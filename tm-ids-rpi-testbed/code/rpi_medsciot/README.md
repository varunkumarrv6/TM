# rpi_medsciot: distributed TM inference over UDP (MedSec-25)

3 sending RPis (one TM each) → laptop (man-in-the-middle, forwards UDP) → central RPi (Hamming decode + TM inference).
Same packet format as `/home/<user>/medsciot` (ecc_pipeline.py) and the receiver logic of `iot_dataset/rpi_customarchi.c`.

## Files

| File | What it is |
|---|---|
| `X_50.bin` | 50 test flows × 23 bytes: 178 Boolean features, feature f = byte f/8, bit f%8 (LSB first), as `rpi_50_test/X_50_bitpacked` |
| `X_50.npy`, `X_50_bitpacked.npy` | same as numpy (50×178 and 50×23) |
| `actual.bin`, `y_50_actual.npy` | true class of each flow (1 byte each) |
| `tmu_pred.bin`, `y_50_tm_predictions.npy` | prediction of the deployed 3-TM model with **no attack** (the reference to compare with) |
| `y_50_tm_predictions_fullweights.npy` | tmu prediction with full integer weights (agrees on 49/50) |
| `class_names.txt`, `class_names.npy` | 0 Benign, 1 Exfiltration, 2 Initial access, 3 Lateral movement, 4 Reconnaissance |
| `tm0_hamming.bin`, `tm1_hamming.bin`, `tm2_hamming.bin` | **split packet streams** for RPi 0/1/2: 445 / 446 / 442 datagrams × 956 bytes |
| `tm0_raw.bin`, `tm1_raw.bin`, `tm2_raw.bin` | same packets without ECC: 548-byte datagrams |
| `rpi_sender.py` | runs on each sending RPi: sends its `.bin` as UDP datagrams |
| `rpi_medsciot_receiver.c` + `rpi_medsciot_config.h` | central node: UDP receive → Hamming decode → TM inference → report |
| `decode_reference.py` | independent Python decoder (numpy only) for checking captured / corrupted streams |
| `make_rpi_files.py` | regenerates everything above from the trained model in `/home/<user>/medsciot` |

## Datagram layout (956 bytes)

```
byte 0        weight nibble (bits 7-4, signed -8..+7) | class nibble (bits 3-0)
byte 1, 2     sequence number, little endian (0 .. packets_of_this_TM - 1)
byte 3        TM node id (0, 1, 2)
byte 4..955   Hamming(7,4) payload = encode(544 bytes):
                 bytes   0..539  12 rules x 45 bytes   (rule r = bytes 45r .. 45r+44)
                 bytes 540..543  copy of the 4 header bytes (receiver uses this protected copy)
```
* **Rule bits**: literal `l` of rule `r` = payload byte `45r + l//8`, bit `l%8` (LSB first). Literal `l < 178` is
  feature `l`; literal `178 + k` is NOT feature `k`. Bits 356-359 of a rule are padding.
* **Hamming(7,4)**: every payload byte → high-nibble codeword, then low-nibble codeword; nibble bit 3 = D1.
  Codeword bits (MSB first) = `P1 P2 D1 P4 D2 D3 D4`, packed back to back: codeword `i` occupies encoded bits
  `7i .. 7i+6` (MSB-first inside each byte) of bytes 4..955. 1088 codewords = 7616 bits = 952 bytes.
* Payload byte `B`, bit `b` (0 = LSB) sits in codeword `2B` if `b >= 4`, else `2B + 1`, at codeword position
  D1 (b%4 = 3) / D2 (2) / D3 (1) / D4 (0) = encoded bit `7*cw + {2, 4, 5, 6}[3 - b%4]` → datagram byte
  `4 + bit//8`, mask `0x80 >> (bit % 8)`. Flipping that one bit changes exactly one literal (Hamming repairs it);
  two flipped data bits in the same codeword cannot be repaired.
* Every packet holds 12 rules of **one class with one weight**; empty (all-zero) rules are padding and never fire.
* Class score per TM = Σ weight of firing rules; each TM's score is capped at ±33; the 3 capped scores are added;
  highest class wins (`CLASS_SUM_T` in the config header).

## Run

```bash
# central RPi
gcc -O3 -march=native -o receiver rpi_medsciot_receiver.c       # NEON is used automatically on ARM
./receiver                     # listens on UDP 6000 (direct) and 5001 (from the laptop); --raw for no-ECC packets
# each sending RPi n (0, 1, 2) -> laptop, or straight to the central node
python3 rpi_sender.py tm<n>_hamming.bin <laptop_or_central_ip> 5001
# offline check of files (e.g. corrupted copies captured on the laptop)
./receiver --files tm0_hamming.bin tm1_hamming.bin tm2_hamming.bin
python3 decode_reference.py tm0_hamming.bin tm1_hamming.bin tm2_hamming.bin
```
The receiver stops when all 1333 datagrams arrived, or after 3 s without traffic (`--idle-ms`).

## Verified here (x86, scalar path)
* file mode, Hamming and raw: decoded predictions = `tmu_pred.bin` on 50/50 flows, accuracy 86.0 %
* one flipped data bit in tm1: 1 codeword corrected, 50/50 still identical
* UDP on localhost: 3 senders in parallel (ports 6000 + 5001), 1333/1333 datagrams, 50/50 identical;
  decode ≈ 10 ms + inference ≈ 18 ms for all packets on one core

## Network deployment (4 RPi + laptop testbed)

```
 10.42.0.1  (RPi) ── tm0_hamming.bin ─────────────────────────────► 10.42.0.59  CENTRAL
 10.42.0.82 (RPi) ── tm1_hamming.bin ─────────────────────────────► 10.42.0.59  (receiver,
 192.168.50.3 (RPi behind laptop) ── tm2_hamming.bin ──► laptop ──► 10.42.0.59   corrects +
                        (thinks it sends to 10.42.0.59)  192.168.50.2 (MITM)      infers)
```
The RPi behind the laptop sends to the CENTRAL ip (10.42.0.59); because its only route is through the laptop,
the laptop receives the packets, modifies them, and forwards them on to the central node, which believes they
came from the RPi.

**1. Central node (10.42.0.59)** — start first:
```bash
gcc -O3 -march=native -o receiver rpi_medsciot_receiver.c
./receiver --idle-ms 10000            # listens on UDP 5001 (laptop) and 6000 (direct)
```

**2. Laptop MITM (192.168.50.2)** — pick ONE attack per run:
```bash
python3 mitm_include.py --dest 10.42.0.59 --dest-port 5001 --listen-port 5001 --src 192.168.50.2 --k 2
python3 mitm_exclude.py --dest 10.42.0.59 --dest-port 5001 --listen-port 5001 --src 192.168.50.2 --k 2
python3 mitm_toggle.py  --dest 10.42.0.59 --dest-port 5001 --listen-port 5001 --src 192.168.50.2 --k 4
```
`--k` = how many of the most-used literals to attack (vary it each run), or `--literals 323,327,...` for a
specific set. The relay listens on the laptop interface facing the RPi, corrupts the same literals in all 12
rules of every datagram (header + Hamming parity untouched), and forwards to the central node.

**3. Sending RPis** — one file each:
```bash
# 10.42.0.1  -> central directly
python3 rpi_sender.py tm0_hamming.bin --dest 10.42.0.59 --port 5001 --src 10.42.0.1
# 10.42.0.82 -> central directly
python3 rpi_sender.py tm1_hamming.bin --dest 10.42.0.59 --port 5001 --src 10.42.0.82
# 192.168.50.3 -> central ip, but routed through the laptop (the MITM)
python3 rpi_sender.py tm2_hamming.bin --dest 10.42.0.59 --port 5001 --src 192.168.50.3
```

Expected on the central node: each TM delivers all its packets; `codewords corrected` counts the bits the MITM
flipped that Hamming repaired. With k = 1-2 the model is fully restored (`decoded == reference : 50/50`, PASS);
from k = 4 the chosen literals start sharing codewords, Hamming can no longer repair them, and accuracy drops
(e.g. toggle k=4 -> 64 %, CHECK) — the same effect as the medsciot SEQ experiments.

Verified on x86 localhost: sender -> each MITM relay -> receiver, all three attacks; k=2 gives 50/50 PASS,
toggle k=4 gives 32/50 (64 %).
