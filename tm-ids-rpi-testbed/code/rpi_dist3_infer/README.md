# rpi_dist3_infer: distributed 3-TM inference under attack on Raspberry Pi (MedSec-25, 70/15/15)

3 sending RPis (one TM each) → central RPi (Hamming decode + ECC + TM inference of 50 test flows, fused over the
3 TMs). TM 2's RPi is routed through the laptop, which is the man-in-the-middle (same topology as `rpi_medsciot`).
One **round** = one complete transmission of the 3 models. A shared `schedule.csv` says, per round, which stream is
sent (no ECC or Hamming), which attack the laptop applies, K, and the K random literals (the **same K literals in
every clause** of the attacked TM). The central RPi logs accuracy, class sums of every TM and timing for every round.

## Files

| File | Where | What |
|---|---|---|
| `tm0/1/2_hamming.bin`, `tm0/1/2_raw.bin` | sending RPi n | packet stream of TM n (394 / 402 / 402 datagrams, 967 B Hamming / 556 B raw) |
| `sender.py` | sending RPis | sends its TM once per round (2-byte round number in front of every datagram) |
| `mitm.py` | laptop | relay: corrupts TM 2's rules as the schedule says, forwards to the central node |
| `receiver.c`, `dist3_config.h`, `neon_compat.h` | central RPi | UDP → decode → ECC → NEON inference, writes `results/rounds.csv` + `results/class_sums.csv` |
| `X_50.bin`, `y_50_actual.bin`, `y_50_ref.bin`, `sums_ref.bin` | central RPi | 50 test flows (10 per class), true class, clean prediction, clean class sums |
| `schedule.csv` | **all devices** | the round plan (`make_schedule.py` makes it: 98 rounds by default) |
| `simulate.py` | anywhere (numpy) | offline twin: same byte changes as `mitm.py`, Python decoder → `expected/` (also `--nodes 0,1,2`) |
| `compare.py` | anywhere | checks the RPi results against the twin, bit for bit |
| `plot_results.py` | anywhere | figures + `summary_by_K.csv` from the results |
| `make_inputs.py` | server | regenerates the streams and the 50 flows from `figures/models/medsec/tm3_node*.npz` |

## Packet format

```
UDP packet = [round lo][round hi] + datagram          (round number: transport field, not part of the model)
datagram   = 8-byte clear header + payload (Hamming(7,4): 959 B, raw: 548 B)
  [0] class   [1] real rules (1..12)   [2..3] packet index (LE)   [4] 0x80 | w4 (4-bit weight, -8..7)
  [5..6] scale s = 7022 / 256 = 27.43 (Q8.8)   [7] node id << 4 | layout (1 = Hamming, 0 = raw)
payload    = 12 rules x 45 bytes (356 literals, LSB first) + copy of the 8-byte header
```
Weights: one common scale for the 3 TMs, w4 = round(w / s); the fused 4-bit model scores 96.03 % on the whole test
split (full weights 96.04 %), 82 % (41/50) on the 50 balanced test flows. Fusion = sum of the raw class sums
(no clipping), argmax (first maximum wins).

## Run

**0. Copy the folder** to every device (`schedule.csv` must be identical everywhere).

**1. Central RPi (10.42.0.59)**, start first:
```bash
gcc -O3 -mcpu=native -o receiver receiver.c
./receiver --port 5001 --out results --idle-ms 10000
```

**2. Laptop (192.168.50.2)**, the MITM for TM 2:
```bash
python3 mitm.py --dest 10.42.0.59 --dest-port 5001 --listen-port 5001 --src 192.168.50.2
```

**3. Sending RPis**: start the three within about 1 s of each other:
```bash
python3 sender.py --node 0 --dest 10.42.0.59 --src 10.42.0.1       # RPi 0, direct
python3 sender.py --node 1 --dest 10.42.0.59 --src 10.42.0.82      # RPi 1, direct
python3 sender.py --node 2 --dest 10.42.0.59 --src 192.168.50.3    # RPi 2, via the laptop
```
With the default gap (2 ms per datagram, 600 ms between rounds) one round takes 1.64 s, so 98 rounds take about
2 min 40 s. If the receiver prints `INCOMPLETE` (lost packets), raise `--gap-us` on all three senders.

**4. Check and plot** (copy `results/` back to the server or a laptop with numpy / pandas / matplotlib):
```bash
python3 simulate.py                                   # -> expected/       (testbed: TM 2 attacked)
python3 simulate.py --nodes 1,2   --out expected_2of3 # what 2 of 3 attacked TMs would give
python3 simulate.py --nodes 0,1,2 --out expected_all3 # what 3 of 3 attacked TMs would give
python3 compare.py results expected                   # PASS = the RPi reproduced every class sum exactly
python3 plot_results.py results --sim "expected_2of3:2 of 3 TMs attacked (simulated)" \
                                --sim "expected_all3:all 3 TMs attacked (simulated)"
```
Figures land in `results/figures/`: `fig_acc_vs_K`, `fig_class_sums_vs_K`, `fig_timing` (PDF + PNG) and
`summary_by_K.csv` / `timing_median_by_layout.csv` for MATLAB.

## Output columns

`rounds.csv`: round, layout, attack, K, nodes, iteration, literals, packets per TM, complete, corrected codewords,
bad headers, acc_pct, correct, agree_ref (predictions equal to the clean model), clean_sums_exact, and times in ms:
`net_ms` first → last packet of the round, `decode_ms` / `ecc_ms` / `infer_ms` busy time of the receiver,
`tail_ms` last packet → predictions, `e2e_ms` first packet → predictions.

`class_sums.csv`: round, flow, actual, ref_pred, pred, source (tm0 / tm1 / tm2 / fused), c0..c4.

## Verified here (x86, localhost, same programs)

* 98 rounds, 3 senders + MITM relay + receiver over UDP: 98/98 complete, no lost packets.
* `compare.py`: class sums of every TM and the fused model identical to the offline twin in **98/98** rounds,
  corrected codewords identical in 98/98.
* Clean rounds reproduce `sums_ref.bin` exactly (50/50 predictions equal to the clean model).
* Receiver work per round (1198 packets, 50 flows, one core): Hamming 5.1 ms (decode 3.2, ECC 1.0, inference 1.0),
  no ECC 1.2 ms. On the RPi expect several times more.
