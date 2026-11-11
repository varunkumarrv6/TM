# rpi_3_c: physical 3-TM testbed in C (MedSec-25, 70/15/15, 4-bit weights)

```
 RPi 0 (TM 0)  RPI0_IP ───────────────────────────────────────────────► CENTRAL_IP:5001
 RPi 1 (TM 1)  RPI1_IP ───────────────────────────────────────────────► CENTRAL_IP:5001   central RPi:
 RPi 2 (TM 2)  RPI2_IP ──► LAPTOP_IP:5001  forwarder (attack, K by hand) ──► CENTRAL_IP:5001   decode + ECC +
                                                                                            inference, prints
                                                                                            class sums + accuracy
```
TM 2 is sent to the laptop. The forwarder changes the same K literals in every rule of TM 2 (include / exclude /
toggle) and forwards to the central RPi. The central RPi shows, after every transmission, the accuracy and class
sums of **each TM and of the fused model**, so you can see TM 2 collapse while the fused accuracy holds.

## 1. Set the addresses (once)
Edit `net_config.h`: `CENTRAL_IP`, `LAPTOP_IP`, `RPI0_IP`, `RPI1_IP`, `RPI2_IP`, ports, `GAP_US`.
Copy the whole folder to all 4 RPis and the laptop (the data files must be next to the programs).

## 2. Build (on each device)
```bash
make receiver           # central RPi
make sender NODE=0      # RPi 0       (NODE=1 on RPi 1, NODE=2 on RPi 2)
make forwarder          # laptop      (no gcc on the laptop?  python3 forwarder.py ... does the same)
```

## What the laptop corrupts
`--target all` (default): K random **bit positions anywhere in TM 1's datagram** -- clear header, rule data,
Hamming parity bits and the protected header copy -- the same K positions in every packet (the attacker does not know
the code). `--target literals`: K literal positions in every rule (the earlier runs 6-12).
The receiver therefore takes the format from the datagram size and, with Hamming, the header from the repaired
protected copy; it reports damaged clear headers and dropped packets (header unusable even after correction).
A Hamming datagram has 7736 bits, so K must be large to hurt (predicted, toggle, Hamming: K=300 -> TM 1 64 %,
K=1000 -> 20 %; raw: K=16 -> 74 %, K=64 -> all TM 1 packets dropped). Fused accuracy stays about 80-82 %.

## 3. One run (repeat with another K)
```bash
# central RPi -- start once, keeps running for all runs
./receiver                     # --show 10 prints the class sums of the first 10 flows

# laptop -- choose attack and K for this run
./forwarder -a none            # first: clean reference run
./forwarder -a toggle -k 300   # then other K (Ctrl-C and restart with a new -k); -t literals = old attack
#           -a exclude | toggle,  -s 2 for another random set of K literals,  -l 12,200,301 for chosen literals

# the three sending RPis -- start them, they send their TM once and exit
./sender                       # Hamming(7,4) stream
./sender raw                   # no ECC
```
Leave about 2 s between runs (the forwarder labels a new run after a 2 s pause), and do one clean run
(`-a none`) for each layout (`./sender` and `./sender raw`), so the plots have a K = 0 reference.

The receiver prints, per run:
```
=== run 4   attack=toggle K=8 seed=1 literals=69 80 81 163 230 251 253 285   layout hamming
packets  TM0 394/394  TM1 402/402  TM2 402/402   corrected codewords 31500   header errors 0
                     accuracy      clean    mean true sum      clean   all sums = 0
TM 0                   82.00%     82.00%            24.06      24.06           0/50
TM 1                   84.00%     84.00%            25.92      25.92           0/50
TM 2 (via laptop)      30.00%     78.00%             5.56      29.02          24/50
fused                  82.00%     82.00%            55.54      79.00           0/50
class sums of the first 3 flows (TM0 | TM1 | TM2 | fused -> prediction, true class):
  flow  0: [   39  -27  -24  -43  -45 ] [   40  -34  -20  -25  -45 ] [   -3    0    1    0    0 ] [   76 ... ] -> 0, true 0
time: decode 3.01 ms, ECC 0.98 ms, inference 1.21 ms (processing 5.19 ms); network first->last packet ...
```
(numbers from the localhost test; the RPi gives the same accuracies and sums, only the times differ.)
If a TM shows fewer packets than expected (`** INCOMPLETE **`), packets were lost: raise `GAP_US` and rebuild.

## 4. Results and plots
`results/runs.csv` gets one row per run (attack, K, literals, layout, packets, corrected codewords, accuracy and mean
true-class sum of TM 0/1/2/fused and their clean values, flows with all-zero sums, decode / ECC / inference time);
`results/run_<id>_class_sums.csv` has all 50 flows x (TM 0, TM 1, TM 2, fused) class sums with the clean ones.
```bash
python3 check_run.py --runs results/runs.csv     # every run vs the Python prediction (needs numpy)
python3 check_run.py --attack include --k 4      # what one run should give, before you start it
python3 plot_runs.py results/runs.csv            # -> results/figures/fig_accuracy_vs_K, fig_true_sum_vs_K, runs_by_K.csv
```

## Packet format (same as rpi_bench_infer)
`[0] class [1] real rules (1..12) [2..3] packet index LE [4] 0x80 | 4-bit weight [5..6] scale (7022/256) [7] node<<4 | layout`
+ payload (12 rules x 45 B + header copy = 548 B), Hamming(7,4) → 959 B. Datagram 967 B (Hamming) / 556 B (raw).
TM packets: 394 / 402 / 402. Fusion: sum of the raw class sums of the 3 TMs, argmax. The 50 flows are 10 per class
from the 70/15/15 test split (clean fused 82 %, 41/50; on the whole test split the fused 4-bit model gives 96.03 %).

## Files
| file | device | |
|---|---|---|
| `net_config.h` | all | IP addresses / ports / packet gap |
| `sender.c` + `tm<n>_hamming.bin`, `tm<n>_raw.bin` | RPi n | sends TM n |
| `forwarder.c` (`forwarder.py` = same in Python) | laptop | attack on TM 2 with your K |
| `receiver.c`, `dist3_config.h`, `neon_compat.h`, `X_50.bin`, `y_50_actual.bin`, `y_50_ref.bin`, `sums_ref.bin` | central RPi | decode, ECC, NEON inference, results |
| `check_run.py`, `plot_runs.py` | any (numpy, pandas, matplotlib) | check and plot |
| `test_localhost.sh` | any Linux | whole testbed on one machine (127.0.0.1) as a dry run |

Verified on x86 (localhost, `bash test_localhost.sh`): 5 runs (clean, include K=4 Hamming / raw, toggle K=8 Hamming,
include K=20 raw), all packets received, every run identical to `check_run.py`.
