# Physical 3-TM testbed on Raspberry Pi: results

MedSec-25, 70/15/15 split, 3 distributed TMs (5 classes x 900 clauses each, 4-bit weights), Hamming(7,4) packets,
50 test flows (10 per class). Code: `../rpi_3_c` (sender.c, forwarder.py, receiver.c, net_config.h).

## Testbed

| Device | IP | Role | Packets |
|---|---|---|---|
| RPi (TM 0) | 10.42.0.1 | sender → central | 394 |
| RPi (TM 1) | 192.168.50.3 | sender → **laptop** 192.168.50.2 (forwarder, attack with K) → central | 402 |
| RPi (TM 2) | 10.42.0.59 | sender → central | 402 |
| central RPi | 10.42.0.145 | receiver: decode + ECC + NEON inference, fusion | 1198 per transmission |

## Run 5 (2026-10-27): clean reference, no attack, Hamming(7,4)

| Model | Route | Packets | Accuracy (50 flows) | Clean accuracy | Mean true-class sum | Clean | Flows with all sums 0 |
|---|---|---|---|---|---|---|---|
| TM 0 | 10.42.0.1 → central | 394 / 394 | 82.00 % | 82.00 % | 24.06 | 24.06 | 0 / 50 |
| TM 1 | 192.168.50.3 → laptop → central | 402 / 402 | 84.00 % | 84.00 % | 25.92 | 25.92 | 0 / 50 |
| TM 2 | 10.42.0.59 → central | 402 / 402 | 78.00 % | 78.00 % | 29.02 | 29.02 | 0 / 50 |
| **Fused (sum of 3 TMs)** | central 10.42.0.145 | 1198 / 1198 | **82.00 %** | 82.00 % | 79.00 | 79.00 | 0 / 50 |

* Corrected codewords 0, header errors 0; fused prediction equal to the clean model for **50 / 50** flows.
* Identical to the offline prediction (`check_run.py --attack none --node 1`, saved in `expected_clean_check_run.txt`).

Timing on the central RPi (one transmission, 1198 packets, 50 flows, one core):

| Stage | Time |
|---|---|
| Decode (codewords + formula syndrome) | 11.49 ms |
| ECC (correction + header checks) | 4.19 ms |
| Inference (16-lane NEON, per packet) | 6.89 ms |
| **Processing total** | **22.57 ms** |
| Last packet → fused predictions | 0.033 ms |
| Network, first → last packet | 9447.5 ms (the 3 senders were started by hand one after another) |

Class sums of the first 5 flows (all true class 0 = Benign, all predicted 0): `clean_run_class_sums_first5.csv`.

## Files

| File | Content |
|---|---|
| `clean_run_per_tm.csv` | the table above |
| `clean_run_timing.csv` | the timing table |
| `clean_run_class_sums_first5.csv` | class sums of TM 0 / 1 / 2 / fused for flows 0-4 |
| `clean_run_receiver_output.txt` | the receiver output as printed on the central RPi |
| `expected_clean_check_run.txt` | the offline prediction it matches |

## Runs 6-12 (2026-10-27): include attack on TM 1 (via the laptop), Hamming(7,4), seed 1 -- LITERAL-ONLY attack

(The forwarder changed only the K literal (data) bits in every rule; headers and Hamming parity bits were not
touched. Later runs use `--target all`: K random bits anywhere in the datagram.)

| Run | K | Corrected codewords | TM 0 | **TM 1 (attacked)** | TM 2 | **Fused** | TM 1 mean true sum | TM 1 flows all 0 | Fused = clean |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 0 (clean) | 0 | 82 % | 84 % | 78 % | **82 %** | 25.92 | 0/50 | 50/50 |
| 6 | 4 | 17,274 | 82 % | 84 % | 78 % | **82 %** | 25.92 | 0/50 | 50/50 |
| 7 | 8 | 30,257 | 82 % | **52 %** | 78 % | **82 %** | 6.68 | 7/50 | 50/50 |
| 8 | 12 | 47,359 | 82 % | **52 %** | 78 % | **82 %** | 6.68 | 7/50 | 50/50 |
| 9 | 16 | 59,596 | 82 % | **42 %** | 78 % | **82 %** | 3.50 | 26/50 | 48/50 |
| 10 | 20 | 72,537 | 82 % | **42 %** | 78 % | **82 %** | 3.50 | 26/50 | 48/50 |
| 11 | 24 | 85,080 | 82 % | **42 %** | 78 % | **82 %** | 3.50 | 29/50 | 48/50 |
| 12 | 30 | 109,780 | 82 % | **42 %** | 78 % | **82 %** | 3.50 | 29/50 | 48/50 |

* K = 4: every changed literal is alone in its codeword, Hamming repairs all of them (TM 1 unchanged).
* From K = 8 two literals share a codeword (80 and 81), Hamming cannot repair it: TM 1 falls to 52 % / 42 % and its
  class sums go towards 0; TM 0 and TM 2 keep the fused accuracy at 82 % for every K.
* Every run is identical to the offline prediction (`check_run.py --attack include --k K --node 1`).
* Processing on the central RPi: 21.9-36.7 ms per transmission (more corrected codewords -> more ECC time).
Files: `attack_include_hamming_runs.csv`, `attack_include_hamming_tm1_class_sums_first5.csv`,
`runs_1_12_receiver_output.txt`.

To keep all 50 flows: copy `results/runs.csv` and `results/run_005_class_sums.csv` from the central RPi
(`~/rpi3tm_c/results/`) into this folder. Attack runs (forwarder `--attack include --k 4`, ...) can be added the same way.

## Runs 13-25 and 41-54 (2026-10-27): blind include attack on TM 1, no ECC vs Hamming(7,4)

The laptop forwarder (`--target all`) sets K random **bit positions anywhere in TM 1's datagram** (clear header,
rule data, parity bits, protected header copy) to 1, the same K positions in every packet. Seed 1.
Runs 41-54: TM 1 sent raw (no ECC; the run header said `layout mixed` because TM 0 / TM 2 still sent Hamming --
they are not attacked, so their sums are unaffected). Runs 13-25: everything Hamming.

| K | TM 1 no ECC | TM 1 Hamming | fused no ECC | fused Hamming | TM 1 true sum, no ECC | TM 1 true sum, Hamming |
|---|---|---|---|---|---|---|
| 0 | 84 % | 84 % | 82 % | 82 % | 25.92 | 25.92 |
| 2 | 80 % | 84 % | 82 % | 82 % | 22.94 | 25.92 |
| 4 | 80 % | 84 % | 82 % | 82 % | 21.42 | 25.92 |
| 8 | 80 % | 84 % | 82 % | 82 % | 20.70 | 25.92 |
| 12 | 78 % | 84 % | 82 % | 82 % | 14.70 | 25.92 |
| 16 | 74 % | 84 % | 82 % | 82 % | 11.92 | 25.92 |
| 20 | 74 % | 82 % | 82 % | 80 % | 9.54 | 24.94 |
| 24-27 | 72 % | -- | 82 % | -- | 9.40 / 8.40 | -- |
| 28, 30 | **20 %** (all 402 packets dropped: clear-header bit hit) | 82 % | 82 % | 80 % | 0.00 | 24.94 |
| 40-60 | -- | 82 % (202 clear headers damaged, all repaired) | -- | 80 % | -- | 24.94 / 24.82 |
| 80 | -- | 80 % | -- | 80 % | -- | 24.14 |

* No ECC: every flipped bit hits the model, TM 1's true-class sum falls 25.9 -> 8.4 by K = 27, and from K = 28 a
  header bit is hit, so all TM 1 packets are unusable (TM 1 = 20 %, all sums 0).
* Hamming: up to K = 16 every bit is repaired (TM 1 unchanged); up to K = 80 TM 1 loses at most 4 points and
  damaged clear headers are rebuilt from the protected copy (0 packets dropped).
* Fused accuracy stays 80-82 % in every run (TM 0 and TM 2 carry the vote).
* All 27 runs equal the offline prediction exactly (see "Check against the prediction" below).

Files: `measured_include_all.csv` (Pi), `predicted_include_all.csv` (check_run.py),
`python plot_include_all.py include` -> `fig_include_all_raw_vs_hamming.pdf/.png` (markers measured, lines predicted).

## Runs 56-66 (2026-10-27): blind TOGGLE attack on TM 1, no ECC (raw), seed 1

The forwarder flips K random bit positions anywhere in every TM 1 datagram (TM 1 sent raw; TM 0 / TM 2 Hamming,
`layout mixed`).

| K | TM 1 accuracy | TM 1 true sum | fused accuracy | fused true sum | note |
|---|---|---|---|---|---|
| 0 | 84 % | 25.92 | 82 % | 79.00 | clean (run 41) |
| 2 | 78 % | 22.78 | 82 % | 75.86 | |
| 4 | 78 % | 21.26 | 82 % | 74.34 | |
| 8 | 80 % | 20.44 | 82 % | 73.52 | |
| 12 | 80 % | 14.34 | 82 % | 67.42 | |
| 16 | 74 % | 11.70 | 82 % | 64.78 | |
| 20 | 74 % | 8.84 | 82 % | 61.92 | |
| 24 | 72 % | 8.70 | 82 % | 61.78 | |
| 26 | -- | -- | -- | -- | run 66 INVALID: TM 0 never arrived, TM 1 sent twice -> repeat |
| 28, 29, 30 | **20 %** | 0.00 | 82 % | 53.08 | clear-header bit flipped: all 402 TM 1 packets dropped |


## Runs 76-91 (2026-10-27): blind TOGGLE attack on TM 1, Hamming(7,4), seed 1

| K | corrected codewords | clear headers damaged | TM 1 accuracy | TM 1 true sum | fused accuracy | fused true sum |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 84 % | 25.92 | 82 % | 79.00 |
| 1 / 4 / 8 / 14 | 402 / 1,608 / 3,216 / 5,628 | 0 | 84 % | 25.92 | 82 % | 79.00 |
| 20 / 24 / 30 | 7,638 / 9,246 / 11,658 | 0 | 82 % | 24.90 | 80 % | 77.98 |
| 50 | 19,296 | 402 (all repaired) | 82 % | 24.90 | 80 % | 77.98 |
| 80 | 30,552 | 402 | 80 % | 23.98 | 80 % | 77.06 |
| 120 | 46,230 | 402 | 80 % | 21.24 | 80 % | 74.32 |
| 160 | 61,506 | 402 | 82 % | 19.42 | 80 % | 72.50 |
| 200 | 75,174 | 402 | 74 % | 14.32 | 82 % | 67.40 |
| 240 | 88,038 | 402 | 76 % | 7.28 | 82 % | 60.36 |
| 280 | 102,510 | 402 | 68 % | 6.14 | 80 % | 59.22 |
| 320 | 114,168 | 402 | 50 % | 1.96 | 82 % | 55.04 |
| 356 | 125,022 | 402 | **34 %** | 0.02 (10 flows all 0) | **82 %** | 53.10 |

* No ECC: TM 1 is lost at K = 28 (header hit). Hamming: TM 1 survives about 10x more flipped bits -- unchanged up to
  K = 14, still 80 % at K = 160, and only at K = 320-356 it collapses (50 % / 34 %); no packet was ever dropped,
  every damaged clear header was rebuilt from the protected copy. The fused 3-TM accuracy stays 80-82 % throughout.
Files: `measured_toggle_all.csv`, `predicted_toggle_all.csv`, `python plot_include_all.py toggle` ->
`fig_toggle_all_raw_vs_hamming.pdf/.png`.
Receiver fix after run 66: repeated packets (same TM and packet number again) no longer count towards the 1198, and the
receiver warns "repeated packets ignored: a sender was started twice".

## Check against the prediction

All valid Pi runs (include 27/27, toggle 28/28) equal `check_run.py` exactly: accuracy and mean class sums of every
TM and of the fused model, and the corrected-codeword counts. (An earlier version of check_run.py ignored the 4
padding bits after the 356 literals of each rule; the receiver scores all 360 bits, so a rule whose padding bit the
attacker sets can never fire. With the padding treated like the receiver, the earlier small differences at raw
K = 8 / 12 and Hamming K >= 240 disappear.)

## Runs 96-106 (2026-10-27): blind EXCLUDE attack on TM 1, no ECC (raw), seed 1

The forwarder clears K random bit positions anywhere in every TM 1 datagram (TM 1 raw, `layout mixed`).

| K | TM 1 accuracy | TM 1 true sum | fused accuracy | fused true sum | note |
|---|---|---|---|---|---|
| 0 | 84 % | 25.92 | 82 % | 79.00 | clean (run 41) |
| 4 | 82 % | 25.76 | 82 % | 78.84 | |
| 12 | 82 % | 25.46 | 82 % | 78.54 | |
| 28 | 84 % | 25.24 | 82 % | 78.32 | |
| 40 / 50 / 60 | 84 % | 25.50 / 25.68 / 25.78 | 82 % | 78.58 - 78.86 | runs 99 and 106 (K = 60) identical |
| 80, 120, 150, 200 | **20 %** | 0.00 | 82 % | 53.08 | all 402 TM 1 packets dropped |

* Exclude barely hurts: it can only change bits that are 1, and only a few % of the rule bits are 1 (included
  literals), so most of the K positions change nothing. Removing a literal also makes a clause fire more easily,
  which moves TM 1's sums only slightly (true sum 25.9 -> 25.2-25.8).
* From K = 80 position 4438 is among the K: it lies in the header COPY at the end of the raw payload (scale byte,
  bit 0x02 = 1 in every packet), so every clear header differs from its copy and all TM 1 packets are dropped
  (raw has no protection to tell which one is right). The fused accuracy still stays 82 %.
* All 12 runs equal `check_run.py` exactly. Hamming exclude runs: not measured yet.
Files: `measured_exclude_all.csv`, `predicted_exclude_all.csv`, `python plot_include_all.py exclude` ->
`fig_exclude_all_raw_vs_hamming.pdf/.png`.

## Runs 107-112 (2026-10-27): blind EXCLUDE attack on TM 1, Hamming(7,4), seed 1

| K | corrected codewords | clear headers damaged | TM 1 accuracy | TM 1 true sum | fused accuracy |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 84 % | 25.92 | 82 % |
| 60 / 80 / 120 | 1,900 / 2,448 / 3,683 | 200 (all repaired) | 84 % | 25.92 | 82 % |
| 200 / 300 | 5,416 / 7,726 | 200 | 84 % | 25.96 / 26.04 | 82 % |
| 356 | 8,992 | 200 | 86 % | 26.38 | 82 % |

Exclude can only clear bits that are 1, so even K = 356 leaves almost every codeword with at most one error; Hamming
repairs them (no packet dropped), while without ECC all TM 1 packets were lost from K = 80.

## All attacks in one figure

`python plot_all_attacks.py` -> `fig_all_attacks_raw_vs_hamming.pdf/.png`: columns include / exclude / toggle, rows
accuracy and mean true-class sum, no ECC vs Hamming(7,4), markers measured on the RPis, lines predicted.
Check against the prediction: include 27/27, exclude 19/19, toggle 28/28 runs identical.

| Attack | no ECC: TM 1 lost (all packets dropped) at | Hamming: TM 1 at the largest K measured | fused accuracy |
|---|---|---|---|
| include | K = 28 | 80 % at K = 80 | 80-82 % |
| exclude | K = 80 | 86 % at K = 356 (unharmed) | 82 % |
| toggle | K = 28 | 34 % at K = 356 (collapses only beyond K ~ 300) | 80-82 % |
