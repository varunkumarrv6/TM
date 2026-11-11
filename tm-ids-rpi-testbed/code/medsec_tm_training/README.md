# new_medsec — MedSec-25, single weighted TM, accuracy vs K literal errors per clause

All inference: raw class sums (no clipping to ±T), prediction = argmax.

| File | Step |
|---|---|
| `prep.py` | clean + dedupe MedSec-25 (`raw/` -> /home/<user>/medsciot/raw), 80/20 stratified train/test split, quantile thermometer (178 bits -> 356 literals), per-class cap 24000 on train -> `data.npz` |
| `train.py` | train one weighted TM (tmu, 1000 clauses/class, T=1000, s=5, 15 epochs), export include/exclude cube + weights -> `tm.npz` (log: `train.log`) |
| `inject.py N` | benchmark (no errors, checked == tmu predict) + error injection, N iterations -> `inject_results.csv` (log: `inject.log`) |
| `plot.py` | `fig_accuracy_vs_K.png`, `summary_accuracy_vs_K.csv` |

Injection: for every clause of every class, K distinct literal positions are drawn at random **independently per
clause**; the same K is used for all clauses. K = 1,2,4,6,8,10,12,14,16,20. Error types: toggle (flip, main),
include (force 1), exclude (force 0). No error correction. 50 iterations per K.

Benchmark (no errors): **96.0168 %** on 105,066 test flows.

| K | toggle mean % | include mean % | exclude mean % |
|---|---|---|---|
| 1 | 93.76 | 94.28 | 95.98 |
| 2 | 85.49 | 85.83 | 95.98 |
| 4 | 60.02 | 55.79 | 95.93 |
| 6 | 42.95 | 42.53 | 95.94 |
| 8 | 32.26 | 33.27 | 95.84 |
| 10 | 15.68 | 12.21 | 95.90 |
| 12 | 6.02 | 5.61 | 95.88 |
| 14 | 2.83 | 2.99 | 95.84 |
| 16 | 2.78 | 2.49 | 95.85 |
| 20 | 2.28 | 2.28 | 95.85 |

Run: `source /home/<user>/iot_dataset/proj/bin/activate; python prep.py; python train.py; python inject.py 50; python plot.py`

## Part 2 — 5 distributed TMs, errors injected into TM 0 only, NO coding

| File | Step |
|---|---|
| `train5.py N` | split the same training set into 5 disjoint stratified shards (~15.7k flows each), train TM N on shard N -> `tm5_nodeN.npz` (logs `train5_nN.log`); test set unchanged and separate |
| `inject5.py` | fused prediction `argmax_c sum_n s_n,c` (raw sums, no clipping); errors (K random literals per clause, same K for all clauses) into TM 0 only, single iteration -> `inject5_results.csv`, `inject5.log` |
| `plot5.py` | `fig5tm_tm0_accuracy_vs_K.png` |

Each TM alone: 95.63 / 95.93 / 95.94 / 96.05 / 95.99 %. Benchmark (5 TMs fused, no errors): **96.0196 %**.

| K | include | exclude | toggle |
|---|---|---|---|
| 0 | 96.020 | 96.020 | 96.020 |
| 1 | 96.033 | 96.020 | 96.032 |
| 2 | 96.018 | 96.016 | 96.028 |
| 4 | 96.037 | 96.014 | 96.018 |
| 6 | 96.041 | 96.021 | 96.031 |
| 8 | 96.035 | 96.018 | 96.039 |
| 10 | 96.035 | 96.021 | 96.036 |
| 12 | 96.036 | 96.018 | 96.035 |
| 14 | 96.036 | 96.018 | 96.036 |
| 16 | 96.036 | 96.030 | 96.036 |
| 20 | 96.036 | 96.015 | 96.036 |

Corrupting one of five TMs costs nothing: include/toggle errors make TM 0's clauses almost never fire, so its class
sums go to ~0 and the fused vote converges to TMs 1-4 alone (96.0358 %). Exclude errors barely change TM 0 at all.

## Part 3 — one test flow, clean vs K=3 include
`one_sample.py [K] [INDEX]` -> `one_sample_K3.txt`: per-class firing clauses, +/- votes and class sum for one flow, test accuracy clean (96.0168 %) vs K=3 include (89.8835 %), plus a literal-level trace of one clause.
