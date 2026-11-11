# rpi_bench_infer — TM inference on the Raspberry Pi from a Hamming-coded datagram stream

MedSec-25 single Tsetlin Machine (70/15/15 model: 5 classes × 900 clauses, 356 literals, T = 623, s = 4.64).
Clause weights (−193 … +147) are sent as **4-bit normalised weights**: `s = max|w| / 7 = 27.5703`,
`w4 = clip(round(w / s), −8, 7)` (16 levels). Test accuracy: 96.2729 % with full weights, **96.2615 % with
4-bit weights** (−0.0114 pp; 99.66 % of the 78,801 test flows keep their prediction).
The model arrives as a stream of UDP-style datagrams; every program decodes it with Hamming(7,4), rebuilds
the clause bank and runs one of 10 inference kernels.

## Datagram format (`data/medsec_tm_hamming.bin`)
```
clear header (8 B): [class][n_real rules][seq lo][seq hi][norm flag bit7 | 4-bit weight bits3..0]
                    [scale s lo][scale s hi]  (Q8.8, s = 7058/256 = 27.5703)  [node bits7..4 | layout bit0]
payload (548 B)   : 12 rules × 45 B (356 literal bits, LSB first, 4 padding bits) + copy of the header
Hamming(7,4)      : each byte -> high-nibble codeword, low-nibble codeword; bits P1 P2 D1 P4 D2 D3 D4, MSB first
                    548 B -> 959 B, datagram = 967 B; 403 datagrams (clauses grouped by class and 4-bit weight)
```
Decoding uses the formula-based syndrome of `rpi_customarchi.c`
(`s1 = P1⊕D1⊕D2⊕D4`, `s2 = P2⊕D1⊕D3⊕D4`, `s4 = P4⊕D2⊕D3⊕D4`, flip bit `7 − s`); the receiver then checks the
header copy and the sequence number and places the `n_real` rules of each packet with the packet's weight.
The receiver reads the normalisation flag and the scale from the header: it keeps the 4-bit weight `w4` for
inference (a positive common scale does not change the arg-max) and can de-normalise it as `w ≈ w4 · s`.

## The 10 kernels (same receiver in all of them)
| # | file | idea | cores |
|---|---|---|---|
| 01 | `tm01_normal.c` | unpacked (1 byte / literal), full scan | 1 |
| 02 | `tm02_packed_bytes.c` | 45-byte packed `rule & ~input` | 1 |
| 03 | `tm03_packed_early_exit.c` | packed, stop at first failing byte | 1 |
| 04 | `tm04_packed64.c` | 6 × uint64 words | 1 |
| 05 | `tm05_packed_batch.c` | NumPy-style whole-array passes | 1 |
| 06 | `tm06_scalar_neon.c` | NEON inside one clause (3 × 128-bit) | 1 |
| 07 | `tm07_persistent_neon.c` | NEON across 64 samples (rpi_customarchi.c kernel) | 1 |
| 08 | `tm08_interclause_neon.c` | **clause parallelism**: 16 clauses per NEON vector | 1 |
| 09 | `tm09_interclause_omp.c` | #08 + **core parallelism** (OpenMP, `TM_THREADS`, default 4; receiver also parallel) | 1–4 |
| 10 | `tm10_final_neon.c` | #08 + empty clauses dropped, masked int16 weights, 2-group ILP, early exit | 1 |

## One-click benchmark: `benchmark_rpi.ipynb`
Open it in Jupyter on the Pi and **Restart & Run All**, or run `./run_notebook.sh` (uses `jupyter nbconvert`,
or plain `python3 benchmark_rpi_cells.py` when Jupyter is missing). It runs, verifies and plots, in order:
pure Python (Hamming decode + 4 inference kernels), NumPy (vectorised decode + 4 kernels), the 10 C kernels,
the OpenMP thread sweep and the pipeline modes. Output: `results_rpi/` — `fig1_inference_latency`,
`fig2_receive_end_to_end`, `fig3_cores_pipeline`, `fig4_stage_breakdown` (PDF + PNG), `all_results.csv`,
`stage_breakdown.csv` (end to end: first datagram → 256 predictions, split into decode / ECC / inference), `summary.txt`.
Needs: `sudo apt install python3-numpy python3-matplotlib build-essential` (+ `jupyter` for the notebook).
`make_notebook.py` regenerates the notebook. `reference_x86/notebook_x86/` shows the output of an x86 run.

## On the Raspberry Pi 4 (64-bit Raspberry Pi OS, gcc), step by step
```bash
tar -xzf rpi_bench_infer.tar.gz && cd rpi_bench_infer
./verify.sh                        # 1. correctness: must end with "VERIFY: PASS"
./run_all.sh data 2.0              # 2. benchmark all 10 kernels  -> results.csv
./run_threads.sh data 1.0 5        # 3. 1 core vs OpenMP 1..4 threads, 5 runs each -> threads_results.csv
python3 plot_results.py            # 4. figures (needs pandas + matplotlib; or copy the CSVs back and plot here)
```
Close other programs and use the performance governor for stable numbers:
`echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`.

## What "verified" means
Every program prints and checks:
* `4-bit weights normalised, s=27.5703` — the header flag and scale were read;
* `hash OK` — the rebuilt model equals the transmitted one (FNV-1a over rules + 4-bit weights = `MODEL_HASH`);
* `scores==ref YES` — the class sums of all 256 test flows are **identical** to those of the 4-bit model,
  computed independently in Python (`scores_ref.npy`), and `pred 256/256`;
* `acc 0.9062` — 232/256 correct (the sample is stratified over the 5 classes, so it includes misclassified flows).

`verify.sh` also runs the stream with **1 % single-bit errors** (`medsec_tm_hamming_err.bin`): the receiver must
report **4459 corrected** codewords and still give the same hash and scores; and the raw (no-ECC) stream must give
the same model. Expected values: `data/expected.txt`. `reference_x86/` holds a run on an x86 PC (NEON mapped to
SSE) — its timings are **not** Pi timings, they only show the expected output format.

## Output columns (`results.csv`)
`receive_ms` = Hamming decode + header checks + rebuild of the whole model (median), `corrected` = codewords
repaired, `prep_ms` = kernel-specific layout (once), `infer_ms_per_sample` = median batch time / 256,
`all_ok` = 1 when hash, scores and predictions all match.

## Re-creating the inputs
`python make_inputs.py [NS=256]` (on the server) rebuilds `data/` and `model_dims.h` from the trained model
(`/home/<user>/figures/models/medsec`), checking the stream with an independent Python decoder.

## Pipelined receiver (`tm11_pipeline.c`, `./run_pipeline.sh data 2.0`)
Every datagram passes three stages, as in `rpi_customarchi.c`: **A** detect (extract codewords, formula-based
syndrome), **B** correct (flip bit `7 − s`, rebuild bytes, header / sequence / normalisation check),
**C** infer (the packet's 12 rules = one 16-lane NEON group, scored against all test samples at once:
`sums[s][class] += w4 · fired`). A packet is scored as soon as it is corrected — inference does not wait for
the whole model. Latency = first packet available → class sums of all 256 samples.

| mode | what runs | cores |
|---|---|---|
| `seq` | A for all packets, then B for all, then C for all | 1 |
| `stream` | A → B → C for packet n, then packet n+1 | 1 |
| `swp` | software pipeline: one iteration does A(n+2), B(n+1), C(n) on three independent packets | 1 |
| `pipe` | 3 persistent threads on cores 1/2/3, 16 packet buffers, lock-free progress counters: while C scores packet n, B corrects n+1 and A decodes n+2 | 3 |
| `hyb2ph` | hybrid, two phases: A + B for all packets with OpenMP (`TM_THREADS`, default 4), then C on one core | 4 |
| `hybpipe` | hybrid pipeline: 3 worker threads (cores 1–3) each do A + B for every 3rd packet; the main thread does C for each packet as soon as it is corrected | 4 |

All modes must give `scores==ref YES` and the same `corrected` count (4459 on the error stream).
On one core the stages cannot run at the same instant; `stream`/`swp` only remove the wait for the whole model
and give the out-of-order core independent work. Real overlap of stages needs `pipe` / `hybpipe` (several
cores). Idle worker threads sleep and are woken outside the timed region before each run.
