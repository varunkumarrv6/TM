# rpi_codes_benchmark

C reconstructions of the 10 TM inference stages (6 classes × 6000 clauses, 526 literals / 66 packed bytes),
all checked against the TMU predictions in `../rpi_50_test/y_50_tm_predictions.npy`.

```
make            # Pi 4: real arm_neon.h, -mcpu=native   |  x86: NEON mapped to SSE via shim/
./run_all.sh [iot_dataset dir=..] [seconds per kernel=1.0]
TM_THREADS=2 ./bin/tm09_interclause_omp ..
```

| # | file | how much is original |
|---|------|----------------------|
| 1 | tm01_normal.c | reconstructed (C port of NumPy baseline: unpacked, full scan) |
| 2 | tm02_packed_bytes.c | reconstructed (66-byte `rule & ~input`) |
| 3 | tm03_packed_early_exit.c | reconstructed (break on first missing byte) |
| 4 | tm04_packed64.c | reconstructed (9 × uint64 words) |
| 5 | tm05_packed_batch.c | reconstructed (NumPy-style whole-array passes) |
| 6 | tm06_scalar_neon.c | reconstructed (one clause at a time, 5 × 128-bit vectors) |
| 7 | tm07_persistent_neon.c | kernel from `../rpi_customarchi.c` (`clause_match_neon`, persistent transposed input vectors, 1 rule × 50 samples); whether this is what the "Persistent NEON" entry was is a guess |
| 8 | tm08_interclause_neon.c | kernel from `original/tm_interclause_neon_single.c` (user-supplied original) |
| 9 | tm09_interclause_omp.c | #8 kernel + OpenMP over clause groups (reconstructed) |
| 10 | tm10_final_neon.c | reconstructed: polarity-sorted groups, fire counting, empty clauses dropped, 2-group ILP, early exit |

`original/tm_interclause_neon_single.c` is kept unmodified; `bin/export_bins` writes the
`rules_bitpacked.bin`, `weights.bin` (int16) and `input.bin` (sample 0) it reads.

Latency = median batch time / 50 samples. Model prep (packing, transposing) is untimed;
per-input transforms are timed. Timings on x86 do not reproduce the Pi numbers, and the
C ports of the Python stages (2–5) lose the interpreter overhead that made them slow.
