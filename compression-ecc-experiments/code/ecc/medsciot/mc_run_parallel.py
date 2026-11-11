"""Runs the 150-iteration Monte-Carlo (mc_attacks.py) in parallel chunks and merges the parts into mc_results.csv.
Resumable: iterations already present in mc_part_*.csv are skipped.   usage: python mc_run_parallel.py [--fresh]"""
import glob
import os
import subprocess
import sys
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ITER, CHUNKS = 150, 5
K_LIST = [2, 4, 6, 8, 10, 12, 16, 20]
os.chdir(BASE)

if "--fresh" in sys.argv:
    for p in glob.glob("mc_part_*.csv"):
        os.remove(p)

done = set()
for p in glob.glob("mc_part_*.csv"):
    d = pd.read_csv(p)
    for it, g in d.groupby("iteration"):
        if set(g.K) >= set(K_LIST):
            done.add(int(it))
missing = [i for i in range(ITER) if i + 1 not in done]
print(f"{len(done)} iterations already done, {len(missing)} to run", flush=True)

ranges = []                                    # contiguous runs of missing iteration indices
for i in missing:
    if ranges and ranges[-1][1] == i:
        ranges[-1][1] = i + 1
    else:
        ranges.append([i, i + 1])
jobs = []                                      # split into about CHUNKS equal pieces
per = max(1, -(-len(missing) // CHUNKS))
for a, b in ranges:
    for s in range(a, b, per):
        jobs.append((s, min(s + per, b)))
threads = str(max(8, 120 // max(1, len(jobs))))
procs = [subprocess.Popen([PY, "mc_attacks.py", str(ITER), str(a), str(b), f"mc_part_{a}_{b}.csv"],
                          env={**os.environ, "NUMBA_NUM_THREADS": threads},
                          stderr=subprocess.DEVNULL) for a, b in jobs]
for p in procs:
    p.wait()

parts = [pd.read_csv(p) for p in glob.glob("mc_part_*.csv")]
df = (pd.concat(parts).drop_duplicates(["iteration", "K", "attack", "layout"])
      .sort_values(["iteration", "K", "attack", "layout"]))
df.to_csv("mc_results.csv", index=False)
print(f"merged {len(df)} runs, iterations {df.iteration.min()}-{df.iteration.max()}, K {sorted(df.K.unique())}")
