"""Compression level sweep of every TM of one configuration (iteration IT, K local TMs; K = 1 = centralized weighted
TM): payload = include actions + clause weights (compress_bench.payload), every method at every level 0..10 the
library supports (compress_bench.METHODS), lossless check, median times.
usage: ./run.sh clause_study/job_levels.py DS IT K   -> levels/<ds>/it<IT>_k<K>.csv (one row per TM x method x level)
"""
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT
from compress_bench import METHODS, payload, bench

DS, IT, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs(f"{ROOT}/levels/{DS}", exist_ok=True)
dst = f"{ROOT}/levels/{DS}/it{IT:02d}_k{K}.csv"
if os.path.exists(dst):
    sys.exit(0)
rows = []
for k in range(K):
    data, n_inc = payload(f"{ROOT}/models/{DS}/it{IT:02d}/k{K}_tm{k}.npz")
    rows += [dict(iteration=IT, K=K, tm=k, include_literals=n_inc, **bench(data, m, l))
             for m, (_, _, levels, _) in METHODS.items() for l in levels]
pd.DataFrame(rows).to_csv(dst + ".tmp", index=False)
os.replace(dst + ".tmp", dst)
print(f"levels {DS} it{IT:02d} K={K}: {len(rows)} rows", flush=True)
