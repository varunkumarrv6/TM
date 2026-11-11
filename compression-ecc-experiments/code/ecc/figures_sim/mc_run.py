"""Runs the 150-iteration Monte Carlo (mc_attacks.py) for one dataset in parallel chunks and merges the parts into
figures/<ds>/monte_carlo/mc_results_<ds>.csv.   usage: python mc_run.py {medsec|iot|mnist} [CHUNKS] [THREADS]"""
import glob
import os
import subprocess
import sys
import pandas as pd
import ecc_common as E

DS = sys.argv[1]
CHUNKS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
THREADS = sys.argv[3] if len(sys.argv) > 3 else "8"
ITER = 150
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{E.ROOT}/{DS}/monte_carlo"
os.makedirs(f"{OUT}/parts", exist_ok=True)

per = -(-ITER // CHUNKS)
jobs = [(a, min(a + per, ITER)) for a in range(0, ITER, per)]
procs = [subprocess.Popen([f"{HERE}/run.sh", f"{HERE}/mc_attacks.py", DS, str(a), str(b),
                           f"{OUT}/parts/mc_part_{a}_{b}.csv"], cwd=HERE,
                          env={**os.environ, "NUMBA_NUM_THREADS": THREADS},
                          stdout=open(f"{OUT}/parts/mc_part_{a}_{b}.log", "w"), stderr=subprocess.STDOUT)
         for a, b in jobs]
codes = [p.wait() for p in procs]
assert not any(codes), f"chunk failed: {codes}"

df = (pd.concat(pd.read_csv(p) for p in glob.glob(f"{OUT}/parts/mc_part_*.csv"))
      .sort_values(["iteration", "K", "attack", "layout"]))
assert len(df) == ITER * 7 * 3 * 4, len(df)
df.to_csv(f"{OUT}/mc_results_{DS}.csv", index=False, float_format="%.10f")
print(f"[{DS}] merged {len(df)} runs, iterations {df.iteration.min()}-{df.iteration.max()}")
