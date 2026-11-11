"""Accuracy vs K: inject errors into K literals of EVERY clause of the trained TM (MedSec-25 test split).

For each iteration and each K, every clause (all classes) gets K distinct literal positions drawn at random
independently for that clause; the same K is used for all clauses. Errors:
  toggle  : flip the include/exclude bit (main error model)
  include : force the literal to include (1)
  exclude : force the literal to exclude (0)
K = 0 is the benchmark (no errors) and is checked to match tmu's own predictions exactly.

usage: python inject.py [ITERS]   -> inject_results.csv
"""
import sys
import time
import numpy as np
import pandas as pd
from numba import njit, prange

BASE = "/home/<user>/new_medsec"
K_LIST = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20]
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ATTACKS = ["toggle", "include", "exclude"]

D = np.load(f"{BASE}/data.npz", allow_pickle=True)
Xte, yte = D["Xte"], D["yte"].astype(np.int64)
M = np.load(f"{BASE}/tm.npz")
cube, weights, T = M["cube"].astype(np.uint8), M["weights"].astype(np.int32), int(M["T"])
C, Cl, L = cube.shape
W64 = (L + 63) // 64


def pack_rows(bits2d):
    """(R, L) {0,1} -> (R, W64) uint64, literal k at word k//64 bit k%64."""
    pad = np.zeros((bits2d.shape[0], W64 * 64), np.uint8)
    pad[:, :bits2d.shape[1]] = bits2d
    return np.packbits(pad, axis=1, bitorder="little").view(np.uint64)


X_lit = pack_rows(np.concatenate([Xte, 1 - Xte], axis=1).astype(np.uint8))


@njit(parallel=True, cache=True)
def _class_sums(Xw, Cw, nonempty, Wt):
    N = Xw.shape[0]
    Cc, Cl_, W = Cw.shape
    out = np.zeros((N, Cc), np.int32)
    for i in prange(N):
        for c in range(Cc):
            s = 0
            for j in range(Cl_):
                if not nonempty[c, j]:
                    continue
                ok = True
                for w in range(W):
                    if Cw[c, j, w] & ~Xw[i, w]:
                        ok = False
                        break
                if ok:
                    s += Wt[c, j]
            out[i, c] = s
    return out


def class_sums(cb):
    Cw = pack_rows(cb.reshape(C * Cl, L)).reshape(C, Cl, W64)
    return _class_sums(X_lit, Cw, cb.any(axis=2), weights)


def predict(cb):
    return np.argmax(class_sums(cb), axis=1)   # raw class sums, no clipping; ties -> lowest class index


def inject(rng, K, attack):
    """K random distinct literals per clause, drawn independently for every clause."""
    lits = rng.random((C * Cl, L)).argpartition(K, axis=1)[:, :K]
    rows = np.repeat(np.arange(C * Cl), K)
    att = cube.reshape(C * Cl, L).copy()
    if attack == "toggle":
        att[rows, lits.ravel()] ^= 1
    elif attack == "include":
        att[rows, lits.ravel()] = 1
    else:
        att[rows, lits.ravel()] = 0
    return att.reshape(C, Cl, L)


# Benchmark (no errors) and check against tmu
sums0 = class_sums(cube)
pred0 = np.argmax(sums0, axis=1)
assert (pred0 == M["test_pred"]).all(), "engine predictions differ from tmu"
BENCH = (pred0 == yte).mean() * 100
print(f"benchmark (no errors): {BENCH:.4f} %  on {len(yte)} test flows  (== tmu predict: OK)", flush=True)

rows = [dict(iteration=0, K=0, attack=a, accuracy=BENCH, drop=0.0) for a in ATTACKS]
for it in range(ITERS):
    t0 = time.time()
    for K in K_LIST:
        for a in ATTACKS:
            rng = np.random.default_rng([it, K, ATTACKS.index(a)])
            acc = (predict(inject(rng, K, a)) == yte).mean() * 100
            rows.append(dict(iteration=it + 1, K=K, attack=a, accuracy=acc, drop=BENCH - acc))
    print(f"iter {it+1}/{ITERS}  {time.time()-t0:.1f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{BASE}/inject_results.csv", index=False)
print(df.groupby(["attack", "K"])["accuracy"].agg(["mean", "std", "min", "max"]).round(3).unstack(0).to_string())
