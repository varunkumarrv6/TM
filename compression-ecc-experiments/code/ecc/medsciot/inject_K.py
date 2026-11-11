"""Accuracy vs K for the 3-node weighted TM (MedSec-25 test split), errors in EVERY clause of EVERY node.

For each iteration and each K, every clause (all classes, all 3 nodes) gets K distinct literal positions drawn at
random independently for that clause; the same K is used for all clauses. Errors:
  toggle  : flip the include/exclude bit
  include : force the literal to include (1)
  exclude : force the literal to exclude (0)
Fusion: y_hat = argmax_c sum_n s_{n,c}  (raw class sums, no clipping). No ECC.
K = 0 is the benchmark; each node's engine sums are checked to equal tmu's saved test_sums exactly.

usage: python inject_K.py [ITERS]   -> inject_K_results.csv
"""
import sys
import time
import numpy as np
import pandas as pd
from numba import njit, prange

BASE = "/home/<user>/medsciot"
N_NODES = 3
K_LIST = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20]
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ATTACKS = ["toggle", "include", "exclude"]

D = np.load(f"{BASE}/data.npz", allow_pickle=True)
Xte, yte = D["Xte"], D["yte"].astype(np.int64)
nodes = [np.load(f"{BASE}/node{n}.npz") for n in range(N_NODES)]
cubes = [nd["cube"].astype(np.uint8) for nd in nodes]
weights = [nd["weights"].astype(np.int32) for nd in nodes]
C, Cl, L = cubes[0].shape
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


def class_sums(cb, wt):
    Cw = pack_rows(cb.reshape(C * Cl, L)).reshape(C, Cl, W64)
    return _class_sums(X_lit, Cw, cb.any(axis=2), wt)


def predict(cbs):
    return np.argmax(sum(class_sums(cb, wt) for cb, wt in zip(cbs, weights)), axis=1)   # raw sums, no clipping


def inject(rng, cube, K, attack):
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


# Benchmark (no errors) and check every node against tmu
for n in range(N_NODES):
    assert (class_sums(cubes[n], weights[n]) == nodes[n]["test_sums"]).all(), f"node {n} sums differ from tmu"
pred0 = predict(cubes)
BENCH = (pred0 == yte).mean() * 100
print(f"benchmark (no errors, 3 nodes fused): {BENCH:.4f} %  on {len(yte)} test flows  (node sums == tmu: OK)",
      flush=True)

rows = [dict(iteration=0, K=0, attack=a, accuracy=BENCH, drop=0.0) for a in ATTACKS]
for it in range(ITERS):
    t0 = time.time()
    for K in K_LIST:
        for a in ATTACKS:
            rng = np.random.default_rng([it, K, ATTACKS.index(a)])
            acc = (predict([inject(rng, cb, K, a) for cb in cubes]) == yte).mean() * 100
            rows.append(dict(iteration=it + 1, K=K, attack=a, accuracy=acc, drop=BENCH - acc))
    print(f"iter {it+1}/{ITERS}  {time.time()-t0:.1f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{BASE}/inject_K_results.csv", index=False)
print(df.groupby(["attack", "K"])["accuracy"].agg(["mean", "std", "min", "max"]).round(3).unstack(0).to_string())
