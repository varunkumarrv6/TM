"""5-TM distributed system, NO coding: inject errors into TM 0 only and measure the fused accuracy vs K.

Fusion: S_c(x) = sum_n s_{n,c}(x)  (raw class sums, no clipping),  y_hat = argmax_c S_c(x).
TM 0: every clause gets K distinct random literal positions (drawn independently per clause, same K for all clauses),
error = include (force 1) / exclude (force 0) / toggle (flip). TMs 1-4 are untouched. Single iteration.
-> inject5_results.csv
"""
import numpy as np
import pandas as pd
from numba import njit, prange

BASE = "/home/<user>/new_medsec"
N_NODES = 5
ATTACKED = 0
K_LIST = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20]
ATTACKS = ["include", "exclude", "toggle"]

D = np.load(f"{BASE}/data.npz", allow_pickle=True)
Xte, yte = D["Xte"], D["yte"].astype(np.int64)
nodes = [np.load(f"{BASE}/tm5_node{n}.npz") for n in range(N_NODES)]
T = int(nodes[0]["T"])
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


raw = [class_sums(cubes[n], weights[n]) for n in range(N_NODES)]
for n in range(N_NODES):
    assert (np.argmax(raw[n], 1) == nodes[n]["test_pred"]).all(), f"engine != tmu on node {n}"
    print(f"TM {n} alone: {(np.argmax(raw[n], 1) == yte).mean()*100:.4f} %  (== tmu predict: OK)")
clean = raw
rest = sum(clean[n] for n in range(N_NODES) if n != ATTACKED)
BENCH = (np.argmax(rest + clean[ATTACKED], 1) == yte).mean() * 100
print(f"benchmark, 5-TM fused, no errors: {BENCH:.4f} %  on {len(yte)} test flows")
print(f"TMs 1-4 fused (TM 0 removed):     {(np.argmax(rest, 1) == yte).mean()*100:.4f} %")

rows = [dict(K=0, attack=a, accuracy=BENCH, drop=0.0) for a in ATTACKS]
for K in K_LIST:
    for a in ATTACKS:
        rng = np.random.default_rng([K, ATTACKS.index(a)])
        s0 = class_sums(inject(rng, cubes[ATTACKED], K, a), weights[ATTACKED])
        acc = (np.argmax(rest + s0, 1) == yte).mean() * 100
        rows.append(dict(K=K, attack=a, accuracy=acc, drop=BENCH - acc))
df = pd.DataFrame(rows)
df.to_csv(f"{BASE}/inject5_results.csv", index=False)
print(df.pivot(index="K", columns="attack", values="accuracy").round(3).to_string())
