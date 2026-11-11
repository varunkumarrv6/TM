"""Single TM (no splitting): class sums for ONE test flow, clean vs K random INCLUDE errors per clause,
plus test accuracy for both. Also traces one clause literal-by-literal.

usage: python one_sample.py [K] [SAMPLE_INDEX]
"""
import sys
import numpy as np

BASE = "/home/<user>/new_medsec"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
D = np.load(f"{BASE}/data.npz", allow_pickle=True)
Xte, yte, cn = D["Xte"], D["yte"].astype(np.int64), D["class_names"]
M = np.load(f"{BASE}/tm.npz")
cube, W, T = M["cube"].astype(np.uint8), M["weights"].astype(np.int64), int(M["T"])
C, Cl, L = cube.shape
Xl = np.concatenate([Xte, 1 - Xte], 1).astype(np.uint8)   # literals: x then NOT x


def clause_out(cb, x):
    """(C, Cl) clause outputs for one literal vector x: AND over included literals; empty clause -> 0."""
    return ((cb & (1 - x)).sum(2) == 0) & (cb.sum(2) > 0)


def accuracy(cb):
    Cw = cb.reshape(C * Cl, L).astype(np.float32)
    viol = (1 - Xl).astype(np.float32) @ Cw.T          # (N, C*Cl): included literals that are false
    fire = (viol == 0) & (Cw.sum(1) > 0)[None, :]
    sums = (fire.reshape(-1, C, Cl) * W[None]).sum(2)
    return (np.argmax(sums, 1) == yte).mean() * 100, sums


acc0, sums0 = accuracy(cube)
assert (np.argmax(sums0, 1) == M["test_pred"]).all()

# K random literals per clause, chosen independently for every clause, forced to include
rng = np.random.default_rng(3)
lits = rng.random((C * Cl, L)).argpartition(K, axis=1)[:, :K]
att = cube.reshape(C * Cl, L).copy()
att[np.repeat(np.arange(C * Cl), K), lits.ravel()] = 1
att = att.reshape(C, Cl, L)
acc1, sums1 = accuracy(att)

i = int(sys.argv[2]) if len(sys.argv) > 2 else int(np.where(yte == 2)[0][0])   # first Initial-access flow
x = Xl[i]
print(f"test flow #{i}, true class = {cn[yte[i]]}   (K = {K} random literals forced to INCLUDE in every clause)\n")
for name, cb, s in [("CLEAN", cube, sums0[i]), (f"K={K} INCLUDE", att, sums1[i])]:
    out = clause_out(cb, x)
    print(f"--- {name} ---")
    print(f"{'class':<17}{'+clauses fire':>14}{'+votes':>9}{'-clauses fire':>15}{'-votes':>9}{'class sum':>11}")
    for c in range(C):
        pos, neg = out[c] & (W[c] > 0), out[c] & (W[c] < 0)
        print(f"{cn[c]:<17}{pos.sum():>14}{W[c][pos].sum():>9}{neg.sum():>15}{W[c][neg].sum():>9}"
              f"{s[c]:>11}")
    p = int(np.argmax(s))
    print(f"prediction: {cn[p]}  ({'correct' if p == yte[i] else 'WRONG'})\n")
print(f"test accuracy on all {len(yte)} flows:  clean {acc0:.4f} %   K={K} include {acc1:.4f} %\n")

# Trace the strongest clause of the true class that fires on this flow
c = yte[i]
fire0 = clause_out(cube, x)[c]
j = int(np.argmax(np.where(fire0, W[c], -10**9)))
inc0 = np.where(cube[c, j])[0]
added = sorted(set(np.where(att[c, j])[0].tolist()) - set(inc0.tolist()))
print(f"trace: class {cn[c]} clause #{j}, weight {W[c, j]:+d}")
print(f"  clean: {len(inc0)} included literals, values on this flow = {x[inc0].tolist()}  -> AND = {int(x[inc0].all())}")
print(f"  K={K}: randomly picked {sorted(lits[c * Cl + j].tolist())}; newly included {added}, "
      f"values on this flow = {x[added].tolist()}")
print(f"         -> AND = {int(x[np.where(att[c, j])[0]].all())}  (contributes {W[c, j] * int(x[np.where(att[c, j])[0]].all()):+d} instead of {W[c, j]:+d})")
