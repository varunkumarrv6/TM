"""MITM attacks on a distributed weighted-TM IDS (MedSec-25) with Hamming(7,4) protection.

System: N nodes each send a clause-bank cube I_n in {0,1}^{C x Cl x 2o} (+ weights, assumed intact).
Fusion:  S_c(x) = sum_n clip(s_{n,c}(x), -T, T),   y_hat = argmax_c S_c(x)  (ties -> lowest index = Benign).

Attacks (applied by the MITM to literal bits in transit, parity bits are NOT recomputed):
  A1(k)  random: flip k random literals in every clause of every class
  A2(F)  force-include all literals (x and not-x thermometer bits) of raw features F
  A2+(F) force-include only the positive literals of F
  A3(F)  force-exclude all literals of F
  A4(F)  toggle all literals of F
Protection schemes:
  none   raw cube
  SEQ    Hamming(7,4), 4 consecutive literals (tmu order) per codeword
  INT    Hamming(7,4), literals interleaved so each codeword holds 4 different raw features
  KEY    Hamming(7,4), whole cube bit-stream permuted by a secret key before grouping
"""
import itertools
import time
import numpy as np
import pandas as pd
from numba import njit, prange

BASE = "/home/<user>/medsciot"
N_NODES = 3
A1_K = [1, 2, 4, 8, 16]
A1_SEEDS = [0, 1, 2]
TARGET_K = [1, 2, 4, 8]
SCHEMES = ["none", "SEQ", "INT", "KEY"]

D = np.load(f"{BASE}/data.npz", allow_pickle=True)
Xte, yte = D["Xte"], D["yte"].astype(np.int64)
class_names = D["class_names"]
feat_names = D["feat_names"]
lit_feature_bool = D["lit_feature"]          # Boolean feature -> raw feature
o = Xte.shape[1]
L = 2 * o
lit_raw = np.concatenate([lit_feature_bool, lit_feature_bool])   # literal -> raw feature
lit_pol = np.concatenate([np.zeros(o, int), np.ones(o, int)])    # 0 = x, 1 = not x
lit_rank = np.concatenate([np.concatenate([np.arange((lit_feature_bool == j).sum())
                                           for j in np.unique(lit_feature_bool)])] * 2)
C = len(class_names)
BENIGN = int(np.where(class_names == "Benign")[0][0])

nodes = [np.load(f"{BASE}/node{n}.npz") for n in range(N_NODES)]
T = int(nodes[0]["T"])
Cl = int(nodes[0]["clauses"])
cubes = [nd["cube"].astype(np.uint8) for nd in nodes]
weights = [nd["weights"].astype(np.int32) for nd in nodes]

# ---------------------------------------------------------------- inference engine
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


def class_sums(cube, wts, X=X_lit):
    Cw = pack_rows(cube.reshape(-1, L)).reshape(C, Cl, W64)
    nonempty = cube.any(axis=2)          # tmu outputs 0 for empty clauses at inference
    return _class_sums(X, Cw, nonempty, wts)


# Validate engine against tmu predict() class sums
for n, nd in enumerate(nodes):
    ref_idx = nd["ref_idx"]
    ours = class_sums(cubes[n], weights[n], X_lit[ref_idx])
    assert np.array_equal(ours, nd["ref_sums"]), f"engine mismatch node {n}"
print(f"inference engine matches tmu class sums exactly on all {N_NODES} nodes")


def fuse(sums_list):
    return np.argmax(sum(np.clip(s, -T, T) for s in sums_list), axis=1)


CLEAN_PRED = None


def metrics(pred):
    acc = (pred == yte).mean()
    f1s = []
    for c in range(C):
        tp = np.sum((pred == c) & (yte == c))
        fp = np.sum((pred == c) & (yte != c))
        fn = np.sum((pred != c) & (yte == c))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    mal = yte != BENIGN
    detect = (pred[mal] != BENIGN).mean()                 # malicious flows raised as attacks
    bal = np.mean([(pred[yte == c] == c).mean() for c in range(C)])
    agree = (pred == CLEAN_PRED).mean() if CLEAN_PRED is not None else 1.0   # same label as clean model
    return dict(accuracy=acc, macro_f1=np.mean(f1s), balanced_acc=bal, detection_rate=detect,
                agreement_with_clean=agree)


# ---------------------------------------------------------------- Hamming(7,4)
# Codeword layout P1 P2 D1 P4 D2 D3 D4 (same as error_coding.ipynb)
DATA_POS = np.array([2, 4, 5, 6])


def hamming_encode(d):                     # d: (M,4) uint8
    d1, d2, d3, d4 = d.T
    return np.stack([d1 ^ d2 ^ d4, d1 ^ d3 ^ d4, d1, d2 ^ d3 ^ d4, d2, d3, d4], axis=1)


def hamming_decode(r):                     # r: (M,7) uint8
    r = r.copy()
    s = (r[:, 0] ^ r[:, 2] ^ r[:, 4] ^ r[:, 6]) \
        | ((r[:, 1] ^ r[:, 2] ^ r[:, 5] ^ r[:, 6]) << 1) \
        | ((r[:, 3] ^ r[:, 4] ^ r[:, 5] ^ r[:, 6]) << 2)
    bad = np.nonzero(s)[0]
    r[bad, s[bad].astype(np.int64) - 1] ^= 1
    return r[:, DATA_POS], len(bad)


def layout_perm(scheme, n):
    """Permutation of the flattened (C, Cl, L) bit-stream; 4 consecutive entries form one codeword."""
    total = C * Cl * L
    if scheme == "SEQ":
        return np.arange(total)
    if scheme == "INT":
        # within each clause order literals by (rank inside feature, polarity, raw feature) so
        # consecutive literals belong to different raw features
        order = np.lexsort((lit_raw, lit_pol, lit_rank))
        return (np.arange(C * Cl)[:, None] * L + order[None, :]).reshape(-1)
    if scheme == "KEY":
        return np.random.default_rng(0xC0FFEE + n).permutation(total)   # secret key per node link
    raise ValueError(scheme)


assert (C * Cl * L) % 4 == 0
PERMS = {(s, n): layout_perm(s, n) for s in ("SEQ", "INT", "KEY") for n in range(N_NODES)}


def transmit(clean_cube, attacked_cube, scheme, n):
    """Return the cube the inference model reconstructs, plus codewords with non-zero syndrome."""
    if scheme == "none":
        return attacked_cube, 0
    p = PERMS[(scheme, n)]
    cw = hamming_encode(clean_cube.reshape(-1)[p].reshape(-1, 4))
    # MITM overwrites the data bits it targets; parity bits are left as originally sent
    cw[:, DATA_POS] = attacked_cube.reshape(-1)[p].reshape(-1, 4)
    data, flagged = hamming_decode(cw)
    out = np.empty(C * Cl * L, np.uint8)
    out[p] = data.reshape(-1)
    return out.reshape(C, Cl, L), flagged


# ---------------------------------------------------------------- attacks
def attack_random(cube, k, seed):
    rng = np.random.default_rng(seed)
    a = cube.copy()
    # k distinct literals per clause: argsort of random keys
    idx = np.argsort(rng.random((C, Cl, L)), axis=2)[:, :, :k]
    np.put_along_axis(a, idx, 1 - np.take_along_axis(a, idx, axis=2), axis=2)
    return a


def target_mask(F, positive_only=False):
    m = np.isin(lit_raw, F)
    if positive_only:
        m &= lit_pol == 0
    return m


def attack_targeted(cube, kind, F):
    """Feature-level (A2/A2+/A3/A4: all literals of raw features F) or literal-level
    (A2L/A3L/A4L: only the single literals F, in every clause of every class)."""
    a = cube.copy()
    if kind.endswith("L"):
        m = np.isin(np.arange(L), F)
    else:
        m = target_mask(F, positive_only=(kind == "A2+"))
    if kind in ("A2", "A2+", "A2L"):
        a[:, :, m] = 1
    elif kind in ("A3", "A3L"):
        a[:, :, m] = 0
    elif kind in ("A4", "A4L"):
        a[:, :, m] ^= 1
    return a


# Targets: raw features ranked by how often their literals are included (attacker's best guess
# of "important" features: it can read the clause banks in transit)
usage = np.zeros(len(feat_names))
for cb in cubes:
    np.add.at(usage, lit_raw, cb.sum(axis=(0, 1)))
ranked = np.argsort(-usage)
TARGETS = {k: ranked[:k] for k in TARGET_K}
print("most-included raw features:", [(feat_names[j], int(usage[j])) for j in ranked[:8]])

# Literal-level targets: single literals ranked by how many clauses include them
lit_usage = sum(cb.sum(axis=(0, 1)).astype(np.int64) for cb in cubes)
lit_ranked = np.argsort(-lit_usage, kind="stable")
LIT_TARGETS = {k: lit_ranked[:k] for k in TARGET_K}
lit_name = lambda k: f"{'NOT ' if lit_pol[k] else ''}{feat_names[lit_raw[k]]} b{lit_rank[k]}"
print("most-included literals:", [(lit_name(k), int(lit_usage[k])) for k in lit_ranked[:8]])

# ---------------------------------------------------------------- run
t0 = time.time()
clean_sums = [class_sums(cubes[n], weights[n]) for n in range(N_NODES)]
rows = []
CLEAN_PRED = fuse(clean_sums)
base = metrics(CLEAN_PRED)
rows.append(dict(attack="clean", param="-", scheme="-", attacked_nodes=0, lit_err_before=0,
                 lit_err_after=0, flagged_codewords=0, **base))
single = metrics(np.argmax(clean_sums[0], axis=1))
print(f"clean fused: {base}\nclean node0 alone: {single}")

scenarios = [("A1", k, s) for k in A1_K for s in A1_SEEDS] + \
            [(kind, k, None) for kind in ("A2", "A2+", "A3", "A4") for k in TARGET_K] + \
            [(kind, k, None) for kind in ("A2L", "A3L", "A4L") for k in TARGET_K]

for kind, k, seed in scenarios:
    for scheme in SCHEMES:
        node_sums, err_b, err_a, flag = [], [], [], []
        for n in range(N_NODES):
            if kind == "A1":
                att = attack_random(cubes[n], k, seed * 10 + n)
            else:
                att = attack_targeted(cubes[n], kind, LIT_TARGETS[k] if kind.endswith("L") else TARGETS[k])
            rec, flagged = transmit(cubes[n], att, scheme, n)
            err_b.append(int((att != cubes[n]).sum()))
            err_a.append(int((rec != cubes[n]).sum()))
            flag.append(flagged)
            node_sums.append(class_sums(rec, weights[n]) if err_a[-1] else clean_sums[n])
        for m in range(1, N_NODES + 1):          # first m nodes compromised
            sums = node_sums[:m] + clean_sums[m:]
            rows.append(dict(attack=kind, param=k, seed=seed, scheme=scheme, attacked_nodes=m,
                             lit_err_before=sum(err_b[:m]), lit_err_after=sum(err_a[:m]),
                             flagged_codewords=sum(flag[:m]), **metrics(fuse(sums))))
    print(f"{kind} k={k} seed={seed} done  ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{BASE}/results_raw.csv", index=False)
# Average random-attack seeds
agg = df.drop(columns="seed").groupby(["attack", "param", "scheme", "attacked_nodes"],
                                     sort=False).mean().reset_index()
agg.to_csv(f"{BASE}/results.csv", index=False)
print(f"done in {time.time()-t0:.0f}s -> results.csv")
