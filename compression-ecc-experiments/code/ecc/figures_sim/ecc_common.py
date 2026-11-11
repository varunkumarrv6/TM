"""Single-TM model + Hamming(7,4) transmission (No ECC / SEQ / INT / KEY) + MITM attacks + inference,
for MedSec-25, Edge-IIoTset (IoT) and MNIST. Same packet / codeword rules as /home/<user>/medsciot/ecc_pipeline.py
and mc_fast.py (rpi_customarchi.c format), applied to one TM per dataset.

Two data/model versions, chosen with the environment variable SPLIT (outputs go to figures/split_<SPLIT>/<ds>):
  SPLIT=70_15_15 (default): figures/models/<ds>/{data.npz, tm.npz}, 70/15/15 train/validation/test split
                            (prep_701515.py), hyperparameters chosen on validation with Optuna (optuna_701515.py,
                            select_701515.py)
  SPLIT=80_20             : the earlier models without a validation split (new_medsec/tm.npz,
                            iot_dataset/rpi_tm_model, mnist/data/tm_mnist.npz)
Everything here is evaluated on the TEST split.

Packets.  The clauses of every class are cut into packets of 12 rules (tmu clause order); the last packet of a class
is padded with empty rules of weight 0.  Each rule is L literal bits, LSB-first, padded to whole bytes (RB bits).
Clause weights travel in the packet header with their exact tmu value (not attacked), so the benchmark is the exact
tmu accuracy.  Inference: raw class sums, argmax, no clipping; empty clauses never fire (as tmu).

Hamming(7,4) as rpi_customarchi.c: every byte -> high-nibble codeword (bits 7..4 = D1..D4), low-nibble codeword
(bits 3..0 = D1..D4); codeword P1 P2 D1 P4 D2 D3 D4.  The MITM only edits data bits (never parity / headers).
Layouts (which 4 rule bits share one codeword):
  none : no Hamming
  SEQ  : literals in tmu order -> 4 neighbouring literals share a codeword
  INT  : literals interleaved so a codeword holds 4 literals of different features
         MedSec: original ecc_pipeline order (lexsort by bit rank, polarity, feature).
         IoT / MNIST: block interleaver inside each polarity half, codeword = literals {m, m+Q, m+2Q, m+3Q},
         Q = ceil(O/4)  (MNIST: 4 pixels 7 rows apart; IoT: 4 bits ~66 positions apart = different features)
  KEY  : the 12*RB rule bits of every packet shuffled with a secret key before Hamming (attacker knows the key)

Attacks on every rule of every packet (all classes):
  A1 random  : flip K random literals per rule (different literals in each rule)
  A2 include : set the given literal set to 1         A3 exclude : set to 0          A4 toggle : flip
"""
import os
import numpy as np
from numba import njit, prange

DATASETS = ["medsec", "iot", "mnist"]
LAYOUTS = ["none", "SEQ", "INT", "KEY"]
RULES_PER_PACKET = 12
SECRET_KEY = 0xC0FFEE
ATTACKS = {"A1": "random: flip K random literals in every clause",
           "A2": "force INCLUDE the targeted literal(s) in every clause",
           "A3": "force EXCLUDE the targeted literal(s) in every clause",
           "A4": "TOGGLE the targeted literal(s) in every clause"}
SPLIT = os.environ.get("SPLIT", "70_15_15")
assert SPLIT in ("70_15_15", "80_20"), SPLIT
ROOT = f"/home/<user>/figures/split_{SPLIT}"                 # output folder: ROOT/<ds>/...
MODELS = "/home/<user>/figures/models"
NAME = {"medsec": "MedSec-25", "iot": "Edge-IIoTset", "mnist": "MNIST"}
UNIT = {"medsec": "test flows", "iot": "test flows", "mnist": "test images"}


class _Info(dict):
    """INFO[ds] = one-line description of the dataset's selected TM (read from figures/models/<ds>/tm.npz)."""

    def __missing__(self, ds):
        if SPLIT == "80_20":
            self[ds] = INFO_80_20[ds]
            return self[ds]
        t = np.load(f"{MODELS}/{ds}/tm.npz")
        d = np.load(f"{MODELS}/{ds}/data.npz")
        C, Cl, L = t["cube"].shape
        self[ds] = (f"{NAME[ds]}, single {'weighted' if t['weighted'] else 'unweighted'} TM ({C} classes x {Cl} "
                    f"clauses, T={int(t['T'])}, s={float(t['s']):g}, {L} literals), 70/15/15 split, "
                    f"{len(d['yte']):,} {UNIT[ds]}")
        return self[ds]


INFO = _Info()
INFO_80_20 = {"medsec": "MedSec-25, single weighted TM (5 classes x 1000 clauses, 356 literals), 80/20 split, "
                        "105,066 test flows",
              "iot": "Edge-IIoTset, single TM (6 classes x 6000 clauses, unweighted, 526 literals), 80/20 split, "
                     "31,560 test flows",
              "mnist": "MNIST, single weighted TM (10 classes x 2000 clauses, 1568 literals), standard 60k/10k split, "
                       "10,000 test images"}


# ------------------------------------------------------------------ models
def _block_interleave(O):
    """Literal order for INT: within each polarity half of O literals, position 4m+j carries literal j*Q+m."""
    Q = -(-O // 4)
    half = [j * Q + m for m in range(Q) for j in range(4) if j * Q + m < O]
    return np.array(half + [O + l for l in half])


def load(ds):
    """Returns dict: Xte (N, O) uint8, yte, cube (C, Cl, L) uint8 include bits, weights (C, Cl) int32,
    int_order (L,), lit_name (L,) str, sums_ref = tmu class sums on the test split (None if not stored)."""
    if SPLIT == "80_20":
        return _load_80_20(ds)
    d = np.load(f"{MODELS}/{ds}/data.npz", allow_pickle=True)
    t = np.load(f"{MODELS}/{ds}/tm.npz")
    Xte, yte, cube, w = d["Xte"], d["yte"], t["cube"], t["weights"]
    O = Xte.shape[1]
    if ds == "medsec":
        lf, fn = d["lit_feature"], d["feat_names"]
        rank = np.concatenate([np.arange((lf == j).sum()) for j in np.unique(lf)])
        raw, pol, rk = np.r_[lf, lf], np.r_[np.zeros(O, int), np.ones(O, int)], np.r_[rank, rank]
        int_order = np.lexsort((raw, pol, rk))
        names = [f"{'NOT ' if pol[k] else ''}{fn[raw[k]]} b{rk[k]}" for k in range(2 * O)]
    elif ds == "iot":
        int_order = _block_interleave(O)
        names = [f"{'NOT ' if k >= O else ''}x{k % O}" for k in range(2 * O)]
    elif ds == "mnist":
        int_order = _block_interleave(O)
        names = [f"{'NOT ' if k >= O else ''}px({(k % O) // 28},{k % 28})" for k in range(2 * O)]
    else:
        raise ValueError(ds)
    cube = np.ascontiguousarray(cube, dtype=np.uint8)
    assert sorted(int_order.tolist()) == list(range(2 * O))
    return dict(ds=ds, Xte=np.asarray(Xte, np.uint8), yte=np.asarray(yte, np.int64), cube=cube,
                weights=np.asarray(w, np.int32), int_order=int_order, lit_name=np.array(names),
                sums_ref=t["test_sums"])


def _load_80_20(ds):
    """Earlier models (no validation split)."""
    sums_ref = None
    if ds == "medsec":
        d = np.load("/home/<user>/new_medsec/data.npz", allow_pickle=True)
        f = np.load("/home/<user>/medsciot/data.npz", allow_pickle=True)
        assert (d["Xte"] == f["Xte"]).all()
        t = np.load("/home/<user>/new_medsec/tm.npz")
        Xte, yte, cube, w = d["Xte"], d["yte"], t["cube"], t["weights"]
        O = Xte.shape[1]
        lf, fn = f["lit_feature"], f["feat_names"]
        rank = np.concatenate([np.arange((lf == j).sum()) for j in np.unique(lf)])
        raw, pol, rk = np.r_[lf, lf], np.r_[np.zeros(O, int), np.ones(O, int)], np.r_[rank, rank]
        int_order = np.lexsort((raw, pol, rk))
        names = [f"{'NOT ' if pol[k] else ''}{fn[raw[k]]} b{rk[k]}" for k in range(2 * O)]
        sums_ref = t["test_sums"]
    elif ds == "iot":
        B = "/home/<user>/iot_dataset"
        Xte = np.load(f"{B}/tree/X_test_tm_6bit_263.npy")
        yte = np.load(f"{B}/tree/y_test_6.npy")
        cube = np.load(f"{B}/rpi_tm_model/rules.npy")
        w = np.load(f"{B}/rpi_tm_model/weights.npy")
        O = Xte.shape[1]
        int_order = _block_interleave(O)
        names = [f"{'NOT ' if k >= O else ''}x{k % O}" for k in range(2 * O)]
    elif ds == "mnist":
        B = "/home/<user>/mnist/data"
        d = np.load(f"{B}/mnist_bool.npz")
        r = np.load(f"{B}/tm_mnist.npz")
        cb = np.stack([r[f"clauses_{c}"].reshape(2000, 49, 8)[:, :, -1] for c in range(10)])  # include = top state bit
        bits = np.unpackbits(np.ascontiguousarray(cb).view(np.uint8), axis=-1, bitorder="little")
        cube = bits.reshape(10, 2000, 49 * 32)[:, :, :1568]
        w = np.stack([r[f"weights_{c}"] for c in range(10)])
        Xte, yte = d["X_test"].reshape(-1, 784), d["y_test"]
        O = 784
        int_order = _block_interleave(O)
        names = [f"{'NOT ' if k >= O else ''}px({(k % O) // 28},{k % 28})" for k in range(2 * O)]
    else:
        raise ValueError(ds)
    return dict(ds=ds, Xte=np.asarray(Xte, np.uint8), yte=np.asarray(yte, np.int64),
                cube=np.ascontiguousarray(cube, dtype=np.uint8), weights=np.asarray(w, np.int32),
                int_order=int_order, lit_name=np.array(names), sums_ref=sums_ref)


# ------------------------------------------------------------------ Hamming(7,4) on bytes (rpi_customarchi.c)
def hamming_encode_bytes(raw):
    raw = np.asarray(raw, np.uint8)
    nib = np.empty(raw.size * 2, np.uint8)
    nib[0::2], nib[1::2] = raw >> 4, raw & 0x0F
    d1, d2, d3, d4 = (nib >> 3) & 1, (nib >> 2) & 1, (nib >> 1) & 1, nib & 1
    cw = np.stack([d1 ^ d2 ^ d4, d1 ^ d3 ^ d4, d1, d2 ^ d3 ^ d4, d2, d3, d4], axis=1)
    return np.packbits(cw.reshape(-1), bitorder="big")


def hamming_decode_bytes(enc):
    bits = np.unpackbits(np.asarray(enc, np.uint8), bitorder="big")
    cw = bits[: bits.size // 7 * 7].reshape(-1, 7).copy()
    s = (cw[:, 0] ^ cw[:, 2] ^ cw[:, 4] ^ cw[:, 6]) | ((cw[:, 1] ^ cw[:, 2] ^ cw[:, 5] ^ cw[:, 6]) << 1) \
        | ((cw[:, 3] ^ cw[:, 4] ^ cw[:, 5] ^ cw[:, 6]) << 2)
    bad = np.nonzero(s)[0]
    cw[bad, s[bad].astype(np.int64) - 1] ^= 1
    nib = (cw[:, 2] << 3) | (cw[:, 4] << 2) | (cw[:, 5] << 1) | cw[:, 6]
    return ((nib[0::2] << 4) | nib[1::2]).astype(np.uint8)


def _residual_table():
    """4-bit data-error pattern (D1..D4 = bits 3..0) -> data error left after Hamming(7,4) decoding.
    The code is linear, so the residual does not depend on the data."""
    tab = np.zeros(16, np.uint8)
    for e in range(16):
        bits = np.unpackbits(hamming_encode_bytes(np.array([0x00], np.uint8)), bitorder="big")
        for i, pos in enumerate([2, 4, 5, 6]):
            if (e >> (3 - i)) & 1:
                bits[pos] ^= 1
        tab[e] = hamming_decode_bytes(np.packbits(bits, bitorder="big"))[0] >> 4
    return tab


RESIDUAL = _residual_table()     # 1-bit errors -> 0 (repaired); 2+ bits -> left or mis-corrected


# ------------------------------------------------------------------ transmission
class Link:
    """Rows = packet*12 + slot, columns = RB rule bits in LITERAL order (column l = literal l; L..RB-1 padding).
    For each Hamming layout: idx (ncw, 4) flat positions D1..D4 of every codeword, and for every flat position the
    codeword it sits in and its slot (0 = D1 .. 3 = D4)."""

    def __init__(self, m):
        C, Cl, L = m["cube"].shape
        self.C, self.Cl, self.L = C, Cl, L
        self.RB = -(-L // 8) * 8
        ppc = -(-Cl // RULES_PER_PACKET)                          # packets per class
        self.npk = C * ppc
        self.R = self.npk * RULES_PER_PACKET
        r = np.arange(self.R)
        self.row_c, self.row_j = r // (ppc * RULES_PER_PACKET), r % (ppc * RULES_PER_PACKET)
        self.real = self.row_j < Cl                               # padding rules: empty, weight 0
        self.rows = np.zeros((self.R, self.RB), np.uint8)
        self.rows[self.real, :L] = m["cube"][self.row_c[self.real], self.row_j[self.real]]
        self.rb = RULES_PER_PACKET * self.RB                      # rule bits per packet
        self.lay = {}
        B = np.arange(self.rb // 8)
        for lay in ("SEQ", "INT", "KEY"):
            order = m["int_order"] if lay == "INT" else np.arange(L)
            col_of_pos = np.r_[order, np.arange(L, self.RB)]
            u = np.arange(self.rb)
            idx = np.empty((self.npk, len(B) * 2, 4), np.int64)
            for s in range(self.npk):
                o = self.key_perm(s) if lay == "KEY" else u       # original rule-area bit carried at position u
                flat = (s * RULES_PER_PACKET + o // self.RB) * self.RB + col_of_pos[o % self.RB]
                hi = np.stack([flat[8 * B + 7], flat[8 * B + 6], flat[8 * B + 5], flat[8 * B + 4]], 1)
                lo = np.stack([flat[8 * B + 3], flat[8 * B + 2], flat[8 * B + 1], flat[8 * B + 0]], 1)
                idx[s] = np.stack([hi, lo], 1).reshape(-1, 4)
            idx = idx.reshape(-1, 4)
            cw_of = np.empty(self.R * self.RB, np.int64)
            slot_of = np.empty(self.R * self.RB, np.uint8)
            cw_of[idx.ravel()] = np.repeat(np.arange(len(idx)), 4)
            slot_of[idx.ravel()] = np.tile(np.arange(4), len(idx))
            self.lay[lay] = (idx, cw_of, slot_of)

    def key_perm(self, seq):
        return np.random.default_rng([SECRET_KEY, seq]).permutation(self.rb)

    def attack(self, atk, lits=None, k=0, rng=None):
        """Flat positions the MITM changes (and their new values come from XOR, i.e. positions that really flip)."""
        if atk == "A1":
            cols = np.argsort(rng.random((self.R, self.L)), axis=1)[:, :k]
            return (np.arange(self.R)[:, None] * self.RB + cols).ravel()
        lits = np.asarray(lits, np.int64)
        flat = (np.arange(self.R)[:, None] * self.RB + lits[None, :]).ravel()
        cur = self.rows.ravel()[flat]
        if atk == "A2":
            return flat[cur == 0]
        if atk == "A3":
            return flat[cur == 1]
        return flat                                              # A4 toggle

    def receive(self, lay, flips):
        """Rows rebuilt by the receiver after the MITM flipped `flips` and the receiver decoded `lay`.
        Returns (rows, histogram of codewords with 1/2/3/4 changed bits or None)."""
        dec = self.rows.ravel().copy()
        if lay == "none":
            dec[flips] ^= 1
            return dec.reshape(self.rows.shape), None
        idx, cw_of, slot_of = self.lay[lay]
        cws, inv = np.unique(cw_of[flips], return_inverse=True)
        e = np.zeros(len(cws), np.uint8)
        np.bitwise_or.at(e, inv, (8 >> slot_of[flips]).astype(np.uint8))
        res = RESIDUAL[e]
        for i in range(4):
            sel = (res >> (3 - i)) & 1 == 1
            dec[idx[cws[sel], i]] ^= 1
        nb = np.unpackbits(e[:, None], axis=1)[:, 4:].sum(1)
        return dec.reshape(self.rows.shape), np.bincount(nb, minlength=5)[1:5]

    def to_cube(self, rows):
        cube = np.zeros((self.C, self.Cl, self.L), np.uint8)
        cube[self.row_c[self.real], self.row_j[self.real]] = rows[self.real, :self.L]
        return cube

    def reference_receive(self, lay, flips, packets):
        """Slow bit-level reference for `packets`: build packet bytes, Hamming-encode, flip data bits in the
        encoded stream, Hamming-decode, undo layout. Used to validate receive()."""
        out = []
        att = self.rows.ravel().copy(); att[flips] ^= 1
        att = att.reshape(self.rows.shape)
        order = None
        for s in packets:
            rr = slice(s * RULES_PER_PACKET, (s + 1) * RULES_PER_PACKET)
            def to_area(rows):
                bits = rows.copy()
                if lay == "INT":
                    bits[:, :self.L] = rows[:, self._int_order]
                b = bits.reshape(-1)
                if lay == "KEY":
                    b = b[self.key_perm(s)]
                return np.packbits(b, bitorder="little")
            clean_bytes, att_bytes = to_area(self.rows[rr]), to_area(att[rr])
            if lay == "none":
                got = att_bytes
            else:
                enc = hamming_encode_bytes(clean_bytes)
                diff = np.unpackbits(clean_bytes ^ att_bytes, bitorder="big")            # data bits that changed
                ebits = np.unpackbits(enc, bitorder="big")
                nib_bit = np.arange(diff.size)                                           # bit i of the byte stream (MSB first)
                cw, d = nib_bit // 4, nib_bit % 4                                        # codeword, D1..D4
                pos = cw * 7 + np.array([2, 4, 5, 6])[d]
                ebits[pos[diff == 1]] ^= 1
                got = hamming_decode_bytes(np.packbits(ebits, bitorder="big"))
            b = np.unpackbits(got, bitorder="little")
            if lay == "KEY":
                o = np.empty_like(b); o[self.key_perm(s)] = b; b = o
            rows = b.reshape(RULES_PER_PACKET, self.RB).copy()
            if lay == "INT":
                r2 = rows.copy(); r2[:, self._int_order] = rows[:, :self.L]; rows = r2
            out.append(rows)
        return np.concatenate(out)


# ------------------------------------------------------------------ inference
def pack_rows(bits2d, W64):
    pad = np.zeros((bits2d.shape[0], W64 * 64), np.uint8)
    pad[:, :bits2d.shape[1]] = bits2d
    return np.packbits(pad, axis=1, bitorder="little").view(np.uint64)


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


@njit(parallel=True, cache=True)
def _delta_sums(Xw, old_w, new_w, ne_old, ne_new, cls, wt, n_cls):
    N = Xw.shape[0]
    out = np.zeros((N, n_cls), np.int32)
    for i in prange(N):
        for j in range(old_w.shape[0]):
            f_old = ne_old[j]
            if f_old:
                for w in range(old_w.shape[1]):
                    if old_w[j, w] & ~Xw[i, w]:
                        f_old = False
                        break
            f_new = ne_new[j]
            if f_new:
                for w in range(new_w.shape[1]):
                    if new_w[j, w] & ~Xw[i, w]:
                        f_new = False
                        break
            if f_old != f_new:
                out[i, cls[j]] += wt[j] if f_new else -wt[j]
    return out


class Engine:
    """Class sums on the whole test set; incremental when few clauses differ from the clean model."""

    def __init__(self, m):
        self.m = m
        C, Cl, L = m["cube"].shape
        self.L, self.W64 = L, -(-L // 64)
        self.X = pack_rows(np.concatenate([m["Xte"], 1 - m["Xte"]], axis=1).astype(np.uint8), self.W64)
        self.clean_sums = self.full(m["cube"])

    def full(self, cube):
        C, Cl, L = cube.shape
        Cw = pack_rows(cube.reshape(C * Cl, L), self.W64).reshape(C, Cl, self.W64)
        return _class_sums(self.X, Cw, cube.any(axis=2), self.m["weights"])

    def sums(self, cube):
        clean = self.m["cube"]
        diff = (clean != cube).any(axis=2)
        c, r = np.nonzero(diff)
        if len(c) == 0:
            return self.clean_sums
        if len(c) > 0.4 * diff.size:
            return self.full(cube)
        d = _delta_sums(self.X, pack_rows(clean[c, r], self.W64), pack_rows(cube[c, r], self.W64),
                        clean[c, r].any(1), cube[c, r].any(1), c.astype(np.int64),
                        self.m["weights"][c, r].astype(np.int32), clean.shape[0])
        return self.clean_sums + d


def setup(ds):
    m = load(ds)
    link = Link(m)
    link._int_order = m["int_order"]
    eng = Engine(m)
    if m.get("sums_ref") is not None:
        assert np.array_equal(eng.clean_sums, m["sums_ref"]), "class sums differ from tmu"
    assert np.array_equal(link.to_cube(link.rows), m["cube"])
    pred0 = eng.clean_sums.argmax(1)
    return m, link, eng, pred0
