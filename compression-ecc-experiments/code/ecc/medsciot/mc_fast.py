"""Fast in-memory equivalent of  encode -> MITM (common literal set in every rule) -> Hamming decode
for the C-format packets of ecc_pipeline.py, plus incremental TM inference. Used for the 150-iteration Monte-Carlo.
validate() checks it bit-for-bit against the real .bin pipeline."""
import numpy as np
from numba import njit, prange
import ecc_pipeline as E

RB = E.RULE_BYTES * 8          # 360 bit columns per rule (356 literals + 4 padding bits)


def _residual_table():
    """4-bit data-error pattern (D1..D4 as bits 3..0) -> 4-bit data error left after Hamming(7,4) decoding."""
    tab = np.zeros(16, np.uint8)
    for e in range(16):
        nib_err = np.array([e], np.uint8)
        enc = E.hamming_encode_bytes(np.array([0x00], np.uint8))           # codeword of byte 0x00
        bits = np.unpackbits(enc, bitorder="big")
        for i, pos in enumerate([2, 4, 5, 6]):                             # D1..D4 = nibble bits 3..0
            if (e >> (3 - i)) & 1:
                bits[pos] ^= 1                                             # error in the HIGH nibble codeword
        dec, _ = E.hamming_decode_bytes(np.packbits(bits, bitorder="big"))
        tab[e] = dec[0] >> 4
    return tab


RESIDUAL = _residual_table()


class NodeStream:
    """Logical view of one TM's transmitted rules: rows = packet*12 + rule slot, columns = 360 rule bits in
    LITERAL order (column l = literal l, 356..359 = padding). cw_idx maps every Hamming codeword of the rule
    areas to the 4 flat indices (D1..D4) it carries, for a given layout."""

    def __init__(self, M, n):
        sched = M["schedules"][n]
        self.n, self.sched = n, sched
        self.rows = np.zeros((len(sched) * E.RULES_PER_PACKET, RB), np.uint8)
        self.row_c = np.repeat(sched[:, 0], E.RULES_PER_PACKET)
        self.row_r = (sched[:, 2][:, None] + np.arange(E.RULES_PER_PACKET)).ravel()
        self.rows[:, :E.L] = M["cubes"][n][self.row_c, self.row_r]
        self.cube_shape = M["cubes"][n].shape
        self.cw = {}
        for lay in ("SEQ", "INT", "KEY"):
            order = E.INT_ORDER if lay == "INT" else np.arange(E.L)
            col_of_pos = np.r_[order, np.arange(E.L, RB)]                  # rule bit position -> column
            u = np.arange(E.RULE_BITS)                                    # transmitted rule-area bit
            idx = []
            for s in range(len(sched)):
                o = E.key_perm(n, s)[u] if lay == "KEY" else u            # original rule-area bit carried
                flat = (s * E.RULES_PER_PACKET + o // RB) * RB + col_of_pos[o % RB]
                B = np.arange(E.RULE_AREA)
                hi = np.stack([flat[8 * B + 7], flat[8 * B + 6], flat[8 * B + 5], flat[8 * B + 4]], 1)
                lo = np.stack([flat[8 * B + 3], flat[8 * B + 2], flat[8 * B + 1], flat[8 * B + 0]], 1)
                idx.append(np.stack([hi, lo], 1).reshape(-1, 4))
            self.cw[lay] = np.concatenate(idx)

    def receive(self, layout, attack, lits):
        """Rows the inference node rebuilds after the MITM sets/clears/toggles columns `lits` in every rule."""
        att = self.rows.copy()
        if attack == "A2":
            att[:, lits] = 1
        elif attack == "A3":
            att[:, lits] = 0
        else:
            att[:, lits] ^= 1
        if layout == "none":
            return att, None
        idx = self.cw[layout]
        err = (self.rows.ravel()[idx] ^ att.ravel()[idx]).astype(np.uint8)          # (ncw, 4)
        e = (err[:, 0] << 3) | (err[:, 1] << 2) | (err[:, 2] << 1) | err[:, 3]
        res = RESIDUAL[e]
        dec = self.rows.ravel().copy()
        hit = np.nonzero(res)[0]
        for i in range(4):
            sel = hit[(res[hit] >> (3 - i)) & 1 == 1]
            dec[idx[sel, i]] ^= 1
        nbits = np.unpackbits(e.astype(np.uint8)[:, None], axis=1)[:, 4:].sum(1)   # changed bits per codeword
        hist = np.bincount(nbits, minlength=5)[1:5]                                 # codewords with 1, 2, 3, 4 changed bits
        return dec.reshape(self.rows.shape), hist

    def to_cube(self, rows):
        cube = np.zeros(self.cube_shape, np.uint8)
        cube[self.row_c, self.row_r] = rows[:, :E.L]
        return cube


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


def class_sums_incremental(clean_cube, new_cube, weights, clean_sums):
    """clean_sums + contribution change of the clauses that differ (exactly equal to a full recomputation)."""
    diff = (clean_cube != new_cube).any(axis=2)
    c, r = np.nonzero(diff)
    if len(c) == 0:
        return clean_sums
    if len(c) > 0.5 * diff.size:
        return E.class_sums_all(new_cube, weights)
    old = E.pack_literals(clean_cube[c, r]); new = E.pack_literals(new_cube[c, r])
    d = _delta_sums(E.X_LIT, old, new, clean_cube[c, r].any(1), new_cube[c, r].any(1),
                    c.astype(np.int64), weights[c, r].astype(np.int32), clean_cube.shape[0])
    return clean_sums + d


def validate(M, streams, n_checks=3, seed=1):
    """Compare with the real file pipeline (encode -> mitm_stream -> decode_node_file)."""
    rng = np.random.default_rng(seed)
    ok = True
    for _ in range(n_checks):
        lits = sorted(rng.choice(E.L, 4, replace=False).tolist())
        for lay in E.LAYOUTS:
            for atk in ("A2", "A3", "A4"):
                n = 0
                src = E.encode_node_file(n, M["cubes"][n], M["weights"][n], lay)
                E.mitm_stream(src, src.replace(".bin", "_mc.bin"), lay, atk, lits)
                cube_file, w, _ = E.decode_node_file(src.replace(".bin", "_mc.bin"), n, lay)
                rows, _ = streams[n].receive(lay, atk, lits)
                cube_fast = streams[n].to_cube(rows)
                same = np.array_equal(cube_file[:, :cube_fast.shape[1]], cube_fast)
                s_full = E.class_sums_all(cube_fast, M["weights"][n])
                s_inc = class_sums_incremental(M["cubes"][n], cube_fast, M["weights"][n], M["clean_sums"][n])
                ok &= same and np.array_equal(s_full, s_inc)
    return ok
