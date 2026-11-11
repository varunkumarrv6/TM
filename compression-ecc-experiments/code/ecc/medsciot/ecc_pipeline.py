"""File-level ECC + TM inference pipeline for the 3-TM MedSec-25 system, in the packet format of
error_coding.ipynb cell 53 (sender) and rpi_customarchi.c (receiver).

  SENDER    each TM's clause bank -> packets of 12 rules that share one class and one weight -> Hamming(7,4)
            -> tx/tm{n}_{layout}.bin  (a stream of UDP datagrams)
  MITM      edits literal bits of the rules inside the .bin bytes (never touches headers or parity bits)
  RECEIVER  model-blind: reads only the .bin files, Hamming-decodes, checks headers, rebuilds banks + weights
  INFERENCE manual TM inference on the rebuilt banks, fused over the 3 TMs

Weights.  The TMs are weighted (integer weights up to |212|).  Each weight is stored as a signed 4-bit nibble
(two's complement, -8..+7) after dividing by one common scale s = max|w| / 7 for all three TMs; the class-sum
threshold becomes T_q = round(T / s).  This costs 0.007 % accuracy (see notebook).

UDP datagram (as rpi_customarchi.c expects: 4-byte header + Hamming payload):
  clear header, 4 bytes
      byte 0   : bits 7-4 = weight nibble (signed -8..+7), bits 3-0 = class id
      byte 1-2 : sequence number, little endian (packet index inside this TM's stream)
      byte 3   : TM node id   (the "flags/reserved" byte of cell 53)
  payload before Hamming, 544 bytes
      bytes 0-539   : 12 rules x 45 bytes (356 literals, LSB-first, 4 padding bits)  -> read from offset 0 like the C code
      bytes 540-543 : copy of the 4-byte header (Hamming protected; receiver checks it against the clear header)
  Hamming(7,4) as rpi_customarchi.c: every byte -> high nibble codeword, low nibble codeword; nibble bit 3 = D1;
  codeword P1 P2 D1 P4 D2 D3 D4 packed MSB-first.  544 bytes -> 952 bytes.  Datagram = 4 + 952 = 956 bytes.
Every (class, weight) group is cut into packets of 12 rules; the last packet of a group is padded with empty rules
(all-zero rules never fire, exactly like the C receiver skips empty clauses).

Layouts (which 4 literal bits share one Hamming codeword):
  none : no Hamming, datagram = header + raw payload (548 bytes)
  SEQ  : literals in tmu order -> 4 neighbouring bits of the same feature share a codeword
  INT  : literals reordered inside each rule so a codeword holds 4 different features
  KEY  : the 4320 rule bits of every packet shuffled with a secret key before Hamming
"""
import os
from functools import lru_cache
import numpy as np
from numba import njit, prange

BASE = "/home/<user>/medsciot"
TX_DIR = os.path.join(BASE, "tx")
N_NODES = 3
NUM_CLASSES = 5
CLAUSES = 1000                       # real clauses per class in every TM
HEADER_BYTES = 4
RULE_BYTES = 45                      # 356 literals -> 45 bytes (4 padding bits)
RULES_PER_PACKET = 12                # as in cell 53 / rpi_customarchi.c
RULE_AREA = RULES_PER_PACKET * RULE_BYTES                         # 540
PAYLOAD_BYTES = RULE_AREA + HEADER_BYTES                          # 544 (rules + protected header copy)
ENC_PAYLOAD = PAYLOAD_BYTES * 14 // 8                             # 952
DG_ENC = HEADER_BYTES + ENC_PAYLOAD                               # 956 bytes per datagram with Hamming
DG_RAW = HEADER_BYTES + PAYLOAD_BYTES                             # 548 bytes per datagram without Hamming
RULE_BITS = RULE_AREA * 8                                         # 4320
SECRET_KEY = 0xC0FFEE                # shared by TMs and inference node only (KEY layout)
LAYOUTS = ["none", "SEQ", "INT", "KEY"]
W_MIN, W_MAX = -8, 7                 # signed 4-bit weight nibble
assert (PAYLOAD_BYTES * 14) % 8 == 0

D = np.load(os.path.join(BASE, "data.npz"), allow_pickle=True)
Xte, yte = D["Xte"], D["yte"].astype(np.int64)
class_names, feat_names = D["class_names"], D["feat_names"]
_lf = D["lit_feature"]
O = Xte.shape[1]
L = 2 * O
assert L <= RULE_BYTES * 8
LIT_RAW = np.concatenate([_lf, _lf])                                  # literal -> raw feature
LIT_POL = np.r_[np.zeros(O, int), np.ones(O, int)]
_rank = np.concatenate([np.arange((_lf == j).sum()) for j in np.unique(_lf)])
LIT_RANK = np.r_[_rank, _rank]
LIT_NAME = np.array([f"{'NOT ' if LIT_POL[k] else ''}{feat_names[LIT_RAW[k]]} b{LIT_RANK[k]}" for k in range(L)])
INT_ORDER = np.lexsort((LIT_RAW, LIT_POL, LIT_RANK))   # INT: position p carries literal INT_ORDER[p]


# ------------------------------------------------------------------ model: 4-bit weights, transmission order
@lru_cache(maxsize=1)
def _model():
    nodes = [np.load(os.path.join(BASE, f"node{n}.npz")) for n in range(N_NODES)]
    T = int(nodes[0]["T"])
    w_full = [nd["weights"].astype(np.int32) for nd in nodes]
    scale = max(np.abs(w).max() for w in w_full) / W_MAX
    T_q = int(round(T / scale))
    cubes, weights, real, schedules, origin = [], [], [], [], []
    for n, nd in enumerate(nodes):
        cube0 = nd["cube"].astype(np.uint8)
        wq0 = np.clip(np.rint(w_full[n] / scale), W_MIN, W_MAX).astype(np.int32)
        rows_c, sched = [], []
        for c in range(NUM_CLASSES):
            rows = []                                   # (original clause index or -1 for padding, weight)
            for wv in sorted(set(wq0[c].tolist())):     # one group per weight value, as cell 53 groups by weight
                idx = np.where(wq0[c] == wv)[0].tolist()
                idx += [-1] * (-len(idx) % RULES_PER_PACKET)
                for p in range(0, len(idx), RULES_PER_PACKET):
                    sched.append((c, wv, len(rows) + p))
                rows += [(i, wv) for i in idx]
            rows_c.append(rows)
        P = max(len(r) for r in rows_c)
        cube = np.zeros((NUM_CLASSES, P, L), np.uint8)
        wq = np.zeros((NUM_CLASSES, P), np.int32)
        is_real = np.zeros((NUM_CLASSES, P), bool)
        orig = -np.ones((NUM_CLASSES, P), int)
        for c, rows in enumerate(rows_c):
            for r, (i, wv) in enumerate(rows):
                wq[c, r] = wv
                if i >= 0:
                    cube[c, r], is_real[c, r], orig[c, r] = cube0[c, i], True, i
        cubes.append(cube); weights.append(wq); real.append(is_real); origin.append(orig)
        schedules.append(np.array(sched, dtype=np.int64))      # rows: (class, weight, first row of the packet)
    return dict(T=T, T_q=T_q, scale=scale, cubes=cubes, weights=weights, real=real, origin=origin,
                schedules=schedules, w_full=w_full,
                cubes_full=[nd["cube"].astype(np.uint8) for nd in nodes],
                tmu_sums=[nd["test_sums"].astype(np.int32) for nd in nodes])


def load_nodes():
    """cubes/weights are the DEPLOYED model in transmission order (4-bit weights, padded groups).
    T is the threshold for those weights. clean_sums = class sums of the deployed model, no attack."""
    m = dict(_model())
    m["T_full"], m["T"] = m["T"], m["T_q"]
    if "clean_sums" not in _model():
        _model()["clean_sums"] = [class_sums_all(c, w) for c, w in zip(m["cubes"], m["weights"])]
    m["clean_sums"] = _model()["clean_sums"]
    return m


# ------------------------------------------------------------------ Hamming(7,4) on bytes (rpi_customarchi.c)
def hamming_encode_bytes(raw):
    raw = np.asarray(raw, np.uint8)
    nib = np.empty(raw.size * 2, np.uint8)
    nib[0::2], nib[1::2] = raw >> 4, raw & 0x0F                  # high nibble first
    d1, d2, d3, d4 = (nib >> 3) & 1, (nib >> 2) & 1, (nib >> 1) & 1, nib & 1     # D1 = nibble MSB
    cw = np.stack([d1 ^ d2 ^ d4, d1 ^ d3 ^ d4, d1, d2 ^ d3 ^ d4, d2, d3, d4], axis=1)
    return np.packbits(cw.reshape(-1), bitorder="big")


def hamming_decode_bytes(enc):
    """Returns (raw bytes, codewords with non-zero syndrome)."""
    bits = np.unpackbits(np.asarray(enc, np.uint8), bitorder="big")
    cw = bits[: bits.size // 7 * 7].reshape(-1, 7).copy()
    s = (cw[:, 0] ^ cw[:, 2] ^ cw[:, 4] ^ cw[:, 6]) | ((cw[:, 1] ^ cw[:, 2] ^ cw[:, 5] ^ cw[:, 6]) << 1) \
        | ((cw[:, 3] ^ cw[:, 4] ^ cw[:, 5] ^ cw[:, 6]) << 2)
    bad = np.nonzero(s)[0]
    cw[bad, s[bad].astype(np.int64) - 1] ^= 1
    nib = (cw[:, 2] << 3) | (cw[:, 4] << 2) | (cw[:, 5] << 1) | cw[:, 6]
    return ((nib[0::2] << 4) | nib[1::2]).astype(np.uint8), len(bad)


def key_perm(node, seq):
    return np.random.default_rng([SECRET_KEY, node, seq]).permutation(RULE_BITS)


def make_header(weight, class_id, seq, node):
    return np.array([((weight & 0x0F) << 4) | class_id, seq & 0xFF, seq >> 8, node], np.uint8)


def parse_header(h):
    w = int(h[0] >> 4)
    return (w - 16 if w > 7 else w), int(h[0] & 0x0F), int(h[1]) | (int(h[2]) << 8), int(h[3])


# ------------------------------------------------------------------ SENDER
def build_datagrams(node, cube, layout):
    """(n_packets, 4 header bytes) and (n_packets, 544 payload bytes) before Hamming."""
    M = _model()
    sched = M["schedules"][node]
    order = INT_ORDER if layout == "INT" else np.arange(L)
    hdr = np.zeros((len(sched), HEADER_BYTES), np.uint8)
    pay = np.zeros((len(sched), PAYLOAD_BYTES), np.uint8)
    for seq, (c, wv, r0) in enumerate(sched):
        bits = np.zeros((RULES_PER_PACKET, RULE_BYTES * 8), np.uint8)
        bits[:, :L] = cube[c, r0:r0 + RULES_PER_PACKET][:, order]
        rule_bytes = np.packbits(bits, axis=1, bitorder="little").reshape(-1)
        if layout == "KEY":
            rule_bytes = np.packbits(np.unpackbits(rule_bytes, bitorder="little")[key_perm(node, seq)], bitorder="little")
        hdr[seq] = make_header(int(wv), int(c), seq, node)
        pay[seq, :RULE_AREA] = rule_bytes
        pay[seq, RULE_AREA:] = hdr[seq]                          # protected copy of the header
    return hdr, pay


def encode_node_file(node, cube, weights, layout):
    """weights are carried by the header nibble of every packet (from the model's schedule)."""
    hdr, pay = build_datagrams(node, cube, layout)
    if layout == "none":
        data = np.concatenate([hdr, pay], axis=1).reshape(-1)
    else:
        data = np.concatenate([np.concatenate([h, hamming_encode_bytes(p)]) for h, p in zip(hdr, pay)])
    os.makedirs(TX_DIR, exist_ok=True)
    path = os.path.join(TX_DIR, f"tm{node}_{layout}.bin")
    data.tofile(path)
    return path


# ------------------------------------------------------------------ literal -> (file byte, bit mask)
@lru_cache(maxsize=None)
def literal_location(node, layout):
    """For every literal (class, row, literal) of the transmitted cube: byte offset and bit mask inside the
    .bin file (-1 for rows that are not real clauses). The MITM uses this to aim at literals."""
    M = _model()
    P = M["cubes"][node].shape[1]
    seq_of = -np.ones((NUM_CLASSES, P), np.int64)
    slot_of = np.zeros((NUM_CLASSES, P), np.int64)
    for seq, (c, wv, r0) in enumerate(M["schedules"][node]):
        seq_of[c, r0:r0 + RULES_PER_PACKET] = seq
        slot_of[c, r0:r0 + RULES_PER_PACKET] = np.arange(RULES_PER_PACKET)
    pos = np.argsort(INT_ORDER) if layout == "INT" else np.arange(L)             # bit position inside the rule
    seq = np.repeat(seq_of[:, :, None], L, axis=2)
    u = slot_of[:, :, None] * RULE_BYTES * 8 + pos[None, None, :]                # bit in rule area (LSB-first)
    if layout == "KEY":
        inv = np.empty((len(M["schedules"][node]), RULE_BITS), np.int64)
        for s in range(len(inv)):
            inv[s, key_perm(node, s)] = np.arange(RULE_BITS)
        u = inv[np.maximum(seq, 0), u]
    raw_byte, raw_bit = u // 8, u % 8
    if layout == "none":
        byte = seq * DG_RAW + HEADER_BYTES + raw_byte
        mask = (1 << raw_bit).astype(np.uint8)
    else:
        cw = 2 * raw_byte + (raw_bit < 4)                    # high nibble (bits 4-7) -> first codeword
        enc_bit = cw * 7 + np.array([6, 5, 4, 2])[raw_bit % 4]   # nibble bit 3 = D1 (pos 2) ... bit 0 = D4 (pos 6)
        file_bit = (seq * DG_ENC + HEADER_BYTES) * 8 + enc_bit
        byte, mask = file_bit // 8, (1 << (7 - file_bit % 8)).astype(np.uint8)
    real = M["real"][node][:, :, None] & np.ones(L, bool)
    return np.where(real, byte, -1), np.where(real, mask, 0).astype(np.uint8)


# ------------------------------------------------------------------ MITM
def mitm_attack(path_in, path_out, node, layout, attack, targets=(), k=1, seed=0):
    """Attacks act on every REAL clause (all classes) of one TM stream:
      A1 / TOG : flip k random literals per clause          INC : include k random excluded literals per clause
      EXC      : exclude k random included literals per clause
      A2 / A3 / A4      : include / exclude / toggle every literal (x and NOT x) of the raw features in targets
      L_INC / L_EXC / L_TOG : include / exclude / toggle the particular literals named in targets."""
    data = np.fromfile(path_in, np.uint8)
    byte, mask = literal_location(node, layout)
    real = byte[:, :, 0] >= 0
    if attack in ("A1", "INC", "EXC"):
        cur = np.zeros(byte.shape, bool)
        cur[real] = (data[byte[real]] & mask[real]) != 0
        eligible = np.broadcast_to(real[:, :, None], byte.shape).copy()
        if attack == "INC":
            eligible &= ~cur
        elif attack == "EXC":
            eligible &= cur
        rng = np.random.default_rng([seed, node] if attack == "A1" else [seed, node, 7])
        key = np.where(eligible, rng.random(byte.shape), np.inf)
        pick = np.argsort(key, axis=2)[:, :, :k]
        ok = np.take_along_axis(eligible, pick, 2).ravel()            # clause may have < k eligible literals
        b, m = np.take_along_axis(byte, pick, 2).ravel()[ok], np.take_along_axis(mask, pick, 2).ravel()[ok]
        op = {"A1": np.bitwise_xor, "INC": np.bitwise_or}.get(attack)
        if op is not None:
            op.at(data, b, m)
        else:
            np.bitwise_and.at(data, b, ~m)
    else:
        if attack.startswith("L_"):
            sel = np.isin(LIT_NAME, list(targets))
            assert sel.sum() == len(targets), f"unknown literal in {targets}"
        else:
            sel = np.isin(LIT_RAW, [int(np.where(feat_names == t)[0][0]) for t in targets])
        b, m = byte[:, :, sel][real], mask[:, :, sel][real]
        b, m = b.ravel(), m.ravel()
        if attack in ("A2", "L_INC"):
            np.bitwise_or.at(data, b, m)
        elif attack in ("A3", "L_EXC"):
            np.bitwise_and.at(data, b, ~m)
        elif attack in ("A4", "L_TOG"):
            np.bitwise_xor.at(data, b, m)
        else:
            raise ValueError(attack)
    changed = int(np.unpackbits(np.fromfile(path_in, np.uint8) ^ data).sum())
    data.tofile(path_out)
    return changed


# ------------------------------------------------------------------ MITM on single datagrams (the 4 attacks)
ATTACKS = {"A1": "random: flip K random literals in every clause",
           "A2": "force INCLUDE the targeted literal(s) in every clause",
           "A3": "force EXCLUDE the targeted literal(s) in every clause",
           "A4": "TOGGLE the targeted literal(s) in every clause"}


def _payload_bit_positions(layout, node, seq, rule_lits):
    """rule_lits: (12, n) literal indices -> (file-independent) bit index inside the datagram, and mask/byte."""
    pos = (np.argsort(INT_ORDER) if layout == "INT" else np.arange(L))[rule_lits]
    u = np.arange(RULES_PER_PACKET)[:, None] * RULE_BYTES * 8 + pos              # rule-area bit, LSB-first
    if layout == "KEY":                                                          # worst case: the attacker knows
        inv = np.empty(RULE_BITS, np.int64); inv[key_perm(node, seq)] = np.arange(RULE_BITS)   # where bits go
        u = inv[u]
    raw_byte, raw_bit = u // 8, u % 8
    if layout == "none":
        return HEADER_BYTES + raw_byte, (1 << raw_bit).astype(np.uint8)
    cw = 2 * raw_byte + (raw_bit < 4)
    bit = HEADER_BYTES * 8 + cw * 7 + np.array([6, 5, 4, 2])[raw_bit % 4]        # D1..D4 slots, parity untouched
    return bit // 8, (1 << (7 - bit % 8)).astype(np.uint8)


def mitm_datagram(dg, layout, attack, lits=(), k=1, rng=None):
    """Man-in-the-middle on ONE intercepted UDP datagram (header + payload). Reads the clear header, then edits
    literal bits of all 12 rules: A1 flips k random literals per rule, A2/A3/A4 include/exclude/toggle `lits`."""
    dg = dg.copy()
    _, _, seq, node = parse_header(dg[:HEADER_BYTES])
    if attack == "A1":
        rule_lits = np.argsort(rng.random((RULES_PER_PACKET, L)), axis=1)[:, :k]
    else:
        rule_lits = np.tile(np.asarray(lits, int), (RULES_PER_PACKET, 1))
    b, m = _payload_bit_positions(layout, node, seq, rule_lits)
    b, m = b.ravel(), m.ravel()
    if attack in ("A1", "A4"):
        np.bitwise_xor.at(dg, b, m)
    elif attack == "A2":
        np.bitwise_or.at(dg, b, m)
    elif attack == "A3":
        np.bitwise_and.at(dg, b, ~m)
    else:
        raise ValueError(attack)
    return dg


def mitm_stream(path_in, path_out, layout, attack, lits=(), k=1, seed=0):
    """Intercepts every datagram of one TM's stream. Returns the number of bits the attacker changed."""
    data = np.fromfile(path_in, np.uint8)
    size = DG_RAW if layout == "none" else DG_ENC
    rng = np.random.default_rng(seed)
    out = np.concatenate([mitm_datagram(data[s:s + size], layout, attack, lits, k, rng)
                          for s in range(0, data.size, size)])
    out.tofile(path_out)
    return int(np.unpackbits(data ^ out).sum())


# ------------------------------------------------------------------ RECEIVER (model-blind)
def decode_node_file(path, expected_node, layout):
    """Reads only the .bin file. Returns (cube, weights, codewords corrected)."""
    data = np.fromfile(path, np.uint8)
    size = DG_RAW if layout == "none" else DG_ENC
    if data.size % size:
        raise RuntimeError(f"{path}: size {data.size} is not a whole number of datagrams")
    order_inv = np.argsort(INT_ORDER) if layout == "INT" else np.arange(L)
    groups = {c: [] for c in range(NUM_CLASSES)}                      # class -> list of (weight, 12 rules)
    flagged = 0
    for s in range(data.size // size):
        dg = data[s * size:(s + 1) * size]
        hdr = dg[:HEADER_BYTES]
        if layout == "none":
            pay = dg[HEADER_BYTES:].copy()
        else:
            pay, f = hamming_decode_bytes(dg[HEADER_BYTES:])
            flagged += f
        w, c, seq, node = parse_header(hdr)
        if not np.array_equal(hdr, pay[RULE_AREA:]) or node != expected_node or c >= NUM_CLASSES or seq != s:
            raise RuntimeError(f"{path}: bad header in packet {s} (weight {w}, class {c}, seq {seq}, node {node})")
        rules = pay[:RULE_AREA]
        if layout == "KEY":
            pb = np.unpackbits(rules, bitorder="little")
            orig = np.empty_like(pb)
            orig[key_perm(expected_node, s)] = pb
            rules = np.packbits(orig, bitorder="little")
        bits = np.unpackbits(rules.reshape(RULES_PER_PACKET, RULE_BYTES), axis=1, bitorder="little")[:, :L]
        groups[c].append((w, bits[:, order_inv]))
    P = max(len(g) for g in groups.values()) * RULES_PER_PACKET
    cube = np.zeros((NUM_CLASSES, P, L), np.uint8)
    weights = np.zeros((NUM_CLASSES, P), np.int32)
    for c, g in groups.items():
        for i, (w, bits) in enumerate(g):
            cube[c, i * RULES_PER_PACKET:(i + 1) * RULES_PER_PACKET] = bits
            weights[c, i * RULES_PER_PACKET:(i + 1) * RULES_PER_PACKET] = w
    return cube, weights, flagged


# ------------------------------------------------------------------ INFERENCE
def pack_literals(bits):
    """(R, L) {0,1} -> (R, 6) uint64 (same bit order as the bytes in the packets)."""
    pad = np.zeros((bits.shape[0], 384), np.uint8)
    pad[:, :bits.shape[1]] = bits
    return np.packbits(pad, axis=1, bitorder="little").view(np.uint64)


X_LIT = pack_literals(np.concatenate([Xte, 1 - Xte], axis=1).astype(np.uint8))


def tm_predict_manual(x_bool, cube, weights):
    """Readable numpy TM inference for ONE sample (as in error_coding.ipynb cell 48)."""
    lit = np.concatenate([x_bool, 1 - x_bool]).astype(np.uint8)
    violated = (cube == 1) & (lit[None, None, :] == 0)          # included literal that is false
    fires = ~violated.any(axis=2) & cube.any(axis=2)            # empty clauses never fire at inference
    return (weights * fires).sum(axis=1)


@njit(parallel=True, cache=True)
def _class_sums(Xw, Cw, nonempty, Wt):
    N, (Cc, Cl_, W) = Xw.shape[0], Cw.shape
    out = np.zeros((N, Cc), np.int32)
    for i in prange(N):
        for c in range(Cc):
            s = 0
            for j in range(Cl_):
                if nonempty[c, j]:
                    ok = True
                    for w in range(W):
                        if Cw[c, j, w] & ~Xw[i, w]:
                            ok = False
                            break
                    if ok:
                        s += Wt[c, j]
            out[i, c] = s
    return out


def class_sums_all(cube, weights):
    """Fast inference on the whole test set (same logic as tm_predict_manual)."""
    Cw = pack_literals(cube.reshape(-1, L)).reshape(cube.shape[0], cube.shape[1], -1)
    return _class_sums(X_LIT, Cw, cube.any(axis=2), weights.astype(np.int32))


def fuse(sums_list, T):
    return np.argmax(sum(np.clip(s, -T, T) for s in sums_list), axis=1)


# ------------------------------------------------------------------ one complete run
def run_pipeline(layout, attack=None, targets=(), k=1, attacked_nodes=(), seed=0, M=None, verbose=False):
    M = M or load_nodes()
    rx = []
    for n in range(N_NODES):
        path = encode_node_file(n, M["cubes"][n], M["weights"][n], layout)
        changed = 0
        if attack and n in attacked_nodes:
            out = path.replace(".bin", "_corrupted.bin")
            changed = mitm_attack(path, out, n, layout, attack, targets, k, seed)
            path = out
        cube, w, flagged = decode_node_file(path, n, layout)
        ref_c, ref_w = M["cubes"][n], M["weights"][n]
        P = max(cube.shape[1], ref_c.shape[1])
        cube_p = np.zeros((NUM_CLASSES, P, L), np.uint8); cube_p[:, :cube.shape[1]] = cube
        w_p = np.zeros((NUM_CLASSES, P), np.int32); w_p[:, :w.shape[1]] = w
        ref_cp = np.zeros_like(cube_p); ref_cp[:, :ref_c.shape[1]] = ref_c
        ref_wp = np.zeros_like(w_p); ref_wp[:, :ref_w.shape[1]] = ref_w
        rx.append(dict(path=path, cube=cube_p, weights=w_p, flagged=flagged, bits_changed=changed,
                       lit_mismatch=int((cube_p != ref_cp).sum()),
                       weight_mismatch=int(((w_p != ref_wp) & ref_cp.any(axis=2)).sum())))
        if verbose:
            print(f"TM {n}: {os.path.basename(path):24s} bits changed by MITM {changed:>9,} | "
                  f"codewords corrected {flagged:>8,} | literal errors left {rx[-1]['lit_mismatch']:>8,}")
    return rx
