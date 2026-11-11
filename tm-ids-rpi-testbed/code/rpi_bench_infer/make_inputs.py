"""Build the inputs for the Raspberry Pi inference benchmark of the MedSec-25 single TM (70/15/15 model:
5 classes x 900 clauses, 356 literals, weighted, T=623, s=4.64).

Weights are sent as 4-bit two's-complement nibbles (16 levels, -8..+7), normalised by one common scale
    s = max|w| / 7,   w4 = clip(round(w / s), -8, 7)      (as ecc_pipeline.py of the 3-node system)
The header says that the weights are normalised and carries s, so the receiver can de-normalise (w ~ w4 * s);
inference uses w4 directly (a positive common scale does not change the argmax).

Datagram stream (one file = everything one TM sends; read by every C kernel):
  clear header, 8 bytes
      [0] class id          [1] real rules in this packet (1..12; the rest are empty padding rules)
      [2..3] sequence no. (uint16, little endian)
      [4] bit 7 = weights normalised (1), bits 3..0 = 4-bit weight w4 (two's complement), bits 6..4 = 0
      [5..6] normalisation scale s in Q8.8 fixed point (uint16, little endian)
      [7] bits 7..4 = node id (0 = single TM), bit 0 = layout (0 = raw / no ECC, 1 = Hamming(7,4) SEQ)
  payload before ECC, 548 bytes = 12 rules x 45 bytes (356 literal bits, LSB first, 4 padding bits) + header copy
  Hamming(7,4) as rpi_customarchi.c: every byte -> high-nibble codeword then low-nibble codeword, codeword bits
      P1 P2 D1 P4 D2 D3 D4 packed MSB first as one bit stream; 548 bytes -> 959 bytes  (datagram = 967 bytes)
  The clauses of every class are grouped by their 4-bit weight (one weight per packet, like error_coding.ipynb
  cell 53), groups cut into packets of 12 rules.

Files written to data/:
  medsec_tm_hamming.bin        Hamming(7,4) stream (the benchmark input)
  medsec_tm_hamming_err.bin    same stream with single-bit errors in 1 % of the codewords (parity bits included);
                               must decode to exactly the same model
  medsec_tm_raw.bin            same packets without ECC (to measure what decoding costs)
  X_test.npy (NS x 178 uint8), y_true.npy (uint32), y_ref.npy (uint32, predictions of the 4-bit model),
  scores_ref.npy (NS x 5 int32 class sums of the 4-bit model), y_tmu.npy (full-weight tmu predictions, info only)
  expected.txt                 numbers the Pi must reproduce
and model_dims.h for the C code.
usage: python make_inputs.py [NS=256]
"""
import os
import sys
import numpy as np

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 256
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)
MODEL = "/home/<user>/figures/models/medsec"
RPP, HDR, RB_BYTES = 12, 8, 45
PAY = RPP * RB_BYTES + HDR                       # 548
ENC = PAY * 14 // 8                              # 959
assert (PAY * 14) % 8 == 0

t = np.load(f"{MODEL}/tm.npz")
d = np.load(f"{MODEL}/data.npz", allow_pickle=True)
cube, W = t["cube"].astype(np.uint8), t["weights"].astype(np.int64)
C, CL, L = cube.shape
O = L // 2
assert L <= RB_BYTES * 8

# ---- 4-bit normalised weights
S_Q88 = int(round(np.abs(W).max() / 7 * 256))                  # scale s in Q8.8 (what the header carries)
SCALE = S_Q88 / 256
W4 = np.clip(np.rint(W / SCALE), -8, 7).astype(np.int64)


def hamming_encode(raw):
    nib = np.empty(raw.size * 2, np.uint8)
    nib[0::2], nib[1::2] = raw >> 4, raw & 0x0F
    d1, d2, d3, d4 = (nib >> 3) & 1, (nib >> 2) & 1, (nib >> 1) & 1, nib & 1
    cw = np.stack([d1 ^ d2 ^ d4, d1 ^ d3 ^ d4, d1, d2 ^ d3 ^ d4, d2, d3, d4], axis=1)
    return np.packbits(cw.reshape(-1), bitorder="big")


def hamming_decode(enc):
    bits = np.unpackbits(enc, bitorder="big")
    cw = bits[: bits.size // 7 * 7].reshape(-1, 7).copy()
    s = (cw[:, 0] ^ cw[:, 2] ^ cw[:, 4] ^ cw[:, 6]) | ((cw[:, 1] ^ cw[:, 2] ^ cw[:, 5] ^ cw[:, 6]) << 1) \
        | ((cw[:, 3] ^ cw[:, 4] ^ cw[:, 5] ^ cw[:, 6]) << 2)
    bad = np.nonzero(s)[0]
    cw[bad, s[bad].astype(np.int64) - 1] ^= 1
    nib = (cw[:, 2] << 3) | (cw[:, 4] << 2) | (cw[:, 5] << 1) | cw[:, 6]
    return ((nib[0::2] << 4) | nib[1::2]).astype(np.uint8), len(bad)


def class_sums(X, weights, chunk=4000):
    """Raw class sums of the TM (empty clauses never fire) for boolean inputs X."""
    out = np.zeros((len(X), C), np.int64)
    ne = cube.any(-1)
    for a in range(0, len(X), chunk):
        lit = np.concatenate([X[a:a + chunk], 1 - X[a:a + chunk]], 1).astype(np.uint8)
        viol = np.einsum("nl,ckl->nck", 1 - lit, cube, dtype=np.int32) > 0       # included literal that is 0
        out[a:a + chunk] = ((~viol & ne[None]) * weights[None]).sum(-1)
    return out


rules = np.packbits(cube, axis=2, bitorder="little")          # (C, CL, 45), literal i -> byte i//8 bit i%8
assert rules.shape[2] == RB_BYTES

# ---- packets: per class, groups of equal 4-bit weight (ascending), 12 rules per packet
packets = []
for c in range(C):
    for wv in np.unique(W4[c]):
        idx = np.flatnonzero(W4[c] == wv)
        for p in range(0, len(idx), RPP):
            packets.append((c, int(wv), idx[p:p + RPP]))
assert len(packets) < 65536


def header(c, n_real, seq, w4, layout):
    h = np.zeros(HDR, np.uint8)
    h[0], h[1] = c, n_real
    h[2], h[3] = seq & 0xFF, seq >> 8
    h[4] = 0x80 | (w4 & 0x0F)                                   # normalised flag + 4-bit weight
    h[5], h[6] = S_Q88 & 0xFF, S_Q88 >> 8                       # scale s, Q8.8
    h[7] = (0 << 4) | layout                                    # node 0, layout
    return h


def stream(layout):
    out = []
    for seq, (c, wv, chunk) in enumerate(packets):
        h = header(c, len(chunk), seq, wv, layout)
        pay = np.zeros(PAY, np.uint8)
        pay[: len(chunk) * RB_BYTES] = rules[c, chunk].reshape(-1)
        pay[RPP * RB_BYTES:] = h
        out += [h, hamming_encode(pay) if layout == 1 else pay]
    return np.concatenate(out)


ham, raw = stream(1), stream(0)
DG = HDR + ENC
assert ham.size == len(packets) * DG and raw.size == len(packets) * (HDR + PAY)

# single-bit errors in 1 % of the codewords of the encoded payloads (headers stay clear)
rng = np.random.default_rng(2026)
bits = np.unpackbits(ham.reshape(len(packets), DG)[:, HDR:], axis=1, bitorder="big")[:, : PAY * 2 * 7]
hit = rng.random((len(packets), PAY * 2)) < 0.01
pk, cw = np.nonzero(hit)
bits[pk, cw * 7 + rng.integers(0, 7, len(cw))] ^= 1
err = ham.reshape(len(packets), DG).copy()
err[:, HDR:] = np.packbits(bits, axis=1, bitorder="big")
err = err.reshape(-1)
N_ERR = int(hit.sum())

# ---- reference receiver in Python (checks the files, gives the expected hash)
FNV_OFF, FNV_PRIME = 0xcbf29ce484222325, 0x100000001b3


def fnv(h, data):
    for b in bytes(data):
        h = ((h ^ b) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def receive(buf, layout):
    size = HDR + (ENC if layout == 1 else PAY)
    got = [[] for _ in range(C)]
    corrected = 0
    for s in range(buf.size // size):
        dg = buf[s * size:(s + 1) * size]
        h = dg[:HDR]
        if layout == 1:
            pay, k = hamming_decode(dg[HDR:]); corrected += k
        else:
            pay = dg[HDR:]
        assert np.array_equal(pay[RPP * RB_BYTES:], h) and int(h[2]) | int(h[3]) << 8 == s
        assert h[4] & 0x80 and (h[7] & 1) == layout and (int(h[5]) | int(h[6]) << 8) == S_Q88
        w4 = int(h[4] & 0x0F); w4 = w4 - 16 if w4 > 7 else w4
        for r in range(h[1]):
            got[h[0]].append((pay[r * RB_BYTES:(r + 1) * RB_BYTES].copy(), w4))
    hsh = FNV_OFF
    for c in range(C):
        for rule, wv in got[c]:
            hsh = fnv(hsh, rule.tobytes() + np.int16(wv).tobytes())
    return got, corrected, hsh


got, k0, H = receive(ham, 1)
got_e, k1, H_e = receive(err, 1)
got_r, k2, H_r = receive(raw, 0)
assert k0 == 0 and k1 == N_ERR and H == H_e == H_r
for c in range(C):                                              # rebuilt model == 4-bit model (as a multiset)
    assert sorted((r.tobytes(), w) for r, w in got[c]) == sorted((rules[c, k].tobytes(), int(W4[c, k])) for k in range(CL))

# ---- accuracy cost of 4-bit weights on the whole test split
Xall, yall = d["Xte"].astype(np.uint8), d["yte"].astype(np.int64)
full = class_sums(Xall, W)
assert np.array_equal(full, t["test_sums"])                     # numpy engine == tmu
q4 = class_sums(Xall, W4)
acc_full = (full.argmax(1) == yall).mean() * 100
acc_q4 = (q4.argmax(1) == yall).mean() * 100
agree = (full.argmax(1) == q4.argmax(1)).mean() * 100

# ---- test samples: stratified (NS // C flows per class, rest from the largest classes), seed 7
srng = np.random.default_rng(7)
sel = np.concatenate([srng.choice(np.flatnonzero(yall == c), NS // C + (c >= C - NS % C), replace=False)
                      for c in range(C)])
sel = srng.permutation(sel)
assert len(sel) == NS
X, y = Xall[sel], yall[sel].astype(np.uint32)
scores = q4[sel].astype(np.int32)
yref = scores.argmax(1).astype(np.uint32)
ytmu = t["test_pred"][sel].astype(np.uint32)

ham.tofile(f"{OUT}/medsec_tm_hamming.bin"); err.tofile(f"{OUT}/medsec_tm_hamming_err.bin")
raw.tofile(f"{OUT}/medsec_tm_raw.bin")
for old in ("scores_tmu.npy",):
    if os.path.exists(f"{OUT}/{old}"):
        os.remove(f"{OUT}/{old}")
np.save(f"{OUT}/X_test.npy", X); np.save(f"{OUT}/y_true.npy", y); np.save(f"{OUT}/y_ref.npy", yref)
np.save(f"{OUT}/scores_ref.npy", scores); np.save(f"{OUT}/y_tmu.npy", ytmu)
cl_max = max(len(g) for g in got)
cl_pad = -(-cl_max // 16) * 16
with open(f"{HERE}/model_dims.h", "w") as f:
    f.write(f"""/* generated by make_inputs.py -- MedSec-25 single TM (70/15/15 model), 4-bit normalised weights */
#define NUM_CLASSES   {C}
#define FEATURES      {O}
#define LITERALS      {L}
#define RULE_BYTES    {RB_BYTES}       /* {L} literal bits + {RB_BYTES * 8 - L} padding bits, LSB first */
#define CL_PAD        {cl_pad}       /* clause slots per class: {CL} clauses rounded up to 16 */
#define NS            {NS}       /* test samples in X_test.npy */
#define RULES_PER_PACKET {RPP}
#define HDR_BYTES     {HDR}
#define PAYLOAD_BYTES {PAY}      /* 12 rules x 45 bytes + header copy */
#define ENC_BYTES     {ENC}      /* Hamming(7,4) encoded payload */
#define N_PACKETS     {len(packets)}
#define WEIGHT_SCALE_Q88 {S_Q88}   /* s = {SCALE:.6f}: weights normalised to 4 bits, w ~ w4 * s */
#define MODEL_HASH    0x{H:016x}ULL   /* FNV-1a over received rules + 4-bit weights (as int16) */
""")
with open(f"{OUT}/expected.txt", "w") as f:
    f.write(f"""model                 MedSec-25 single TM, {C} classes x {CL} clauses, {L} literals
weights               normalised to 4 bits: s = max|w|/7 = {np.abs(W).max()}/7 -> Q8.8 {S_Q88} = {SCALE:.6f}, w4 in -8..7
                      (full weights {W.min()}..{W.max()}; {int((W4 == 0).sum())} clauses get w4 = 0)
packets               {len(packets)} datagrams (grouped by class and 4-bit weight), {DG} bytes each with Hamming
hamming stream        medsec_tm_hamming.bin  {ham.size} bytes, corrected codewords 0
error stream          medsec_tm_hamming_err.bin  corrected codewords {N_ERR} (1 % single-bit errors)
raw stream            medsec_tm_raw.bin  {raw.size} bytes
model hash            0x{H:016x}  (identical for all three streams)
test samples          {NS} (stratified over the 5 classes, seed 7, from the 70/15/15 test split)
reference             class sums of the 4-bit model must be identical, predictions {NS}/{NS}
accuracy on samples   {(yref == y).mean() * 100:.4f} % ({int((yref == y).sum())}/{NS}) with 4-bit weights; full weights {(ytmu == y).mean() * 100:.4f} %
whole test split      full weights {acc_full:.4f} %, 4-bit weights {acc_q4:.4f} % (change {acc_q4 - acc_full:+.4f} pp),
                      same prediction for {agree:.4f} % of the {len(yall)} test flows
""")
print(open(f"{OUT}/expected.txt").read())
