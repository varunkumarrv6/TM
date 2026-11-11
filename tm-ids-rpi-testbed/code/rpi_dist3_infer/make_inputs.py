"""Inputs for the distributed 3-TM Raspberry Pi testbed (MedSec-25, 70/15/15 split, 3 node TMs from
figures/train5_701515.py with NTM=3: 5 classes x 900 clauses each, 356 literals, weighted, T=623, s=4.64).

Weights: 4-bit two's-complement nibbles (-8..+7) after dividing by ONE common scale for all three TMs,
    s = max|w| / 7 over the 3 TMs,   w4 = clip(round(w / s), -8, 7)
The header flags the weights as normalised and carries s (Q8.8); the receiver infers with w4 (a common positive
scale does not change the argmax of the fused sum).

Datagram of TM n (same format as rpi_bench_infer, node id = n):
  clear header, 8 bytes
      [0] class id   [1] real rules in this packet (1..12; the rest are empty padding rules, never fire)
      [2..3] packet index inside this TM's stream (uint16 LE)
      [4] bit 7 = weights normalised, bits 3..0 = w4   [5..6] scale s, Q8.8 (uint16 LE)
      [7] bits 7..4 = node id n, bit 0 = layout (0 raw / no ECC, 1 Hamming(7,4) SEQ)
  payload 548 bytes = 12 rules x 45 bytes (356 literal bits, LSB first, 4 padding bits) + copy of the header
  Hamming(7,4) (rpi_customarchi.c): byte -> high-nibble codeword, low-nibble codeword, P1 P2 D1 P4 D2 D3 D4 packed
      MSB first; 548 -> 959 bytes.  Datagram = 967 bytes (Hamming) / 556 bytes (raw).
On the wire the sender puts a 2-byte round number (uint16 LE) in front of every datagram (transport field, not
part of the model): 969 / 558 bytes per UDP packet. Every round = one complete transmission of all 3 models.

Test flows: 50 flows of the 70/15/15 TEST split, 10 per class (seed 7).

Files (this folder):
  tm{0,1,2}_hamming.bin, tm{0,1,2}_raw.bin   datagram streams of the 3 sending RPis (no round prefix)
  X_50.bin        50 x 178 uint8 booleanized features (row-major)
  y_50_actual.bin true class (50 x uint8)          y_50_ref.bin  prediction of the clean fused 4-bit model
  sums_ref.bin    clean class sums, int32 [3 nodes][50][5] (fused = sum over nodes)
  *.npy           the same as numpy arrays;  dist3_config.h  sizes for receiver.c;  expected.txt  summary
usage: python make_inputs.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "/home/<user>/figures/models/medsec"
N_NODES, NS_PER_CLASS = 3, 10
RPP, HDR, RB_BYTES = 12, 8, 45
PAY = RPP * RB_BYTES + HDR                       # 548
ENC = PAY * 14 // 8                              # 959
assert (PAY * 14) % 8 == 0

nodes = [np.load(f"{MODEL}/tm3_node{n}.npz") for n in range(N_NODES)]
d = np.load(f"{MODEL}/data.npz", allow_pickle=True)
cubes = [m["cube"].astype(np.uint8) for m in nodes]
Ws = [m["weights"].astype(np.int64) for m in nodes]
C, CL, L = cubes[0].shape
assert L <= RB_BYTES * 8

# ---- one common 4-bit scale for the 3 TMs
S_Q88 = int(round(max(np.abs(W).max() for W in Ws) / 7 * 256))
SCALE = S_Q88 / 256
W4s = [np.clip(np.rint(W / SCALE), -8, 7).astype(np.int64) for W in Ws]


def hamming_encode(raw):
    nib = np.empty(raw.size * 2, np.uint8)
    nib[0::2], nib[1::2] = raw >> 4, raw & 0x0F
    d1, d2, d3, d4 = (nib >> 3) & 1, (nib >> 2) & 1, (nib >> 1) & 1, nib & 1
    cw = np.stack([d1 ^ d2 ^ d4, d1 ^ d3 ^ d4, d1, d2 ^ d3 ^ d4, d2, d3, d4], axis=1)
    return np.packbits(cw.reshape(-1), bitorder="big")


def class_sums(X, cube, weights, chunk=4000):
    """Raw class sums (empty clauses never fire) for boolean inputs X."""
    out = np.zeros((len(X), C), np.int64)
    ne = cube.any(-1)
    for a in range(0, len(X), chunk):
        lit = np.concatenate([X[a:a + chunk], 1 - X[a:a + chunk]], 1).astype(np.uint8)
        viol = np.einsum("nl,ckl->nck", 1 - lit, cube, dtype=np.int32) > 0
        out[a:a + chunk] = ((~viol & ne[None]) * weights[None]).sum(-1)
    return out


def build_stream(n, layout):
    rules = np.packbits(cubes[n], axis=2, bitorder="little")        # (C, CL, 45)
    out, count = [], 0
    for c in range(C):
        for wv in np.unique(W4s[n][c]):
            idx = np.flatnonzero(W4s[n][c] == wv)
            for p in range(0, len(idx), RPP):
                chunk = idx[p:p + RPP]
                h = np.zeros(HDR, np.uint8)
                h[0], h[1] = c, len(chunk)
                h[2], h[3] = count & 0xFF, count >> 8
                h[4] = 0x80 | (int(wv) & 0x0F)
                h[5], h[6] = S_Q88 & 0xFF, S_Q88 >> 8
                h[7] = (n << 4) | layout
                pay = np.zeros(PAY, np.uint8)
                pay[: len(chunk) * RB_BYTES] = rules[c, chunk].reshape(-1)
                pay[RPP * RB_BYTES:] = h
                out += [h, hamming_encode(pay) if layout == 1 else pay]
                count += 1
    return np.concatenate(out), count


pk = []
for n in range(N_NODES):
    ham, P = build_stream(n, 1)
    raw, P2 = build_stream(n, 0)
    assert P == P2 and P < 512
    ham.tofile(f"{HERE}/tm{n}_hamming.bin"); raw.tofile(f"{HERE}/tm{n}_raw.bin")
    pk.append(P)

# ---- 50 test flows, 10 per class
Xte, yte = d["Xte"].astype(np.uint8), d["yte"].astype(np.int64)
rng = np.random.default_rng(7)
sel = np.concatenate([np.sort(rng.choice(np.flatnonzero(yte == c), NS_PER_CLASS, replace=False)) for c in range(C)])
X50, y50 = Xte[sel], yte[sel]
NS = len(sel)

sums = np.stack([class_sums(X50, cubes[n], W4s[n]) for n in range(N_NODES)])       # (3, 50, 5)
ref = sums.sum(0).argmax(1)
full = sum(class_sums(Xte, cubes[n], W4s[n]) for n in range(N_NODES)).argmax(1)       # whole test split, 4-bit
full_w = sum(class_sums(Xte, cubes[n], Ws[n]) for n in range(N_NODES)).argmax(1)      # whole test split, full w

X50.tofile(f"{HERE}/X_50.bin"); y50.astype(np.uint8).tofile(f"{HERE}/y_50_actual.bin")
ref.astype(np.uint8).tofile(f"{HERE}/y_50_ref.bin"); sums.astype(np.int32).tofile(f"{HERE}/sums_ref.bin")
np.save(f"{HERE}/X_50.npy", X50); np.save(f"{HERE}/y_50_actual.npy", y50); np.save(f"{HERE}/y_50_ref.npy", ref)
np.save(f"{HERE}/sums_ref.npy", sums.astype(np.int32)); np.save(f"{HERE}/test_indices.npy", sel)

with open(f"{HERE}/dist3_config.h", "w") as f:
    f.write(f"""/* generated by make_inputs.py -- distributed 3-TM MedSec-25 (70/15/15), 4-bit normalised weights */
#define N_NODES       {N_NODES}
#define NUM_CLASSES   {C}
#define FEATURES      {L // 2}
#define LITERALS      {L}
#define RULE_BYTES    {RB_BYTES}
#define RULES_PER_PACKET {RPP}
#define HDR_BYTES     {HDR}
#define PAYLOAD_BYTES {PAY}
#define ENC_BYTES     {ENC}
#define PREFIX_BYTES  2         /* round number in front of every UDP packet */
#define NS            {NS}
#define MAX_PACKETS   512
#define WEIGHT_SCALE_Q88 {S_Q88}   /* s = {SCALE:.6f} */
static const int PACKETS_PER_NODE[N_NODES] = {{{", ".join(map(str, pk))}}};
""")
names = [str(c) for c in d["class_names"]]
txt = f"""distributed 3-TM MedSec-25 (70/15/15), 4-bit weights, scale s = {SCALE:.6f} (Q8.8 {S_Q88})
packets per TM      {pk}  (total {sum(pk)}), datagram 967 B Hamming / 556 B raw (+2 B round prefix on the wire)
classes             {names}
50 test flows       10 per class, indices in test_indices.npy
clean fused 4-bit   50 flows: {(ref == y50).mean()*100:.2f} % ({int((ref == y50).sum())}/50)
whole test split    4-bit fused {(full == yte).mean()*100:.4f} %, full-weight fused {(full_w == yte).mean()*100:.4f} %
per-node test acc   {[round(float(m['test_acc']), 4) for m in nodes]} (tmu, full weights)
"""
open(f"{HERE}/expected.txt", "w").write(txt)
print(txt)
