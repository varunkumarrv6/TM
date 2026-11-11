"""ECC under the MITM attack model: the same K random literal positions are flipped in EVERY rule (6 x 6000 rules
of 526 literals, bit-packed LSB-first to 66 bytes, rules stored back to back), then each code decodes.
Reports the literal bits still wrong after decoding and the share of rules left corrupted (mean of DRAWS draws).
The attack is identical in every rule, so the first N_RULES = 3600 rules (10 %, all 6 classes interleaved by taking
600 per class) are used; they cover every alignment of rules to BCH / RS codewords many times.

  Hamming(7,4) : byte -> 2 codewords (rpi_customarchi.c), i.e. 4 consecutive literals per codeword (SEQ layout)
  BCH(255,239), BCH(255,191) : 255-bit codewords over the packed bit stream (one codeword spans ~half a rule)
  RS(255,223)  : 255-byte codewords (16 byte errors correctable; one codeword spans ~3.4 rules)
usage: NUMBA_NUM_THREADS=8 python attack_ecc.py   -> results_attack.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import galois

sys.path.insert(0, "/home/<user>/figures")
import ecc_common as E

OUT = os.path.dirname(os.path.abspath(__file__))
K_LIST = [1, 2, 3, 4, 6, 8]
DRAWS = 5
packed = np.load("/home/<user>/iot_dataset/rules_bitpacked.npy")[:, :600]         # (6, 600, 66): 600 rules / class
R, L, RB = 6 * 600, 526, 528
rows_bits = np.unpackbits(packed.reshape(R, 66), axis=1, bitorder="little")        # (R, 528) literal-order bits
DATA = packed.reshape(-1)


def attacked_bytes(lits):
    b = rows_bits.copy(); b[:, lits] ^= 1
    return np.packbits(b, axis=1, bitorder="little").reshape(-1)


def wrong_lits(dec_bytes):
    d = np.unpackbits(dec_bytes[:DATA.size].reshape(R, 66), axis=1, bitorder="little")[:, :L]
    w = d != rows_bits[:, :L]
    return int(w.sum()), float(w.any(1).mean() * 100)


def run_hamming(att):
    """Flip the attacked data bits inside the Hamming(7,4)-encoded stream, then decode (attacker hits data bits)."""
    enc = E.hamming_encode_bytes(DATA)
    diff = np.unpackbits(att ^ DATA)                                   # stream bit i (MSB-first per byte)
    bits = np.unpackbits(enc)
    i = np.flatnonzero(diff)
    bits[(i // 4) * 7 + np.array([2, 4, 5, 6])[i % 4]] ^= 1
    return E.hamming_decode_bytes(np.packbits(bits))


codes = {"BCH(255,239)": galois.BCH(255, 239), "BCH(255,191)": galois.BCH(255, 191),
         "RS(255,223)": galois.ReedSolomon(255, 223)}
enc_clean = {}
msg_bits = np.unpackbits(DATA)
for name, c in codes.items():
    if name.startswith("BCH"):
        k = c.k; pad = (-msg_bits.size) % k
        enc_clean[name] = c.encode(galois.GF2(np.r_[msg_bits, np.zeros(pad, np.uint8)].reshape(-1, k)))
    else:
        k = c.k; pad = (-DATA.size) % k
        enc_clean[name] = c.encode(c.field(np.r_[DATA, np.zeros(pad, np.uint8)].reshape(-1, k)))


def run_code(name, att):
    c, C = codes[name], enc_clean[name]
    k = c.k
    if name.startswith("BCH"):
        e = np.unpackbits(att ^ DATA)
        E_ = np.zeros(C.size, np.uint8).reshape(C.shape)
        pad = (-e.size) % k
        E_[:, :k] = np.r_[e, np.zeros(pad, np.uint8)].reshape(-1, k)       # systematic: message bits first
        out = np.asarray(c.decode(C ^ galois.GF2(E_))).reshape(-1)[:msg_bits.size]
        return np.packbits(out.astype(np.uint8))
    e = att ^ DATA
    pad = (-e.size) % k
    E_ = np.zeros(C.shape, np.uint8)
    E_[:, :k] = np.r_[e, np.zeros(pad, np.uint8)].reshape(-1, k)
    return np.asarray(c.decode(C + c.field(E_))).astype(np.uint8).reshape(-1)[:DATA.size]


rows = []
for K in K_LIST:
    for d in range(DRAWS):
        lits = np.random.default_rng([2026, K, d]).choice(L, K, replace=False)
        att = attacked_bytes(lits)
        none_w, none_r = wrong_lits(att)
        rows.append(dict(K=K, draw=d, code="No ECC", wrong_literal_bits=none_w, rules_corrupted_pct=none_r))
        w, r = wrong_lits(run_hamming(att))
        rows.append(dict(K=K, draw=d, code="Hamming(7,4)", wrong_literal_bits=w, rules_corrupted_pct=r))
        for name in codes:
            w, r = wrong_lits(run_code(name, att))
            rows.append(dict(K=K, draw=d, code=name, wrong_literal_bits=w, rules_corrupted_pct=r))
        print(f"K={K} draw {d}: " + "  ".join(f"{x['code']} {x['rules_corrupted_pct']:.1f}%" for x in rows[-5:]),
              flush=True)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/results_attack.csv", index=False, float_format="%.6f")
print(df.groupby(["K", "code"], sort=False)["rules_corrupted_pct"].mean().unstack().round(2).to_string())
