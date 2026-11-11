"""Monte Carlo with INDEPENDENT bit flips: every transmitted bit flips with probability p (channel bit error rate),
150 iterations per p, unprotected TM vs Hamming(7,4) (SEQ layout; parity bits are hit too).

  No ECC      : each of the L literal bits of every rule flips with probability p
  Hamming(7,4): each of the 7 bits of every codeword flips with probability p; the decoder output for every 7-bit
                error pattern is taken from the real decoder (ecc_common.hamming_decode_bytes), so residual data
                errors (2 or 3 bits for a 2-error codeword) are exact
Per iteration: accuracy, codewords with >= 2 channel errors, residual wrong literal bits, clauses left corrupted.
Predictions: P(>=2 errors in a codeword) = 1-(1-p)^7-7p(1-p)^6 ; corrupted clause share
             No ECC 1-(1-p)^L, Hamming 1-(1-P2)^(RB/4)   (RB/4 codewords per rule).

usage: SPLIT=70_15_15 python mc_ber.py {medsec|iot}   -> split_<SPLIT>/<ds>/monte_carlo_ber/mc_ber_<ds>.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import ecc_common as E

DS = sys.argv[1]
OUT = f"{E.ROOT}/{DS}/monte_carlo_ber"
os.makedirs(OUT, exist_ok=True)
P_LIST = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1, 2e-1]
ITERS = 150

m, link, eng, pred0 = E.setup(DS)
y, N = m["yte"], len(m["yte"])
BENCH = (pred0 == y).mean() * 100
idx = link.lay["SEQ"][0]                                   # (ncw, 4) flat positions of D1..D4 of every codeword
NCW = len(idx)
real_clauses = link.real.sum()

# decoder output (4-bit data error, D1 = bit 3) for each of the 128 error patterns on P1 P2 D1 P4 D2 D3 D4
DEC7 = np.zeros(128, np.uint8)
for e in range(128):
    bits = np.unpackbits(E.hamming_encode_bytes(np.array([0], np.uint8)), bitorder="big")
    for i in range(7):
        if (e >> (6 - i)) & 1:
            bits[i] ^= 1
    DEC7[e] = E.hamming_decode_bytes(np.packbits(bits, bitorder="big"))[0] >> 4


def received(coding, p, rng):
    dec = link.rows.ravel().copy()
    if coding == "No ECC":
        n = link.R * link.L
        pos = rng.choice(n, rng.binomial(n, p), replace=False)
        dec[(pos // link.L) * link.RB + pos % link.L] ^= 1
        return dec.reshape(link.rows.shape), np.nan
    n = NCW * 7
    pos = rng.choice(n, rng.binomial(n, p), replace=False)
    cws, inv = np.unique(pos // 7, return_inverse=True)
    ev = np.zeros(len(cws), np.uint8)
    np.bitwise_or.at(ev, inv, (64 >> (pos % 7)).astype(np.uint8))
    multi = int((np.unpackbits(ev[:, None], axis=1).sum(1) >= 2).sum())
    res = DEC7[ev]
    for i in range(4):
        sel = (res >> (3 - i)) & 1 == 1
        dec[idx[cws[sel], i]] ^= 1
    return dec.reshape(link.rows.shape), multi / NCW


rows, t0 = [], time.time()
for p in P_LIST:
    for coding in ("No ECC", "Hamming(7,4)"):
        for it in range(ITERS):
            rng = np.random.default_rng([int(round(p * 1e9)), it, 0 if coding == "No ECC" else 1])
            rx, frac_multi = received(coding, p, rng)
            cube = link.to_cube(rx)
            pred = eng.sums(cube).argmax(1)
            diff = cube != m["cube"]
            rows.append(dict(p=p, coding=coding, iteration=it + 1, accuracy=(pred == y).mean() * 100,
                             correct=int((pred == y).sum()), N=N, benchmark=BENCH,
                             frac_cw_2plus=frac_multi, wrong_literal_bits=int(diff.sum()),
                             frac_clauses_corrupted=diff.any(2).sum() / real_clauses))
    d = pd.DataFrame(rows)
    g = d[d.p == p].groupby("coding")["accuracy"].mean()
    print(f"[{DS}] p={p:g}: No ECC {g['No ECC']:.3f} %, Hamming {g['Hamming(7,4)']:.3f} %  ({time.time() - t0:.0f} s)",
          flush=True)
    d.to_csv(f"{OUT}/mc_ber_{DS}.csv", index=False, float_format="%.10f")
