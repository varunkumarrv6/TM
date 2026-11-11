"""Total time (encode + decode) of Hamming(7,4), BCH(255,239), BCH(255,191), RS(255,223) and a polar code (256,224)
on the IoT TM rules (rules_bitpacked.npy, 2.376 MB), for bit error rates 0, 1e-5, 1e-4, 1e-3.
Same machine, single thread, batch implementations, warm-up excluded, median of runs; residual errors are checked.

Polar(256,224): rate 0.875 (= RS(255,223)), non-systematic x = u F^{(x)8}; frozen set = the 32 least reliable
synthetic channels by the Bhattacharyya recursion (BEC design z0 = 0.5); decoder = successive cancellation (min-sum),
vectorised over all codewords, with rate-0 / rate-1 node shortcuts; hard-decision channel LLRs (BSC).
usage: NUMBA_NUM_THREADS=1 python total_time.py   -> results_total_time.csv, total_time.mat, fig6_total_time.png
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.io
import galois
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, "/home/<user>/figures")
import ecc_common as E

OUT = os.path.dirname(os.path.abspath(__file__))
BERS = [0.0, 1e-5, 1e-4, 1e-3]
DATA = np.load("/home/<user>/iot_dataset/rules_bitpacked.npy").reshape(-1)          # 2,376,000 bytes
MSG = np.unpackbits(DATA)                                                          # 19,008,000 bits


def timed(fn, repeats):
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts)) * 1000


def flip(bits, ber, seed=7):
    out = bits.copy()
    if ber > 0:
        out[np.random.default_rng(seed).random(bits.size) < ber] ^= 1
    return out


# ------------------------------------------------------------------ polar code
class Polar:
    def __init__(self, N=256, K=224):
        self.N, self.K = N, K
        z = np.array([0.5])
        while z.size < N:                                  # first half = worse (f) channels, second half = better (g)
            z = np.concatenate([2 * z - z ** 2, z ** 2])
        self.frozen = np.zeros(N, bool)
        self.frozen[np.argsort(-z, kind="stable")[:N - K]] = True
        self.info = np.flatnonzero(~self.frozen)

    @staticmethod
    def transform(u):                                      # x = u F^{(x)n} over GF(2), batch (B, N)
        x = u.copy(); B, N = x.shape; h = 1
        while h < N:
            v = x.reshape(B, N // (2 * h), 2, h)
            v[:, :, 0, :] ^= v[:, :, 1, :]
            h *= 2
        return x

    def encode(self, msg_bits):                            # (B, K) -> (B, N)
        u = np.zeros((msg_bits.shape[0], self.N), np.uint8)
        u[:, self.info] = msg_bits
        return self.transform(u)

    def _sc(self, llr, frozen):
        M = frozen.size
        if frozen.all():
            z = np.zeros(llr.shape, np.uint8); return z, z
        if not frozen.any():                               # rate-1 node: hard decision, u = x F (F involutory)
            x = (llr < 0).astype(np.uint8); return self.transform(x), x
        if M == 1:
            u = (llr < 0).astype(np.uint8); return u, u
        a, b = llr[:, :M // 2], llr[:, M // 2:]
        f = np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
        u1, x1 = self._sc(f, frozen[:M // 2])
        g = b + (1 - 2 * x1.astype(np.float32)) * a
        u2, x2 = self._sc(g, frozen[M // 2:])
        return np.concatenate([u1, u2], 1), np.concatenate([x1 ^ x2, x2], 1)

    def decode(self, y_bits):                              # hard bits (B, N) -> message bits (B, K)
        llr = (1 - 2 * y_bits.astype(np.float32))
        u, _ = self._sc(llr, self.frozen)
        return u[:, self.info]


# ------------------------------------------------------------------ benchmarks: (encode fn, decode fn, compare fn)
def run_hamming():
    enc, te = timed(lambda: E.hamming_encode_bytes(DATA), 5)
    rows = []
    for ber in BERS:
        y = np.packbits(flip(np.unpackbits(enc), ber))
        dec, td = timed(lambda: E.hamming_decode_bytes(y), 5)
        rows.append((ber, te, td, int(np.unpackbits(dec ^ DATA).sum())))
    return "Hamming(7,4)", 4 / 7, rows


def run_bch(n, k):
    c = galois.BCH(n, k); pad = (-MSG.size) % k
    M = galois.GF2(np.r_[MSG, np.zeros(pad, np.uint8)].reshape(-1, k))
    C, te = timed(lambda: c.encode(M), 3)
    rows = []
    for ber in BERS:
        Y = galois.GF2(flip(np.asarray(C).reshape(-1).astype(np.uint8), ber).reshape(C.shape))
        D, td = timed(lambda: c.decode(Y), 3)
        rows.append((ber, te, td, int((np.asarray(D).reshape(-1)[:MSG.size] != MSG).sum())))
    return f"BCH({n},{k})", k / n, rows


def run_rs(n, k):
    c = galois.ReedSolomon(n, k); pad = (-DATA.size) % k
    M = c.field(np.r_[DATA, np.zeros(pad, np.uint8)].reshape(-1, k))
    C, te = timed(lambda: c.encode(M), 3)
    rows = []
    for ber in BERS:
        bits = flip(np.unpackbits(np.asarray(C).astype(np.uint8).reshape(-1)), ber)
        Y = c.field(np.packbits(bits).reshape(C.shape))
        D, td = timed(lambda: c.decode(Y), 3)
        rows.append((ber, te, td, int(np.unpackbits(np.asarray(D).astype(np.uint8).reshape(-1)[:DATA.size] ^ DATA).sum())))
    return f"RS({n},{k})", k / n, rows


def run_polar(N=256, K=224):
    p = Polar(N, K); pad = (-MSG.size) % K
    Mb = np.r_[MSG, np.zeros(pad, np.uint8)].reshape(-1, K)
    X, te = timed(lambda: p.encode(Mb), 5)
    rows = []
    for ber in BERS:
        Y = flip(X.reshape(-1), ber).reshape(X.shape)
        D, td = timed(lambda: p.decode(Y), 3)
        rows.append((ber, te, td, int((D.reshape(-1)[:MSG.size] != MSG).sum())))
    return f"Polar({N},{K}) SC", K / N, rows


res = []
for f in (run_hamming, lambda: run_bch(255, 239), lambda: run_bch(255, 191), lambda: run_rs(255, 223), run_polar):
    name, rate, rows = f()
    for ber, te, td, resid in rows:
        res.append(dict(code=name, rate=rate, overhead_pct=(1 / rate - 1) * 100, BER=ber, encode_ms=te, decode_ms=td,
                        total_ms=te + td, residual_bit_errors=resid, data_bits=MSG.size))
    print(f"{name}: " + "  ".join(f"BER {r[0]:g}: {r[1] + r[2]:.1f} ms (resid {r[3]})" for r in rows), flush=True)
R = pd.DataFrame(res)
R.to_csv(f"{OUT}/results_total_time.csv", index=False, float_format="%.6f")
assert (R[R.BER == 0].residual_bit_errors == 0).all()

# ------------------------------------------------------------------ figure
COL = {"Hamming(7,4)": "#1baf7a", "BCH(255,239)": "#2a78d6", "BCH(255,191)": "#7b5ea7", "RS(255,223)": "#eb6834",
       "Polar(256,224) SC": "#d4a017"}
codes = list(dict.fromkeys(R.code))
fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw=dict(width_ratios=[1, 1.15]))
ax = axes[0]
c0 = R[R.BER == 0].set_index("code").reindex(codes)
x = np.arange(len(codes))
ax.bar(x, c0.encode_ms, color=[COL[c] for c in codes], label="encode")
ax.bar(x, c0.decode_ms, bottom=c0.encode_ms, color=[COL[c] for c in codes], alpha=0.5, hatch="//", edgecolor="white",
       label="decode")
for xi, e, d in zip(x, c0.encode_ms, c0.decode_ms):
    ax.text(xi, (e + d) * 1.08, f"{e + d:,.1f} ms\n({e:.1f} + {d:,.1f})", ha="center", va="bottom", fontsize=9)
ax.set_yscale("log"); ax.set_ylim(10, 3e4)
ax.set_xticks(x, [f"{c}\nrate {r:.3f}" for c, r in zip(codes, c0.rate)], fontsize=9.5)
ax.set_ylabel("total time = encode + decode (ms, log scale)")
ax.set_title("Total time, error-free stream (whole 19.0 Mbit rule file)", loc="left")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#999999", label="encode"),
                   Patch(facecolor="#999999", alpha=0.5, hatch="//", label="decode")], loc="upper left")
ax = axes[1]
for c in codes:
    d = R[R.code == c].sort_values("BER")
    xb = np.where(d.BER == 0, 3e-6, d.BER)
    ax.plot(xb, d.total_ms, marker="o", lw=2, ms=7, color=COL[c], label=c)
    for xv, yv, rv in zip(xb, d.total_ms, d.residual_bit_errors):
        if rv:
            ax.annotate(f"{rv:,} left", (xv, yv), xytext=(4, 6), textcoords="offset points", fontsize=8, color=COL[c])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([3e-6, 1e-5, 1e-4, 1e-3], ["0", "1e-5", "1e-4", "1e-3"])
ax.set_xlabel("bit error rate on the channel"); ax.set_ylabel("total time = encode + decode (ms, log scale)")
ax.set_title("Total time vs error rate  (label = bit errors left after decoding)", loc="left")
ax.legend(loc="center left", fontsize=9.5)
fig.suptitle("Hamming vs BCH vs RS vs Polar on the IoT TM rules — same machine, single thread, median of runs",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_total_time.png", dpi=170)

piv = lambda col: np.array([R[R.code == c].sort_values("BER")[col].to_numpy() for c in codes], float)
scipy.io.savemat(f"{OUT}/total_time.mat", {
    "codes": np.array(codes, dtype=object), "BER": np.array(BERS, float), "dims": "arrays are (code, BER)",
    "encode_ms": piv("encode_ms"), "decode_ms": piv("decode_ms"), "total_ms": piv("total_ms"),
    "residual_bit_errors": piv("residual_bit_errors"), "rate": c0.rate.to_numpy(float),
    "data_bits": float(MSG.size)}, do_compression=True)
print(R.pivot_table(index="code", columns="BER", values="total_ms", sort=False).round(1).to_string())
