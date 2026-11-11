"""Fair re-benchmark of bit packing, compression and error-correcting codes on the IoT (Edge-IIoTset) TM rules
(iot_dataset/rules_bitpacked.npy: 6 classes x 6000 clauses x 526 literals, bit-packed to 66 bytes per clause).

Same machine, same data, single thread (NUMBA_NUM_THREADS=1, like one Raspberry Pi core), every code with its
fastest batch implementation, warm-up (JIT compile) excluded, median of REPEATS runs.
  A. representation : raw (1 byte per literal) vs np.packbits; general-purpose compressors on raw and on packed
  B. ECC            : Hamming(7,4) (byte -> 2 codewords, rpi_customarchi.c layout, vectorised numpy),
                      BCH(255,239) t=2, BCH(255,191) t=8 (bits), RS(255,223) t=16 (bytes) via galois (batch)
                      encode, decode of a clean stream, decode after random bit errors (BER 1e-5 and 1e-4)
usage: NUMBA_NUM_THREADS=1 python benchmark.py   -> results_*.csv, benchmark.mat (this folder)
"""
import bz2
import lzma
import os
import sys
import time
import zlib
import numpy as np
import pandas as pd
import scipy.io
import brotli
import lz4.frame
import snappy
import zstandard
import galois

sys.path.insert(0, "/home/<user>/figures")
import ecc_common as E                                             # Hamming(7,4) of the pipeline

OUT = os.path.dirname(os.path.abspath(__file__))
REPEATS = 5
packed = np.load("/home/<user>/iot_dataset/rules_bitpacked.npy")    # (6, 6000, 66) uint8
raw = np.unpackbits(packed, axis=-1, bitorder="little")[..., :526].astype(np.uint8)  # (6, 6000, 526): 1 byte / literal
assert np.array_equal(np.packbits(raw, axis=-1, bitorder="little"), packed)
DATA = packed.reshape(-1)                                          # 2,376,000 bytes
NBITS = DATA.size * 8


def timed(fn, repeats=REPEATS):
    fn()                                                           # warm-up (JIT, caches)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts)) * 1000, float(np.min(ts)) * 1000


# ------------------------------------------------------------------ A. bit packing + compression
rows = []
_, t_pack, _ = timed(lambda: np.packbits(raw, axis=-1, bitorder="little"))
_, t_unpack, _ = timed(lambda: np.unpackbits(packed, axis=-1, bitorder="little")[..., :526])
rows.append(dict(method="raw (1 byte / literal)", input="raw", size_MB=raw.nbytes / 1e6, encode_ms=0.0, decode_ms=0.0))
rows.append(dict(method="bit-packing (np.packbits)", input="raw", size_MB=packed.nbytes / 1e6, encode_ms=t_pack,
                 decode_ms=t_unpack))
COMP = {"lz4-0": (lambda b: lz4.frame.compress(b, compression_level=0), lz4.frame.decompress),
        "snappy": (snappy.compress, snappy.decompress),
        "zstd-1": (zstandard.ZstdCompressor(level=1).compress, zstandard.ZstdDecompressor().decompress),
        "zstd-3": (zstandard.ZstdCompressor(level=3).compress, zstandard.ZstdDecompressor().decompress),
        "brotli-1": (lambda b: brotli.compress(b, quality=1), brotli.decompress),
        "zlib-6": (lambda b: zlib.compress(b, 6), zlib.decompress),
        "bzip2-9": (lambda b: bz2.compress(b, 9), bz2.decompress),
        "lzma-0": (lambda b: lzma.compress(b, preset=0), lzma.decompress)}
for src_name, src in (("raw", raw.tobytes()), ("bit-packed", packed.tobytes())):
    for name, (c, d) in COMP.items():
        comp, tc, _ = timed(lambda: c(src))
        back, td, _ = timed(lambda: d(comp))
        assert back == src
        rows.append(dict(method=f"{name} on {src_name}", input=src_name, size_MB=len(comp) / 1e6,
                         encode_ms=tc + (t_pack if src_name == "bit-packed" else 0),
                         decode_ms=td + (t_unpack if src_name == "bit-packed" else 0)))
A = pd.DataFrame(rows)
A["reduction_vs_raw_%"] = (1 - A.size_MB / A.size_MB.iloc[0]) * 100
A["total_ms"] = A.encode_ms + A.decode_ms
A.to_csv(f"{OUT}/results_bitpacking_compression.csv", index=False, float_format="%.6f")
print(A.round(3).to_string(index=False), flush=True)


# ------------------------------------------------------------------ B. ECC
def flip_bits(bitarr, ber, seed):
    """Random independent bit flips at bit error rate `ber` on a flat 0/1 array; returns (copy, positions)."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(rng.random(bitarr.size) < ber)
    out = bitarr.copy(); out[pos] ^= 1
    return out, pos


def bench_hamming():
    enc, t_enc, _ = timed(lambda: E.hamming_encode_bytes(DATA))
    dec, t_dec, _ = timed(lambda: E.hamming_decode_bytes(enc))
    assert np.array_equal(dec, DATA)
    res = dict(code="Hamming(7,4)", n=7, k=4, t="1 bit / 7", encoded_bytes=enc.size, encode_ms=t_enc,
               decode_clean_ms=t_dec)
    for ber in (1e-5, 1e-4):
        bits = np.unpackbits(enc)
        bad, pos = flip_bits(bits, ber, 7)
        badb = np.packbits(bad)
        out, t, _ = timed(lambda: E.hamming_decode_bytes(badb))
        res[f"decode_ber{ber:g}_ms"] = t
        res[f"errors_injected_ber{ber:g}"] = len(pos)
        res[f"residual_bit_errors_ber{ber:g}"] = int(np.unpackbits(out ^ DATA).sum())
    return res


def bench_bch(n, k):
    bch = galois.BCH(n, k)
    msg_bits = np.unpackbits(DATA)
    pad = (-msg_bits.size) % k
    M = galois.GF2(np.r_[msg_bits, np.zeros(pad, np.uint8)].reshape(-1, k))
    C, t_enc, _ = timed(lambda: bch.encode(M), repeats=3)
    D, t_dec, _ = timed(lambda: bch.decode(C), repeats=3)
    assert np.array_equal(np.asarray(D).reshape(-1)[:msg_bits.size], msg_bits)
    res = dict(code=f"BCH({n},{k})", n=n, k=k, t=f"{bch.t} bits / {n}", encoded_bytes=int(np.ceil(C.size / 8)),
               encode_ms=t_enc, decode_clean_ms=t_dec)
    for ber in (1e-5, 1e-4):
        bad, pos = flip_bits(np.asarray(C).reshape(-1).astype(np.uint8), ber, 7)
        Cb = galois.GF2(bad.reshape(C.shape))
        out, t, _ = timed(lambda: bch.decode(Cb), repeats=3)
        res[f"decode_ber{ber:g}_ms"] = t
        res[f"errors_injected_ber{ber:g}"] = len(pos)
        res[f"residual_bit_errors_ber{ber:g}"] = int((np.asarray(out).reshape(-1)[:msg_bits.size] != msg_bits).sum())
    return res


def bench_rs(n, k):
    rs = galois.ReedSolomon(n, k)
    GF = rs.field
    pad = (-DATA.size) % k
    M = GF(np.r_[DATA, np.zeros(pad, np.uint8)].reshape(-1, k))
    C, t_enc, _ = timed(lambda: rs.encode(M), repeats=3)
    D, t_dec, _ = timed(lambda: rs.decode(C), repeats=3)
    assert np.array_equal(np.asarray(D).reshape(-1)[:DATA.size], DATA)
    res = dict(code=f"RS({n},{k})", n=n, k=k, t=f"{rs.t} bytes / {n}", encoded_bytes=C.size, encode_ms=t_enc,
               decode_clean_ms=t_dec)
    for ber in (1e-5, 1e-4):
        bits = np.unpackbits(np.asarray(C).astype(np.uint8).reshape(-1))
        bad, pos = flip_bits(bits, ber, 7)
        Cb = GF(np.packbits(bad).reshape(C.shape))
        out, t, _ = timed(lambda: rs.decode(Cb), repeats=3)
        res[f"decode_ber{ber:g}_ms"] = t
        res[f"errors_injected_ber{ber:g}"] = len(pos)
        res[f"residual_bit_errors_ber{ber:g}"] = int(np.unpackbits(np.asarray(out).astype(np.uint8).reshape(-1)
                                                                   [:DATA.size] ^ DATA).sum())
    return res


B = []
for f in (bench_hamming, lambda: bench_bch(255, 239), lambda: bench_bch(255, 191), lambda: bench_rs(255, 223)):
    r = f(); B.append(r)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
B = pd.DataFrame(B)
B["code_rate"] = B.k / B.n
B["overhead_%"] = (B.encoded_bytes / DATA.size - 1) * 100
B["encode_Mbit_s"] = NBITS / (B.encode_ms / 1000) / 1e6
B["decode_Mbit_s"] = NBITS / (B.decode_clean_ms / 1000) / 1e6
B["data_bits"] = NBITS
B.to_csv(f"{OUT}/results_ecc.csv", index=False, float_format="%.6f")
print(B.round(3).to_string(index=False))

scipy.io.savemat(f"{OUT}/benchmark.mat", {
    "packing_method": np.array(A.method.tolist(), dtype=object), "packing_size_MB": A.size_MB.to_numpy(),
    "packing_encode_ms": A.encode_ms.to_numpy(), "packing_decode_ms": A.decode_ms.to_numpy(),
    "ecc_code": np.array(B.code.tolist(), dtype=object), "ecc_encode_ms": B.encode_ms.to_numpy(),
    "ecc_decode_clean_ms": B.decode_clean_ms.to_numpy(), "ecc_decode_ber1e-4_ms".replace("-", "m"):
    B["decode_ber0.0001_ms"].to_numpy(), "ecc_overhead_pct": B["overhead_%"].to_numpy(),
    "ecc_code_rate": B.code_rate.to_numpy(), "ecc_residual_ber1em4": B["residual_bit_errors_ber0.0001"].to_numpy(),
    "data_bits": float(NBITS), "repeats": float(REPEATS)}, do_compression=True)
