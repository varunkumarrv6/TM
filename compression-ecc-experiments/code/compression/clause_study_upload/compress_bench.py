"""Upload-compression benchmark of the trained local TMs (models/<ds>/k<K>_tm<k>.npz, K = 2..7).
Payload a local TM uploads for the joined clause bank = its inference model:
  include actions  : the top TA-state bit of every literal of every clause, per class (C x chunks uint32, bit-packed)
  clause weights   : C int32 per class
(the full 8-bit TA states are kept in the model files, if the full training state is wanted instead)
Methods (Python bindings): Zstandard, Brotli, Gzip, Zlib, Bzip2, LZMA2 (xz), LZ4 (frame), Zopfli (zlib format),
Snappy. Every compression is checked by decompressing it (lossless).
  level sweep    : one local TM (K = 2, TM 0), every method at every level 0..10 the library supports
  default levels : every local TM of every K, every method at its library default level
Times: median of up to 5 runs (single core, coder workspace CPU).
usage: ./run.sh clause_study/compress_bench.py DS   -> compression/<ds>/{level_sweep,default_levels}.csv
"""
import bz2
import gzip
import lzma
import os
import sys
import time
import zlib
import numpy as np
import pandas as pd
import brotli
import lz4.frame
import snappy
import zopfli.zlib
import zstandard
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, KS

L = range(0, 11)
METHODS = {   # name: (compress(data, level), decompress, levels in the sweep, default level)
    "Zstandard": (lambda b, l: zstandard.ZstdCompressor(level=l).compress(b),
                  lambda c: zstandard.ZstdDecompressor().decompress(c), list(L), 3),
    "Brotli": (lambda b, l: brotli.compress(b, quality=l), brotli.decompress, list(L), 11),
    "Gzip": (lambda b, l: gzip.compress(b, compresslevel=l, mtime=0), gzip.decompress, list(range(0, 10)), 9),
    "Zlib": (lambda b, l: zlib.compress(b, l), zlib.decompress, list(range(0, 10)), 6),
    "Bzip2": (lambda b, l: bz2.compress(b, l), bz2.decompress, list(range(1, 10)), 9),
    "LZMA2": (lambda b, l: lzma.compress(b, preset=l), lzma.decompress, list(range(0, 10)), 6),
    "LZ4": (lambda b, l: lz4.frame.compress(b, compression_level=l), lz4.frame.decompress, list(L), 0),
    "Zopfli": (lambda b, l: zopfli.zlib.compress(b, numiterations=max(l, 1)), zlib.decompress, list(range(1, 11)), 15),
    "Snappy": (lambda b, l: snappy.compress(b), snappy.decompress, [0], 0),
}


def payload(f):
    """Bytes a local TM uploads: include-action bitmaps + clause weights of every class."""
    M = np.load(f)
    ncls, C, ch, sb = M["ta_state"].shape[0], int(M["clauses"]), int(M["chunks"]), int(M["state_bits"])
    inc = np.ascontiguousarray(M["ta_state"].reshape(ncls, C, ch, sb)[..., sb - 1])
    w = np.ascontiguousarray(M["weights"].astype(np.int32))
    return inc.tobytes() + w.tobytes(), int(np.unpackbits(inc.view(np.uint8)).sum())


def timed(fn, *a):
    out, t = None, []
    for _ in range(5):
        t0 = time.perf_counter(); out = fn(*a); t.append(time.perf_counter() - t0)
        if sum(t) > 1.0:
            break
    return out, float(np.median(t))


def bench(data, name, level):
    comp, dec, _, _ = METHODS[name]
    c, tc = timed(comp, data, level)
    d, td = timed(dec, c)
    assert d == data, f"{name} level {level} is not lossless"
    return dict(method=name, level=level, raw_bytes=len(data), compressed_bytes=len(c),
                compress_s=tc, decompress_s=td)


if __name__ == "__main__":
    DS = sys.argv[1]
    out = f"{ROOT}/compression/{DS}"
    os.makedirs(out, exist_ok=True)
    data, n_inc = payload(f"{ROOT}/models/{DS}/k2_tm0.npz")
    rows = [dict(bench(data, m, l), include_literals=n_inc) for m, (_, _, levels, _) in METHODS.items() for l in levels]
    pd.DataFrame(rows).to_csv(f"{out}/level_sweep.csv", index=False)
    print(f"{DS}: level sweep done ({len(data) / 1e6:.3f} MB payload)", flush=True)
    rows = []
    for K in KS:
        for k in range(K):
            data, n_inc = payload(f"{ROOT}/models/{DS}/k{K}_tm{k}.npz")
            for m, (_, _, _, dflt) in METHODS.items():
                rows.append(dict(K=K, tm=k, include_literals=n_inc, **bench(data, m, dflt)))
        pd.DataFrame(rows).to_csv(f"{out}/default_levels.csv", index=False)
        print(f"{DS}: K={K} done", flush=True)
