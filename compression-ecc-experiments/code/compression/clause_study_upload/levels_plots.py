"""Compression level vs upload size / compression time, mean over 10 iterations (MNIST, Fashion-MNIST), for the
centralized weighted TM (1 TM) and for K = 2, 3, 4, 5 local TMs (data: levels/<ds>/it<IT>_k<K>.csv).
Per iteration and configuration: upload size = sum over the K local TMs' compressed models (what the server receives),
compression time = slowest local TM (they compress in parallel); then the mean over the iterations.
Figures (figures/<ds>/): fig10_levels_k<K>   (a) level vs upload size, (b) level vs compression time
                         fig10_levels_all    the 5 configurations in one figure (rows)
CSV: csv/<ds>/compression_levels_mean.csv
usage: ./run.sh clause_study/levels_plots.py
"""
import glob
import os
import sys
import pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, NAME
MCOL = {"Zstandard": "#0072B2", "Brotli": "#D55E00", "Gzip": "#E69F00", "Zlib": "#009E73", "Bzip2": "#56B4E9",
        "LZMA2": "#7B2D8E", "LZ4": "#A50F15", "Zopfli": "#1F4E99", "Snappy": "#666666"}   # as compress_plots.py

mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 10, "axes.labelsize": 10.5, "axes.grid": True, "grid.alpha": 0.3,
                     "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
KS = (1, 2, 3, 4, 5)


def config_name(K):
    return "centralized weighted TM (1 TM)" if K == 1 else f"K = {K} local TMs"


def draw(a, b, M, K, legend=True):
    for m, G in M[M.K == K].groupby("method", sort=False):
        if m == "Snappy":
            continue
        G = G.sort_values("level")
        a.plot(G.level, G.compressed_bytes, "o-", color=MCOL[m], ms=4, lw=1.4, label=m)
        b.plot(G.level, G.compress_s * 1e3, "o-", color=MCOL[m], ms=4, lw=1.4, label=m)
    raw = M[M.K == K].raw_bytes.iloc[0]
    a.axhline(raw, color="black", ls="--", lw=1.0)
    a.annotate(f"raw upload = {raw:,.0f} bytes", (0.05, raw), xytext=(2, 4), textcoords="offset points", fontsize=8)
    a.set_yscale("log"); b.set_yscale("log")
    a.set_ylim(top=raw * 1.6)
    a.set_ylabel("upload size (bytes, log scale)" if K == 1 else "total upload, all K TMs\n(bytes, log scale)")
    b.set_ylabel("compression time (ms, log scale)" if K == 1 else "compression time, slowest TM\n(ms, log scale)")
    for ax in (a, b):
        ax.set_xticks(range(0, 11)); ax.set_xlabel("compression level (Zopfli: iterations)")
        if legend:
            ax.legend(ncol=2, fontsize=7.5)


for ds in ("mnist", "fmnist"):
    fs = sorted(glob.glob(f"{ROOT}/levels/{ds}/it*_k*.csv"))
    if not fs:
        continue
    R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    # per iteration x configuration x method x level: total bytes over the K TMs, slowest compression
    P = R.groupby(["iteration", "K", "method", "level"]).agg(raw_bytes=("raw_bytes", "sum"),
                                                             compressed_bytes=("compressed_bytes", "sum"),
                                                             compress_s=("compress_s", "max"),
                                                             decompress_s=("decompress_s", "sum")).reset_index()
    M = P.groupby(["K", "method", "level"]).agg(iterations=("iteration", "nunique"), raw_bytes=("raw_bytes", "mean"),
                                                compressed_bytes=("compressed_bytes", "mean"),
                                                compress_s=("compress_s", "mean"),
                                                decompress_s=("decompress_s", "mean")).reset_index()
    order = {m: i for i, m in enumerate(R.method.drop_duplicates())}
    M = M.sort_values(["K", "method", "level"], key=lambda c: c.map(order) if c.name == "method" else c)
    os.makedirs(f"{ROOT}/csv/{ds}", exist_ok=True)
    M.round(6).to_csv(f"{ROOT}/csv/{ds}/compression_levels_mean.csv", index=False)
    for K in KS:
        if not (M.K == K).any():
            continue
        n = int(M[M.K == K].iterations.min())
        fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4))
        draw(a, b, M, K)
        a.set_title("(a) compression level vs upload size", fontsize=10)
        b.set_title("(b) compression level vs compression time", fontsize=10)
        fig.suptitle(f"{NAME[ds]}: {config_name(K)}, 2000 clauses each, mean of {n} iterations",
                     fontweight="bold", y=1.01)
        for ext in ("pdf", "png"):
            fig.savefig(f"{ROOT}/figures/{ds}/fig10_levels_k{K}.{ext}", dpi=300)
        plt.close(fig)
    ks = [K for K in KS if (M.K == K).any()]
    fig, axs = plt.subplots(len(ks), 2, figsize=(12.5, 3.6 * len(ks)), squeeze=False,
                            gridspec_kw=dict(hspace=0.45, top=1 - 1.1 / (3.6 * len(ks))))
    for r, K in enumerate(ks):
        draw(axs[r, 0], axs[r, 1], M, K, legend=False)
        short = "centralized TM" if K == 1 else f"K = {K} local TMs"
        for c, t in ((0, "(a) upload size"), (1, "(b) compression time")):
            axs[r, c].set_title(f"{short}: {t}", fontsize=10, fontweight="bold")
    n = int(M.iterations.min())
    h, l = axs[0, 0].get_legend_handles_labels()
    H = 3.6 * len(ks)                                           # figure height (in): legend + title above the axes
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1 - 0.75 / H), ncol=len(l), fontsize=9)
    fig.suptitle(f"{NAME[ds]}: compression level vs upload size and compression time, mean of {n} iterations",
                 fontweight="bold", y=1 - 0.35 / H, va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(f"{ROOT}/figures/{ds}/fig10_levels_all.{ext}", dpi=300)
    plt.close(fig)
    print(ds, "configurations", ks, "iterations", int(M.iterations.min()))
print("->", f"{ROOT}/figures/<ds>/fig10_levels_*", f"{ROOT}/csv/<ds>/compression_levels_mean.csv")
