"""Figures + tables of the local-TM upload compression study (data: compression/<ds>/*.csv from compress_bench.py).
Upload model for K local TMs over a shared link of B Mbit/s (default 1 Mbps):
  raw upload time          = sum_k raw_bytes_k * 8 / B
  compressed pipeline time = max_k compress_k   (the TMs compress in parallel on their own devices)
                           + sum_k compressed_bytes_k * 8 / B   (shared link to the server)
                           + sum_k decompress_k (the server decompresses every upload)
Per dataset (figures/<ds>/):
  fig7_compression_level    (a) level vs upload size, (b) level vs compression time   [one local TM, K = 2]
  fig8_latency_by_method    compression / upload / decompression time per method, K = 2, 1 Mbps (stacked)
  fig9_local_tms_upload     K vs total upload size (left) and raw upload / compressed pipeline time (right), Zstandard
Tables (compression/ and csv/<ds>/):
  table3_compression  every dataset x K at 1 Mbps: best method, raw / compressed MB, size saving, raw upload time,
                      compressed pipeline time, time saving        (+ .tex)
  table4_bandwidth    every dataset, the K with the highest time saving at 1 Mbps, B = 1, 5, 10, 50, 100 Mbps (+ .tex)
usage: ./run.sh clause_study/compress_plots.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATASETS, NAME

mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 10, "axes.labelsize": 10.5, "axes.grid": True, "grid.alpha": 0.3,
                     "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
MCOL = {"Zstandard": "#0072B2", "Brotli": "#D55E00", "Gzip": "#E69F00", "Zlib": "#009E73", "Bzip2": "#56B4E9",
        "LZMA2": "#7B2D8E", "LZ4": "#A50F15", "Zopfli": "#1F4E99", "Snappy": "#666666"}
BWS = (1, 5, 10, 50, 100)
FIG_K = 2


def pipeline(G, bw):
    """G: default-level rows of one (K, method). Times in ms."""
    up = G.compressed_bytes.sum() * 8 / (bw * 1e6) * 1e3
    return dict(raw_MB=G.raw_bytes.sum() / 1e6, comp_MB=G.compressed_bytes.sum() / 1e6,
                raw_upload_ms=G.raw_bytes.sum() * 8 / (bw * 1e6) * 1e3,
                compress_ms=G.compress_s.max() * 1e3, upload_ms=up, decompress_ms=G.decompress_s.sum() * 1e3,
                pipeline_ms=G.compress_s.max() * 1e3 + up + G.decompress_s.sum() * 1e3)


def table(D, ds, bw):
    rows = []
    for K, GK in D.groupby("K"):
        P = {m: pipeline(G, bw) for m, G in GK.groupby("method")}
        best = min(P, key=lambda m: P[m]["pipeline_ms"])
        p = P[best]
        rows.append(dict(dataset=NAME[ds], K=K, bandwidth_Mbps=bw, best_method=best, raw_upload_MB=p["raw_MB"],
                         compressed_upload_MB=p["comp_MB"], size_saving_pct=100 * (1 - p["comp_MB"] / p["raw_MB"]),
                         raw_upload_time_ms=p["raw_upload_ms"], compressed_pipeline_ms=p["pipeline_ms"],
                         time_saving_pct=100 * (1 - p["pipeline_ms"] / p["raw_upload_ms"])))
    return pd.DataFrame(rows)


def fig_levels(L, ds):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for m, G in L.groupby("method", sort=False):
        if m == "Snappy":
            continue
        a.plot(G.level, G.compressed_bytes, "o-", color=MCOL[m], ms=4, lw=1.4, label=m)
        b.plot(G.level, G.compress_s * 1e3, "o-", color=MCOL[m], ms=4, lw=1.4, label=m)
    raw = L.raw_bytes.iloc[0]
    a.axhline(raw, color="black", ls="--", lw=1.0)
    a.annotate(f"raw upload = {raw:,} bytes", (0.05, raw), xytext=(2, 4), textcoords="offset points", fontsize=8)
    a.set_yscale("log"); b.set_yscale("log")
    a.set_ylabel("upload size (bytes, log scale)"); b.set_ylabel("compression time (ms, log scale)")
    for ax, t in ((a, "(a) compression level vs upload size"), (b, "(b) compression level vs compression time")):
        ax.set_xlabel("compression level (Zopfli: iterations)"); ax.set_xticks(range(0, 11))
        ax.set_title(t, fontsize=10); ax.legend(ncol=2, fontsize=8)
    fig.suptitle(f"{NAME[ds]}: one local TM (K = {FIG_K}, 2000 clauses), include actions + clause weights",
                 fontweight="bold", y=1.01)
    return fig


def fig_methods(D, ds, bw=1):
    G = D[D.K == FIG_K]
    P = pd.DataFrame([dict(method=m, **pipeline(g, bw)) for m, g in G.groupby("method")]).sort_values("pipeline_ms")
    raw = P.raw_upload_ms.iloc[0]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(P) + 1)
    c = list(P.compress_ms) + [0]; u = list(P.upload_ms) + [raw]; d = list(P.decompress_ms) + [0]
    ax.bar(x, c, color="#FDE3C4", edgecolor="#555", lw=0.6, label="compression (slowest local TM)")
    ax.bar(x, u, bottom=c, color="#D3E0F5", edgecolor="#555", lw=0.6, label=f"upload ({bw} Mbps, all K TMs)")
    ax.bar(x, d, bottom=np.add(c, u), color="#D2EBD8", edgecolor="#555", lw=0.6, label="decompression (server)")
    for i, t in enumerate(np.add(np.add(c, u), d)):
        ax.annotate(f"{t:,.0f}", (i, t), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_yscale("log"); ax.set_ylim(min(min(v for v in c if v > 0), min(d[:-1])) * 0.5, raw * 3)
    ax.set_xticks(x, list(P.method) + ["Raw"], rotation=35, ha="right")
    ax.set_ylabel("end-to-end latency (ms, log scale)"); ax.set_xlabel("compression method")
    ax.set_title(f"{NAME[ds]}: K = {FIG_K} local TMs upload their models, {bw} Mbps", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    return fig


def fig_k(D, ds, method="Zstandard", bw=1):
    P = pd.DataFrame([dict(K=K, **pipeline(G[G.method == method], bw)) for K, G in D.groupby("K")])
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    a2 = ax.twinx(); a2.grid(False)
    h = [ax.plot(P.K, P.raw_MB * 1e6, "o-", color="#0072B2", lw=1.5, mfc="white", label="raw upload size")[0],
         ax.plot(P.K, P.comp_MB * 1e6, "s-", color="#0072B2", lw=1.5, label=f"compressed upload size ({method})")[0],
         a2.plot(P.K, P.raw_upload_ms, "^-", color="#D55E00", lw=1.5, mfc="white", label="raw upload time")[0],
         a2.plot(P.K, P.pipeline_ms, "D-", color="#D55E00", lw=1.5, label="compressed pipeline time")[0]]
    ax.set_yscale("log"); a2.set_yscale("log")
    lo, hi = P.comp_MB.min() * 1e6, P.raw_MB.max() * 1e6               # sizes in the lower half, times in the upper
    ax.set_ylim(lo / 1.5, hi * 100)
    a2.set_ylim(P.pipeline_ms.min() / 100, P.raw_upload_ms.max() * 1.5)
    ax.set_ylabel("total upload size (bytes, log scale)", color="#0072B2")
    a2.set_ylabel("end-to-end communication latency (ms, log scale)", color="#D55E00")
    ax.set_xlabel("number of local TMs K (2000 clauses each)"); ax.set_xticks(P.K)
    ax.legend(handles=h, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.set_title(f"{NAME[ds]}: all K local TMs upload their models, {bw} Mbps, {method}", fontweight="bold")
    return fig


def tex(T, cols, caption, path):
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", "\\footnotesize",
             "\\begin{tabular}{" + "l" * 2 + "r" * (len(cols) - 2) + "}", "\\toprule",
             " & ".join(c for c, _ in cols) + " \\\\", "\\midrule"]
    prev = None
    for _, r in T.iterrows():
        cells = []
        for (_, k) in cols:
            v = r[k]
            cells.append((v if v != prev else "") if k == "dataset" else (f"{v:.2f}" if isinstance(v, float) else str(v)))
        prev = r["dataset"]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n")


T3, T4 = [], []
for ds in DATASETS:
    d = f"{ROOT}/compression/{ds}"
    if not os.path.exists(f"{d}/default_levels.csv"):
        continue
    L, D = pd.read_csv(f"{d}/level_sweep.csv"), pd.read_csv(f"{d}/default_levels.csv")
    os.makedirs(f"{ROOT}/figures/{ds}", exist_ok=True); os.makedirs(f"{ROOT}/csv/{ds}", exist_ok=True)
    for name, fig in (("fig7_compression_level", fig_levels(L, ds)), ("fig8_latency_by_method", fig_methods(D, ds)),
                      ("fig9_local_tms_upload", fig_k(D, ds))):
        for ext in ("pdf", "png"):
            fig.savefig(f"{ROOT}/figures/{ds}/{name}.{ext}", dpi=300)
        plt.close(fig)
    t3 = table(D, ds, 1)
    kbest = int(t3.loc[t3.time_saving_pct.idxmax(), "K"])
    t4 = pd.concat([table(D[D.K == kbest], ds, bw) for bw in BWS], ignore_index=True)
    t3.round(4).to_csv(f"{ROOT}/csv/{ds}/compression_table3.csv", index=False)
    t4.round(4).to_csv(f"{ROOT}/csv/{ds}/compression_table4_bandwidth.csv", index=False)
    L.to_csv(f"{ROOT}/csv/{ds}/compression_level_sweep.csv", index=False)
    D.to_csv(f"{ROOT}/csv/{ds}/compression_default_levels.csv", index=False)
    T3.append(t3); T4.append(t4)
if T3:
    T3, T4 = pd.concat(T3, ignore_index=True), pd.concat(T4, ignore_index=True)
    T3.round(4).to_csv(f"{ROOT}/compression/table3_compression.csv", index=False)
    T4.round(4).to_csv(f"{ROOT}/compression/table4_bandwidth.csv", index=False)
    c3 = [("Dataset", "dataset"), ("K", "K"), ("Best method", "best_method"), ("Raw (MB)", "raw_upload_MB"),
          ("Compressed (MB)", "compressed_upload_MB"), ("Size saving (\\%)", "size_saving_pct"),
          ("Raw upload (ms)", "raw_upload_time_ms"), ("Compressed pipeline (ms)", "compressed_pipeline_ms"),
          ("Time saving (\\%)", "time_saving_pct")]
    tex(T3, c3, "Local-TM model upload compression at 1 Mbps (K local TMs, 2000 clauses each). The compressed "
        "pipeline time includes compression, compressed upload and decompression.", f"{ROOT}/compression/table3.tex")
    c4 = [("Dataset", "dataset"), ("K", "K"), ("Bandwidth (Mbps)", "bandwidth_Mbps"), ("Best method", "best_method"),
          ("Raw upload (ms)", "raw_upload_time_ms"), ("Compressed pipeline (ms)", "compressed_pipeline_ms"),
          ("Time saving (\\%)", "time_saving_pct")]
    tex(T4, c4, "Bandwidth sensitivity; for each dataset the K with the highest time saving at 1 Mbps.",
        f"{ROOT}/compression/table4.tex")
    print(T3.round(2).to_string(index=False)); print(T4.round(2).to_string(index=False))
print("->", f"{ROOT}/figures/<ds>/fig7-9, {ROOT}/compression/table3/4 (.csv, .tex), {ROOT}/csv/<ds>/compression_*")
