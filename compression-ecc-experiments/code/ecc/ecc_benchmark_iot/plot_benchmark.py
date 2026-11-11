"""Figures + tables for the IoT bit-packing / compression / ECC benchmark (benchmark.py, attack_ecc.py), plus the
timings recorded earlier in iot_dataset/error_coding.ipynb and compression.ipynb (as reported there).

usage: python plot_benchmark.py   -> fig1..fig5 *.png, table_*.csv, benchmark_report.pdf (this folder)
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = os.path.dirname(os.path.abspath(__file__))
A = pd.read_csv(f"{OUT}/results_bitpacking_compression.csv")
B = pd.read_csv(f"{OUT}/results_ecc.csv")
ATT = pd.read_csv(f"{OUT}/results_attack.csv") if os.path.exists(f"{OUT}/results_attack.csv") else None
mpl.rcParams.update({"font.size": 11, "axes.grid": True, "grid.color": "#e6e6e6", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"Hamming(7,4)": "#1baf7a", "BCH(255,239)": "#2a78d6", "BCH(255,191)": "#7b5ea7", "RS(255,223)": "#eb6834",
       "No ECC": "#888888"}
figs = []

# ---- notebook-reported numbers (error_coding.ipynb, as printed there)
NB = pd.DataFrame([
    dict(code="Hamming(7,4)", encode_ms=np.nan, decode_ms=57.727, implementation="vectorised NumPy",
         note="decode of 4,734,000 codewords; encode time not printed"),
    dict(code="BCH(255,239)", encode_ms=21.499, decode_ms=124381.177, implementation="galois, per-codeword loop",
         note="2 bit errors in codeword 0"),
    dict(code="BCH(255,191)", encode_ms=48.382, decode_ms=136443.574, implementation="galois, per-codeword loop",
         note="8 bit errors in codeword 0 (cell 38 plot used 140251 ms)"),
    dict(code="RS(255,223)", encode_ms=110.628, decode_ms=17567.561, implementation="galois, per-codeword loop",
         note="16 byte errors in codeword 0; first run encode 1269 ms"),
    dict(code="Conv. K=7 r=1/2 (Viterbi)", encode_ms=4658.824, decode_ms=np.nan, implementation="Python",
         note="decoding interrupted (too slow)")])
NB.to_csv(f"{OUT}/table_notebook_reported_ecc.csv", index=False)

# ---- fig 1: ECC encode / decode time (fair benchmark)
codes = B.code.tolist()
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(codes)); w = 0.26
for i, (col, lab, hatch) in enumerate([("encode_ms", "encode", ""), ("decode_clean_ms", "decode (no errors)", "//"),
                                       ("decode_ber0.0001_ms", "decode (BER 1e-4)", "..")]):
    v = B[col].to_numpy()
    bars = ax.bar(x + (i - 1) * w, v, w * 0.95, color=[COL[c] for c in codes], hatch=hatch, edgecolor="white",
                  alpha=[1, 0.8, 0.6][i], label=lab)
    for xb, vb in zip(x + (i - 1) * w, v):
        ax.text(xb, vb * 1.08, f"{vb:.1f}", ha="center", va="bottom", fontsize=8.5, rotation=90)
ax.set_yscale("log"); ax.set_ylim(5, 2e4)
ax.set_xticks(x, [f"{c}\nrate {r:.3f}, overhead {o:.1f} %" for c, r, o in zip(codes, B.code_rate, B["overhead_%"])])
ax.set_ylabel("time for the whole rule file (ms, log scale)")
ax.set_title("ECC on the IoT TM rules (2.376 MB bit-packed, 19.0 Mbit) — same machine, single thread, "
             "median of runs", loc="left")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#bbbbbb", label="encode"), Patch(facecolor="#bbbbbb", hatch="//", label="decode, no errors"),
                   Patch(facecolor="#bbbbbb", hatch="..", label="decode, BER 1e-4")], loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_ecc_time.png", dpi=170); figs.append(fig)

# ---- fig 2: throughput vs overhead (bubble = correction capability)
fig, ax = plt.subplots(figsize=(10, 6))
cap = {"Hamming(7,4)": 1 / 7, "BCH(255,239)": 2 / 255, "BCH(255,191)": 8 / 255, "RS(255,223)": 16 * 8 / (255 * 8)}
for _, r in B.iterrows():
    ax.scatter(r["overhead_%"], r.decode_Mbit_s, s=200 + 6000 * cap[r.code], color=COL[r.code], alpha=0.85,
               edgecolor="k", zorder=3)
    ax.annotate(f"{r.code}\ndecode {r.decode_Mbit_s:.1f} Mbit/s\nencode {r.encode_Mbit_s:.0f} Mbit/s",
                (r["overhead_%"], r.decode_Mbit_s), xytext=(14, 4), textcoords="offset points", fontsize=9)
ax.set_yscale("log"); ax.set_xlim(0, 95)
ax.set_xlabel("storage / transmission overhead (%)"); ax.set_ylabel("decode throughput (Mbit/s, log scale)")
ax.set_title("Speed vs overhead (bubble size = correctable errors per transmitted bit)", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_ecc_speed_vs_overhead.png", dpi=170); figs.append(fig)

# ---- fig 3: bit packing — size
sel = A[A.method.str.contains("raw \\(|bit-packing|lz4-0|zstd-1|zstd-3|snappy|brotli-1")].copy()
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
ax = axes[0]
cols = ["#888888" if "raw (" in m else "#1baf7a" if "bit-packing" in m else ("#9ecae1" if "on raw" in m else "#2a78d6")
        for m in sel.method]
ax.barh(sel.method, sel.size_MB, color=cols)
for y_, v, red in zip(range(len(sel)), sel.size_MB, sel["reduction_vs_raw_%"]):
    ax.text(v + 0.2, y_, f"{v:.3f} MB  (−{red:.1f} %)", va="center", fontsize=9)
ax.invert_yaxis(); ax.set_xlabel("size (MB)"); ax.set_xlim(0, 24)
ax.set_title("Size of the rule file (6 × 6000 rules × 526 literals)", loc="left")
ax = axes[1]
ax.barh(sel.method, sel.encode_ms, color=cols, label="encode / compress")
ax.barh(sel.method, sel.decode_ms, left=sel.encode_ms, color=cols, alpha=0.45, label="decode / decompress")
for y_, e, d in zip(range(len(sel)), sel.encode_ms, sel.decode_ms):
    ax.text(e + d + 0.5, y_, f"{e:.2f} + {d:.2f} = {e + d:.2f} ms", va="center", fontsize=9)
ax.invert_yaxis(); ax.set_xlabel("time (ms): encode + decode"); ax.set_xlim(0, 95)
ax.set_title("Time (compressors on packed data include the pack/unpack time)", loc="left")
ax.legend(loc="lower right")
fig.suptitle("Bit-packing: 8× smaller than raw in 2.4 ms; compressing the PACKED file is 3–5× faster than compressing raw",
             x=0.01, ha="left", fontsize=12.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_bitpacking.png", dpi=170); figs.append(fig)

# ---- fig 4: notebook-reported decode times (as recorded) vs fair benchmark
fig, ax = plt.subplots(figsize=(11, 6))
nb = NB[NB.code.isin(codes)].set_index("code").reindex(codes)
x = np.arange(len(codes))
ax.bar(x - 0.2, nb.decode_ms, 0.38, color=[COL[c] for c in codes], alpha=0.45, label="notebook (galois per-codeword loop)")
ax.bar(x + 0.2, B.decode_clean_ms, 0.38, color=[COL[c] for c in codes], label="fair re-benchmark (batch decode)")
for xi, a, b in zip(x, nb.decode_ms, B.decode_clean_ms):
    ax.text(xi - 0.2, a * 1.1, f"{a:,.0f}", ha="center", fontsize=8.5, rotation=90, va="bottom")
    ax.text(xi + 0.2, b * 1.1, f"{b:,.0f}", ha="center", fontsize=8.5, rotation=90, va="bottom")
ax.set_yscale("log"); ax.set_ylim(10, 5e6); ax.set_xticks(x, codes)
ax.set_ylabel("decode time, whole file (ms, log)")
ax.set_title("Decode time: earlier notebook numbers vs fair batch benchmark (Hamming wins in both)", loc="left")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig4_notebook_vs_fair.png", dpi=170); figs.append(fig)

# ---- fig 5: ECC under the MITM attack (same K literals in every rule)
if ATT is not None:
    g = ATT.groupby(["K", "code"])["rules_corrupted_pct"].agg(["mean", "min", "max"]).reset_index()
    fig, ax = plt.subplots(figsize=(11, 6))
    for code in ["No ECC", "Hamming(7,4)", "BCH(255,239)", "BCH(255,191)", "RS(255,223)"]:
        d = g[g.code == code]
        ax.plot(d.K, d["mean"], marker="o", lw=2, color=COL[code], label=code)
        ax.fill_between(d.K, d["min"], d["max"], color=COL[code], alpha=0.12, lw=0)
    ax.set_xlabel("K = literals flipped at the same position in every rule (MITM)")
    ax.set_ylabel("rules still corrupted after decoding (%)")
    ax.set_xticks(sorted(ATT.K.unique())); ax.set_ylim(-3, 103)
    ax.set_title(f"ECC under the structured MITM attack (mean of {ATT.draw.nunique()} random literal sets, band = min–max)",
                 loc="left")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig5_ecc_under_attack.png", dpi=170); figs.append(fig)
    g.pivot(index="K", columns="code", values="mean").to_csv(f"{OUT}/table_ecc_under_attack.csv", float_format="%.4f")

# ---- tables
T = B[["code", "code_rate", "overhead_%", "t", "encode_ms", "decode_clean_ms", "decode_ber1e-05_ms",
       "decode_ber0.0001_ms", "encode_Mbit_s", "decode_Mbit_s", "errors_injected_ber0.0001",
       "residual_bit_errors_ber0.0001"]].copy()
T["decode speed-up of Hamming (x)"] = T.decode_clean_ms / T.decode_clean_ms.iloc[0]
T.to_csv(f"{OUT}/table_ecc_fair.csv", index=False, float_format="%.4f")
A.to_csv(f"{OUT}/table_bitpacking_compression.csv", index=False, float_format="%.4f")

with PdfPages(f"{OUT}/benchmark_report.pdf") as pdf:
    for f in figs:
        pdf.savefig(f)
    for title, tab in (("ECC — fair benchmark", T), ("Bit-packing and compression", A),
                       ("ECC — numbers recorded in error_coding.ipynb", NB)):
        f, ax = plt.subplots(figsize=(16.5, 0.6 + 0.42 * len(tab))); ax.axis("off")
        fmt = lambda v: f"{v:,.3f}" if isinstance(v, float) and not np.isnan(v) else ("—" if isinstance(v, float) else str(v))
        tb = ax.table(cellText=[[fmt(v) for v in r] for r in tab.values], colLabels=[str(c) for c in tab.columns],
                      loc="center", cellLoc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.5)
        ax.set_title(title, loc="left"); pdf.savefig(f, bbox_inches="tight"); plt.close(f)
plt.close("all")
print(T.round(3).to_string(index=False))
