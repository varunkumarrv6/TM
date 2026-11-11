"""Paper figures (vector PDF) from the Raspberry Pi benchmark output.
  fig_rpi_latency.pdf : (a) inference latency per kernel (ms/sample, log), threads under each bar, fastest marked;
                        (b) end-to-end time for one model load + one inference = receive (Hamming decode +
                            rebuild) + prep + 1 sample
  fig_rpi_threads.pdf : single-core kernels 08 / 10 vs OpenMP kernel 09 at 1-4 threads (median of runs)
usage: python plot_results.py [results.csv] [threads_results.csv]     (run where matplotlib is available)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/results.csv"
THR = sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/threads_results.csv"
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "axes.titlelocation": "left",
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
BLUE, VERM, GREEN, GREY, ORANGE = "#0072B2", "#D55E00", "#009E73", "#7F7F7F", "#E69F00"

R = pd.read_csv(RES)
R["label"] = R.kernel.str.replace(r"^(\d\d) ", r"\1\n", regex=True)
best = R.infer_ms_per_sample.idxmin()
cols = [GREEN if i == best else (ORANGE if t > 1 else BLUE) for i, t in zip(R.index, R.threads)]
host = "Raspberry Pi 4" if "aarch64" in open(RES).read() or len(sys.argv) > 3 else "host"

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axes[0]
x = np.arange(len(R))
ax.bar(x, R.infer_ms_per_sample * 1000, 0.7, color=cols)
for xi, v, t in zip(x, R.infer_ms_per_sample * 1000, R.threads):
    ax.text(xi, v * 1.1, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
    ax.text(xi, 0.02, f"{t}T", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5,
            color="white" if v > 0 else "k", fontweight="bold")
ax.set_yscale("log"); ax.set_ylim(top=R.infer_ms_per_sample.max() * 1000 * 12)
ax.set_xticks(x, [k.split("\n")[0] for k in R.label])
ax.set_xlabel("kernel (08: clause-parallel NEON, 09: + OpenMP cores, 10: final NEON)")
ax.set_ylabel(r"inference latency ($\mu$s / sample)")
ax.set_title(f"(a) inference latency, {int(R.tmu_match.iloc[0])} test flows, all scores = TMU")
ax.grid(axis="x", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="single core (1T)"), Patch(color=ORANGE, label="OpenMP (multi-core)"),
                   Patch(color=GREEN, label=f"fastest: {R.kernel[best]}")], loc="upper right")
ax = axes[1]
y = np.arange(len(R))
one = R.receive_ms + R.prep_ms + R.infer_ms_per_sample
ax.barh(y, R.receive_ms, 0.7, color=GREY, label="receive (Hamming decode + rebuild)")
ax.barh(y, R.prep_ms, 0.7, left=R.receive_ms, color=BLUE, label="kernel preparation")
ax.barh(y, R.infer_ms_per_sample, 0.7, left=R.receive_ms + R.prep_ms, color=VERM, label="1 inference")
ax.set_yticks(y, R.kernel); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("time (ms)"); ax.set_title("(b) model load + one inference")
ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.2), ncol=2, fontsize=6.5)
fig.tight_layout(w_pad=1.0)
fig.savefig(f"{HERE}/fig_rpi_latency.pdf"); plt.close(fig)
print("saved fig_rpi_latency.pdf")

if os.path.exists(THR):
    T = pd.read_csv(THR)
    M = T.groupby(["kernel", "threads"])[["infer_ms_per_sample", "receive_ms"]].median().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))
    for ax, col, lab in zip(axes, ["infer_ms_per_sample", "receive_ms"],
                            [r"inference latency ($\mu$s / sample)", "receive: Hamming decode + rebuild (ms)"]):
        scale = 1000 if col == "infer_ms_per_sample" else 1
        d9 = M[M.kernel == "tm09_interclause_omp"].sort_values("threads")
        ax.plot(d9.threads, d9[col] * scale, "-o", color=ORANGE, mfc=ORANGE, mec="white",
                label="09 clause-parallel NEON + OpenMP")
        for k, colr, mk in (("tm08_interclause_neon", BLUE, "s"), ("tm10_final_neon", GREEN, "D")):
            v = M[M.kernel == k][col].iloc[0] * scale
            ax.axhline(v, color=colr, ls="--", lw=1.0, label=f"{k[2:4]} single core: {v:.2f}")
        ax.set(xticks=[1, 2, 3, 4], xlabel="OpenMP threads (cores)", ylabel=lab)
        ax.legend(fontsize=6.5)
    axes[0].set_title("(a) inference"); axes[1].set_title("(b) receive")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(f"{HERE}/fig_rpi_threads.pdf"); plt.close(fig)
    print("saved fig_rpi_threads.pdf")
