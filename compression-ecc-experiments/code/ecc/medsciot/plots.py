"""Summary tables + figures for the MITM / Hamming(7,4) simulation."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/<user>/medsciot"
df = pd.read_csv(f"{BASE}/results.csv")
clean = df[df.attack == "clean"].iloc[0]
df = df[df.attack != "clean"].copy()
df["param"] = df["param"].astype(int)

SCHEME_LABEL = {"none": "No ECC", "SEQ": "Hamming – sequential", "INT": "Hamming – feature-interleaved",
                "KEY": "Hamming – keyed shuffle"}
COLORS = {"none": "#2a78d6", "SEQ": "#eb6834", "INT": "#1baf7a", "KEY": "#eda100"}
MARKERS = {"none": "o", "SEQ": "s", "INT": "^", "KEY": "D"}
TITLES = {"A1": "A1 – random flips (k literals / clause)", "A2": "A2 – force include, x and ¬x",
          "A2+": "A2⁺ – force include, x only", "A3": "A3 – force exclude",
          "A4": "A4 – toggle"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.8})


def figure(metric, ylabel, fname, m):
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.6), sharey=True)
    for ax, atk in zip(axes, TITLES):
        sub = df[(df.attack == atk) & (df.attacked_nodes == m)]
        for s in SCHEME_LABEL:
            r = sub[sub.scheme == s].sort_values("param")
            ax.plot(r.param.astype(int), r[metric] * 100, color=COLORS[s], lw=2, marker=MARKERS[s],
                    ms=7, mec="white", mew=1.5, label=SCHEME_LABEL[s])
        ax.axhline(clean[metric] * 100, color="#52514e", lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ticks = sorted(sub.param.astype(int).unique())
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_title(TITLES[atk], fontsize=10.5, color="#0b0b0b", loc="left")
        ax.set_xlabel("k random literals / clause" if atk == "A1" else "targeted raw features |F|")
    axes[0].set_ylabel(ylabel)
    axes[0].annotate(f"clean {clean[metric]*100:.1f}%", xy=(1, clean[metric] * 100), xytext=(0, 4),
                     textcoords="offset points", fontsize=9, color="#52514e")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle(f"MedSec-25 · {m} of 3 TM nodes attacked · fused prediction", x=0.01, ha="left",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{BASE}/{fname}", dpi=150)
    plt.close(fig)


for m in (1, 2, 3):
    figure("accuracy", "accuracy (%)", f"fig_accuracy_{m}node.png", m)
    figure("macro_f1", "macro-F1 (%)", f"fig_macroF1_{m}node.png", m)

# ----- headline table: all nodes attacked, strongest setting of each attack
pd.set_option("display.width", 200)
cols = ["attack", "param", "scheme", "accuracy", "macro_f1", "detection_rate", "lit_err_before",
        "lit_err_after", "flagged_codewords"]
for m in (1, 3):
    t = df[(df.attacked_nodes == m) & (df.attack != "clean")][cols].copy()
    for c in ("accuracy", "macro_f1", "detection_rate"):
        t[c] = (t[c] * 100).round(2)
    t.to_csv(f"{BASE}/table_{m}node.csv", index=False)
print("clean:", {k: round(clean[k] * 100, 2) for k in ("accuracy", "macro_f1", "balanced_acc", "detection_rate")})
