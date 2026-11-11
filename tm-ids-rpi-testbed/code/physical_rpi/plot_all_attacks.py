"""All three blind attacks on TM 1 in one figure (physical Raspberry Pi testbed, MedSec-25, 3 TMs, 50 test flows):
columns = include / exclude / toggle (K random bit positions anywhere in every TM 1 datagram set to 1 / 0 / flipped
by the laptop forwarder), rows = accuracy and mean true-class sum, no ECC (raw) vs Hamming(7,4).
Markers = measured on the RPis, lines = prediction (check_run.py); filled black-edged markers = fused 3-TM model.
usage: python plot_all_attacks.py   -> fig_all_attacks_raw_vs_hamming.pdf / .png
"""
import os
import pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACKS = [("include", "set to 1"), ("exclude", "set to 0"), ("toggle", "flipped")]
LAY = {"raw": ("no ECC", "#D55E00", "o"), "hamming": ("Hamming(7,4)", "#0072B2", "s")}
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "axes.titlelocation": "left",
                     "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
fig, axs = plt.subplots(2, 3, figsize=(7.16, 4.6))
for j, (atk, verb) in enumerate(ATTACKS):
    M = pd.read_csv(f"{HERE}/measured_{atk}_all.csv")
    if "valid" in M:
        M = M[M.valid == "yes"]
    P = pd.read_csv(f"{HERE}/predicted_{atk}_all.csv")
    for lay, (name, col, mk) in LAY.items():
        m = M[M.layout == lay].groupby("K").mean(numeric_only=True).reset_index()
        p = P[P.layout == lay].sort_values("K")
        for row, (meas, pred) in enumerate((("acc", "acc"), ("true_sum", "true_sum"))):
            ax = axs[row, j]
            if len(p):
                ax.plot(p.K, p[f"tm1_{pred}"], "-", color=col, lw=1.1, alpha=0.55)
                ax.plot(p.K, p[f"fused_{pred}"], "--", color=col, lw=0.9, alpha=0.55)
            if len(m):
                ax.plot(m.K, m[f"{meas}_tm1"], mk, color=col, ms=4.5, mec="white", mew=0.5,
                        label=f"TM 1 (attacked), {name}")
                ax.plot(m.K, m[f"{meas}_fused"], mk, color=col, ms=4.5, mec="black", mew=0.6,
                        label=f"fused 3 TMs, {name}")
    lost = M[(M.layout == "raw") & (M.tm1_dropped >= 402)].K
    if len(lost):
        k0 = int(lost.min())
        axs[0, j].annotate(f"no ECC: header hit at K = {k0},\nall TM 1 packets dropped", (k0, 20),
                           xytext=(0.32, 0.3), textcoords="axes fraction", fontsize=6, color="#D55E00",
                           arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#D55E00"))
    axs[0, j].axhline(82, color="#009E73", lw=0.7, ls=":")
    axs[0, j].axhline(78, color="#CC79A7", lw=0.7, ls=":")
    axs[0, j].set_ylim(0, 100)
    axs[1, j].axhline(0, color="grey", lw=0.6)
    axs[0, j].set_title(f"({'abc'[j]}) {atk}: accuracy")
    axs[1, j].set_title(f"({'def'[j]}) {atk}: true-class sum")
    axs[1, j].set_xlabel(f"K (bits {verb} per TM 1 datagram)")
axs[0, 0].set_ylabel("accuracy, 50 flows (%)")
axs[1, 0].set_ylabel("mean true-class sum")
h, l = axs[0, 2].get_legend_handles_labels()
h += [plt.Line2D([], [], color="#009E73", ls=":", lw=0.9), plt.Line2D([], [], color="#CC79A7", ls=":", lw=0.9),
      plt.Line2D([], [], color="grey", ls="-", lw=1.1), plt.Line2D([], [], color="grey", ls="--", lw=0.9)]
l += ["TM 0 (82 %), not attacked", "TM 2 (78 %), not attacked", "prediction, TM 1", "prediction, fused"]
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=6.8)
fig.suptitle("Raspberry Pi testbed, MedSec-25, 3 TMs: blind attacks on TM 1 via the laptop, no ECC vs Hamming(7,4)",
             fontsize=8.5, y=1.01)
fig.tight_layout(h_pad=0.8)
fig.savefig(f"{HERE}/fig_all_attacks_raw_vs_hamming.pdf"); fig.savefig(f"{HERE}/fig_all_attacks_raw_vs_hamming.png", dpi=300)
print("saved fig_all_attacks_raw_vs_hamming.pdf / .png")
