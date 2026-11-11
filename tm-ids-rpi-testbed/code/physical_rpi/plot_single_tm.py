"""Single TM view of the physical Raspberry Pi runs: only TM 1 (the TM attacked through the laptop), measured on the
central RPi, no ECC (raw) vs Hamming(7,4), for the three blind attacks (K random bit positions anywhere in every TM 1
datagram set to 1 / set to 0 / flipped).
  row 1: accuracy of TM 1 alone on the 50 test flows        row 2: mean true-class sum of TM 1
usage: python plot_single_tm.py   -> fig_single_tm_rpi.pdf / .png  (+ single_tm_rpi.csv)
"""
import os
import pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACKS = [("include", "Include (bit → 1)", "bits set to 1"), ("exclude", "Exclude (bit → 0)", "bits set to 0"),
           ("toggle", "Toggle (bit flipped)", "bits flipped")]
LAY = {"raw": ("without ECC", "#D55E00", "o"), "hamming": ("with Hamming(7,4) ECC", "#0072B2", "s")}
CLEAN_ACC, CLEAN_SUM = 84.0, 25.92
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "legend.frameon": False, "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
                     "axes.linewidth": 0.9, "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
fig, axs = plt.subplots(2, 3, figsize=(10, 6.2), sharex="col")
rows = []
for j, (atk, title, verb) in enumerate(ATTACKS):
    M = pd.read_csv(f"{HERE}/measured_{atk}_all.csv")
    if "valid" in M:
        M = M[M.valid == "yes"]
    for lay, (name, col, mk) in LAY.items():
        m = M[M.layout == lay].groupby("K")[["acc_tm1", "true_sum_tm1", "tm1_dropped"]].mean().reset_index()
        if not len(m):
            continue
        for row, c in enumerate(("acc_tm1", "true_sum_tm1")):
            axs[row, j].plot(m.K, m[c], "-", marker=mk, color=col, lw=2.4, ms=7, mec="white", mew=0.8,
                             label=name, zorder=3)
        rows += [dict(attack=atk, layout=lay, K=int(r.K), tm1_accuracy=r.acc_tm1, tm1_mean_true_class_sum=r.true_sum_tm1)
                 for r in m.itertuples()]
        if lay == "raw" and (m.tm1_dropped >= 402).any():           # mark where all TM 1 packets are lost
            k0 = int(m.K[m.tm1_dropped >= 402].min())
            axs[0, j].annotate(f"without ECC, K = {k0}:\nheader bit hit,\nall TM 1 packets dropped", xy=(k0, 20),
                               xytext=(0.30, 0.36), textcoords="axes fraction", fontsize=8.5, color=col,
                               ha="left", va="center",
                               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
                               arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0))
    for row, (clean, lab) in enumerate(((CLEAN_ACC, f"clean TM 1 ({CLEAN_ACC:.0f} %)"),
                                        (CLEAN_SUM, f"clean TM 1 ({CLEAN_SUM})"))):
        axs[row, j].axhline(clean, color="#555555", lw=1.2, ls=":", label=lab if row == 0 else None, zorder=2)
    axs[1, j].axhline(0, color="black", lw=0.7)
    axs[0, j].set_ylim(0, 100)
    axs[1, j].set_ylim(-2, 30)
    axs[0, j].set_title(title, fontweight="bold", pad=8)
    axs[1, j].set_xlabel(f"K ({verb} per TM 1 datagram)")
    for row in range(2):
        axs[row, j].set_xlim(left=-0.03 * M.K.max())
axs[0, 0].set_ylabel("TM 1 accuracy (%)\n(50 test flows)")
axs[1, 0].set_ylabel("TM 1 mean\ntrue-class sum")
for row, letter in ((0, "abc"), (1, "def")):
    for j in range(3):
        axs[row, j].text(0.02, 0.96, f"({letter[j]})", transform=axs[row, j].transAxes, va="top", fontsize=10,
                         fontweight="bold")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.965), fontsize=10)
fig.suptitle("Single attacked TM on the Raspberry Pi testbed (TM 1, MedSec-25): without ECC vs with Hamming(7,4)",
             fontsize=11.5, y=1.045)
fig.tight_layout(h_pad=1.0, w_pad=1.2)
fig.savefig(f"{HERE}/fig_single_tm_rpi.pdf"); fig.savefig(f"{HERE}/fig_single_tm_rpi.png", dpi=300)
pd.DataFrame(rows).to_csv(f"{HERE}/single_tm_rpi.csv", index=False)
print("saved fig_single_tm_rpi.pdf / .png, single_tm_rpi.csv")
