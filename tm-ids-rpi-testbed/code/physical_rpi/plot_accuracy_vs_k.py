"""Accuracy vs K on the physical Raspberry Pi testbed (MedSec-25, 3 TMs, 50 test flows), with and without ECC.
One panel per blind attack on TM 1 (K random bit positions anywhere in every TM 1 datagram set to 1 / set to 0 /
flipped by the laptop forwarder). Lines connect the measured runs only (no prediction):
  TM 1 (attacked) and the fused 3-TM model, no ECC (raw) vs Hamming(7,4); TM 0 / TM 2 (not attacked) as references.
usage: python plot_accuracy_vs_k.py   -> fig_accuracy_vs_k_rpi.pdf / .png  (+ accuracy_vs_k_rpi.csv for MATLAB)
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
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.7), sharey=True)
rows = []
for j, (atk, verb) in enumerate(ATTACKS):
    M = pd.read_csv(f"{HERE}/measured_{atk}_all.csv")
    if "valid" in M:
        M = M[M.valid == "yes"]
    ax = axs[j]
    for lay, (name, col, mk) in LAY.items():
        m = M[M.layout == lay].groupby("K")[["acc_tm1", "acc_fused"]].mean().reset_index()
        if not len(m):
            continue
        ax.plot(m.K, m.acc_tm1, "-", marker=mk, color=col, lw=1.4, ms=4.5, mec="white", mew=0.5,
                label=f"TM 1 (attacked), {name}")
        ax.plot(m.K, m.acc_fused, "--", marker=mk, color=col, lw=1.1, ms=4.5, mfc="white", mec=col, mew=0.9,
                label=f"fused 3 TMs, {name}")
        rows += [dict(attack=atk, layout=lay, K=int(r.K), acc_tm1=r.acc_tm1, acc_fused=r.acc_fused)
                 for r in m.itertuples()]
    ax.axhline(82, color="#009E73", lw=0.7, ls=":", label="TM 0, not attacked (82 %)")
    ax.axhline(78, color="#CC79A7", lw=0.7, ls=":", label="TM 2, not attacked (78 %)")
    ax.set_ylim(0, 100)
    ax.set_title(f"({'abc'[j]}) {atk}")
    ax.set_xlabel(f"K (bits {verb} per TM 1 datagram)")
axs[0].set_ylabel("accuracy, 50 test flows (%)")
h, l = axs[2].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=6.8)
fig.suptitle("Raspberry Pi testbed (MedSec-25, 3 TMs): accuracy vs K, TM 1 attacked via the laptop, "
             "no ECC vs Hamming(7,4)", fontsize=8.5, y=1.03)
fig.tight_layout()
fig.savefig(f"{HERE}/fig_accuracy_vs_k_rpi.pdf"); fig.savefig(f"{HERE}/fig_accuracy_vs_k_rpi.png", dpi=300)
pd.DataFrame(rows).to_csv(f"{HERE}/accuracy_vs_k_rpi.csv", index=False)
print("saved fig_accuracy_vs_k_rpi.pdf / .png, accuracy_vs_k_rpi.csv")
