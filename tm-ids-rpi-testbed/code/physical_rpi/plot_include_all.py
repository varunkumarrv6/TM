"""Physical 3-TM testbed: blind include attack on TM 1 (K random bit positions anywhere in its datagrams, set to 1 by
the laptop forwarder), no ECC (raw) vs Hamming(7,4). Markers = measured on the Raspberry Pis, lines = offline
prediction (check_run.py).
  (a) accuracy on the 50 flows of TM 1 (attacked) and of the fused 3-TM model (TM 0 / TM 2 constant)
  (b) mean true-class sum of TM 1 and of the fused model
usage: python plot_include_all.py [include|toggle|exclude]   -> fig_<attack>_all_raw_vs_hamming.pdf / .png
"""
import os
import sys
import pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ATK = sys.argv[1] if len(sys.argv) > 1 else "include"
M = pd.read_csv(f"{HERE}/measured_{ATK}_all.csv")
if "valid" in M:
    M = M[M.valid == "yes"]                                   # e.g. run 66: TM 0 missing, TM 1 sent twice
PF = f"{HERE}/predicted_{ATK}_all.csv"
P = pd.read_csv(PF) if os.path.exists(PF) else None
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "axes.titlelocation": "left",
                     "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
LAY = {"raw": ("no ECC", "#D55E00", "o"), "hamming": ("Hamming(7,4)", "#0072B2", "s")}
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.9))
for lay, (name, col, mk) in LAY.items():
    m = M[M.layout == lay].groupby("K").mean(numeric_only=True).reset_index()
    if not len(m):
        if P is None or not (P.layout == lay).any():
            continue                                             # neither measured nor predicted: leave it out
        name += " (predicted only)"
    for ax, tm1, fu in ((axs[0], "acc_tm1", "acc_fused"), (axs[1], "true_sum_tm1", "true_sum_fused")):
        if P is not None:
            p = P[P.layout == lay].sort_values("K")
            pred = {"acc_tm1": "tm1_acc", "true_sum_tm1": "tm1_true_sum",
                    "acc_fused": "fused_acc", "true_sum_fused": "fused_true_sum"}
            ax.plot(p.K, p[pred[tm1]], "-", color=col, lw=1.2, alpha=0.6)
            ax.plot(p.K, p[pred[fu]], "--", color=col, lw=1.0, alpha=0.6)
        ax.plot(m.K, m[tm1], mk, color=col, ms=5.5, mec="white", label=f"TM 1 (attacked), {name}")
        ax.plot(m.K, m[fu], mk, color="black", mfc=col, ms=5.5, mec="black", mew=0.6, label=f"fused 3 TMs, {name}")
axs[0].axhline(82, color="#009E73", lw=0.8, ls=":", label="TM 0 (82 %), not attacked")
axs[0].axhline(78, color="#CC79A7", lw=0.8, ls=":", label="TM 2 (78 %), not attacked")
axs[0].set_ylim(0, 100); axs[0].set_ylabel("accuracy, 50 test flows (%)")
axs[0].set_title("(a) accuracy")
axs[1].set_ylabel("mean true-class sum"); axs[1].set_title("(b) class sum of the true class")
axs[1].axhline(0, color="grey", lw=0.6)
lost = M[(M.layout == "raw") & (M.tm1_dropped >= 402)].K                  # first K that kills every TM 1 packet
if len(lost):
    k0 = int(lost.min())
    axs[0].annotate(f"no ECC, K = {k0}: a clear-header bit is changed,\nall 402 TM 1 packets dropped", (k0, 20),
                    xytext=(k0 + 0.08 * M.K.max(), 8), fontsize=6.5, color="#D55E00",
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#D55E00"))
for ax in axs:
    ax.set_xlabel("K (bits %s in every TM 1 datagram)" % {"include": "set to 1", "exclude": "set to 0"}.get(ATK, "flipped"))
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=6.8)
fig.suptitle(f"Raspberry Pi testbed: blind {ATK} attack on TM 1 via the laptop -- markers measured, lines predicted",
             fontsize=8.5, y=1.02)
fig.tight_layout()
fig.savefig(f"{HERE}/fig_{ATK}_all_raw_vs_hamming.pdf"); fig.savefig(f"{HERE}/fig_{ATK}_all_raw_vs_hamming.png", dpi=300)
print(f"saved fig_{ATK}_all_raw_vs_hamming.pdf / .png")
