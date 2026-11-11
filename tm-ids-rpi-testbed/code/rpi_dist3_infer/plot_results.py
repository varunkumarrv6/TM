"""Figures + plotting data from the receiver's CSV files (Raspberry Pi run) and, optionally, the offline twin runs.

  fig_acc_vs_K         accuracy of the 50 flows vs K per attack: no ECC vs Hamming (testbed: TM 2 attacked), plus
                       the simulated cases with 2 of 3 and all 3 TMs attacked (dashed) if given
  fig_class_sums_vs_K  why: mean class sums over the 50 flows vs K -- the attacked TM's true-class sum and largest
                       other sum (top), and the fused margin true - best other (bottom); TM 0/1 sums stay constant
  fig_timing           per-round decode / ECC / inference time on the central RPi, no ECC vs Hamming, and the
                       end-to-end time split into network span and tail (last packet -> predictions)
  summary_by_K.csv     the plotted numbers (mean / min / max over iterations) for MATLAB or other tools

usage: python plot_results.py [results] [--sim expected_2of3:"2 of 3 TMs attacked (sim.)"] [--sim expected_all3:...]
                              [--out figures]
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="?", default=os.path.join(HERE, "results"))
ap.add_argument("--sim", action="append", default=[], help='dir:label of an offline run (simulate.py --nodes ...)')
ap.add_argument("--out", default=None)
a = ap.parse_args()
OUT = a.out or os.path.join(a.results, "figures")
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "axes.titlelocation": "left",
                     "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
LAY = {"raw": ("No ECC", "#D55E00", "o"), "hamming": ("Hamming(7,4)", "#0072B2", "s")}
ATTACKS = ["include", "exclude", "toggle"]
KT = [0, 2, 4, 6, 8, 12, 16, 20]                                      # K tick labels (K = 1 is plotted too)


def load(d, label):
    R = pd.read_csv(f"{d}/rounds.csv")
    S = pd.read_csv(f"{d}/class_sums.csv")
    cs = S[["c0", "c1", "c2", "c3", "c4"]].to_numpy()
    true = cs[np.arange(len(S)), S.actual.to_numpy()]
    other = cs.copy(); other[np.arange(len(S)), S.actual.to_numpy()] = np.iinfo(np.int64).min
    S["true_sum"], S["best_other"] = true, other.max(1)
    S["margin"], S["all_zero"] = S.true_sum - S.best_other, (cs == 0).all(1)
    g = S.groupby(["round", "source"]).agg(true_sum=("true_sum", "mean"), best_other=("best_other", "mean"),
                                           margin=("margin", "mean"), all_zero=("all_zero", "mean")).reset_index()
    w = g.pivot(index="round", columns="source")
    w.columns = [f"{s}_{v}" for v, s in w.columns]
    R = R.merge(w.reset_index(), on="round")
    R["label"] = label
    return R


def clean_for(R, layout):
    """The clean round of this layout, repeated for every attack at K = 0."""
    c = R[(R.attack == "none") & (R.layout == layout)]
    return pd.concat([c.assign(attack=atk) for atk in ATTACKS])


runs = [load(a.results, "testbed (TM 2 attacked)")]
for spec in a.sim:
    d, _, lab = spec.partition(":")
    runs.append(load(d, lab or os.path.basename(d)))
allr = []
for R in runs:
    for lay in LAY:
        allr.append(pd.concat([clean_for(R, lay), R[(R.attack != "none") & (R.layout == lay)]]))
A = pd.concat(allr)
val = ["acc_pct", "tm0_true_sum", "tm1_true_sum", "tm2_true_sum", "tm2_best_other", "fused_margin", "fused_all_zero"]
summ = A.groupby(["label", "layout", "attack", "K"])[val].agg(["mean", "min", "max"])
summ.columns = [f"{v}_{s}" for v, s in summ.columns]
summ = summ.reset_index()
summ.to_csv(f"{OUT}/summary_by_K.csv", index=False)

# ---- accuracy vs K
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=True)
ls = ["-", "--", ":"]
for j, atk in enumerate(ATTACKS):
    ax = axs[j]
    for i, R in enumerate(runs):
        for lay, (name, col, mk) in LAY.items():
            d = summ[(summ.label == R.label.iloc[0]) & (summ.layout == lay) & (summ.attack == atk)].sort_values("K")
            if i == 0:
                ax.fill_between(d.K, d.acc_pct_min, d.acc_pct_max, color=col, alpha=0.15, lw=0)
            ax.plot(d.K, d.acc_pct_mean, ls[i % 3], marker=mk if i == 0 else None, color=col, ms=4, mec="white",
                    lw=1.6 if i == 0 else 1.2, label=f"{name}, {R.label.iloc[0]}")
    ax.set_title(f"({'abc'[j]}) {atk} K literals")
    ax.set_xticks(KT)
    ax.set_ylim(0, 100)
axs[0].set_ylabel("accuracy, 50 test flows (%)")
fig.supxlabel("K (literals changed per clause, same K literals in every clause of an attacked TM)", fontsize=8, y=-0.06)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=7)
fig.suptitle("Distributed 3-TM MedSec-25 on Raspberry Pi 4: fused accuracy vs K (mean, band = min-max)",
             fontsize=8.5, y=1.02)
fig.savefig(f"{OUT}/fig_acc_vs_K.pdf"); fig.savefig(f"{OUT}/fig_acc_vs_K.png", dpi=300); plt.close(fig)

# ---- class sums vs K (testbed run and, dashed, the last simulated run)
fig, axs = plt.subplots(2, 3, figsize=(7.16, 4.2), sharex=True)
show = [runs[0]] + ([runs[-1]] if len(runs) > 1 else [])
for j, atk in enumerate(ATTACKS):
    for i, R in enumerate(show):
        lab = R.label.iloc[0]
        for lay, (name, col, mk) in LAY.items():
            d = summ[(summ.label == lab) & (summ.layout == lay) & (summ.attack == atk)].sort_values("K")
            st = "-" if i == 0 else "--"
            if i == 0:
                axs[0, j].plot(d.K, d.tm2_true_sum_mean, st, marker=mk, color=col, ms=4, mec="white", lw=1.5,
                               label=f"TM 2 true-class sum, {name}")
                axs[0, j].plot(d.K, d.tm2_best_other_mean, ":", marker=mk, color=col, ms=3, mfc="white", lw=1.1,
                               label=f"TM 2 largest other sum, {name}")
            axs[1, j].plot(d.K, d.fused_margin_mean, st, marker=mk if i == 0 else None, color=col, ms=4, mec="white",
                           lw=1.5 if i == 0 else 1.1, label=f"{name}, {lab}")
    d0 = summ[(summ.label == runs[0].label.iloc[0]) & (summ.layout == "raw") & (summ.attack == atk)].sort_values("K")
    axs[0, j].plot(d0.K, d0.tm0_true_sum_mean, "-", color="#009E73", lw=1, label="TM 0 true-class sum (not attacked)")
    axs[0, j].plot(d0.K, d0.tm1_true_sum_mean, "-", color="#CC79A7", lw=1, label="TM 1 true-class sum (not attacked)")
    for row in range(2):
        axs[row, j].axhline(0, color="grey", lw=0.6)
    axs[0, j].set_title(f"({'abc'[j]}) {atk}: class sums of the TMs")
    axs[1, j].set_title(f"({'def'[j]}) {atk}: fused margin")
    axs[1, j].set_xlabel("K (literals changed per clause)")
    axs[1, j].set_xticks(KT)
axs[0, 0].set_ylabel("mean class sum (50 flows)")
axs[1, 0].set_ylabel("mean (true - best other)")
for row, (y, ncol) in enumerate(((-0.02, 3), (-0.16, 2))):
    h, l = axs[row, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y), fontsize=6.5)
fig.tight_layout(h_pad=0.6)
fig.savefig(f"{OUT}/fig_class_sums_vs_K.pdf"); fig.savefig(f"{OUT}/fig_class_sums_vs_K.png", dpi=300); plt.close(fig)

# ---- timing (testbed only)
R = runs[0]
if R.e2e_ms.notna().any():
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.4))
    t = R.groupby("layout")[["decode_ms", "ecc_ms", "infer_ms", "net_ms", "tail_ms", "e2e_ms"]].median()
    t["processing_ms"] = t.decode_ms + t.ecc_ms + t.infer_ms
    x = np.arange(len(t))
    bot = np.zeros(len(t))
    for k, (c, col) in enumerate((("decode_ms", "#56B4E9"), ("ecc_ms", "#E69F00"), ("infer_ms", "#009E73"))):
        axs[0].bar(x, t[c], 0.55, bottom=bot, color=col, label=c.replace("_ms", ""))
        bot += t[c]
    for xi, v in zip(x, bot):
        axs[0].text(xi, v, f"{v:.2f} ms", ha="center", va="bottom", fontsize=7)
    axs[0].set_xticks(x, [LAY[l][0] for l in t.index]); axs[0].set_ylabel("busy time per round (ms)")
    axs[0].set_ylim(0, bot.max() * 1.18)
    axs[0].set_title("(a) receiver work per round (median)"); axs[0].legend(fontsize=7)
    R["proc_ms"] = R.decode_ms + R.ecc_ms + R.infer_ms
    for lay, (name, col, mk) in LAY.items():
        d = R[R.layout == lay].sort_values("round")
        axs[1].plot(d["round"], d.proc_ms, mk, color=col, ms=3, label=f"{name}: processing (decode + ECC + infer)")
        axs[1].plot(d["round"], d.net_ms, mk, color=col, ms=3, mfc="white", label=f"{name}: network span (first -> last packet)")
    axs[1].set_yscale("log"); axs[1].set_xlabel("round"); axs[1].set_ylabel("time per round (ms, log)")
    axs[1].set_title("(b) processing vs network span, every round")
    axs[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_timing.pdf"); fig.savefig(f"{OUT}/fig_timing.png", dpi=300); plt.close(fig)
    t.round(3).to_csv(f"{OUT}/timing_median_by_layout.csv")
print(summ[summ.attack == "include"][["label", "layout", "K", "acc_pct_mean", "tm2_true_sum_mean",
                                      "fused_margin_mean"]].round(2).to_string(index=False))
print("figures ->", OUT)
