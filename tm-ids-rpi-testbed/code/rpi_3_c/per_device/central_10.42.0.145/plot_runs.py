"""Plots of the physical runs (results/runs.csv written by the central RPi), one line per TM + fused, against K:
  fig_accuracy_vs_K    accuracy on the 50 flows of TM 0, TM 1, TM 2 (attacked via the laptop) and the fused model
  fig_true_sum_vs_K    mean true-class sum of the same four (TM 2's collapses, the fused one keeps a margin)
one column per (attack, layout) that occurs in the runs. Runs with K = 0 / attack none are the clean reference.
Also writes runs_by_K.csv (mean over repeated runs with the same attack, layout and K) for MATLAB.

usage: python3 plot_runs.py [results/runs.csv] [--out results/figures]
"""
import argparse
import os
import pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="?", default=os.path.join(HERE, "results", "runs.csv"))
ap.add_argument("--out", default=None)
a = ap.parse_args()
OUT = a.out or os.path.join(os.path.dirname(a.runs), "figures")
os.makedirs(OUT, exist_ok=True)

R = pd.read_csv(a.runs)
R["attack"] = R.attack.fillna("none")
R = R[R.complete == 1]
AN = int(R.attacked_node.iloc[0]) if "attacked_node" in R else 2       # TM routed through the laptop
src = {f"tm{n}": (f"TM {n}" + (" (attacked via laptop)" if n == AN else ""), "#D55E00" if n == AN else c, m)
       for n, c, m in ((0, "#009E73", "o"), (1, "#CC79A7", "s"), (2, "#56B4E9", "^"))}
src["fused"] = ("fused 3 TMs", "#000000", "D")
cols = [c for n in src for c in (f"acc_{n}", f"true_sum_{n}")]
groups = []
for (atk, lay), g in R[R.attack != "none"].groupby(["attack", "layout"]):
    clean = R[(R.attack == "none") & (R.layout == lay)]
    groups.append((atk, lay, pd.concat([clean.assign(K=0), g])))
if not groups:
    raise SystemExit("no attacked runs in " + a.runs)
summ = pd.concat([g.assign(attack=atk).groupby(["attack", "layout", "K"])[cols].mean().reset_index()
                  for atk, lay, g in groups])
summ.to_csv(f"{OUT}/runs_by_K.csv", index=False)

mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "axes.titlelocation": "left",
                     "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
for what, ylab, fname in (("acc", "accuracy, 50 test flows (%)", "fig_accuracy_vs_K"),
                          ("true_sum", "mean true-class sum", "fig_true_sum_vs_K")):
    fig, axs = plt.subplots(1, len(groups), figsize=(max(3.5, 2.6 * len(groups)), 2.6), squeeze=False, sharey=True)
    for j, (atk, lay, _) in enumerate(groups):
        ax = axs[0, j]
        d = summ[(summ.attack == atk) & (summ.layout == lay)].sort_values("K")
        for n, (name, col, mk) in src.items():
            ax.plot(d.K, d[f"{what}_{n}"], "-", marker=mk, color=col, ms=5, mec="white", lw=2.0 if n == "fused" else 1.4,
                    label=name)
        ax.set_title(f"({chr(97 + j)}) {atk}, {'Hamming(7,4)' if lay == 'hamming' else 'no ECC'}")
        ax.set_xlabel(f"K (literals changed in every rule of TM {AN})")
        ax.set_xticks(sorted(d.K.unique()))
        if what == "acc":
            ax.set_ylim(0, 100)
        else:
            ax.axhline(0, color="grey", lw=0.6)
    axs[0, 0].set_ylabel(ylab)
    axs[0, -1].legend(loc="best")
    fig.suptitle(f"Physical 3-TM testbed (Raspberry Pi 4): TM {AN} attacked by the laptop forwarder", fontsize=8.5, y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}.pdf"); fig.savefig(f"{OUT}/{fname}.png", dpi=300); plt.close(fig)
print(summ.round(2).to_string(index=False))
print("figures ->", OUT)
