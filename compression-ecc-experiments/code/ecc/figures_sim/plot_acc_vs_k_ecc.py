"""Accuracy vs K for No ECC / Hamming SEQ / INT / KEY, from the 150-iteration Monte Carlo (same K random literals in
every clause, new set each iteration). Line = mean over iterations, band = 95 % CI of the mean, dotted = benchmark.

usage: SPLIT=70_15_15 python plot_acc_vs_k_ecc.py {medsec|iot|mnist}
  -> split_<SPLIT>/<ds>/monte_carlo/fig_acc_vs_K_ecc_<ds>.png, acc_vs_K_ecc_<ds>.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
import ecc_common as E

DS = sys.argv[1]
OUT = f"{E.ROOT}/{DS}/monte_carlo"
MC = pd.read_csv(f"{OUT}/mc_results_{DS}.csv")
BENCH = float(MC["benchmark %"].iloc[0])
NIT = int(MC.iteration.max())
LNAME = {"none": "No ECC", "SEQ": "Hamming SEQ", "INT": "Hamming INT", "KEY": "Hamming KEY"}
LCOL = {"none": "#d95319", "SEQ": "#edb120", "INT": "#77ac30", "KEY": "#0072bd"}
MK = {"none": "o", "SEQ": "s", "INT": "D", "KEY": "^"}
ATK = {"A2": "include", "A3": "exclude", "A4": "toggle"}

S = (MC.groupby(["attack", "K", "layout"])["accuracy %"]
     .agg(mean="mean", std=lambda s: s.std(ddof=1), n="count", min="min", max="max").reset_index())
S["ci95"] = 1.96 * S["std"] / np.sqrt(S["n"])
S.to_csv(f"{OUT}/acc_vs_K_ecc_{DS}.csv", index=False, float_format="%.6f")

mpl.rcParams.update({"font.size": 11.5, "axes.grid": True, "grid.color": "#e6e6e6", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=True)
for ax, (atk, name) in zip(axes, ATK.items()):
    for lay in E.LAYOUTS:
        d = S[(S.attack == atk) & (S.layout == lay)].sort_values("K")
        ax.plot(d.K, d["mean"], marker=MK[lay], ms=7, lw=2.2, color=LCOL[lay], mec="white", label=LNAME[lay])
        ax.fill_between(d.K, d["mean"] - d.ci95, d["mean"] + d.ci95, color=LCOL[lay], alpha=0.18, lw=0)
    ax.axhline(BENCH, color="k", ls=":", lw=1.3, label=f"benchmark {BENCH:.2f} %")
    ax.set_title(f"{atk} · {name} K random literals (same in every clause)", loc="left")
    ax.set_xlabel("K (literals with errors per clause)")
    ax.set_xticks(sorted(S.K.unique())); ax.set_ylim(0, 102)
axes[0].set_ylabel(f"test accuracy (%) — mean of {NIT} iterations, band = 95 % CI")
axes[0].legend(loc="center left")
fig.suptitle(f"{E.INFO[DS]}: accuracy vs K for No ECC and Hamming SEQ / INT / KEY", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_acc_vs_K_ecc_{DS}.png", dpi=170)
print(f"[{DS}] {OUT}/fig_acc_vs_K_ecc_{DS}.png")
print(S[S.attack == "A2"].pivot(index="K", columns="layout", values="mean")[E.LAYOUTS].round(2).to_string())
