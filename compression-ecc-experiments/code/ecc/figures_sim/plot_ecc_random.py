"""Bar figures (style of medsciot fig2_A2 ...) for RANDOM literals that are the same in every clause:
from the 150-iteration Monte Carlo (mc_results_<ds>.csv). In every iteration K random literals are drawn and included
(A2) / excluded (A3) / toggled (A4) in every clause; bar = mean accuracy over the 150 iterations, table = mean, Δ vs
benchmark, std and min-max.

usage: python plot_ecc_random.py {medsec|iot|mnist}
  -> figures/<ds>/ecc_attacks/fig_random_A{2,3,4}_<ds>.png, random_bars_<ds>.csv, random_bars_<ds>.mat
"""
import sys
import numpy as np
import pandas as pd
import scipy.io
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import plot_style as P
import ecc_common as E

DS = sys.argv[1]
OUT = f"{E.ROOT}/{DS}/ecc_attacks"
MC = pd.read_csv(f"{E.ROOT}/{DS}/monte_carlo/mc_results_{DS}.csv")
BENCH = float(MC["benchmark %"].iloc[0])
N = int(MC.N.iloc[0])
NIT = int(MC.iteration.max())
K_VALUES = sorted(MC.K.unique())
ATKS = {"A2": "INCLUDE", "A3": "EXCLUDE", "A4": "TOGGLE"}
NAME = {"medsec": "MedSec-25", "iot": "Edge-IIoTset", "mnist": "MNIST"}[DS]

S = (MC.groupby(["attack", "K", "layout"])["accuracy %"]
     .agg(mean="mean", std=lambda s: s.std(ddof=1), min="min", max="max").reset_index())
S["mean correct"] = MC.groupby(["attack", "K", "layout"])["correct"].mean().values
S["Δ mean vs benchmark"] = S["mean"] - BENCH
S["iterations"] = NIT
S.to_csv(f"{OUT}/random_bars_{DS}.csv", index=False, float_format="%.10f")


def val(atk, k, lay, col):
    return S[(S.attack == atk) & (S.K == k) & (S.layout == lay)][col].iloc[0]


P.setup()
for atk, verb in ATKS.items():
    groups = [f"K = {k}\n({k} random literals,\nsame in every clause)" for k in K_VALUES]
    series = [(lab, [val(atk, k, s, "mean") for k in K_VALUES], c) for s, lab, c in P.SCHEMES]
    cells = [[f"mean {val(atk, k, s, 'mean'):.4f} %   Δ {val(atk, k, s, 'mean') - BENCH:+.4f}\n"
              f"std {val(atk, k, s, 'std'):.4f} · min {val(atk, k, s, 'min'):.4f} · max {val(atk, k, s, 'max'):.4f}"
              for k in K_VALUES] for s, _, _ in P.SCHEMES]
    fig, (ax,) = P.figure(1, 4, width=26)
    P.grouped_bars(ax, groups, series, bench=BENCH, n_total=N, cell_text=cells)
    ax.set_title(f"{atk} · {verb} K random literals (same literals in every clause, new random set each "
                 f"iteration) · mean of {NIT} iterations  ·  {E.INFO[DS]}", pad=8)
    P.top_legend(fig, ax, extra_note=f"value on each bar = mean accuracy % over {NIT} iterations  (change vs benchmark)"
                                     "  ·  table = mean, std, min, max")
    fig.savefig(f"{OUT}/fig_random_{atk}_{DS}.png", dpi=150)
    plt.close(fig)

arr = lambda col: np.array([[[val(a, k, l, col) for l in E.LAYOUTS] for k in K_VALUES] for a in ATKS], dtype=float)
scipy.io.savemat(f"{OUT}/random_bars_{DS}.mat", {
    "dataset": DS, "description": E.INFO[DS], "dims": "arrays are (attack, K, layout)",
    "attacks": np.array([f"{a} {v.lower()}" for a, v in ATKS.items()], dtype=object),
    "layouts": np.array(["No ECC", "Hamming SEQ", "Hamming INT", "Hamming KEY"], dtype=object),
    "K": np.array(K_VALUES, float), "iterations": float(NIT), "N_test": float(N), "benchmark_pct": BENCH,
    "mean_pct": arr("mean"), "std_pct": arr("std"), "min_pct": arr("min"), "max_pct": arr("max"),
    "mean_correct": arr("mean correct")}, do_compression=True)
print(f"[{DS}] wrote fig_random_A2/A3/A4_{DS}.png, random_bars_{DS}.csv, random_bars_{DS}.mat")
