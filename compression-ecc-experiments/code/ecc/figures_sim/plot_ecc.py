"""Figures for ecc_attacks.py (same style as medsciot fig1_A1 ... fig4_A4) + accuracy-vs-K line summary + MATLAB data.

usage: python plot_ecc.py {medsec|iot|mnist}
  -> figures/<ds>/ecc_attacks/fig{1..4}_A{1..4}_<ds>.png, fig5_accuracy_vs_K_<ds>.png, ecc_attacks_<ds>.mat
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
R = pd.read_csv(f"{OUT}/results_4attacks_{DS}.csv")
TG = pd.read_csv(f"{OUT}/targeted_literals_{DS}.csv")
K_VALUES = sorted(R.K.unique())
N = int(R.N.iloc[0])
BENCH = float(R["benchmark %"].iloc[0])
ATK = list(E.ATTACKS)


def acc(atk, k, lay, col="accuracy %"):
    return R[(R.attack == atk) & (R.K == k) & (R.layout == lay)][col].iloc[0]


P.setup()
for i, atk in enumerate(ATK, 1):
    if atk == "A1":
        groups = [f"K = {k}\n({k} random literal{'s' if k > 1 else ''} / clause)" for k in K_VALUES]
    else:
        groups = [f"1 literal\n({TG.literal[0]})" if k == 1 else f"{k} literals\n(+ {k - 1} more)" for k in K_VALUES]
    series = [(lab, [acc(atk, k, s) for k in K_VALUES], c) for s, lab, c in P.SCHEMES]
    fig, (ax,) = P.figure(1, 4, width=24)
    P.grouped_bars(ax, groups, series, bench=BENCH, n_total=N)
    ax.set_title(f"{DS.upper() if DS != 'medsec' else 'MedSec-25'} · {atk} · {E.ATTACKS[atk]}  ·  {E.INFO[DS]}", pad=8)
    P.top_legend(fig, ax)
    fig.savefig(f"{OUT}/fig{i}_{atk}_{DS}.png", dpi=150)
    plt.close(fig)

# accuracy vs K, one panel per attack
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
for ax, atk in zip(axes.ravel(), ATK):
    for (s, lab, c), mk in zip(P.SCHEMES, "osD^"):
        ax.plot(K_VALUES, [acc(atk, k, s) for k in K_VALUES], marker=mk, color=c, lw=2, ms=7, mec="white", label=lab)
    ax.axhline(BENCH, color="#222", ls="--", lw=1.2, label=f"benchmark {BENCH:.4f} %")
    ax.set_title(f"{atk} · {E.ATTACKS[atk]}", fontsize=11.5)
    ax.set_ylim(0, 102); ax.set_xticks(K_VALUES)
    ax.set_ylabel("accuracy (%)")
for ax in axes[1]:
    ax.set_xlabel("K  (A1: random literals per clause · A2-A4: number of most-used literals targeted)")
axes[0, 0].legend(loc="lower left", fontsize=9.5)
fig.suptitle(f"{E.INFO[DS]}: accuracy vs K for every attack and coding", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_accuracy_vs_K_{DS}.png", dpi=150)
plt.close(fig)

# MATLAB: arrays indexed (attack, K, layout)
cube = lambda col: np.array([[[acc(a, k, l, col) for l in E.LAYOUTS] for k in K_VALUES] for a in ATK], dtype=float)
scipy.io.savemat(f"{OUT}/ecc_attacks_{DS}.mat", {
    "dataset": DS, "description": E.INFO[DS], "attacks": np.array(ATK, dtype=object),
    "attack_desc": np.array([E.ATTACKS[a] for a in ATK], dtype=object),
    "layouts": np.array(["No ECC", "Hamming SEQ", "Hamming INT", "Hamming KEY"], dtype=object),
    "K": np.array(K_VALUES, float), "N_test": float(N), "benchmark_pct": BENCH,
    "benchmark_correct": float(R["benchmark correct"].iloc[0]),
    "accuracy_pct": cube("accuracy %"), "correct": cube("correct"), "delta_pct": cube("Δ vs benchmark"),
    "bits_changed": cube("bits changed by MITM"), "wrong_bits_after_decode": cube("wrong bits after decode"),
    "targeted_literals": np.array(TG.literal.tolist(), dtype=object)}, do_compression=True)
print(f"[{DS}] wrote fig1-4, fig5 and ecc_attacks_{DS}.mat in {OUT}")
