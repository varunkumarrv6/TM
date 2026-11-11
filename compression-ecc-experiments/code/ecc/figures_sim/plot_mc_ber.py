"""Figure + summary for mc_ber.py (independent bit flips with probability p, 150 iterations per p).
  (a) accuracy vs p, unprotected vs Hamming(7,4), mean +- 95 % CI
  (b) codewords with >= 2 errors: measured vs 1-(1-p)^7-7p(1-p)^6
  (c) clauses left corrupted: measured vs 1-(1-p)^L (No ECC) and 1-(1-P2)^(RB/4) (Hamming)

usage: SPLIT=70_15_15 python plot_mc_ber.py {medsec|iot}
  -> split_<SPLIT>/<ds>/monte_carlo_ber/fig_mc_ber_<ds>.png, summary_mc_ber_<ds>.csv, mc_ber_<ds>.mat
"""
import sys
import numpy as np
import pandas as pd
import scipy.io
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
import ecc_common as E

DS = sys.argv[1]
OUT = f"{E.ROOT}/{DS}/monte_carlo_ber"
D = pd.read_csv(f"{OUT}/mc_ber_{DS}.csv")
m = E.load(DS)
C, CL, L = m["cube"].shape
RB = -(-L // 8) * 8
BENCH = float(D.benchmark.iloc[0])
FLOOR = (m["yte"] == 0).mean() * 100
NIT = int(D.iteration.max())

S = (D.groupby(["coding", "p"]).agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                                    acc_min=("accuracy", "min"), acc_max=("accuracy", "max"),
                                    n=("accuracy", "count"), cw2_measured=("frac_cw_2plus", "mean"),
                                    clauses_corrupted_measured=("frac_clauses_corrupted", "mean"),
                                    wrong_literal_bits=("wrong_literal_bits", "mean")).reset_index())
S["acc_ci95"] = 1.96 * S.acc_std / np.sqrt(S.n)
p = S.p.to_numpy()
P2 = 1 - (1 - p) ** 7 - 7 * p * (1 - p) ** 6
S["cw2_formula"] = np.where(S.coding == "Hamming(7,4)", P2, np.nan)
S["clauses_corrupted_predicted"] = np.where(S.coding == "No ECC", 1 - (1 - p) ** L, 1 - (1 - P2) ** (RB // 4))
S.to_csv(f"{OUT}/summary_mc_ber_{DS}.csv", index=False, float_format="%.10g")

COL = {"No ECC": "#d95319", "Hamming(7,4)": "#1baf7a"}
mpl.rcParams.update({"font.size": 11.5, "axes.grid": True, "grid.color": "#e6e6e6", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
ax = axes[0]
for c in ("No ECC", "Hamming(7,4)"):
    d = S[S.coding == c]
    ax.plot(d.p, d.acc_mean, marker="o", lw=2.2, color=COL[c], label=c)
    ax.fill_between(d.p, d.acc_mean - d.acc_ci95, d.acc_mean + d.acc_ci95, color=COL[c], alpha=0.2, lw=0)
ax.axhline(BENCH, color="k", ls=":", label=f"benchmark {BENCH:.2f} %")
ax.axhline(FLOOR, color="gray", ls="--", lw=1, label=f"class-0 share {FLOOR:.2f} %")
ax.set_xscale("log"); ax.set_ylim(0, 100)
ax.set(xlabel="channel bit error rate p", ylabel=f"test accuracy (%), mean of {NIT} iterations ± 95 % CI",
       title="(a) accuracy vs p")
ax.legend(loc="lower left")
ax = axes[1]
d = S[S.coding == "Hamming(7,4)"]
ax.plot(d.p, d.cw2_measured, "o", ms=8, color=COL["Hamming(7,4)"], label="measured")
pp = np.logspace(np.log10(p.min()), np.log10(p.max()), 200)
ax.plot(pp, 1 - (1 - pp) ** 7 - 7 * pp * (1 - pp) ** 6, "k-", lw=1.4, label=r"$1-(1-p)^7-7p(1-p)^6$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set(xlabel="channel bit error rate p", ylabel="share of codewords with ≥ 2 errors",
       title="(b) Hamming(7,4) codewords that cannot be repaired")
ax.legend()
ax = axes[2]
for c in ("No ECC", "Hamming(7,4)"):
    d = S[S.coding == c]
    ax.plot(d.p, d.clauses_corrupted_measured * 100, "o", ms=8, color=COL[c], label=f"{c} measured")
    pr = (1 - (1 - pp) ** L) if c == "No ECC" else 1 - (1 - (1 - (1 - pp) ** 7 - 7 * pp * (1 - pp) ** 6)) ** (RB // 4)
    ax.plot(pp, pr * 100, "-", color=COL[c], lw=1.4,
            label=f"{c} predicted: " + (rf"$1-(1-p)^{{{L}}}$" if c == "No ECC" else rf"$1-(1-P_2)^{{{RB // 4}}}$"))
ax.set_xscale("log"); ax.set_ylim(-2, 102)
ax.set(xlabel="channel bit error rate p", ylabel="clauses left corrupted (%)",
       title="(c) clauses with at least one wrong literal after decoding")
ax.legend(fontsize=9.5, loc="upper left")
fig.suptitle(f"{E.INFO[DS]}: independent bit flips with probability p ({NIT} Monte Carlo iterations per p)",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_mc_ber_{DS}.png", dpi=170)
arr = lambda c, col: S[S.coding == c].sort_values("p")[col].to_numpy(float)
scipy.io.savemat(f"{OUT}/mc_ber_{DS}.mat", {
    "p": np.sort(S.p.unique()), "benchmark_pct": BENCH, "class0_share_pct": FLOOR, "iterations": float(NIT),
    "acc_mean_noecc": arr("No ECC", "acc_mean"), "acc_ci95_noecc": arr("No ECC", "acc_ci95"),
    "acc_mean_hamming": arr("Hamming(7,4)", "acc_mean"), "acc_ci95_hamming": arr("Hamming(7,4)", "acc_ci95"),
    "cw2_measured": arr("Hamming(7,4)", "cw2_measured"), "cw2_formula": arr("Hamming(7,4)", "cw2_formula"),
    "clauses_corrupted_noecc": arr("No ECC", "clauses_corrupted_measured"),
    "clauses_corrupted_hamming": arr("Hamming(7,4)", "clauses_corrupted_measured"),
    "clauses_pred_noecc": arr("No ECC", "clauses_corrupted_predicted"),
    "clauses_pred_hamming": arr("Hamming(7,4)", "clauses_corrupted_predicted")}, do_compression=True)
print(S[["coding", "p", "acc_mean", "acc_ci95", "cw2_measured", "cw2_formula", "clauses_corrupted_measured",
         "clauses_corrupted_predicted"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
