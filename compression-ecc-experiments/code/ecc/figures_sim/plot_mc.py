"""Monte Carlo figures (same style as medsciot fig6_mc_A2_K2-8.png ...) + statistics, steep-drop list, MATLAB data.

Marker = largest number of bits the MITM changed inside one Hamming codeword in that run (2 / 3 / 4 bits cannot be
repaired by Hamming(7,4)). Steep drop = accuracy below the median of its own series by more than
max(3 x robust std, 0.02 pp); the 3 deepest of every panel get a callout.

usage: python plot_mc.py {medsec|iot|mnist}
  -> figures/<ds>/monte_carlo/fig_mc_<atk>_K<a>-<b>_<ds>.png, mc_stats_<ds>.csv, mc_steep_drops_<ds>.csv,
     mc_<ds>.mat, plot_mc_panel.m
"""
import sys
import numpy as np
import pandas as pd
import scipy.io
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
import ecc_common as E

DS = sys.argv[1]
OUT = f"{E.ROOT}/{DS}/monte_carlo"
MC = pd.read_csv(f"{OUT}/mc_results_{DS}.csv").fillna({"shared codeword (SEQ)": "", "shared codeword (INT)": ""})
BENCH = float(MC["benchmark %"].iloc[0])
N = int(MC.N.iloc[0])
NIT = int(MC.iteration.max())
ALL_K = sorted(MC.K.unique())
ATKS = ["A2", "A3", "A4"]
ATK_NAME = {"A2": "include", "A3": "exclude", "A4": "toggle"}
ACTION = {"A2": "included", "A3": "excluded", "A4": "toggled"}


def steep_drops(d):
    acc = d["accuracy %"].to_numpy()
    med = np.median(acc)
    sig = 1.4826 * np.median(np.abs(acc - med))
    return d[acc < med - max(3 * sig, 0.02)]


MC["steep drop"] = False
for _, d in MC.groupby(["attack", "K", "layout"]):
    MC.loc[steep_drops(d).index, "steep drop"] = True

# ---- statistics and steep-drop table
stats = (MC.groupby(["K", "attack", "layout"], sort=False)["accuracy %"]
         .agg(mean="mean", variance="var", std="std", min="min", max="max").reset_index())
stats["mean Δ vs benchmark"] = stats["mean"] - BENCH
g = MC.groupby(["K", "attack", "layout"], sort=False)
stats["runs with ≥2 bits in one cw"] = g["max bits in one cw"].apply(lambda s: int((s >= 2).sum())).values
stats["steep drops"] = g["steep drop"].sum().values
stats.to_csv(f"{OUT}/mc_stats_{DS}.csv", index=False, float_format="%.10f")
cols = ["iteration", "K", "attack", "layout", "accuracy %", "correct", "Δ vs benchmark", "max bits in one cw",
        "cw 2 bits", "cw 3 bits", "cw 4 bits", "bits changed by MITM", "wrong bits after decode", "literals",
        "shared codeword (SEQ)", "shared codeword (INT)"]
MC[MC["steep drop"]].sort_values(["attack", "K", "layout", "accuracy %"])[cols].to_csv(
    f"{OUT}/mc_steep_drops_{DS}.csv", index=False, float_format="%.10f")

# ---- figures
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5,
    "axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "#000000", "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": 0.6,
    "legend.frameon": True, "legend.edgecolor": "#000000", "legend.fancybox": False, "legend.framealpha": 1.0,
    "figure.dpi": 100, "savefig.dpi": 200})
LCOL = {"none": "#d95319", "SEQ": "#edb120", "INT": "#77ac30", "KEY": "#0072bd"}        # MATLAB default colours
LNAME = {"none": "No ECC", "SEQ": "Hamming SEQ", "INT": "Hamming INT", "KEY": "Hamming KEY"}
BITSTYLE = {2: ("#ff7f0e", "^", "2 bits in one codeword"), 3: ("#d62728", "s", "3 bits in one codeword"),
            4: ("#7b1fa2", "D", "4 bits in one codeword")}


def callout(r, lay):
    t = f"it {int(r.iteration)}: {r['accuracy %']:.4f} %\nΔ {r['Δ vs benchmark']:+.4f} pp"
    if lay == "none":
        return t + f"\n{int(r['bits changed by MITM']):,} bits {ACTION[r.attack]}"
    parts = [f"{int(r[f'cw {b} bits']):,}×{b}-bit" for b in (2, 3, 4) if r[f"cw {b} bits"] > 0]
    return t + "\ncw: " + ("\n    ".join(parts) if parts else "all 1-bit")


K_BLOCKS = [ALL_K[:4], ALL_K[4:]]
for atk in ATKS:
    for KS in K_BLOCKS:
        fig, axes = plt.subplots(len(KS), 4, figsize=(26, 5.2 * len(KS)), sharex=True, squeeze=False)
        for r, K in enumerate(KS):
            for c, lay in enumerate(E.LAYOUTS):
                ax = axes[r, c]
                d = MC[(MC.attack == atk) & (MC.K == K) & (MC.layout == lay)].sort_values("iteration")
                acc = d["accuracy %"].to_numpy()
                ax.plot(d.iteration, acc, color=LCOL[lay], lw=1.2, zorder=2, label=LNAME[lay])
                ax.plot(d.iteration, acc, ls="none", marker="o", ms=2.6, color=LCOL[lay], zorder=3)
                ax.axhline(BENCH, color="#000000", ls="--", lw=1.1, zorder=1, label=f"benchmark {BENCH:.4f} %")
                ax.axhline(acc.mean(), color=LCOL[lay], ls=":", lw=1.5, zorder=1, label=f"mean {acc.mean():.4f} %")
                if lay != "none":
                    for b, (col, mk, lab) in BITSTYLE.items():
                        sel = d[d["max bits in one cw"] == b]
                        if len(sel):
                            ax.scatter(sel.iteration, sel["accuracy %"], marker=mk, s=46, color=col,
                                       edgecolor="black", linewidth=0.6, zorder=5, label=f"{lab} ({len(sel)}×)")
                dr = steep_drops(d).sort_values("accuracy %").head(3)
                if len(dr):
                    ax.scatter(dr.iteration, dr["accuracy %"], s=150, facecolor="none", edgecolor="black", lw=1.4,
                               zorder=6, label="steep drop")
                    for j, (_, row) in enumerate(dr.iterrows()):
                        ax.annotate(callout(row, lay), xy=(row.iteration, row["accuracy %"]),
                                    xytext=(0.19 + 0.31 * j, 0.30), textcoords="axes fraction", fontsize=7.6,
                                    ha="center", va="center",
                                    bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.6),
                                    arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8), zorder=7)
                lo, hi = min(acc.min(), BENCH), max(acc.max(), BENCH)
                pad = max((hi - lo) * 0.15, 0.01)
                ax.set_ylim(lo - 1.6 * pad, hi + pad)
                ax.set_xlim(0, NIT + 1)
                ax.set_title(f"{LNAME[lay]}  ·  K = {K} literals {ACTION[atk]}", loc="left", fontweight="bold")
                ax.text(0.99, 0.97, f"mean {acc.mean():.4f}\nstd {acc.std(ddof=1):.4f}\nvar {acc.var(ddof=1):.4f}\n"
                                    f"min {acc.min():.4f}", transform=ax.transAxes, ha="right", va="top", fontsize=8.8,
                        family="DejaVu Sans Mono", bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.6))
                ax.legend(loc="upper left", fontsize=8, ncol=1, borderaxespad=0.3)
                if c == 0:
                    ax.set_ylabel(f"accuracy on {N:,} {E.UNIT[DS]} (%)")
                if r == len(KS) - 1:
                    ax.set_xlabel("iteration (new random literal set each time)")
        forced = (MC[(MC.attack == atk) & (MC.layout == "none") & MC.K.isin(KS)].fillna({"forced shared cw": ""})
                  .query("`forced shared cw` != ''"))
        note = ("" if forced.empty else "   ·   re-drawn to contain a shared codeword: " +
                ", ".join(f"K={r.K} it {r.iteration} ({r['forced shared cw']})" for _, r in forced.iterrows()))
        fig.suptitle(f"{E.INFO[DS]}\n{atk}: {ACTION[atk]} K random literals (same set in every clause of the TM) · "
                     f"K = {KS[0]}-{KS[-1]} · {NIT} iterations{note}", x=0.005, ha="left", fontsize=15,
                     fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
        fig.savefig(f"{OUT}/fig_mc_{atk}_K{KS[0]}-{KS[-1]}_{DS}.png")
        plt.close(fig)

# ---- MATLAB: arrays indexed (iteration, K, attack, layout)
def arr(col):
    a = np.full((NIT, len(ALL_K), len(ATKS), len(E.LAYOUTS)), np.nan)
    for (ki, K) in enumerate(ALL_K):
        for ai, atk in enumerate(ATKS):
            for li, lay in enumerate(E.LAYOUTS):
                d = MC[(MC.K == K) & (MC.attack == atk) & (MC.layout == lay)].sort_values("iteration")
                a[d.iteration.to_numpy() - 1, ki, ai, li] = d[col].astype(float).to_numpy()
    return a


scipy.io.savemat(f"{OUT}/mc_{DS}.mat", {
    "dataset": DS, "description": E.INFO[DS], "N_test": float(N), "benchmark_pct": BENCH,
    "benchmark_correct": float(MC["benchmark correct"].iloc[0]),
    "iteration": np.arange(1, NIT + 1, dtype=float), "K": np.array(ALL_K, float),
    "attacks": np.array([f"{a} {ATK_NAME[a]}" for a in ATKS], dtype=object),
    "layouts": np.array([LNAME[l] for l in E.LAYOUTS], dtype=object),
    "dims": "arrays are (iteration, K, attack, layout)",
    "accuracy_pct": arr("accuracy %"), "correct": arr("correct"), "delta_pct": arr("Δ vs benchmark"),
    "max_bits_in_one_cw": arr("max bits in one cw"), "cw_1bit": arr("cw 1 bit"), "cw_2bits": arr("cw 2 bits"),
    "cw_3bits": arr("cw 3 bits"), "cw_4bits": arr("cw 4 bits"), "bits_changed": arr("bits changed by MITM"),
    "wrong_bits_after_decode": arr("wrong bits after decode"), "steep_drop": arr("steep drop")},
    do_compression=True)

with open(f"{OUT}/plot_mc_panel.m", "w") as f:
    f.write(f"""% One Monte Carlo panel from mc_{DS}.mat (arrays are iteration x K x attack x layout).
S = load('mc_{DS}.mat');
k = find(S.K == 4); a = 1; l = 2;          % K = 4, attack 1 = A2 include, layout 2 = Hamming SEQ
acc = S.accuracy_pct(:, k, a, l); mb = S.max_bits_in_one_cw(:, k, a, l); sd = S.steep_drop(:, k, a, l) == 1;
figure; plot(S.iteration, acc, '-o', 'MarkerSize', 3); hold on; grid on; box on;
yline(S.benchmark_pct, 'k--', sprintf('benchmark %.4f %%', S.benchmark_pct));
yline(mean(acc), ':', sprintf('mean %.4f %%', mean(acc)));
plot(S.iteration(mb == 2), acc(mb == 2), '^', 'MarkerFaceColor', [1 .5 .05], 'MarkerEdgeColor', 'k');
plot(S.iteration(mb == 3), acc(mb == 3), 's', 'MarkerFaceColor', [.84 .15 .16], 'MarkerEdgeColor', 'k');
plot(S.iteration(mb == 4), acc(mb == 4), 'd', 'MarkerFaceColor', [.48 .12 .64], 'MarkerEdgeColor', 'k');
plot(S.iteration(sd), acc(sd), 'ko', 'MarkerSize', 12, 'LineWidth', 1.2);
xlabel('iteration (new random literal set each time)'); ylabel('accuracy (%)');
title(sprintf('%s, %s, K = %d', S.layouts{{l}}, S.attacks{{a}}, S.K(k)));
""")
print(f"[{DS}] wrote MC figures, mc_stats_{DS}.csv, mc_steep_drops_{DS}.csv, mc_{DS}.mat, plot_mc_panel.m")
print(stats[stats.layout != "none"].groupby(["attack", "layout"])["runs with ≥2 bits in one cw"].sum().to_string())
