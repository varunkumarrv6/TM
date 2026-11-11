"""Line graph from the Raspberry Pi 4 run: end-to-end latency (first datagram -> predictions of 256 flows) and its
decode / ECC / inference parts per method, with the accuracy of every method (identical: all reproduce the same
predictions).  usage: python plot_best_method.py   -> fig_best_method_pi.pdf / .png (this folder)
"""
import csv
import os
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(f"{HERE}/stage_breakdown_pi.csv")))
rows.sort(key=lambda r: -float(r["e2e_ms"]))                      # slowest -> fastest
ACC_TEST, ACC_SAMPLE = 96.2615, 232 / 256 * 100                   # 4-bit model: test split / 256-flow sample

mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "axes.titlelocation": "left",
                     "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
x = range(len(rows))
SHORT = {"C hybrid 2-phase (OpenMP decode+ECC + 1-core infer)": "C hybrid 2-phase",
         "C hybrid pipeline (3 decode+ECC workers + 1 infer core)": "C hybrid pipeline"}
lab = [f"{SHORT.get(r['method'], r['method'])}\n({r['cores']} core{'s' if r['cores'] != '1' else ''})" for r in rows]
fig, (axa, ax) = plt.subplots(2, 1, figsize=(7.16, 4.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 3.2]))
axa.plot(x, [ACC_TEST] * len(rows), "-o", color="#CC79A7", ms=4, lw=1.4, label=f"test split (78,801 flows): {ACC_TEST:.2f} %")
axa.plot(x, [ACC_SAMPLE] * len(rows), ":s", color="#CC79A7", ms=4, lw=1.4, label=f"256-flow sample: {ACC_SAMPLE:.2f} %")
axa.set_ylim(85, 100); axa.set_ylabel("accuracy (%)")
axa.set_title("(a) accuracy: identical for every method (all give the same predictions)")
axa.legend(loc="lower right", ncol=2, fontsize=6.5)
for key, name, col, mk, ls, lw in (("e2e_ms", "end-to-end", "#000000", "D", "-", 2.2),
                                    ("infer_ms", "inference", "#009E73", "o", "--", 1.3),
                                    ("decode_ms", "decode", "#56B4E9", "s", "--", 1.3),
                                    ("ecc_ms", "ECC", "#E69F00", "^", "--", 1.3)):
    ax.plot(x, [float(r[key]) for r in rows], ls=ls, marker=mk, color=col, lw=lw, ms=5, mec="white", label=name)
for i, r in enumerate(rows):
    v = float(r["e2e_ms"])
    ax.annotate(f"{v:,.0f} ms" if v >= 100 else f"{v:.3g} ms", (i, v), xytext=(4 if i == 0 else 0, 8),
                textcoords="offset points", ha="left" if i == 0 else "center", fontsize=7)
best = len(rows) - 1
bv = float(rows[best]["e2e_ms"])
ax.plot(best, bv, "o", ms=15, mfc="none", mec="#D55E00", mew=1.6)
ax.annotate(f"best: {rows[best]['method']} ({rows[best]['cores']} cores)\n{bv:.2f} ms end-to-end, "
            f"{float(rows[0]['e2e_ms']) / bv:.0f}x faster than pure Python", (best, bv), xytext=(best - 2.6, 400),
            textcoords="data", fontsize=7.5, color="#D55E00", ha="center",
            arrowprops=dict(arrowstyle="-|>", color="#D55E00", lw=0.8))
ax.set_yscale("log")
ax.set_ylim(0.2, 20000)
ax.set_xticks(list(x), lab, rotation=25, ha="right", fontsize=6.8)
ax.set_ylabel("time (ms, log)")
ax.set_title("(b) Raspberry Pi 4: first datagram -> predictions of 256 flows (403 Hamming datagrams)")
ax.set_xlim(-0.4, len(rows) - 0.6)
ax.legend(loc="lower left", ncol=4, fontsize=6.8)
fig.tight_layout(h_pad=0.4)
fig.savefig(f"{HERE}/fig_best_method_pi.pdf"); fig.savefig(f"{HERE}/fig_best_method_pi.png", dpi=200)
print("saved fig_best_method_pi.pdf / .png")
