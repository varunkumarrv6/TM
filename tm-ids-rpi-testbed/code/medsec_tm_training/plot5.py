"""Accuracy vs K, 5-TM distributed system, errors in TM 0 only, no coding."""
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = "/home/<user>/new_medsec"
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.color": "#e8e8e8", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False})
df = pd.read_csv(f"{BASE}/inject5_results.csv")
BENCH = df[df.K == 0]["accuracy"].iloc[0]
STYLE = {"include": ("#eb6834", "s", "include (set to 1)"),
         "exclude": ("#1baf7a", "D", "exclude (set to 0)"),
         "toggle": ("#2a78d6", "o", "toggle (bit flip)")}

fig, ax = plt.subplots(figsize=(11, 6.5))
for a, (col, mk, lab) in STYLE.items():
    g = df[(df.attack == a) & (df.K > 0)].sort_values("K")
    ax.plot(g.K, g.accuracy, color=col, marker=mk, ms=8, lw=2, mec="white", mew=1.2, label=lab)
ax.axhline(BENCH, color="#222", ls="--", lw=1.3, label=f"benchmark, no errors ({BENCH:.2f}%)")
ax.set_xlabel("K  —  literals with errors per clause in TM 0 (positions random per clause)")
ax.set_ylabel("fused test accuracy (%)")
ax.set_title("MedSec-25, 5 distributed TMs, no coding: errors injected into TM 0 only", loc="left")
ax.set_xticks(sorted(df[df.K > 0].K.unique())); ax.set_ylim(0, 100)
ax.legend(loc="lower left", frameon=True)
lo, hi = df[df.K > 0].accuracy.min(), df[df.K > 0].accuracy.max()
ax.text(0.99, 0.80, f"all 3 attacks, all K: {lo:.2f}% – {hi:.2f}%\n(lines overlap; TMs 1–4 outvote the corrupted TM 0)",
        transform=ax.transAxes, ha="right", va="top", fontsize=11, color="#333")
fig.tight_layout()
fig.savefig(f"{BASE}/fig5tm_tm0_accuracy_vs_K.png", dpi=150)
print("saved")
