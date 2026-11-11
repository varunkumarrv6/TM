"""Accuracy vs K (literals with errors per clause), 3-node weighted TM on MedSec-25. Mean over iterations, min-max band."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = "/home/<user>/medsciot"
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.color": "#e8e8e8", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False})
df = pd.read_csv(f"{BASE}/inject_K_results.csv")
BENCH = df[df.K == 0]["accuracy"].iloc[0]
NIT = df.iteration.max()
STYLE = {"toggle": ("#2a78d6", "o", "toggle (bit flip)"),
         "include": ("#eb6834", "s", "include (set to 1)"),
         "exclude": ("#1baf7a", "D", "exclude (set to 0)")}

fig, ax = plt.subplots(figsize=(11, 6.5))
for a, (col, mk, lab) in STYLE.items():
    g = df[(df.attack == a) & (df.K > 0)].groupby("K")["accuracy"].agg(["mean", "min", "max"])
    ax.fill_between(g.index, g["min"], g["max"], color=col, alpha=0.12, lw=0)
    ax.plot(g.index, g["mean"], color=col, marker=mk, ms=8, lw=2, mec="white", mew=1.2, label=lab)
    if a == "toggle":
        for k, m in g["mean"].items():
            ax.annotate(f"{m:.1f}", (k, m), textcoords="offset points", xytext=(8, -14), fontsize=8.5, color="#333")
ax.axhline(BENCH, color="#222", ls="--", lw=1.3, label=f"benchmark, no errors ({BENCH:.2f}%)")
ax.set_xlabel("K  —  literals with errors per clause (same K for every clause, positions random per clause)")
ax.set_ylabel("test accuracy (%)")
ax.set_title(f"MedSec-25, 3-node weighted TM (errors in all nodes, raw fused sums): accuracy vs K\n"
             f"mean of {NIT} iterations, band = min–max", loc="left")
ax.set_xticks(sorted(df[df.K > 0].K.unique())); ax.set_ylim(0, 100)
ax.legend(loc="center right", frameon=True)
fig.tight_layout()
fig.savefig(f"{BASE}/figures/fig_accuracy_vs_K.png", dpi=150)

s = df.groupby(["K", "attack"])["accuracy"].agg(["mean", "std"]).round(3).unstack("attack")
s.to_csv(f"{BASE}/summary_accuracy_vs_K.csv")
print(s.to_string())
