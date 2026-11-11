"""Plot the RAM-over-time CSV written by the receiver (ram_usage.csv).

usage:  python3 plot_ram.py [ram_usage.csv] [out.png]
Produces a line chart of resident RAM (MB) vs time, marking the peak. If matplotlib is not installed on the Pi,
copy ram_usage.csv to a machine that has it and run this there.
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "ram_usage.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "ram_usage.png"

t, rss = [], []
with open(src) as f:
    for row in csv.DictReader(f):
        t.append(float(row["t_seconds"]))
        rss.append(float(row["rss_mb"]))

if not t:
    sys.exit(f"{src} has no samples")

peak = max(rss)
peak_t = t[rss.index(peak)]

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.color": "#dddddd"})
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, rss, color="#0072bd", lw=1.6)
ax.fill_between(t, rss, color="#0072bd", alpha=0.12)
ax.axhline(peak, color="#d62728", ls="--", lw=1, label=f"peak {peak:.2f} MB")
ax.scatter([peak_t], [peak], color="#d62728", zorder=5)
ax.annotate(f"peak {peak:.2f} MB @ {peak_t:.2f}s", (peak_t, peak),
            textcoords="offset points", xytext=(8, -12), color="#d62728", fontsize=10)
ax.set_xlabel("time since start (s)")
ax.set_ylabel("resident RAM — VmRSS (MB)")
ax.set_title("Central RPi receiver — RAM usage during decode + inference")
ax.set_ylim(0, peak * 1.25)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"{len(t)} samples, peak {peak:.2f} MB at {peak_t:.2f}s  ->  {out}")
