"""Plot power/current over time from power_monitor.py's CSV.

usage:  python3 plot_power.py [power_usage.csv] [out.png]
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "power_usage.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "power_usage.png"

t, p, i = [], [], []
with open(src) as f:
    for row in csv.DictReader(f):
        t.append(float(row["t_seconds"]))
        p.append(float(row["power_w"]))
        i.append(float(row.get("current_a", 0.0)))

if not t:
    sys.exit(f"{src} has no samples")
mean_p, peak_p = sum(p) / len(p), max(p)

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.color": "#dddddd"})
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(t, p, color="#d62728", lw=1.5, label="total power (W)")
ax.fill_between(t, p, color="#d62728", alpha=0.10)
ax.axhline(mean_p, color="#555555", ls=":", lw=1.2, label=f"mean {mean_p:.2f} W")
ax.axhline(peak_p, color="#d62728", ls="--", lw=1, label=f"peak {peak_p:.2f} W")
ax.set_xlabel("time since start (s)")
ax.set_ylabel("power (W)")
ax.set_title("Central RPi receiver — power during decode + inference")
ax.set_ylim(0, peak_p * 1.25)
ax2 = ax.twinx()                       # current on a second axis
ax2.plot(t, i, color="#0072bd", lw=1.0, alpha=0.6)
ax2.set_ylabel("current on 5V (A)", color="#0072bd")
ax2.set_ylim(0, max(i) * 1.25 if max(i) else 1)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"{len(t)} samples -> {out}  (mean {mean_p:.2f} W, peak {peak_p:.2f} W)")
