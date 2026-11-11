"""Measure Raspberry Pi 5 power/current over time via the on-board PMIC (no extra hardware).

Reads `vcgencmd pmic_read_adc` every --interval ms, pairs each rail's current (*_A) with its voltage (*_V),
sums power = sum(I*V) across rails, and logs t_seconds, total_W, total_A_on_5V to CSV.

usage (on the central Pi, while the receiver runs):
    # terminal 1:
    ./receiver --idle-ms 60000
    # terminal 2 (start before/with the senders):
    python3 power_monitor.py --interval 200 --out power_usage.csv --duration 40
Then plot:
    python3 plot_ram.py power_usage.csv power.png      # works for any t,value csv (see plot_power below)

Only works on Pi 5 (older Pis have no pmic_read_adc -> use an external USB/inline power meter instead).
"""
import argparse
import re
import subprocess
import time

RE_CUR = re.compile(r"(\w+?)_A\s+current\(\d+\)=([\d.]+)A")
RE_VExpr = re.compile(r"(\w+?)_V\s+volt\(\d+\)=([\d.]+)V")


def read_pmic():
    """Return (total_watts, ext5v_volts, ext5v_amps_estimate). total_watts = sum(I_rail * V_rail)."""
    out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True).stdout
    cur = {m.group(1): float(m.group(2)) for m in RE_CUR.finditer(out)}
    volt = {m.group(1): float(m.group(2)) for m in RE_VExpr.finditer(out)}
    watts = sum(cur[name] * volt.get(name, 0.0) for name in cur)
    ext5v = volt.get("EXT5V", 0.0)
    return watts, ext5v, (watts / ext5v if ext5v else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=200.0, help="sample period in ms")
    ap.add_argument("--out", default="power_usage.csv")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds to sample")
    a = ap.parse_args()

    try:
        read_pmic()
    except FileNotFoundError:
        raise SystemExit("vcgencmd not found - run this on the Raspberry Pi (Pi 5 for pmic_read_adc).")

    rows = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a.duration:
        w, v5, i5 = read_pmic()
        rows.append((time.perf_counter() - t0, w, v5, i5))
        time.sleep(a.interval / 1000.0)

    with open(a.out, "w") as f:
        f.write("t_seconds,power_w,ext5v_v,current_a\n")
        for t, w, v, i in rows:
            f.write(f"{t:.3f},{w:.4f},{v:.4f},{i:.4f}\n")
    ws = [r[1] for r in rows]
    print(f"{len(rows)} samples over {rows[-1][0]:.1f}s -> {a.out}")
    print(f"power: mean {sum(ws)/len(ws):.2f} W, min {min(ws):.2f} W, max {max(ws):.2f} W")


if __name__ == "__main__":
    main()
