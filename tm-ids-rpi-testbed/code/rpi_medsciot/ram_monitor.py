"""Sample a process's RAM over time and log it to CSV (Linux / Raspberry Pi).

It launches the command you give after `--`, samples /proc/<pid>/status every --interval ms until the process
exits, and writes t_seconds, rss_mb, peak_mb to --out. VmRSS is the instantaneous resident RAM; VmHWM is the
kernel's running peak (catches spikes between samples).

examples (run on the central RPi):
    # receiver reading the packet files
    python3 ram_monitor.py --out ram_usage.csv -- ./receiver --files tm0_hamming.bin tm1_hamming.bin tm2_hamming.bin
    # receiver waiting on UDP (start this, then start the senders)
    python3 ram_monitor.py --interval 20 --out ram_usage.csv -- ./receiver
Then plot:
    python3 plot_ram.py ram_usage.csv
"""
import argparse
import subprocess
import sys
import time


def read_rss_kb(pid):
    rss = hwm = 0
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        return None
    return rss, hwm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=20.0, help="sample period in ms (default 20)")
    ap.add_argument("--out", default="ram_usage.csv")
    if "--" not in sys.argv:
        ap.error("put the command to run after `--`, e.g.  ram_monitor.py --out ram.csv -- ./receiver")
    split = sys.argv.index("--")
    a = ap.parse_args(sys.argv[1:split])
    cmd = sys.argv[split + 1:]

    proc = subprocess.Popen(cmd)
    rows = []
    t0 = time.perf_counter()
    while proc.poll() is None:
        r = read_rss_kb(proc.pid)
        if r is not None:
            rows.append((time.perf_counter() - t0, r[0] / 1024.0, r[1] / 1024.0))
        time.sleep(a.interval / 1000.0)
    proc.wait()

    with open(a.out, "w") as f:
        f.write("t_seconds,rss_mb,peak_mb\n")
        for t, rss, hwm in rows:
            f.write(f"{t:.4f},{rss:.4f},{hwm:.4f}\n")
    peak = max((r[2] for r in rows), default=0.0)
    print(f"\n[ram_monitor] {len(rows)} samples over {rows[-1][0] if rows else 0:.2f} s -> {a.out}")
    print(f"[ram_monitor] peak RAM {peak:.2f} MB")


if __name__ == "__main__":
    main()
