"""Runs on each sending RPi: sends its TM's model as UDP datagrams, once per round of the schedule.

Every UDP packet = 2-byte round number (uint16 LE) + one datagram of the stream file (967 B Hamming / 556 B raw).
The schedule says, per round, whether the Hamming or the raw stream is sent (the MITM / receiver read the same file).

usage:
  python3 sender.py --node <n> --dest <central_ip> [--port 5001] [--src <local_ip>] [--schedule schedule.csv]
                    [--gap-us 2000] [--round-gap-ms 600] [--period-ms P] [--rounds 0:97]
examples (testbed of rpi_medsciot):
  python3 sender.py --node 0 --dest 10.42.0.59 --src 10.42.0.1        # RPi 0 -> central directly
  python3 sender.py --node 1 --dest 10.42.0.59 --src 10.42.0.82       # RPi 1 -> central directly
  python3 sender.py --node 2 --dest 10.42.0.59 --src 192.168.50.3     # RPi 2 -> central ip, routed via the laptop
Files needed next to this script: tm<n>_hamming.bin, tm<n>_raw.bin, schedule.csv.
  --gap-us        pause between datagrams (default 2000 us; raise it if the receiver reports missing packets)
  --round-gap-ms  idle time after a round (default 600 ms); round period = 520 x gap + round gap, on a fixed
                  timeline, so the 3 senders keep the offset they were started with (start them within ~1 s)
"""
import argparse
import csv
import os
import socket
import time

HDR, DG = 8, {"hamming": 8 + 959, "raw": 8 + 548}
HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--node", type=int, required=True, help="TM id of this RPi (0, 1, 2)")
ap.add_argument("--dest", required=True, help="central node IP (also for the RPi that is routed via the laptop)")
ap.add_argument("--port", type=int, default=5001)
ap.add_argument("--src", default=None, help="local IP to bind as source (optional)")
ap.add_argument("--schedule", default=os.path.join(HERE, "schedule.csv"))
ap.add_argument("--gap-us", type=float, default=2000.0)
ap.add_argument("--round-gap-ms", type=float, default=600.0)
ap.add_argument("--period-ms", type=float, default=None,
                help="time from one round start to the next (default 520 x gap + round gap); same on all senders")
ap.add_argument("--rounds", default=None, help="first:last round to send (default: the whole schedule)")
a = ap.parse_args()

sched = list(csv.DictReader(open(a.schedule)))
if a.rounds:
    lo, hi = (int(x) for x in a.rounds.split(":"))
    sched = [r for r in sched if lo <= int(r["round"]) <= hi]
streams = {lay: open(os.path.join(HERE, f"tm{a.node}_{lay}.bin"), "rb").read() for lay in DG}
for lay, data in streams.items():
    assert len(data) % DG[lay] == 0, f"tm{a.node}_{lay}.bin is not a whole number of {DG[lay]}-byte datagrams"
    assert data[7] >> 4 == a.node, f"tm{a.node}_{lay}.bin belongs to node {data[7] >> 4}"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
if a.src:
    sock.bind((a.src, 0))
gap = a.gap_us * 1e-6
n_dg = len(streams["hamming"]) // DG["hamming"]
# fixed timeline: round i starts at t_start + i * period, packet k of a round at round start + k * gap, so sleep
# overshoot never accumulates and the 3 senders keep the same offset as when they were started
period = a.period_ms * 1e-3 if a.period_ms else 520 * gap + a.round_gap_ms * 1e-3      # 520 > max packets per TM
assert period > n_dg * gap, "--period-ms must be longer than one round of packets"
print(f"TM {a.node}: {len(sched)} rounds -> {a.dest}:{a.port}  ({n_dg} datagrams per round, gap {a.gap_us:.0f} us, "
      f"round period {period * 1e3:.0f} ms)", flush=True)
t_start = time.perf_counter()
for i, row in enumerate(sched):
    r, lay = int(row["round"]), row["layout"]
    data, size = streams[lay], DG[lay]
    prefix = bytes((r & 0xFF, r >> 8))
    t_round = t_start + i * period
    for k, off in enumerate(range(0, len(data), size)):
        wait = t_round + k * gap - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        sock.sendto(prefix + data[off:off + size], (a.dest, a.port))
    print(f"\rround {r:3d} ({lay:7s}, {row['attack']:7s} K={int(row['K']):2d}) sent in "
          f"{time.perf_counter() - t_round:.2f} s", end="", flush=True)
wait = t_start + len(sched) * period - time.perf_counter()
if wait > 0:
    time.sleep(wait)
print(f"\nTM {a.node}: done, {len(sched)} rounds in {time.perf_counter() - t_start:.1f} s")
