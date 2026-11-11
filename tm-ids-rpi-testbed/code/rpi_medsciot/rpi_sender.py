"""Runs on each sending RPi: streams one TM's packet file as UDP datagrams to a destination.

usage:
  python3 rpi_sender.py <file> --dest <ip> [--port 5001] [--src <local_ip>] [--raw] [--gap-us 200]

examples (this testbed):
  # RPi 10.42.0.1  -> central directly
  python3 rpi_sender.py tm0_hamming.bin --dest 10.42.0.59 --port 5001 --src 10.42.0.1
  # RPi 10.42.0.82 -> central directly
  python3 rpi_sender.py tm1_hamming.bin --dest 10.42.0.59 --port 5001 --src 10.42.0.82
  # RPi 192.168.50.3 (behind the laptop) -> it believes it sends to the central node, but the
  # laptop (192.168.50.2) is the gateway/MITM and forwards to 10.42.0.59 after modifying the packets
  python3 rpi_sender.py tm2_hamming.bin --dest 10.42.0.59 --port 5001 --src 192.168.50.3

Notes:
  --dest   where this RPi sends. The RPi behind the laptop still uses the CENTRAL ip (10.42.0.59);
           routing sends it through the laptop, which is why the laptop can intercept it.
  --src    optional: bind the socket to this local address (sets the source IP of the packets).
  --port   UDP port on the destination (default 5001).
  --raw    the file holds 548-byte datagrams (no ECC) instead of 956-byte Hamming ones.
  --gap-us microseconds between datagrams, so the receiver/laptop is not flooded (default 200).
"""
import argparse
import socket
import time

DG_ENC, DG_RAW = 956, 548

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--dest", required=True, help="destination IP (central node, even when routed via the laptop)")
ap.add_argument("--port", type=int, default=5001)
ap.add_argument("--src", default=None, help="local IP to bind as source (optional)")
ap.add_argument("--raw", action="store_true")
ap.add_argument("--gap-us", type=float, default=2000.0,
                help="microseconds between datagrams (default 2000 = 2 ms; raise it if packets are lost)")
a = ap.parse_args()

size = DG_RAW if a.raw else DG_ENC
data = open(a.file, "rb").read()
assert len(data) % size == 0, f"{a.file}: {len(data)} bytes is not a whole number of {size}-byte datagrams"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
if a.src:
    sock.bind((a.src, 0))                       # fix the source IP of the outgoing packets
gap = a.gap_us * 1e-6

total = len(data) // size
t0 = time.perf_counter()
n = 0
for off in range(0, len(data), size):
    sock.sendto(data[off:off + size], (a.dest, a.port))
    n += 1
    if n % 50 == 0 or n == total:
        print(f"\rsent {n} / {total} datagrams to {a.dest}:{a.port}", end="", flush=True)
    if gap:
        time.sleep(gap)
src = a.src or "default"
print(f"\nsent {n} datagrams of {size} B from {a.file} (src {src}) to {a.dest}:{a.port} in "
      f"{time.perf_counter()-t0:.3f} s  (gap {a.gap_us:.0f} us)")
