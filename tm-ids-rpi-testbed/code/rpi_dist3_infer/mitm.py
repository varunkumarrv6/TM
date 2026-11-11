"""Laptop man-in-the-middle relay: receives the UDP packets of the TM(s) routed through the laptop, corrupts the
rules as the schedule says for that round, and forwards them to the central node (headers and Hamming parity bits
are never touched, so the receiver cannot tell the packet was changed).

Per round (2-byte round number in front of every packet) the schedule gives: attack (none / include / exclude /
toggle), the attacked TM ids, and K literal positions. The SAME K literals are changed in every real rule of every
packet of an attacked TM (rules 0 .. n_real-1; padding rules stay empty):
  include : literal bit -> 1      exclude : literal bit -> 0      toggle : literal bit flipped
For a Hamming packet the literal is a data bit (D1..D4) of a codeword; one changed bit per codeword is repaired by
the receiver, two or more in the same codeword are not.

usage:
  python3 mitm.py --dest 10.42.0.59 [--dest-port 5001] [--listen-port 5001] [--src 192.168.50.2]
                  [--schedule schedule.csv] [--log mitm_log.csv]
"""
import argparse
import csv
import os
import signal
import socket
import time

PREFIX, HDR, RULE_BYTES, RULES_PER_PACKET = 2, 8, 45, 12
DG = {1: HDR + 959, 0: HDR + 548}                    # layout bit -> datagram size (without the round prefix)
HERE = os.path.dirname(os.path.abspath(__file__))


def literal_bit(rule, lit, layout):
    """(byte index in the UDP packet, bit mask) of literal `lit` of rule `rule`."""
    B, b = RULE_BYTES * rule + lit // 8, lit % 8             # payload byte / bit (LSB first)
    if layout == 0:
        return PREFIX + HDR + B, 1 << b
    cw = 2 * B if b >= 4 else 2 * B + 1                       # high nibble -> first codeword
    enc_bit = 7 * cw + (2, 4, 5, 6)[3 - b % 4]                # D1..D4 inside P1 P2 D1 P4 D2 D3 D4
    return PREFIX + HDR + enc_bit // 8, 0x80 >> (enc_bit % 8)


def targets(literals, n_real, layout):
    return [literal_bit(r, l, layout) for r in range(n_real) for l in literals]


def corrupt(pkt, tgt, attack):
    buf = bytearray(pkt)
    for byte, mask in tgt:
        if attack == "include":
            buf[byte] |= mask
        elif attack == "exclude":
            buf[byte] &= ~mask & 0xFF
        else:
            buf[byte] ^= mask
    return bytes(buf)


def load_schedule(path):
    out = {}
    for row in csv.DictReader(open(path)):
        out[int(row["round"])] = dict(attack=row["attack"], K=int(row["K"]),
                                      nodes={int(x) for x in row["nodes"].split()},
                                      literals=[int(x) for x in row["literals"].split()])
    return out


def on_term(signum, frame):
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="10.42.0.59", help="central node IP")
    ap.add_argument("--dest-port", type=int, default=5001)
    ap.add_argument("--listen-ip", default="0.0.0.0")
    ap.add_argument("--listen-port", type=int, default=5001)
    ap.add_argument("--src", default=None, help="local IP to bind as source of the forwarded packets")
    ap.add_argument("--schedule", default=os.path.join(HERE, "schedule.csv"))
    ap.add_argument("--log", default=os.path.join(HERE, "mitm_log.csv"))
    a = ap.parse_args()

    sched = load_schedule(a.schedule)
    signal.signal(signal.SIGTERM, on_term)                     # kill -> stop like Ctrl-C and save the log
    cache = {}
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    rx.bind((a.listen_ip, a.listen_port))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if a.src:
        tx.bind((a.src, 0))
    print(f"MITM: listen {a.listen_ip}:{a.listen_port} -> forward {a.dest}:{a.dest_port}, {len(sched)} scheduled rounds",
          flush=True)
    stats = {}                                                   # (round, node) -> [forwarded, corrupted, bits]
    last_round = None
    try:
        while True:
            pkt, _ = rx.recvfrom(2048)
            if len(pkt) >= PREFIX + HDR:
                r = pkt[0] | (pkt[1] << 8)
                h7, n_real = pkt[PREFIX + 7], pkt[PREFIX + 1]
                node, layout = h7 >> 4, h7 & 1
                st = stats.setdefault((r, node), [0, 0, 0])
                st[0] += 1
                row = sched.get(r)
                if (row and row["attack"] != "none" and node in row["nodes"] and len(pkt) == PREFIX + DG[layout]
                        and 1 <= n_real <= RULES_PER_PACKET):
                    key = (r, n_real, layout)
                    if key not in cache:
                        cache[key] = targets(row["literals"], n_real, layout)
                    pkt = corrupt(pkt, cache[key], row["attack"])
                    st[1] += 1; st[2] += len(cache[key])
                if r != last_round:
                    if row:
                        print(f"round {r:3d}: {row['attack']:7s} K={row['K']:2d} TMs {sorted(row['nodes'])}", flush=True)
                    last_round = r
            tx.sendto(pkt, (a.dest, a.dest_port))
    except KeyboardInterrupt:
        pass
    finally:
        with open(a.log, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "node", "forwarded", "corrupted", "bits_changed_attempted"])
            for (r, n), v in sorted(stats.items()):
                w.writerow([r, n] + v)
        print(f"\nstopped; per-round counts in {a.log}")


if __name__ == "__main__":
    main()
