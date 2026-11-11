"""Laptop MITM relay -- INCLUDE attack.

Place on the laptop (192.168.50.2). The RPi behind the laptop (e.g. 192.168.50.3) sends its Hamming-encoded
TM packets believing they go to the central node 10.42.0.59; routing delivers them here. This program:
  1. receives each 956-byte UDP datagram on --listen-port,
  2. forces to INCLUDE (set to 1) the target literal(s) in ALL 12 rules of the packet (header + Hamming parity left untouched),
  3. forwards the (corrupted) datagram to the central node, which Hamming-decodes and corrects it.

usage:
  python3 mitm_include.py --dest 10.42.0.59 --dest-port 5001 --listen-port 5001 \
                  --src 192.168.50.2 --k 2
  --k         number of target literals (most-used first); or --literals 323,327,...
  --listen-ip 0.0.0.0 by default (listen on all laptop interfaces)
  --src       optional local IP to bind as the SOURCE of the forwarded packets (so the central node
              sees them coming from the laptop/gateway, as if from the RPi)

The attack hits the SAME literals in every clause, matching the medsciot A2/A3/A4 experiments. Run with a
different --k or --literals each time to vary the corruption.
"""
import argparse
import socket

# ---- packet geometry (Hamming .bin format, SEQ layout) -------------------------------------------
HEADER = 4
RULE_BYTES = 45
RULES_PER_PACKET = 12
PAYLOAD = 544
DG_ENC = 956
# most-used literals in the trained clause banks (model-blind: precomputed indices, 0..355)
TOP_LITERALS = [323, 327, 326, 294, 322, 328, 349, 185, 324, 353, 293, 321, 292, 351, 348, 350, 330, 184, 186, 291]


def literal_bit(rule, lit):
    """Datagram byte index and bit mask of literal `lit` of rule `rule` (Hamming data bit; parity untouched)."""
    B, b = RULE_BYTES * rule + lit // 8, lit % 8
    cw = 2 * B if b >= 4 else 2 * B + 1            # high nibble -> first codeword, low nibble -> second
    enc_bit = 7 * cw + (2, 4, 5, 6)[3 - b % 4]     # D1..D4 positions inside the 7-bit codeword
    return HEADER + enc_bit // 8, 0x80 >> (enc_bit % 8)


def build_targets(literals):
    """(byte, mask) for every (rule, literal) pair -> what to change in each datagram."""
    out = []
    for r in range(RULES_PER_PACKET):
        for l in literals:
            out.append(literal_bit(r, l))
    return out


def corrupt(dg, targets):
    buf = bytearray(dg)
    for byte, mask in targets:
        buf[byte] |= mask      # INCLUDE
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="10.42.0.59", help="central node IP")
    ap.add_argument("--dest-port", type=int, default=5001)
    ap.add_argument("--listen-ip", default="0.0.0.0")
    ap.add_argument("--listen-port", type=int, default=5001)
    ap.add_argument("--src", default=None, help="local IP to bind as source of forwarded packets")
    ap.add_argument("--k", type=int, default=2, help="number of target literals (most-used first)")
    ap.add_argument("--literals", default=None, help="comma-separated literal indices (overrides --k)")
    a = ap.parse_args()

    literals = [int(x) for x in a.literals.split(",")] if a.literals else TOP_LITERALS[:a.k]
    targets = build_targets(literals)

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    rx.bind((a.listen_ip, a.listen_port))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if a.src:
        tx.bind((a.src, 0))

    ATTACK = "include"
    print(f"MITM {ATTACK}: listen {a.listen_ip}:{a.listen_port} -> forward {a.dest}:{a.dest_port}")
    print(f"target literals: {literals}  ({len(targets)} bit positions changed per datagram)")
    n_fwd = n_other = 0
    try:
        while True:
            dg, _ = rx.recvfrom(2048)
            if len(dg) == DG_ENC:
                tx.sendto(corrupt(dg, targets), (a.dest, a.dest_port))
                n_fwd += 1
            else:
                tx.sendto(dg, (a.dest, a.dest_port))        # wrong size -> pass through untouched
                n_other += 1
            if (n_fwd + n_other) % 200 == 0:
                print(f"forwarded {n_fwd} corrupted, {n_other} passthrough", flush=True)
    except KeyboardInterrupt:
        print(f"\nstopped: {n_fwd} corrupted + {n_other} passthrough datagrams forwarded")


if __name__ == "__main__":
    main()
