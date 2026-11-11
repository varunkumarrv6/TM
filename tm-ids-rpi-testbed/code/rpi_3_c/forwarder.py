"""Laptop forwarder (Python version of forwarder.c, same behaviour and same random positions per seed) for the TM
routed through the laptop (TM 1): receives its datagrams, corrupts them, forwards them to the central RPi.
Change K (and the attack) by hand between runs: stop with Ctrl-C, start again with a new --k.

--target all (default): K random BIT positions anywhere in the datagram -- clear header, rule data, Hamming parity
    bits and the protected header copy -- the same K positions in every packet (the attacker does not know the code).
    Positions are drawn from all 967 x 8 bits of a Hamming datagram (556 x 8 for a raw one).
--target literals: K literal positions (0..355), changed in every real rule (the earlier experiment, runs 6-12).
  include : bit -> 1      exclude : bit -> 0      toggle : flip      none : forward unchanged
Positions: splitmix64 + partial Fisher-Yates from the seed, identical to forwarder.c.
Before the first packet of every transmission a label packet ("TMLABEL attack=.. K=.. seed=.. target=..
literals=..") goes to the central RPi, which stores it with that run.

usage (on the laptop):
  python forwarder.py --attack toggle  --k 4           # 4 random bits anywhere in every packet flipped
  python forwarder.py --attack include --k 8 --seed 3  # another random set of 8 bit positions set to 1
  python forwarder.py --attack none                    # clean pass-through (reference run)
  python forwarder.py --attack include --k 8 --target literals   # literal-only attack of runs 6-12
Addresses: read from net_config.h next to this file (CENTRAL_IP, CENTRAL_PORT, LAPTOP_PORT); the options
--listen-port / --central-ip / --central-port override them.
"""
import argparse
import os
import re
import socket
import time


def net_config(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "net_config.h")):
    """#define NAME value pairs of net_config.h (the same file the C programs use)."""
    out = {}
    if os.path.exists(path):
        for m in re.finditer(r'^\s*#define\s+(\w+)\s+"?([^"\s]+)"?', open(path).read(), re.M):
            out[m.group(1)] = m.group(2)
    return out


CFG = net_config()
LISTEN_IP, LISTEN_PORT = "0.0.0.0", int(CFG.get("LAPTOP_PORT", 5001))          # where the RPi behind the laptop sends
CENTRAL_IP, CENTRAL_PORT = CFG.get("CENTRAL_IP", "10.42.0.145"), int(CFG.get("CENTRAL_PORT", 5001))   # central RPi
LITERALS, HDR, RULE_BYTES, RULES_PER_PACKET = 356, 8, 45, 12
DG = {1: HDR + 959, 0: HDR + 548}                 # layout (1 Hamming, 0 raw) -> datagram size
LAYOUT_OF_SIZE = {v: k for k, v in DG.items()}
NEW_RUN_GAP_S = 2.0                               # a pause this long starts a new transmission (new label)


def literal_bit(rule, lit, layout):
    """(byte index in the datagram, bit mask) of literal `lit` of rule `rule`."""
    B, b = RULE_BYTES * rule + lit // 8, lit % 8          # payload byte / bit, LSB first
    if layout == 0:
        return HDR + B, 1 << b
    cw = 2 * B if b >= 4 else 2 * B + 1                    # high nibble -> first codeword
    enc_bit = 7 * cw + (2, 4, 5, 6)[3 - b % 4]             # D1..D4 inside P1 P2 D1 P4 D2 D3 D4 (MSB first)
    return HDR + enc_bit // 8, 0x80 >> (enc_bit % 8)


def literal_targets(literals, n_real, layout):
    return [literal_bit(r, l, layout) for r in range(n_real) for l in literals]


def bit_targets(bits):
    """Bit positions of the whole datagram (bit 0 = MSB of byte 0) -> (byte, mask)."""
    return [(b // 8, 0x80 >> (b % 8)) for b in bits]


def corrupt(dg, tgt, attack):
    buf = bytearray(dg)
    for byte, mask in tgt:
        if attack == "include":
            buf[byte] |= mask
        elif attack == "exclude":
            buf[byte] &= ~mask & 0xFF
        else:
            buf[byte] ^= mask
    return bytes(buf)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def choose(k, seed, n):
    """K distinct positions of 0..n-1 (sorted): splitmix64 + partial Fisher-Yates, identical to forwarder.c."""
    pool, state = list(range(n)), seed
    for i in range(k):
        state, r = splitmix64(state)
        j = i + r % (n - i)
        pool[i], pool[j] = pool[j], pool[i]
    return sorted(pool[:k])


def choose_literals(k, seed):
    return choose(k, seed, LITERALS)


def positions(target, k, seed, literals=None):
    """{layout: positions}: literal indices (target literals) or datagram bit positions (target all)."""
    if target == "literals":
        lits = literals if literals is not None else choose_literals(k, seed)
        return {0: lits, 1: lits}
    if literals is not None:
        return {0: literals, 1: literals}
    return {lay: choose(k, seed, size * 8) for lay, size in DG.items()}


def label_positions(target, pos):
    if target == "literals":
        return " ".join(map(str, pos[1]))
    return f"hamming:{' '.join(map(str, pos[1]))} raw:{' '.join(map(str, pos[0]))}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", choices=["none", "include", "exclude", "toggle"], default="none")
    ap.add_argument("--k", type=int, default=0, help="number of bits (target all) or literals (target literals)")
    ap.add_argument("--seed", type=int, default=1, help="which random set of K positions")
    ap.add_argument("--target", choices=["all", "literals"], default="all")
    ap.add_argument("--literals", default=None, help="comma-separated explicit positions (overrides --k/--seed)")
    ap.add_argument("--listen-port", type=int, default=LISTEN_PORT)
    ap.add_argument("--central-ip", default=CENTRAL_IP)
    ap.add_argument("--central-port", type=int, default=CENTRAL_PORT)
    a = ap.parse_args()

    explicit = [int(x) for x in a.literals.split(",")] if a.literals else None
    pos = positions(a.target, a.k, a.seed, explicit) if a.attack != "none" else {0: [], 1: []}
    K = len(pos[1])
    label = (f"TMLABEL attack={a.attack} K={K} seed={a.seed if explicit is None else -1} target={a.target} "
             f"literals={label_positions(a.target, pos) if K else ''}").encode()
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    rx.bind((LISTEN_IP, a.listen_port))
    rx.settimeout(0.5)                 # wake up twice a second so Ctrl-C also works on Windows
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (a.central_ip, a.central_port)
    print(f"forwarder: {LISTEN_IP}:{a.listen_port} -> {dest[0]}:{dest[1]}   attack {a.attack}, K = {K}, "
          f"target {a.target}", flush=True)
    if K:
        print(f"  positions (Hamming datagram): {pos[1]}\n  positions (raw datagram):     {pos[0]}", flush=True)
    cache, last, n_run, n_fwd, n_hit = {}, 0.0, 0, 0, 0
    try:
        while True:
            try:
                dg, _ = rx.recvfrom(2048)
            except socket.timeout:
                continue
            now = time.monotonic()
            if now - last > NEW_RUN_GAP_S:                       # first packet of a new transmission
                if n_fwd:
                    print(f"  transmission {n_run}: {n_fwd} datagrams forwarded, {n_hit} changed", flush=True)
                n_run += 1; n_fwd = n_hit = 0
                tx.sendto(label, dest)
            last = now
            lay = LAYOUT_OF_SIZE.get(len(dg))
            if K and lay is not None:
                if a.target == "all":
                    key = ("all", lay)
                    if key not in cache:
                        cache[key] = bit_targets(pos[lay])
                    dg = corrupt(dg, cache[key], a.attack); n_hit += 1
                elif 1 <= dg[1] <= RULES_PER_PACKET:
                    key = ("lit", dg[1], lay)
                    if key not in cache:
                        cache[key] = literal_targets(pos[lay], dg[1], lay)
                    dg = corrupt(dg, cache[key], a.attack); n_hit += 1
            tx.sendto(dg, dest)
            n_fwd += 1
    except KeyboardInterrupt:
        print(f"\nstopped; last transmission: {n_fwd} datagrams forwarded, {n_hit} changed")


if __name__ == "__main__":
    main()
