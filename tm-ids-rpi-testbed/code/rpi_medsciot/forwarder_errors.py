"""LAPTOP MITM FORWARDER -- random Hamming-codeword error injection (Python 3, no compiler).

Each run picks a RANDOM number of RANDOM codewords in the rule area and flips a chosen number of RANDOM bits
INSIDE each one (parity P1 P2 P4 and data D1 D2 D3 D4 are all possible). The SAME codewords/bits are hit in every
datagram, so every clause of that TM is corrupted identically. A new random choice is drawn per run (--seed).

  --bits-per-cw 1  -> 1 bit per codeword  -> Hamming(7,4) REPAIRS it  -> no drop (PASS)
  --bits-per-cw 2  -> 2 bits per codeword -> UNCORRECTABLE            -> accuracy drops   (default)

Attack op applied to each selected bit:  0 = INCLUDE (force 1)   1 = EXCLUDE (force 0)   2 = TOGGLE (flip)

run:
    python forwarder_errors.py 2                          # toggle, random #codewords, 2 bits each (uncorrectable)
    python forwarder_errors.py 0 --cwmin 1 --cwmax 20     # include, 1..20 corrupted codewords
    python forwarder_errors.py 2 --bits-per-cw 1          # 1 bit each -> Hamming repairs (sanity check, PASS)
    python forwarder_errors.py 1 --seed 42                # reproducible

Prints SEED, #codewords, bits-per-cw, parity/data split. Replay on the full test set here:
    python3 map_to_fulltest.py --attack toggle --seed 42 --cwmin .. --cwmax .. --bits-per-cw ..
"""
import argparse
import random
import socket

# ----------------------- CONFIG (edit for your testbed) -----------------------
LISTEN_IP    = "0.0.0.0"
LISTEN_PORT  = 5001
CENTRAL_IP   = "10.42.0.59"
CENTRAL_PORT = 5001
SRC_IP       = ""
# -----------------------------------------------------------------------------

HEADER, DG_ENC = 4, 956
RULE_CODEWORDS = 1080                 # codewords of the 540-byte rule area (header copy left intact)
PARITY_POS = {0, 1, 3}                # P1 P2 P4 inside the 7-bit codeword ; data = {2,4,5,6}


def gen_positions(seed, cwmin, cwmax, bits_per_cw):
    """Encoded-bit positions flipped in every datagram. Same arguments -> identical attack (forwarder & mapper)."""
    r = random.Random(seed)
    n_cw = r.randint(cwmin, cwmax)
    codewords = sorted(r.sample(range(RULE_CODEWORDS), n_cw))
    positions = []
    for c in codewords:
        for b in sorted(r.sample(range(7), bits_per_cw)):     # which bits inside this codeword
            positions.append(c * 7 + b)
    return n_cw, sorted(positions)


def split(positions):
    parity = sum(1 for p in positions if p % 7 in PARITY_POS)
    return parity, len(positions) - parity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attack", type=int, choices=[0, 1, 2], help="0=include(set1) 1=exclude(set0) 2=toggle(flip)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cwmin", type=int, default=1, help="min codewords corrupted per run")
    ap.add_argument("--cwmax", type=int, default=10, help="max codewords corrupted per run")
    ap.add_argument("--bits-per-cw", type=int, default=2, choices=range(1, 8),
                    help="bits flipped inside each chosen codeword (>=2 is uncorrectable)")
    a = ap.parse_args()

    seed = a.seed if a.seed is not None else random.randrange(1_000_000)
    n_cw, positions = gen_positions(seed, a.cwmin, a.cwmax, a.bits_per_cw)
    parity, data = split(positions)
    uncor = n_cw if a.bits_per_cw >= 2 else 0
    name = ("INCLUDE", "EXCLUDE", "TOGGLE")[a.attack]
    hits = [(HEADER + p // 8, 0x80 >> (p % 8)) for p in positions]

    print("=" * 74)
    print(f"MITM codeword error forwarder: {name}")
    print(f"SEED={seed}  CODEWORDS={n_cw}  BITS_PER_CW={a.bits_per_cw}  "
          f"TOTAL_BITS={len(positions)} (parity {parity}, data {data})")
    print(f"UNCORRECTABLE_CODEWORDS={uncor}  "
          f"({'accuracy will drop' if uncor else 'all single-bit -> Hamming repairs, PASS'})")
    print(f"map to full dataset:  python3 map_to_fulltest.py --attack {name.lower()} --seed {seed} "
          f"--cwmin {a.cwmin} --cwmax {a.cwmax} --bits-per-cw {a.bits_per_cw}")
    print("=" * 74)

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    except OSError:
        pass
    rx.bind((LISTEN_IP, LISTEN_PORT))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if SRC_IP:
        tx.bind((SRC_IP, 0))
    rx.settimeout(2.0)

    print("waiting for packets from the RPi...  (Ctrl+C to stop)")
    n_fwd = n_pass = 0
    first = True
    try:
        while True:
            try:
                dg, addr = rx.recvfrom(2048)
            except socket.timeout:
                if n_fwd + n_pass == 0:
                    print("\r[still waiting] 0 packets -- check RPi --dest, firewall, route", end="", flush=True)
                continue
            if first:
                print(f"\nfirst packet: {len(dg)} bytes from {addr[0]}:{addr[1]}")
                first = False
            if len(dg) == DG_ENC:
                buf = bytearray(dg)
                for byte, mask in hits:
                    if a.attack == 0:
                        buf[byte] |= mask
                    elif a.attack == 1:
                        buf[byte] &= 0xFF ^ mask
                    else:
                        buf[byte] ^= mask
                tx.sendto(bytes(buf), (CENTRAL_IP, CENTRAL_PORT))
                n_fwd += 1
            else:
                tx.sendto(dg, (CENTRAL_IP, CENTRAL_PORT))
                n_pass += 1
            if (n_fwd + n_pass) % 50 == 0:
                print(f"\rforwarded {n_fwd} corrupted, {n_pass} passthrough", end="", flush=True)
    except KeyboardInterrupt:
        print(f"\nstopped: {n_fwd} corrupted + {n_pass} passthrough datagrams forwarded")


if __name__ == "__main__":
    main()
