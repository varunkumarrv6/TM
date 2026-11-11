"""LAPTOP MITM FORWARDER (single script, no compiler needed -- runs on Windows/Linux with Python 3).

Receives the TM packets the RPi (connected to the laptop) sends, corrupts the chosen literals,
and routes the packets on to the CENTRAL node.

Attack is selected by a number:   0 = INCLUDE   1 = EXCLUDE   2 = TOGGLE

run (PowerShell / cmd / bash):
    python forwarder.py 0            # include, default k
    python forwarder.py 1 2          # exclude, 2 literals
    python forwarder.py 2 4          # toggle, 4 literals

Edit the CONFIG block below for your testbed (IPs / ports), then just run it again.
"""
import socket
import sys

# ----------------------- CONFIG (edit for your testbed) -----------------------
LISTEN_IP    = "0.0.0.0"        # laptop interface facing the RPi ("0.0.0.0" = all)
LISTEN_PORT  = 5001             # port the RPi sends to on the laptop
CENTRAL_IP   = "10.42.0.59"     # DEST: the central node (routes back here)
CENTRAL_PORT = 5001             # port the central receiver listens on
SRC_IP       = ""               # SRC of forwarded packets ("" = OS default; or laptop's 10.42.0.x addr)
DEFAULT_K    = 2                # literals attacked if k not given on the command line
# -----------------------------------------------------------------------------

HEADER, RULE_BYTES, RULES_PER_PACKET, DG_ENC = 4, 45, 12, 956
# most-used literals in the trained clause banks (0..355), most-used first
TOP_LITERALS = [323, 327, 326, 294, 322, 328, 349, 185, 324, 353,
                293, 321, 292, 351, 348, 350, 330, 184, 186, 291]


def literal_bit(rule, lit):
    """Datagram byte index and bit mask of literal `lit` of rule `rule` (Hamming data bit; parity untouched)."""
    B, b = RULE_BYTES * rule + lit // 8, lit % 8
    cw = 2 * B if b >= 4 else 2 * B + 1            # high nibble -> 1st codeword, low -> 2nd
    enc_bit = 7 * cw + (2, 4, 5, 6)[3 - b % 4]     # D1..D4 positions in the 7-bit codeword
    return HEADER + enc_bit // 8, 0x80 >> (enc_bit % 8)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("0", "1", "2"):
        print("usage: python forwarder.py <attack 0=include 1=exclude 2=toggle> [k]")
        return
    attack = int(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_K
    name = ("INCLUDE", "EXCLUDE", "TOGGLE")[attack]

    targets = [literal_bit(r, l) for r in range(RULES_PER_PACKET) for l in TOP_LITERALS[:k]]

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

    print(f"MITM forwarder: {name} k={k}  ({len(targets)} bit positions per datagram)")
    print(f"listen {LISTEN_IP}:{LISTEN_PORT}  ->  forward {CENTRAL_IP}:{CENTRAL_PORT}")
    print("waiting for packets from the RPi...  (Ctrl+C to stop)")
    rx.settimeout(2.0)
    n_fwd = n_pass = 0
    first = True
    try:
        while True:
            try:
                dg, addr = rx.recvfrom(2048)
            except socket.timeout:
                if n_fwd + n_pass == 0:
                    print("\r[still waiting] 0 packets received -- check RPi --dest, firewall, route", end="", flush=True)
                continue
            if first:
                print(f"\nfirst packet: {len(dg)} bytes from {addr[0]}:{addr[1]}")
                first = False
            if len(dg) == DG_ENC:
                buf = bytearray(dg)
                for byte, mask in targets:
                    if attack == 0:
                        buf[byte] |= mask            # include
                    elif attack == 1:
                        buf[byte] &= 0xFF ^ mask     # exclude
                    else:
                        buf[byte] ^= mask            # toggle
                tx.sendto(bytes(buf), (CENTRAL_IP, CENTRAL_PORT))
                n_fwd += 1
            else:
                tx.sendto(dg, (CENTRAL_IP, CENTRAL_PORT))     # wrong size -> pass through
                n_pass += 1
            if (n_fwd + n_pass) % 50 == 0:
                print(f"\rforwarded {n_fwd} corrupted, {n_pass} passthrough", end="", flush=True)
    except KeyboardInterrupt:
        print(f"\nstopped: {n_fwd} corrupted + {n_pass} passthrough datagrams forwarded")


if __name__ == "__main__":
    main()
