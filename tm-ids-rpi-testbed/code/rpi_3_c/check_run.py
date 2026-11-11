"""What the central RPi should print for one run: applies the forwarder's exact byte changes to the datagrams of the
TM routed through the laptop (--node, default 1; runs.csv says which), then runs the receiver's logic in Python
(numpy): format from the datagram size, Hamming decode with the formula syndrome, header taken from the repaired
protected copy (raw: clear header must equal its copy), invalid or duplicate (TM, packet) dropped, class sums of the
50 flows per TM and fused. With --runs it compares every complete row of the receiver's results/runs.csv.

Packets are processed in TM order 0, 2, then the attacked TM. If a corrupted header copy decodes to a (TM, packet)
of another TM, which copy the RPi keeps depends on the arrival order; such cases are reported as "header collisions".

usage: python3 check_run.py --attack toggle --k 4 [--seed 1] [--target all|literals] [--layout hamming|raw]
       python3 check_run.py --runs results/runs.csv
"""
import argparse
import csv
import os
import numpy as np
import forwarder as fw

HERE = os.path.dirname(os.path.abspath(__file__))
HDR, RB, RPP, PAY, C = 8, 45, 12, 548, 5
DG = {"hamming": HDR + 959, "raw": HDR + PAY}
PKTS = [394, 402, 402]
SCALE_Q88 = 7022
X = np.load(f"{HERE}/X_50.npy").astype(np.uint8)
y = np.load(f"{HERE}/y_50_actual.npy")
# 360 bits per rule like receiver.c: the 4 padding bits after the 356 literals count as literals that are always 0,
# so a rule whose padding bit is set (by an attack) can never fire, and is no longer "empty"
NOT_LIT = np.concatenate([1 - np.concatenate([X, 1 - X], 1), np.ones((len(X), 4), np.uint8)], 1).astype(np.float32)


def decode(enc):
    bits = np.unpackbits(enc, axis=1, bitorder="big")[:, : PAY * 14]
    cw = bits.reshape(len(enc), -1, 7).copy()
    s = (cw[..., 0] ^ cw[..., 2] ^ cw[..., 4] ^ cw[..., 6]) | ((cw[..., 1] ^ cw[..., 2] ^ cw[..., 5] ^ cw[..., 6]) << 1) \
        | ((cw[..., 3] ^ cw[..., 4] ^ cw[..., 5] ^ cw[..., 6]) << 2)
    p, k = np.nonzero(s)
    cw[p, k, s[p, k].astype(np.int64) - 1] ^= 1
    nib = (cw[..., 2] << 3) | (cw[..., 4] << 2) | (cw[..., 5] << 1) | cw[..., 6]
    return ((nib[:, 0::2] << 4) | nib[:, 1::2]).astype(np.uint8), len(p)


def receive(streams, layout):
    """streams: list of (P, size) datagram arrays in arrival order -> sums (3, 50, 5), stats."""
    sums = np.zeros((3, len(X), C), np.int64)
    seen, st = set(), dict(corrected=0, dropped=0, clear_damaged=0, collisions=0)
    for owner, dgs in streams:
        if layout == "hamming":
            pay, fixed = decode(dgs[:, HDR:])
            st["corrected"] += fixed
        else:
            pay = dgs[:, HDR:]
        for p in range(len(dgs)):
            copy, clear = pay[p, RPP * RB:], dgs[p, :HDR]
            clear_ok = bool((copy == clear).all())
            st["clear_damaged"] += not clear_ok
            h = copy if layout == "hamming" else clear
            node, idx, scale = int(h[7]) >> 4, int(h[2]) | (int(h[3]) << 8), int(h[5]) | (int(h[6]) << 8)
            cls, n, w = int(h[0]), int(h[1]), int(h[4]) & 0x0F
            ok = ((layout == "hamming") or clear_ok) and node < 3 and (int(h[7]) & 1) == (layout == "hamming") \
                and idx < PKTS[node] and (int(h[4]) & 0x80) and scale == SCALE_Q88 and cls < C and 1 <= n <= RPP \
                and (node, idx) not in seen
            if not ok:
                st["dropped"] += 1
                continue
            if node != owner:
                st["collisions"] += 1
            seen.add((node, idx))
            w = w - 16 if w >= 8 else w
            if w == 0:
                continue
            rules = np.unpackbits(pay[p, : n * RB].reshape(n, RB), axis=1, bitorder="little").astype(np.float32)
            sums[node, :, cls] += w * (((NOT_LIT @ rules.T) == 0) & (rules.sum(1) > 0)[None]).sum(1)
    return sums, st


def predict(attack, pos, layout, target="all", node=1):
    """pos: {0: raw positions, 1: Hamming positions} (forwarder.positions)."""
    lay = 1 if layout == "hamming" else 0
    order = [n for n in range(3) if n != node] + [node]
    streams = []
    for n in order:
        dgs = np.fromfile(f"{HERE}/tm{n}_{layout}.bin", np.uint8).reshape(-1, DG[layout])
        if n == node and attack != "none" and pos[lay]:
            out = []
            for d in dgs:
                t = fw.bit_targets(pos[lay]) if target == "all" else fw.literal_targets(pos[lay], int(d[1]), lay)
                out.append(np.frombuffer(fw.corrupt(d.tobytes(), t, attack), np.uint8))
            dgs = np.stack(out)
        streams.append((n, dgs))
    sums, st = receive(streams, layout)
    res = {}
    for name, s in zip(["tm0", "tm1", "tm2", "fused"], list(sums) + [sums.sum(0)]):
        res[name] = (100 * (s.argmax(1) == y).mean(), s[np.arange(len(y)), y].mean())
    return res, st


def parse_positions(text, target):
    if not text.strip():
        return {0: [], 1: []}
    if target == "all" and "hamming:" in text:
        ham, raw = text.split("raw:")
        return {1: [int(x) for x in ham.replace("hamming:", "").split()], 0: [int(x) for x in raw.split()]}
    v = [int(x) for x in text.split()]
    return {0: v, 1: v}


ap = argparse.ArgumentParser()
ap.add_argument("--attack", default="none")
ap.add_argument("--k", type=int, default=0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--target", default="all", choices=["all", "literals"])
ap.add_argument("--literals", default=None)
ap.add_argument("--layout", default="hamming")
ap.add_argument("--node", type=int, default=1, help="TM routed through the laptop")
ap.add_argument("--runs", default=None)
a = ap.parse_args()

if a.runs:
    bad = 0
    for row in csv.DictReader(open(a.runs)):
        if row.get("complete") != "1":
            continue
        target = row.get("target") or "literals"           # rows written before the 'target' column: literal attack
        res, st = predict(row["attack"] or "none", parse_positions(row["literals"], target), row["layout"], target,
                          int(row.get("attacked_node") or 2))
        an = f"tm{int(row.get('attacked_node') or 2)}"
        got = {n: (float(row[f"acc_{n}"]), float(row[f"true_sum_{n}"])) for n in res}
        ok = all(abs(got[n][0] - res[n][0]) < 0.01 and abs(got[n][1] - res[n][1]) < 0.001 for n in res) \
            and int(row["corrected"]) == st["corrected"]
        bad += not ok
        print(f"run {row['run']:>3} {row['attack']:7s} K={row['K']:>2} {target:8s} {row['layout']:7s} "
              f"fused {got['fused'][0]:6.2f} % (exp. {res['fused'][0]:6.2f})  {an.upper()} {got[an][0]:6.2f} % "
              f"(exp. {res[an][0]:6.2f})  corrected {row['corrected']} (exp. {st['corrected']})  "
              f"dropped {row.get('dropped', '-')} (exp. {st['dropped']})  {'OK' if ok else 'DIFFERENT'}"
              + (f"  [{st['collisions']} header collisions]" if st["collisions"] else ""))
    print("all complete runs match" if not bad else f"{bad} run(s) differ")
else:
    explicit = [int(x) for x in a.literals.split(",")] if a.literals else None
    pos = fw.positions(a.target, a.k, a.seed, explicit) if a.attack != "none" else {0: [], 1: []}
    res, st = predict(a.attack, pos, a.layout, a.target, a.node)
    lay = 1 if a.layout == "hamming" else 0
    print(f"attack {a.attack}, K = {len(pos[lay])}, target {a.target}, layout {a.layout}, positions {pos[lay]}")
    print(f"  corrected codewords {st['corrected']}, dropped packets {st['dropped']}, damaged clear headers "
          f"{st['clear_damaged']}" + (f", header collisions {st['collisions']}" if st["collisions"] else ""))
    for n, (acc, ts) in res.items():
        print(f"  {n:6s} accuracy {acc:6.2f} %   mean true-class sum {ts:8.2f}")
