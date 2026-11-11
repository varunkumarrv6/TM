"""Offline twin of the testbed: for every round of the schedule, applies exactly the byte changes of mitm.py to the
attacked TMs' datagrams, Hamming-decodes them in Python (numpy), rebuilds the rules and computes the class sums of the
50 flows per TM and fused. Writes the same rounds.csv / class_sums.csv columns as receiver.c (timing columns empty),
so compare.py can check the Raspberry Pi results bit for bit, and plot_results.py can plot either.

usage: python simulate.py [--schedule schedule.csv] [--out expected] [--nodes 0,1,2]
  --nodes  override the attacked TMs of every attacked round (e.g. 0,1,2 to see what happens when all three are
           attacked, which the testbed topology cannot do); default: as in the schedule
"""
import argparse
import csv
import os
import numpy as np
import mitm

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX, HDR, RB, RPP, PAY = 2, 8, 45, 12, 548
DG = {"hamming": HDR + 959, "raw": HDR + PAY}
N_NODES, C = 3, 5

ap = argparse.ArgumentParser()
ap.add_argument("--schedule", default=os.path.join(HERE, "schedule.csv"))
ap.add_argument("--out", default=os.path.join(HERE, "expected"))
ap.add_argument("--nodes", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

X = np.load(f"{HERE}/X_50.npy").astype(np.uint8)
y_act, y_ref = np.load(f"{HERE}/y_50_actual.npy"), np.load(f"{HERE}/y_50_ref.npy")
sums_ref = np.load(f"{HERE}/sums_ref.npy")
NS = len(X)
NOT_LIT = (1 - np.concatenate([X, 1 - X], 1)).astype(np.float32)          # (50, 356): literal is 0
streams = {(n, lay): np.fromfile(f"{HERE}/tm{n}_{lay}.bin", np.uint8).reshape(-1, DG[lay])
           for n in range(N_NODES) for lay in DG}


def hamming_decode(enc):
    """(P, 959) encoded payloads -> (P, 548) bytes, number of corrected codewords (formula syndrome)."""
    bits = np.unpackbits(enc, axis=1, bitorder="big")[:, : PAY * 2 * 7]
    cw = bits.reshape(len(enc), -1, 7).copy()
    s = (cw[..., 0] ^ cw[..., 2] ^ cw[..., 4] ^ cw[..., 6]) | ((cw[..., 1] ^ cw[..., 2] ^ cw[..., 5] ^ cw[..., 6]) << 1) \
        | ((cw[..., 3] ^ cw[..., 4] ^ cw[..., 5] ^ cw[..., 6]) << 2)
    p, k = np.nonzero(s)
    cw[p, k, s[p, k].astype(np.int64) - 1] ^= 1
    nib = (cw[..., 2] << 3) | (cw[..., 4] << 2) | (cw[..., 5] << 1) | cw[..., 6]
    return ((nib[:, 0::2] << 4) | nib[:, 1::2]).astype(np.uint8), len(p)


def node_sums(dgs, layout):
    """Class sums (50, 5) of one TM from its (possibly corrupted) datagrams (without the round prefix)."""
    hdr = dgs[:, :HDR]
    pay, fixed = hamming_decode(dgs[:, HDR:]) if layout == "hamming" else (dgs[:, HDR:].copy(), 0)
    ok = (pay[:, RPP * RB:] == hdr).all(1)
    out = np.zeros((NS, C), np.int64)
    for p in np.flatnonzero(ok):
        cls, n, w4 = int(hdr[p, 0]), int(hdr[p, 1]), int(hdr[p, 4]) & 0x0F
        w4 = w4 - 16 if w4 >= 8 else w4                                     # signed 4-bit weight
        if w4 == 0 or not 1 <= n <= RPP or cls >= C:
            continue
        rules = np.unpackbits(pay[p, : n * RB].reshape(n, RB), axis=1, bitorder="little")[:, :356].astype(np.float32)
        fired = ((NOT_LIT @ rules.T) == 0) & (rules.sum(1) > 0)[None]
        out[:, cls] += w4 * fired.sum(1)
    return out, fixed, int((~ok).sum())


sched = list(csv.DictReader(open(a.schedule)))
fr = open(f"{a.out}/rounds.csv", "w", newline="")
fs = open(f"{a.out}/class_sums.csv", "w", newline="")
wr, ws = csv.writer(fr), csv.writer(fs)
wr.writerow(["round", "layout", "attack", "K", "nodes", "iteration", "literals", "pkts_tm0", "pkts_tm1", "pkts_tm2",
             "complete", "corrected", "bad_headers", "acc_pct", "correct", "agree_ref", "clean_sums_exact", "net_ms",
             "decode_ms", "ecc_ms", "infer_ms", "tail_ms", "e2e_ms"])
ws.writerow(["round", "flow", "actual", "ref_pred", "pred", "source", "c0", "c1", "c2", "c3", "c4"])
for row in sched:
    r, lay, attack = int(row["round"]), row["layout"], row["attack"]
    nodes = {int(x) for x in (a.nodes.split(",") if a.nodes and attack != "none" else row["nodes"].split())}
    lits = [int(x) for x in row["literals"].split()]
    sums, fixed, bad = [], 0, 0
    for n in range(N_NODES):
        dgs = streams[(n, lay)]
        if attack != "none" and n in nodes:
            pk = np.concatenate([np.zeros((len(dgs), PREFIX), np.uint8), dgs], 1)     # same offsets as on the wire
            out = []
            for d in pk:
                t = mitm.targets(lits, int(d[PREFIX + 1]), d[PREFIX + 7] & 1)
                out.append(np.frombuffer(mitm.corrupt(d.tobytes(), t, attack), np.uint8))
            dgs = np.stack(out)[:, PREFIX:]
        s, f, b = node_sums(dgs, lay)
        sums.append(s); fixed += f; bad += b
    sums = np.stack(sums)
    fused = sums.sum(0)
    pred = fused.argmax(1)
    correct, agree = int((pred == y_act).sum()), int((pred == y_ref).sum())
    exact = int(np.array_equal(sums, sums_ref))
    wr.writerow([r, lay, attack, row["K"], " ".join(map(str, sorted(nodes))) if attack != "none" else "",
                 row["iteration"], row["literals"]] + [len(streams[(n, lay)]) for n in range(N_NODES)] +
                [1, fixed, bad, f"{100 * correct / NS:.4f}", correct, agree, exact, "", "", "", "", "", ""])
    for s in range(NS):
        for n, name in enumerate(["tm0", "tm1", "tm2", "fused"]):
            v = sums[n, s] if n < N_NODES else fused[s]
            ws.writerow([r, s, y_act[s], y_ref[s], pred[s], name] + v.tolist())
    print(f"round {r:3d} {lay:7s} {attack:7s} K={int(row['K']):2d} TMs {sorted(nodes) if attack != 'none' else '-'}  "
          f"corrected {fixed:6d}  acc {100 * correct / NS:6.2f} %  agree {agree}/{NS}" + ("  exact" if exact else ""))
fr.close(); fs.close()
print("->", a.out)
