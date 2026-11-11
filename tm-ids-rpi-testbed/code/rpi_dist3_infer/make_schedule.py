"""Round schedule shared by the senders, the laptop MITM and the receiver (copy schedule.csv to every device).

One round = one complete transmission of the 3 models. For every round the schedule says which stream the senders
send (raw or Hamming), which attack the MITM applies (none / include / exclude / toggle), K, the attacked TMs and
the K literal positions (random, distinct, seeded; the SAME K positions in every clause of the attacked TMs).

Default: for each layout (raw, hamming) one clean round, then attacks x K x ITERS.
usage: python make_schedule.py [--iters 2] [--k 1,2,4,6,8,12,16,20] [--nodes 2] [--layouts raw,hamming]
                               [--attacks include,exclude,toggle] [--seed 2026] [--out schedule.csv]
  --nodes  TMs whose packets the MITM corrupts (space-free list, e.g. 2 or 1,2). In the testbed only TM 2 is routed
           through the laptop, so only 2 can really be attacked there; other values are for the simulator.
"""
import argparse
import csv
import os
import numpy as np

LITERALS = 356
ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=2)
ap.add_argument("--k", default="1,2,4,6,8,12,16,20")
ap.add_argument("--nodes", default="2")
ap.add_argument("--layouts", default="raw,hamming")
ap.add_argument("--attacks", default="include,exclude,toggle")
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "schedule.csv"))
a = ap.parse_args()

ks = [int(k) for k in a.k.split(",")]
nodes = " ".join(a.nodes.split(","))
rows, r = [], 0
for layout in a.layouts.split(","):
    rows.append(dict(round=r, layout=layout, attack="none", K=0, nodes="", iteration=0, literals="")); r += 1
    for attack in a.attacks.split(","):
        for K in ks:
            for it in range(1, a.iters + 1):
                rng = np.random.default_rng([a.seed, K, it])          # same positions for raw and hamming
                lits = np.sort(rng.choice(LITERALS, K, replace=False))
                rows.append(dict(round=r, layout=layout, attack=attack, K=K, nodes=nodes, iteration=it,
                                 literals=" ".join(map(str, lits))))
                r += 1
assert r < 65536
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rounds -> {a.out}")
