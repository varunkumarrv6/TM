"""4 attacks (A1 random, A2 include, A3 exclude, A4 toggle) x K x {No ECC, Hamming SEQ, INT, KEY} on the single TM
of one dataset. A1: K random literals per rule (seed 0). A2-A4: the K most-used literals (most included in the clean TM).

First validates the fast decoder against a bit-level packet reference (Hamming encode -> flip -> decode).
usage: python ecc_attacks.py {medsec|iot|mnist}   -> figures/<ds>/ecc_attacks/results_4attacks_<ds>.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import ecc_common as E

DS = sys.argv[1]
OUT = f"{E.ROOT}/{DS}/ecc_attacks"
os.makedirs(OUT, exist_ok=True)
K_VALUES = [1, 2, 4, 8, 12, 16, 20]
A1_SEED = 0

m, link, eng, pred0 = E.setup(DS)
y, N = m["yte"], len(m["yte"])
BENCH_CORRECT = int((pred0 == y).sum())
BENCH = BENCH_CORRECT / N * 100
use = m["cube"].sum(axis=(0, 1)).astype(np.int64)
TARGETS = [int(i) for i in np.argsort(-use, kind="stable")[:max(K_VALUES)]]
print(f"[{DS}] {E.INFO[DS]}\nbenchmark {BENCH:.4f} % ({BENCH_CORRECT} / {N}); packets {link.npk}, rule bits {link.RB}",
      flush=True)

# ---- validation against the bit-level reference
rng = np.random.default_rng(5)
pk = np.r_[0, 1, rng.choice(link.npk, 6, replace=False), link.npk - 1]
rows_pk = np.concatenate([np.arange(s * 12, (s + 1) * 12) for s in pk])
for atk, kw in [("A1", dict(k=6, rng=np.random.default_rng(1))), ("A2", dict(lits=TARGETS[:8])),
                ("A3", dict(lits=TARGETS[:8])), ("A4", dict(lits=rng.choice(link.L, 12, replace=False)))]:
    flips = link.attack(atk, **kw)
    for lay in E.LAYOUTS:
        fast, _ = link.receive(lay, flips)
        assert np.array_equal(fast[rows_pk], link.reference_receive(lay, flips, pk)), (atk, lay)
print("fast decoder == bit-level Hamming packet reference: OK", flush=True)

rows = []
t0 = time.time()
for atk in E.ATTACKS:
    for k in K_VALUES:
        if atk == "A1":
            flips = link.attack("A1", k=k, rng=np.random.default_rng([A1_SEED, k]))
        else:
            flips = link.attack(atk, lits=TARGETS[:k])
        for lay in E.LAYOUTS:
            rx, hist = link.receive(lay, flips)
            cube = link.to_cube(rx)
            pred = eng.sums(cube).argmax(1)
            correct = int((pred == y).sum())
            rows.append({"attack": atk, "K": k, "layout": lay,
                         "targeted literals": "" if atk == "A1" else " | ".join(m["lit_name"][TARGETS[:k]]),
                         "bits changed by MITM": len(flips),
                         "cw 1 bit": None if hist is None else int(hist[0]),
                         "cw 2 bits": None if hist is None else int(hist[1]),
                         "cw 3 bits": None if hist is None else int(hist[2]),
                         "cw 4 bits": None if hist is None else int(hist[3]),
                         "wrong bits after decode": int((cube != m["cube"]).sum()),
                         "clauses changed": int((cube != m["cube"]).any(2).sum()),
                         "correct": correct, "N": N, "accuracy %": correct / N * 100,
                         "benchmark correct": BENCH_CORRECT, "benchmark %": BENCH,
                         "Δ vs benchmark": correct / N * 100 - BENCH,
                         "samples changed %": (pred != pred0).mean() * 100})
    print(f"{atk} done ({time.time() - t0:.0f} s)", flush=True)

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/results_4attacks_{DS}.csv", index=False, float_format="%.10f")
pd.DataFrame({"rank": np.arange(1, len(TARGETS) + 1), "literal index (0-based)": TARGETS,
              "literal": m["lit_name"][TARGETS], "clauses including it": use[TARGETS]}).to_csv(
    f"{OUT}/targeted_literals_{DS}.csv", index=False)
print(R.pivot_table(index=["attack", "K"], columns="layout", values="accuracy %", sort=False)
      [E.LAYOUTS].round(4).to_string())
