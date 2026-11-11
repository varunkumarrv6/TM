"""Monte Carlo, 150 iterations: in every iteration and for every K, K random literals are drawn; the SAME literals are
included (A2) / excluded (A3) / toggled (A4) in every rule of every packet of the single TM, sent with No ECC / Hamming
SEQ / INT / KEY. A new random set is drawn in the next iteration (seed 10_000 + iteration, as medsciot/mc_attacks.py).

usage: python mc_attacks.py {medsec|iot|mnist} START STOP OUTCSV   (iterations START..STOP-1)
       import mc_attacks; mc_attacks.init(ds); mc_attacks.run_set(it, K, lits)   (used by mc_force_shared.py)
"""
import sys
import time
import numpy as np
import pandas as pd
import ecc_common as E

K_LIST = [2, 4, 6, 8, 12, 16, 20]
ATTACKS = ["A2", "A3", "A4"]


def init(ds):
    global DS, m, link, eng, pred0, y, N, BENCH_CORRECT, BENCH, L, cw_of
    DS = ds
    m, link, eng, pred0 = E.setup(DS)
    y, N = m["yte"], len(m["yte"])
    BENCH_CORRECT = int((pred0 == y).sum())
    BENCH = BENCH_CORRECT / N * 100
    L = link.L
    cw_of = {"SEQ": np.arange(L) // 4, "INT": np.argsort(m["int_order"]) // 4}   # codeword of a literal inside a rule


def shared_groups(lits, lay):
    """Groups (>= 2 literals) of `lits` that share one Hamming codeword inside a rule for layout SEQ / INT."""
    groups = {}
    for l in lits:
        groups.setdefault(int(cw_of[lay][l]), []).append(int(l))
    return [g for g in groups.values() if len(g) >= 2]


def run_set(it, K, lits):
    """All attacks x layouts for one literal set (applied to every rule). Returns result rows."""
    rows = []
    if True:
        names = " | ".join(m["lit_name"][lits])
        shared = {lay: [[str(m["lit_name"][l]) for l in g] for g in shared_groups(lits, lay)] for lay in cw_of}
        for atk in ATTACKS:
            flips = link.attack(atk, lits=lits)
            for lay in E.LAYOUTS:
                rx, hist = link.receive(lay, flips)
                cube = link.to_cube(rx)
                pred = eng.sums(cube).argmax(1)
                correct = int((pred == y).sum())
                h = [None] * 4 if hist is None else [int(v) for v in hist]
                rows.append({"iteration": it + 1, "K": K, "attack": atk, "layout": lay,
                             "literal indices": " ".join(map(str, lits)), "literals": names,
                             "shared codeword (SEQ)": "; ".join(" + ".join(g) for g in shared["SEQ"]),
                             "shared codeword (INT)": "; ".join(" + ".join(g) for g in shared["INT"]),
                             "bits changed by MITM": len(flips),
                             "cw 1 bit": h[0], "cw 2 bits": h[1], "cw 3 bits": h[2], "cw 4 bits": h[3],
                             "max bits in one cw": None if hist is None else
                             (int(np.max(np.nonzero(hist)[0]) + 1) if hist.any() else 0),
                             "wrong bits after decode": int((cube != m["cube"]).sum()),
                             "clauses changed": int((cube != m["cube"]).any(2).sum()),
                             "correct": correct, "N": N, "accuracy %": correct / N * 100,
                             "benchmark correct": BENCH_CORRECT, "benchmark %": BENCH,
                             "Δ vs benchmark": correct / N * 100 - BENCH,
                             "samples changed %": (pred != pred0).mean() * 100})
    return rows


if __name__ == "__main__":
    START, STOP, OUT = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    init(sys.argv[1])
    rows = []
    t0 = time.time()
    for it in range(START, STOP):
        rng = np.random.default_rng(10_000 + it)
        for K in K_LIST:
            rows += run_set(it, K, sorted(rng.choice(L, K, replace=False).tolist()))
        if (it + 1 - START) % 5 == 0 or it + 1 == STOP:
            pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.10f")
            print(f"[{DS}] iteration {it + 1} of chunk {START}-{STOP}  ({time.time() - t0:.0f} s)", flush=True)
    print(f"[{DS}] done chunk {START}-{STOP}: {len(rows)} runs in {time.time() - t0:.0f} s, benchmark {BENCH:.4f} %")
