"""Monte-Carlo: 150 iterations. In every iteration, for K = 2, 4, 6, pick K random literals; the SAME literals are
attacked in every clause of every class of every attacked TM (include / exclude / toggle), sent as C-format packets
with no ECC / Hamming SEQ / INT / KEY. A new random set is drawn in the next iteration. Writes mc_results.csv."""
import sys
import time
import numpy as np
import pandas as pd
import ecc_pipeline as E
import mc_fast as F

ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 150
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0          # iterations START .. STOP-1 (for parallel chunks)
STOP = int(sys.argv[3]) if len(sys.argv) > 3 else ITER
OUT = sys.argv[4] if len(sys.argv) > 4 else "mc_results.csv"
K_LIST = [2, 4, 6, 8, 10, 12, 16, 20]
ATTACKS = ["A2", "A3", "A4"]
ATTACKED_NODES = [0, 1, 2]
BASE = E.BASE

M = E.load_nodes(); T = M["T"]; y = E.yte; N = len(y)
CLEAN = E.fuse(M["clean_sums"], T)
BENCH = (CLEAN == y).mean() * 100
streams = [F.NodeStream(M, n) for n in range(E.N_NODES)]
cw_of = {"SEQ": np.arange(E.L) // 4, "INT": np.argsort(E.INT_ORDER) // 4}     # codeword of a literal in a rule

rows = []
t0 = time.time()
for it in range(START, STOP):
    rng = np.random.default_rng(10_000 + it)
    for K in K_LIST:
        lits = sorted(rng.choice(E.L, K, replace=False).tolist())
        names = " | ".join(str(E.LIT_NAME[l]) for l in lits)
        shared = {}
        for lay, cw in cw_of.items():                                  # literals that share one codeword
            groups = {}
            for l in lits:
                groups.setdefault(int(cw[l]), []).append(str(E.LIT_NAME[l]))
            shared[lay] = [g for g in groups.values() if len(g) >= 2]
        for atk in ATTACKS:
            for lay in E.LAYOUTS:
                sums, hist, wrong, changed = [], np.zeros(4, int), 0, 0
                for n in range(E.N_NODES):
                    if n not in ATTACKED_NODES:
                        sums.append(M["clean_sums"][n]); continue
                    r, h = streams[n].receive(lay, atk, lits)
                    att, _ = streams[n].receive("none", atk, lits)
                    changed += int((att != streams[n].rows).sum())
                    cube = streams[n].to_cube(r)
                    wrong += int((cube != M["cubes"][n]).sum())
                    if h is not None:
                        hist += h
                    sums.append(F.class_sums_incremental(M["cubes"][n], cube, M["weights"][n], M["clean_sums"][n]))
                pred = E.fuse(sums, T)
                correct = int((pred == y).sum())
                rows.append({"iteration": it + 1, "K": K, "attack": atk, "layout": lay, "literals": names,
                             "shared codeword (SEQ)": "; ".join(" + ".join(g) for g in shared["SEQ"]),
                             "shared codeword (INT)": "; ".join(" + ".join(g) for g in shared["INT"]),
                             "bits changed by MITM": changed,
                             "cw 1 bit": hist[0] if lay != "none" else None,
                             "cw 2 bits": hist[1] if lay != "none" else None,
                             "cw 3 bits": hist[2] if lay != "none" else None,
                             "cw 4 bits": hist[3] if lay != "none" else None,
                             "max bits in one cw": (int(np.max(np.nonzero(hist)[0]) + 1) if hist.any() else 0)
                                                   if lay != "none" else None,
                             "wrong bits after decode": wrong, "correct": correct,
                             "accuracy %": correct / N * 100, "Δ vs benchmark": correct / N * 100 - BENCH,
                             "flows changed %": (pred != CLEAN).mean() * 100})
    if (it + 1) % 10 == 0:
        pd.DataFrame(rows).to_csv(f"{BASE}/{OUT}", index=False)
        print(f"iteration {it + 1} of chunk {START}-{STOP}  ({time.time() - t0:.0f} s)", flush=True)
pd.DataFrame(rows).to_csv(f"{BASE}/{OUT}", index=False)
print(f"done: {len(rows)} runs in {time.time() - t0:.0f} s, benchmark {BENCH:.4f} %")
