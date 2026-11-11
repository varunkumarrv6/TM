"""Make every Hamming SEQ / INT Monte Carlo panel contain at least MIN_SHARED iterations in which 2+ of the K random
literals share one Hamming codeword (the case Hamming(7,4) cannot repair).

For a (K, layout) with fewer: pick an iteration whose set has no shared codeword in SEQ or INT, re-draw that
iteration's K random literals (seed [20_000, iteration, K, attempt]) until the set has a shared codeword in that layout,
and recompute that iteration for all attacks and all 4 layouts (they use the same literal set). The re-drawn iterations
are marked in the column "forced shared cw" (+ "redraw seed"). The original results are kept as
mc_results_<ds>_unforced.csv.

usage: python mc_force_shared.py {medsec|iot|mnist}
"""
import os
import sys
import numpy as np
import pandas as pd
import mc_attacks as MCA

MIN_SHARED = 2
DS = sys.argv[1]
OUT = f"{MCA.E.ROOT}/{DS}/monte_carlo"
src = f"{OUT}/mc_results_{DS}.csv"
orig = f"{OUT}/mc_results_{DS}_unforced.csv"
if not os.path.exists(orig):
    os.rename(src, orig)
MC = pd.read_csv(orig)
MC["forced shared cw"], MC["redraw seed"] = "", ""
MCA.init(DS)

ref = MC[(MC.attack == "A2") & (MC.layout == "none")].fillna({"shared codeword (SEQ)": "", "shared codeword (INT)": ""})
has = {(int(r.iteration), int(r.K), lay): r[f"shared codeword ({lay})"] != ""
       for _, r in ref.iterrows() for lay in ("SEQ", "INT")}
for K in MCA.K_LIST:
    for lay in ("SEQ", "INT"):
        while sum(has[(it, K, lay)] for it in range(1, 151)) < MIN_SHARED:
            free = [it for it in range(1, 151) if not has[(it, K, "SEQ")] and not has[(it, K, "INT")]
                    and not (MC[(MC.iteration == it) & (MC.K == K)]["forced shared cw"] != "").any()]
            it = int(np.random.default_rng([30_000, K, len(free)]).choice(free))
            attempt = 0
            while True:
                lits = sorted(np.random.default_rng([20_000, it, K, attempt]).choice(MCA.L, K, replace=False).tolist())
                if MCA.shared_groups(lits, lay):
                    break
                attempt += 1
            new = pd.DataFrame(MCA.run_set(it - 1, K, lits))
            new["forced shared cw"], new["redraw seed"] = lay, f"[20000, {it}, {K}, {attempt}]"
            MC = pd.concat([MC[~((MC.iteration == it) & (MC.K == K))], new])
            for l2 in ("SEQ", "INT"):
                has[(it, K, l2)] = bool(MCA.shared_groups(lits, l2))
            print(f"[{DS}] K={K} {lay}: iteration {it} re-drawn (attempt {attempt}): "
                  f"{' | '.join(MCA.m['lit_name'][lits])}", flush=True)

MC = MC.sort_values(["iteration", "K", "attack", "layout"])
assert len(MC) == 150 * 7 * 3 * 4
MC.to_csv(src, index=False, float_format="%.10f")
cnt = {lay: {K: sum(has[(it, K, lay)] for it in range(1, 151)) for K in MCA.K_LIST} for lay in ("SEQ", "INT")}
print(f"[{DS}] iterations with a shared codeword: {cnt}")
