"""Check the Raspberry Pi results against the offline twin: per round, the class sums of every TM and of the fused
model for all 50 flows must be identical to simulate.py, and so must the predictions and corrected-codeword counts.

usage: python compare.py [results] [expected]
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
exp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "expected")
R, E = pd.read_csv(f"{res}/rounds.csv"), pd.read_csv(f"{exp}/rounds.csv")
SR, SE = pd.read_csv(f"{res}/class_sums.csv"), pd.read_csv(f"{exp}/class_sums.csv")
key = ["round", "flow", "source"]
m = SR.merge(SE, on=key, suffixes=("", "_exp"))
cols = ["c0", "c1", "c2", "c3", "c4", "pred"]
m["same"] = (m[cols].to_numpy() == m[[c + "_exp" for c in cols]].to_numpy()).all(1)
per_round = m.groupby("round").same.all()
rr = R.merge(E[["round", "acc_pct", "corrected"]], on="round", suffixes=("", "_exp"))
rr["sums_match"] = rr["round"].map(per_round).fillna(False)
rr["corrected_match"] = rr.corrected == rr.corrected_exp
bad = rr[~(rr.sums_match & rr.corrected_match & (rr.complete == 1))]
print(f"{len(rr)} rounds compared: class sums identical in {int(rr.sums_match.sum())}, corrected codewords identical "
      f"in {int(rr.corrected_match.sum())}, complete {int((rr.complete == 1).sum())}")
missing = sorted(set(E["round"]) - set(R["round"]))
if missing:
    print(f"rounds missing in {res}: {missing}")
if len(bad):
    print("rounds that differ:")
    print(bad[["round", "layout", "attack", "K", "complete", "pkts_tm0", "pkts_tm1", "pkts_tm2", "acc_pct", "acc_pct_exp",
               "corrected", "corrected_exp"]].to_string(index=False))
else:
    print("PASS: every round reproduces the offline twin exactly")
