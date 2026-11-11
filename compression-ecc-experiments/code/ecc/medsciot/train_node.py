"""Train one weighted TM node with tmu and export its clause bank cube + weights.

usage: python train_node.py NODE_ID [EPOCHS] [CLAUSES] [T] [S]
"""
import sys
import time
import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier

node = int(sys.argv[1])
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
CLAUSES = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
T = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
S = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0

D = np.load("/home/<user>/medsciot/data.npz", allow_pickle=True)
idx = D[f"node{node}"]
Xtr, ytr = D["Xtr"][idx], D["ytr"][idx]
Xte, yte = D["Xte"], D["yte"]
C = len(D["class_names"])

# Fixed stratified validation subset for progress reporting
rng = np.random.default_rng(100 + node)
val = np.concatenate([rng.permutation(np.where(yte == c)[0])[:600] for c in range(C)])

tm = TMClassifier(CLAUSES, T, S, platform="CPU", weighted_clauses=True, seed=1000 + node)
for ep in range(EPOCHS):
    t0 = time.time()
    tm.fit(Xtr, ytr)
    t1 = time.time()
    acc = (tm.predict(Xte[val]) == yte[val]).mean()
    print(f"node{node} epoch {ep+1}/{EPOCHS} train {t1-t0:.1f}s  balanced-val acc {acc:.4f}", flush=True)

# Export include/exclude cube: action = MSB of the TA state
o2 = tm.clause_banks[0].number_of_literals
cube = np.zeros((C, CLAUSES, o2), dtype=np.uint8)
for c in range(C):
    cb = tm.clause_banks[c]
    bank = cb.clause_bank.reshape(CLAUSES, cb.number_of_ta_chunks, cb.number_of_state_bits_ta)
    msb = bank[:, :, cb.number_of_state_bits_ta - 1].astype(np.uint32)
    bits = ((msb[:, :, None] >> np.arange(32, dtype=np.uint32)) & 1).reshape(CLAUSES, -1)
    cube[c] = bits[:, :o2]
weights = np.stack([tm.weight_banks[c].get_weights() for c in range(C)]).astype(np.int32)

# Spot-check the extraction against tmu's own TA action accessor
for _ in range(200):
    c, j, k = rng.integers(C), rng.integers(CLAUSES), rng.integers(o2)
    assert cube[c, j, k] == tm.get_ta_action(j, k, the_class=c)

# Reference predictions + class sums from tmu on a fixed subset (used to validate our inference engine)
ref_idx = val
ref_pred, ref_sums = tm.predict(Xte[ref_idx], return_class_sums=True)
# Original-TM reference on the WHOLE test set (tmu's own predict), used by the ECC + inference pipeline
t0 = time.time()
test_pred, test_sums = tm.predict(Xte, return_class_sums=True)
print(f"node{node} tmu predict on {len(yte)} test flows: {time.time()-t0:.1f}s", flush=True)
np.savez_compressed(f"/home/<user>/medsciot/node{node}.npz", cube=cube, weights=weights, T=T, s=S,
                    clauses=CLAUSES, ref_idx=ref_idx, ref_pred=ref_pred, ref_sums=ref_sums,
                    test_pred=test_pred, test_sums=test_sums.astype(np.int32))
print(f"node{node} saved; included literals/clause mean {cube.sum(2).mean():.2f}", flush=True)
