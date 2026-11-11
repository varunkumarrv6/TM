"""Distributed setup: split the training set into 5 disjoint stratified shards and train one weighted TM per shard.
The test set stays separate. Exports each TM's include/exclude cube + weights -> tm5_node{N}.npz

usage: python train5.py NODE_ID [EPOCHS] [CLAUSES] [T] [S]
"""
import sys
import time
import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier

N_NODES = 5
node = int(sys.argv[1])
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
CLAUSES = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
T = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
S = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0

D = np.load("/home/<user>/new_medsec/data.npz", allow_pickle=True)
Xtr, ytr, Xte, yte = D["Xtr"], D["ytr"], D["Xte"], D["yte"]
C = len(D["class_names"])

# Same seed in every process -> identical disjoint stratified shards
rng = np.random.default_rng(2026)
shards = [[] for _ in range(N_NODES)]
for c in range(C):
    for n, part in enumerate(np.array_split(rng.permutation(np.where(ytr == c)[0]), N_NODES)):
        shards[n].append(part)
idx = rng.permutation(np.concatenate(shards[node]))
Xtr, ytr = Xtr[idx], ytr[idx]
print(f"node{node} shard {len(idx)} {np.bincount(ytr, minlength=C)}", flush=True)

tm = TMClassifier(CLAUSES, T, S, platform="CPU", weighted_clauses=True, seed=1000 + node)
for ep in range(EPOCHS):
    t0 = time.time()
    tm.fit(Xtr, ytr)
    t1 = time.time()
    acc = (tm.predict(Xte) == yte).mean()
    print(f"node{node} epoch {ep+1}/{EPOCHS} train {t1-t0:.1f}s  test acc {acc:.4f}", flush=True)

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

for _ in range(200):
    c, j, k = rng.integers(C), rng.integers(CLAUSES), rng.integers(o2)
    assert cube[c, j, k] == tm.get_ta_action(j, k, the_class=c)

test_pred, test_sums = tm.predict(Xte, return_class_sums=True)
np.savez_compressed(f"/home/<user>/new_medsec/tm5_node{node}.npz", cube=cube, weights=weights, T=T, s=S,
                    clauses=CLAUSES, train_idx=idx, test_pred=test_pred, test_sums=test_sums.astype(np.int32))
print(f"node{node} saved; tmu test acc {(test_pred == yte).mean():.4f}", flush=True)
