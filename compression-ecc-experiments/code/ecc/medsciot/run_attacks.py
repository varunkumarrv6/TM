"""Runs every MITM attack through the real C-format packet files (ecc_pipeline.py) and writes
results.csv, clause_attack_results.csv and particular_literal_results.csv for the notebooks."""
import time
import numpy as np
import pandas as pd
import ecc_pipeline as E

BASE = E.BASE
M = E.load_nodes()
T = M["T"]
CLEAN = E.fuse(M["clean_sums"], T)
y = E.yte
BENIGN = int(np.where(E.class_names == "Benign")[0][0])

# targets: most-used raw features / literals in the trained clause banks
usage_f = np.zeros(len(E.feat_names))
lit_use = np.zeros(E.L, np.int64)
for cube in M["cubes_full"]:
    np.add.at(usage_f, E.LIT_RAW, cube.sum(axis=(0, 1)))
    lit_use += cube.sum(axis=(0, 1)).astype(np.int64)
TOP_F = [str(f) for f in E.feat_names[np.argsort(-usage_f)[:8]]]
TOP_L = [str(E.LIT_NAME[i]) for i in np.argsort(-lit_use, kind="stable")[:8]]
print("features:", TOP_F)
print("literals:", TOP_L)


def metrics(pred):
    mal = y != BENIGN
    return dict(correct=int((pred == y).sum()), accuracy=(pred == y).mean(),
                balanced_acc=np.mean([(pred[y == c] == c).mean() for c in range(E.NUM_CLASSES)]),
                detection_rate=(pred[mal] != BENIGN).mean(), agreement_with_clean=(pred == CLEAN).mean())


# encode every TM once per layout
for lay in E.LAYOUTS:
    for n in range(E.N_NODES):
        E.encode_node_file(n, M["cubes"][n], M["weights"][n], lay)


def node_result(n, lay, attack, targets, k, seed):
    src = f"{E.TX_DIR}/tm{n}_{lay}.bin"
    dst = src.replace(".bin", "_corrupted.bin")
    changed = E.mitm_attack(src, dst, n, lay, attack, targets, k, seed)
    cube, w, flagged = E.decode_node_file(dst, n, lay)
    err = int((cube != M["cubes"][n]).sum())
    sums = E.class_sums_all(cube, w) if err or (w != M["weights"][n]).any() else M["clean_sums"][n]
    return changed, err, flagged, sums


scen = [("A1", k, [], s) for k in (1, 2, 4, 8, 16) for s in (0, 1, 2)]
scen += [(a, k, TOP_F[:k], 0) for a in ("A2", "A3", "A4") for k in (1, 2, 4, 8)]
scen += [(a, k, TOP_L[:k], 0) for a in ("L_INC", "L_EXC", "L_TOG") for k in (1, 2, 4, 8)]
scen += [(a, k, [], 0) for a in ("INC", "EXC") for k in (1, 2, 4, 8)]
NAME = {"L_INC": "A2L", "L_EXC": "A3L", "L_TOG": "A4L"}

rows = []
c0 = metrics(CLEAN)
rows.append(dict(attack="clean", param="-", seed=None, scheme="-", attacked_nodes=0, lit_err_before=0,
                 lit_err_after=0, flagged_codewords=0, **c0))
t0 = time.time()
for attack, k, targets, seed in scen:
    for lay in E.LAYOUTS:
        res = [node_result(n, lay, attack, targets, k, seed) for n in range(E.N_NODES)]
        for m in (1, 2, 3):
            sums = [r[3] for r in res[:m]] + M["clean_sums"][m:]
            rows.append(dict(attack=NAME.get(attack, attack), param=k, seed=seed, scheme=lay, attacked_nodes=m,
                             lit_err_before=sum(r[0] for r in res[:m]), lit_err_after=sum(r[1] for r in res[:m]),
                             flagged_codewords=sum(r[2] for r in res[:m]), **metrics(E.fuse(sums, T))))
    print(f"{attack:6s} k={k:<2} seed={seed}  {time.time() - t0:6.0f}s", flush=True)

raw = pd.DataFrame(rows)
raw.to_csv(f"{BASE}/results_raw.csv", index=False)
agg = raw.drop(columns="seed").groupby(["attack", "param", "scheme", "attacked_nodes"], sort=False).mean().reset_index()
agg.to_csv(f"{BASE}/results.csv", index=False)


def side_table(codes, label_col):
    t = raw[raw.attack.isin(codes) & (raw.attacked_nodes == 3) & (raw.seed.fillna(0) == 0)].copy()
    t["attack"] = t.attack.map(codes)
    return pd.DataFrame({"attack": t.attack, label_col: t.param.astype(int), "layout": t.scheme,
                         "bits changed": t.lit_err_before, "wrong bits after decode": t.lit_err_after,
                         "correct flows": t.correct, "accuracy %": t.accuracy * 100,
                         "same as original %": t.agreement_with_clean * 100})


side_table({"INC": "Include", "EXC": "Exclude", "A1": "Toggle"}, "K").to_csv(f"{BASE}/clause_attack_results.csv", index=False)
side_table({"A2L": "Include", "A3L": "Exclude", "A4L": "Toggle"}, "literals").to_csv(
    f"{BASE}/particular_literal_results.csv", index=False)
print(f"done in {time.time() - t0:.0f}s; clean: {c0}")
