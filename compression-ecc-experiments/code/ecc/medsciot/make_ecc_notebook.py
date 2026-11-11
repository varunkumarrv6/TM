"""Builds ecc_tm_inference.ipynb: sender -> MITM corruption -> model-blind Hamming decoder -> manual TM inference."""
import json

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n")})


md(r"""
# MedSec-25 · 3 Tsetlin Machines → MITM corruption → Hamming(7,4) → manual TM inference

Same flow as `error_coding.ipynb` (cells 53 → 54 → 55), now for the 3 TMs trained on MedSec-25:

```
TM 0 ─┐                                  ┌─ decode + correct ─┐
TM 1 ─┼─ clause bank → packets → Hamming ┤   (model-blind)    ├─ rebuild 3 banks → manual inference → fuse → compare
TM 2 ─┘        tx/tm{n}_{layout}.bin     └── MITM edits bits ─┘                    with the original (no-attack) predictions
```

**Run:** kernel **"Python (Edge-IIoTset)"** → Run All (about 1–2 minutes). Change the settings in the next cell to try
other attacks / layouts / attacked TMs. All code lives in `ecc_pipeline.py`.
""")

code(r"""
import numpy as np, pandas as pd, os, time
import ecc_pipeline as E

# ---------------- SETTINGS ----------------
LAYOUT = "KEY"                # "none" (no Hamming), "SEQ", "INT", "KEY"
ATTACK = "A4"                 # "A1" random, "A2" force include, "A3" force exclude, "A4" toggle
TARGETS = ["Dst Port"]        # raw features attacked by A2/A3/A4
K = 8                         # A1: random literals flipped per clause
ATTACKED_NODES = [0, 1, 2]    # which TM files the man-in-the-middle corrupts (try [1] = one file)
# ------------------------------------------

M = E.load_nodes()
T = M["T"]
original_pred = E.fuse(M["clean_sums"], T)        # deployed model (4-bit weights), no attack = reference
rng = np.random.default_rng(2026)
SHOW = np.concatenate([rng.choice(np.where(E.yte == c)[0], 10, replace=False) for c in range(E.NUM_CLASSES)])
print("classes        :", list(E.class_names))
print("test flows     :", len(E.yte), "| shown per-sample:", len(SHOW))
print("original TM    : accuracy", f"{(original_pred == E.yte).mean()*100:.4f} %  ({(original_pred == E.yte).sum():,} correct)")
""")

md("## Step 1 · Sender: encode each TM's clause bank into a Hamming `.bin` file")
code(r"""
print("=" * 72)
print(f"            HAMMING(7,4) SENDER   (layout = {LAYOUT})")
print("=" * 72)
print(f"Header byte 0       : weight nibble (bits 7-4, signed -8..7) | class nibble (bits 3-0); bytes 1-2 seq; byte 3 node")
print(f"Rules per packet    : {E.RULES_PER_PACKET} rules of ONE class and ONE weight, {E.RULE_BYTES} bytes each")
print(f"Payload             : {E.PAYLOAD_BYTES} bytes (rules + header copy) -> Hamming {E.ENC_PAYLOAD} bytes")
print(f"Datagram            : {E.HEADER_BYTES} clear header + {E.ENC_PAYLOAD} = {E.DG_ENC} bytes ({E.DG_RAW} without Hamming)")
print(f"Datagrams per TM    : {[len(s) for s in M['schedules']]}")
for n in range(E.N_NODES):
    path = E.encode_node_file(n, M["cubes"][n], M["weights"][n], LAYOUT)
    print(f"TM {n} -> {path}   {os.path.getsize(path):,} bytes")

# which 4 literals of clause 0 share the first Hamming codeword in this layout
order = E.INT_ORDER if LAYOUT == "INT" else np.arange(E.L)
if LAYOUT in ("SEQ", "INT"):
    print("\nFirst codeword carries :", list(E.LIT_NAME[order[:4]]))
elif LAYOUT == "KEY":
    print("\nKEY: payload bits are shuffled with the secret key before Hamming, so a codeword's 4 bits are random literals")
""")

md("## Step 2 · Man-in-the-middle corrupts the files in transit")
code(r"""
print("=" * 72)
print(f"            MITM CORRUPTION   attack = {ATTACK}   TMs = {ATTACKED_NODES}")
print("=" * 72)
for n in ATTACKED_NODES:
    src = os.path.join(E.TX_DIR, f"tm{n}_{LAYOUT}.bin")
    dst = src.replace(".bin", "_corrupted.bin")
    changed = E.mitm_attack(src, dst, n, LAYOUT, ATTACK, TARGETS, K)
    a, b = np.fromfile(src, np.uint8), np.fromfile(dst, np.uint8)
    byte, mask = E.literal_location(n, LAYOUT)
    hit = np.argwhere(((a[byte] ^ b[byte]) & mask) != 0)
    print(f"\nTM {n}: {os.path.basename(dst)}   bits flipped = {changed:,}")
    for c, j, l in hit[:5]:
        fb = byte[c, j, l]
        print(f"   class {c} clause {j:4d} literal '{E.LIT_NAME[l]}':  file byte {fb:7d}  0x{a[fb]:02X} -> 0x{b[fb]:02X}")
    if len(hit) > 5:
        print(f"   ... {len(hit) - 5:,} more")
""")

md("## Step 3 · Model-blind receiver: decode the 3 files (no access to the original clause banks)")
code(r"""
print("=" * 72)
print("            MODEL-BLIND HAMMING DECODER")
print("=" * 72)
t0 = time.perf_counter()
decoded = []
for n in range(E.N_NODES):
    path = os.path.join(E.TX_DIR, f"tm{n}_{LAYOUT}" + ("_corrupted" if n in ATTACKED_NODES else "") + ".bin")
    cube, w, flagged = E.decode_node_file(path, n, LAYOUT)       # raises if a header is broken
    decoded.append((cube, w))
    status = "CLEAN -> DIRECT INFERENCE" if flagged == 0 else "ERRORS DETECTED -> ECC CORRECTION"
    print(f"TM {n}: {os.path.basename(path):24s} packets {len(M['schedules'][n])} | headers OK | "
          f"codewords corrected {flagged:>8,} | {status}")
print(f"\nDecode + rebuild time : {(time.perf_counter() - t0)*1000:.1f} ms")
""")

md("## Step 4 · Rule verification (original clause banks used ONLY for this check)")
code(r"""
all_ok = True
for n, (cube, w) in enumerate(decoded):
    lit_bad = int((cube != M["cubes"][n]).sum())
    cl_bad = int((cube != M["cubes"][n]).any(axis=2).sum())
    w_bad = int((w != M["weights"][n]).sum())
    ok = lit_bad == 0 and w_bad == 0
    all_ok &= ok
    print(f"TM {n}: wrong literal bits {lit_bad:>8,} | damaged clauses {cl_bad:>5,} / {E.NUM_CLASSES*E.CLAUSES} | "
          f"wrong weights {w_bad} | {'PASS' if ok else 'FAIL'}")
print("\nRULE VERIFICATION :", "PASS - all clause banks restored exactly" if all_ok else "FAIL - some errors remain")
""")

md(r"""
## Step 5 · Manual TM inference on the rebuilt clause banks
`tm_predict_manual` is plain numpy (included literal that is false ⇒ clause off; empty clause ⇒ off;
score = Σ weight × clause). Each TM's score is capped at ±T and the 3 are added.
The *No ECC* column shows what the same attack does when the files are sent without Hamming.
""")
code(r"""
# same attack, sent WITHOUT Hamming, for comparison
noecc = E.run_pipeline("none", ATTACK, TARGETS, K, ATTACKED_NODES, M=M)

def manual_fused(i, banks):
    return int(np.argmax(sum(np.clip(E.tm_predict_manual(E.Xte[i], c, w), -T, T) for c, w in banks)))

print(f"{'Sample':<8}{'Actual':<18}{'Original TM':<18}{'ECC decoded':<18}{'No ECC':<18}{'Match'}")
print("-" * 86)
cn = E.class_names
for i in SHOW:
    p_ecc = manual_fused(i, decoded)
    p_raw = manual_fused(i, [(r["cube"], r["weights"]) for r in noecc])
    print(f"{i:<8}{cn[E.yte[i]]:<18}{cn[original_pred[i]]:<18}{cn[p_ecc]:<18}{cn[p_raw]:<18}"
          f"{'YES' if p_ecc == original_pred[i] else 'NO'}")
""")

code(r"""
# whole test set (fast engine, same logic)
sums_ecc = [E.class_sums_all(c, w) for c, w in decoded]
sums_raw = [E.class_sums_all(r["cube"], r["weights"]) for r in noecc]
pred_ecc, pred_raw = E.fuse(sums_ecc, T), E.fuse(sums_raw, T)
y = E.yte
print("=" * 72)
print("                    FINAL RESULTS (whole test set)")
print("=" * 72)
print(f"Test flows                       : {len(y):,}")
print(f"Original TM accuracy             : {(original_pred == y).mean()*100:8.4f} %")
print(f"After attack, NO ECC             : {(pred_raw == y).mean()*100:8.4f} %   "
      f"(same prediction as original: {(pred_raw == original_pred).mean()*100:.2f} %)")
print(f"After attack, Hamming {LAYOUT:<4}       : {(pred_ecc == y).mean()*100:8.4f} %   "
      f"(same prediction as original: {(pred_ecc == original_pred).mean()*100:.2f} %)")
same_scores = all(np.array_equal(a, b) for a, b in zip(sums_ecc, M["clean_sums"]))
print(f"Class scores identical to clean   : {'YES' if same_scores else 'NO'}")
print(f"Prediction mismatches vs original: {(pred_ecc != original_pred).sum():,}")
print("=" * 72)
print("ECC + INFERENCE RESULT :", "PASS" if (pred_ecc == original_pred).all() else "CHECK")
print("=" * 72)
""")

md(r"""
## Step 6 · All attacks × all layouts (same file pipeline every time)
`PASS` = after decoding, every one of the 105,066 test flows gets exactly the original TM's prediction.
""")
code(r"""
usage = sum(np.bincount(E.LIT_RAW, weights=c.sum(axis=(0, 1)), minlength=len(E.feat_names)) for c in M["cubes"])
top8 = list(E.feat_names[np.argsort(-usage)[:8]])
runs = [("A1", [], 8, "A1 random, 8 literals / clause"),
        ("A2", ["Dst Port"], 1, "A2 include Dst Port"),
        ("A2", ["Dst Port", "Flow Duration"], 1, "A2 include Dst Port + Flow Duration"),
        ("A3", top8, 1, "A3 exclude top-8 features"),
        ("A4", ["Dst Port"], 1, "A4 toggle Dst Port"),
        ("A4", ["Dst Port", "Flow Duration"], 1, "A4 toggle Dst Port + Flow Duration")]
rows = []
for atk, tg, k, label in runs:
    for lay in E.LAYOUTS:
        rx = E.run_pipeline(lay, atk, tg, k, ATTACKED_NODES, M=M)
        pred = E.fuse([E.class_sums_all(r["cube"], r["weights"]) for r in rx], T)
        rows.append({"attack": label, "layout": lay,
                     "bits flipped": sum(r["bits_changed"] for r in rx),
                     "literal errors left": sum(r["lit_mismatch"] for r in rx),
                     "correct flows": int((pred == E.yte).sum()), "accuracy %": (pred == E.yte).mean() * 100,
                     "same as original %": (pred == original_pred).mean() * 100,
                     "result": "PASS" if (pred == original_pred).all() else "FAIL"})
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(E.BASE, "ecc_pipeline_summary.csv"), index=False)
print(f"Attacked TMs: {ATTACKED_NODES} | original accuracy {(original_pred == E.yte).mean()*100:.2f} %")
summary
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "proj", "display_name": "Python (Edge-IIoTset)", "language": "python"},
                   "language_info": {"name": "python"}}}
json.dump(nb, open("/home/<user>/medsciot/ecc_tm_inference.ipynb", "w"), indent=1)
print("written ecc_tm_inference.ipynb")
