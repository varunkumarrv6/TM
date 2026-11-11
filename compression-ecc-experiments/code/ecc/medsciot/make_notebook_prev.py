"""Builds medsciot_results.ipynb (explanations + clearer figures). Run once; then open the notebook."""
import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(r"""
# MedSec-25 · Man-in-the-middle attacks on 3 Tsetlin Machines · Hamming(7,4) defence

**How to run:** pick the kernel **"Python (Edge-IIoTset)"** (top-right in VS Code), then *Run All*.
By default the notebook reads the saved results (takes seconds). Set `RERUN = True` in the next cell to
redo everything from the raw CSV (about 5 minutes).

### The system in plain words
1. The MedSec-25 flows are split: **20 % test set** (never seen in training) and **80 % training**.
2. The training part is cut into **3 separate pieces**. Each piece trains its own Tsetlin Machine (TM 0, TM 1, TM 2).
   No TM sees another TM's data.
3. Each TM sends its learned **clause bank** (which literals each clause includes / excludes) to an
   **inference node** over the network. *This link is where the man-in-the-middle sits.*
4. The inference node runs every test flow through all 3 clause banks, caps each TM's class score at ±T,
   adds the 3 scores, and picks the highest class.
5. **Defence:** each TM encodes its clause bank with Hamming(7,4) before sending; the inference node decodes
   (and fixes errors) before using it.
6. **Packet format = `error_coding.ipynb` cell 53 + `rpi_customarchi.c`:** 4-byte header whose first byte is
   **weight nibble | class nibble**, then the Hamming-coded payload of 12 rules. All results below come from the real
   packet files (`ecc_pipeline.py`, `run_attacks.py`).
""")

code(r"""
import subprocess, numpy as np, pandas as pd, matplotlib.pyplot as plt
BASE = "/home/<user>/medsciot"
PY = "/home/<user>/iot_dataset/proj/bin/python3"
RERUN = False          # True = redo preprocessing, training of the 3 TMs and all attacks

if RERUN:
    subprocess.run(["bash", f"{BASE}/run_all.sh"], check=True)

df = pd.read_csv(f"{BASE}/results.csv")
clean = df[df.attack == "clean"].iloc[0]
df = df[df.attack != "clean"].copy()
df["param"] = df["param"].astype(int)
print(f"Deployed model, no attack: accuracy {clean.accuracy*100:.4f} %  ({int(clean.correct):,} / 105,066 correct), "
      f"balanced {clean.balanced_acc*100:.2f} %, attacks detected {clean.detection_rate*100:.2f} %")
""")

md(r"""
## Packet format (as in `rpi_customarchi.c`) and 4-bit weights
The TMs are **weighted** TMs: every clause has its own integer weight (|w| up to 212). The packet format puts the
weight in the **high nibble of header byte 0** (class in the low nibble), so each weight is stored as a signed
4-bit value (−8 … +7) after dividing all weights by one common scale; the threshold T is scaled the same way.
Clauses are grouped by (class, weight) and every group is cut into packets of 12 rules (last packet padded with empty
rules, which never fire — exactly like the C receiver skips empty clauses).

```
UDP datagram (956 bytes)
 ├─ clear header (4 B):  [weight nibble | class nibble] [seq lo] [seq hi] [TM node id]
 └─ Hamming(7,4) payload (952 B) = encode( 12 rules × 45 B  +  copy of the header (4 B) )      544 B → 952 B
      every byte → high-nibble codeword, low-nibble codeword;  nibble bit 3 = D1;  codeword P1 P2 D1 P4 D2 D3 D4
```
The receiver takes class/weight/sequence from the header and checks the clear header against the Hamming-protected
copy. The MITM attacks below change **literal bits of the rules** only.
""")

code(r"""
import sys; sys.path.insert(0, BASE)
import ecc_pipeline as E
M = E.load_nodes()
full = E.fuse(M["tmu_sums"], M["T_full"]); dep = E.fuse(M["clean_sums"], M["T"])
print(f"weight scale s = {M['scale']:.3f}  ->  weight nibble = round(w / s) in [-8, 7],  T = {M['T_full']} -> {M['T']}")
print(f"full integer weights (tmu)        : {(full == E.yte).mean()*100:.4f} %  ({(full == E.yte).sum():,} correct)")
print(f"4-bit weights (deployed, benchmark): {(dep == E.yte).mean()*100:.4f} %  ({(dep == E.yte).sum():,} correct)")
print(f"same prediction                     : {(full == dep).mean()*100:.4f} % of flows\n")
for n in range(E.N_NODES):
    sch = M["schedules"][n]
    print(f"TM {n}: {len(sch)} datagrams x {E.DG_ENC} B = {len(sch)*E.DG_ENC:,} B   weight values used: "
          f"{sorted(set(int(w) for w in sch[:, 1]))}")
c, wv, r0 = M["schedules"][0][0]
h = E.make_header(int(wv), int(c), 0, 0)
print(f"\nfirst datagram of TM 0: header bytes {[f'0x{b:02X}' for b in h]}  -> weight {E.parse_header(h)[0]:+d}, "
      f"class {E.class_names[E.parse_header(h)[1]]}, seq {E.parse_header(h)[2]}, node {E.parse_header(h)[3]}")
""")

md(r"""
## Benchmark and comparison of every attack
**Benchmark** = accuracy of the deployed model (3 fused TMs, 4-bit weights) on the 105,066 MedSec-25 test flows
with no attack (printed above).
Every attack below is compared with it (all 3 TMs attacked). `Δ` = accuracy − benchmark in percentage points;
`flows changed` = % of test flows whose prediction differs from the no-attack prediction.
""")

code(r"""
BENCH = clean.accuracy * 100
desc = {"A1": "flip K random literals per clause", "A2": "include ALL literals of top-K features",
        "A3": "exclude ALL literals of top-K features", "A4": "toggle ALL literals of top-K features",
        "A2L": "include top-K particular literals", "A3L": "exclude top-K particular literals",
        "A4L": "toggle top-K particular literals"}
rows = [{"attack": "benchmark (no attack)", "K": "", "No ECC %": round(BENCH, 4)}]
for atk, d in desc.items():
    for k in sorted(df[df.attack == atk].param.unique()):
        g = df[(df.attack == atk) & (df.param == k) & (df.attacked_nodes == 3)].set_index("scheme")
        row = {"attack": f"{atk}: {d}", "K": k}
        for sch, lab in [("none", "No ECC"), ("SEQ", "SEQ"), ("INT", "INT"), ("KEY", "KEY")]:
            row[f"{lab} %"] = round(g.loc[sch, "accuracy"] * 100, 4)
        row["Δ No ECC"] = round(row["No ECC %"] - BENCH, 4)
        row["Δ KEY"] = round(row["KEY %"] - BENCH, 4)
        row["flows changed, No ECC %"] = round((1 - g.loc["none", "agreement_with_clean"]) * 100, 4)
        row["flows changed, KEY %"] = round((1 - g.loc["KEY", "agreement_with_clean"]) * 100, 4)
        rows.append(row)
rc = pd.read_csv(f"{BASE}/clause_attack_results.csv")
for name in ["Include", "Exclude", "Toggle"]:
    for k in (1, 2, 4, 8):
        g = rc[(rc.attack == name) & (rc.K == k)].set_index("layout")
        row = {"attack": f"{name} K random literals per clause (different in each clause)", "K": k}
        for sch, lab in [("none", "No ECC"), ("SEQ", "SEQ"), ("INT", "INT"), ("KEY", "KEY")]:
            row[f"{lab} %"] = g.loc[sch, "accuracy %"]
        row["Δ No ECC"] = round(row["No ECC %"] - BENCH, 4)
        row["Δ KEY"] = round(row["KEY %"] - BENCH, 4)
        row["flows changed, No ECC %"] = round(100 - g.loc["none", "same as original %"], 4)
        row["flows changed, KEY %"] = round(100 - g.loc["KEY", "same as original %"], 4)
        rows.append(row)
bench_table = pd.DataFrame(rows).fillna("")
bench_table.to_csv(f"{BASE}/benchmark_comparison.csv", index=False)
pd.set_option("display.max_rows", 100); pd.set_option("display.float_format", "{:.4f}".format)
bench_table
""")

md(r"""
### The four attacks (done by the man-in-the-middle on the clause bank in transit)
| Attack | What the attacker does |
|---|---|
| **A1 random** | flips *k* random literals in every clause |
| **A2 force include** | sets every literal of the targeted feature(s) to *include* |
| **A3 force exclude** | sets every literal of the targeted feature(s) to *exclude* |
| **A4 toggle** | flips every literal of the targeted feature(s) (include ↔ exclude) |

Targeted features are the ones the TMs use most: **Dst Port**, then Flow Duration, Bwd IAT Mean, Bwd IAT Max, …
""")

md(r"""
### What are SEQ, INT and KEY?
Hamming(7,4) protects **4 bits at a time** (plus 3 parity bits = a 7-bit block) and can repair **only 1 wrong
bit per block**. With 2 or more wrong bits in the same block it fails (and can even flip a correct bit).

So the question is: **which 4 literals are packed together into one block?** We tried three ways:

| Name | Which 4 literals share a block | Attack on one feature (e.g. Dst Port) |
|---|---|---|
| **SEQ** (sequential) | the next 4 literals in the normal order, usually 4 bits of the **same** feature | all 4 bits in a block change → **cannot repair** |
| **INT** (interleaved) | one bit from **4 different features** | only 1 bit per block changes → **repaired** (but 2 attacked features that sit side by side break it) |
| **KEY** (keyed shuffle) | the bits are shuffled with a **secret random key** first, so the 4 bits are random | attacker can't know which bits share a block → errors spread out, most blocks get ≤ 1 error |

The cell below prints a real block from each layout so you can see it.
""")

code(r"""
D = np.load(f"{BASE}/data.npz", allow_pickle=True)
feat, lf = D["feat_names"], D["lit_feature"]
o = len(lf); L = 2 * o
rank = np.concatenate([np.arange((lf == j).sum()) for j in np.unique(lf)])
raw = np.concatenate([lf, lf]); pol = np.r_[np.zeros(o, int), np.ones(o, int)]; rk = np.r_[rank, rank]
name = lambda k: f"{'NOT ' if pol[k] else ''}{feat[raw[k]]} bit{rk[k]}"

seq = np.arange(L)
inter = np.lexsort((raw, pol, rk))
import sys; sys.path.insert(0, BASE)
import ecc_pipeline as E
key = E.key_perm(0, 0)        # secret shuffle of the 12 x 45 x 8 = 4320 rule bits of TM 0, packet 0

print("SEQ  block 0 :", [name(k) for k in seq[:4]])
print("INT  block 0 :", [name(k) for k in inter[:4]])
print("KEY  block 0 :", [(name(p % 360) if p % 360 < L else "padding bit") + f" (rule {p // 360} of the packet)"
                        for p in key[:4]])
""")

md(r"""
## Figure 1 · The main result (all 3 TMs attacked)
One strong setting per attack. Grey = no attack, orange = attacked with no protection,
blue = attacked but sent with Hamming(7,4) using the keyed shuffle (KEY).
""")

code(r"""
import sys; sys.path.insert(0, BASE)
import plot_style as P
P.setup()
BENCH = clean.accuracy * 100          # benchmark = accuracy of the 3 TMs on the MedSec-25 test set, no attack
COL = {k: c for k, _, c in P.SCHEMES}

def get(attack, k, scheme, m=3, col="accuracy"):
    r = df[(df.attack == attack) & (df.param == k) & (df.scheme == scheme) & (df.attacked_nodes == m)]
    return float(r[col].iloc[0]) * 100

def scheme_series(attack, params, m=3):
    return [(lab, [get(attack, k, s, m) for k in params], c) for s, lab, c in P.SCHEMES]

cases = [("A1", 8, "A1 random flips\n(8 per clause)"), ("A2", 1, "A2 include all of\nDst Port"),
         ("A3", 8, "A3 exclude all of\ntop-8 features"), ("A4", 1, "A4 toggle all of\nDst Port")]
fig, (ax,) = P.figure(1, 3)
P.grouped_bars(ax, [c[2] for c in cases],
               [("No attack (benchmark)", [BENCH] * 4, "#a8a79f"),
                ("Attacked, no protection", [get(a, k, "none") for a, k, _ in cases], COL["none"]),
                ("Attacked + Hamming – KEY", [get(a, k, "KEY") for a, k, _ in cases], COL["KEY"])], bench=BENCH)
P.top_legend(fig, ax)
ax.set_title("Figure 1 · All 3 TMs attacked: accuracy before / after the attack", pad=8)
fig.savefig(f"{BASE}/fig1_main_result.png", dpi=150); plt.show()
""")

md(r"""
## Figure 2 · Whole-feature attacks, getting stronger (all 3 TMs attacked)
**What "1 feature" means here:** the attacker takes a raw feature (first *Dst Port*) and changes **all of its
literals in every one of the 15,000 clauses**. Dst Port has 8 thresholds → 8 literals `Dst Port b0..b7` plus their
8 negations `NOT Dst Port b0..b7` = **16 literals per clause**. "2 features" adds *Flow Duration*, and so on
(most-used features first). The group labels give the feature names and the literals changed per clause.

**Example (A2 include, 1 feature):** every clause now contains both `Dst Port b0` ("port above threshold 0") and
`NOT Dst Port b0` ("port NOT above threshold 0"). No flow can satisfy both, so **no clause ever fires**, all class
scores are 0, and every flow falls to the first class, Benign → accuracy = share of Benign flows = **2.28 %**.
Toggle (A4) does the same for the clauses where the feature was absent. Exclude (A3) only *removes* conditions, so
clauses get looser but keep working → small drop.
""")

code(r"""
usage_f = np.zeros(len(feat))
for n in range(3):
    np.add.at(usage_f, raw, np.load(f"{BASE}/node{n}.npz")["cube"].sum(axis=(0, 1)))
top_f = np.argsort(-usage_f)[:8]
nlit = lambda fs: int(2 * np.isin(lf, fs).sum())            # literals per clause for a feature set
groups = [f"{feat[top_f[0]]}\n({nlit(top_f[:1])} literals)",
          f"+ {feat[top_f[1]]}\n({nlit(top_f[:2])} literals)",
          f"+ {feat[top_f[2]]}, {feat[top_f[3]]}\n({nlit(top_f[:4])} literals)",
          f"+ 4 more features\n({nlit(top_f[:8])} literals)"]
print("Features in attack order:", list(feat[top_f]))
panels = {"A2": "A2 · force INCLUDE all literals of the feature(s) in every clause",
          "A3": "A3 · force EXCLUDE all literals of the feature(s) in every clause",
          "A4": "A4 · TOGGLE all literals of the feature(s) in every clause"}
fig, axes = P.figure(3, 4)
for ax, (atk, title) in zip(axes, panels.items()):
    P.grouped_bars(ax, groups, scheme_series(atk, [1, 2, 4, 8]), bench=BENCH)
    ax.set_title(title, pad=6)
P.top_legend(fig, axes[0])
fig.savefig(f"{BASE}/fig2_attack_strength.png", dpi=150); plt.show()
""")

md(r"""
### Figure 2b · Random attack (A1): K random literals flipped in every clause
Each clause gets its **own** K random literals flipped (different literals in different clauses).
""")

code(r"""
fig, (ax,) = P.figure(1, 4)
P.grouped_bars(ax, [f"K = {k}" for k in (1, 2, 4, 8, 16)], scheme_series("A1", [1, 2, 4, 8, 16]), bench=BENCH)
ax.set_title("Figure 2b · A1: flip K random literals in every clause (all 3 TMs attacked)", pad=8)
P.top_legend(fig, ax)
fig.savefig(f"{BASE}/fig2b_random_flips.png", dpi=150); plt.show()
""")

md(r"""
## Figure 3 · Why the grouping (SEQ / INT / KEY) matters
Toggle attack (A4) on 1 feature vs 2 features, all 3 TMs attacked.
""")

code(r"""
fig, (ax,) = P.figure(1, 4)
P.grouped_bars(ax, ["Toggle 1 feature\n(Dst Port, 16 literals)", "Toggle 2 features\n(Dst Port + Flow Duration, 32 literals)"],
               scheme_series("A4", [1, 2]), bench=BENCH)
ax.set_title("Figure 3 · Same Hamming code, different grouping of bits into blocks", pad=8)
P.top_legend(fig, ax)
fig.savefig(f"{BASE}/fig3_grouping.png", dpi=150); plt.show()
""")

md(r"""
## Figure 4 · How many TMs does the attacker need to hit? (no protection)
With 1 or 2 TMs attacked, the clean TM(s) still out-vote the damaged ones.
""")

code(r"""
cases4 = [("A1", 8, "A1 random flips\n(8 per clause)"), ("A2", 1, "A2 include all of\nDst Port"),
          ("A3", 8, "A3 exclude all of\ntop-8 features"), ("A4", 1, "A4 toggle all of\nDst Port")]
fig, (ax,) = P.figure(1, 3)
P.grouped_bars(ax, [c[2] for c in cases4],
               [(f"{m} of 3 TMs attacked", [get(a, k, "none", m) for a, k, _ in cases4], c)
                for m, c in [(1, "#8fb9ea"), (2, "#4f90dd"), (3, "#1c4f8f")]], bench=BENCH)
ax.set_title("Figure 4 · No protection: accuracy vs number of attacked TMs", pad=8)
P.top_legend(fig, ax)
fig.savefig(f"{BASE}/fig4_nodes_attacked.png", dpi=150); plt.show()
""")

md(r"""
## Single-literal attacks (one literal changed in every clause)
Instead of attacking a whole feature (all its thermometer bits, x and NOT x), the MITM now changes **only one
literal** in every clause of every class: force it to include (A2), force it to exclude (A3) or toggle it (A4).
Then the top-2, top-4 and top-8 literals. Literals are picked by how many clauses include them (the attacker can read
the clause banks in transit). The cell below prints which literals were selected.
""")

code(r"""
cubes = [np.load(f"{BASE}/node{n}.npz")["cube"] for n in range(3)]
use = sum(c.sum(axis=(0, 1)).astype(np.int64) for c in cubes)
lit_order = np.argsort(-use, kind="stable")
total_clauses = 3 * cubes[0].shape[0] * cubes[0].shape[1]
print("Selected literals (in order):")
for i, k in enumerate(lit_order[:8], 1):
    print(f"  {i}. {name(k):28s} included in {use[k]:5d} of {total_clauses} clauses ({use[k]/total_clauses*100:.1f} %)")
print("\nNOT <flag> b0 means 'this flag count is at its lowest level' (e.g. no RST flag in the flow).")
""")

code(r"""
lit_groups = [f"1 literal\n({name(lit_order[0])})", "2 literals\n(+ 1 more)", "4 literals\n(+ 3 more)", "8 literals\n(+ 7 more)"]
panels = {"A2L": "Figure 5 · A2 · force INCLUDE the same literal(s) in every clause",
          "A3L": "A3 · force EXCLUDE the same literal(s) in every clause",
          "A4L": "A4 · TOGGLE the same literal(s) in every clause"}
fig, axes = P.figure(3, 4)
for ax, (atk, title) in zip(axes, panels.items()):
    P.grouped_bars(ax, lit_groups, scheme_series(atk, [1, 2, 4, 8]), bench=BENCH)
    ax.set_title(title, pad=6)
P.top_legend(fig, axes[0])
fig.savefig(f"{BASE}/fig5_single_literal.png", dpi=150); plt.show()
""")

md(r"""
### Why SEQ and INT give exactly the benchmark when 1 or 2 literals are attacked
Hamming(7,4) protects 4 literal bits per block and repairs **1 wrong bit per block**. If every changed bit lands
in a **different** block, the decoder repairs all of them, the clause banks are restored bit-for-bit, and the
accuracy is *exactly* the benchmark (no-attack accuracy, same correct-flow count). With 4 literals, two of them share a block → 2 errors →
the block cannot be repaired.
""")

code(r"""
import ecc_pipeline as E
lit_top = [int(i) for i in lit_order[:8]]
inv_int = np.argsort(E.INT_ORDER)
lay_pos = {"SEQ": np.arange(E.L), "INT": inv_int}
lay_ord = {"SEQ": np.arange(E.L), "INT": E.INT_ORDER}
print("Hamming block of each attacked literal (same position in every clause):")
for lay in ("SEQ", "INT"):
    print(f"\n{lay}:")
    for k in lit_top[:4]:
        b = lay_pos[lay][k] // 4
        print(f"  {E.LIT_NAME[k]:24s} -> block {b:3d}  holds {[str(E.LIT_NAME[i]) for i in lay_ord[lay][4*b:4*b+4]]}")

rows = []
for k in (1, 2, 4):
    for lay in ("SEQ", "INT"):
        per = np.bincount(lay_pos[lay][lit_top[:k]] // 4, minlength=E.L // 4)   # toggled bits per block
        rows.append({"literals": k, "layout": lay, "blocks with 1 error (all 3 TMs)": int((per == 1).sum()) * 15000,
                     "blocks with 2+ errors": int((per >= 2).sum()) * 15000})
print("\nToggle attack, errors per Hamming block:")
print(pd.DataFrame(rows).to_string(index=False))

# one real block, step by step
c = cubes[0][4, 5]; b = lit_top[0] // 4
lits_in_block = [4*b + 3, 4*b + 2, 4*b + 1, 4*b]            # nibble bit 3 = D1 ... bit 0 = D4 (rpi_customarchi.c)
data = c[lits_in_block].astype(int)
d1, d2, d3, d4 = data
code = np.array([d1 ^ d2 ^ d4, d1 ^ d3 ^ d4, d1, d2 ^ d3 ^ d4, d2, d3, d4])
slot = [6, 5, 4, 2][lit_top[0] % 4]                        # codeword position of the attacked literal
rx = code.copy(); rx[slot] ^= 1
syn = (rx[0]^rx[2]^rx[4]^rx[6]) | ((rx[1]^rx[2]^rx[5]^rx[6]) << 1) | ((rx[3]^rx[4]^rx[5]^rx[6]) << 2)
fx = rx.copy(); fx[syn - 1] ^= 1
print(f"\nTM 0, class Reconnaissance, clause 5, SEQ block {b} (D1..D4): {[str(E.LIT_NAME[i]) for i in lits_in_block]}")
print("  data bits sent      :", data.tolist())
print("  7-bit codeword      :", code.tolist(), " (P1 P2 D1 P4 D2 D3 D4)")
print(f"  after MITM toggle   : {rx.tolist()}  (bit {slot+1} flipped = '{E.LIT_NAME[lit_top[0]]}')")
print(f"  syndrome            : {syn}  -> decoder flips bit {syn}")
print("  after correction    :", fx.tolist(), "-> data", fx[[2, 4, 5, 6]].tolist(), "= sent:", np.array_equal(fx[[2, 4, 5, 6]], data))

r = pd.read_csv(f"{BASE}/particular_literal_results.csv")
r = r[r.layout.isin(["none", "SEQ", "INT"]) & r.literals.isin([1, 2, 4])]
print("\nExact results from the real .bin pipeline:")
print(r[["attack", "literals", "layout", "bits changed", "wrong bits after decode", "correct flows", "accuracy %"]]
      .to_string(index=False, float_format="%.4f"))
""")

md(r"""
## Figure 6 · Whole feature vs single literal (no protection, all 3 TMs attacked)
Same three attacks, one target each: the whole feature **Dst Port** (all 16 of its literals) vs the single most-used
literal **NOT RST Flag Cnt b0**.
""")

code(r"""
pairs = [("A2", "A2L", "Include"), ("A3", "A3L", "Exclude"), ("A4", "A4L", "Toggle")]
fig, (ax,) = P.figure(1, 2)
P.grouped_bars(ax, [p[2] for p in pairs],
               [("whole feature: Dst Port (16 literals)", [get(a, 1, "none") for a, _, _ in pairs], COL["none"]),
                (f"single literal: {name(lit_order[0])}", [get(b, 1, "none") for _, b, _ in pairs], "#8fb9ea")],
               bench=BENCH)
ax.set_title("Figure 6 · Attacking one literal is far weaker than attacking a whole feature", pad=8)
P.top_legend(fig, ax)
fig.savefig(f"{BASE}/fig6_feature_vs_literal.png", dpi=150); plt.show()
""")

md(r"""
## Figure 7 · Different random literals in every clause (all 3 TMs attacked, real `.bin` files)
In **every clause** the attacker picks **its own K random literals** (different in each clause):
* **Include K**: K literals that are currently 0 (excluded) are set to 1
* **Exclude K**: K literals that are currently 1 (included) are set to 0
* **Toggle K**: K random literals are flipped

Example (TM 0, class Reconnaissance, clause 5, Include 2): before `[Dst Port b0, NOT Bwd IAT Tot b5, NOT Bwd Header Len b0]`
→ attacker adds `Fwd Pkts/s b1` and `NOT Active Max b0` → after Hamming the 2 added literals are removed again.
Values come from `clause_attack_results.csv` (made by `clause_attack_demo.ipynb`, which prints every changed value).
""")

code(r"""
rc = pd.read_csv(f"{BASE}/clause_attack_results.csv")
fig, axes = P.figure(3, 4)
for ax, name in zip(axes, ["Include", "Exclude", "Toggle"]):
    series = [(lab, list(rc[(rc.attack == name) & (rc.layout == s) & (rc.K <= 8)].sort_values("K")["accuracy %"]), c)
              for s, lab, c in P.SCHEMES]
    P.grouped_bars(ax, [f"K = {k}" for k in (1, 2, 4, 8)], series, bench=BENCH)
    ax.set_title(f"{name.upper()} K random literals in every clause (different literals in each clause)", pad=6)
P.top_legend(fig, axes[0])
fig.savefig(f"{BASE}/fig7_per_clause_attacks.png", dpi=150); plt.show()
""")

md(r"""
## Table · Full numbers (all 3 TMs attacked)
`agreement` = % of test flows that get exactly the same prediction as the clean model.
""")

code(r"""
t = df[df.attacked_nodes == 3].pivot_table(index=["attack", "param"], columns="scheme",
                                           values=["accuracy", "agreement_with_clean"], sort=False) * 100
t = t.reindex(columns=["none", "SEQ", "INT", "KEY"], level=1).round(2)
t.columns = [f"{'acc' if a == 'accuracy' else 'agreement'} · {s}" for a, s in t.columns]
t
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "proj", "display_name": "Python (Edge-IIoTset)", "language": "python"},
                   "language_info": {"name": "python"}}}
json.dump(nb, open("/home/<user>/medsciot/medsciot_results.ipynb", "w"), indent=1)
print("written medsciot_results.ipynb")
