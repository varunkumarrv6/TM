"""Builds medsciot_results.ipynb: the 4 MITM attacks on the C-format packets (ecc_pipeline.py) vs the benchmark.
(The previous, larger version of this builder is kept as make_notebook_prev.py.)"""
import json

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n")})


md(r"""
# MedSec-25 · 3 weighted Tsetlin Machines · 4 man-in-the-middle attacks · Hamming(7,4)

**Run:** kernel **"Python (Edge-IIoT)"** (or "Python (Edge-IIoTset)", same environment) → *Run All* (≈ 3 minutes; every attack is run live through the packet files).

### System
1. MedSec-25 is split into a **test set** (105,066 flows, never used for training) and a training set.
   The training set is cut into **3 parts**; each part trains one weighted TM (TM 0, TM 1, TM 2).
2. Each TM sends its **clause bank** to the inference node as UDP datagrams in the format of
   `error_coding.ipynb` cell 53 / `rpi_customarchi.c`. **The man-in-the-middle sits on this link.**
3. The inference node decodes the datagrams (Hamming(7,4) repairs single-bit errors), rebuilds the 3 clause banks,
   caps each TM's class score at ±T, adds the three scores and picks the highest class.

### The 4 attacks (applied by the MITM to every intercepted datagram, i.e. every clause of every class)
| Attack | What the MITM does in **every clause** |
|---|---|
| **A1 random** | flips **K random literals** (different literals in each clause) |
| **A2 include** | sets the **targeted literal(s)** to **1 = include** |
| **A3 exclude** | sets the **targeted literal(s)** to **0 = exclude** |
| **A4 toggle** | **flips** the **targeted literal(s)** (0 → 1, 1 → 0) |

Targeted literals = the literals the TMs use most (K = 1, 2, 4, 8, 12, 16, 20 of them). The MITM only changes literal bits of the
rules; it never touches headers or Hamming parity bits.

**Benchmark** = accuracy of the same system with **no attack**. Every result is compared with it.
""")

code(r"""
import sys, time, numpy as np, pandas as pd, matplotlib.pyplot as plt
BASE = "/home/<user>/medsciot"
sys.path.insert(0, BASE)
import ecc_pipeline as E
import plot_style as P

# ---------------- SETTINGS ----------------
K_VALUES = [1, 2, 4, 8, 12, 16, 20]  # A1: random literals per clause; A2-A4: number of targeted literals
ATTACKED_NODES = [0, 1, 2]       # TMs whose datagrams the MITM modifies (try [0] = one TM)
A1_SEED = 0
SHOW_LAYOUT = "SEQ"              # layout used in the clause-by-clause showcase
# ------------------------------------------

M = E.load_nodes(); T = M["T"]; y = E.yte; N = len(y); cn = E.class_names
CLEAN = E.fuse(M["clean_sums"], T)
BENCH = (CLEAN == y).mean() * 100
BENCH_CORRECT = int((CLEAN == y).sum())
lit_use = sum(c.sum(axis=(0, 1)).astype(np.int64) for c in M["cubes_full"])
TARGETS = [int(i) for i in np.argsort(-lit_use, kind="stable")[:max(K_VALUES)]]

print(f"BENCHMARK (no attack): {BENCH:.4f} %  =  {BENCH_CORRECT:,} / {N:,} test flows correct\n")
print(f"{'class':18s}{'test flows':>11s}{'correct':>9s}{'recall':>9s}")
for c, name in enumerate(cn):
    m = y == c
    print(f"{name:18s}{m.sum():>11,}{(CLEAN[m] == c).sum():>9,}{(CLEAN[m] == c).mean()*100:>8.2f}%")
print("\nTargeted literals (A2-A4), most-used first:")
for i, t in enumerate(TARGETS, 1):
    print(f"  {i}. {E.LIT_NAME[t]:24s} included in {lit_use[t]:5,} of 15,000 clauses")
""")

md(r"""
## Packet format (`rpi_customarchi.c`)
```
UDP datagram (956 bytes)
 ├─ clear header (4 B):  [weight nibble | class nibble] [seq lo] [seq hi] [TM node id]
 └─ Hamming(7,4) payload (952 B) = encode( 12 rules × 45 B  +  copy of the header (4 B) )      544 B → 952 B
      every byte → high-nibble codeword, low-nibble codeword;  nibble bit 3 = D1;  codeword P1 P2 D1 P4 D2 D3 D4
```
The TMs are weighted: every clause weight is stored as a **signed 4-bit nibble (−8 … +7)** in header byte 0, next to
the class nibble. Clauses are grouped by (class, weight); each group is cut into packets of 12 rules.

**Protection layouts** (which 4 literal bits share one Hamming codeword):
**No ECC** = payload sent without Hamming · **SEQ** = 4 neighbouring literals · **INT** = 4 literals of different
features · **KEY** = rule bits shuffled with a secret key.
""")

code(r"""
full = E.fuse(M["tmu_sums"], M["T_full"])
print(f"weight scale {M['scale']:.3f}: nibble = round(w / scale) in [-8, 7];  T = {M['T_full']} -> {M['T']}")
print(f"full integer weights : {(full == y).mean()*100:.4f} %  ({(full == y).sum():,} correct)")
print(f"4-bit weight nibbles : {BENCH:.4f} %  ({BENCH_CORRECT:,} correct)   <- benchmark\n")
for n in range(E.N_NODES):
    print(f"TM {n}: {len(M['schedules'][n])} datagrams x {E.DG_ENC} B, weights used {sorted(set(int(w) for w in M['schedules'][n][:, 1]))}")
c, wv, r0 = M["schedules"][0][0]
h = E.make_header(int(wv), int(c), 0, 0)
print(f"\nfirst datagram of TM 0: header {[f'0x{b:02X}' for b in h]} -> weight {E.parse_header(h)[0]:+d}, class {cn[E.parse_header(h)[1]]}, seq 0, node 0")
""")

md(r"""
## Run the 4 attacks
For every attack, K and layout: each attacked TM's stream is encoded, every datagram is intercepted and modified by the
MITM, the inference node decodes the stream, rebuilds the clause banks and classifies all 105,066 test flows.
""")

code(r"""
for lay in E.LAYOUTS:                                  # clean streams, one file per TM and layout
    for n in range(E.N_NODES):
        E.encode_node_file(n, M["cubes"][n], M["weights"][n], lay)

rows, examples = [], {}
t0 = time.time()
for atk in E.ATTACKS:
    for k in K_VALUES:
        for lay in E.LAYOUTS:
            sums, changed, wrong, fixed = [], 0, 0, 0
            for n in range(E.N_NODES):
                if n not in ATTACKED_NODES:
                    sums.append(M["clean_sums"][n]); continue
                src = f"{E.TX_DIR}/tm{n}_{lay}.bin"
                dst = src.replace(".bin", "_corrupted.bin")
                changed += E.mitm_stream(src, dst, lay, atk, TARGETS[:k], k, A1_SEED + n)
                cube, w, f = E.decode_node_file(dst, n, lay)
                fixed += f
                wrong += int((cube != M["cubes"][n]).sum())
                sums.append(E.class_sums_all(cube, w))
                if n == 0 and k == 1:
                    examples[(atk, lay)] = cube
            pred = E.fuse(sums, T)
            correct = int((pred == y).sum())
            rows.append({"attack": atk, "K": k, "layout": lay, "bits changed by MITM": changed,
                         "codewords corrected": fixed, "wrong bits after decode": wrong,
                         "correct": correct, "accuracy %": correct / N * 100,
                         "Δ vs benchmark": correct / N * 100 - BENCH,
                         "flows changed %": (pred != CLEAN).mean() * 100})
    print(f"{atk} done ({time.time() - t0:.0f} s)")
R = pd.DataFrame(rows)
R.to_csv(f"{BASE}/results_4attacks.csv", index=False)
print("saved results_4attacks.csv")
""")

md(r"""
## Results vs benchmark (all numbers exact)
`Δ` = accuracy − benchmark (percentage points). The full table with correct-flow counts, bits changed and bits left
wrong after decoding is in `results_4attacks.csv` (and below the figures).
""")

code(r"""
pd.set_option("display.float_format", "{:.4f}".format); pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 220)
LAB = {"none": "No ECC", "SEQ": "SEQ", "INT": "INT", "KEY": "KEY"}
tab = [{"attack": "benchmark (no attack)", "K": "", **{f"{LAB[l]} %": BENCH for l in E.LAYOUTS}}]
for atk, desc in E.ATTACKS.items():
    for k in K_VALUES:
        g = R[(R.attack == atk) & (R.K == k)].set_index("layout")
        row = {"attack": f"{atk} · {desc}", "K": k}
        for lay in E.LAYOUTS:
            row[f"{LAB[lay]} %"] = g.loc[lay, "accuracy %"]
        for lay in E.LAYOUTS:
            row[f"Δ {LAB[lay]}"] = g.loc[lay, "Δ vs benchmark"]
        tab.append(row)
summary = pd.DataFrame(tab).fillna("")
summary.to_csv(f"{BASE}/benchmark_comparison.csv", index=False)
print(f"Benchmark (no attack): {BENCH:.4f} %  ({BENCH_CORRECT:,} / {N:,})   ·   TMs attacked: {ATTACKED_NODES}")
summary
""")

md("## Figures: one per attack (bars = accuracy, table under the bars = exact values)")

code(r"""
P.setup()
def attack_figure(atk, fname):
    if atk == "A1":
        groups = [f"K = {k}\n({k} random literal{'s' if k > 1 else ''} / clause)" for k in K_VALUES]
    else:
        groups = [f"1 literal\n({E.LIT_NAME[TARGETS[0]]})" if k == 1 else f"{k} literals\n(+ {k - 1} more)" for k in K_VALUES]
    series = [(lab, [R[(R.attack == atk) & (R.K == k) & (R.layout == s)]["accuracy %"].iloc[0] for k in K_VALUES], c)
              for s, lab, c in P.SCHEMES]
    fig, (ax,) = P.figure(1, 4, width=24)
    P.grouped_bars(ax, groups, series, bench=BENCH)
    ax.set_title(f"{atk} · {E.ATTACKS[atk]}  ·  TMs attacked: {ATTACKED_NODES}", pad=8)
    P.top_legend(fig, ax)
    fig.savefig(f"{BASE}/{fname}", dpi=150); plt.show()

for i, atk in enumerate(E.ATTACKS, 1):
    attack_figure(atk, f"fig{i}_{atk}.png")
""")

md("## Summary figure: all 4 attacks side by side (smallest and largest K)")

code(r"""
fig, axes = P.figure(2, 4)
groups = ["A1 random\nflips", "A2 include\nliteral(s)", "A3 exclude\nliteral(s)", "A4 toggle\nliteral(s)"]
for ax, k in zip(axes, [K_VALUES[0], K_VALUES[-1]]):
    series = [(lab, [R[(R.attack == a) & (R.K == k) & (R.layout == s)]["accuracy %"].iloc[0] for a in E.ATTACKS], c)
              for s, lab, c in P.SCHEMES]
    P.grouped_bars(ax, groups, series, bench=BENCH)
    ax.set_title(f"K = {k}   (A1: {k} random literal(s) per clause · A2-A4: top-{k} literal(s))  ·  TMs attacked: {ATTACKED_NODES}", pad=6)
P.top_legend(fig, axes[0])
fig.savefig(f"{BASE}/fig5_summary.png", dpi=150); plt.show()
""")

md("## Full result table (exact counts)")

code(r"""
R[["attack", "K", "layout", "bits changed by MITM", "codewords corrected", "wrong bits after decode",
   "correct", "accuracy %", "Δ vs benchmark", "flows changed %"]]
""")

md(r"""
## What the attack does inside one clause (K = 1, TM 0)
Before the attack → what the inference node receives without ECC → what it gets after Hamming decoding.
""")

code(r"""
orig = M["cubes"][0]
t = TARGETS[0]
c1, j1 = np.argwhere((orig[:, :, t] == 1) & (orig.sum(2) <= 6))[0]                       # includes the target
c2, j2 = np.argwhere((orig[:, :, t] == 0) & (orig.sum(2) >= 3) & (orig.sum(2) <= 6))[0]  # does not
lits = lambda r: ", ".join(str(E.LIT_NAME[i]) for i in np.nonzero(r)[0])
for atk in E.ATTACKS:
    print("=" * 110)
    print(f"{atk} · {E.ATTACKS[atk]}" + ("" if atk == "A1" else f"   (target literal: {E.LIT_NAME[t]})"))
    print("=" * 110)
    for c, j in ((c1, j1), (c2, j2)):
        before = orig[c, j]; after = examples[(atk, "none")][c, j]; fixed = examples[(atk, SHOW_LAYOUT)][c, j]
        print(f"\nclass {cn[c]}, clause row {j}, weight nibble {M['weights'][0][c, j]:+d}")
        print(f"  before attack        : [{lits(before)}]")
        changed = np.nonzero(before != after)[0]
        for i in changed:
            print(f"  MITM changes         : '{E.LIT_NAME[i]}'  {before[i]} -> {after[i]}")
        if len(changed) == 0:
            print(f"  MITM changes         : nothing (the literal already had that value)")
        print(f"  received, no ECC     : [{lits(after)}]")
        print(f"  after Hamming {SHOW_LAYOUT:<4}   : [{lits(fixed)}]  -> {'RESTORED' if np.array_equal(fixed, before) else 'NOT restored'}")
""")

md(r"""
## Why SEQ / INT give exactly the benchmark for 1–2 targeted literals
Hamming(7,4) repairs **one** wrong bit per codeword. A targeted literal sits at the same position in every rule, so the
question is whether two targeted literals share a codeword.
""")

code(r"""
pos = {"SEQ": np.arange(E.L), "INT": np.argsort(E.INT_ORDER)}
order = {"SEQ": np.arange(E.L), "INT": E.INT_ORDER}
for lay in ("SEQ", "INT"):
    print(f"{lay}:")
    for t_ in TARGETS[:4]:
        b = pos[lay][t_] // 4
        print(f"  {E.LIT_NAME[t_]:24s} -> codeword {b:3d}, shares it with {[str(E.LIT_NAME[i]) for i in order[lay][4*b:4*b+4] if i != t_]}")
    for k in K_VALUES:
        per = np.bincount(pos[lay][TARGETS[:k]] // 4)
        print(f"    top-{k}: codewords holding 2+ targeted literals = {int((per >= 2).sum())}")
print()
print(R[R.attack.isin(["A2", "A3", "A4"]) & R.layout.isin(["SEQ", "INT"])]
      [["attack", "K", "layout", "bits changed by MITM", "wrong bits after decode", "correct", "accuracy %"]].to_string(index=False))
""")

md(r"""
---
# Monte Carlo: 150 iterations with a NEW random literal set every iteration
In every iteration and for K = 2, 4, 6, 8, 10, 12, 16, 20 the attacker picks **K random literals**. The **same** literals are attacked in
**every clause of every class of all 3 TMs** (include / exclude / toggle). In the next iteration a **new random set**
is drawn. When 2 or more of the chosen literals fall into the **same Hamming codeword**, Hamming(7,4) cannot repair it
(and may even flip a third bit), so accuracy should drop in exactly those iterations.

The 14,400 runs (150 × 8 K × 3 attacks × 4 layouts) are computed by `mc_attacks.py` (5 parallel chunks, `mc_run_parallel.sh`) with an in-memory copy of the packet
encode → MITM → decode that was checked **bit-for-bit** against the real `.bin` files (`mc_fast.validate`). Results are
stored in `mc_results.csv`; set `RERUN_MC = True` to recompute (≈ 25-30 minutes).
""")

code(r"""
import subprocess, os
RERUN_MC = False
if RERUN_MC or not os.path.exists(f"{BASE}/mc_results.csv"):
    subprocess.run(["bash", f"{BASE}/mc_run_parallel.sh"], check=True, cwd=BASE)
MC = pd.read_csv(f"{BASE}/mc_results.csv").fillna({"shared codeword (SEQ)": "", "shared codeword (INT)": ""})
MC["collision"] = np.where(MC.layout == "SEQ", MC["shared codeword (SEQ)"] != "",
                  np.where(MC.layout == "INT", MC["shared codeword (INT)"] != "", False))
ATK_NAME = {"A2": "include", "A3": "exclude", "A4": "toggle"}
print(f"{MC.iteration.max()} iterations, {len(MC):,} runs, benchmark {BENCH:.4f} %")

stats = (MC.groupby(["K", "attack", "layout"], sort=False)["accuracy %"]
           .agg(mean="mean", variance="var", std="std", min="min", max="max").reset_index())
stats["attack"] = stats.attack.map(lambda a: f"{a} {ATK_NAME[a]}")
stats["layout"] = stats.layout.map({"none": "No ECC", "SEQ": "SEQ", "INT": "INT", "KEY": "KEY"})
coll = MC[MC.layout.isin(["SEQ", "INT"])].groupby(["K", "attack", "layout"]).collision.sum()
stats["iterations with shared codeword"] = [
    int(coll.get((k, a.split()[0], l), 0)) if l in ("SEQ", "INT") else "" for k, a, l in zip(stats.K, stats.attack, stats.layout)]
stats["mean Δ vs benchmark"] = stats["mean"] - BENCH
pd.set_option("display.float_format", "{:.4f}".format)
stats
""")

md("### Iterations where 2 or more chosen literals share one Hamming codeword (SEQ / INT)")

code(r"""
for lay in ("SEQ", "INT"):
    sub = MC[(MC.layout == lay) & MC.collision]
    print("=" * 120)
    print(f"{lay}: {sub[['iteration', 'K']].drop_duplicates().shape[0]} (iteration, K) cases with a shared codeword")
    print("=" * 120)
    if len(sub):
        wide = sub.pivot_table(index=["iteration", "K", "literals", f"shared codeword ({lay})"], columns="attack",
                               values="accuracy %", sort=False).reset_index()
        wide.columns = [c if c not in ATK_NAME else f"{c} {ATK_NAME[c]} %" for c in wide.columns]
        print(wide.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ok = MC[(MC.layout == lay) & ~MC.collision]
    print(f"\n  without shared codeword: accuracy min {ok['accuracy %'].min():.4f} %, max {ok['accuracy %'].max():.4f} % "
          f"(benchmark {BENCH:.4f} %)\n")
""")

md(r"""
### Steep drops: which iterations, how many bits inside one codeword
A point is a **steep drop** when its accuracy is clearly below the typical value of its own series:
below the series median by more than max(3 × robust std, 0.02 percentage points).
For every run we also count the Hamming codewords in which the MITM changed **1, 2, 3 or 4** bits
(1 bit is always repaired; 2+ bits cannot be repaired and may be mis-corrected).
""")

code(r"""
def steep_drops(d):
    acc = d["accuracy %"].to_numpy()
    med = np.median(acc)
    sig = 1.4826 * np.median(np.abs(acc - med))
    thr = max(3 * sig, 0.02)
    return d[acc < med - thr]

drops = []
for (atk, K, lay), d in MC.groupby(["attack", "K", "layout"], sort=False):
    drops.append(steep_drops(d))
DROPS = pd.concat(drops)
cols = ["iteration", "K", "attack", "layout", "accuracy %", "Δ vs benchmark", "max bits in one cw",
        "cw 2 bits", "cw 3 bits", "cw 4 bits", "bits changed by MITM", "literals", "shared codeword (SEQ)", "shared codeword (INT)"]
print(f"{len(DROPS)} steep-drop points out of {len(MC):,} runs\n")
print("steep drops per layout:", DROPS.layout.value_counts().to_dict())
DROPS.sort_values(["attack", "K", "layout", "accuracy %"])[cols]
""")

md(r"""
### Accuracy over the 150 iterations
Marker = largest number of bits the MITM changed inside one Hamming codeword in that run
(**· 1 bit → repaired**, **▲ 2 bits**, **■ 3 bits**, **◆ 4 bits**). Callouts label the deepest steep drops with the
iteration, accuracy, Δ vs benchmark and the number of codewords that had 2 / 3 / 4 bits changed.
Dashed black = benchmark, dotted = mean of the 150 iterations. Every panel has its own y-axis.
""")

code(r"""
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5,
    "axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "#000000", "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.color": "#000000", "ytick.color": "#000000", "axes.labelcolor": "#000000",
    "axes.grid": True, "axes.grid.axis": "both", "grid.color": "#d9d9d9", "grid.linewidth": 0.6, "grid.linestyle": "-",
    "legend.frameon": True, "legend.edgecolor": "#000000", "legend.fancybox": False, "legend.framealpha": 1.0,
    "figure.dpi": 100, "savefig.dpi": 200})
LCOL = {"none": "#d95319", "SEQ": "#edb120", "INT": "#77ac30", "KEY": "#0072bd"}        # MATLAB default colours
LNAME = {"none": "No ECC", "SEQ": "Hamming SEQ", "INT": "Hamming INT", "KEY": "Hamming KEY"}
BITSTYLE = {2: ("#ff7f0e", "^", "2 bits in one codeword"), 3: ("#d62728", "s", "3 bits in one codeword"),
            4: ("#7b1fa2", "D", "4 bits in one codeword")}
ACTION = {"A2": "included", "A3": "excluded", "A4": "toggled"}

def callout(d_row, lay):
    t = f"it {int(d_row.iteration)}: {d_row['accuracy %']:.4f} %\nΔ {d_row['Δ vs benchmark']:+.4f} pp"
    if lay == "none":
        return t + f"\n{int(d_row['bits changed by MITM']):,} bits {ACTION[d_row.attack]}"
    parts = [f"{int(d_row[f'cw {b} bits']):,}×{b}-bit" for b in (2, 3, 4) if d_row[f"cw {b} bits"] > 0]
    return t + "\ncw: " + ("\n    ".join(parts) if parts else "all 1-bit")

ALL_K = sorted(MC.K.unique())
K_BLOCKS = [ALL_K[i:i + 4] for i in range(0, len(ALL_K), 4)]            # 4 K values per figure (2-8, 10-20)
fig_no = 6
for atk in ["A2", "A3", "A4"]:
  for KS in K_BLOCKS:
    fig, axes = plt.subplots(len(KS), 4, figsize=(26, 5.2 * len(KS)), sharex=True, squeeze=False)
    for r, K in enumerate(KS):
        for c, lay in enumerate(E.LAYOUTS):
            ax = axes[r, c]
            d = MC[(MC.attack == atk) & (MC.K == K) & (MC.layout == lay)].sort_values("iteration")
            acc = d["accuracy %"].to_numpy()
            ax.plot(d.iteration, acc, color=LCOL[lay], lw=1.2, zorder=2, label=LNAME[lay])
            ax.plot(d.iteration, acc, ls="none", marker="o", ms=2.6, color=LCOL[lay], zorder=3)
            ax.axhline(BENCH, color="#000000", ls="--", lw=1.1, zorder=1, label=f"benchmark {BENCH:.4f} %")
            ax.axhline(acc.mean(), color=LCOL[lay], ls=":", lw=1.5, zorder=1, label=f"mean {acc.mean():.4f} %")
            if lay != "none":
                for b, (col, mk, lab) in BITSTYLE.items():
                    sel = d[d["max bits in one cw"] == b]
                    if len(sel):
                        ax.scatter(sel.iteration, sel["accuracy %"], marker=mk, s=46, color=col, edgecolor="black",
                                   linewidth=0.6, zorder=5, label=f"{lab} ({len(sel)}×)")
            dr = steep_drops(d).sort_values("accuracy %").head(3)
            if len(dr):
                ax.scatter(dr.iteration, dr["accuracy %"], s=150, facecolor="none", edgecolor="black", lw=1.4, zorder=6,
                           label="steep drop")
                lo, hi = min(acc.min(), BENCH), max(acc.max(), BENCH)
                for j, (_, row) in enumerate(dr.iterrows()):
                    x_txt = 0.19 + 0.31 * j
                    ax.annotate(callout(row, lay), xy=(row.iteration, row["accuracy %"]),
                                xytext=(x_txt, 0.30), textcoords="axes fraction", fontsize=7.6, ha="center", va="center",
                                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.6),
                                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8), zorder=7)
            lo, hi = min(acc.min(), BENCH), max(acc.max(), BENCH)
            pad = max((hi - lo) * 0.15, 0.01)
            ax.set_ylim(lo - 1.6 * pad, hi + pad)
            ax.set_xlim(0, d.iteration.max() + 1)
            ax.set_title(f"{LNAME[lay]}  ·  K = {K} literals {ACTION[atk]}", loc="left", fontweight="bold")
            ax.text(0.99, 0.97, f"mean {acc.mean():.4f}\nstd {acc.std(ddof=1):.4f}\nvar {acc.var(ddof=1):.4f}\n"
                                f"min {acc.min():.4f}", transform=ax.transAxes, ha="right", va="top", fontsize=8.8,
                    family="DejaVu Sans Mono", bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.6))
            ax.legend(loc="upper left", fontsize=8, ncol=1, borderaxespad=0.3)
            if c == 0:
                ax.set_ylabel("accuracy on 105,066 test flows (%)")
            if r == len(KS) - 1:
                ax.set_xlabel("iteration (new random literal set each time)")
    fig.suptitle(f"{atk}: {ACTION[atk]} K random literals (same set in every clause of all 3 TMs) · "
                 f"K = {KS[0]}-{KS[-1]} · {MC.iteration.max()} iterations", x=0.005, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))
    fig.savefig(f"{BASE}/fig{fig_no}_mc_{atk}_K{KS[0]}-{KS[-1]}.png"); plt.show()
    fig_no += 1
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "edge-iiot", "display_name": "Python (Edge-IIoT)", "language": "python"},
                   "language_info": {"name": "python"}}}
json.dump(nb, open("/home/<user>/medsciot/medsciot_results.ipynb", "w"), indent=1)
print("written medsciot_results.ipynb")
