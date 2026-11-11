"""Builds clause_attack_demo.ipynb
  Part A: ONE particular literal included / excluded / toggled in ALL clauses (printed values, scores, accuracy)
  Part B: different random literals in every clause
Everything goes through the real Hamming(7,4) .bin files (ecc_pipeline.py)."""
import json

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n")})


md(r"""
# Clause-bank attacks, shown value by value

Every clause is a row of 356 literal bits (**1 = include, 0 = exclude**). Each TM has 5 classes × 1000 clauses.

* **Part A** – the attacker picks **one particular literal** (e.g. `NOT RST Flag Cnt b0`) and changes **that same
  literal in all clauses** of all classes: force it to 1 (include), force it to 0 (exclude) or flip it (toggle).
* **Part B** – the attacker changes **different random literals in every clause**.

All runs use the real files: clause bank → packets → Hamming(7,4) → `.bin` → MITM edits bytes → model-blind decode.

**Run:** kernel **"Python (Edge-IIoTset)"** → Run All (~2 minutes).
""")

code(r"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import ecc_pipeline as E

# ---------------- SETTINGS ----------------
TARGET_LITERAL = "NOT RST Flag Cnt b0"   # Part A: the particular literal (any name from E.LIT_NAME)
K = 2                                    # Part B: random literals changed per clause
LAYOUT = "KEY"                           # Hamming grouping used in the showcase: "SEQ", "INT", "KEY"
ATTACKED_NODES = [0, 1, 2]               # TMs whose files are attacked
SHOW_TM, SHOW_CLASS = 0, 4               # example clauses come from this TM / class (4 = Reconnaissance)
# ------------------------------------------

M = E.load_nodes(); T = M["T"]; cn = E.class_names
original_pred = E.fuse(M["clean_sums"], T)        # deployed model (4-bit weights), no attack
lits = lambda row: ", ".join(str(E.LIT_NAME[i]) for i in np.nonzero(row)[0])
use = sum(c.sum(axis=(0, 1)).astype(np.int64) for c in M["cubes"])
TOP_LITERALS = [str(E.LIT_NAME[i]) for i in np.argsort(-use, kind="stable")[:8]]
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "axes.edgecolor": "#8a8984"})
STYLE = {"none": ("#eb6834", "o", "No protection"), "SEQ": ("#eda100", "D", "Hamming – SEQ"),
         "INT": ("#1baf7a", "^", "Hamming – INT"), "KEY": ("#2a78d6", "s", "Hamming – KEY")}
CLEAN_ACC = (original_pred == E.yte).mean() * 100      # benchmark: no attack


def bench(ax):
    ax.axhline(CLEAN_ACC, color="#0b0b0b", ls="--", lw=1.4, zorder=2, label=f"Benchmark (no attack) {CLEAN_ACC:.2f} %")


def show_drop(ax, xs, ys, color="#eb6834"):
    for x_, y_ in zip(xs, ys):
        ax.annotate(f"{y_:.2f}\n({y_ - CLEAN_ACC:+.2f})", (x_, y_), textcoords="offset points", xytext=(0, -26),
                    ha="center", fontsize=8.5, color=color)


def show_clause(j, before, after, fixed, weight, focus=None):
    print(f"\nClause {j}  (weight {weight:+d})")
    print(f"  before attack     : {before.sum()} included  [{lits(before)}]")
    for i in (np.nonzero(before != after)[0] if focus is None else focus):
        print(f"  literal '{E.LIT_NAME[i]}' : original {before[i]} -> after attack {after[i]} -> after Hamming {fixed[i]}"
              + ("" if before[i] != after[i] else "   (already that value: nothing to change)"))
    print(f"  received, no ECC  : {after.sum()} included  [{lits(after)}]")
    print(f"  after Hamming {LAYOUT:<4}: {fixed.sum()} included  [{lits(fixed)}]"
          f"   -> {'RESTORED' if np.array_equal(fixed, before) else 'NOT fully restored'}")


def fires(cube, x):
    lit = np.concatenate([x, 1 - x])
    return (~((cube == 1) & (lit[None, None, :] == 0)).any(axis=2)) & cube.any(axis=2)


def flow_effect(i, versions):
    x, table = E.Xte[i], []
    for label, banks in versions:
        total = np.zeros(E.NUM_CLASSES)
        for n, (cube, w) in enumerate(banks):
            s = E.tm_predict_manual(x, cube, w)
            total += np.clip(s, -T, T)
            table.append({"version": label, "TM": n, "clauses firing": int(fires(cube, x).sum()),
                          **{cn[c]: int(s[c]) for c in range(E.NUM_CLASSES)}, "prediction": ""})
        table.append({"version": label, "TM": "fused (±T capped, summed)", "clauses firing": "",
                      **{cn[c]: int(total[c]) for c in range(E.NUM_CLASSES)},
                      "prediction": cn[int(np.argmax(total))]})
    return pd.DataFrame(table)


banks = lambda rx: [(r["cube"], r["weights"]) for r in rx]
print("Most-used literals (attacker's choice for 1, 2, 4, 8 literals):")
for i, name in enumerate(TOP_LITERALS, 1):
    print(f"  {i}. {name}")
""")

# ======================================================================= PART A
md(r"""
---
# Part A · One particular literal changed in ALL clauses

| Attack | What happens to `TARGET_LITERAL` in each of the 15,000 clauses |
|---|---|
| **Include** | set to **1** everywhere. Clauses that already had it: no change. Others: the literal is added. |
| **Exclude** | set to **0** everywhere. Only clauses that had it are changed (it is removed). |
| **Toggle** | flipped everywhere: 0 → 1 in clauses that lacked it, 1 → 0 in clauses that had it. |
""")

code(r"""
tid = int(np.where(E.LIT_NAME == TARGET_LITERAL)[0][0])
ATTACKS_A = {"Include": "L_INC", "Exclude": "L_EXC", "Toggle": "L_TOG"}
recA, fixA = {}, {}
for name, c in ATTACKS_A.items():
    recA[name] = E.run_pipeline("none", c, [TARGET_LITERAL], attacked_nodes=ATTACKED_NODES, M=M)
    fixA[name] = E.run_pipeline(LAYOUT, c, [TARGET_LITERAL], attacked_nodes=ATTACKED_NODES, M=M)

orig = M["cubes"][SHOW_TM][SHOW_CLASS]
n_inc = orig.sum(axis=1)
small = (n_inc >= 3) & (n_inc <= 7)
EX_HAS = np.where(small & (orig[:, tid] == 1))[0][:2]       # clauses that include the target literal
EX_NOT = np.where(small & (orig[:, tid] == 0))[0][:2]       # clauses that do not
print(f"Target literal: '{TARGET_LITERAL}'  (literal #{tid})")
print(f"TM {SHOW_TM}, class '{cn[SHOW_CLASS]}': clauses {list(EX_HAS)} include it, clauses {list(EX_NOT)} do not")
for name in ATTACKS_A:
    print("\n" + "=" * 100)
    print(f"  {name.upper()} '{TARGET_LITERAL}' in ALL clauses")
    print("=" * 100)
    for j in list(EX_HAS) + list(EX_NOT):
        show_clause(j, orig[j], recA[name][SHOW_TM]["cube"][SHOW_CLASS, j],
                    fixA[name][SHOW_TM]["cube"][SHOW_CLASS, j], M["weights"][SHOW_TM][SHOW_CLASS, j], focus=[tid])
""")

md("### How many clauses changed (all classes of each attacked TM)")
code(r"""
rows = []
for name in ATTACKS_A:
    for n in ATTACKED_NODES:
        before = M["cubes"][n][:, :, tid]
        after = recA[name][n]["cube"][:, :, tid]
        rows.append({"attack": f"{name} '{TARGET_LITERAL}'", "TM": n,
                     "clauses that had it (=1)": int(before.sum()), "clauses 0 -> 1": int(((before == 0) & (after == 1)).sum()),
                     "clauses 1 -> 0": int(((before == 1) & (after == 0)).sum()),
                     f"wrong bits after Hamming {LAYOUT}": fixA[name][n]["lit_mismatch"]})
pd.DataFrame(rows)
""")

md(r"""
### Effect on one network flow
Score of a class = Σ weight × (clause fires). Final decision = class with the highest sum of the three ±T-capped TM scores.
""")
code(r"""
pred_inc = E.fuse([E.class_sums_all(c, w) for c, w in banks(recA["Include"])], T)
cand = np.where((pred_inc != original_pred) & (original_pred == E.yte))[0]
flow = int(cand[0]) if len(cand) else 0
print(f"Flow #{flow}: actual = {cn[E.yte[flow]]}, original prediction = {cn[original_pred[flow]]}, "
      f"its literal '{TARGET_LITERAL}' is {'TRUE' if np.concatenate([E.Xte[flow], 1 - E.Xte[flow]])[tid] else 'FALSE'}")
versions = [("original", [(M["cubes"][n], M["weights"][n]) for n in range(3)])]
for name in ATTACKS_A:
    versions += [(f"{name}, no ECC", banks(recA[name])), (f"{name}, Hamming {LAYOUT}", banks(fixA[name]))]
pd.set_option("display.max_rows", 200)
flow_effect(flow, versions)
""")

md("### Accuracy on all 105,066 test flows: 1, 2, 4, 8 particular literals (most-used first)")
code(r"""
rows = []
for name, c in ATTACKS_A.items():
    for k in (1, 2, 4, 8):
        for lay in E.LAYOUTS:
            rx = E.run_pipeline(lay, c, TOP_LITERALS[:k], attacked_nodes=ATTACKED_NODES, M=M)
            pred = E.fuse([E.class_sums_all(cu, w) for cu, w in banks(rx)], T)
            rows.append({"attack": name, "literals": k, "layout": lay,
                         "bits changed": sum(r["bits_changed"] for r in rx),
                         "wrong bits after decode": sum(r["lit_mismatch"] for r in rx),
                         "correct flows": int((pred == E.yte).sum()), "accuracy %": (pred == E.yte).mean() * 100,
                         "same as original %": (pred == original_pred).mean() * 100})
resA = pd.DataFrame(rows)
resA.to_csv(f"{E.BASE}/particular_literal_results.csv", index=False)
pA = resA.pivot_table(index=["attack", "literals"], columns="layout", values="accuracy %", sort=False)[E.LAYOUTS]
pA.insert(0, "benchmark", round(CLEAN_ACC, 2)); pA["Δ No ECC"] = (pA["none"] - CLEAN_ACC).round(2); pA["Δ KEY"] = (pA["KEY"] - CLEAN_ACC).round(2)
pA
""")

code(r"""
import sys; sys.path.insert(0, E.BASE)
import plot_style as P
P.setup()
ticks = [f"1 literal\n({TOP_LITERALS[0]})", "2 literals\n(+ 1 more)", "4 literals\n(+ 3 more)", "8 literals\n(+ 7 more)"]
fig, axes = P.figure(3, 4)
for ax, name in zip(axes, ATTACKS_A):
    series = [(lab, list(resA[(resA.attack == name) & (resA.layout == s)]["accuracy %"]), c) for s, lab, c in P.SCHEMES]
    P.grouped_bars(ax, ticks, series, bench=CLEAN_ACC)
    ax.set_title(f"{name.upper()} the same literal(s) in ALL clauses  ·  TMs attacked: {ATTACKED_NODES}", pad=6)
P.top_legend(fig, axes[0])
fig.savefig(f"{E.BASE}/fig8_particular_literal.png", dpi=150); plt.show()
""")

# ======================================================================= PART B
md(r"""
---
# Part B · Different random literals in every clause

| Attack | In every clause, the attacker… |
|---|---|
| **Include K** | picks K literals that are currently 0 and sets them to 1 (different ones in each clause) |
| **Exclude K** | picks K literals that are currently 1 and sets them to 0 |
| **Toggle K** | picks K random literals and flips them |
""")

code(r"""
ATTACKS_B = {"Include": "INC", "Exclude": "EXC", "Toggle": "A1"}
recB, fixB = {}, {}
for name, c in ATTACKS_B.items():
    recB[name] = E.run_pipeline("none", c, k=K, attacked_nodes=ATTACKED_NODES, M=M)
    fixB[name] = E.run_pipeline(LAYOUT, c, k=K, attacked_nodes=ATTACKED_NODES, M=M)
EX = np.where(small)[0][:2]
for name in ATTACKS_B:
    print("\n" + "=" * 100)
    print(f"  {name.upper()} {K} random literals — different in each clause")
    print("=" * 100)
    for j in EX:
        show_clause(j, orig[j], recB[name][SHOW_TM]["cube"][SHOW_CLASS, j],
                    fixB[name][SHOW_TM]["cube"][SHOW_CLASS, j], M["weights"][SHOW_TM][SHOW_CLASS, j])
""")

code(r"""
rows = []
for name, c in ATTACKS_B.items():
    for k in (1, 2, 4, 8):
        for lay in E.LAYOUTS:
            rx = E.run_pipeline(lay, c, k=k, attacked_nodes=ATTACKED_NODES, M=M)
            pred = E.fuse([E.class_sums_all(cu, w) for cu, w in banks(rx)], T)
            rows.append({"attack": name, "K": k, "layout": lay,
                         "bits changed": sum(r["bits_changed"] for r in rx),
                         "wrong bits after decode": sum(r["lit_mismatch"] for r in rx),
                         "correct flows": int((pred == E.yte).sum()), "accuracy %": (pred == E.yte).mean() * 100,
                         "same as original %": (pred == original_pred).mean() * 100})
resB = pd.DataFrame(rows)
resB.to_csv(f"{E.BASE}/clause_attack_results.csv", index=False)

fig, axes = P.figure(3, 4)
for ax, name in zip(axes, ATTACKS_B):
    series = [(lab, list(resB[(resB.attack == name) & (resB.layout == s)]["accuracy %"]), c) for s, lab, c in P.SCHEMES]
    P.grouped_bars(ax, [f"K = {k}" for k in (1, 2, 4, 8)], series, bench=CLEAN_ACC)
    ax.set_title(f"{name.upper()} K random literals in every clause (different in each clause)  ·  TMs attacked: {ATTACKED_NODES}", pad=6)
P.top_legend(fig, axes[0])
fig.savefig(f"{E.BASE}/fig7_per_clause_attacks.png", dpi=150); plt.show()
pB = resB.pivot_table(index=["attack", "K"], columns="layout", values="accuracy %", sort=False)[E.LAYOUTS]
pB.insert(0, "benchmark", round(CLEAN_ACC, 2)); pB["Δ No ECC"] = (pB["none"] - CLEAN_ACC).round(2); pB["Δ KEY"] = (pB["KEY"] - CLEAN_ACC).round(2)
pB
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "proj", "display_name": "Python (Edge-IIoTset)", "language": "python"},
                   "language_info": {"name": "python"}}}
json.dump(nb, open("/home/<user>/medsciot/clause_attack_demo.ipynb", "w"), indent=1)
print("written clause_attack_demo.ipynb")
