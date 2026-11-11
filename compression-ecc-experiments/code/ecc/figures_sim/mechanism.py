"""Why a single TM's class sums and accuracy go to 0 / the class-0 share as K grows (MedSec version of
mnist/single_tm_k_mechanism.ipynb). The SAME K random literals are forced into every clause of every class
(include / toggle); 5 trials of nested literal sets (K uses the first K of the trial's list).

Chain, measured on the whole test set:
  (1) input -> literals : share of samples with at least one forced literal = 0 (computed from the inputs only)
  (2) literals -> clauses: clauses firing (observed) vs predicted = fired before AND all K forced literals = 1
  (3) clauses -> sums    : true-class sum, best wrong-class sum, mean |class sum|  -> all 0
  (4) sums -> accuracy   : all sums 0 -> argmax = class 0 -> accuracy = share of class 0 in the test set

usage: SPLIT=80_20 python mechanism.py {medsec|iot|mnist}
  -> split_<SPLIT>/<ds>/mechanism/mechanism_<ds>.pdf, fig_mechanism_<ds>.png, fig_mechanism_samples_<ds>.png,
     mechanism_<ds>.csv, mechanism_<ds>.mat
"""
import os
import sys
import textwrap
import numpy as np
import pandas as pd
import scipy.io
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numba import njit, prange
import ecc_common as E

DS = sys.argv[1]
OUT = f"{E.ROOT}/{DS}/mechanism"
os.makedirs(OUT, exist_ok=True)
K_VALUES = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 20]
ATTACKS = ["include", "toggle"]
N_TRIALS = 5
CLASS_NAMES = {"medsec": ["Benign", "Exfiltration", "Initial access", "Lateral movement", "Reconnaissance"],
               "iot": ["Normal", "DoS/DDoS", "Information Gathering", "MITM", "Injection", "Malware"],
               "mnist": [str(c) for c in range(10)]}[DS]

m = E.load(DS)
X, y, cube0, W = m["Xte"], m["yte"], m["cube"], m["weights"]
N, (C, CL, L) = len(y), cube0.shape
W64 = -(-L // 64)
LIT = np.concatenate([X, 1 - X], 1).astype(bool)                       # value of every literal for every sample
Xw = E.pack_rows(LIT.astype(np.uint8), W64)
cls = np.repeat(np.arange(C), CL)
wt = W.reshape(-1).astype(np.int64)


@njit(parallel=True, cache=True)
def _eval(Xw, Cw, nonempty, wt, cls, n_cls):
    """Class sums (N, C) and number of firing clauses per sample."""
    n = Xw.shape[0]
    sums = np.zeros((n, n_cls), np.int64)
    fire = np.zeros(n, np.int64)
    for i in prange(n):
        for j in range(Cw.shape[0]):
            if not nonempty[j]:
                continue
            ok = True
            for w in range(Cw.shape[1]):
                if Cw[j, w] & ~Xw[i, w]:
                    ok = False
                    break
            if ok:
                sums[i, cls[j]] += wt[j]
                fire[i] += 1
    return sums, fire


def evaluate(cube):
    flat = cube.reshape(C * CL, L)
    return _eval(Xw, E.pack_rows(flat, W64), flat.any(1), wt, cls, C)


SUMS0, FIRE0 = evaluate(cube0)
assert m.get("sums_ref") is None or np.array_equal(SUMS0, m["sums_ref"])
PRED0 = SUMS0.argmax(1)
CLASS0_SHARE = (y == 0).mean() * 100
rng = np.random.default_rng(11)
TRIAL_LITS = [rng.choice(L, max(K_VALUES), replace=False) for _ in range(N_TRIALS)]

# two example samples, both correctly classified and NOT class 0 (so the collapse to class 0 is a visible error):
# the first sample of the largest class, and the sample (any other class) whose first 0-literal comes latest
first_zero = np.argmax(~LIT[:, TRIAL_LITS[0]], axis=1).astype(float)
first_zero[LIT[:, TRIAL_LITS[0]].all(1)] = np.inf
ok = (PRED0 == y) & (y != 0)
big = np.bincount(y[y != 0], minlength=C).argmax()
s1 = int(np.flatnonzero(ok & (y == big))[0])
late = np.where(ok & (y != big) & np.isfinite(first_zero), first_zero, -1)
SAMPLES = [s1, int(np.argmax(late))]

rows, sample_sums = [], {}
idx = np.arange(N)
for tr in range(N_TRIALS):
    for mode in ATTACKS:
        for k in K_VALUES:
            lits = TRIAL_LITS[tr][:k]
            cube = cube0.copy()
            if mode == "include":
                cube[:, :, lits] = 1
            else:
                cube[:, :, lits] ^= 1
            sums, fire = evaluate(cube)
            all_true = LIT[:, lits].all(1) if k else np.ones(N, bool)
            wrong = sums.copy(); wrong[idx, y] = np.iinfo(np.int64).min
            rows.append({"trial": tr, "attack": mode, "K": k, "literals": " ".join(map(str, lits)),
                         "accuracy %": (sums.argmax(1) == y).mean() * 100,
                         "samples with a forced literal = 0 %": (~all_true).mean() * 100,
                         "all class sums = 0 %": (sums == 0).all(1).mean() * 100,
                         "clauses firing % (observed)": fire.mean() / (C * CL) * 100,
                         "clauses firing % (predicted)": (FIRE0 * all_true).mean() / (C * CL) * 100,
                         "true-class sum": sums[idx, y].mean(), "best wrong-class sum": wrong.max(1).mean(),
                         "mean |class sum|": np.abs(sums).mean()})
            if tr == 0 and mode == "include":
                sample_sums[k] = (sums[SAMPLES], LIT[np.ix_(SAMPLES, lits)])
    print(f"[{DS}] trial {tr + 1}/{N_TRIALS} done", flush=True)
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/mechanism_{DS}.csv", index=False, float_format="%.10f")
G = R.drop(columns=["literals"]).groupby(["attack", "K"])
MEAN, STD = G.mean(), G.std(ddof=1)

# ---------------------------------------------------------------- figure 1: the 4-link proof
ks = K_VALUES
STY = {"include": dict(color="C1", marker="o"), "toggle": dict(color="C0", marker="s", ls="--")}


def band(ax, metric, a, label, **kw):
    mu = MEAN.loc[a][metric].reindex(ks).to_numpy(); sd = STD.loc[a][metric].reindex(ks).to_numpy()
    lo, hi = mu - sd, mu + sd
    if "%" in metric:
        lo, hi = np.clip(lo, 0, 100), np.clip(hi, 0, 100)
    ax.plot(ks, mu, label=label, **kw)
    ax.fill_between(ks, lo, hi, color=kw.get("color"), alpha=0.13, lw=0)


name = E.NAME[DS]
fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
ax = axes[0, 0]
band(ax, "samples with a forced literal = 0 %", "include", "≥1 forced literal is 0 (from inputs only)", color="C3",
     marker="^")
for a in ATTACKS:
    band(ax, "all class sums = 0 %", a, f"all class sums = 0 ({a})", **STY[a])
ax.set(title="① input → literals: some forced literal is 0  ⇒  every class sum is 0", ylabel=f"% of {E.UNIT[DS]}",
       ylim=(-3, 103))
ax = axes[0, 1]
band(ax, "clauses firing % (predicted)", "include", "predicted: fired before AND all K forced literals = 1",
     color="k", ls=":", lw=3)
for a in ATTACKS:
    band(ax, "clauses firing % (observed)", a, f"observed ({a})", **STY[a])
ax.set(title="② literals → clauses: clauses stop firing", ylabel="% of clauses firing (average per sample)")
ax = axes[1, 0]
for a, ls in zip(ATTACKS, ["-", "--"]):
    band(ax, "true-class sum", a, f"true-class sum ({a})", color="C2", ls=ls, marker="o")
    band(ax, "best wrong-class sum", a, f"best wrong-class sum ({a})", color="C4", ls=ls, marker="s")
    band(ax, "mean |class sum|", a, f"mean |class sum| ({a})", color="C7", ls=ls, marker=".")
ax.axhline(0, color="k", lw=0.8)
ax.set(title="③ clauses → class sums: all class sums collapse to 0", ylabel="class sum (average over test samples)")
ax = axes[1, 1]
for a in ATTACKS:
    band(ax, "accuracy %", a, f"accuracy ({a})", **STY[a])
ax.axhline(CLASS0_SHARE, color="gray", ls=":", label=f"share of class 0 ({CLASS_NAMES[0]}) = {CLASS0_SHARE:.2f} %")
ax.set(title=f"④ class sums → accuracy: argmax of {C} zeros = class 0", ylabel="test accuracy (%)", ylim=(0, 100))
for a_ in axes.ravel():
    a_.set_xticks(ks); a_.set_xlabel("K (forced literals in every clause)"); a_.grid(alpha=0.3)
    a_.legend(fontsize=8.5)
fig.suptitle(f"{name}, single TM: why accuracy → {CLASS0_SHARE:.2f} % as K grows "
             f"(mean ± std over {N_TRIALS} trials of nested random literals, same literals in every clause)",
             fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_mechanism_{DS}.png", dpi=150)
FIG1 = fig

# ---------------------------------------------------------------- figure 2: single samples
show_k = [0, 1, 2, 4, 6, 8, 10, 20]
fig, axes = plt.subplots(len(SAMPLES), len(show_k), figsize=(21, 3.8 * len(SAMPLES)), sharey="row")
names = lambda lits: [str(m["lit_name"][l]) for l in lits]
for r, smp in enumerate(SAMPLES):
    true = int(y[smp])
    lim = np.abs(sample_sums[0][0][r]).max() * 1.15
    for ax, k in zip(axes[r], show_k):
        s, lt = sample_sums[k][0][r], sample_sums[k][1][r]
        ax.bar(range(C), s, color=["C2" if c == true else "C7" for c in range(C)])
        ax.axhline(0, color="k", lw=0.8)
        zeros = [n for n, v in zip(names(TRIAL_LITS[0][:k]), lt) if not v]
        ttl = f"K={k}: pred {CLASS_NAMES[int(s.argmax())]}\n" + ("all forced literals = 1" if not zeros else
                                                              "0-literal: " + textwrap.shorten(", ".join(zeros), 34))
        ax.set_title(ttl, fontsize=8.5, color="C3" if zeros else "k")
        ax.set(xticks=range(C), ylim=(-lim, lim))
        ax.set_xticklabels([n[:6] for n in CLASS_NAMES], rotation=60, fontsize=7.5)
    axes[r, 0].set_ylabel(f"sample #{smp}\n(true: {CLASS_NAMES[true]})\nclass sum")
fig.suptitle(f"{name}: class sums of single test samples as forced literals are added (include, trial 0). "
             "Red title = a forced literal is 0 for this sample → all class sums 0", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_mechanism_samples_{DS}.png", dpi=150)
FIG2 = fig

# ---------------------------------------------------------------- PDF
inc = MEAN.loc["include"]
k_half = next((k for k in ks if inc.loc[k, "accuracy %"] < (inc.loc[0, "accuracy %"] + CLASS0_SHARE) / 2), ks[-1])
k_zero = next((k for k in ks if inc.loc[k, "all class sums = 0 %"] > 99), None)
text = f"""
{name}: why accuracy collapses when the same K literals are forced into every clause

Model: {E.INFO[DS]}.
Benchmark (K = 0): {inc.loc[0, 'accuracy %']:.4f} %.  Class 0 = {CLASS_NAMES[0]}, {CLASS0_SHARE:.2f} % of the test set.
Experiment: {N_TRIALS} trials; in each trial a random order of literals is drawn and for K = {', '.join(map(str, ks))}
the first K literals of that order are forced into EVERY clause of EVERY class (include = set to 1, toggle = flip).

The argument, one link at a time (Figure 1):

1. Input -> literals.  A clause outputs 1 only if every literal it includes is 1 for the input. After the attack every
   clause includes the K forced literals, so if ANY forced literal is 0 for a sample, every clause of every class
   outputs 0 for that sample. Panel 1: the share of samples with at least one forced literal = 0 is computed from the
   inputs alone, and it coincides with the share of samples whose class sums are all 0 (include).

2. Literals -> clauses.  For the remaining samples (all forced literals = 1) the forced literals are harmless, so a
   clause fires exactly when it fired before. Prediction: firing = clean firing x [all K forced literals = 1].
   Panel 2: the observed firing rate follows this prediction and falls to 0 as K grows. (Observed is a little
   above predicted at small K: clauses that were EMPTY before - empty clauses never fire - now contain only the
   forced literals and fire for samples where those literals are all 1.)

3. Clauses -> class sums.  A class sum is the weighted vote of the firing clauses. With no clause firing, the
   true-class sum, the best wrong-class sum and |class sum| all go to 0 (Panel 3).

4. Class sums -> accuracy.  When all {C} class sums are 0, argmax returns the first class (class 0 = {CLASS_NAMES[0]}),
   so accuracy tends to the share of class 0 in the test set, {CLASS0_SHARE:.2f} % (Panel 4).

Per sample it is a cliff, not a slope (Figure 2): while all forced literals are 1 the class sums hardly change; the
moment one forced literal is 0 for that sample, all class sums drop to 0 and the prediction becomes class 0.
The average curve declines gradually only because different samples hit their first 0-literal at different K.

Key numbers (include, mean over {N_TRIALS} trials):
   accuracy is halfway to the floor by K = {k_half};
   {'all class sums are 0 for >99 % of samples from K = ' + str(k_zero) if k_zero is not None else 'all class sums reach 0 for most samples by K = ' + str(ks[-1])}.
Toggle behaves the same: a forced literal that was excluded becomes included (as in include); one that was included
becomes excluded, which can only make a clause fire more, but forced literals are rarely already included.
"""
table = MEAN.reset_index()[["attack", "K", "accuracy %", "samples with a forced literal = 0 %", "all class sums = 0 %",
                            "clauses firing % (observed)", "clauses firing % (predicted)", "true-class sum",
                            "best wrong-class sum", "mean |class sum|"]]
with PdfPages(f"{OUT}/mechanism_{DS}.pdf") as pdf:
    f = plt.figure(figsize=(11.7, 8.3))
    f.text(0.05, 0.96, text, va="top", ha="left", fontsize=10.5, family="DejaVu Sans Mono")
    pdf.savefig(f); plt.close(f)
    pdf.savefig(FIG1)
    pdf.savefig(FIG2)
    for a in ATTACKS:
        f, ax = plt.subplots(figsize=(11.7, 8.3)); ax.axis("off")
        t = table[table.attack == a].drop(columns="attack")
        cells = [[str(int(r.K))] + [f"{v:.4f}" for v in r.iloc[1:]] for _, r in t.iterrows()]
        tb = ax.table(cellText=cells, colLabels=[textwrap.fill(c, 13) for c in t.columns], loc="center",
                      cellLoc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.9)
        for c_ in range(len(t.columns)):
            tb[0, c_].set_height(tb[0, c_].get_height() * 2.2)
        ax.set_title(f"{name}: mean over {N_TRIALS} trials, attack = {a} (exact values)", fontsize=12)
        pdf.savefig(f); plt.close(f)
plt.close("all")

scipy.io.savemat(f"{OUT}/mechanism_{DS}.mat", {
    "dataset": DS, "K": np.array(ks, float), "attacks": np.array(ATTACKS, dtype=object),
    "class0_share_pct": CLASS0_SHARE, "dims": "arrays are (attack, K): mean over trials; *_std = std",
    **{c.replace(" ", "_").replace("%", "pct").replace("|", "abs").replace("(", "").replace(")", "")
       .replace("-", "_").replace("=", "eq").replace(">", "ge"):
       np.array([[MEAN.loc[a].loc[k, c] for k in ks] for a in ATTACKS]) for c in table.columns[2:]},
    **{(c.replace(" ", "_").replace("%", "pct").replace("|", "abs").replace("(", "").replace(")", "")
        .replace("-", "_").replace("=", "eq").replace(">", "ge") + "_std"):
       np.array([[STD.loc[a].loc[k, c] for k in ks] for a in ATTACKS]) for c in table.columns[2:]}},
    do_compression=True)
print(f"[{DS}] wrote {OUT}/mechanism_{DS}.pdf and figures; benchmark {inc.loc[0, 'accuracy %']:.4f} %, "
      f"class-0 share {CLASS0_SHARE:.2f} %, samples {SAMPLES}")
