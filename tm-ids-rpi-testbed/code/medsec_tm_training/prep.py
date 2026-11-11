"""MedSec-25: clean, dedupe, 80/20 stratified train/test split, quantile-thermometer booleanization."""
import numpy as np
import pandas as pd

RAW = "/home/<user>/new_medsec/raw/MedSec-25.csv"
OUT = "/home/<user>/new_medsec/data.npz"
BITS_PER_FEATURE = 8           # max thermometer thresholds per raw feature
TEST_FRAC = 0.2
CAP_PER_CLASS = 24000          # undersample majority classes for training
SEED = 2026

rng = np.random.default_rng(SEED)

df = pd.read_csv(RAW, low_memory=False)
df = df.drop(columns=["Flow ID", "Src IP", "Dst IP", "Timestamp", "Src Port"])
y_str = df.pop("Label")
X = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
keep = X.notna().all(axis=1)
X, y_str = X[keep], y_str[keep]
X = X.loc[:, X.nunique() > 1]

# Remove exact duplicates so train/test do not share identical rows
d = pd.concat([X, y_str], axis=1).drop_duplicates()
y_str = d.pop("Label")
X = d
class_names = np.array(sorted(y_str.unique()))
y = np.searchsorted(class_names, y_str.values).astype(np.uint32)
feat_names = np.array(X.columns)
Xv = X.values.astype(np.float64)
print("rows", len(y), "features", Xv.shape[1], "classes", list(class_names))

# Stratified train/test split
test_mask = np.zeros(len(y), bool)
for c in range(len(class_names)):
    idx = np.where(y == c)[0]
    rng.shuffle(idx)
    test_mask[idx[: int(round(TEST_FRAC * len(idx)))]] = True
Xtr, ytr, Xte, yte = Xv[~test_mask], y[~test_mask], Xv[test_mask], y[test_mask]

# Quantile thermometer thresholds fitted on training data only
thresholds = []
for j in range(Xtr.shape[1]):
    q = np.quantile(Xtr[:, j], np.arange(1, BITS_PER_FEATURE + 1) / (BITS_PER_FEATURE + 1))
    t = np.unique(q)
    t = t[t < Xtr[:, j].max()]
    if len(t) == 0:
        t = np.array([np.median(Xtr[:, j])])
    thresholds.append(t)


def booleanize(A):
    return np.concatenate([(A[:, [j]] > thresholds[j][None, :]) for j in range(A.shape[1])],
                          axis=1).astype(np.uint32)


Btr, Bte = booleanize(Xtr), booleanize(Xte)
print("boolean features o =", Btr.shape[1], "-> literals 2o =", 2 * Btr.shape[1])

# Per-class undersampling of the training split
sel = np.concatenate([rng.permutation(np.where(ytr == c)[0])[:CAP_PER_CLASS] for c in range(len(class_names))])
sel = rng.permutation(sel)
print("train", np.bincount(ytr[sel], minlength=len(class_names)))
print("test ", np.bincount(yte, minlength=len(class_names)))

np.savez_compressed(OUT, Xtr=Btr[sel], ytr=ytr[sel], Xte=Bte, yte=yte,
                    class_names=class_names, feat_names=feat_names)
