import sys, numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier
Cl, T, S = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
D = np.load("data.npz", allow_pickle=True); idx = D["node0"]
X, y = D["Xtr"][idx], D["ytr"][idx]
n = int(0.8 * len(y)); tm = TMClassifier(Cl, T, S, platform="CPU", weighted_clauses=True, seed=7)
best = 0
for ep in range(15):
    tm.fit(X[:n], y[:n]); p = tm.predict(X[n:])
    bacc = np.mean([(p[y[n:] == c] == c).mean() for c in range(5)]); best = max(best, bacc)
print(f"Cl={Cl} T={T} s={S}  final balanced-acc={bacc:.4f} best={best:.4f}", flush=True)
