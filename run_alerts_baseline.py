"""Alerts-only baseline (binary Brenk/PAINS/NIH flags -> L1-logistic), same splits & contamination as run_benchmark.py."""
import sys, os, csv, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from sklearn.metrics import roc_auc_score, average_precision_score
from robusttox import prep, scaffold_split
from robusttox.methods import L1Logit
NOISE = [("none", 0.0), ("random", 0.1), ("random", 0.2), ("random", 0.3), ("margin", 0.1), ("margin", 0.2)]
out = csv.writer(open(f"{ROOT}/results/alerts_only.csv", "w")); out.writerow(["ds","task","seed","kind","noise","method","auc","prauc","nnz"])
for ds in ("tox21", "clintox", "sider", "toxcast_hep"):
    z = np.load(f"{ROOT}/data/{ds}.npz", allow_pickle=True)
    for task in z["tasks"]:
        y = z["y_" + str(task)]; ok = ~np.isnan(y); X, A, sc, y = z["X"][ok], z["A"][ok].astype(float), z["scaf"][ok], y[ok].astype(int)
        for seed in range(3):
            tr, te = scaffold_split(sc, seed)
            if min(y[te].mean(), y[tr].mean()) in (0, 1) or y[te].sum() < 5: continue
            Xtr, Xte, _ = prep(X[tr], X[te]); ref = L1Logit(seed=seed).fit(Xtr, y[tr]); margin = np.abs(ref.decision_function(Xtr))
            for kind, noise in NOISE:
                rng = np.random.RandomState(1000 + seed); ytr = y[tr].copy()
                if kind == "random": flip = rng.rand(len(ytr)) < noise
                elif kind == "margin": flip = np.zeros(len(ytr), bool); flip[np.argsort(-margin)[: int(noise * len(ytr))]] = True
                else: flip = np.zeros(len(ytr), bool)
                ytr[flip] = 1 - ytr[flip]
                m = L1Logit(seed=seed, Cs=(0.03, 0.1, 0.3, 1.0)).fit(A[tr], ytr); p = m.decision_function(A[te])
                out.writerow([ds, task, seed, kind, noise, "Alerts-only", roc_auc_score(y[te], p), average_precision_score(y[te], p), len(m.support_)])
print("done")
