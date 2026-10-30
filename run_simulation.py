"""Synthetic check of the theory: support recovery & direction under label-flip contamination.
Design: x ~ N(0, Sigma_ij = 0.3^|i-j|), p=300, s=8 active coefs (+-1), n=1000, logistic truth.
Contamination: 'random' = flip each label w.p. eps (independent of x);
               'targeted' = flip the labels of the eps*n points with the largest |margin| (worst case for logistic loss).
"""
import sys, os, numpy as np, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from robusttox.methods import L1Logit, TrimLogit, AdaTrim, StabilitySelection, RobustTox
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30; n, p, s = 1000, 300, 8
os.makedirs(f"{ROOT}/results", exist_ok=True); out = open(f"{ROOT}/results/simulation.csv", "w"); w = csv.writer(out)
w.writerow(["contam", "eps", "rep", "method", "f1", "exact", "cos", "tpr", "fdr", "n_sel", "flips_removed_frac"])
idx = np.arange(p); L = np.linalg.cholesky(0.3 ** np.abs(idx[:, None] - idx[None, :]))
for contam in ("random", "targeted"):
    for eps in (0.0, 0.1, 0.2, 0.3, 0.4):
        for rep in range(REPS):
            rng = np.random.RandomState(rep * 100 + int(eps * 10)); X = rng.randn(n, p) @ L.T
            beta = np.zeros(p); S = rng.choice(p, s, replace=False); beta[S] = rng.choice([-1, 1], s) * 1.0
            z = X @ beta; y = (rng.rand(n) < 1 / (1 + np.exp(-z))).astype(int); flip = np.zeros(n, bool)
            if contam == "random": flip = rng.rand(n) < eps
            else: flip[np.argsort(-np.abs(z))[: int(eps * n)]] = True
            yc = y.copy(); yc[flip] = 1 - yc[flip]
            hh = max(0.5, 1 - eps - 0.02)  # trimming level that satisfies h <= 1 - eps (oracle-informed)
            models = {"L1-Logit": L1Logit(seed=rep), "TrimLogit h=0.9": TrimLogit(h=0.9, seed=rep), "TrimLogit h=1-eps (oracle)": TrimLogit(h=hh, seed=rep),
                      "AdaTrim": AdaTrim(seed=rep), "StabSel": StabilitySelection(seed=rep), "RobustTox": RobustTox(seed=rep)}
            for name, m in models.items():
                m.fit(X, yc); sel = np.zeros(p, bool); sel[np.asarray(m.support_)] = True
                true = beta != 0; tp = (sel & true).sum(); fp = (sel & ~true).sum(); fn = (~sel & true).sum()
                f1 = 2 * tp / max(2 * tp + fp + fn, 1); coef = np.zeros(p); c = m.model_.coef_[0]
                if hasattr(m, "cols_"): coef[m.cols_] = c
                else: coef = c
                cos = coef @ beta / (np.linalg.norm(coef) * np.linalg.norm(beta) + 1e-12)
                kept = getattr(m, "kept_", np.arange(n)); rem = np.setdiff1d(np.arange(n), kept)
                fr = flip[rem].sum() / max(flip.sum(), 1) if flip.sum() else np.nan
                w.writerow([contam, eps, rep, name, f1, int(tp == s and fp == 0), cos, tp / s, fp / max(sel.sum(), 1), sel.sum(), fr]); out.flush()
        print(contam, eps, "done", flush=True)
