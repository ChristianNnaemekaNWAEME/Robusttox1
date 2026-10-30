"""Estimators.  Every class has fit(X, y) / decision_function(X) / support_ (indices of nonzero coefs).

L1Logit            L1-logistic regression, penalty chosen by 3-fold CV AUC.
StratTrimLinear    The J-5 estimator: sparse-LTS concentration steps on the 0/1 label
                   (linear residuals), class-stratified, then L1-logistic on kept records.
TrimLogit          FIX 1. Concentration steps with the *logistic* loss: fit L1-logistic,
                   drop the h-fraction-complement of highest-loss records per class, refit.
AdaTrim            FIX 2. TrimLogit with the trimming level h chosen by cross-validation.
StabilitySelection FIX 3. Repeated half-subsample L1 fits; keep features selected with
                   frequency >= pi; refit a ridge logistic model on the stable set.
RobustTox          Combination: AdaTrim -> StabilitySelection on the kept records.
"""
import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

CS = (0.003, 0.01, 0.03)

def _l1(C, seed=0):
    return LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=200, random_state=seed)

def cv_C(X, y, Cs=CS, seed=0, folds=3):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed); best = (-1, Cs[0])
    for C in Cs:
        sc = []
        for a, b in skf.split(X, y):
            if y[a].min() == y[a].max() or y[b].min() == y[b].max():
                continue
            m = _l1(C, seed).fit(X[a], y[a]); sc.append(roc_auc_score(y[b], m.decision_function(X[b])))
        s = np.mean(sc) if sc else -1
        if s > best[0]: best = (s, C)
    return best[1]

def _strat_keep(score, y, h):
    """Keep, within each class, the ceil(h*n_c) records with the smallest score."""
    keep = []
    for c in np.unique(y):
        ic = np.where(y == c)[0]; k = max(int(np.ceil(h * len(ic))), 1)
        keep.append(ic[np.argsort(score[ic])[:k]])
    return np.concatenate(keep)

class _Base:
    def decision_function(self, X): return self.model_.decision_function(X[:, self.cols_] if hasattr(self, "cols_") else X)
    @property
    def support_(self):
        if hasattr(self, "support_idx_"): return self.support_idx_
        return np.where(self.model_.coef_[0] != 0)[0]

class L1Logit(_Base):
    name = "L1-Logit"
    def __init__(self, seed=0, C=None, Cs=CS): self.seed, self.C, self.Cs = seed, C, Cs
    def fit(self, X, y):
        self.C_ = self.C or cv_C(X, y, Cs=self.Cs, seed=self.seed)
        self.model_ = _l1(self.C_, self.seed).fit(X, y); self.kept_ = np.arange(len(y)); return self

class StratTrimLinear(_Base):
    """Reproduces the J-5 'SLTS+LASSO' estimator (linear-residual trimming)."""
    name = "SLTS-linear (J-5)"
    def __init__(self, h=0.9, alpha=0.02, starts=3, csteps=4, seed=0, C=None):
        self.h, self.alpha, self.starts, self.csteps, self.seed, self.C = h, alpha, starts, csteps, seed, C
    def fit(self, X, y):
        rng = np.random.RandomState(self.seed); n = len(y); best = (np.inf, None)
        for _ in range(self.starts):
            idx = rng.choice(n, size=max(int(self.h * n) // 2, 50), replace=False)
            if len(np.unique(y[idx])) < 2: idx = np.arange(n)
            for _ in range(self.csteps):
                m = Lasso(alpha=self.alpha, max_iter=300, tol=1e-3).fit(X[idx], y[idx])
                r = np.abs(y - m.predict(X)); idx = _strat_keep(r, y, self.h)
            obj = np.sum(r[idx] ** 2) + self.alpha * np.abs(m.coef_).sum()
            if obj < best[0]: best = (obj, idx)
        idx = best[1]
        if y[idx].min() == y[idx].max(): idx = np.arange(n)
        self.kept_ = np.sort(idx); self.C_ = self.C or cv_C(X[idx], y[idx], seed=self.seed)
        self.model_ = _l1(self.C_, self.seed).fit(X[idx], y[idx]); return self

def _logloss(model, X, y):
    z = model.decision_function(X); return np.logaddexp(0, -(2 * y - 1) * z)

class TrimLogit(_Base):
    name = "TrimLogit"
    def __init__(self, h=0.9, csteps=3, seed=0, C=None):
        self.h, self.csteps, self.seed, self.C = h, csteps, seed, C
    def fit(self, X, y):
        self.C_ = self.C or cv_C(X, y, seed=self.seed)
        idx = np.arange(len(y))
        for _ in range(self.csteps):
            m = _l1(self.C_, self.seed).fit(X[idx], y[idx])
            new = np.sort(_strat_keep(_logloss(m, X, y), y, self.h))
            if len(new) == len(idx) and np.all(new == idx): break
            idx = new
        self.kept_ = idx; self.model_ = _l1(self.C_, self.seed).fit(X[idx], y[idx]); return self

class AdaTrim(_Base):
    """Choose h among `grid` by CV AUC.  Held-out folds are scored on their (possibly noisy) labels:
    label noise attenuates AUC roughly uniformly across candidate models, so ranking of h survives."""
    name = "AdaTrim"
    def __init__(self, grid=(1.0, 0.95, 0.9, 0.8), seed=0, folds=3, C=None):
        self.grid, self.seed, self.folds, self.C = grid, seed, folds, C
    def fit(self, X, y):
        C = self.C or cv_C(X, y, seed=self.seed); skf = StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)
        scores = {h: [] for h in self.grid}
        for a, b in skf.split(X, y):
            if y[b].min() == y[b].max(): continue
            for h in self.grid:
                m = TrimLogit(h=h, csteps=2, seed=self.seed, C=C).fit(X[a], y[a])
                scores[h].append(roc_auc_score(y[b], m.decision_function(X[b])))
        mean = {h: (np.mean(v) if v else -1) for h, v in scores.items()}
        top = max(mean.values())
        # tie-break: within 0.003 AUC of the best, prefer *more* trimming (smaller h) -> safer under contamination
        self.h_ = min(h for h in self.grid if mean[h] >= top - 0.003); self.cv_scores_ = mean
        self.inner_ = TrimLogit(h=self.h_, seed=self.seed, C=C).fit(X, y)
        self.model_, self.kept_, self.C_ = self.inner_.model_, self.inner_.kept_, C; return self

class StabilitySelection(_Base):
    name = "StabSel"
    def __init__(self, B=20, pi=0.6, seed=0, C=None, ridge_C=0.05):
        self.B, self.pi, self.seed, self.C, self.ridge_C = B, pi, seed, C, ridge_C
    def fit(self, X, y, kept=None):
        idx = np.arange(len(y)) if kept is None else kept
        C = self.C or cv_C(X[idx], y[idx], seed=self.seed); rng = np.random.RandomState(self.seed)
        freq = np.zeros(X.shape[1]); done = 0
        for _ in range(self.B):
            sub = []
            for c in np.unique(y[idx]):
                ic = idx[y[idx] == c]; sub.append(rng.choice(ic, size=max(len(ic) // 2, 1), replace=False))
            sub = np.concatenate(sub)
            if y[sub].min() == y[sub].max(): continue
            m = _l1(C, int(rng.randint(1 << 30))).fit(X[sub], y[sub]); freq += (m.coef_[0] != 0); done += 1
        self.freq_ = freq / max(done, 1); cols = np.where(self.freq_ >= self.pi)[0]
        if len(cols) == 0: cols = np.argsort(-self.freq_)[:5]   # never return an empty model
        self.cols_ = cols; self.support_idx_ = cols; self.C_ = C
        self.model_ = LogisticRegression(penalty="l2", C=self.ridge_C, max_iter=500).fit(X[idx][:, cols], y[idx]); return self

class RobustTox(_Base):
    """Adaptive log-loss trimming, then stability selection on the kept records, then ridge refit."""
    name = "RobustTox"
    def __init__(self, seed=0, grid=(1.0, 0.95, 0.9, 0.8), B=20, pi=0.6):
        self.seed, self.grid, self.B, self.pi = seed, grid, B, pi
    def fit(self, X, y):
        self.ada_ = AdaTrim(grid=self.grid, seed=self.seed).fit(X, y)
        self.h_, self.kept_ = self.ada_.h_, self.ada_.kept_
        self.ss_ = StabilitySelection(B=self.B, pi=self.pi, seed=self.seed, C=self.ada_.C_).fit(X, y, kept=self.kept_)
        self.model_, self.cols_, self.support_idx_, self.freq_ = self.ss_.model_, self.ss_.cols_, self.ss_.support_idx_, self.ss_.freq_
        return self

METHODS = {
    "L1-Logit": lambda seed: L1Logit(seed=seed),
    "SLTS-linear (J-5)": lambda seed: StratTrimLinear(h=0.9, seed=seed),
    "TrimLogit": lambda seed: TrimLogit(h=0.9, seed=seed),
    "AdaTrim": lambda seed: AdaTrim(seed=seed),
    "StabSel": lambda seed: StabilitySelection(seed=seed),
    "RobustTox": lambda seed: RobustTox(seed=seed),
}
