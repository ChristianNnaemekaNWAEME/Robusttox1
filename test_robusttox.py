import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robusttox.methods import L1Logit, TrimLogit, AdaTrim, StabilitySelection, RobustTox, StratTrimLinear, _strat_keep
from robusttox.split import scaffold_split

def toy(n=400, p=40, s=3, eps=0.0, seed=0):
    r = np.random.RandomState(seed); X = r.randn(n, p); b = np.zeros(p); b[:s] = 1.5
    y = (r.rand(n) < 1 / (1 + np.exp(-X @ b))).astype(int); f = r.rand(n) < eps; y[f] = 1 - y[f]; return X, y, b

def test_stratified_trim_keeps_both_classes():
    y = np.array([0] * 95 + [1] * 5); score = np.r_[np.zeros(95), np.ones(5) * 10]   # minority has the worst scores
    keep = _strat_keep(score, y, 0.9)
    assert set(y[keep]) == {0, 1}
    unstrat = np.argsort(score)[: 90]
    assert set(y[unstrat]) == {0}          # un-stratified trimming would have deleted the whole minority class

def test_c_steps_do_not_increase_objective():          # Proposition 1
    X, y, _ = toy(eps=0.1); C = 0.05
    def obj(m, idx):
        z = m.decision_function(X[idx]); return np.logaddexp(0, -(2 * y[idx] - 1) * z).sum() / 1 + np.abs(m.coef_[0]).sum() / C
    from robusttox.methods import _l1, _logloss
    idx = np.arange(len(y)); prev = np.inf
    for _ in range(4):
        m = _l1(C).fit(X[idx], y[idx]); idx = np.sort(_strat_keep(_logloss(m, X, y), y, 0.9))
        m2 = _l1(C).fit(X[idx], y[idx]); o = obj(m2, idx)
        assert o <= prev + 1e-6; prev = o

def test_methods_recover_support_on_clean_data():
    X, y, b = toy()
    for M in (L1Logit(), TrimLogit(h=0.9), AdaTrim()):
        sel = set(np.asarray(M.fit(X, y).support_).tolist()); assert {0, 1, 2} <= sel, (M.name, sel)
    for M in (StabilitySelection(), RobustTox()):   # stability selection is deliberately conservative: expect high precision, most of the truth
        sel = set(np.asarray(M.fit(X, y).support_).tolist()); assert len(sel & {0, 1, 2}) >= 2 and len(sel - {0, 1, 2}) <= 2, (M.name, sel)

def test_scaffold_split_disjoint_and_complete():
    sc = np.array(list("aabbbccdeeeffg") * 20); tr, te = scaffold_split(sc, 1)
    assert not set(sc[tr]) & set(sc[te]) and len(tr) + len(te) == len(sc)

def test_stabsel_never_empty():
    r = np.random.RandomState(0); X = r.randn(200, 30); y = r.randint(0, 2, 200)
    assert len(StabilitySelection(B=5).fit(X, y).support_) > 0
