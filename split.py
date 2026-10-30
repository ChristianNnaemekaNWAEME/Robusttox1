"""Randomised Bemis-Murcko scaffold splits (a different split per seed)."""
import numpy as np

def scaffold_split(scaf, seed, test_frac=0.2):
    rng = np.random.RandomState(seed)
    groups = {}
    for i, s in enumerate(scaf):
        groups.setdefault(s, []).append(i)
    keys = list(groups); rng.shuffle(keys)
    thr = len(scaf) * test_frac / 2
    big = sorted([k for k in keys if len(groups[k]) > thr], key=lambda k: -len(groups[k]))
    small = [k for k in keys if len(groups[k]) <= thr]  # already shuffled
    n_tr = int((1 - test_frac) * len(scaf)); tr, te = [], []
    for k in big + small:
        (tr if len(tr) + len(groups[k]) <= n_tr else te).extend(groups[k])
    return np.array(sorted(tr)), np.array(sorted(te))
