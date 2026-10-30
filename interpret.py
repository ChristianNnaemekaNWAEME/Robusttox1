"""Map selected Morgan bits back to substructures and compare with Brenk / PAINS / NIH alerts.

For each task: take RobustTox (and L1-Logit) features selected on clean labels across splits, keep Morgan bits
with positive coefficient (toxic-associated), recover the atom environments that set each bit, and measure
  overlap_rate(bit) = share of molecules carrying the bit in which the environment (radius>=1) shares atoms with a matched alert.
Significance: compare the mean overlap_rate of the selected bits with equally sized random sets of bits of similar frequency.
Also: an alerts-only L1-logistic baseline AUC is in the benchmark (method 'Alerts-only').
"""
import os, sys, glob, json, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from collections import Counter, defaultdict
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from robusttox.features import N_DESC, FP_SIZE, feature_names
from robusttox.alerts import alert_matrix
NAMES = feature_names(); gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)
METHOD = sys.argv[1] if len(sys.argv) > 1 else "RobustTox"; TOPK = 10; MAXMOL = 400
S = [json.loads(l) for f in glob.glob(f"{ROOT}/results/support_*.jsonl") for l in open(f)]
S = [s for s in S if s["kind"] == "none" and s["method"] in (METHOD, "L1-Logit")]
cache = {}
def dataset(ds):
    if ds in cache: return cache[ds]
    z = np.load(f"{ROOT}/data/{ds}.npz", allow_pickle=True); smi = z["smiles"]
    _, _, atoms = alert_matrix(smi)
    cache[ds] = (smi, atoms, {t: z["y_" + t] for t in z["tasks"]}); return cache[ds]
def env_atoms(mol, atom, rad):
    if rad == 0: return {atom}, None
    bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom); at = set()
    for b in bonds: bd = mol.GetBondWithIdx(b); at.update((bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()))
    return at, list(bonds)
def bit_stats(ds, bit, rng):
    smi, atoms, _ = dataset(ds); key = (ds, bit)
    if key in cache: return cache[key]
    hits = []
    for i, s in enumerate(smi):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap(); gen.GetCountFingerprint(m, additionalOutput=ao)
        bi = ao.GetBitInfoMap()
        if bit in bi: hits.append((i, m, bi[bit]))
    cache[key] = hits; return hits
# faster: single pass collecting bitInfo for every molecule once per dataset
def bitinfo_all(ds):
    key = (ds, "bi")
    if key in cache: return cache[key]
    smi, atoms, _ = dataset(ds); out = []
    for s in smi:
        m = Chem.MolFromSmiles(s); ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
        if m is not None: gen.GetCountFingerprint(m, additionalOutput=ao); out.append((m, ao.GetBitInfoMap()))
        else: out.append((None, {}))
    cache[key] = out; return out
def overlap_for_bit(ds, bit, rng):
    smi, atoms, _ = dataset(ds); bi = bitinfo_all(ds); idx = [i for i, (m, b) in enumerate(bi) if bit in b]
    if len(idx) > MAXMOL: idx = list(rng.choice(idx, MAXMOL, replace=False))
    ov, envs, alerts = [], Counter(), Counter()
    for i in idx:
        m, b = bi[i]; hit = False
        for atom, rad in b[bit]:
            if rad == 0: continue
            at, bonds = env_atoms(m, atom, rad)
            for name, aa in atoms[i].items():
                if at & aa: hit = True; alerts[name] += 1
            envs[Chem.MolFragmentToSmiles(m, atomsToUse=sorted(at), bondsToUse=bonds, rootedAtAtom=atom)] += 1
        ov.append(hit)
    return (np.mean(ov) if ov else np.nan), len(idx), envs.most_common(1), alerts.most_common(2)
rng = np.random.RandomState(0); rows = []; detail = []
for (ds, task), g in pd.DataFrame(S).groupby(["ds", "task"]):
    if g.method.nunique() < 1: continue
    for method, gm in g.groupby("method"):
        agg = defaultdict(lambda: [0, 0.0]); nsplit = gm.seed.nunique()
        for _, r in gm.iterrows():
            for f, c in zip(r.feat, r.coef): agg[f][0] += 1; agg[f][1] += c
        pos = sorted([(f, v[0] / nsplit, v[1] / v[0]) for f, v in agg.items() if f >= N_DESC and v[1] > 0], key=lambda t: (-t[1], -t[2]))[:TOPK]
        if len(pos) < 3: continue
        bits = [f - N_DESC for f, _, _ in pos]; sel = [overlap_for_bit(ds, b, rng) for b in bits]
        rates = np.array([x[0] for x in sel]); allb = np.arange(FP_SIZE)
        smi, atoms, _ = dataset(ds); bi = bitinfo_all(ds); freq = np.array([sum(1 for m, b in bi if bb in b) for bb in allb])
        pool = [b for b in allb if 30 <= freq[b]]; nulls = []
        for _ in range(60):
            pick = rng.choice(pool, len(bits), replace=False); nulls.append(np.nanmean([overlap_for_bit(ds, int(b), rng)[0] for b in pick]))
        obs = np.nanmean(rates); p = (1 + sum(n >= obs for n in nulls)) / (1 + len(nulls))
        rows.append(dict(ds=ds, task=task, method=method, n_bits=len(bits), mean_overlap_selected=obs, mean_overlap_random=np.mean(nulls), p_perm=p))
        for (f, fr, c), s_ in zip(pos, sel): detail.append(dict(ds=ds, task=task, method=method, bit=f - N_DESC, select_freq=fr, coef=c, overlap_rate=s_[0], n_mol=s_[1],
                                                              top_substructure=s_[2][0][0] if s_[2] else "", top_alerts="; ".join(a for a, _ in s_[3])))
        print(ds, task, method, round(obs, 3), round(np.mean(nulls), 3), p, flush=True)
os.makedirs(f"{ROOT}/results/tables", exist_ok=True)
pd.DataFrame(rows).to_csv(f"{ROOT}/results/tables/interpretability_alert_overlap.csv", index=False)
pd.DataFrame(detail).to_csv(f"{ROOT}/results/tables/interpretability_bit_detail.csv", index=False)
