"""Featurisation: 217 RDKit descriptors + 512-bin log Morgan count fingerprint."""
import warnings
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")

FP_SIZE = 512
N_DESC = len(Descriptors._descList)

def feature_names():
    return [d[0] for d in Descriptors._descList] + [f"morgan_count_{i}" for i in range(FP_SIZE)]

def featurize(smiles):
    """Return (X, kept_index, murcko_scaffolds). Invalid SMILES are dropped."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)
    X, keep, scaf = [], [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None or m.GetNumAtoms() == 0:
            continue
        try:
            d = [f(m) for _, f in Descriptors._descList]
            fp = gen.GetCountFingerprintAsNumPy(m).astype(float)
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            continue
        X.append(np.concatenate([d, np.log1p(fp)])); keep.append(i); scaf.append(sc)
    X = np.array(X, float); X[~np.isfinite(X)] = np.nan
    return X, np.array(keep), np.array(scaf)

def prep(Xtr, Xte):
    """Median-impute, standardise on train, clip to +-5. Returns (Xtr, Xte, kept_columns)."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    med = np.nanmedian(Xtr, 0); med = np.where(np.isfinite(med), med, 0)
    for X in (Xtr, Xte):
        r, c = np.where(~np.isfinite(X)); X[r, c] = med[c]
    mu, sd = Xtr.mean(0), Xtr.std(0); keep = sd > 1e-8
    f = lambda X: np.clip((X[:, keep] - mu[keep]) / sd[keep], -5, 5)
    return f(Xtr), f(Xte), np.where(keep)[0]
