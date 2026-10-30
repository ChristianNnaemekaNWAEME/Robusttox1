"""Featurise every dataset once and cache to data/<name>.npz  (X, scaffolds, alert matrix, labels)."""
import sys, os, numpy as np, pandas as pd, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robusttox.features import featurize
from robusttox.alerts import alert_matrix
SRC = sys.argv[1]; OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TOX = ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER","NR-ER-LBD","NR-PPAR-gamma","SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]
HEP = ["APR_HepG2_CellLoss_72h_dn","APR_HepG2_MitoMembPot_24h_dn","APR_HepG2_OxidativeStress_24h_up"]
jobs = {"tox21": ("tox21.csv", {t: t for t in TOX}),
        "clintox": ("clintox.csv", {"CT_TOX": "CT_TOX"}),
        "sider": ("sider.csv", {"SIDER-Hepatobiliary": "Hepatobiliary disorders"}),
        "toxcast_hep": ("toxcast_data.csv", {h: h for h in HEP})}
os.makedirs(OUT, exist_ok=True)
for ds, (fn, tasks) in jobs.items():
    df = pd.read_csv(os.path.join(SRC, fn)); X, keep, sc = featurize(df.smiles.values)
    d = df.iloc[keep].reset_index(drop=True); A, anames, _ = alert_matrix(d.smiles.values)
    Y = {t: d[c].values.astype(float) for t, c in tasks.items()}
    np.savez_compressed(os.path.join(OUT, ds + ".npz"), X=X, scaf=sc, A=A, smiles=d.smiles.values, anames=np.array(anames),
                        tasks=np.array(list(Y)), **{"y_" + t: v for t, v in Y.items()})
    print(ds, X.shape, A.shape, flush=True)
