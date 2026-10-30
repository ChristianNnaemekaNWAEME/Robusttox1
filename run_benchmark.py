"""Usage: python run_benchmark.py WORKER NWORKERS [seeds] .  Writes results/bench_<worker>.csv and results/support_<worker>.jsonl"""
import sys, os, json, time, csv, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from robusttox import prep, scaffold_split, METHODS
from robusttox.methods import L1Logit
W, NW = int(sys.argv[1]), int(sys.argv[2]); SEEDS = range(int(sys.argv[3]) if len(sys.argv) > 3 else 3)
NOISE = [("none", 0.0), ("random", 0.1), ("random", 0.2), ("random", 0.3), ("margin", 0.1), ("margin", 0.2)]
os.makedirs(f"{ROOT}/results", exist_ok=True)
out = open(f"{ROOT}/results/bench_{W}.csv", "a")
if out.tell() == 0: out.write("ds,task,seed,kind,noise,method,auc,prauc,nnz,n_train,n_test,prev_test,sec,h,flips_removed,n_removed,n_flips\n")
sup = open(f"{ROOT}/results/support_{W}.jsonl", "a")
jobs = []
for ds in ("tox21", "clintox", "sider", "toxcast_hep"):
    z = np.load(f"{ROOT}/data/{ds}.npz", allow_pickle=True)
    for t in z["tasks"]: jobs.append((ds, str(t)))
BASE = {"Random Forest": lambda s: RandomForestClassifier(200, n_jobs=1, random_state=s, class_weight="balanced_subsample", min_samples_leaf=2),
        "Gradient boosting": lambda s: HistGradientBoostingClassifier(max_iter=150, random_state=s)}
for j, (ds, task) in enumerate(jobs):
    if j % NW != W: continue
    z = np.load(f"{ROOT}/data/{ds}.npz", allow_pickle=True); y = z["y_" + task]; ok = ~np.isnan(y)
    X, A, sc, y = z["X"][ok], z["A"][ok].astype(float), z["scaf"][ok], y[ok].astype(int)
    for seed in SEEDS:
        tr, te = scaffold_split(sc, seed)
        if min(y[te].mean(), y[tr].mean()) in (0, 1) or y[te].sum() < 5: continue
        Xtr, Xte, kc = prep(X[tr], X[te])
        ref = L1Logit(seed=seed).fit(Xtr, y[tr]); margin = np.abs(ref.decision_function(Xtr))  # reference margins from CLEAN labels (design device)
        for kind, noise in NOISE:
            rng = np.random.RandomState(1000 + seed); ytr = y[tr].copy()
            if kind == "random": flip = rng.rand(len(ytr)) < noise
            elif kind == "margin":  # worst case for direction: flip the most confidently-labelled compounds
                flip = np.zeros(len(ytr), bool); flip[np.argsort(-margin)[: int(noise * len(ytr))]] = True
            else: flip = np.zeros(len(ytr), bool)
            ytr[flip] = 1 - ytr[flip]
            runs = {k: f for k, f in METHODS.items()}
            for name in list(runs) + list(BASE) + ["Alerts-only"]:
                t0 = time.time(); extra = {}
                if name in runs:
                    m = runs[name](seed).fit(Xtr, ytr); p = m.decision_function(Xte); nz = len(m.support_)
                    kept = getattr(m, "kept_", np.arange(len(ytr)))
                    removed = np.setdiff1d(np.arange(len(ytr)), kept); extra = dict(h=getattr(m, "h_", getattr(m, "h", 1.0)),
                        flips_removed=int(flip[removed].sum()) if len(removed) else 0, n_removed=len(removed), n_flips=int(flip.sum()))
                    coef = m.model_.coef_[0]
                    cols = getattr(m, "cols_", None); idx = np.asarray(m.support_)
                    if cols is not None: orig = kc[cols]; vals = coef
                    else: orig = kc[idx]; vals = coef[idx]
                    if kind in ("none", "random") and noise in (0.0, 0.2): sup.write(json.dumps(dict(ds=ds, task=task, seed=seed, kind=kind, noise=noise, method=name, feat=[int(a) for a in orig], coef=[float(a) for a in vals])) + "\n"); sup.flush()
                elif name in BASE:
                    m = BASE[name](seed).fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; nz = -1
                else:
                    m = L1Logit(seed=seed).fit(A[tr], ytr); p = m.decision_function(A[te]); nz = len(m.support_)
                row = dict(ds=ds, task=task, seed=seed, kind=kind, noise=noise, method=name, auc=roc_auc_score(y[te], p), prauc=average_precision_score(y[te], p),
                           nnz=nz, n_train=len(tr), n_test=len(te), prev_test=y[te].mean(), sec=time.time() - t0, **extra)
                csv.writer(out).writerow([row.get(k, "") for k in ["ds","task","seed","kind","noise","method","auc","prauc","nnz","n_train","n_test","prev_test","sec","h","flips_removed","n_removed","n_flips"]]); out.flush()
print("DONE", W, flush=True)
