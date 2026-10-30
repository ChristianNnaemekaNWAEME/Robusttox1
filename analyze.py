import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
"""Aggregate benchmark: per-task CIs over scaffold splits, across-task paired tests, stability, tables & figures."""
import os, sys, glob, json, numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ROOT}/results/bench_*.csv")], ignore_index=True)
df = df[df.method != "Alerts-only"]
al = pd.read_csv(f"{ROOT}/results/alerts_only.csv"); df = pd.concat([df, al], ignore_index=True)
df = df.drop_duplicates(["ds", "task", "seed", "kind", "noise", "method"])
df["noise"] = df.kind + "-" + df.noise.astype(str)   # condition label, e.g. random-0.2 / margin-0.1 / none-0.0
CLEAN = "none-0.0"
os.makedirs(f"{ROOT}/results/tables", exist_ok=True)
T = lambda name, d: d.to_csv(f"{ROOT}/results/tables/{name}.csv")
# complete cases only (every method present for a task/seed/noise)
nm = df.method.nunique(); g = df.groupby(["ds", "task", "seed", "noise"]).method.transform("nunique"); df = df[g == nm]
print("rows", len(df), "tasks", df.task.nunique(), "seeds", df.seed.nunique(), "methods", nm)
def boot_ci(v, B=4000, rng=np.random.RandomState(0)):
    v = np.asarray(v); b = rng.choice(v, (B, len(v))).mean(1); return np.percentile(b, [2.5, 97.5])
# 1. mean AUC across tasks (average over seeds within task first), 95% bootstrap CI over tasks
pt = df.groupby(["task", "noise", "method"]).auc.mean().reset_index()
rows = []
for (noise, m), d in pt.groupby(["noise", "method"]):
    lo, hi = boot_ci(d.auc); rows.append(dict(noise=noise, method=m, mean_auc=d.auc.mean(), lo=lo, hi=hi, n_tasks=len(d)))
tab1 = pd.DataFrame(rows).pivot(index="method", columns="noise", values="mean_auc"); T("mean_auc_by_noise", tab1); print(tab1.round(3))
T("mean_auc_by_noise_ci", pd.DataFrame(rows).set_index(["noise", "method"]))
# 2. drop in AUC from clean to noise, per method
cl = pt[pt.noise == CLEAN].set_index(["task", "method"]).auc
drop = pt[pt.noise != CLEAN].copy(); drop["clean"] = [cl[(t, m)] for t, m in zip(drop.task, drop.method)]; drop["drop"] = drop.clean - drop.auc
dr = drop.groupby(["noise", "method"])["drop"].mean().unstack(0); T("auc_drop", dr); print(dr.round(3))
# 3. paired tests vs J-5 estimator and vs L1-Logit (across tasks, Wilcoxon), per noise
res = []
for noise in sorted(df.noise.unique()):
    p = pt[pt.noise == noise].pivot(index="task", columns="method", values="auc")
    for ref in ("SLTS-linear (J-5)", "L1-Logit"):
        for m in p.columns:
            if m == ref: continue
            d = (p[m] - p[ref]).dropna(); lo, hi = boot_ci(d)
            try: pv = stats.wilcoxon(d).pvalue
            except Exception: pv = np.nan
            res.append(dict(noise=noise, ref=ref, method=m, mean_diff=d.mean(), lo=lo, hi=hi, wins=int((d > 0).sum()), n=len(d), p_wilcoxon=pv))
R = pd.DataFrame(res); T("paired_tests", R.set_index(["noise", "ref", "method"]))
print(R[(R.ref == "SLTS-linear (J-5)") & R.method.isin(["RobustTox", "AdaTrim", "TrimLogit", "StabSel", "Random Forest"])].round(4).to_string())
# 4. sparsity and trimming diagnostics
sp = df[df.nnz >= 0].groupby(["noise", "method"]).nnz.median().unstack(0); T("median_nnz", sp); print(sp)
tr = df[df.method.isin(["TrimLogit", "AdaTrim", "RobustTox", "SLTS-linear (J-5)"]) & (df.noise != CLEAN)].copy()
tr["flip_recall"] = tr.flips_removed / tr.n_flips.replace(0, np.nan); tr["trim_frac"] = tr.n_removed / tr.n_train
tr["precision"] = tr.flips_removed / tr.n_removed.replace(0, np.nan)
dg = tr.groupby(["noise", "method"])[["trim_frac", "flip_recall", "precision"]].mean(); T("trimming_diagnostics", dg); print(dg.round(3))
hs = df[(df.method == "AdaTrim")].groupby("noise").h.value_counts().unstack(fill_value=0); T("adatrim_h_choice", hs); print(hs)
# 5. per-task AUC with CI over splits (t-interval), for the supplement
ci = df.groupby(["task", "noise", "method"]).auc.agg(["mean", "std", "count"]); ci["half"] = stats.t.ppf(.975, ci["count"] - 1) * ci["std"] / np.sqrt(ci["count"]); T("per_task_auc_ci", ci)
# 6. selection stability (mean pairwise Jaccard of selected features across splits)
S = [json.loads(l) for f in glob.glob(f"{ROOT}/results/support_*.jsonl") for l in open(f)]
rows = []
SD = pd.DataFrame([{k: v for k, v in s.items() if k != 'coef'} | {"set": frozenset(s["feat"])} for s in S]); SD["noise"] = SD.kind + "-" + SD.noise.astype(str)
for (m, noise), g in SD.groupby(["method", "noise"]):
    js = []
    for task, gg in g.groupby("task"):
        sets = list(gg.set)
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = len(sets[i] | sets[j]); js.append(len(sets[i] & sets[j]) / u if u else np.nan)
    rows.append(dict(method=m, noise=noise, jaccard=np.nanmean(js), n_pairs=len(js)))
J = pd.DataFrame(rows).pivot(index="method", columns="noise", values="jaccard"); T("selection_stability_jaccard", J); print(J.round(3))
# 7. figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4)); cols = plt.cm.tab10(np.arange(10))
for k, m in enumerate(tab1.index):
    ax[0].plot(range(len(tab1.columns)), tab1.loc[m], marker="o", label=m, color=cols[k]); ax[1].plot(range(len(dr.columns)), dr.loc[m], marker="o", label=m, color=cols[k])
for a, cc in ((ax[0], tab1.columns), (ax[1], dr.columns)): a.set_xticks(range(len(cc))); a.set_xticklabels(cc, rotation=40, ha="right", fontsize=7)
ax[0].set_ylabel("mean test ROC-AUC (scaffold split)"); ax[1].set_ylabel("AUC lost vs clean labels")
ax[1].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{ROOT}/results/fig_auc_vs_noise.png", dpi=160)
