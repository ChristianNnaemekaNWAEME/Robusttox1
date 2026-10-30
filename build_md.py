import pandas as pd, numpy as np, glob, json
R = "/home/claude/robusttox/results"; T = R + "/tables"
def md(df, align=None):
    cols = list(df.columns); out = "| " + " | ".join(str(c) for c in cols) + " |\n|" + "|".join((":---" if i == 0 else "---:") for i in range(len(cols))) + "|\n"
    for _, r in df.iterrows(): out += "| " + " | ".join(str(v) for v in r.values) + " |\n"
    return out
NM = {"SLTS-linear (J-5)": "SLTS-linear"}; nm = lambda m: NM.get(m, m)
conds = ["none-0.0", "random-0.1", "random-0.2", "random-0.3", "margin-0.1", "margin-0.2"]; ch = ["Clean", "Random 10%", "Random 20%", "Random 30%", "Margin 10%", "Margin 20%"]
order = ["Random Forest", "Gradient boosting", "L1-Logit", "SLTS-linear (J-5)", "TrimLogit", "AdaTrim", "StabSel", "RobustTox", "Alerts-only"]
# datasets
rows = []
for ds, lab in [("tox21", "Tox21"), ("clintox", "ClinTox"), ("sider", "SIDER"), ("toxcast_hep", "ToxCast (HepG2 / hepatocyte)")]:
    z = np.load(f"/home/claude/robusttox/data/{ds}.npz", allow_pickle=True)
    for t in z["tasks"]:
        y = z["y_" + str(t)]; y = y[~np.isnan(y)]; rows.append([lab, str(t), f"{len(y):,}", f"{100 * y.mean():.1f}%"])
T_DATA = md(pd.DataFrame(rows, columns=["Collection", "Endpoint", "Compounds", "Positive rate"]))
ci = pd.read_csv(f"{T}/mean_auc_by_noise_ci.csv")
a = ci.pivot(index="method", columns="noise", values="mean_auc").loc[order, conds]
a_fmt = a.map(lambda v: f"{v:.3f}"); a_fmt.index = [nm(m) for m in a_fmt.index]; a_fmt.columns = ch; a_fmt = a_fmt.reset_index().rename(columns={"index": "Method", "method": "Method"})
T_AUC = md(a_fmt)
sel = ["Random Forest", "L1-Logit", "SLTS-linear (J-5)", "AdaTrim", "RobustTox"]; rows = []
for m in sel:
    d = ci[ci.method == m].set_index("noise").loc[["none-0.0", "random-0.2", "margin-0.2"]]; rows.append([nm(m)] + [f"{r.mean_auc:.3f} [{r.lo:.3f}, {r.hi:.3f}]" for r in d.itertuples()])
T_CI = md(pd.DataFrame(rows, columns=["Method", "Clean", "Random 20%", "Margin 20%"]))
dr = pd.read_csv(f"{T}/auc_drop.csv").set_index("method").loc[order, ["random-0.1", "random-0.2", "random-0.3", "margin-0.1", "margin-0.2"]]
dr = dr.map(lambda v: f"{v:.3f}"); dr.index = [nm(m) for m in dr.index]; dr.columns = ch[1:]; T_DROP = md(dr.reset_index().rename(columns={"method": "Method", "index": "Method"}))
P = pd.read_csv(f"{T}/paired_tests.csv"); rows = []
def prow(ref, m, c):
    r = P[(P.ref == ref) & (P.method == m) & (P.noise == c)].iloc[0]; return f"{r.mean_diff:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]; {int(r.wins)}/{int(r.n)}; p={r.p_wilcoxon:.3f}" if r.p_wilcoxon >= .001 else f"{r.mean_diff:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]; {int(r.wins)}/{int(r.n)}; p<0.001"
for m in ["SLTS-linear (J-5)", "TrimLogit", "AdaTrim", "StabSel", "RobustTox", "Random Forest"]:
    rows.append([nm(m)] + [prow("L1-Logit", m, c) for c in ["none-0.0", "random-0.2", "margin-0.2"]])
T_PAIR_L1 = md(pd.DataFrame(rows, columns=["Method (minus L1-Logit)", "Clean", "Random 20%", "Margin 20%"]))
rows = []
for m in ["TrimLogit", "AdaTrim", "RobustTox", "StabSel"]:
    rows.append([nm(m)] + [prow("SLTS-linear (J-5)", m, c) for c in ["none-0.0", "random-0.2", "margin-0.2"]])
T_PAIR_J5 = md(pd.DataFrame(rows, columns=["Method (minus SLTS-linear)", "Clean", "Random 20%", "Margin 20%"]))
dg = pd.read_csv(f"{T}/trimming_diagnostics.csv"); rows = []
for c in conds[1:]:
    d = dg[(dg.noise == c) & (dg.method == "AdaTrim")].iloc[0]; base = float(c.split("-")[1])
    rows.append([ch[conds.index(c)], f"{100 * d.trim_frac:.1f}%", f"{100 * d.flip_recall:.1f}%", f"{d.flip_recall / d.trim_frac:.2f}", f"{100 * d.precision:.1f}% (base rate {100 * base:.0f}%)"])
T_TRIM = md(pd.DataFrame(rows, columns=["Condition", "Records trimmed", "Flipped labels removed", "Enrichment (removed / trimmed)", "Share of trimmed records that were flipped"]))
sp = pd.read_csv(f"{T}/median_nnz.csv").set_index("method"); J = pd.read_csv(f"{T}/selection_stability_jaccard.csv").set_index("method"); rows = []
for m in ["L1-Logit", "SLTS-linear (J-5)", "TrimLogit", "AdaTrim", "StabSel", "RobustTox"]:
    rows.append([nm(m), int(sp.loc[m, "none-0.0"]), int(sp.loc[m, "random-0.2"]), f"{J.loc[m, 'none-0.0']:.3f}", f"{J.loc[m, 'random-0.2']:.3f}"])
T_SPARS = md(pd.DataFrame(rows, columns=["Method", "Median features (clean)", "Median features (random 20%)", "Selection stability, clean (Jaccard)", "Selection stability, random 20% (Jaccard)"]))
S = pd.read_csv(f"{R}/simulation.csv"); rows = []
for c, cl in [("random", "random"), ("targeted", "margin")]:
    for m, ml in [("L1-Logit", "L1-Logit"), ("TrimLogit h=1-eps (oracle)", "TrimLogit, h = 1 - eps - 0.02"), ("AdaTrim", "AdaTrim"), ("RobustTox", "RobustTox")]:
        g = S[(S.contam == c) & (S.method == m)].groupby("eps").cos.mean(); rows.append([cl, ml] + [f"{v:.2f}" for v in g.values])
T_SIM = md(pd.DataFrame(rows, columns=["Flips", "Method", "eps = 0", "0.1", "0.2", "0.3", "0.4"]))
A = pd.read_csv(f"{T}/interpretability_alert_overlap.csv"); rows = []
from scipy import stats
for m, g in A.groupby("method"):
    d = g.mean_overlap_selected - g.mean_overlap_random
    rows.append([m, len(g), f"{g.mean_overlap_selected.mean():.3f}", f"{g.mean_overlap_random.mean():.3f}", f"{d.mean():+.3f}", f"{stats.wilcoxon(d).pvalue:.3f}", int((g.p_perm < .05).sum()), f"{stats.combine_pvalues(g.p_perm, method='fisher')[1]:.3f}"])
T_INT = md(pd.DataFrame(rows, columns=["Model", "Endpoints analysed", "Overlap, selected bits", "Overlap, random bits", "Difference", "Wilcoxon p (across endpoints)", "Endpoints with permutation p < 0.05", "Fisher combined p"]))
B = pd.read_csv(f"{T}/interpretability_bit_detail.csv"); x = B[(B.method == "RobustTox") & (B.task == "SR-ARE")].sort_values("select_freq", ascending=False).head(3)
T_BITS = md(x[["task", "bit", "select_freq", "overlap_rate", "n_mol", "top_substructure", "top_alerts"]].rename(columns={"task": "Endpoint", "bit": "Morgan bin", "select_freq": "Selection frequency", "overlap_rate": "Alert overlap rate", "n_mol": "Molecules examined", "top_substructure": "Most frequent environment (SMILES)", "top_alerts": "Overlapping alerts"}).round(2).assign(**{"Most frequent environment (SMILES)": lambda d: d["Most frequent environment (SMILES)"].map(lambda s: "`" + str(s) + "`")}))
al = pd.read_csv(f"{R}/alerts_only.csv"); print("alerts rows", len(al))
tmpl = open("report_template.md").read()
for k, v in dict(T_DATA=T_DATA, T_AUC=T_AUC, T_CI=T_CI, T_DROP=T_DROP, T_PAIR_L1=T_PAIR_L1, T_PAIR_J5=T_PAIR_J5, T_TRIM=T_TRIM, T_SPARS=T_SPARS, T_SIM=T_SIM, T_INT=T_INT, T_BITS=T_BITS).items(): tmpl = tmpl.replace("{{" + k + "}}", v)
open("report.md", "w").write(tmpl); print("built", len(tmpl))
