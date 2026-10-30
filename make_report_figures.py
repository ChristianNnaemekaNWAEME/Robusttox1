import os, sys, glob, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150, "savefig.dpi": 200, "legend.frameon": False})
COL = {"L1-Logit": "#1b6ca8", "SLTS-linear (J-5)": "#d1495b", "TrimLogit": "#8d99ae", "AdaTrim": "#2a9d8f", "StabSel": "#e9c46a", "RobustTox": "#6a4c93", "Random Forest": "#3d3d3d", "Gradient boosting": "#9e9e9e", "Alerts-only": "#f4a261"}
LAB = {"SLTS-linear (J-5)": "Earlier estimator (SLTS-linear)"}
nm = lambda m: LAB.get(m, m)
def save(f, n): f.tight_layout(); f.savefig(f"{OUT}/{n}.png"); plt.close(f)
# ---- prior repositories (values transcribed from their README tables) ----
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
x = [0, 10, 20]; prior = {"Random Forest": [.818, .775, .726], "Gradient boosting": [.807, .736, .688], "LASSO": [.772, .736, .694], "SLTS+LASSO (keep 90%)": [.742, .723, .687], "SLTS+LASSO (keep 75%)": [.716, .709, .682]}
cc = ["#3d3d3d", "#9e9e9e", "#1b6ca8", "#d1495b", "#e76f51"]
for (k, v), c in zip(prior.items(), cc): ax[0].plot(x, v, marker="o", label=k, color=c)
ax[0].set_xlabel("training labels flipped (%)"); ax[0].set_ylabel("mean ROC-AUC (42 endpoint-split cases)"); ax[0].set_xticks(x); ax[0].legend(fontsize=7)
ax[0].set_title("toxicity-estimator-benchmark", fontsize=9)
xo = [0, 5, 10, 20]; es = {"Delaney 4-term (refit)": [1.105, 1.112, 1.145, 1.136], "LASSO": [.845, .912, .992, 1.043], "Random Forest": [.885, .933, .997, 1.105], "SLTS+LASSO (keep 75%)": [.877, .882, .875, .880], "SLTS+LASSO (keep 90%)": [.860, .870, .867, 1.356]}
for (k, v), c in zip(es.items(), ["#bdbdbd", "#1b6ca8", "#3d3d3d", "#e76f51", "#d1495b"]): ax[1].plot(xo, v, marker="o", label=k, color=c)
ax[1].set_xlabel("training solubilities replaced by gross outliers (%)"); ax[1].set_ylabel("mean test RMSE (log units; lower is better)"); ax[1].set_xticks(xo); ax[1].legend(fontsize=7); ax[1].set_title("esol-solubility-qsar-reproduction", fontsize=9)
save(fig, "fig_prior_repos")
fig, ax = plt.subplots(figsize=(3.8, 3)); mods = ["Random Forest", "Gradient boosting", "LASSO"]; r = [.850, .839, .808]; s = [.825, .813, .779]; w = .36; i = np.arange(3)
ax.bar(i - w / 2, r, w, label="random split", color="#8d99ae"); ax.bar(i + w / 2, s, w, label="scaffold split", color="#2a9d8f"); ax.set_ylim(.7, .9); ax.set_xticks(i); ax.set_xticklabels(mods, fontsize=8); ax.set_ylabel("mean ROC-AUC (12 assays x 3 seeds)"); ax.legend(); ax.set_title("tox21-challenge-protocol-reproduction", fontsize=9)
save(fig, "fig_prior_tox21_split")
# ---- RobustTox benchmark ----
T = f"{ROOT}/results/tables"; ci = pd.read_csv(f"{T}/mean_auc_by_noise_ci.csv"); conds = ["none-0.0", "random-0.1", "random-0.2", "random-0.3", "margin-0.1", "margin-0.2"]
lab = ["clean", "random\n10%", "random\n20%", "random\n30%", "margin\n10%", "margin\n20%"]; pos = [0, 1.2, 2.2, 3.2, 4.4, 5.4]
order = ["Random Forest", "Gradient boosting", "L1-Logit", "SLTS-linear (J-5)", "TrimLogit", "AdaTrim", "StabSel", "RobustTox", "Alerts-only"]
fig, ax = plt.subplots(figsize=(7.4, 3.8))
for m in order:
    d = ci[ci.method == m].set_index("noise").loc[conds]; ax.errorbar(pos, d.mean_auc, yerr=[d.mean_auc - d.lo, d.hi - d.mean_auc], marker="o", ms=4, lw=1.3, capsize=2, label=nm(m), color=COL[m], alpha=.95 if m in ("L1-Logit", "AdaTrim", "Random Forest", "SLTS-linear (J-5)") else .6)
ax.axhline(.5, color="k", lw=.8, ls=":"); ax.text(5.55, .505, "chance", fontsize=7); ax.set_xticks(pos); ax.set_xticklabels(lab); ax.set_ylabel("mean test ROC-AUC (17 endpoints)"); ax.legend(fontsize=7, ncol=3, loc="lower left")
ax.set_title("Accuracy under label contamination (95% bootstrap CI over endpoints)", fontsize=9); save(fig, "fig_auc_by_condition")
# paired differences vs L1-Logit
R = pd.read_csv(f"{T}/paired_tests.csv"); R = R[R.ref == "L1-Logit"]; ms = ["SLTS-linear (J-5)", "TrimLogit", "AdaTrim", "StabSel", "RobustTox"]
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=True)
for a, (c, t) in zip(axs, [("none-0.0", "clean labels"), ("random-0.2", "20% random flips"), ("margin-0.2", "20% margin flips")]):
    d = R[R.noise == c].set_index("method").loc[ms]; y = np.arange(len(ms))[::-1]
    a.errorbar(d.mean_diff, y, xerr=[d.mean_diff - d.lo, d.hi - d.mean_diff], fmt="o", color="#1b6ca8", capsize=2); a.axvline(0, color="k", lw=.8); a.set_title(t, fontsize=9); a.set_yticks(y); a.set_yticklabels([nm(m) for m in ms], fontsize=7); a.set_xlabel("AUC difference vs L1-Logit")
save(fig, "fig_paired_vs_l1")
# per-endpoint heatmap (clean)
df = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ROOT}/results/bench_*.csv")]); df = df[df.method != "Alerts-only"]; al = pd.read_csv(f"{ROOT}/results/alerts_only.csv"); df = pd.concat([df, al]); df["cond"] = df.kind + "-" + df.noise.astype(str)
H = df[df.cond == "none-0.0"].groupby(["task", "method"]).auc.mean().unstack()[order]
fig, ax = plt.subplots(figsize=(7.4, 4.8)); im = ax.imshow(H.values, aspect="auto", cmap="viridis", vmin=.5, vmax=.95)
ax.set_xticks(range(len(order))); ax.set_xticklabels([nm(m) for m in order], rotation=40, ha="right", fontsize=7); ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index, fontsize=7); ax.grid(False)
for i in range(H.shape[0]):
    for j in range(H.shape[1]): ax.text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="w" if H.values[i, j] < .75 else "k")
plt.colorbar(im, ax=ax, label="ROC-AUC (clean labels, mean of 3 scaffold splits)", shrink=.8); ax.set_title("Per-endpoint accuracy", fontsize=9); save(fig, "fig_endpoint_heatmap")
# trimming diagnostics
d = pd.read_csv(f"{T}/trimming_diagnostics.csv"); d = d[d.method.isin(["TrimLogit", "AdaTrim", "SLTS-linear (J-5)"])]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3))
cs = ["random-0.1", "random-0.2", "random-0.3", "margin-0.1", "margin-0.2"]; wd = .26
for k, m in enumerate(["SLTS-linear (J-5)", "TrimLogit", "AdaTrim"]):
    dd = d[d.method == m].set_index("noise").loc[cs]; ax[0].bar(np.arange(5) + (k - 1) * wd, dd.flip_recall, wd, label=nm(m), color=COL[m])
tf = d[d.method == "AdaTrim"].set_index("noise").loc[cs].trim_frac
ax[0].plot(np.arange(5), tf, "k_", ms=18, mew=1.5, label="share of records trimmed (AdaTrim)"); ax[0].set_xticks(range(5)); ax[0].set_xticklabels([c.replace("-", "\n") for c in cs], fontsize=7); ax[0].set_ylabel("flipped labels removed (share)"); ax[0].legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 1.28), ncol=2); ax[0].set_ylim(0, .24)
hc = pd.read_csv(f"{T}/adatrim_h_choice.csv").set_index("noise").loc[["none-0.0"] + cs]; hc = hc.div(hc.sum(1), axis=0); bt = np.zeros(len(hc))
for h, c in zip(hc.columns, ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]): ax[1].bar(range(len(hc)), hc[h], bottom=bt, color=c, label=f"h = {float(h):.2f}"); bt += hc[h].values
ax[1].set_xticks(range(len(hc))); ax[1].set_xticklabels([c.replace("-", "\n") for c in hc.index], fontsize=7); ax[1].set_ylabel("share of fits choosing each h"); ax[1].legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.12)); ax[1].set_ylim(0, 1); save(fig, "fig_trimming_diagnostics")
# sparsity & stability
sp = pd.read_csv(f"{T}/median_nnz.csv").set_index("method").drop("Alerts-only", errors="ignore"); J = pd.read_csv(f"{T}/selection_stability_jaccard.csv").set_index("method")
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3)); ms2 = ["L1-Logit", "SLTS-linear (J-5)", "TrimLogit", "AdaTrim", "StabSel", "RobustTox"]
ax[0].bar(range(6), sp.loc[ms2, "none-0.0"], color=[COL[m] for m in ms2]); ax[0].set_xticks(range(6)); ax[0].set_xticklabels([nm(m).replace(" (", "\n(") for m in ms2], rotation=35, ha="right", fontsize=7); ax[0].set_ylabel("median features selected (clean)")
w = .38; ax[1].bar(np.arange(6) - w / 2, J.loc[ms2, "none-0.0"], w, label="clean labels", color="#2a9d8f"); ax[1].bar(np.arange(6) + w / 2, J.loc[ms2, "random-0.2"], w, label="20% random flips", color="#e76f51")
ax[1].set_xticks(range(6)); ax[1].set_xticklabels([nm(m).replace(" (", "\n(") for m in ms2], rotation=35, ha="right", fontsize=7); ax[1].set_ylabel("selection stability (Jaccard)"); ax[1].legend(fontsize=7); save(fig, "fig_sparsity_stability")
# simulation
S = pd.read_csv(f"{ROOT}/results/simulation.csv"); fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.2), sharex=True)
sm = {"L1-Logit": COL["L1-Logit"], "TrimLogit h=0.9": "#8d99ae", "TrimLogit h=1-eps (oracle)": "#e76f51", "AdaTrim": COL["AdaTrim"], "StabSel": COL["StabSel"], "RobustTox": COL["RobustTox"]}
for j, (c, t) in enumerate([("random", "random flips"), ("targeted", "margin flips (worst case)")]):
    for m, col in sm.items():
        g = S[(S.contam == c) & (S.method == m)].groupby("eps"); ax[0, j].plot(g.cos.mean().index, g.cos.mean().values, marker="o", ms=3, color=col, label=m); ax[1, j].plot(g.f1.mean().index, g.f1.mean().values, marker="o", ms=3, color=col)
    ax[0, j].set_title(t, fontsize=9); ax[0, j].axhline(0, color="k", lw=.6); ax[1, j].set_xlabel("flip rate")
ax[0, 0].set_ylabel("cosine(estimate, truth)"); ax[1, 0].set_ylabel("support-recovery F1"); ax[0, 1].legend(fontsize=6); save(fig, "fig_simulation")
# interpretability
a = pd.read_csv(f"{T}/interpretability_alert_overlap.csv"); fig, ax = plt.subplots(figsize=(3.9, 3.6))
for m, c, mk in [("L1-Logit", COL["L1-Logit"], "o"), ("RobustTox", COL["RobustTox"], "s")]:
    g = a[a.method == m]; ax.scatter(g.mean_overlap_random, g.mean_overlap_selected, c=c, marker=mk, s=26, label=m)
ax.plot([0, .6], [0, .6], "k--", lw=.8); ax.set_xlim(0, .6); ax.set_ylim(0, .6); ax.set_xlabel("overlap with alerts: random bits (same frequency)"); ax.set_ylabel("overlap with alerts: selected bits"); ax.legend(fontsize=7)
ax.set_title("Selected substructures vs Brenk/PAINS/NIH alerts", fontsize=8); save(fig, "fig_interpretability")
print("ok")
