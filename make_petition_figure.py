import os, pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "legend.frameon": False})
ci = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/tables/mean_auc_by_noise_ci.csv"))
conds = ["none-0.0", "random-0.1", "random-0.2", "random-0.3", "margin-0.1", "margin-0.2"]
lab = ["clean", "random\n10%", "random\n20%", "random\n30%", "worst-case\n10%", "worst-case\n20%"]; pos = [0, 1.2, 2.2, 3.2, 4.4, 5.4]
show = [("Random Forest", "Random Forest", "#3d3d3d"), ("L1-Logit", "Plain L1 logistic regression", "#1b6ca8"), ("AdaTrim", "AdaTrim (new trimmed estimator)", "#2a9d8f"), ("SLTS-linear (J-5)", "Earlier estimator", "#d1495b")]
fig, ax = plt.subplots(figsize=(6.6, 3.6))
for m, l, c in show:
    d = ci[ci.method == m].set_index("noise").loc[conds]; ax.errorbar(pos, d.mean_auc, yerr=[d.mean_auc - d.lo, d.hi - d.mean_auc], marker="o", ms=4, lw=1.5, capsize=2, label=l, color=c)
ax.axhline(.5, color="k", ls=":", lw=.8); ax.text(5.45, .51, "chance", fontsize=8); ax.set_xticks(pos); ax.set_xticklabels(lab, fontsize=8.5)
ax.set_ylabel("mean test ROC-AUC (17 endpoints)"); ax.legend(fontsize=8, loc="lower left"); ax.set_title("Accuracy as training labels are corrupted (95% intervals)", fontsize=10)
fig.tight_layout(); fig.savefig("robusttox_auc.png", dpi=220)
