---
title: "RobustTox: A Contamination-Robust, Interpretable Toxicity-Prediction Library and Benchmark"
subtitle: "Technical report on the research project, with theory, simulations, a 17-endpoint benchmark and an interpretability analysis"
author:
  - "Christian N. Nwaeme (Baylor University, Waco, TX, USA)"
  - "Adewale F. Lukman (University of North Dakota, Grand Forks, ND, USA)"
date: "October 29, 2026"
---

**Status of this document.** This is a technical report, not a peer-reviewed paper. All numbers were computed by the code in the accompanying RobustTox repository and are reproduced here from its result tables. The software, analysis and first draft of this report were produced with the assistance of Claude (Anthropic); the authors are responsible for reviewing and verifying them before any publication or submission.

# Abstract

Toxicity labels used to train computational models are noisy, and small, outlier-prone data sets are the norm in preclinical drug-safety work. This report asks: *when does robust, sparse estimation beat standard models on real toxicity data, and can it be made accurate and explainable at the same time?* Starting from the SLTS+LASSO estimator of Nwaeme and Lukman [@nwaeme2023] (the authors' earlier paper) and from three earlier test repositories, we built RobustTox, a Python library with three candidate fixes for the weakness of the earlier estimator (trimming on the logistic loss, trimming level chosen by cross-validation, and stability selection) and a benchmark of 17 public endpoints (Tox21, ClinTox, SIDER and three ToxCast hepatocyte assays) with repeated scaffold splits, two kinds of label contamination and nine methods (2,754 model fits). The adaptive trimmed estimator beats the earlier estimator (+0.028 ROC-AUC on clean labels, 15 of 17 endpoints, p = 0.0002) but does not significantly beat plain L1-penalised logistic regression in any condition. Under worst-case label flips every linear model falls to or below chance while Random Forest keeps an AUC of 0.65. Trimming removes flipped labels only modestly more often than chance. Selected substructures do not overlap standard structural alerts more than random substructures do. A theory note proves several elementary properties, explains why random flips are benign, and shows by simulation that the trimming algorithm fails when it is initialised from a heavily corrupted fit.

# 1. Introduction

Predicting toxicity from chemical structure is a screening step used to decide which compounds deserve expensive animal and clinical testing. Training data are assembled from high-throughput assays, regulatory databases and clinical reports, and they contain errors: assay interference, conflicting results between laboratories, and mis-recorded outcomes. The statistical literature on label noise is large [@frenay2014], and robust regression has a long history, from least trimmed squares [@rousseeuw1984; @rousseeuw1987] to sparse versions for high-dimensional data [@alfons2013] and provable methods under adversarial corruption [@chen2013; @bhatia2015]. What is less clear is whether these ideas help on real toxicity data, where the number of predictors is large, the positive class is rare, and test compounds have new chemical scaffolds.

**Research question.** When does robust, sparse estimation beat standard models on real toxicity data, and can it be made accurate and explainable at the same time?

**Contributions.**

1. An open library (RobustTox) with a scaffold-aware, contamination-aware benchmark over 17 public endpoints.
2. Three candidate fixes for the weakness shown by the earlier estimator, evaluated against that estimator, against plain L1-penalised logistic regression and against tree ensembles.
3. A theory note with results labelled by status (proved, cited, claim, empirical), and a simulation study that tests the claims.
4. An interpretability analysis that maps selected fingerprint bins back to substructures and compares them with the Brenk, PAINS and NIH structural alerts shipped with RDKit [@brenk2008; @baell2010; @rdkit].
5. An honest account of what did *not* work.

# 2. Background

**The starting estimator.** Nwaeme and Lukman [@nwaeme2023] introduced a two-step hybrid for data that are both high-dimensional and contaminated: sparse least trimmed squares (sparse LTS) selects variables while down-weighting outlying observations, and a second-stage learner is fitted on the selected variables; the combination SLTS+LASSO achieved the lowest prediction error among the methods compared on a QSAR data set of anticancer compounds. Sparse LTS is the $\ell_1$-penalised version of least trimmed squares [@rousseeuw1984; @alfons2013], and the LASSO is the $\ell_1$-penalised least-squares estimator [@tibshirani1996].

**Trimmed estimators and their guarantees.** For linear regression with a fraction of corrupted records, trimmed estimators have a replacement breakdown point that depends on the trimming level [@rousseeuw1984; @donoho1983], and estimation and support-recovery bounds for corrupted sparse regression are available [@chen2013; @bhatia2015]. For logistic regression with corrupted *labels only*, the situation is different, as discussed in Section 5.

**Stability selection** [@meinshausen2010] repeatedly refits a sparse model on random half-samples and keeps the variables that are chosen often, which gives smaller and more reproducible feature sets.

**Benchmarks and splits.** MoleculeNet [@wu2018] provides standard toxicity data sets; the Tox21 Data Challenge [@huang2016; @mayr2016] set a community protocol; random train/test splits overstate accuracy on new chemistry [@wallach2018], so this work uses Bemis–Murcko scaffold splits [@bemis1996].

**Structural alerts.** Brenk and co-workers [@brenk2008] compiled substructures associated with unwanted reactivity or toxicity, and Baell and Holloway [@baell2010] compiled substructures that interfere with assays (PAINS). Both are distributed with RDKit [@rdkit].

# 3. Earlier repositories this work builds on

Three public repositories preceded RobustTox. Each is referred to by its name.

**toxicity-estimator-benchmark.** It adapted the published SLTS+LASSO estimator to binary toxicity labels and compared it with LASSO (L1-penalised logistic regression), Random Forest [@breiman2001] and gradient boosting [@friedman2001] on 14 public endpoints (12 Tox21 assays [@huang2016], ClinTox [@gayvert2016] and SIDER hepatobiliary [@kuhn2016]) with three scaffold splits, 42 endpoint-split cases in all, and 0, 10 and 20 percent of training labels flipped at random. Random Forest was the most accurate model at every noise level (mean ROC-AUC 0.818, 0.775, 0.726). SLTS+LASSO lost the least accuracy as labels were corrupted (0.055 AUC lost from 0 to 20 percent with 90 percent of records kept, against 0.078 for LASSO and 0.092 for Random Forest) and selected fewer descriptors, but it was less accurate than LASSO and Random Forest on clean labels (Figure 1, left). Trimming on the raw 0/1 residuals discarded most toxic compounds because they are the minority class, so the adaptation trims within each class; this is an adaptation, not the procedure of the original paper. The repository states its own limitations: random flips are not the gross-outlier contamination the estimator was designed for, baselines were untuned, and only three splits were used.

**tox21-challenge-protocol-reproduction.** It evaluated 12 Tox21 assays in the style of the Tox21 Data Challenge [@huang2016] with RDKit features and three models, and measured how much a random split flatters the models relative to a scaffold split. The original challenge files were not available, so a MoleculeNet copy of Tox21 was used and scores are not comparable with the published leaderboard. A random split overstated accuracy by 0.025 to 0.029 AUC for all three models, and the ranking of models did not change (Figure 2).

**esol-solubility-qsar-reproduction.** It reproduced the classical ESOL aqueous-solubility equation [@delaney2004] with RDKit descriptors on 1,128 compounds (published equation: $R^2 = 0.725$, RMSE 1.10 log units; the predictions stored in the DeepChem file: $R^2 = 0.811$, RMSE 0.91), and then tested SLTS+LASSO in the setting it was designed for, a continuous response with gross outliers. LASSO and Random Forest degraded steadily as outliers were added, SLTS+LASSO that kept 75 percent of the records stayed flat (RMSE 0.877 at 0 percent and 0.880 at 20 percent contamination), and the variant that kept 90 percent worked until the contamination (20 percent) exceeded the trimming (10 percent), when it failed (RMSE 1.36) (Figure 1, right). The rule that the trimming level must exceed the contamination rate comes from this experiment.

![**Figure 1.** Results reported by the earlier repositories. Left: toxicity-estimator-benchmark, mean ROC-AUC against the percentage of flipped training labels. Right: esol-solubility-qsar-reproduction, mean test RMSE against the percentage of gross outliers (lower is better). Values are transcribed from the repositories' result tables.](figs/fig_prior_repos.png){width=6.4in}

![**Figure 2.** tox21-challenge-protocol-reproduction: mean ROC-AUC (12 assays, 3 seeds) under a random split and a scaffold split.](figs/fig_prior_tox21_split.png){width=3.3in}

The toxicity result left a specific weakness: a robust estimator that was the most stable under label corruption but not the most accurate. RobustTox is built to test whether that weakness can be fixed.

# 4. Data, methods and experimental design

## 4.1 Data

All data are public MoleculeNet-format collections [@wu2018] (Table 1). The three ToxCast endpoints [@richard2016] are in-vitro hepatocyte high-content-screening readouts (HepG2 cell loss at 72 h, mitochondrial membrane potential at 24 h, oxidative stress at 24 h). **They are not drug-induced liver injury labels;** the DILIrank list could not be retrieved for this study. The SIDER endpoint is a clinical side-effect category, not an assay.

**Table 1.** Endpoints (after removing molecules RDKit could not parse, and compounds without a label for the endpoint).

{{T_DATA}}

## 4.2 Features and splits

Each molecule is described by 217 RDKit 2-D descriptors and a 512-bin Morgan count fingerprint of radius 2 with a log transform [@rogers2010; @rdkit], 729 numbers in all. Features are median-imputed, standardised on the training set and clipped at five standard deviations. Each endpoint is split 80/20 by Bemis–Murcko scaffold [@bemis1996]; the split is randomised by seed, and three seeds are used. Test labels are never altered.

## 4.3 Contamination models

* **Random flips.** Each training label is flipped independently with probability $\varepsilon \in \{0.1, 0.2, 0.3\}$.
* **Margin flips (worst case).** The $\varepsilon \in \{0.1, 0.2\}$ fraction of training compounds that a reference L1-logistic model, fitted on the *clean* labels, classifies most confidently are flipped. This is a worst case for the direction of a linear fit, and it is built from a linear model, so it is adversarial to linear models by construction.

## 4.4 Methods

**Table 2.** Methods compared.

| Method | Description |
|:---|:---|
| L1-Logit | L1-penalised logistic regression; penalty chosen by 3-fold cross-validated AUC [@tibshirani1996] |
| SLTS-linear | The estimator of the toxicity-estimator-benchmark repository: sparse-LTS concentration steps on the 0/1 label, class-stratified, then L1-logistic regression on the kept records |
| TrimLogit | *Fix 1.* Concentration steps with the logistic loss: fit, remove the highest-loss records within each class (keeping a fraction $h = 0.9$), refit |
| AdaTrim | *Fix 2.* TrimLogit with $h \in \{1, 0.95, 0.9, 0.8\}$ chosen by cross-validated AUC (ties within 0.003 go to more trimming) |
| StabSel | *Fix 3.* Stability selection [@meinshausen2010]: 20 half-sample L1 fits, keep features selected in at least 60% of them, ridge-logistic refit |
| RobustTox | AdaTrim, then stability selection on the kept records, then ridge refit |
| Random Forest | 200 trees, balanced class weights [@breiman2001] |
| Gradient boosting | Histogram gradient boosting, 150 iterations [@friedman2001] |
| Alerts-only | L1-logistic regression on the binary Brenk, PAINS and NIH alert flags only |

Models are implemented with scikit-learn [@pedregosa2011] and RDKit [@rdkit]. Random Forest and gradient boosting use fixed settings and are not tuned.

## 4.5 Evaluation and statistics

The metric is ROC-AUC on the clean test set (PR-AUC is also recorded in the result files). For each condition, AUC is averaged over the three splits within an endpoint, and 95% confidence intervals for the mean over endpoints are bootstrap percentile intervals [@efron1979]. Paired differences between methods are computed per endpoint, with bootstrap intervals and two-sided Wilcoxon signed-rank tests [@wilcoxon1945] across the 17 endpoints. Because the 12 Tox21 endpoints share molecules, the endpoints are not independent and the intervals are optimistic. Feature-selection stability is the mean pairwise Jaccard index of the selected feature sets across splits.

# 5. Theory

Notation: $x_i \in \mathbb{R}^p$, $y_i \in \{0,1\}$, $P(y=1 \mid x) = \sigma(x^\top\beta^*)$ with $\|\beta^*\|_0 = s$; the observed label is $\tilde y_i = y_i \oplus z_i$ for a flip indicator $z_i$. The loss is $\ell_i(\beta) = \log(1 + e^{-(2\tilde y_i - 1)x_i^\top\beta})$. The class-stratified trimmed estimator is
$$(\hat\beta, \hat S) = \arg\min_{\beta,\, S \in \mathcal{S}_h} \sum_{i \in S} \ell_i(\beta) + \lambda\|\beta\|_1,$$
where $\mathcal{S}_h$ is the set of subsets that keep exactly $\lceil h\, n_c \rceil$ records in each observed class $c$. Every statement carries a status: **[Proved here]**, **[Cited]**, **[Claim, proof outstanding]** or **[Empirical]**.

**Proposition 1 (minority-class wipe-out) [Proved here].** If the minority class has share $\pi_1 < 1 - h$ and the fitted values start at the intercept-only value $\bar y = \pi_1$, every minority record has residual $1 - \pi_1 > \pi_1$, the residual of every majority record, so an *unstratified* rule that keeps the $\lceil hn \rceil$ smallest residuals removes the entire minority class. The stratified rule keeps at least one record of each class. Eight of the 12 Tox21 endpoints have a toxic share below 10% (2.9% to 6.3%), which explains the failure of the first prototype of the earlier estimator (AUC 0.65, one non-zero coefficient) before stratification was added.

**Proposition 2 (monotone descent) [Proved here].** With $F(\beta, S) = \sum_{i\in S}\ell_i(\beta) + \lambda\|\beta\|_1$, alternating (a) $\beta_{t+1} \in \arg\min_\beta F(\beta, S_t)$ and (b) $S_{t+1} =$ the $\lceil h n_c \rceil$ smallest losses in each class at $\beta_{t+1}$ gives $F(\beta_{t+1}, S_{t+1}) \le F(\beta_{t+1}, S_t) \le F(\beta_t, S_t)$, so the objective never increases and the iteration terminates (finitely many subsets, objective bounded below by 0). The fixed point is a local minimum of a non-convex problem. A unit test checks the monotonicity numerically.

**Proposition 3 (basic inequality under a sufficient trimming budget) [Proved here].** If, in each observed class, at least $\lceil h n_c \rceil$ records have uncorrupted labels, then for every $\beta^\circ$ there is a feasible clean subset $S^\circ$ with
$$\sum_{i \in \hat S \cap C} \ell_i(\hat\beta) + \lambda\|\hat\beta\|_1 \le \sum_{i \in S^\circ} \ell_i(\beta^\circ) + \lambda\|\beta^\circ\|_1,$$
where $C$ is the set of clean records. *Proof.* $F(\hat\beta, \hat S) \le F(\beta^\circ, S^\circ)$ by optimality, and the corrupted records inside $\hat S$ contribute non-negative terms that can be dropped. $\square$ This is the first step of the analysis of trimmed estimators for linear regression [@alfons2013; @chen2013]. It is **not** an error bound.

**Claim 3′ (estimation and support recovery under adversarial flips) [Claim, proof outstanding].** For Gaussian designs with $n \gtrsim s\log p$ and contamination $\varepsilon \le (1 - h)/2$, we expect $\|\hat\beta - \beta^*\|_2 \lesssim \sqrt{s\log p / n} + \varepsilon$ up to the trimming bias below. The route is Proposition 3 followed by a restricted-eigenvalue bound that holds uniformly over all subsets that remove at most a fraction $2\varepsilon$ of the rows; the combinatorial entropy term of that union bound is the whole difficulty. It should not be called a theorem.

**Two limitations [Claim, with empirical checks].** (i) Trimming removes the records with the largest loss, which are the legitimately unlikely outcomes of a logistic model, so the trimmed estimator is not consistent for $\beta^*$ even with clean labels; the scale of the coefficients is over-estimated. (ii) For linear LTS the replacement breakdown point is $(n - h + 1)/n$ [@rousseeuw1984; @donoho1983] because one high-leverage point can move the fit arbitrarily. With labels only, each corrupted record has bounded influence, since $\|\nabla \ell_i\| \le \|x_i\|$ and $\|\hat\beta\|_1 \le n\log 2/\lambda$ always holds; what fails is the *direction* of the fit.

**Proposition 4 (random flips preserve direction) [Cited].** If $x \sim N(0, \Sigma)$ and each label is flipped independently with probability $\varepsilon < 1/2$, then $E[\tilde y \mid x] = \varepsilon + (1 - 2\varepsilon)\sigma(x^\top\beta^*)$, a single-index model with a strictly increasing link. For Gaussian designs the population minimiser of the logistic loss under such a model is proportional to $\beta^*$ with a positive constant [@brillinger1983; @liduan1989], and $\mathrm{Cov}(\tilde y, x^\top\beta^*) = (1 - 2\varepsilon)\,\mathrm{Cov}(y, x^\top\beta^*)$. So random flips *shrink the signal by the factor $1 - 2\varepsilon$ without rotating the solution*; the sample size needed for support recovery then grows roughly like $(1 - 2\varepsilon)^{-2}$ (a heuristic scaling; sparse-recovery results for single-index models with Gaussian design are in [@plan2013], whose constants should be checked before quoting). This predicts that trimming has little to repair under random flips, which is what the benchmark shows.

**Stability selection [Cited].** Under exchangeability of the noise features and a no-worse-than-random condition, the expected number of falsely selected variables is at most $q^2 / ((2\pi - 1)p)$ [@meinshausen2010], with $q$ the expected number selected on a half-sample and $\pi$ the threshold. For $q \approx 60$, $p = 729$ and $\pi = 0.6$ the bound is about 25, which is larger than the sets actually selected, so it gives no useful guarantee here; the conditions also fail for correlated Morgan bins. Stability selection is therefore used for the smaller and more reproducible feature sets it produces, with no false-discovery claim.

**Open items, in order of value:** (1) the uniform restricted-eigenvalue lemma behind Claim 3′; (2) direction preservation of the trimmed estimator with clean labels and the scale bias as a function of $h$; (3) a finite-sample argument for the cross-validated choice of $h$, since the validation folds are themselves contaminated [@wainwright2019 gives the standard high-dimensional toolkit].

# 6. Simulation study

*Design.* $n = 1000$, $p = 300$, $s = 8$ active coefficients of $\pm 1$, Gaussian design with correlation $0.3^{|i-j|}$, logistic truth, 20 replicates per cell. Random flips and margin flips (the $\varepsilon n$ points with the largest $|x^\top\beta^*|$ are flipped) were applied. The "oracle" trimmed estimator is given a trimming level $h = 1 - \varepsilon - 0.02$.

**Table 3.** Mean cosine between the estimated and true coefficient vectors.

{{T_SIM}}

![**Figure 3.** Simulation results. Top: cosine between the estimate and the truth. Bottom: F1 of support recovery. The F1 score ignores the *sign* of coefficients, so the rise at flip rates 0.3 and 0.4 under margin flips reflects correct *positions* with reversed signs, not recovery.](figs/fig_simulation.png){width=6.4in}

*Findings.* (i) **Random flips** degrade the direction gradually, as Proposition 4 predicts, and trimming gives no gain (0.91 for both AdaTrim and L1-Logit at $\varepsilon = 0.2$). (ii) **Margin flips at $\varepsilon = 0.1$:** trimming with $h \le 1 - \varepsilon$ restores the direction (cosine 0.96 to 0.97 against 0.72 for plain L1) and removes 90 to 97 percent of the flipped labels. (iii) **Margin flips at $\varepsilon \ge 0.2$:** every method fails, including the oracle trimmed estimator, which removes only 21 percent of the flipped labels. Proposition 3 concerns the global minimiser, but the concentration steps are local and start from the full-sample fit, which at $\varepsilon = 0.2$ has lost its direction (cosine about 0). The trimming-budget condition is therefore necessary but not sufficient: *the initial estimate must itself be correlated with the truth.* A robust initialisation is the open algorithmic problem.

# 7. Benchmark results

All results below are from 2,754 model fits: 17 endpoints $\times$ 3 scaffold splits $\times$ 6 conditions $\times$ 9 methods.

## 7.1 Accuracy under contamination

**Table 4.** Mean test ROC-AUC over the 17 endpoints.

{{T_AUC}}

**Table 5.** Mean ROC-AUC with 95% bootstrap confidence intervals over endpoints, for selected methods and conditions.

{{T_CI}}

![**Figure 4.** Mean test ROC-AUC over 17 endpoints under each contamination condition, with 95% bootstrap intervals over endpoints. The dotted line is chance.](figs/fig_auc_by_condition.png){width=6.4in}

**Table 6.** Mean AUC lost relative to clean labels.

{{T_DROP}}

The main observations are as follows.

1. **Random Forest is the most accurate method in every condition** (clean 0.797; 0.719 at 20 percent random flips), and gradient boosting is next on clean data.
2. **Under random flips all linear methods lose accuracy in the same way** (about 0.03, 0.07 and 0.12 AUC at 10, 20 and 30 percent for plain L1-Logit), as Proposition 4 predicts. The earlier estimator loses less (0.040 at 20 percent) only because it starts lower; its AUC is not different from that of L1-Logit at any noise level.
3. **Under margin flips every linear model falls to or below chance at 20 percent** (L1-Logit 0.432, AdaTrim 0.429, RobustTox 0.426, SLTS-linear 0.390), while Random Forest keeps 0.646 and gradient boosting 0.646. AUC below 0.5 means the model ranks compounds in the *wrong* order, because the flipped compounds carried the strongest signal.
4. **An alerts-only model reaches 0.594 on clean data**, against 0.751 for L1-Logit on descriptors and fingerprints, so the models learn much more than the standard alerts encode.

## 7.2 Do the fixes work?

**Table 7.** Paired differences in mean AUC against the earlier estimator (SLTS-linear), per condition: mean difference [95% bootstrap CI]; endpoints won out of 17; Wilcoxon p.

{{T_PAIR_J5}}

**Table 8.** Paired differences in mean AUC against plain L1-Logit.

{{T_PAIR_L1}}

![**Figure 5.** Paired AUC differences against L1-Logit (mean and 95% bootstrap interval over endpoints). Points to the right of zero would indicate an improvement.](figs/fig_paired_vs_l1.png){width=6.4in}

* **Against the earlier estimator,** AdaTrim improves clean-label AUC by +0.028 (15 of 17 endpoints; p = 0.0002), under margin flips by +0.030 (10 percent; p = 0.027) and +0.039 (20 percent; p = 0.011), and under 10 percent random flips by +0.012 (p = 0.009). At 20 and 30 percent random flips the earlier estimator is as good as any trimmed variant.
* **Against plain L1-Logit,** no method is significantly better in any condition. AdaTrim is statistically indistinguishable from it in every condition (differences from -0.003 to +0.004). Stability selection and RobustTox cost 0.01 to 0.035 AUC (significant in most of the clean and random-flip conditions).
* **Against Random Forest,** no linear method comes close under margin flips.

So the weakness of the earlier estimator, lower accuracy than the plain L1 model on clean labels (-0.026, p = 0.001), is repaired by AdaTrim, but the repair brings it only to parity with a model that has no robustness machinery at all.

![**Figure 6.** Clean-label ROC-AUC for every endpoint and method (mean of three scaffold splits).](figs/fig_endpoint_heatmap.png){width=6.4in}

## 7.3 Why trimming does not help more

**Table 9.** Trimming diagnostics for AdaTrim: share of records trimmed, share of the flipped labels that trimming removed, the ratio of the two (1.0 would be chance), and the share of trimmed records that were truly flipped.

{{T_TRIM}}

Under random flips trimming removes flipped labels 1.2 to 1.7 times more often than chance, an enrichment that is real but far too weak to clean the data. Under margin flips the enrichment is *below* chance (0.8 and 0.4): the flipped compounds are the ones the fitted model is most confident about, and after the model has absorbed them they no longer look like high-loss records. This is the mechanism behind the failure in Section 6. AdaTrim also chose the least trimming on clean data and the most trimming under heavy random noise (Figure 7, right), which is the intended behaviour, but the cross-validation criterion cannot tell margin flips from clean data.

![**Figure 7.** Left: share of flipped labels removed by trimming (bars) against the share of records trimmed (black marks). Right: the trimming level $h$ chosen by AdaTrim in each condition.](figs/fig_trimming_diagnostics.png){width=6.4in}

## 7.4 Sparsity and selection stability

**Table 10.** Median number of selected features and selection stability across splits.

{{T_SPARS}}

![**Figure 8.** Left: median number of selected features with clean labels. Right: selection stability (mean Jaccard index across splits) with clean labels and with 20 percent random flips.](figs/fig_sparsity_stability.png){width=6.4in}

Stability selection and RobustTox use far fewer features (a median of 8, against 38 for L1-Logit on clean labels). Selection stability is highest for stability selection on clean labels (0.535, against 0.447 for L1-Logit), but with 20 percent random flips *all* methods fall to about 0.08 to 0.13 and none is better than plain L1-Logit (0.127). The reproducibility benefit of the extra machinery does not survive label noise in this benchmark.

# 8. Interpretability: selected substructures against structural alerts

*Method.* For each endpoint, the Morgan bins with positive coefficients (associated with toxicity) selected on the clean labels were ranked by how often the split selected them, and the ten most frequent were taken. For each bin, the atom environments that set the bin were recovered with RDKit, and the *alert-overlap rate* was computed: the share of molecules carrying the bin in which an environment of radius 1 or 2 shares atoms with a matched Brenk, PAINS or NIH alert [@brenk2008; @baell2010; @rdkit] (up to 400 molecules per bin). The mean overlap rate of the ten selected bins was compared with the mean of 60 random sets of ten bins carried by at least 30 molecules (permutation test).

**Table 11.** Alert overlap of selected bins against random bins.

{{T_INT}}

![**Figure 9.** Mean alert-overlap rate of the selected bins against random bins, one point per endpoint. Points below the dashed line mean the selected bins overlap alerts less than random bins do.](figs/fig_interpretability.png){width=3.9in}

The selected bins do **not** overlap the standard alerts more than random bins do (differences of -0.035 for L1-Logit and -0.059 for RobustTox; no endpoint is significant at 0.05; the Fisher-combined p-values are 0.97 and 1.00). The Wilcoxon test across endpoints suggests a slightly *lower* overlap (p = 0.035 and 0.042), which should not be over-read. One mechanistically sensible example exists: for the Tox21 SR-ARE endpoint, an electrophile-stress response, one of the three bins selected in every split is a carbon–carbon double-bond environment that overlaps the Brenk Michael-acceptor and NIH acrylate alerts in 61 percent of the molecules that carry it (Table 12); the other two bins overlap alerts in 7 and 38 percent of their molecules.

**Table 12.** Top selected bins for SR-ARE (RobustTox).

{{T_BITS}}

Caveats: the 512-bin fingerprint folds many environments into one bin, so the substructure shown is only the most frequent environment; Brenk and PAINS alerts are not toxicity labels (PAINS flag assay interference); only the top ten bins were analysed; and RobustTox selected at least three positive bins in only 11 of the 17 endpoints. Whether the selected substructures make mechanistic sense needs review by a toxicologist.

# 9. Discussion

**Answer to the research question, as far as the evidence goes.** On these 17 public endpoints, robust sparse estimation of the kinds tested did not beat plain L1-penalised logistic regression, either with clean labels or with random or worst-case label flips. It did beat the earlier estimator of the toxicity-estimator-benchmark repository. Under random flips the data agree with the theory (Proposition 4): the signal is attenuated but not rotated, so there is little for trimming to fix. Under worst-case flips, trimming cannot find the corrupted labels, and the one family of methods that is robust here is tree ensembles. Sparse models can be made accurate to within 0.01 to 0.035 AUC of the dense L1 model with about a fifth of the features, but the interpretability of those features, measured against standard alerts, is not evidently better than random.

**What was not achieved.** No fix beat the plain baseline; no linear method survived margin flips; no theorem was proved for estimation or support recovery under adversarial flips (Claim 3′ is open); no real cross-database label conflicts were analysed; no drug-induced liver injury set was used.

**Threats to validity.**

1. *Margin flips are built from a linear model*, so they are adversarial to linear models by construction, and part of Random Forest's advantage there is by design. A contamination model that depends on no model family (assay-interference clusters, real label conflicts between databases) is needed.
2. *Three scaffold splits per endpoint* is a minimum; ten should be used before submission. The 12 Tox21 endpoints share molecules, so bootstrap intervals over endpoints are optimistic.
3. *Hyperparameters* (penalty grid, trimming grid, 20 half-samples, threshold 0.6) were fixed in advance and not tuned on test data, but they were not optimised either, and Random Forest and boosting were not tuned. A tuned comparison could change the picture.
4. *Hepatotoxicity.* The ToxCast endpoints are in-vitro readouts and the SIDER endpoint is a clinical side-effect category; neither is drug-induced liver injury.
5. *Comparability with the earlier repositories.* Numbers in Section 3 and Section 7 differ (for example, L1-penalised logistic regression scores 0.772 there and 0.751 here) because the endpoint sets and random splits differ; they should not be compared directly.

**Next experiments, in order.** (1) A robust initialisation for the concentration steps, for example multiple starts from sub-samples or starting from out-of-fold predictions of a tree ensemble. (2) Real contamination: compare labels for the same compounds across Tox21, ToxCast and ChEMBL and treat disagreements as noise. (3) Robust and calibrated tree ensembles as the main competitor. (4) A drug-induced liver injury set and a review of the Section 8 substructures by a toxicologist. (5) Ten scaffold splits and the proof of the uniform restricted-eigenvalue lemma.

# 10. Reproducibility and availability

The repository contains the library (`robusttox/`), the benchmark and analysis scripts (`scripts/`), the cached features, five unit tests, the result tables and the theory note. Software: Python 3.13, RDKit 2026.09.1, scikit-learn 1.9.1. To reproduce: prepare the data (`prepare_data.py`), run the benchmark with two single-threaded workers (`run_benchmark.py`, about two hours on two cores), then `run_alerts_baseline.py`, `run_simulation.py`, `analyze.py` and `interpret.py`. The code has not yet been published; its URL should be added here once it is.

# Appendix A. Skills developed by each phase

**Table A1.** Skills each project phase requires, and where the evidence is in the repository.

| Phase | Skills | Evidence in the repository |
|:---|:---|:---|
| 1. Foundations | Reproducible pipelines; scaffold-aware validation; bootstrap intervals and paired non-parametric tests across endpoints; class imbalance; ROC-AUC against PR-AUC | `split.py`, `run_benchmark.py`, `analyze.py` |
| 2. Theory | High-dimensional statistics ($\ell_1$ penalties, restricted eigenvalues); robust statistics (trimming, breakdown, influence); single-index models; proof writing | Section 5, `run_simulation.py` |
| 3. Method | Optimisation (concentration steps, local minima); model selection under label noise; stability selection; honest ablation | `methods.py` |
| 4. Interpretability | Cheminformatics (Morgan environments, bin collisions, substructure search); structural alerts; permutation null models | `alerts.py`, `interpret.py` |
| Cross-cutting | Software engineering (package layout, unit tests); scientific writing; reporting negative results | `tests/`, this report |

Toxicology itself (mechanisms, regulatory endpoints, assay design) is not developed by this project; a toxicologist co-author or reviewer is needed for the interpretability findings.

# References

::: {#refs}
:::
