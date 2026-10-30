# RobustTox: theory note (working draft)

**Status of each statement.** Every result below is labelled **[Proved here]** (elementary; full proof given), **[Cited]** (a published theorem we apply; check the exact hypotheses against the source before you submit), **[Claim, proof outstanding]** (we believe it, we give the route, but the proof is not written), or **[Empirical]** (checked only by simulation or by the benchmark). Please keep these labels in any paper; they are what a referee will test.

## 0. Setting and estimators

Data $(x_i,y_i)_{i\le n}$, $x_i\in\mathbb R^p$, $y_i\in\{0,1\}$, with $P(y=1\mid x)=\sigma(x^\top\beta^*)$, $\sigma(t)=1/(1+e^{-t})$, and $\|\beta^*\|_0=s\ll p$. We observe $\tilde y_i=y_i\oplus z_i$ with $z_i\in\{0,1\}$ a flip indicator. Two contamination models:

* **(C1) random flips:** $z_i\sim\text{Bernoulli}(\varepsilon)$, independent of $(x_i,y_i)$, $\varepsilon<1/2$.
* **(C2) arbitrary flips:** an adversary chooses a set $M$, $|M|=m=\varepsilon n$, and sets $\tilde y_i$ for $i\in M$ (we test the worst case "flip the most confidently labelled points", called *margin* contamination in the benchmark).

Write $\ell_i(\beta)=\log(1+e^{-(2\tilde y_i-1)x_i^\top\beta})$ and $n_c=|\{i:\tilde y_i=c\}|$.

* **L1-Logit:** $\hat\beta=\arg\min_\beta \sum_i\ell_i(\beta)+\lambda\|\beta\|_1$.
* **Class-stratified trimmed L1-logistic (TrimLogit):** for a trimming level $h\in(0,1]$,
$$(\hat\beta,\hat S)=\arg\min_{\beta,\;S\in\mathcal S_h}\ \sum_{i\in S}\ell_i(\beta)+\lambda\|\beta\|_1,\qquad \mathcal S_h=\{S:\ |S\cap\{\tilde y=c\}|=\lceil h\,n_c\rceil\ \ \forall c\}.$$
* **J-5 estimator (SLTS-linear):** the same, but the subset $S$ is chosen by the residuals of a *linear* LASSO fit to the 0/1 label, followed by an L1-logistic fit on $S$.
* **AdaTrim:** $h$ chosen from a grid by cross-validated AUC. **StabSel:** features kept if selected in at least a fraction $\pi$ of $B$ random half-samples, then ridge-logistic refit. **RobustTox:** AdaTrim, then StabSel on the kept records.

## 1. Why the J-5 estimator needed class stratification

**Proposition 1 (minority-class wipe-out) [Proved here].** Let the minority class have share $\pi_1<1-h$. Take any start whose fitted values are the intercept-only value $\bar y=\pi_1$. Then every minority record has residual $|1-\pi_1|=1-\pi_1$ and every majority record has residual $\pi_1$. Because $1-\pi_1>\pi_1$, all minority records are ranked as the largest residuals, and an *unstratified* rule that keeps the $\lceil hn\rceil$ smallest residuals, with $n-\lceil hn\rceil\ge\pi_1 n$, deletes the entire minority class. The stratified rule keeps $\lceil h n_c\rceil\ge1$ records of each class, so the fit is always defined.

*Remark.* Eight of the twelve Tox21 tasks have toxic prevalence below 10% (2.9% to 6.3%) and are therefore exposed to this failure at $h=0.9$. It is the failure we observed in the first J-5 prototype (AUC 0.65, one non-zero coefficient), which is why stratification is in the J-5 estimator at all.

## 2. Concentration steps converge

**Proposition 2 (monotone descent) [Proved here].** Let $F(\beta,S)=\sum_{i\in S}\ell_i(\beta)+\lambda\|\beta\|_1$. Starting from any $S_0\in\mathcal S_h$, iterate: (a) $\beta_{t+1}\in\arg\min_\beta F(\beta,S_t)$; (b) $S_{t+1}=$ for each class, the $\lceil hn_c\rceil$ records with the smallest $\ell_i(\beta_{t+1})$. Then $F(\beta_{t+1},S_{t+1})\le F(\beta_{t+1},S_t)\le F(\beta_t,S_t)$, so $F$ is non-increasing; since $\mathcal S_h$ is finite and $F$ is bounded below by $0$, the iteration reaches a fixed point after finitely many steps.

*Proof.* (a) is a minimisation over $\beta$ at fixed $S_t$, hence $F(\beta_{t+1},S_t)\le F(\beta_t,S_t)$. For (b), $\mathcal S_h$ is a product over classes of "choose $k_c$ records", and for fixed $\beta$ the sum $\sum_{i\in S}\ell_i(\beta)$ over such a product is minimised by choosing the $k_c$ smallest losses in each class. Hence $F(\beta_{t+1},S_{t+1})\le F(\beta_{t+1},S_t)$. $\square$

*Caveat.* The fixed point is a local minimum of a non-convex problem. The software runs at most three steps from the full sample; the unit test `test_c_steps_do_not_increase_objective` checks the proposition numerically. Different starts can give different fixed points; the J-5 code used several random starts.

## 3. The trimming budget and an oracle inequality

**Proposition 3 (basic inequality under a sufficient trimming budget) [Proved here].** Let $C$ be the set of records with uncorrupted labels. Suppose, for each observed class $c$, there are at least $\lceil hn_c\rceil$ clean records, i.e. the number of corrupted records in each observed class is at most $n_c-\lceil hn_c\rceil\ (\approx(1-h)n_c)$. Then there is a feasible subset $S^\circ\in\mathcal S_h$ with $S^\circ\subseteq C$, and for every $\beta^\circ$:
$$\sum_{i\in\hat S\cap C}\ell_i(\hat\beta)+\lambda\|\hat\beta\|_1\ \le\ \sum_{i\in S^\circ}\ell_i(\beta^\circ)+\lambda\|\beta^\circ\|_1 .$$
Moreover $|\hat S\cap C|\ge|\hat S|-m$.

*Proof.* $(\hat\beta,\hat S)$ minimises $F$ over all feasible pairs, so $F(\hat\beta,\hat S)\le F(\beta^\circ,S^\circ)$. The corrupted records inside $\hat S$ contribute non-negative terms $\ell_i(\hat\beta)\ge0$; dropping them from the left-hand side only decreases it. The count follows from $|\hat S\cap M|\le m$. $\square$

*What this does and does not give.* It says that when the trimming budget covers the contamination, whatever corrupted records survive in $\hat S$ cannot hurt the fit to the surviving clean records beyond what an oracle fitted on a clean subset would achieve. It is the first step in the analysis of trimmed estimators for linear regression (see the analyses of sparse LTS by Alfons, Croux and Gelper, 2013, and of robust sparse regression under corruption by Chen, Caramanis and Mannor, 2013). **It is not an error bound.** An $\ell_2$ or support-recovery bound needs, in addition, restricted strong convexity on *every* large subset of the sample (uniformly over the choice of $S$) and a compatibility condition; for Gaussian designs this is standard in the linear case.

**Claim 3′ (estimation and support recovery under (C2)) [Claim, proof outstanding].** For $x\sim N(0,\Sigma)$ with bounded condition number, $n\gtrsim s\log p$, $\varepsilon\le(1-h)/2$ and $\lambda\asymp\sqrt{\log p/n}$, the trimmed estimator satisfies $\|\hat\beta-\beta^*\|_2\lesssim\sqrt{s\log p/n}+\varepsilon$ (up to the trimming bias of §4). Route: Proposition 3, then a uniform-over-subsets restricted-eigenvalue bound for the logistic Hessian on subsets of size $\ge(1-2\varepsilon)n$ (union bound over $\binom{n}{\varepsilon n}$ subsets; the entropy term is $\varepsilon\log(1/\varepsilon)n$, which has to be dominated by the curvature gain). Do not state this as a theorem in the paper until the uniform bound is written and checked.

## 4. Two honest limitations

**Bias from trimming (even with clean labels) [Claim, with an empirical check].** Trimming removes the records with the *largest* loss, which are exactly the legitimately unlikely outcomes of a logistic model. So the trimmed score equations are not unbiased at $\beta^*$ when $\varepsilon=0$, and the trimmed estimator is not consistent for $\beta^*$: it over-estimates the scale of the coefficients (a sharper fit to a truncated sample). For symmetric Gaussian designs the *direction* may still be preserved, but we have not proved it; the simulation reports the cosine between $\hat\beta$ and $\beta^*$ to see how large the effect is. This is the price of trimming and the reason for choosing $h$ adaptively.

**A breakdown statement needs the right notion.** For linear LTS the replacement breakdown point is $(n-h+1)/n$ (Rousseeuw, 1984), because a single bad record with high leverage can move the fit arbitrarily. For *labels only*, with clean $x$, each corrupted record has bounded influence: $\|\nabla\ell_i\|\le\|x_i\|$, and with the $\ell_1$ penalty $\|\hat\beta\|_1\le n\log 2/\lambda$ always. So there is no "cliff to infinity". What can fail is the **direction**: if a fraction $\delta$ of the *retained* records is corrupted in a coordinated way, the gradient at $\beta^*$ is shifted by up to $\delta\,\mathbb E\|x\|$-order terms, and when that exceeds the curvature the minimiser rotates. We therefore state the operational rule that the data bear out: **the trimming level $1-h$ must exceed the contamination rate** (Proposition 3's hypothesis). The simulation (`results/simulation.csv`) and the ESOL regression experiment in Exhibit J-7 (where the estimator broke down at 90% trimming level against 20% contamination, i.e. when too little was trimmed) are checks, not proofs.

## 5. What random flips do, and why they are the wrong stress test

**Proposition 4 (direction is preserved under (C1), Gaussian design) [Cited, applied here].** If $x\sim N(0,\Sigma)$ and $y\sim\text{Bernoulli}(\sigma(x^\top\beta^*))$, then under (C1)
$$\mathbb E[\tilde y\mid x]=\varepsilon+(1-2\varepsilon)\,\sigma(x^\top\beta^*)=:g(x^\top\beta^*),$$
a single-index model with a strictly increasing link $g$ (slope scaled by $1-2\varepsilon>0$). For Gaussian $x$ the population minimiser of the logistic loss (an *arbitrarily mis-specified* convex GLM loss) is $\bar\beta=\kappa\beta^*$ with $\kappa>0$ (Brillinger, 1983; Li and Duan, 1989), because $\mathbb E[x\mid x^\top\beta^*]$ is linear in $x^\top\beta^*$. Also $\operatorname{Cov}(\tilde y,x^\top\beta^*)=(1-2\varepsilon)\operatorname{Cov}(y,x^\top\beta^*)$.

*Consequences.* (i) Random flips **shrink the signal by the factor $(1-2\varepsilon)$ but do not change the direction or support** of the population solution; so plain L1-logistic regression should remain *consistent for the support*, needing roughly $(1-2\varepsilon)^{-2}$ times more data **[heuristic scaling, from the signal-to-noise ratio; sparse-recovery theory for single-index models with Gaussian design is in Plan and Vershynin, 2013; check their constants before quoting]**. (ii) Hence there is little for trimming to fix under (C1), which is a direct explanation of the J-5 finding that the robust estimator lost the least accuracy but was not the most accurate. (iii) The right stress tests are contamination that **depends on $x$ or on the true label** — (C2) margin flips here, assay-interference clusters and cross-database label conflicts in the next phase. These are the cases where Claim 3′ is needed.

## 6. Stability selection

**Theorem (Meinshausen and Bühlmann, 2010) [Cited].** Under exchangeability of the noise features and a "no worse than random guessing" condition for the base selector, with $q$ the expected number of variables selected on a half-sample and threshold $\pi\in(1/2,1)$, the expected number of falsely selected variables is at most $q^2/\big((2\pi-1)p\big)$.

For $q\approx 60$, $p=729$, $\pi=0.6$ the bound is $\approx 25$, which is **vacuous** here (it is larger than the number of features we select). We therefore use stability selection for what it demonstrably buys — much smaller, much more reproducible feature sets (measured by the Jaccard index across splits in the benchmark) — and make no formal false-discovery claim. The exchangeability condition also fails for correlated Morgan bits and descriptors.

## 7. What to prove next, in order of value

1. The uniform restricted-eigenvalue bound behind Claim 3′ (a self-contained lemma; the combinatorial entropy term is the whole difficulty).
2. Direction preservation of the trimmed estimator at $\varepsilon=0$ for Gaussian designs (symmetry argument), and a quantified scale bias as a function of $h$.
3. A finite-sample bound for the CV choice of $h$ in AdaTrim (needs a model for how contamination moves the validation AUC; at present this is only argued heuristically in the code comments).

## 8. What the experiments say about the theory (measured; see `results/`)

*Simulation (n=1000, p=300, s=8, 20 replicates per cell; `results/simulation.csv`).* Mean cosine between $\hat\beta$ and $\beta^*$ for plain L1-Logit:

| flip rate $\varepsilon$ | 0 | 0.1 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|---|
| random flips (C1) | 0.99 | 0.98 | 0.91 | 0.72 | 0.22 |
| margin flips (C2) | 0.99 | 0.72 | -0.01 | -0.52 | -0.90 |

* **Random flips (supports Proposition 4):** the direction degrades gradually, as expected from signal attenuation by $1-2\varepsilon$ at fixed $n$, and not by rotation. Trimming gives no gain in direction (AdaTrim 0.91 and plain 0.91 at $\varepsilon=0.2$).
* **Margin flips at $\varepsilon=0.1$ (supports the trimming budget, Proposition 3):** TrimLogit with $h\le 1-\varepsilon$ restores the direction (cosine 0.96–0.97 against 0.72 for plain L1), and removes 90–97% of the flipped labels.
* **Margin flips at $\varepsilon\ge0.2$ (a limit of the algorithm, not of Proposition 3):** every method fails, *including TrimLogit with $h$ set to $1-\varepsilon-0.02$*, which removes only 21% of the flipped labels. Proposition 3 concerns the global minimiser; the concentration steps are local and start from the full-sample fit, which at $\varepsilon=0.2$ has already lost its direction (cosine $\approx0$). So the budget condition is necessary, not sufficient: **the initial estimate must itself be correlated with $\beta^*$.** A robust initialisation that does not depend on the corrupted fit is the open algorithmic problem. This sharpens, and should replace, the "trimming level must exceed contamination" rule of §4 in any write-up.

*Real data (17 endpoints, 3 scaffold splits, `results/tables/`).* Details in `docs/PAPER_OUTLINE.md`. Two measurements bear directly on the theory: under random flips the fraction of flipped labels that trimming removes (17–20% at a 10–15% trimming rate) is barely above chance, so the trimmed fit is mostly *not* identifying corrupted records; under margin flips it removes 3–8%.
