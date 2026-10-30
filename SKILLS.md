# Skills map: what each phase trains, and the evidence it leaves

| Phase | Skills you will have to use | Where it is in this repository | Evidence it produces |
|---|---|---|---|
| 1. Foundations | Reproducible ML pipelines; scaffold-aware validation; confidence intervals and paired non-parametric tests across tasks; handling severe class imbalance; ROC-AUC vs PR-AUC | `robusttox/split.py`, `scripts/run_benchmark.py`, `scripts/analyze.py` | Tables with 95% bootstrap CIs; per-task intervals over splits; Wilcoxon tests |
| 2. Theory | High-dimensional statistics (L1 penalties, restricted eigenvalues); robust statistics (trimming, breakdown, influence); single-index models; proof writing | `docs/THEORY.md`, `scripts/run_simulation.py` | Labelled propositions, a list of open lemmas, simulations that test each claim |
| 3. Method | Convex and non-convex optimisation (concentration steps, local minima); model selection under label noise; stability selection; honest ablation | `robusttox/methods.py` | Four estimators that each fix one suspected weakness, plus the combination |
| 4. Interpretability | Cheminformatics (Morgan environments, bit collisions, substructure search); structural alerts; permutation null models | `robusttox/alerts.py`, `scripts/interpret.py` | A table of selected substructures vs Brenk/PAINS/NIH alerts with a permutation p-value |
| Cross-cutting | Software engineering (package layout, unit tests, determinism); scientific writing; reporting negative results | `tests/`, `README.md`, `docs/PAPER_OUTLINE.md` | An installable library and a preprint |

## The gap this closes for your endeavor
Your petition's endeavor is *contamination-robust and interpretable* toxicity prediction. Phases 2 and 3 are where the *robust* skills live (mathematical statistics, which is your training); Phase 4 is where the *interpretable* and domain skills live (cheminformatics, which is the part you do not yet have). Phase 4 is also the natural place to ask a toxicologist whether the substructures the model selects make mechanistic sense — a concrete, answerable question that makes a good first email to a domain expert.

## What this does not give you
Toxicology itself (mechanisms, regulatory endpoints, assay design). Computation can flag a substructure; deciding what it means biologically needs a toxicologist. Plan to co-author with one.
