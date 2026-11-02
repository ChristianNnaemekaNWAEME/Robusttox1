# RobustTox

A research library and benchmark for **contamination-robust, interpretable toxicity prediction** from molecular structure.

Research question: *when does robust, sparse estimation beat standard models on real toxicity data, and can it be made accurate and explainable at the same time?*

This repository extends the earlier `toxicity-estimator-benchmark` (which showed that a trimmed L1-logistic estimator was the most stable under label noise but not the most accurate) by testing three candidate fixes, adding structured contamination, and mapping the selected features back to substructures.

## Layout
```
robusttox/        features.py (RDKit descriptors + Morgan counts)  split.py (randomised scaffold splits)
                  methods.py (L1-Logit, SLTS-linear, TrimLogit, AdaTrim, StabilitySelection, RobustTox)
                  alerts.py (Brenk / PAINS / NIH via RDKit FilterCatalog)
scripts/          prepare_data.py  run_benchmark.py  run_alerts_baseline.py  run_simulation.py  analyze.py  interpret.py
tests/            pytest suite (stratified trimming, monotone C-steps, split disjointness, support recovery)
docs/             THEORY.md  SKILLS.md  PAPER_OUTLINE.md
results/          tables/*.csv, figures
```

## Reproduce
```bash
pip install -r requirements.txt
bash scripts/download_data.sh                         # four public CSV files -> data_raw/
python scripts/prepare_data.py data_raw               # features, scaffolds, structural alerts -> data/ (about 10 min)
export OMP_NUM_THREADS=1                              # important: avoids thread oversubscription
python scripts/run_benchmark.py 0 2 3 & python scripts/run_benchmark.py 1 2 3   # two workers, 3 scaffold splits (~2 h)
wait
python scripts/run_alerts_baseline.py
python scripts/run_simulation.py 20
python scripts/analyze.py                             # tables in results/tables
python scripts/interpret.py RobustTox                 # substructure / alert analysis
python scripts/make_report_figures.py figs
pytest tests
```
The `results/` folder already contains the raw result files and tables from the run reported in `docs/report/RobustTox_Technical_Report.pdf`, so you can inspect the numbers without rerunning anything.

## Work in progress
The next phase extends this benchmark to FDA's DILIrank list of liver-injury drugs and further Tox21 and ToxCast data, with a comparison against a published graph neural network. Status and scope are in [`ROADMAP.md`](ROADMAP.md). No results from that work are reported here yet.

## Author and licence
Christian N. Nwaeme. MIT licence.

## Data (all public, via MoleculeNet)
Tox21 (12 tasks), ClinTox (CT_TOX), SIDER (Hepatobiliary disorders), and three ToxCast hepatocyte high-content-screening endpoints (HepG2 cell loss 72 h, mitochondrial membrane potential 24 h, oxidative stress 24 h). **The ToxCast endpoints are in-vitro hepatocyte readouts, not clinical drug-induced liver injury**; a DILI set (e.g. DILIrank) is being added; see `ROADMAP.md`.

## Contamination models
`random`: each training label flipped with probability eps. `margin`: the eps-fraction of training compounds that a reference model, fitted on the clean labels, classifies most confidently are flipped (a worst case for the direction of the fit). Test labels are never altered.

## Caveats
* Only the training labels are corrupted; the test set is clean.
* Scaffold splits are randomised per seed; three splits per endpoint is a minimum, not a recommendation for the final paper (use 10).
* Sparse-model feature identity is only meaningful up to the correlation among Morgan bits and descriptors.
