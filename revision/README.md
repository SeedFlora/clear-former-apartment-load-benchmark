# CLEAR-Former revision: corrected protocol and new experiments

This folder holds the code and results of the ICoDiT 2026 revision. The
submitted pipeline in `../src` is left unchanged, so the submitted numbers can
still be reproduced (`reproduce_submitted.py` re-runs it and matches the
submitted CLEAR-Former MAE to the last digit: 5.091749509175618 kW at 1 h and
15.394672419230144 kW at 24 h).

The revision code, tables, figures and BibTeX in this folder were written with
Claude (Anthropic); see the paper's AI Usage Declaration.

## What the revised protocol changes

| # | Submitted pipeline (`../src`) | Revised protocol (`common.py`) |
|---|---|---|
| 1 | Hour-of-day climatology and clipping range of the gap reconstruction estimated from every observed hour, the 48 test hours included | Estimated from observed hours before the first test target (grid index < 672, i.e. before 2026-06-27 20:00) |
| 2 | Noise term `N(0, (0.5 sigma_r)^2)`; `sigma_r` was taken against an interpolant that passes through every observation, so it was exactly 0 and no noise was ever added | Term removed; fills are `c[h(t)] + tau(t)` |
| 3 | Covariate fills not clipped | Every channel clipped to its pre-test observed range (11 covariate fills change) |
| 4 | Early stopping on all validation targets, most of them reconstructed (0 of 93 day-ahead validation windows fully observed) | Validation loss and selection on observed target hours only |
| 5 | Day-ahead training windows whose targets overlap validation targets kept | 23 such windows purged |
| 6 | One configuration per model; CLEAR-Former's was fixed while test scores were visible | Each learning model selects from a predeclared grid (`train.py: GRIDS`) by validation MAE on observed hours; the test block is read once, after selection |
| 7 | SVR fitted on raw kW targets | SVR fitted on standardised targets |

Everything else is reused from `../src` unchanged: input window (24 h),
horizons (1 h, 24 h), features, model code, Adam (lr 1e-3, L2 weight decay
1e-4, ReduceLROnPlateau factor 0.5 / patience 8), SmoothL1 loss on
standardised targets, batch 32, at most 200 epochs, patience 25, gradient
clipping 3.0, seeds 0-4, 5-seed ensemble.

## Files

| File | Purpose |
|---|---|
| `common.py` | reconstruction, windows, chronological split with purge, masked metrics |
| `train.py` | model factory, hyperparameter grids, training loop with masked validation |
| `jobs.py` | worker functions for the process pools |
| `reproduce_submitted.py` | re-runs the submitted CLEAR-Former protocol |
| `recon_validation.py` | masking experiment for the gap reconstruction |
| `run_benchmark.py` | grid, validation-only selection, single test evaluation of all 11 models |
| `run_followup.py` | ablation, reconstruction arms and the correction chain (exploratory, after selection) |
| `run_sensitivity.py`, `robust.py` | post hoc sensitivity of the neural ranking to the validation design |
| `analysis.py` | metrics, per-origin variability, Diebold-Mariano tests, window composition, parameter counts |
| `check_data.py` | channel types and the power-factor encoding of the Excel export, gap bounds (`results/data_and_ref_checks.json`) |
| `val_diagnostics.py` | validation MAE by target type, validation-test rank correlation, autocorrelation of the DM loss differentials (`val_by_target_type.json`, `val_test_rank.json`, `dm_autocorr.json`) |
| `summarise_sensitivity.py` | the sensitivity table, rank correlations, winners and the matched-validation fill effect (`sensitivity_summary.json`) |
| `make_tables.py`, `export_figdata.py` | LaTeX tables and CSV figure data from `results/` |
| `make_figures.py` | Fig. 1 (record and protocol) and Fig. 3 (per-origin day-ahead MAE) from `figdata/` |
| `figures/fig2_architecture.tex` | Fig. 2 (CLEAR-Former data flow), TikZ, compiled with `pdflatex` |
| `figures/CAPTIONS_draft.md` | caption drafts with the source of every number |
| `references/check_refs.py`, `ref_checks.json` | Crossref/OpenAlex/arXiv records of every DOI-bearing reference |
| `references/build_bib.py`, `references_revised.bib`, `recency.json` | verified bibliography (16 submitted + 14 recommended + 11 optional entries) and its share of recent references |
| `references/REFERENSI.md`, `ref_review.json` | per-reference guide (Indonesian) and the verification records (evidence quotes, skeptic notes) |
| `references/submitted_references.bib` | the 16 references of the submitted manuscript, input of `build_bib.py` |
| `results/` | every JSON and NPZ the revision reports (`summary.json` collects the numbers) |
| `logs/` | console logs of each run |

## Reproduce

Run from this folder, in this order (PowerShell and bash alike):

```text
python check_data.py                   # channel and gap checks of the Excel export
python reproduce_submitted.py          # about 5 min
python recon_validation.py             # 102 s, CPU
python run_benchmark.py --workers 6    # 620 s on an idle RTX 3070 Ti Laptop GPU
python run_followup.py                 # 2,025 s (GPU shared with other jobs)
python analysis.py
python val_diagnostics.py              # after run_benchmark, analysis and run_followup
python run_sensitivity.py              # post hoc; one process per job, restartable
python summarise_sensitivity.py        # after run_sensitivity
python make_tables.py                  # tables/*.tex from results/
python export_figdata.py               # figdata/*.csv for the figures
cd figures; pdflatex fig2_architecture.tex; cd ..
python make_figures.py                 # Figs. 1 and 3 (+ PNG previews) and figures_manifest.json
python references/check_refs.py        # network: Crossref, OpenAlex, arXiv (generic User-Agent)
python references/build_bib.py         # references/references_revised.bib, recency.json
```

`check_data.py`, `val_diagnostics.py` and `summarise_sensitivity.py` compare what
they compute with the stored result files and stop if a value differs. The
neural jobs need an NVIDIA GPU with CUDA (the device is fixed to `cuda` in
`jobs.py`). Result keys keep the old component name: `-Exogenous MLP` is the
variant without the feature MLP f_z.

Input data: `../../Data Kelistrikan 1 Bulan.xlsx` (457 rows, SHA-256
`6cdb55d3d00a8470274d20aae5434b788a2ff4f0c67bf05ccdbd34a0bf0aa161`), read by
`../src/data.py`. The unchanged source workbook is included in the public release at
`../data/Data Kelistrikan 1 Bulan.xlsx`; the reader also supports the legacy
local path above.

`run_sensitivity.py` uses `robust.py`, which runs every job in a fresh process,
stores each result as it finishes and retries jobs that fail when another
program fills the GPU; a retried job starts from the same seed.

Hardware: Intel Core i7-12700H (14 cores), 32 GB RAM, NVIDIA GeForce RTX 3070
Ti Laptop GPU (8 GB), CUDA 12.8. Software: Python 3.13.5, PyTorch 2.11.0+cu128,
NumPy 2.4.6, scikit-learn 1.9.0, statsmodels 0.14.6, XGBoost 3.3.0,
Prophet 1.3.0, pandas 2.3.3. GPU runs are deterministic on this machine:
repeating a run with the same seed gives identical predictions.
