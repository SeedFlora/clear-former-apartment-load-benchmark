# CLEAR-Former: corrected forecasting protocol

Research code accompanying ICoDiT 2026 submission 78, CLEAR-Former: Reconstructing Gaps and Forecasting Apartment Electrical Load from IoT Power-Meter Streams with a Climatology-Anchored Decomposition–Linear Transformer.

This public package contains the original apartment meter workbook, algorithm code, configuration grids, split definitions and aggregate metrics. It also includes the architecture diagram source. The workbook is available in [`data/`](data/README.md). Generated reconstruction/prediction arrays, job caches and fitted models are excluded.

## Data availability

The raw apartment meter workbook, code, configurations, split definitions and aggregate results are publicly available at https://github.com/SeedFlora/clear-former-apartment-load-benchmark. The original workbook is provided in the data directory. Generated prediction series, fitted models and per-run caches are not included.

[Download the original workbook](data/Data%20Kelistrikan%201%20Bulan.xlsx) or read the [dataset notes](data/README.md).

## Layout

- `data/`: the unchanged source workbook and dataset notes.
- `src/`: the original submitted data reader, reconstruction, model implementations, experiment runner and plotting/table code.
- `revision/`: the corrected reconstruction, chronological validation/purge, benchmark, sensitivity and analysis code. `revision/README.md` explains the changes and original run order.
- `aggregate-results/`: configuration and aggregate error/statistical summaries. Files ending `_metrics` or `_summary` are reduced exports; timestamp/placement-level arrays were removed. These are reference results, not the full private run cache.
- `protocol.json`: verified grid constants, split definitions and counts, training settings and parameter grids; no target values or per-run time series.
- `requirements.txt`: the versions recorded in the revision environment, with unrecorded utility versions left unpinned.
- `MANIFEST.sha256`: SHA-256 for the packaged files except the manifest itself.

## Input and reproduction

The original input workbook is included at `data/Data Kelistrikan 1 Bulan.xlsx`. The reader `src/data.py` uses that bundled file, with a fallback to the legacy parent-directory location for existing local projects. Its SHA-256 is `6cdb55d3d00a8470274d20aae5434b788a2ff4f0c67bf05ccdbd34a0bf0aa161`.

The raw source workbook was added to this public repository on 3 November 2026. Building identity and internal research notes are excluded.

Install dependencies in a fresh Python 3.13 environment. The recorded machine used PyTorch 2.11.0 with CUDA 12.8, an NVIDIA RTX 3070 Ti Laptop GPU and an Intel Core i7-12700H. The original neural worker fixes the device to `cuda`; a CPU-only installation cannot run those jobs unchanged. No retraining was performed to prepare this package.

From `revision/`, run the scientific steps listed in its README using the bundled workbook. Their full generated outputs are intentionally absent here. `analysis.py`, figure generation and diagnostics require prediction caches produced by the preceding experiment scripts; the reduced summaries in `aggregate-results/` do not replace those caches. Reference-harvesting utilities, manuscript sources and internal notes are outside this code package.

The revision README records that the analysis, table, plotting and diagram code was written with Claude (Anthropic). Preserve the actual provenance when describing the work.

## Rights

No separate code or data licence is supplied. Third-party dependencies retain their respective licences.
