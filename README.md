# CLEAR-Former: corrected forecasting protocol

Research code accompanying ICoDiT 2026 submission 78, CLEAR-Former: Reconstructing Gaps and Forecasting Apartment Electrical Load from IoT Power-Meter Streams with a Climatology-Anchored Decompositionâ€“Linear Transformer.

This public package contains algorithm code, configuration grids, anonymous split definitions and aggregate metrics. It also includes the architecture diagram source. The original one-month apartment meter workbook and every raw or reconstructed load trace, true target array, per-run prediction, job cache and fitted model are excluded. Code availability does not make the meter data public.

## Layout

- `src/`: the original submitted data reader, reconstruction, model implementations, experiment runner and plotting/table code.
- `revision/`: the corrected reconstruction, chronological validation/purge, benchmark, sensitivity and analysis code. `revision/README.md` explains the changes and original run order.
- `aggregate-results/`: configuration and aggregate error/statistical summaries. Files ending `_metrics` or `_summary` are reduced exports; timestamp/placement-level arrays were removed. These are reference results, not the full private run cache.
- `protocol.json`: verified grid constants, split definitions and counts, training settings and parameter grids; no target values or per-run time series.
- `requirements.txt`: the versions recorded in the revision environment, with unrecorded utility versions left unpinned.
- `MANIFEST.sha256`: SHA-256 for the packaged files except the manifest itself.

## Input and reproduction

Full reproduction requires the authors' private input workbook. The original reader `src/data.py` expects `Data Kelistrikan 1 Bulan.xlsx` in the parent directory of this package. Do not commit that workbook or generated meter-level artifacts. The exact workbook recorded in the local revision has SHA-256 `6cdb55d3d00a8470274d20aae5434b788a2ff4f0c67bf05ccdbd34a0bf0aa161`.

The local project describes access to the hourly readings on request with the building manager's permission. This package does not contain or establish that permission; each data release must follow the applicable author and manager authorization. The building identity and internal research notes are excluded.

Install dependencies in a fresh Python 3.13 environment. The recorded machine used PyTorch 2.11.0 with CUDA 12.8, an NVIDIA RTX 3070 Ti Laptop GPU and an Intel Core i7-12700H. The original neural worker fixes the device to `cuda`; a CPU-only installation cannot run those jobs unchanged. No retraining was performed to prepare this package.

From `revision/`, run the scientific steps listed in its README after obtaining authorized data. Their full generated outputs are intentionally absent here. `analysis.py`, figure generation and diagnostics require prediction caches produced by the preceding experiment scripts; the reduced summaries in `aggregate-results/` do not replace those caches. Reference-harvesting utilities, manuscript sources and internal notes are outside this code package.

The revision README records that the analysis, table, plotting and diagram code was written with Claude (Anthropic). Preserve the actual provenance when describing the work.

## Rights

No separate code or data licence was found in the local project, and this staging package assigns none. Third-party dependencies retain their respective licences. Meter readings remain restricted; no public-data claim is made.
