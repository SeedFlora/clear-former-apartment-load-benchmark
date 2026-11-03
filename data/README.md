# Apartment meter source data

The original Excel workbook used for CLEAR-Former paper 78 is provided here:

[Download Data Kelistrikan 1 Bulan.xlsx](./Data%20Kelistrikan%201%20Bulan.xlsx)

The workbook is unchanged from the research source. It contains one worksheet,
457 source records plus the header row, a Time column and 22 measurement
channels. The study aggregates the records to 456 observed hourly points on
a 720-hour grid. Missing hours are reconstructed by the supplied code.

SHA-256: `6cdb55d3d00a8470274d20aae5434b788a2ff4f0c67bf05ccdbd34a0bf0aa161`

Original Excel cell formats are retained, including time/duration formats in
some measurement channels. Use `src/data.py` and the corrected protocol in
`revision/` for the study's handling of these channels. The workbook is the
raw export, not a cleaned or gap-filled series. Generated prediction arrays,
fitted models and per-run caches are not part of this release.

The source data were added to the public repository on 3 November 2026.
No separate data licence is supplied.
