"""Data checks behind the channel and gap facts (writes results/data_and_ref_checks.json).

* the Python type of every cell in each of the 22 channels of the Excel export;
* the five float channels that hold day fractions of whole minutes
  (stored = reading x 1000 / 1440), recovered with a factor of 1.44;
* the observations that bound each reconstructed gap.

The bibliographic checks that an earlier version of this file carried now live
in references/ref_checks.json (references/check_refs.py).

    python check_data.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import openpyxl

import common as C

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import data as D  # noqa: E402


def channel_checks():
    wb = openpyxl.load_workbook(D.XLSX, data_only=True)
    ws = wb.active
    header = [str(c.value).strip() if c.value is not None else "" for c in ws[1]]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    types, encoding, hit = {}, {}, 0
    for j in range(1, len(header)):
        vals = [r[j] for r in rows if r[j] is not None]
        names = sorted({type(v).__name__ for v in vals})
        types[header[j]] = names
        time_like = any(n in ("time", "timedelta", "datetime") for n in names)
        day_fraction = names == ["float"] and all(abs(v * 1440 - round(v * 1440)) < 1e-9 for v in vals)
        if day_fraction:
            dev = max(abs(v * 1440 - round(v * 1440)) for v in vals)
            encoding[header[j]] = {"stored_min": min(vals), "stored_max": max(vals),
                                   "max_dev_from_whole_minutes": dev,
                                   "recovered_min": round(min(vals) * 1.44, 3),
                                   "recovered_max": round(max(vals) * 1.44, 3)}
        hit += time_like or day_fraction
    return {"channel_encoding": encoding, "channel_types": types, "n_channels": len(header) - 1,
            "n_channels_mixed_or_time_types": hit}


def gap_bounds():
    P = C.reconstruct()
    obs = ~P["is_imputed"]
    out = {}
    for s, e in C.missing_runs(P["is_imputed"]):
        lo, hi = s - 1, e          # missing_runs gives [start, end) on the grid
        while not obs[lo]:
            lo -= 1
        while not obs[hi]:
            hi += 1
        out[f"{e - s}h"] = {"left": [str(P["times"][lo]), float(P["target"][lo])],
                            "right": [str(P["times"][hi]), float(P["target"][hi])]}
    return out


def main():
    res = channel_checks()
    res["gap_bounds"] = gap_bounds()
    path = os.path.join(C.RESULTS, "data_and_ref_checks.json")
    if os.path.exists(path):                       # the stored file must not change
        old = json.load(open(path, encoding="utf-8"))
        for k, v in res.items():
            assert old.get(k) == v, f"{k} differs from the stored file"
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_channels", "n_channels_mixed_or_time_types", "gap_bounds")}))


if __name__ == "__main__":
    main()
