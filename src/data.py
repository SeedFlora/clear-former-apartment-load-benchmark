"""
Data pipeline for the apartment short-term load forecasting (STLF) paper.

Responsibilities
----------------
1. Read the raw Excel export (Data Kelistrikan 1 Bulan.xlsx).
2. Recover the clean numeric columns and DROP columns that the source file
   corrupted with Excel time/duration number-formats (they decode to
   nonsensical values such as frequency == 2.08).  A column is kept only if
   >= 95% of its cells are real int/float numbers.
3. Reconstruct the hourly series on its native timestamps, detect gaps
   (> 1 h steps) and label contiguous segments.
4. Build gap-aware supervised windows for two canonical STLF tasks:
     * 1-hour-ahead   (scalar target)
     * day-ahead      (24-vector target, horizon 1..24)
   A window is only emitted when its L input hours AND its H target hours are
   fully contiguous (no internal gap), so no sample ever crosses a data hole.
5. Chronological train / val / test split by window origin time.

The module is deliberately dependency-light (openpyxl + numpy) so the exact
same arrays are reproducible.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # paper_build/
PROJECT = os.path.dirname(ROOT)                    # listrik apart/
XLSX = os.path.join(ROOT, "data", "Data Kelistrikan 1 Bulan.xlsx")
if not os.path.isfile(XLSX):
    XLSX = os.path.join(PROJECT, "Data Kelistrikan 1 Bulan.xlsx")
DATA_DIR = os.path.join(ROOT, "data")

TARGET = "active_power_total"
INPUT_LEN = 24          # look-back window (hours) = one day
HORIZON = 24            # day-ahead horizon
SEED = 20260707


# --------------------------------------------------------------------------- #
# 1. Raw read + column cleaning
# --------------------------------------------------------------------------- #
def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def read_clean_table():
    """Return (times, colnames, matrix) keeping only reliably-numeric columns."""
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb.active
    header = [str(c.value).strip() if c.value is not None else "" for c in ws[1]]

    rows = list(ws.iter_rows(min_row=2, values_only=True))
    n = len(rows)

    # Time column (index 0)
    times = []
    for r in rows:
        t = r[0]
        times.append(t if isinstance(t, _dt.datetime) else None)

    # Assess each non-time column for numeric reliability
    keep_idx, keep_names, dropped = [], [], []
    for j in range(1, len(header)):
        vals = [r[j] for r in rows]
        numeric = sum(1 for v in vals if _is_number(v))
        frac = numeric / max(1, n)
        if frac >= 0.95:
            keep_idx.append(j)
            keep_names.append(header[j])
        else:
            dropped.append((header[j], round(frac, 3)))

    mat = np.full((n, len(keep_idx)), np.nan, dtype=float)
    for i, r in enumerate(rows):
        for k, j in enumerate(keep_idx):
            v = r[j]
            if _is_number(v):
                mat[i, k] = float(v)

    # sort by time (defensive) and drop rows with no timestamp
    order = [i for i in range(n) if times[i] is not None]
    order.sort(key=lambda i: times[i])
    times = [times[i] for i in order]
    mat = mat[order]
    return times, keep_names, mat, dropped, header


# --------------------------------------------------------------------------- #
# 2. Segment / gap detection
# --------------------------------------------------------------------------- #
def segment_series(times):
    """Return list of (start_idx, end_idx_exclusive) contiguous 1-hour runs."""
    segs = []
    start = 0
    for i in range(1, len(times)):
        dh = (times[i] - times[i - 1]).total_seconds() / 3600.0
        if abs(dh - 1.0) > 1e-6:
            segs.append((start, i))
            start = i
    segs.append((start, len(times)))
    return segs


# --------------------------------------------------------------------------- #
# 3. Feature engineering
# --------------------------------------------------------------------------- #
def calendar_features(times):
    """Cyclical hour-of-day / day-of-week + weekend flag; plus raw hour & dow ids."""
    hod = np.array([t.hour for t in times], dtype=float)
    dow = np.array([t.weekday() for t in times], dtype=float)  # 0=Mon
    feats = np.stack([
        np.sin(2 * np.pi * hod / 24.0),
        np.cos(2 * np.pi * hod / 24.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        (dow >= 5).astype(float),
    ], axis=1)
    names = ["hod_sin", "hod_cos", "dow_sin", "dow_cos", "weekend"]
    return feats, names, hod.astype(int), dow.astype(int)


def select_covariates(colnames, mat):
    """Pick a compact, sensible set of clean electrical covariates."""
    wanted_prefixes = ["current_a", "current_b", "current_c",
                       "voltage_a", "voltage_b", "voltage_c",
                       "power_factor_total"]
    idx, names = [], []
    for w in wanted_prefixes:
        for j, c in enumerate(colnames):
            if c.replace(" ", "").lower().startswith(w.replace(" ", "").lower()):
                idx.append(j); names.append(c.strip()); break
    return np.array(idx, dtype=int), names


# --------------------------------------------------------------------------- #
# 4. Gap-aware windowing
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    # feature matrix used for windows: columns = [target, covariates..., calendar...]
    feat_names: list
    target_col: int            # index of target inside feature matrix
    hour_ids: np.ndarray       # (N,) hour-of-day id per timestep
    dow_ids: np.ndarray        # (N,) day-of-week id per timestep
    times: list
    # windows
    Xw: np.ndarray             # (S, L, F)
    hidw: np.ndarray           # (S, L) hour ids over the input window
    didw: np.ndarray           # (S, L) dow ids over the input window
    y1: np.ndarray             # (S,) 1-step-ahead target (raw scale)
    y24: np.ndarray            # (S, H) day-ahead targets (raw scale)
    origin_time: list          # timestamp of last input step per window
    split: np.ndarray          # (S,) 0=train 1=val 2=test


def build(input_len=INPUT_LEN, horizon=HORIZON, val_frac=0.15, test_frac=0.15):
    times, colnames, mat, dropped, header = read_clean_table()
    segs = segment_series(times)

    cal, cal_names, hour_ids, dow_ids = calendar_features(times)
    cov_idx, cov_names = select_covariates(colnames, mat)

    # target index in original clean matrix
    tcol = None
    for j, c in enumerate(colnames):
        if c.replace(" ", "").lower() == TARGET.replace(" ", "").lower():
            tcol = j; break
    if tcol is None:
        raise RuntimeError(f"target {TARGET} not found in {colnames}")

    target = mat[:, tcol:tcol + 1]
    cov = mat[:, cov_idx] if len(cov_idx) else np.zeros((len(times), 0))

    # simple forward/back fill for any isolated NaN inside covariates/target
    def _fill(a):
        a = a.copy()
        for c in range(a.shape[1]):
            col = a[:, c]
            # forward fill
            last = np.nan
            for i in range(len(col)):
                if np.isnan(col[i]):
                    col[i] = last
                else:
                    last = col[i]
            # back fill
            nxt = np.nan
            for i in range(len(col) - 1, -1, -1):
                if np.isnan(col[i]):
                    col[i] = nxt
                else:
                    nxt = col[i]
            a[:, c] = col
        return a
    target = _fill(target)
    cov = _fill(cov) if cov.shape[1] else cov

    feat = np.concatenate([target, cov, cal], axis=1)
    feat_names = [TARGET] + [c.strip() for c in cov_names] + cal_names
    target_col = 0

    L, H = input_len, horizon
    Xw, hidw, didw, y1, y24, origin_time = [], [], [], [], [], []
    for (s, e) in segs:
        seg_len = e - s
        # need L input + H future within the SAME segment
        for start in range(s, e - L - H + 1):
            in_lo, in_hi = start, start + L           # input [in_lo, in_hi)
            fut_lo, fut_hi = in_hi, in_hi + H          # future [fut_lo, fut_hi)
            Xw.append(feat[in_lo:in_hi])
            hidw.append(hour_ids[in_lo:in_hi])
            didw.append(dow_ids[in_lo:in_hi])
            y1.append(target[fut_lo, 0])
            y24.append(target[fut_lo:fut_hi, 0])
            origin_time.append(times[in_hi - 1])

    Xw = np.asarray(Xw, dtype=float)
    hidw = np.asarray(hidw, dtype=int)
    didw = np.asarray(didw, dtype=int)
    y1 = np.asarray(y1, dtype=float)
    y24 = np.asarray(y24, dtype=float)
    S = len(Xw)

    # chronological split by origin time
    order = np.argsort([t.timestamp() for t in origin_time])
    split = np.zeros(S, dtype=int)
    n_test = int(round(test_frac * S))
    n_val = int(round(val_frac * S))
    test_ids = set(order[-n_test:].tolist())
    val_ids = set(order[-(n_test + n_val):-n_test].tolist())
    for i in range(S):
        if i in test_ids:
            split[i] = 2
        elif i in val_ids:
            split[i] = 1

    ds = Dataset(feat_names=feat_names, target_col=target_col,
                 hour_ids=hour_ids, dow_ids=dow_ids, times=times,
                 Xw=Xw, hidw=hidw, didw=didw, y1=y1, y24=y24,
                 origin_time=origin_time, split=split)

    meta = {
        "n_raw_rows": len(times),
        "date_start": str(times[0]), "date_end": str(times[-1]),
        "kept_columns": colnames,
        "dropped_columns": dropped,
        "covariates_used": [c.strip() for c in cov_names],
        "feature_names": feat_names,
        "n_segments": len(segs),
        "segment_lengths": [e - s for (s, e) in segs],
        "segment_spans": [(str(times[s]), str(times[e - 1])) for (s, e) in segs],
        "input_len": L, "horizon": H,
        "n_windows": int(S),
        "n_train": int((split == 0).sum()),
        "n_val": int((split == 1).sum()),
        "n_test": int((split == 2).sum()),
        "target": TARGET,
        "target_mean": float(np.mean(target)), "target_std": float(np.std(target)),
        "target_min": float(np.min(target)), "target_max": float(np.max(target)),
    }
    return ds, meta, segs, times, target[:, 0]


def prepare(input_len=INPUT_LEN):
    """Return the cleaned feature matrix + target + timestamps + segments.

    This is the entry point used by the rolling-origin backtest in
    run_experiments.py.  Everything is on the native hourly grid; segment
    boundaries mark the data gaps so no window/forecast crosses a hole.
    """
    times, colnames, mat, dropped, header = read_clean_table()
    segs = segment_series(times)
    cal, cal_names, hour_ids, dow_ids = calendar_features(times)
    cov_idx, cov_names = select_covariates(colnames, mat)

    tcol = None
    for j, c in enumerate(colnames):
        if c.replace(" ", "").lower() == TARGET.replace(" ", "").lower():
            tcol = j; break

    target = mat[:, tcol:tcol + 1]
    cov = mat[:, cov_idx] if len(cov_idx) else np.zeros((len(times), 0))

    def _fill(a):
        a = a.copy()
        for c in range(a.shape[1]):
            col = a[:, c]
            last = np.nan
            for i in range(len(col)):
                if np.isnan(col[i]): col[i] = last
                else: last = col[i]
            nxt = np.nan
            for i in range(len(col) - 1, -1, -1):
                if np.isnan(col[i]): col[i] = nxt
                else: nxt = col[i]
            a[:, c] = col
        return a
    target = _fill(target)
    cov = _fill(cov) if cov.shape[1] else cov

    feat = np.concatenate([target, cov, cal], axis=1)
    feat_names = [TARGET] + [c.strip() for c in cov_names] + cal_names

    return {
        "feat": feat, "feat_names": feat_names, "target_col": 0,
        "target": target[:, 0], "cov": cov, "cov_names": [c.strip() for c in cov_names],
        "cal": cal, "cal_names": cal_names,
        "times": times, "segs": segs,
        "hour_ids": hour_ids, "dow_ids": dow_ids,
        "dropped_columns": dropped, "kept_columns": colnames,
    }


def _save(ds, meta):
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(DATA_DIR, "windows.npz"),
        Xw=ds.Xw, hidw=ds.hidw, didw=ds.didw, y1=ds.y1, y24=ds.y24,
        split=ds.split,
        origin_ts=np.array([t.timestamp() for t in ds.origin_time]),
        target_col=ds.target_col,
    )
    with open(os.path.join(DATA_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    ds, meta, segs, times, target = build()
    _save(ds, meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
