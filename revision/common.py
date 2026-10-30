"""
Shared pieces of the revised protocol (revision of the ICoDiT 2026 submission).

What changes against the submitted pipeline in ../src (which is left untouched
so the submitted numbers stay reproducible):
  1. The reconstruction (hour-of-day climatology, interpolated level, clipping
     range) is estimated from observed hours before the first test target only.
     The submitted code used every observed hour, the 48 test hours included.
  2. No noise term. In src/impute.py the residual SD was taken against an
     interpolant that passes through every observation, so it was exactly 0 and
     the submitted fills were already deterministic; the term is dropped.
  3. Validation scores observed target hours only. Reconstructed targets are
     used for training, never for early stopping or model selection.
  4. Training windows whose targets overlap validation targets are purged
     (only the 24 h task has such windows).
Windows, features, models, optimiser and early-stopping rule are reused from
../src unchanged.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import data as D            # noqa: E402  raw reader and channel cleaning
import metrics as M         # noqa: E402
import run_experiments as R  # noqa: E402  flat_features / static_features

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

L = 24                       # input window (h)
TEST_HOURS = 48
N_GRID = 720
TEST_START = N_GRID - TEST_HOURS   # grid index 672 = 2026-06-27 20:00, first test target
VAL_FRAC = 0.15
SEEDS = (0, 1, 2, 3, 4)
COV_SLICE = (1, 8)           # seven meter covariates in the feature matrix
TARGET_IDX = 0


# --------------------------------------------------------------------------- #
# reconstruction
# --------------------------------------------------------------------------- #
def _fill_channel(vals, is_obs, hour, fit_mask, clip=True):
    """c[h(t)] + tau(t) for every missing hour t.

    c[h] is the mean of fitted observations at hour of day h; tau is the
    de-seasonalised value d = x - c[h] interpolated linearly in time between
    the two fitted observations that bound the gap.  Returns the filled
    channel and the climatology."""
    use = is_obs & fit_mask
    clim = np.array([vals[use & (hour == h)].mean() for h in range(24)])
    d = vals - clim[hour]
    idx = np.where(use)[0]
    tau = np.interp(np.arange(len(vals)), idx, d[idx])
    out = vals.copy()
    miss = ~is_obs
    out[miss] = clim[hour[miss]] + tau[miss]
    n_clipped = 0
    if clip:
        lo, hi = vals[use].min(), vals[use].max()
        n_clipped = int(((out[miss] < lo) | (out[miss] > hi)).sum())
        out[miss] = np.clip(out[miss], lo, hi)
    return out, clim, n_clipped


def reconstruct(fit_end=TEST_START, clip_covariates=True):
    """Hourly 720 h grid with observed values and reconstructed gaps.

    fit_end: grid index; only observed hours with index < fit_end enter the
    reconstruction.  fit_end=None reproduces the submitted leakage (all
    observed hours, test block included) apart from the zero noise term."""
    times, colnames, mat, dropped, header = D.read_clean_table()
    t0 = times[0].replace(minute=0, second=0, microsecond=0)
    t1 = times[-1].replace(minute=0, second=0, microsecond=0)
    n = int((t1 - t0).total_seconds() // 3600) + 1
    assert n == N_GRID, n
    grid = [t0 + dt.timedelta(hours=i) for i in range(n)]

    tcol = next(j for j, c in enumerate(colnames)
                if c.replace(" ", "").lower() == D.TARGET)
    cov_idx, cov_names = D.select_covariates(colnames, mat)

    is_obs = np.zeros(n, bool)
    target = np.full(n, np.nan)
    cov = np.full((n, len(cov_idx)), np.nan)
    n_dup = 0
    for k, tt in enumerate(times):
        gi = int((tt.replace(minute=0, second=0, microsecond=0) - t0).total_seconds() // 3600)
        if is_obs[gi]:
            n_dup += 1                       # later row inside an observed hour: first row kept
            continue
        target[gi] = mat[k, tcol]
        cov[gi] = mat[k, cov_idx]
        is_obs[gi] = True

    hour = np.array([t.hour for t in grid])
    dow = np.array([t.weekday() for t in grid])
    fit_mask = np.ones(n, bool) if fit_end is None else (np.arange(n) < fit_end)

    target_full, clim_t, clip_t = _fill_channel(target, is_obs, hour, fit_mask, clip=True)
    cov_full = np.zeros_like(cov)
    clip_c = []
    for c in range(cov.shape[1]):
        cov_full[:, c], _, k = _fill_channel(cov[:, c], is_obs, hour, fit_mask, clip=clip_covariates)
        clip_c.append(k)

    cal = np.stack([np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
                    np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0),
                    (dow >= 5).astype(float)], axis=1)
    feat = np.concatenate([target_full[:, None], cov_full, cal], axis=1)
    return {
        "feat": feat, "target": target_full, "target_raw": target, "cov_raw": cov,
        "times": grid, "hour_ids": hour, "dow_ids": dow,
        "is_imputed": ~is_obs, "clim_target": clim_t,
        "n_clipped_target": clip_t, "n_clipped_cov": clip_c,
        "cov_names": [c.strip() for c in cov_names], "n_duplicate_rows": n_dup,
        "dropped_columns": dropped, "fit_end": fit_end,
    }


def missing_runs(is_imputed):
    runs, i, n = [], 0, len(is_imputed)
    while i < n:
        if is_imputed[i]:
            j = i
            while j < n and is_imputed[j]:
                j += 1
            runs.append((i, j))              # [start, end)
            i = j
        else:
            i += 1
    return runs


# --------------------------------------------------------------------------- #
# windows and splits
# --------------------------------------------------------------------------- #
def make_windows(P, H):
    feat, target, imp = P["feat"], P["target"], P["is_imputed"]
    out = []
    for o in range(L - 1, N_GRID - H):
        lo, hi = o - L + 1, o + 1
        out.append({
            "origin": o, "X": feat[lo:hi], "hid": P["hour_ids"][lo:hi], "dow": P["dow_ids"][lo:hi],
            "y": target[o + 1:o + 1 + H].copy(),
            "in_obs": ~imp[lo:hi], "tg_obs": ~imp[o + 1:o + 1 + H], "tgt_max": o + H,
        })
    return out


def long_gap(P):
    runs = missing_runs(P["is_imputed"])
    return max(runs, key=lambda r: r[1] - r[0])


def split(P, ws, H, scheme="recon", purge=True):
    """Chronological split of one task.

    scheme: 'recon'        every window with targets before the test block;
            'observed'     fully observed windows only (no reconstructed hour);
            'short_fills'  like 'recon' minus windows touching the longest gap.
    The latest VAL_FRAC of the pool (by origin) is validation; training windows
    whose targets reach the first validation target are purged (purge=False
    keeps them, as the submitted protocol did)."""
    pool = sorted([w for w in ws if w["tgt_max"] < TEST_START], key=lambda w: w["origin"])
    if scheme == "recon_blockval":
        # post hoc sensitivity design: validation = latest VAL_FRAC of the fully
        # observed windows that end before the longest gap; training = every
        # reconstructed-pool window whose span does not touch the validation span
        g0, _ = long_gap(P)
        full = [w for w in pool if w["in_obs"].all() and w["tg_obs"].all() and w["tgt_max"] < g0]
        val = full[-max(8, int(VAL_FRAC * len(full))):]
        lo, hi = val[0]["origin"] - L + 1, val[-1]["tgt_max"]
        vset = {w["origin"] for w in val}
        rest = [w for w in pool if w["origin"] not in vset]
        train = [w for w in rest if w["tgt_max"] < lo or w["origin"] - L + 1 > hi]
        test = sorted([w for w in ws if w["origin"] + 1 >= TEST_START], key=lambda w: w["origin"])
        return {"train": train, "val": val, "test": test, "n_purged": len(rest) - len(train)}
    if scheme == "observed":
        pool = [w for w in pool if w["in_obs"].all() and w["tg_obs"].all()]
    elif scheme == "short_fills":
        g0, g1 = long_gap(P)
        pool = [w for w in pool if (w["tgt_max"] < g0) or (w["origin"] - L + 1 >= g1)]
    elif scheme != "recon":
        raise ValueError(scheme)
    n_val = max(8, int(VAL_FRAC * len(pool)))
    val = pool[-n_val:]
    first_val_target = val[0]["origin"] + 1
    head = pool[:-n_val]
    train = [w for w in head if w["tgt_max"] < first_val_target] if purge else list(head)
    test = sorted([w for w in ws if w["origin"] + 1 >= TEST_START], key=lambda w: w["origin"])
    assert all(w["in_obs"].all() and w["tg_obs"].all() for w in test), "test touches a reconstructed hour"
    assert max(w["tgt_max"] for w in train + val) < TEST_START
    return {"train": train, "val": val, "test": test, "n_purged": len(head) - len(train)}


def stack(ws):
    X = np.array([w["X"] for w in ws], float)
    hid = np.array([w["hid"] for w in ws], int)
    dow = np.array([w["dow"] for w in ws], int)
    y = np.array([w["y"] for w in ws], float)
    mask = np.array([w["tg_obs"] for w in ws], bool)
    return X, hid, dow, y, mask


def masked_mae(y, p, mask):
    y, p, mask = np.asarray(y, float), np.asarray(p, float), np.asarray(mask, bool)
    return float(np.abs(y - p)[mask].mean())


def composition(ws, P):
    """How many windows carry reconstructed values, and where."""
    if not ws:
        return {}
    g0, g1 = long_gap(P)
    n_in = sum(1 for w in ws if not w["in_obs"].all())
    n_tg = sum(1 for w in ws if not w["tg_obs"].all())
    full = sum(1 for w in ws if w["in_obs"].all() and w["tg_obs"].all())
    in_frac = float(np.mean([1 - w["in_obs"].mean() for w in ws]))
    tg_frac = float(np.mean([1 - w["tg_obs"].mean() for w in ws]))
    touch_long = sum(1 for w in ws if not ((w["tgt_max"] < g0) or (w["origin"] - L + 1 >= g1)))
    return {"n": len(ws), "fully_observed": full, "synthetic_input": n_in, "synthetic_target": n_tg,
            "any_observed_target": sum(1 for w in ws if w["tg_obs"].any()),
            "observed_target_hours": int(sum(w["tg_obs"].sum() for w in ws)),
            "share_synthetic_input_positions": in_frac, "share_synthetic_target_positions": tg_frac,
            "touch_longest_gap": touch_long,
            "first_origin": int(ws[0]["origin"]), "last_origin": int(ws[-1]["origin"])}


# --------------------------------------------------------------------------- #
# scaling and features (as in src/run_experiments.py)
# --------------------------------------------------------------------------- #
class Scaler:
    def __init__(self, P):
        f, t = P["feat"][:TEST_START], P["target"][:TEST_START]
        self.fmean, self.fstd = f.mean(0), f.std(0) + 1e-8
        self.tmean, self.tstd = float(t.mean()), float(t.std() + 1e-8)

    def X(self, X):
        return (X - self.fmean) / self.fstd

    def y(self, y):
        return (y - self.tmean) / self.tstd

    def inv(self, y):
        return y * self.tstd + self.tmean


def static_feats(ws):
    return R.static_features(ws, TARGET_IDX, COV_SLICE)


def flat_feats(ws):
    return R.flat_features(ws, TARGET_IDX, COV_SLICE)


def grid_time(P, i):
    return str(P["times"][i])
