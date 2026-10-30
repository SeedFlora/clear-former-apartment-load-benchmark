"""
Missing-span reconstruction (data augmentation for the gaps).

The raw recording has three holes (2 h, 48 h and 217 h) that break the month
into four continuous segments.  Rather than discard the surrounding context, we
reconstruct a COMPLETE hourly series with a seasonal-trend imputation and a
small residual-noise injection:

    x_hat(t) = clim[hour(t)] + trend(t) + eps,   eps ~ N(0, s * sigma_r)

where clim is the learned hour-of-day profile, trend is the linearly
interpolated slow level (the de-seasonalised signal), and the noise makes the
fill breathe like real data instead of a flat line.  Covariates are imputed the
same way (without noise).

Honesty guarantee: imputed timestamps are flagged in ``is_imputed`` and are used
for TRAINING context only; the experiment harness forbids any imputed value from
entering a test window, so every reported test score is measured on genuinely
observed data.
"""
from __future__ import annotations
import datetime as _dt
import os
import numpy as np

import data as D

TARGET = D.TARGET


def _impute_channel(vals, is_obs, hour, noise_std_scale, rng, add_noise=True,
                    lo=None, hi=None):
    n = len(vals)
    # hour-of-day climatology from observed points
    clim = np.zeros(24)
    for h in range(24):
        m = is_obs & (hour == h)
        clim[h] = vals[m].mean() if m.any() else np.nanmean(vals[is_obs])
    deseason = vals - clim[hour]
    obs_idx = np.where(is_obs)[0]
    # linear-interpolate the de-seasonalised level across the whole grid
    trend = np.interp(np.arange(n), obs_idx, deseason[obs_idx])
    resid = deseason[obs_idx] - trend[obs_idx]
    sigma_r = resid.std() if resid.size > 1 else 0.0
    out = vals.copy()
    for i in range(n):
        if not is_obs[i]:
            e = rng.normal(0, noise_std_scale * sigma_r) if add_noise else 0.0
            out[i] = clim[hour[i]] + trend[i] + e
    if lo is not None:
        out = np.clip(out, lo, hi)
    return out


def prepare_continuous(input_len=24, noise_scale=0.5, seed=20260707):
    times, colnames, mat, dropped, header = D.read_clean_table()

    # complete hourly grid spanning the observation window
    t0 = times[0].replace(minute=0, second=0, microsecond=0)
    t1 = times[-1].replace(minute=0, second=0, microsecond=0)
    n = int((t1 - t0).total_seconds() // 3600) + 1
    grid = [t0 + _dt.timedelta(hours=i) for i in range(n)]

    # target + covariate column indices (same selection as data.prepare)
    tcol = next(j for j, c in enumerate(colnames)
                if c.replace(" ", "").lower() == TARGET.replace(" ", "").lower())
    cov_idx, cov_names = D.select_covariates(colnames, mat)

    # place observed values on the grid, NaN elsewhere
    is_obs = np.zeros(n, dtype=bool)
    target = np.full(n, np.nan)
    cov = np.full((n, len(cov_idx)), np.nan)
    for k, tt in enumerate(times):
        h = tt.replace(minute=0, second=0, microsecond=0)
        gi = int((h - t0).total_seconds() // 3600)
        if 0 <= gi < n and not is_obs[gi]:
            if np.isfinite(mat[k, tcol]):
                target[gi] = mat[k, tcol]
                cov[gi] = mat[k, cov_idx]
                is_obs[gi] = True

    hour = np.array([t.hour for t in grid])
    dow = np.array([t.weekday() for t in grid])
    rng = np.random.default_rng(seed)

    lo, hi = np.nanmin(target[is_obs]), np.nanmax(target[is_obs])
    target_full = _impute_channel(target, is_obs, hour, noise_scale, rng,
                                  add_noise=True, lo=lo, hi=hi)
    cov_full = np.zeros_like(cov)
    for c in range(cov.shape[1]):
        col = cov[:, c]
        obs_c = is_obs & np.isfinite(col)
        cov_full[:, c] = _impute_channel(np.where(np.isfinite(col), col, np.nan),
                                         obs_c, hour, 0.0, rng, add_noise=False)

    is_imputed = ~is_obs

    # calendar features on the full grid
    cal = np.stack([
        np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0),
        (dow >= 5).astype(float)], axis=1)
    cal_names = ["hod_sin", "hod_cos", "dow_sin", "dow_cos", "weekend"]

    feat = np.concatenate([target_full[:, None], cov_full, cal], axis=1)
    feat_names = [TARGET] + [c.strip() for c in cov_names] + cal_names

    return {
        "feat": feat, "feat_names": feat_names, "target_col": 0,
        "target": target_full, "cov": cov_full, "cov_names": [c.strip() for c in cov_names],
        "cal": cal, "cal_names": cal_names,
        "times": grid, "segs": [(0, n)],           # one continuous segment now
        "hour_ids": hour, "dow_ids": dow,
        "is_imputed": is_imputed,
        "dropped_columns": dropped, "kept_columns": colnames,
        "n_observed": int(is_obs.sum()), "n_imputed": int(is_imputed.sum()),
        "n_total": n,
    }


if __name__ == "__main__":
    P = prepare_continuous()
    import json
    print(json.dumps({k: P[k] for k in ("n_observed", "n_imputed", "n_total", "feat_names")},
                     indent=2, default=str))
    print("imputed fraction: %.1f%%" % (100 * P["n_imputed"] / P["n_total"]))
    # sanity: imputed target within observed range, continuous grid
    steps = np.diff([t.timestamp() for t in P["times"]]) / 3600
    print("all 1h steps:", np.allclose(steps, 1.0), " n=", P["n_total"])
