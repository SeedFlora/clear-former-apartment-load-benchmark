"""
Masking experiment for the gap reconstruction.

Artificial gaps of G hours are cut into the observed pre-test record (grid
hours < 672) and each method rebuilds them from the remaining observed
pre-test hours.  A gap covers [s, s+G); it needs an observed hour on each side
(s-1 and s+G) and ends before the test block.  Real gaps that fall inside it
become part of the hole, so the hole is exactly G hours long; only its
originally observed hours are scored, and at least half of them must be
observed.  G = 1, 6, 24, 47, 96 and 216 h (1, 47 and 216 h are the lengths of
the record's real gaps); starts every 6 h.

Methods
  LOCF            last observed value before the gap
  Linear          linear interpolation between the two bounding observations
  HourMean        hour-of-day mean c[h]
  PrevDay         value 24k h earlier, smallest k >= 1 with an observation
  Kalman          smoothed signal of a local-level model with a 4-harmonic
                  24 h cycle, fitted by maximum likelihood on the masked record
  Ours            c[h(t)] + tau(t), tau = de-seasonalised level interpolated
                  between the bounding observations (the paper's reconstruction)
Output: results/recon_validation.json
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import common as C  # noqa: E402

LENGTHS = (1, 6, 24, 47, 96, 216)
STEP = 6
METHODS = ("LOCF", "Linear", "HourMean", "PrevDay", "Kalman", "Ours")


def placements(obs, G, end=C.TEST_START):
    out = []
    for s in range(1, end - G, STEP):
        e = s + G
        if e >= end or not obs[s - 1] or not obs[e]:
            continue
        if obs[s:e].sum() < 0.5 * G:
            continue
        out.append(s)
    return out


def reconstruct_block(y, obs_fit, hour, s, G, methods=METHODS):
    """y: record with NaN where unobserved or masked; obs_fit: usable hours."""
    e = s + G
    t = np.arange(s, e)
    clim = np.array([y[obs_fit & (hour == h)].mean() for h in range(24)])
    lo, hi = y[obs_fit].min(), y[obs_fit].max()
    yl, yr = y[s - 1], y[e]
    w = (t - (s - 1)) / (e - (s - 1))
    out = {}
    if "LOCF" in methods:
        out["LOCF"] = np.full(G, yl)
    if "Linear" in methods:
        out["Linear"] = yl + w * (yr - yl)
    if "HourMean" in methods:
        out["HourMean"] = clim[hour[t]]
    if "PrevDay" in methods:
        pd_ = np.empty(G)
        for i, tt in enumerate(t):
            k, v = 1, np.nan
            while tt - 24 * k >= 0:
                if obs_fit[tt - 24 * k]:
                    v = y[tt - 24 * k]
                    break
                k += 1
            pd_[i] = v if np.isfinite(v) else clim[hour[tt]]
        out["PrevDay"] = pd_
    if "Ours" in methods:
        dl, dr = yl - clim[hour[s - 1]], yr - clim[hour[e]]
        out["Ours"] = np.clip(clim[hour[t]] + dl + w * (dr - dl), lo, hi)
    if "Kalman" in methods:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        yy = np.where(obs_fit, y, np.nan)
        r = UnobservedComponents(yy, level="llevel",
                                 freq_seasonal=[{"period": 24, "harmonics": 4}]).fit(disp=False, maxiter=200)
        out["Kalman"] = r.smoother_results.smoothed_forecasts[0][s:e]
    return out


def main():
    t0 = time.time()
    P = C.reconstruct()
    n = C.TEST_START
    obs = ~P["is_imputed"][:n]
    hour = P["hour_ids"][:n]
    chans = {"active_power_total": P["target_raw"][:n]}
    for j, name in enumerate(P["cov_names"]):
        chans[name] = P["cov_raw"][:n, j]

    res = {"lengths": LENGTHS, "step_h": STEP, "methods": METHODS, "target": {}, "covariates": {}}
    for G in LENGTHS:
        S = placements(obs, G)
        per = {m: {"mae": [], "rmse": [], "bias": [], "n_scored": []} for m in METHODS}
        cov_err = {}
        for s in S:
            fit = obs.copy()
            fit[s:s + G] = False
            sc = np.where(obs[s:s + G])[0]            # scored positions inside the hole
            y = np.where(fit, chans["active_power_total"], np.nan)
            rec = reconstruct_block(y, fit, hour, s, G)
            truth = chans["active_power_total"][s:s + G][sc]
            for m in METHODS:
                err = rec[m][sc] - truth
                per[m]["mae"].append(float(np.abs(err).mean()))
                per[m]["rmse"].append(float(np.sqrt((err ** 2).mean())))
                per[m]["bias"].append(float(err.mean()))
                per[m]["n_scored"].append(int(len(sc)))
            for name in P["cov_names"]:                # covariates: our method and the two cheapest rivals
                v = chans[name]
                sd = float(v[obs].std())
                yc = np.where(fit, v, np.nan)
                rc = reconstruct_block(yc, fit, hour, s, G, methods=("Linear", "HourMean", "Ours"))
                for m, p in rc.items():
                    cov_err.setdefault(name, {}).setdefault(m, []).append(
                        float(np.abs(p[sc] - v[s:s + G][sc]).mean() / sd))
        summ = {}
        for m in METHODS:
            a = np.array(per[m]["mae"])
            summ[m] = {"n_placements": len(S), "mae_mean": float(a.mean()), "mae_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                       "mae_median": float(np.median(a)), "mae_q25": float(np.quantile(a, 0.25)),
                       "mae_q75": float(np.quantile(a, 0.75)),
                       "rmse_mean": float(np.mean(per[m]["rmse"])), "bias_mean": float(np.mean(per[m]["bias"])),
                       "scored_hours_total": int(np.sum(per[m]["n_scored"])),
                       "ours_better_in": int(np.sum(np.array(per["Ours"]["mae"]) < a)) if m != "Ours" else None}
        res["target"][str(G)] = {"starts": S, "summary": summ,
                                 "per_placement_mae": {m: per[m]["mae"] for m in METHODS}}
        res["covariates"][str(G)] = {name: {m: float(np.mean(v)) for m, v in d.items()} for name, d in cov_err.items()}
        best = min(METHODS, key=lambda m: summ[m]["mae_mean"])
        print(f"G={G:3d} h  placements={len(S):3d}  " +
              "  ".join(f"{m}={summ[m]['mae_mean']:.2f}" for m in METHODS) + f"  best={best}", flush=True)
    res["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(C.RESULTS, "recon_validation.json"), "w"), indent=1)
    print(f"[done] {res['elapsed_sec']}s")


if __name__ == "__main__":
    main()
