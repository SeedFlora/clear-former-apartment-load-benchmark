"""Validation diagnostics quoted in the letter and the brief.

* results/val_by_target_type.json: validation MAE of each selected neural ensemble on observed and on
  reconstructed target positions (from grid_val.json; the reconstructed part follows from the
  all-target and observed-target MAE and the position counts of common.split);
* results/val_test_rank.json: Spearman rank correlation of validation and test MAE across the eight
  ablation variants (followup.json) and across the four neural models;
* results/dm_autocorr.json: lag-1 autocorrelation and Ljung-Box(4) p-value of the one-hour absolute-error
  loss differentials, CLEAR-Former against each other model (test block, 48 origins).

    python val_diagnostics.py      # after run_benchmark.py, analysis.py and run_followup.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr

import common as C

NEURAL = ["MLP", "LSTM", "Transformer", "CLEAR-Former"]
OTHERS = ["SeasonalNaive", "SARIMA", "HoltWinters", "Prophet", "SVR", "RandomForest", "XGBoost", "MLP", "LSTM",
          "Transformer"]


def j(name):
    return json.load(open(os.path.join(C.RESULTS, name), encoding="utf-8"))


def check_and_write(name, new, tol=1e-9):
    path = os.path.join(C.RESULTS, name)
    if os.path.exists(path):                       # a stored file must be reproduced, not silently changed
        old = json.load(open(path, encoding="utf-8"))

        def cmp(a, b, where):
            if isinstance(a, dict):
                assert set(a) == set(b), f"{name}{where}: keys differ"
                for k in a:
                    cmp(a[k], b[k], f"{where}/{k}")
            elif isinstance(a, list):
                assert len(a) == len(b), f"{name}{where}: lengths differ"
                for i, (x, y) in enumerate(zip(a, b)):
                    cmp(x, y, f"{where}[{i}]")
            elif isinstance(a, float) or isinstance(b, float):
                assert abs(a - b) <= tol * max(1.0, abs(a)), f"{name}{where}: {a} != {b}"
            else:
                assert a == b, f"{name}{where}: {a} != {b}"
        cmp(old, new, "")
    json.dump(new, open(path, "w", encoding="utf-8"), indent=1)


def val_by_target_type():
    grid, sel = j("grid_val.json"), j("selection.json")["neural"]
    P = C.reconstruct()
    out = {}
    for task, H in (("1h", 1), ("day", 24)):
        val = C.split(P, C.make_windows(P, H), H)["val"]
        n_obs = int(sum(w["tg_obs"].sum() for w in val))
        n_all = len(val) * H
        for m in NEURAL:
            cid = sel[f"{task}|{m}"]["selected_config_id"]
            row = next(r for r in grid[f"{task}|{m}"] if r["config_id"] == cid)
            obs, allv = row["val_mae_masked_ensemble"], row["val_mae_all_ensemble"]
            recon = (allv * n_all - obs * n_obs) / (n_all - n_obs)
            out[f"{task}|{m}"] = {"obs": obs, "recon": recon, "all": allv, "n_obs": n_obs, "n_recon": n_all - n_obs}
    return out


def val_test_rank(vbt):
    F, S = j("followup.json"), j("summary.json")
    out = {}
    for task in ("1h", "day"):
        keys = [k for k in F if k.startswith(f"ablation|{task}|")]
        v = [float(np.mean(F[k]["val_mae_masked_seeds"])) for k in keys]
        t = [F[k]["MAE"] for k in keys]
        r = spearmanr(v, t)
        out[f"{task}|ablation_variants"] = {"n": len(keys), "spearman": float(r.statistic), "p": float(r.pvalue)}
        v = [vbt[f"{task}|{m}"]["obs"] for m in NEURAL]
        t = [S["metrics"][task][m]["MAE"] for m in NEURAL]
        r = spearmanr(v, t)
        out[f"{task}|neural_models"] = {"n": 4, "spearman": float(r.statistic), "p": float(r.pvalue), "val": v, "test": t}
    return out


def dm_autocorr():
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf
    z = np.load(os.path.join(C.RESULTS, "bench_predictions.npz"), allow_pickle=True)
    y = z["1h__y_true"].reshape(-1)
    cf = np.abs(z["1h__CLEAR-Former__selected"].reshape(-1) - y)
    out = {}
    for m in OTHERS:
        key = f"1h__{m}__selected" if m in NEURAL else f"1h__{m}"
        d = cf - np.abs(z[key].reshape(-1) - y)
        lb = acorr_ljungbox(d, lags=[4], return_df=True)
        out[m] = {"acf_lag1": float(acf(d, nlags=1, fft=False)[1]), "ljung_box4_p": float(lb["lb_pvalue"].iloc[0])}
    return out


def main():
    vbt = val_by_target_type()
    check_and_write("val_by_target_type.json", vbt)
    check_and_write("val_test_rank.json", val_test_rank(vbt))
    ac = dm_autocorr()
    json.dump(ac, open(os.path.join(C.RESULTS, "dm_autocorr.json"), "w", encoding="utf-8"), indent=1)
    learners = ["SVR", "RandomForest", "XGBoost", "MLP", "LSTM", "Transformer"]
    a = [ac[m]["acf_lag1"] for m in learners]
    print("lag-1 ACF against the six learners:", round(min(a), 2), "to", round(max(a), 2),
          "| Ljung-Box(4) p:", {m: round(ac[m]["ljung_box4_p"], 4) for m in learners})
    print("others:", {m: round(ac[m]["acf_lag1"], 2) for m in OTHERS if m not in learners})


if __name__ == "__main__":
    main()
