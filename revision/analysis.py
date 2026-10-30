"""
Collects every number the revision reports into results/summary.json.

  metrics       MAE, RMSE, MAPE, R2 per model and task (test block, pooled)
  origins       per-origin MAE: 48 one-hour origins, 25 day-ahead origins
                (hourly stride, 48 distinct target hours, two non-overlapping)
  seeds         per-seed test MAE of the neural models (selected configurations)
  dm_1h         Diebold-Mariano tests on absolute errors at one hour with the
                Harvey-Leybourne-Newbold correction, Holm over the 10 rivals
  windows       composition of training, validation and test windows
  params        parameter counts per component
  recon_days    day-over-day change of observed and reconstructed load
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict

import numpy as np
from scipy import stats

import common as C
import train as T

ORDER = ["SeasonalNaive", "SARIMA", "HoltWinters", "Prophet", "SVR", "RandomForest", "XGBoost",
         "MLP", "LSTM", "Transformer", "CLEAR-Former"]
TASKS = (("1h", 1), ("day", 24))


def load_npz(name):
    z = np.load(os.path.join(C.RESULTS, name))
    return {k.replace("__", "|"): z[k] for k in z.files}


def model_pred(B, task, m, role="selected"):
    return B[f"{task}|{m}|{role}"] if m in T.NEURAL else B[f"{task}|{m}"]


def dm_hln(e1, e2, h=1, lags=None):
    """DM on absolute-error loss differential d = |e1| - |e2| (negative: model 1 better)."""
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    lags = h - 1 if lags is None else lags
    dc = d - d.mean()
    gamma = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(lags + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lags + 1)) * gamma[k] for k in range(1, lags + 1)) if lags else gamma[0]
    dm = d.mean() / np.sqrt(var / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return float(d.mean()), float(hln), float(p)


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def component_params(model):
    groups = OrderedDict()
    for n, p in model.named_parameters():
        top = n.split(".")[0]
        groups[top] = groups.get(top, 0) + p.numel()
    return groups


def main():
    B = load_npz("bench_predictions.npz")
    sel = json.load(open(os.path.join(C.RESULTS, "selection.json")))
    grid = json.load(open(os.path.join(C.RESULTS, "grid_val.json")))
    out = {"selection": sel, "metrics": {}, "origins": {}, "seeds": {}, "dm_1h": {}, "windows": {},
           "params": {}, "submitted_config_test": {}}

    for task, H in TASKS:
        y = B[f"{task}|y_true"]
        origins = B[f"{task}|origins"]
        out["metrics"][task], out["origins"][task] = {}, {"n_origins": int(len(origins)),
                                                          "first_origin": int(origins[0]),
                                                          "last_origin": int(origins[-1]),
                                                          "distinct_target_hours": int(len(set(
                                                              int(o) + 1 + k for o in origins for k in range(H))))}
        per_origin = {}
        for m in ORDER:
            p = model_pred(B, task, m)
            met = C.M.all_metrics(y, p)
            po = np.abs(p - y).mean(axis=1)
            per_origin[m] = po
            met.update({"origin_MAE_median": float(np.median(po)), "origin_MAE_q25": float(np.quantile(po, .25)),
                        "origin_MAE_q75": float(np.quantile(po, .75)), "origin_MAE_min": float(po.min()),
                        "origin_MAE_max": float(po.max())})
            if H > 1:
                met["MAE_first_origin"] = float(po[0])       # the two non-overlapping origins
                met["MAE_last_origin"] = float(po[-1])
            if m in T.NEURAL:
                seeds = B[f"{task}|{m}|selected|seeds"]
                sm = [C.M.mae(y, s) for s in seeds]
                met.update({"seed_MAE": sm, "seed_MAE_mean": float(np.mean(sm)), "seed_MAE_sd": float(np.std(sm, ddof=1)),
                            "seed_R2": [C.M.r2(y, s) for s in seeds]})
                subp = B[f"{task}|{m}|submitted"]
                out["submitted_config_test"][f"{task}|{m}"] = C.M.all_metrics(y, subp)
            out["metrics"][task][m] = met
        best = np.argmin(np.array([per_origin[m] for m in ORDER]), axis=0)
        out["origins"][task]["wins"] = {m: int((best == i).sum()) for i, m in enumerate(ORDER)}
        out["origins"][task]["per_origin_MAE"] = {m: per_origin[m].tolist() for m in ORDER}
        # cross-horizon mean
    out["cross_horizon_mean_MAE"] = {m: 0.5 * (out["metrics"]["1h"][m]["MAE"] + out["metrics"]["day"][m]["MAE"])
                                     for m in ORDER}

    # Diebold-Mariano at one hour: CLEAR-Former against every other model
    y = B["1h|y_true"].ravel()
    e_cf = model_pred(B, "1h", "CLEAR-Former").ravel() - y
    rivals = [m for m in ORDER if m != "CLEAR-Former"]
    rows = [dm_hln(e_cf, model_pred(B, "1h", m).ravel() - y) for m in rivals]
    rows_hac = [dm_hln(e_cf, model_pred(B, "1h", m).ravel() - y, lags=4) for m in rivals]
    adj = holm(np.array([r[2] for r in rows]))
    adj_hac = holm(np.array([r[2] for r in rows_hac]))
    out["dm_1h"] = {m: {"mean_loss_diff_kW": rows[i][0], "DM_HLN": rows[i][1], "p": rows[i][2], "p_holm": float(adj[i]),
                        "DM_HAC4": rows_hac[i][1], "p_HAC4": rows_hac[i][2], "p_holm_HAC4": float(adj_hac[i])}
                    for i, m in enumerate(rivals)}

    # windows
    import jobs as J
    for task, H in TASKS:
        for scheme in ("recon", "observed", "short_fills"):
            P, S, sc, A = J._data(C.TEST_START, scheme, H)
            out["windows"][f"{task}|{scheme}"] = {part: C.composition(S[part], P) for part in ("train", "val", "test")}
            out["windows"][f"{task}|{scheme}"]["n_purged"] = S["n_purged"]
            out["windows"][f"{task}|{scheme}"]["val_first_target"] = C.grid_time(P, S["val"][0]["origin"] + 1)
            out["windows"][f"{task}|{scheme}"]["val_last_target"] = C.grid_time(P, S["val"][-1]["tgt_max"])
            out["windows"][f"{task}|{scheme}"]["train_last_target"] = C.grid_time(P, max(w["tgt_max"] for w in S["train"]))
            out["windows"][f"{task}|{scheme}"]["test_first_origin"] = C.grid_time(P, S["test"][0]["origin"])
            out["windows"][f"{task}|{scheme}"]["test_last_origin"] = C.grid_time(P, S["test"][-1]["origin"])

    # parameters per component
    for task, H in TASKS:
        for role in ("selected", "submitted"):
            cfg = (sel["neural"][f"{task}|CLEAR-Former"]["selected_config"] if role == "selected"
                   else T.SUBMITTED["CLEAR-Former"])
            mdl = T.build_neural("CLEAR-Former", cfg, H, 0)
            comp = dict(component_params(mdl))
            attn = sum(v for k, v in comp.items() if k in ("patch_embed", "pos", "hour_emb", "dow_emb", "cov_gru",
                                                            "gate", "cov_proj", "encoder", "norm", "corr_head"))
            out["params"][f"{task}|CLEAR-Former|{role}"] = {"config": cfg, "total": T.n_params(mdl), "components": comp,
                                                             "attention_branch": attn,
                                                             "attention_branch_with_alpha": attn + comp["alpha"]}
        for m in ("MLP", "LSTM", "Transformer"):
            cfg = sel["neural"][f"{task}|{m}"]["selected_config"]
            out["params"][f"{task}|{m}|selected"] = {"config": cfg, "total": T.n_params(T.build_neural(m, cfg, H, 0))}

    # reconstructed vs observed day-over-day change
    P = C.reconstruct()
    x, imp = P["target"], P["is_imputed"]
    g0, g1 = C.long_gap(P)
    idx = np.arange(24, C.TEST_START)
    both_obs = idx[~imp[idx] & ~imp[idx - 24]]
    in_fill = idx[(idx >= g0 + 24) & (idx < g1)]
    pre_obs = ~imp[:C.TEST_START]
    per_hour = [int((pre_obs & (P["hour_ids"][:C.TEST_START] == h)).sum()) for h in range(24)]
    out["recon_days"] = {
        "n_pretest_observed": int(pre_obs.sum()), "pretest_obs_per_hour_min": min(per_hour),
        "pretest_obs_per_hour_max": max(per_hour),
        "mean_abs_day_change_observed_kW": float(np.abs(x[both_obs] - x[both_obs - 24]).mean()),
        "n_observed_pairs": int(len(both_obs)),
        "mean_abs_day_change_in_216h_fill_kW": float(np.abs(x[in_fill] - x[in_fill - 24]).mean()),
        "n_fill_pairs": int(len(in_fill)),
        "sd_observed_pretest_kW": float(x[:C.TEST_START][~imp[:C.TEST_START]].std()),
        "sd_216h_fill_kW": float(x[g0:g1].std()),
        "n_clipped_target": P["n_clipped_target"], "n_clipped_cov": P["n_clipped_cov"],
        "missing_runs": [(C.grid_time(P, a), C.grid_time(P, b - 1), b - a) for a, b in C.missing_runs(imp)],
    }
    out["grid_val"] = grid
    json.dump(out, open(os.path.join(C.RESULTS, "summary.json"), "w"), indent=1, default=float)
    for task, _ in TASKS:
        print(f"== {task}")
        for m in ORDER:
            r = out["metrics"][task][m]
            extra = f" seeds {r['seed_MAE_mean']:.2f}+/-{r['seed_MAE_sd']:.2f}" if m in T.NEURAL else ""
            print(f"  {m:14s} MAE={r['MAE']:6.2f} RMSE={r['RMSE']:6.2f} R2={r['R2']:6.3f} "
                  f"origin-MAE median={r['origin_MAE_median']:.2f} [{r['origin_MAE_min']:.2f},{r['origin_MAE_max']:.2f}]{extra}")
    print("cross-horizon mean:", {m: round(v, 2) for m, v in out["cross_horizon_mean_MAE"].items()})
    print("DM 1h:", {m: (round(v['mean_loss_diff_kW'], 2), round(v['p_holm'], 4)) for m, v in out["dm_1h"].items()})
    print("recon_days:", out["recon_days"])


if __name__ == "__main__":
    main()
