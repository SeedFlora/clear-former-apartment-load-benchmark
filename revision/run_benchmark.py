"""
Revised benchmark (protocol in common.py).

  1. Neural grid: 4 models x 4 predeclared configurations x 5 seeds x 2 tasks,
     trained on training windows, early-stopped on observed validation hours;
     only validation predictions are read, model states are saved.
  2. Selection: the configuration with the lowest masked validation MAE of its
     5-seed ensemble.
  3. Classical ML: each configuration is fitted on training windows and scored
     on observed validation hours; the selected one is refitted on the pool.
  4. Test: selected neural configurations, classical ML, statistical models
     (refitted at every origin) and the seasonal-naive reference forecast the
     test block once.  The submitted neural configurations are scored as a
     labelled post hoc comparison and play no part in selection.

Outputs (results/): grid_val.json, selection.json, bench_predictions.npz,
bench_metrics.json.  Usage: python run_benchmark.py [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as C
import train as T

MODELS_DIR = os.path.join(C.RESULTS, "models")
TASKS = (("1h", 1), ("day", 24))


def _key(task, kind, ci):
    return f"{task}|{kind}|{ci}"


def neural_grid(workers):
    os.makedirs(MODELS_DIR, exist_ok=True)
    jobs = []
    for task, H in TASKS:
        for kind in T.NEURAL:
            for ci, cfg in enumerate(T.GRIDS[kind]):
                for sd in C.SEEDS:
                    jobs.append(dict(tag=_key(task, kind, ci), task=task, H=H, kind=kind, cfg=cfg, seed=sd,
                                     fit_end=C.TEST_START, scheme="recon", mask_val=True,
                                     save_state=os.path.join(MODELS_DIR, f"{task}_{kind}_{ci}_{sd}.pt")))
    import jobs as J
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, r in enumerate(ex.map(J.neural_job, jobs)):
            res.setdefault(r["tag"], []).append(r)
            if (k + 1) % 10 == 0:
                print(f"  grid {k + 1}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    return res


def select_neural(res):
    sel, table = {}, {}
    for task, H in TASKS:
        P, S, sc, A = __import__("jobs")._data(C.TEST_START, "recon", H)
        yv, mv = A["val"]["y"], A["val"]["mask"]
        for kind in T.NEURAL:
            rows = []
            for ci, cfg in enumerate(T.GRIDS[kind]):
                rs = sorted(res[_key(task, kind, ci)], key=lambda r: r["seed"])
                ens = np.mean([r["val_pred"] for r in rs], axis=0)
                rows.append({"config_id": ci, "config": cfg,
                             "val_mae_masked_ensemble": C.masked_mae(yv, ens, mv),
                             "val_mae_all_ensemble": C.masked_mae(yv, ens, np.ones_like(mv)),
                             "val_mae_masked_seeds": [r["val_mae_masked"] for r in rs],
                             "best_epochs": [r["info"]["best_epoch"] for r in rs],
                             "n_params": rs[0]["n_params"]})
            best = min(rows, key=lambda r: r["val_mae_masked_ensemble"])
            table[f"{task}|{kind}"] = rows
            sel[f"{task}|{kind}"] = best["config_id"]
    return sel, table


def ml_models(task, H, A_pool, A_tr, A_va, workers=None):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    import xgboost as xgb

    def make(name, cfg):
        if name == "SVR":
            est = TransformedTargetRegressor(SVR(C=cfg["C"], gamma="scale", epsilon=0.1),
                                             transformer=StandardScaler())
        elif name == "RandomForest":
            est = RandomForestRegressor(n_estimators=400, max_depth=cfg["max_depth"], min_samples_leaf=2,
                                        random_state=0, n_jobs=-1)
        else:
            est = xgb.XGBRegressor(n_estimators=400, max_depth=cfg["max_depth"], learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                   random_state=0, n_jobs=-1)
        return MultiOutputRegressor(est) if H > 1 else est

    out, rows_all = {}, {}
    fsc = StandardScaler().fit(A_tr["F"])
    for name in ("SVR", "RandomForest", "XGBoost"):
        rows = []
        for ci, cfg in enumerate(T.GRIDS[name]):
            est = make(name, cfg)
            yt = A_tr["y"] if H > 1 else A_tr["y"].ravel()
            est.fit(fsc.transform(A_tr["F"]), yt)
            vp = est.predict(fsc.transform(A_va["F"])).reshape(len(A_va["y"]), -1)
            rows.append({"config_id": ci, "config": cfg,
                         "val_mae_masked": C.masked_mae(A_va["y"], vp, A_va["mask"])})
        best = min(rows, key=lambda r: r["val_mae_masked"])
        psc = StandardScaler().fit(A_pool["F"])
        est = make(name, T.GRIDS[name][best["config_id"]])
        yp = A_pool["y"] if H > 1 else A_pool["y"].ravel()
        est.fit(psc.transform(A_pool["F"]), yp)
        out[name] = est.predict(psc.transform(A_pool["Fte"])).reshape(len(A_pool["yte"]), -1)
        rows_all[f"{task}|{name}"] = {"grid": rows, "selected": best["config_id"]}
    return out, rows_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--stat-workers", type=int, default=8)
    args = ap.parse_args()
    t_start = time.time()
    import jobs as J

    # ---------------- 1-2: neural grid and selection (validation only) -------------
    print("[1] neural grid", flush=True)
    res = neural_grid(args.workers)
    sel, table = select_neural(res)
    json.dump({k: [{kk: vv for kk, vv in r.items()} for r in v] for k, v in table.items()},
              open(os.path.join(C.RESULTS, "grid_val.json"), "w"), indent=2, default=float)
    print("[2] selected:", {k: T.GRIDS[k.split("|")[1]][v] for k, v in sel.items()}, flush=True)

    # ---------------- 3: classical ML grid -----------------------------------------
    print("[3] classical ML", flush=True)
    preds, ml_sel = {}, {}
    for task, H in TASKS:
        P, S, sc, A = J._data(C.TEST_START, "recon", H)
        pool = sorted([w for w in C.make_windows(P, H) if w["tgt_max"] < C.TEST_START], key=lambda w: w["origin"])
        A_tr = {"F": C.flat_feats(S["train"]), "y": C.stack(S["train"])[3]}
        A_va = {"F": C.flat_feats(S["val"]), "y": C.stack(S["val"])[3], "mask": C.stack(S["val"])[4]}
        A_pool = {"F": C.flat_feats(pool), "y": C.stack(pool)[3], "Fte": C.flat_feats(S["test"]),
                  "yte": C.stack(S["test"])[3]}
        out, rows = ml_models(task, H, A_pool, A_tr, A_va)
        ml_sel.update(rows)
        for k, v in out.items():
            preds[f"{task}|{k}"] = v
        y_true = C.stack(S["test"])[3]
        preds[f"{task}|y_true"] = y_true
        preds[f"{task}|origins"] = np.array([w["origin"] for w in S["test"]])
        X = np.array([w["X"][:, C.TARGET_IDX] for w in S["test"]])         # unscaled inputs
        preds[f"{task}|SeasonalNaive"] = X[:, :H] if H > 1 else X[:, :1]   # value 24 h before each target

    # ---------------- 4a: test predictions of selected and submitted neural configs --
    print("[4a] neural test predictions", flush=True)
    pjobs = []
    for task, H in TASKS:
        for kind in T.NEURAL:
            ids = {"selected": sel[f"{task}|{kind}"], "submitted": T.GRIDS[kind].index(T.SUBMITTED[kind])}
            for role, ci in ids.items():
                for sd in C.SEEDS:
                    pjobs.append(dict(tag=f"{task}|{kind}|{role}", task=task, H=H, kind=kind,
                                      cfg=T.GRIDS[kind][ci], seed=sd, fit_end=C.TEST_START, scheme="recon",
                                      save_state=os.path.join(MODELS_DIR, f"{task}_{kind}_{ci}_{sd}.pt")))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(J.predict_saved, pjobs):
            preds.setdefault(r["tag"] + "|seeds", []).append((r["seed"], r["test_pred"]))
    for k in [k for k in preds if k.endswith("|seeds")]:
        arr = np.array([p for _, p in sorted(preds[k], key=lambda t: t[0])])
        preds[k] = arr
        preds[k.replace("|seeds", "")] = arr.mean(0)

    # ---------------- 4b: statistical models, refitted at each origin ---------------
    print("[4b] statistical models", flush=True)
    sjobs = []
    for task, H in TASKS:
        for o in preds[f"{task}|origins"]:
            for m in ("SARIMA", "HoltWinters", "Prophet"):
                sjobs.append(dict(model=m, H=H, origin=int(o), fit_end=C.TEST_START))
    stat = {}
    with ProcessPoolExecutor(max_workers=args.stat_workers) as ex:
        for r in ex.map(J.stat_job, sjobs, chunksize=4):
            stat[(r["model"], r["H"], r["origin"])] = r["pred"]
    for task, H in TASKS:
        for m in ("SARIMA", "HoltWinters", "Prophet"):
            preds[f"{task}|{m}"] = np.array([stat[(m, H, int(o))] for o in preds[f"{task}|origins"]])

    np.savez_compressed(os.path.join(C.RESULTS, "bench_predictions.npz"), **{k.replace("|", "__"): np.asarray(v)
                                                                           for k, v in preds.items()})
    json.dump({"neural": {k: {"selected_config_id": v, "selected_config": T.GRIDS[k.split("|")[1]][v]}
                          for k, v in sel.items()},
               "ml": ml_sel, "elapsed_sec": round(time.time() - t_start, 1)},
              open(os.path.join(C.RESULTS, "selection.json"), "w"), indent=2, default=str)
    print(f"[done] {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
