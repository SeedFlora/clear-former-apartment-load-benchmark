"""
Post hoc sensitivity of the neural ranking to the validation design and to
the training composition, added after the main results were seen.  Each
neural model keeps the configuration that validation selected in the main
protocol; only the data that train and early-stop it change.

  recon_obs       main protocol (taken from the benchmark, not re-run)
  recon_all       all fills; early stopping on every validation target (as submitted)
  recon_blockval  all fills; early stopping on fully observed windows before the
                  216 h gap, with every training window touching their span removed
  short_fills     windows touching the 216 h fill removed; observed validation
  observed        no reconstructed hour anywhere
CLEAR-Former's observed, short_fills and recon_all arms come from run_followup.py.
Three CLEAR-Former variants (linear backbone only, without the exogenous MLP,
without the attention branch) run under the same four designs, to test whether
the non-linear paths hurt the 24 h task only when the long fill is in training.
Output: results/sensitivity.json
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as C
import train as T

TASKS = (("1h", 1), ("day", 24))
ARMS = {
    "recon_all": dict(fit_end=C.TEST_START, scheme="recon", mask_val=False, purge=True),
    "recon_blockval": dict(fit_end=C.TEST_START, scheme="recon_blockval", mask_val=True, purge=True),
    "short_fills": dict(fit_end=C.TEST_START, scheme="short_fills", mask_val=True, purge=True),
    "observed": dict(fit_end=C.TEST_START, scheme="observed", mask_val=True, purge=True),
}


def main(workers=4):
    t0 = time.time()
    import jobs as J
    sel = json.load(open(os.path.join(C.RESULTS, "selection.json")))["neural"]
    jobs = []
    variants = {"Linear backbone only": {"use_featmlp": False, "use_attn": False},
                "-Exogenous MLP": {"use_featmlp": False}, "-Attention branch": {"use_attn": False}}
    for task, H in TASKS:
        for kind in T.NEURAL:
            cfg = sel[f"{task}|{kind}"]["selected_config"]
            for aname, arm in ARMS.items():
                if kind == "CLEAR-Former" and aname != "recon_blockval":
                    continue
                for sd in C.SEEDS:
                    jobs.append(dict(tag=f"{task}|{kind}|{aname}", task=task, H=H, kind=kind, cfg=cfg, seed=sd,
                                     predict_test=True, **arm))
        cfg = sel[f"{task}|CLEAR-Former"]["selected_config"]
        for vname, abl in variants.items():
            for aname, arm in ARMS.items():
                for sd in C.SEEDS:
                    jobs.append(dict(tag=f"{task}|CLEAR-Former {vname}|{aname}", task=task, H=H, kind="CLEAR-Former",
                                     cfg=cfg, seed=sd, ablate=abl, predict_test=True, **arm))
    import robust
    res = {}
    for r in robust.run_jobs(jobs, "neural_job", os.path.join(C.RESULTS, "sensitivity_jobs"), workers=workers):
        res.setdefault(r["tag"], []).append(r)
    out = {}
    for tag, rs in res.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        task, kind, aname = tag.split("|")
        arm = ARMS[aname]
        P, S, sc, A = J._data(arm["fit_end"], arm["scheme"], dict(TASKS)[task], arm["purge"])
        y = A["test"]["y"]
        seeds = np.array([r["test_pred"] for r in rs])
        m = C.M.all_metrics(y, seeds.mean(0))
        m.update({"seed_MAE": [C.M.mae(y, p) for p in seeds], "seed_MAE_sd": float(np.std([C.M.mae(y, p) for p in seeds], ddof=1)),
                  "val_mae_masked_seeds": [r["val_mae_masked"] for r in rs],
                  "n_train": len(S["train"]), "n_val": len(S["val"]), "n_purged": S["n_purged"],
                  "val_first_origin": int(S["val"][0]["origin"]), "val_last_origin": int(S["val"][-1]["origin"]),
                  "best_epochs": [r["info"]["best_epoch"] for r in rs]})
        out[tag] = m
        print(f"{tag:36s} MAE={m['MAE']:.3f} R2={m['R2']:.3f} seedSD={m['seed_MAE_sd']:.3f} n_train={m['n_train']} n_val={m['n_val']}")
    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(C.RESULTS, "sensitivity.json"), "w"), indent=2)
    print(f"[done] {out['elapsed_sec']}s")


if __name__ == "__main__":
    main()
