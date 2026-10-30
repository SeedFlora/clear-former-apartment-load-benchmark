"""
Analyses run after selection, on the CLEAR-Former configuration that
validation selected for each task (results/selection.json).  All are
exploratory test-block analyses; none feeds back into selection.

  ablation   one path or sub-component removed at a time (8 variants)
  arms       where the reconstruction enters:
               observed     no reconstructed hour anywhere (fully observed windows)
               recon_all    reconstructed training, validation scored on every target
               recon_obs    reconstructed training, validation on observed hours (main protocol)
               short_fills  as recon_obs without windows that touch the 216 h fill
  chain      submitted configuration, one correction added at a time:
               a) submitted pipeline (leaky reconstruction, no purge, all-target validation)
               b) + reconstruction fitted on pre-test hours
               c) + purge between training and validation
               d) + validation on observed hours only
Output: results/followup.json (+ test predictions in followup_predictions.npz)
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
ABLATIONS = {
    "Full": {},
    "-Climatology": {"use_clim": False},
    "-Decomposition": {"use_decomp": False},
    "-Exogenous MLP": {"use_featmlp": False},
    "-Attention branch": {"use_attn": False},
    "Linear backbone only": {"use_featmlp": False, "use_attn": False},
    "-Covariate gate (in attention)": {"use_covfuse": False},
    "-Hour/day embeddings (in attention)": {"use_periodic": False},
}
ARMS = {
    "observed": dict(fit_end=C.TEST_START, scheme="observed", mask_val=True, purge=True),
    "recon_all": dict(fit_end=C.TEST_START, scheme="recon", mask_val=False, purge=True),
    "recon_obs": dict(fit_end=C.TEST_START, scheme="recon", mask_val=True, purge=True),
    "short_fills": dict(fit_end=C.TEST_START, scheme="short_fills", mask_val=True, purge=True),
}
CHAIN = {
    "a_submitted": dict(fit_end=None, scheme="recon", mask_val=False, purge=False),
    "b_pretest_recon": dict(fit_end=C.TEST_START, scheme="recon", mask_val=False, purge=False),
    "c_purge": dict(fit_end=C.TEST_START, scheme="recon", mask_val=False, purge=True),
    "d_masked_val": dict(fit_end=C.TEST_START, scheme="recon", mask_val=True, purge=True),
}


def main(workers=6):
    t0 = time.time()
    import jobs as J
    sel = json.load(open(os.path.join(C.RESULTS, "selection.json")))["neural"]
    jobs = []
    for task, H in TASKS:
        cfg = sel[f"{task}|CLEAR-Former"]["selected_config"]
        for vname, abl in ABLATIONS.items():
            for sd in C.SEEDS:
                jobs.append(dict(tag=f"ablation|{task}|{vname}", task=task, H=H, kind="CLEAR-Former", cfg=cfg,
                                 seed=sd, ablate=abl, predict_test=True, **ARMS["recon_obs"]))
        for aname, arm in ARMS.items():
            if aname == "recon_obs":
                continue                                    # identical to ablation|Full
            for sd in C.SEEDS:
                jobs.append(dict(tag=f"arm|{task}|{aname}", task=task, H=H, kind="CLEAR-Former", cfg=cfg,
                                 seed=sd, predict_test=True, **arm))
        for cname, arm in CHAIN.items():
            for sd in C.SEEDS:
                jobs.append(dict(tag=f"chain|{task}|{cname}", task=task, H=H, kind="CLEAR-Former",
                                 cfg=T.SUBMITTED["CLEAR-Former"], seed=sd, predict_test=True, **arm))
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, r in enumerate(ex.map(J.neural_job, jobs)):
            res.setdefault(r["tag"], []).append(r)
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)

    out, preds = {}, {}
    for tag, rs in res.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        _, task, _ = tag.split("|")
        H = dict(TASKS)[task]
        arm = (ARMS["recon_obs"] if tag.startswith("ablation") else
               ARMS[tag.split("|")[2]] if tag.startswith("arm") else CHAIN[tag.split("|")[2]])
        P, S, sc, A = J._data(arm["fit_end"], arm["scheme"], H, arm["purge"])
        y = A["test"]["y"]
        seeds = np.array([r["test_pred"] for r in rs])
        ens = seeds.mean(0)
        m = C.M.all_metrics(y, ens)
        m.update({"seed_MAE": [C.M.mae(y, p) for p in seeds],
                  "seed_MAE_sd": float(np.std([C.M.mae(y, p) for p in seeds])),
                  "val_mae_masked_seeds": [r["val_mae_masked"] for r in rs],
                  "n_params": rs[0]["n_params"], "best_epochs": [r["info"]["best_epoch"] for r in rs],
                  "n_train": len(S["train"]), "n_val": len(S["val"]), "n_purged": S["n_purged"]})
        out[tag] = m
        preds[tag.replace("|", "__")] = seeds
        print(f"{tag:55s} MAE={m['MAE']:.3f} R2={m['R2']:.3f} seedSD={m['seed_MAE_sd']:.3f} n_train={m['n_train']}")
    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(C.RESULTS, "followup.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(C.RESULTS, "followup_predictions.npz"), **preds)
    print(f"[done] {out['elapsed_sec']}s")


if __name__ == "__main__":
    main()
