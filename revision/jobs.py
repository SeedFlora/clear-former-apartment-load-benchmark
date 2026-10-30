"""
Worker functions for the process pools (one training run or one statistical
refit per job).  Data are rebuilt once per worker process and cached.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import common as C  # noqa: E402

_CACHE = {}


def _data(fit_end, scheme, H, purge=True):
    key = (fit_end, scheme, H, purge)
    if key not in _CACHE:
        Pk = ("P", fit_end)
        if Pk not in _CACHE:
            _CACHE[Pk] = C.reconstruct(fit_end=fit_end, clip_covariates=fit_end is not None)
        P = _CACHE[Pk]
        ws = C.make_windows(P, H)
        S = C.split(P, ws, H, scheme, purge=purge)
        sc = C.Scaler(P)
        from sklearn.preprocessing import StandardScaler
        arrs = {}
        for part in ("train", "val", "test"):
            X, hid, dow, y, mask = C.stack(S[part])
            arrs[part] = dict(X=sc.X(X), hid=hid, dow=dow, y=y, mask=mask, SF=C.static_feats(S[part]))
        sfs = StandardScaler().fit(np.vstack([arrs["train"]["SF"], arrs["val"]["SF"]]))
        for part in arrs:
            arrs[part]["SF"] = sfs.transform(arrs[part]["SF"])
        _CACHE[key] = (P, S, sc, arrs)
    return _CACHE[key]


def neural_job(job):
    import torch
    torch.set_num_threads(2)
    import train as T
    P, S, sc, A = _data(job["fit_end"], job["scheme"], job["H"], job.get("purge", True))
    kind, cfg, seed, H = job["kind"], job["cfg"], job["seed"], job["H"]
    ablate = job.get("ablate")
    use_sf = T.uses_static(kind, ablate)
    model = T.build_neural(kind, cfg, H, seed, ablate)
    mask = A["val"]["mask"] if job["mask_val"] else np.ones_like(A["val"]["mask"])
    model, info = T.train_masked(model, A["train"]["X"], sc.y(A["train"]["y"]), A["train"]["hid"], A["train"]["dow"],
                                 A["val"]["X"], sc.y(A["val"]["y"]), A["val"]["hid"], A["val"]["dow"], mask,
                                 "cuda", seed=seed,
                                 sftr=A["train"]["SF"] if use_sf else None,
                                 sfva=A["val"]["SF"] if use_sf else None)
    import deep_models as DM
    vp = sc.inv(DM.predict_torch(model, A["val"]["X"], A["val"]["hid"], A["val"]["dow"], "cuda",
                                 sf=A["val"]["SF"] if use_sf else None)).reshape(len(A["val"]["y"]), -1)
    out = {"tag": job["tag"], "seed": seed, "info": info, "n_params": T.n_params(model), "val_pred": vp,
           "val_mae_masked": C.masked_mae(A["val"]["y"], vp, A["val"]["mask"]),
           "val_mae_all": C.masked_mae(A["val"]["y"], vp, np.ones_like(A["val"]["mask"]))}
    if job.get("save_state"):
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, job["save_state"])
    if job.get("predict_test"):
        tp = sc.inv(DM.predict_torch(model, A["test"]["X"], A["test"]["hid"], A["test"]["dow"], "cuda",
                                     sf=A["test"]["SF"] if use_sf else None)).reshape(len(A["test"]["y"]), -1)
        out["test_pred"] = tp
    return out


def predict_saved(job):
    """Test predictions of a model trained in the grid phase (state on disk)."""
    import torch
    torch.set_num_threads(2)
    import train as T
    import deep_models as DM
    P, S, sc, A = _data(job["fit_end"], job["scheme"], job["H"])
    model = T.build_neural(job["kind"], job["cfg"], job["H"], job["seed"])
    model.load_state_dict(torch.load(job["save_state"], weights_only=True))
    model.to("cuda")
    use_sf = T.uses_static(job["kind"])
    tp = sc.inv(DM.predict_torch(model, A["test"]["X"], A["test"]["hid"], A["test"]["dow"], "cuda",
                                 sf=A["test"]["SF"] if use_sf else None)).reshape(len(A["test"]["y"]), -1)
    return {"tag": job["tag"], "seed": job["seed"], "test_pred": tp}


def stat_job(job):
    """One statistical forecast at one origin, fitted on the grid history up to
    the origin (reconstructed hours included, as in the submission)."""
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    logging.getLogger("prophet").setLevel(logging.CRITICAL)
    import run_experiments as R
    P, S, sc, A = _data(job["fit_end"], "recon", job["H"])
    o, H = job["origin"], job["H"]
    hist = P["target"][:o + 1]
    if job["model"] == "SARIMA":
        fc = R.fit_forecast_sarima(hist, H)
    elif job["model"] == "HoltWinters":
        fc = R.fit_forecast_ets(hist, H)
    else:
        fc = R.fit_forecast_prophet(hist, P["times"][:o + 1], H)
    return {"model": job["model"], "H": H, "origin": o, "pred": np.asarray(fc, float).reshape(-1)[:H]}
