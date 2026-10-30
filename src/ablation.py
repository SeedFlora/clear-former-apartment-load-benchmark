"""
Ablation study for CLEAR-Former: remove one component at a time and re-measure
both STLF tasks (1h-ahead, day-ahead), 5 seeds, ensemble metric.

Variants:
  Full                 - the proposed model
  -Climatology         - drop the hour/day-of-week climatology anchor
  -Decomposition       - drop trend/seasonal series decomposition
  -FeatureFusion       - drop the nonlinear exogenous feature branch
  -Attention           - drop the gated attention module (backbone only)
  LinearBackboneOnly   - climatology + decomposition-linear only
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D
import impute as IMP
import metrics as M
import deep_models as DM
import run_experiments as R
import torch

RESULTS = os.path.join(os.path.dirname(HERE), "results")
N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
IMPUTE = os.environ.get("IMPUTE", "1") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VARIANTS = {
    "Full":               dict(use_clim=True,  use_decomp=True,  use_featmlp=True,  use_attn=True),
    "-Climatology":       dict(use_clim=False, use_decomp=True,  use_featmlp=True,  use_attn=True),
    "-Decomposition":     dict(use_clim=True,  use_decomp=False, use_featmlp=True,  use_attn=True),
    "-FeatureFusion":     dict(use_clim=True,  use_decomp=True,  use_featmlp=False, use_attn=True),
    "-Attention":         dict(use_clim=True,  use_decomp=True,  use_featmlp=True,  use_attn=False),
    "LinearBackboneOnly": dict(use_clim=True,  use_decomp=True,  use_featmlp=False, use_attn=False),
}
LIGHT = dict(d_model=16, heads=1, layers=1, ff=24, dropout=0.3)


def main():
    P = IMP.prepare_continuous(input_len=R.L) if IMPUTE else D.prepare(input_len=R.L)
    is_imputed = P.get("is_imputed", np.zeros(len(P["target"]), dtype=bool))
    feat = P["feat"]; target = P["target"]; times = P["times"]; segs = P["segs"]
    target_idx = P["target_col"]; n_feat = feat.shape[1]
    n_cov = len(P["cov_names"]); cov_slice = (1, 1 + n_cov)
    eval_seg = max(segs, key=lambda se: se[1] - se[0]); es, ee = eval_seg
    test_start = ee - R.TEST_HOURS
    tr_ts = [i for i in range(len(target)) if i < test_start]
    fmean = feat[tr_ts].mean(0); fstd = feat[tr_ts].std(0) + 1e-8
    tmean = float(target[tr_ts].mean()); tstd = float(target[tr_ts].std() + 1e-8)

    out = {"1h": {}, "day": {}}
    from sklearn.preprocessing import StandardScaler
    for task, horizon in [("1h", 1), ("day", R.H)]:
        allw = R.valid_windows(P, horizon)
        train_ws = sorted([w for w in allw if w["tgt_max"] < test_start], key=lambda w: w["origin"])
        test_ws = sorted([w for w in allw if w["seg"] == eval_seg and w["origin"] + 1 >= test_start
                          and w["tgt_max"] < ee
                          and not is_imputed[w["origin"] - R.L + 1: w["tgt_max"] + 1].any()],
                         key=lambda w: w["origin"])
        n_val = max(8, int(0.15 * len(train_ws)))
        tr_ws, va_ws = train_ws[:-n_val], train_ws[-n_val:]
        Xtr, hidtr, dowtr, ytr = R.stack(tr_ws, horizon)
        Xva, hidva, dowva, yva = R.stack(va_ws, horizon)
        Xte, hidte, dowte, yte = R.stack(test_ws, horizon)
        y_true = yte.reshape(len(test_ws), -1) if horizon > 1 else yte.reshape(-1, 1)
        sX = lambda X: (X - fmean) / fstd
        XtrS, XvaS, XteS = sX(Xtr), sX(Xva), sX(Xte)
        ytrS = (ytr - tmean) / tstd; yvaS = (yva - tmean) / tstd
        SFtr = R.static_features(tr_ws, target_idx, cov_slice)
        SFva = R.static_features(va_ws, target_idx, cov_slice)
        SFte = R.static_features(test_ws, target_idx, cov_slice)
        sf_sc = StandardScaler().fit(np.vstack([SFtr, SFva]))
        SFtr, SFva, SFte = sf_sc.transform(SFtr), sf_sc.transform(SFva), sf_sc.transform(SFte)
        n_static = SFtr.shape[1]

        for vname, flags in VARIANTS.items():
            seed_preds, seed_maes = [], []
            for sd in range(N_SEEDS):
                torch.manual_seed(sd)
                model = DM.ApartElectricFormer(n_feat, R.L, horizon, target_idx=target_idx,
                                               cov_slice=cov_slice, n_static=n_static,
                                               **flags, **LIGHT)
                use_sf = flags["use_featmlp"]
                model = DM.train_torch(model, XtrS, ytrS, hidtr, dowtr, XvaS, yvaS, hidva, dowva,
                                       DEVICE, seed=sd, epochs=200,
                                       sftr=SFtr if use_sf else None, sfva=SFva if use_sf else None)
                p = DM.predict_torch(model, XteS, hidte, dowte, DEVICE, sf=SFte if use_sf else None)
                p = (p * tstd + tmean).reshape(len(test_ws), -1)
                seed_preds.append(p); seed_maes.append(M.mae(y_true, p))
            ens = np.mean(seed_preds, axis=0)
            met = M.all_metrics(y_true, ens)
            met["MAE_std"] = float(np.std(seed_maes))
            out[task][vname] = met
            print(f"[{task}] {vname:20s} MAE={met['MAE']:.3f} RMSE={met['RMSE']:.3f} R2={met['R2']:.3f}")

    with open(os.path.join(RESULTS, "ablation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("-> results/ablation.json")


if __name__ == "__main__":
    main()
