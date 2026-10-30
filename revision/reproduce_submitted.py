"""Re-run the submitted protocol for CLEAR-Former only (original reconstruction,
unmasked validation, submitted light configuration, 5 seeds) to check that the
submitted numbers (1 h MAE 5.092, 24 h MAE 15.395) reproduce on this machine.
Writes revision/results/reproduce_submitted.json; touches no original file."""
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import impute as IMP, deep_models as DM, run_experiments as R, metrics as M
from sklearn.preprocessing import StandardScaler
P = IMP.prepare_continuous(input_len=R.L)          # original: statistics from all observed hours
feat, target = P["feat"], P["target"]; test_start = 720 - R.TEST_HOURS
fmean = feat[:test_start].mean(0); fstd = feat[:test_start].std(0) + 1e-8
tmean = float(target[:test_start].mean()); tstd = float(target[:test_start].std() + 1e-8)
out = {}
for task, H in [("1h", 1), ("day", 24)]:
    allw = R.valid_windows(P, H)
    tr = sorted([w for w in allw if w["tgt_max"] < test_start], key=lambda w: w["origin"])
    te = sorted([w for w in allw if w["origin"] + 1 >= test_start and w["tgt_max"] < 720], key=lambda w: w["origin"])
    nv = max(8, int(0.15 * len(tr))); trw, vaw = tr[:-nv], tr[-nv:]
    (Xtr, htr, dtr, ytr), (Xva, hva, dva, yva), (Xte, hte, dte, yte) = R.stack(trw, H), R.stack(vaw, H), R.stack(te, H)
    SF = [R.static_features(w, 0, (1, 8)) for w in (trw, vaw, te)]
    sc = StandardScaler().fit(np.vstack(SF[:2])); SF = [sc.transform(s) for s in SF]
    yt = yte.reshape(len(te), -1)
    preds = []
    for sd in range(5):
        torch.manual_seed(sd)
        m = DM.ApartElectricFormer(13, 24, H, target_idx=0, cov_slice=(1, 8), n_static=SF[0].shape[1],
                                   use_featmlp=True, d_model=16, heads=1, layers=1, ff=24, dropout=0.3)
        m = DM.train_torch(m, (Xtr - fmean) / fstd, (ytr - tmean) / tstd, htr, dtr, (Xva - fmean) / fstd,
                           (yva - tmean) / tstd, hva, dva, "cuda", seed=sd, sftr=SF[0], sfva=SF[1])
        preds.append((DM.predict_torch(m, (Xte - fmean) / fstd, hte, dte, "cuda", sf=SF[2]) * tstd + tmean).reshape(len(te), -1))
    out[task] = {"ens_MAE": M.mae(yt, np.mean(preds, 0)), "seed_MAE": [M.mae(yt, p) for p in preds]}
    print(task, out[task], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "reproduce_submitted.json"), "w"), indent=2)
