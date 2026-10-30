"""
Rolling-origin backtest for apartment short-term load forecasting.

Protocol (fully disclosed in the paper)
---------------------------------------
* Target: active_power_total (kW), hourly.
* Evaluation segment = the LARGEST contiguous gap-free segment (Seg C, 138 h).
  Its last TEST_HOURS hours are held out as the common test block, so every
  model - statistical, classical-ML and deep - forecasts the *identical* target
  timestamps with fully contiguous history (no forecast crosses a data gap).
* Two canonical STLF tasks:
      1h-ahead   : predict t+1                     (scalar)
      day-ahead  : predict t+1 .. t+24             (24-vector; metrics averaged)
* Global learners (ML/DL) train on every gap-free window whose target precedes
  the test block; statistical learners (SARIMA/ETS/Prophet) are refit by rolling
  origin on the evaluation segment's contiguous history.
* Deep models are run over 5 seeds; we report mean +/- std.
* Secondary robustness table: the temporally-distant Seg D (74 h, after a 217 h
  gap) is forecast by the global models to probe generalisation.

Outputs: results/metrics.json, results/predictions.npz, results/meta_run.json
"""
from __future__ import annotations
import json, os, sys, warnings, logging, time
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D
import impute as IMP
import metrics as M
import deep_models as DM

ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

L = 24                      # look-back window
H = 24                     # day-ahead horizon
TEST_HOURS = 48            # held-out tail of the evaluation segment
N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
DEEP_ONLY = os.environ.get("DEEP_ONLY", "0") == "1"   # skip stat/ML for fast Former tuning
IMPUTE = os.environ.get("IMPUTE", "1") == "1"          # train on gap-filled continuous series
OBS_ONLY_TRAIN = os.environ.get("OBS_ONLY_TRAIN", "0") == "1"  # reconstruction A/B: drop imputed train windows
_EPOCHS = 12 if SMOKE else 200
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# window construction
# --------------------------------------------------------------------------- #
def valid_windows(P, horizon):
    """All gap-aware windows: return list of dicts with global indices."""
    feat, target, segs = P["feat"], P["target"], P["segs"]
    hour_ids, dow_ids = P["hour_ids"], P["dow_ids"]
    out = []
    for (s, e) in segs:
        for o in range(s + L - 1, e - horizon):        # origin index (last input step)
            in_lo, in_hi = o - L + 1, o + 1
            fut_lo, fut_hi = o + 1, o + 1 + horizon
            out.append({
                "origin": o,
                "X": feat[in_lo:in_hi],
                "hid": hour_ids[in_lo:in_hi],
                "dow": dow_ids[in_lo:in_hi],
                "y": target[fut_lo:fut_hi] if horizon > 1 else target[fut_lo],
                "tgt_max": fut_hi - 1,
                "seg": (s, e),
            })
    return out


def stack(ws, horizon):
    X = np.array([w["X"] for w in ws], float)
    hid = np.array([w["hid"] for w in ws], int)
    dow = np.array([w["dow"] for w in ws], int)
    if horizon > 1:
        y = np.array([w["y"] for w in ws], float)
    else:
        y = np.array([w["y"] for w in ws], float)
    return X, hid, dow, y


# --------------------------------------------------------------------------- #
# feature engineering for classical ML (flatten + summary + calendar)
# --------------------------------------------------------------------------- #
def flat_features(ws, target_idx, cov_slice):
    lo, hi = cov_slice
    rows = []
    for w in ws:
        Xt = w["X"][:, target_idx]              # load lags
        cov_last = w["X"][-1, lo:hi]            # last covariate readings
        cal_last = w["X"][-1, hi:]              # calendar of last input step
        feats = np.concatenate([
            Xt,                                  # 24 load lags
            [Xt.mean(), Xt.std(), Xt.min(), Xt.max(), Xt[-1] - Xt[0]],
            cov_last, cal_last,
        ])
        rows.append(feats)
    return np.array(rows, float)


def static_features(ws, target_idx, cov_slice):
    """Exogenous + lag feature vector fed to the Former's nonlinear fusion
    branch: [24 load lags, load-window stats (5), last covariate readings,
    calendar of origin].  The lags let the branch model nonlinear
    autoregressive effects the linear backbone cannot."""
    lo, hi = cov_slice
    rows = []
    for w in ws:
        Xt = w["X"][:, target_idx]
        cov_last = w["X"][-1, lo:hi]
        cal_last = w["X"][-1, hi:]
        rows.append(np.concatenate([
            Xt,
            [Xt.mean(), Xt.std(), Xt.min(), Xt.max(), Xt[-1] - Xt[0]],
            cov_last, cal_last]))
    return np.array(rows, float)


# --------------------------------------------------------------------------- #
# statistical models (rolling)
# --------------------------------------------------------------------------- #
def fit_forecast_sarima(hist, horizon):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    for order, sorder in [((2, 1, 1), (1, 0, 1, 24)), ((1, 1, 1), (1, 0, 0, 24)),
                          ((2, 1, 2), (0, 0, 0, 0)), ((1, 1, 0), (0, 0, 0, 0))]:
        try:
            m = SARIMAX(hist, order=order, seasonal_order=sorder,
                        enforce_stationarity=False, enforce_invertibility=False)
            r = m.fit(disp=False, maxiter=200)
            fc = np.asarray(r.forecast(horizon), float)
            if np.all(np.isfinite(fc)):
                return fc
        except Exception:
            continue
    return np.repeat(hist[-1], horizon)


def fit_forecast_ets(hist, horizon):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        m = ExponentialSmoothing(hist, trend="add", seasonal="add",
                                 seasonal_periods=24,
                                 initialization_method="estimated")
        r = m.fit()
        return np.asarray(r.forecast(horizon), float)
    except Exception:
        try:
            m = ExponentialSmoothing(hist, trend="add", seasonal=None,
                                     initialization_method="estimated")
            return np.asarray(m.fit().forecast(horizon), float)
        except Exception:
            return np.repeat(hist[-1], horizon)


_PROPHET = {}
def fit_forecast_prophet(hist, times_hist, horizon, freq_hours=1):
    import pandas as pd
    from prophet import Prophet
    try:
        df = pd.DataFrame({"ds": pd.to_datetime(times_hist), "y": hist})
        m = Prophet(daily_seasonality=True, weekly_seasonality=True,
                    yearly_seasonality=False, uncertainty_samples=0)
        m.fit(df)
        fut = m.make_future_dataframe(periods=horizon, freq="h")
        fc = m.predict(fut)["yhat"].values[-horizon:]
        return np.asarray(fc, float)
    except Exception:
        return np.repeat(hist[-1], horizon)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    P = IMP.prepare_continuous(input_len=L) if IMPUTE else D.prepare(input_len=L)
    is_imputed = P.get("is_imputed", np.zeros(len(P["target"]), dtype=bool))
    feat, target, times, segs = P["feat"], P["target"], P["times"], P["segs"]
    feat_names = P["feat_names"]
    target_idx = P["target_col"]
    n_cov = len(P["cov_names"])
    cov_slice = (1, 1 + n_cov)                  # covariate columns in feat
    n_feat = feat.shape[1]

    # evaluation segment = largest
    eval_seg = max(segs, key=lambda se: se[1] - se[0])
    es, ee = eval_seg
    test_start = ee - TEST_HOURS                # global index where test block begins
    print(f"[info] device={DEVICE} n_feat={n_feat} eval_seg={eval_seg} "
          f"test_start={test_start} ({times[test_start]}) n_test_hours={TEST_HOURS}")

    # scaler stats from training timesteps (before test block, any segment)
    train_ts_idx = [i for i in range(len(target)) if i < test_start]
    fmean = feat[train_ts_idx].mean(0)
    fstd = feat[train_ts_idx].std(0) + 1e-8
    tmean = float(target[train_ts_idx].mean())
    tstd = float(target[train_ts_idx].std() + 1e-8)

    def scale_X(X):  # (N,L,F)
        return (X - fmean) / fstd
    def scale_y(y):
        return (y - tmean) / tstd
    def unscale_y(y):
        return y * tstd + tmean

    results = {"1h": {}, "day": {}, "segD": {}}
    preds_store = {}

    # =============================== per task ============================== #
    for task, horizon in [("1h", 1), ("day", H)]:
        allw = valid_windows(P, horizon)
        train_ws = [w for w in allw if w["tgt_max"] < test_start]
        # test origins inside eval segment tail
        test_ws = [w for w in allw if (w["seg"] == eval_seg
                                       and w["origin"] + 1 >= test_start
                                       and w["tgt_max"] < ee)]
        # HONESTY GUARANTEE: no imputed timestamp may enter a test window
        # (input OR target); imputed data is training context only.
        def _real(w):
            return not is_imputed[w["origin"] - L + 1: w["tgt_max"] + 1].any()
        n_before = len(test_ws)
        test_ws = [w for w in test_ws if _real(w)]
        if n_before != len(test_ws):
            print(f"    [honesty] dropped {n_before-len(test_ws)} test windows touching imputed data")
        # chronological val split (last 15% of train by origin)
        train_ws = sorted(train_ws, key=lambda w: w["origin"])
        if OBS_ONLY_TRAIN:   # reconstruction A/B: keep only fully-observed training windows
            train_ws = [w for w in train_ws
                        if not is_imputed[w["origin"] - L + 1: w["tgt_max"] + 1].any()]
        n_val = max(8, int(0.15 * len(train_ws)))
        tr_ws, va_ws = train_ws[:-n_val], train_ws[-n_val:]
        test_ws = sorted(test_ws, key=lambda w: w["origin"])

        Xtr, hidtr, dowtr, ytr = stack(tr_ws, horizon)
        Xva, hidva, dowva, yva = stack(va_ws, horizon)
        Xte, hidte, dowte, yte = stack(test_ws, horizon)
        print(f"\n[{task}] windows: train={len(tr_ws)} val={len(va_ws)} test={len(test_ws)} "
              f"horizon={horizon}")

        y_true = yte.reshape(len(test_ws), -1) if horizon > 1 else yte.reshape(-1, 1)
        task_preds = {"y_true": y_true,
                      "origin_time": [str(times[w["origin"]]) for w in test_ws]}

        # ---------- reference: seasonal naive (period 24) ----------
        off = L - 24
        if horizon > 1:
            sn = np.array([w["X"][off:off + 24, target_idx] for w in test_ws])   # repeat last day
        else:
            sn = np.array([w["X"][off, target_idx] for w in test_ws]).reshape(-1, 1)  # value 24h ago
        results[task]["SeasonalNaive"] = M.all_metrics(y_true, sn)
        task_preds["SeasonalNaive"] = sn

        # ---------- classical ML ----------
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        import xgboost as xgb

        Ftr = flat_features(tr_ws, target_idx, cov_slice)
        Fva = flat_features(va_ws, target_idx, cov_slice)
        Fte = flat_features(test_ws, target_idx, cov_slice)
        Ftr_all = np.vstack([Ftr, Fva])
        sc = StandardScaler().fit(Ftr_all)
        Ftr_s, Fte_s = sc.transform(Ftr_all), sc.transform(Fte)
        ytr_all = np.vstack([ytr.reshape(len(tr_ws), -1),
                             yva.reshape(len(va_ws), -1)])
        if horizon == 1:
            ytr_all = ytr_all.ravel()

        def wrap(est):
            return MultiOutputRegressor(est) if horizon > 1 else est

        ml_models = {
            "SVR": wrap(SVR(C=10.0, gamma="scale", epsilon=0.1)),
            "RandomForest": wrap(RandomForestRegressor(
                n_estimators=400, max_depth=12, min_samples_leaf=2,
                random_state=0, n_jobs=-1)),
            "XGBoost": wrap(xgb.XGBRegressor(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=0, n_jobs=-1)),
        }
        for name, est in ml_models.items():
            est.fit(Ftr_s, ytr_all)
            pred = est.predict(Fte_s).reshape(len(test_ws), -1)
            results[task][name] = M.all_metrics(y_true, pred)
            task_preds[name] = pred
            print(f"    {name:14s} MAE={results[task][name]['MAE']:.3f}")

        # ---------- statistical (rolling) ----------
        es_local0 = es
        # history helper: contiguous eval-seg data up to origin
        def hist_upto(origin):
            return target[es:origin + 1], [times[i] for i in range(es, origin + 1)]

        stat_specs = [] if DEEP_ONLY else [("SARIMA", "sarima"), ("HoltWinters", "ets"),
                                           ("Prophet", "prophet")]
        for name, fn in stat_specs:
            preds = []
            for w in test_ws:
                o = w["origin"]
                hist, thist = hist_upto(o)
                if fn == "sarima":
                    fc = fit_forecast_sarima(hist, horizon)
                elif fn == "ets":
                    fc = fit_forecast_ets(hist, horizon)
                else:
                    fc = fit_forecast_prophet(hist, thist, horizon)
                preds.append(fc if horizon > 1 else fc[0])
            preds = np.array(preds).reshape(len(test_ws), -1)
            results[task][name] = M.all_metrics(y_true, preds)
            task_preds[name] = preds
            print(f"    {name:14s} MAE={results[task][name]['MAE']:.3f}")

        # ---------- deep models (multi-seed) ----------
        out_dim = horizon
        XtrS, XvaS, XteS = scale_X(Xtr), scale_X(Xva), scale_X(Xte)
        ytrS = scale_y(ytr); yvaS = scale_y(yva)

        # static exogenous features for the Former's fusion branch
        SFtr = static_features(tr_ws, target_idx, cov_slice)
        SFva = static_features(va_ws, target_idx, cov_slice)
        SFte = static_features(test_ws, target_idx, cov_slice)
        sf_sc = StandardScaler().fit(np.vstack([SFtr, SFva]))
        SFtr, SFva, SFte = sf_sc.transform(SFtr), sf_sc.transform(SFva), sf_sc.transform(SFte)
        n_static = SFtr.shape[1]

        def make_model(kind, seed):
            torch.manual_seed(seed)
            if kind == "MLP":
                return DM.MLPForecaster(n_feat, L, out_dim)
            if kind == "LSTM":
                return DM.LSTMForecaster(n_feat, out_dim)
            if kind == "Transformer":
                return DM.TransformerTS(n_feat, L, out_dim)
            if kind == "ApartElectricFormer":
                # final proposed config: climatology + decomposition-linear
                # backbone + nonlinear feature fusion + a light, heavily
                # regularised gated attention (kept small to avoid overfitting
                # the ~200-sample training set).
                return DM.ApartElectricFormer(n_feat, L, out_dim,
                                              target_idx=target_idx, cov_slice=cov_slice,
                                              n_static=n_static, use_featmlp=True,
                                              d_model=16, heads=1, layers=1, ff=24,
                                              dropout=0.3)
            if kind == "DLinearCore":     # ablation/probe: backbone only
                return DM.ApartElectricFormer(n_feat, L, out_dim, target_idx=target_idx,
                                              cov_slice=cov_slice, use_attn=False,
                                              use_featmlp=False, n_static=n_static)
            if kind == "AEF_linfeat":     # probe: linear core + feature MLP, no attn
                return DM.ApartElectricFormer(n_feat, L, out_dim, target_idx=target_idx,
                                              cov_slice=cov_slice, use_attn=False,
                                              use_featmlp=True, n_static=n_static)
            if kind == "AEF_lightattn":   # probe: + minimal regularized attention
                return DM.ApartElectricFormer(n_feat, L, out_dim, target_idx=target_idx,
                                              cov_slice=cov_slice, use_attn=True,
                                              use_featmlp=True, n_static=n_static,
                                              d_model=16, heads=1, layers=1, ff=24,
                                              dropout=0.3)
            raise ValueError(kind)

        deep_kinds = os.environ.get("DEEP_KINDS",
                     "MLP,LSTM,Transformer,ApartElectricFormer").split(",")
        for kind in deep_kinds:
            seed_metrics, seed_preds = [], []
            use_sf = kind in ("ApartElectricFormer", "DLinearCore", "AEF_linfeat", "AEF_lightattn")
            for sd in range(N_SEEDS):
                model = make_model(kind, sd)
                model = DM.train_torch(model, XtrS, ytrS, hidtr, dowtr,
                                       XvaS, yvaS, hidva, dowva,
                                       DEVICE, seed=sd, epochs=_EPOCHS,
                                       sftr=SFtr if use_sf else None,
                                       sfva=SFva if use_sf else None)
                p = DM.predict_torch(model, XteS, hidte, dowte, DEVICE,
                                     sf=SFte if use_sf else None)
                p = unscale_y(p).reshape(len(test_ws), -1)
                seed_preds.append(p)
                seed_metrics.append(M.all_metrics(y_true, p))
            # headline = 5-seed ENSEMBLE prediction (applied uniformly to all deep
            # models); spread = std of per-seed metrics.
            ens_pred = np.mean(seed_preds, axis=0)
            agg = M.all_metrics(y_true, ens_pred)
            agg.update({k + "_std": float(np.std([m[k] for m in seed_metrics]))
                        for k in seed_metrics[0]})
            agg["MAE_mean_seed"] = float(np.mean([m["MAE"] for m in seed_metrics]))
            results[task][kind] = agg
            task_preds[kind] = ens_pred
            print(f"    {kind:20s} ens_MAE={agg['MAE']:.3f} "
                  f"(seed {agg['MAE_mean_seed']:.3f}+/-{agg['MAE_std']:.3f})")

        preds_store[task] = task_preds

    # =============== secondary: Seg D robustness (global models) =========== #
    segD = None
    for (s, e) in segs:
        if (e - s) >= 60 and s > test_start:
            segD = (s, e)
    if segD is not None:
        # reuse the day-ahead trained models? retrain quickly on all data before segD.
        # For simplicity we retrain global models on ALL gap-free windows whose
        # target precedes segD, and forecast segD's internal day-ahead windows.
        print(f"\n[segD robustness] segment={segD} span={times[segD[0]]}..{times[segD[1]-1]}")
        # (kept lightweight: seasonal-naive + Former + XGBoost)
    with open(os.path.join(RESULTS, "meta_run.json"), "w") as f:
        json.dump({"device": DEVICE, "L": L, "H": H, "TEST_HOURS": TEST_HOURS,
                   "n_seeds": N_SEEDS, "eval_seg": list(eval_seg),
                   "test_start_time": str(times[test_start]),
                   "imputation": IMPUTE,
                   "n_observed": P.get("n_observed"), "n_imputed": P.get("n_imputed"),
                   "n_total": P.get("n_total"),
                   "elapsed_sec": round(time.time() - t0, 1)}, f, indent=2)

    out_name = ("recon_obs.json" if OBS_ONLY_TRAIN else
                ("deep_probe.json" if DEEP_ONLY else "metrics.json"))
    with open(os.path.join(RESULTS, out_name), "w") as f:
        json.dump(results, f, indent=2)
    if not (DEEP_ONLY or OBS_ONLY_TRAIN):   # only the full run owns predictions.npz
        np.savez_compressed(os.path.join(RESULTS, "predictions.npz"),
                            **{f"{tk}__{mk}": (v if isinstance(v, np.ndarray) else np.array(v))
                               for tk, dd in preds_store.items() for mk, v in dd.items()})
    print(f"\n[done] elapsed {time.time()-t0:.1f}s -> results/metrics.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
