"""
Generate publication figures (vector PDF) for the paper from the REAL
experiment outputs.  Colour-blind-safe palette (Okabe-Ito); proposed model
(CLEAR-Former) highlighted throughout.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D
import impute as IMP

ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})
OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "yellow": "#F0E442", "black": "#333333", "grey": "#999999"}
PROP = OK["red"]

DISP = {"ApartElectricFormer": "CLEAR-Former", "SeasonalNaive": "Seasonal-Naive",
        "HoltWinters": "Holt-Winters", "RandomForest": "Random Forest"}
def disp(k): return DISP.get(k, k)


def _load():
    P = D.prepare(24)
    pred = np.load(os.path.join(RES, "predictions.npz"), allow_pickle=True)
    metrics = json.load(open(os.path.join(RES, "metrics.json")))
    return P, pred, metrics


# --------------------------------------------------------------------------- #
def fig_series(P):
    t = P["times"]; y = P["target"]; segs = P["segs"]
    fig, ax = plt.subplots(figsize=(5.5, 2.3))
    # plot each contiguous segment separately (do not draw across gaps)
    for i, (s, e) in enumerate(segs):
        if e - s < 2:
            continue
        ax.plot(t[s:e], y[s:e], color=OK["blue"], lw=0.9)
    # shade gaps
    for i in range(len(segs) - 1):
        g0 = t[segs[i][1] - 1]; g1 = t[segs[i + 1][0]]
        ax.axvspan(g0, g1, color=OK["grey"], alpha=0.18)
    ax.set_ylabel("Active power (kW)")
    ax.set_xlabel("Time")
    ax.set_title("Hourly total active-power load (grey = data gaps)")
    fig.autofmt_xdate(rotation=25)
    fig.savefig(os.path.join(FIG, "fig_series.pdf")); plt.close(fig)


def fig_imputation():
    """Reconstructed continuous month: observed vs seasonal-trend-imputed spans."""
    Q = IMP.prepare_continuous()
    t = Q["times"]; y = Q["target"]; imp = Q["is_imputed"]
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    # shade imputed spans
    i = 0
    labelled = False
    while i < len(imp):
        if imp[i]:
            j = i
            while j < len(imp) and imp[j]:
                j += 1
            ax.axvspan(t[i], t[min(j, len(t) - 1)], color=OK["orange"], alpha=0.18,
                       label=("imputed span" if not labelled else None))
            labelled = True
            i = j
        else:
            i += 1
    # observed points and imputed points
    yo = np.where(imp, np.nan, y)
    yi = np.where(imp, y, np.nan)
    ax.plot(t, yo, color=OK["blue"], lw=0.9, label="observed")
    ax.plot(t, yi, color=OK["red"], lw=0.9, label="imputed fill")
    ax.set_ylabel("Active power (kW)"); ax.set_xlabel("Time")
    ax.set_title("Reconstructed continuous month (264 of 720 h imputed)")
    ax.legend(loc="upper right", ncol=3, fontsize=6.5)
    fig.autofmt_xdate(rotation=25)
    fig.savefig(os.path.join(FIG, "fig_imputation.pdf")); plt.close(fig)


def fig_profiles(P):
    t = P["times"]; y = P["target"]
    hour = np.array([x.hour for x in t]); dow = np.array([x.weekday() for x in t])
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1))
    # hour-of-day
    hm = [y[hour == h] for h in range(24)]
    means = [v.mean() for v in hm]; sd = [v.std() for v in hm]
    axes[0].fill_between(range(24), np.array(means) - np.array(sd),
                         np.array(means) + np.array(sd), color=OK["sky"], alpha=0.3)
    axes[0].plot(range(24), means, color=OK["blue"], marker="o", ms=2.5, lw=1)
    axes[0].set_xlabel("Hour of day"); axes[0].set_ylabel("Active power (kW)")
    axes[0].set_title("Daily profile (mean $\\pm$ sd)")
    axes[0].set_xticks([0, 6, 12, 18, 23])
    # day-of-week
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dm = [y[dow == d] for d in range(7)]
    axes[1].boxplot(dm, tick_labels=labels, showfliers=False,
                    medianprops=dict(color=PROP), widths=0.6)
    axes[1].set_title("Load by day of week"); axes[1].set_ylabel("kW")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_profiles.pdf")); plt.close(fig)


def fig_decomposition(P):
    from statsmodels.tsa.seasonal import STL
    t = P["times"]; y = P["target"]; segs = P["segs"]
    s, e = max(segs, key=lambda se: se[1] - se[0])
    seg = y[s:e]; tt = t[s:e]
    stl = STL(seg, period=24, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(5.5, 4.2), sharex=True)
    for ax, series, name, c in [
        (axes[0], seg, "Observed", OK["black"]),
        (axes[1], stl.trend, "Trend", OK["blue"]),
        (axes[2], stl.seasonal, "Seasonal (24 h)", OK["green"]),
        (axes[3], stl.resid, "Residual", OK["orange"])]:
        ax.plot(tt, series, color=c, lw=0.9)
        ax.set_ylabel(name)
    axes[0].set_title("STL decomposition of the largest gap-free segment")
    fig.autofmt_xdate(rotation=20)
    fig.savefig(os.path.join(FIG, "fig_decomposition.pdf")); plt.close(fig)


def fig_forecast_1h(pred):
    yt = pred["1h__y_true"].ravel()
    times = [np.datetime64(x) for x in pred["1h__origin_time"]]
    fig, ax = plt.subplots(figsize=(5.5, 2.3))
    ax.plot(times, yt, color=OK["black"], lw=1.4, label="Actual", marker="o", ms=2)
    for key, c, ls in [("HoltWinters", OK["blue"], "--"),
                       ("XGBoost", OK["green"], "-."),
                       ("ApartElectricFormer", PROP, "-")]:
        ax.plot(times, pred[f"1h__{key}"].ravel(), color=c, lw=1.2, ls=ls,
                label=disp(key), alpha=0.9)
    ax.set_ylabel("Active power (kW)"); ax.set_title("One-hour-ahead forecasts on the test block")
    ax.legend(ncol=2, loc="upper right")
    fig.autofmt_xdate(rotation=20)
    fig.savefig(os.path.join(FIG, "fig_forecast_1h.pdf")); plt.close(fig)


def fig_forecast_day(pred):
    # day__* arrays: (n_windows, 24). Show 3 example origins.
    yt = pred["day__y_true"]
    n = yt.shape[0]
    idxs = [0, n // 2, n - 1]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0), sharey=True)
    for ax, i in zip(axes, idxs):
        h = np.arange(1, 25)
        ax.plot(h, yt[i], color=OK["black"], lw=1.4, label="Actual", marker="o", ms=2)
        ax.plot(h, pred["day__Transformer"][i], color=OK["green"], ls="-.", lw=1.1, label="Transformer")
        ax.plot(h, pred["day__ApartElectricFormer"][i], color=PROP, lw=1.2, label="CLEAR-Former")
        ax.set_xlabel("Horizon (h)")
        ax.set_title(str(pred["day__origin_time"][i])[5:16], fontsize=7)
    axes[0].set_ylabel("kW"); axes[0].legend(loc="best", fontsize=6.5)
    fig.suptitle("Day-ahead (24 h) forecast examples", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_forecast_day.pdf")); plt.close(fig)


def fig_mae_bars(metrics):
    order_1h = sorted(metrics["1h"], key=lambda k: metrics["1h"][k]["MAE"])
    models = [m for m in order_1h if m != "SeasonalNaive"]
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    x = np.arange(len(models)); w = 0.4
    v1 = [metrics["1h"][m]["MAE"] for m in models]
    vd = [metrics["day"][m]["MAE"] for m in models]
    b1 = ax.bar(x - w/2, v1, w, label="1 h-ahead", color=OK["sky"])
    b2 = ax.bar(x + w/2, vd, w, label="Day-ahead", color=OK["orange"])
    # highlight proposed
    for bars in (b1, b2):
        for i, m in enumerate(models):
            if m == "ApartElectricFormer":
                bars[i].set_edgecolor(PROP); bars[i].set_linewidth(1.8)
    ax.set_xticks(x); ax.set_xticklabels([disp(m) for m in models], rotation=35, ha="right")
    ax.set_ylabel("MAE (kW)"); ax.set_title("Test MAE by model and horizon")
    ax.legend()
    fig.savefig(os.path.join(FIG, "fig_mae_bars.pdf")); plt.close(fig)


def fig_ablation():
    p = os.path.join(RES, "ablation.json")
    if not os.path.exists(p):
        print("  (ablation.json not found, skipping)")
        return
    ab = json.load(open(p))
    variants = list(ab["1h"].keys())
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    x = np.arange(len(variants)); w = 0.4
    v1 = [ab["1h"][v]["MAE"] for v in variants]
    vd = [ab["day"][v]["MAE"] for v in variants]
    ax.bar(x - w/2, v1, w, label="1 h-ahead", color=OK["sky"])
    ax.bar(x + w/2, vd, w, label="Day-ahead", color=OK["orange"])
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_ylabel("MAE (kW)"); ax.set_title("CLEAR-Former ablation")
    ax.legend()
    fig.savefig(os.path.join(FIG, "fig_ablation.pdf")); plt.close(fig)


def main():
    P, pred, metrics = _load()
    fig_series(P); print("  fig_series")
    fig_imputation(); print("  fig_imputation")
    fig_profiles(P); print("  fig_profiles")
    fig_decomposition(P); print("  fig_decomposition")
    fig_forecast_1h(pred); print("  fig_forecast_1h")
    fig_forecast_day(pred); print("  fig_forecast_day")
    fig_mae_bars(metrics); print("  fig_mae_bars")
    fig_ablation(); print("  fig_ablation")
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
