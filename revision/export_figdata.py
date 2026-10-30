"""
Plain CSV exports from which the authors can draw the revised figures
(no figure is drawn here).  Output: figdata/*.csv
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

import common as C
import impute as IMP

OUT = os.path.join(C.HERE, "figdata")
os.makedirs(OUT, exist_ok=True)
ORDER = ["SeasonalNaive", "SARIMA", "HoltWinters", "Prophet", "SVR", "RandomForest", "XGBoost",
         "MLP", "LSTM", "Transformer", "CLEAR-Former"]
NEURAL = ("MLP", "LSTM", "Transformer", "CLEAR-Former")


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    P = C.reconstruct()
    old = IMP.prepare_continuous()
    runs = C.missing_runs(P["is_imputed"])
    seg_of = np.full(C.N_GRID, "", dtype=object)
    k = 0
    for i in range(C.N_GRID):
        if P["is_imputed"][i]:
            seg_of[i] = "gap_%dh" % next(b - a for a, b in runs if a <= i < b)
        else:
            if i > 0 and P["is_imputed"][i - 1]:
                k += 1
            seg_of[i] = "segment_" + "ABCD"[k]
    rows = []
    for i in range(C.N_GRID):
        rows.append([str(P["times"][i]), i, "" if P["is_imputed"][i] else f"{P['target_raw'][i]:.0f}",
                     f"{P['target'][i]:.3f}", f"{old['target'][i]:.3f}", int(P["is_imputed"][i]), seg_of[i],
                     int(i >= C.TEST_START)])
    write("grid_series.csv", ["time", "grid_index", "observed_kW", "revised_series_kW", "submitted_series_kW",
                              "is_reconstructed", "segment_or_gap", "in_test_block"], rows)

    S = json.load(open(os.path.join(C.RESULTS, "summary.json")))
    rows = []
    for key, w in S["windows"].items():
        task, scheme = key.split("|")
        for part in ("train", "val", "test"):
            c = w[part]
            rows.append([task, scheme, part, c["n"], C.grid_time(P, c["first_origin"]), C.grid_time(P, c["last_origin"]),
                         c["fully_observed"], c["n"] - c["fully_observed"], c["observed_target_hours"]])
        rows.append([task, scheme, "purged", w["n_purged"], "", "", "", "", ""])
    write("protocol_windows.csv", ["task", "scheme", "part", "n_windows", "first_origin", "last_origin",
                                   "fully_observed", "with_reconstructed_hour", "observed_target_hours"], rows)

    z = np.load(os.path.join(C.RESULTS, "bench_predictions.npz"))
    B = {k.replace("__", "|"): z[k] for k in z.files}

    def pred(task, m):
        return B[f"{task}|{m}|selected"] if m in NEURAL else B[f"{task}|{m}"]
    o1 = B["1h|origins"]
    write("test_predictions_1h.csv", ["origin", "target_time", "observed_kW"] + ORDER,
          [[C.grid_time(P, int(o)), C.grid_time(P, int(o) + 1), f"{B['1h|y_true'][i, 0]:.0f}"] +
           [f"{pred('1h', m)[i, 0]:.3f}" for m in ORDER] for i, o in enumerate(o1)])
    od = B["day|origins"]
    rows = []
    for i, o in enumerate(od):
        for h in range(24):
            rows.append([C.grid_time(P, int(o)), h + 1, C.grid_time(P, int(o) + 1 + h), f"{B['day|y_true'][i, h]:.0f}"] +
                        [f"{pred('day', m)[i, h]:.3f}" for m in ORDER])
    write("test_predictions_24h.csv", ["origin", "lead_h", "target_time", "observed_kW"] + ORDER, rows)
    po = S["origins"]["day"]["per_origin_MAE"]
    write("per_origin_mae_24h.csv", ["origin"] + ORDER,
          [[C.grid_time(P, int(o))] + [f"{po[m][i]:.3f}" for m in ORDER] for i, o in enumerate(od)])

    R = json.load(open(os.path.join(C.RESULTS, "recon_validation.json")))
    rows = []
    for G in R["lengths"]:
        for m in R["methods"]:
            s = R["target"][str(G)]["summary"][m]
            rows.append([G, m, s["n_placements"], f"{s['mae_mean']:.3f}", f"{s['mae_median']:.3f}", f"{s['mae_q25']:.3f}",
                         f"{s['mae_q75']:.3f}", f"{s['rmse_mean']:.3f}", f"{s['bias_mean']:.3f}"])
    write("recon_validation.csv", ["gap_h", "method", "n_placements", "mae_mean_kW", "mae_median_kW", "mae_q25_kW",
                                   "mae_q75_kW", "rmse_mean_kW", "bias_mean_kW"], rows)
    print("figdata ->", OUT, sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
