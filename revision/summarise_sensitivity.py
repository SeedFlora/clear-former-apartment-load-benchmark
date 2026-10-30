"""Collect the post hoc sensitivity analysis into results/sensitivity_summary.json.

Rows: the four neural models and three CLEAR-Former variants; columns: the five designs.
* recon_obs (main protocol): neural models from summary.json (test) and grid_val.json (mean single-seed
  validation MAE of the selected configuration); CLEAR-Former variants from followup.json (ablation);
* recon_all, short_fills, observed for the full CLEAR-Former: followup.json (reconstruction arms);
* every other cell: sensitivity.json (run_sensitivity.py).
Also the validation-test Spearman correlation over the seven rows, the best neural model per design and
the effect of the fills at matched validation (observed -> recon_blockval).

    python summarise_sensitivity.py      # after analysis.py, run_followup.py and run_sensitivity.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr

import common as C
from val_diagnostics import check_and_write, j

DESIGNS = ["recon_obs", "recon_all", "recon_blockval", "short_fills", "observed"]
NEURAL = ["MLP", "LSTM", "Transformer", "CLEAR-Former"]
VARIANTS = {"CLEAR-Former Linear backbone only": "Linear backbone only",
            "CLEAR-Former -Exogenous MLP": "-Exogenous MLP",       # result key of the feature-MLP ablation
            "CLEAR-Former -Attention branch": "-Attention branch"}
MODELS = NEURAL + list(VARIANTS)


def cell(r):
    return {"test_MAE": r["MAE"], "val_MAE_seed_mean": float(np.mean(r["val_mae_masked_seeds"])), "R2": r["R2"]}


def main():
    S, F, SE, G, SEL = j("summary.json"), j("followup.json"), j("sensitivity.json"), j("grid_val.json"), \
        j("selection.json")["neural"]
    table = {}
    for task in ("1h", "day"):
        for m in MODELS:
            for d in DESIGNS:
                key = f"{task}|{m}|{d}"
                if d == "recon_obs" and m in NEURAL:
                    cid = SEL[f"{task}|{m}"]["selected_config_id"]
                    row = next(r for r in G[f"{task}|{m}"] if r["config_id"] == cid)
                    met = S["metrics"][task][m]
                    table[key] = {"test_MAE": met["MAE"], "val_MAE_seed_mean": float(np.mean(row["val_mae_masked_seeds"])),
                                  "R2": met["R2"]}
                elif d == "recon_obs":
                    table[key] = cell(F[f"ablation|{task}|{VARIANTS[m]}"])
                elif m == "CLEAR-Former" and d != "recon_blockval":
                    table[key] = cell(F[f"arm|{task}|{d}"])
                else:
                    table[key] = cell(SE[key])
    rank, winners = {}, {}
    for task in ("1h", "day"):
        for d in DESIGNS:
            v = [table[f"{task}|{m}|{d}"]["val_MAE_seed_mean"] for m in MODELS]
            t = [table[f"{task}|{m}|{d}"]["test_MAE"] for m in MODELS]
            r = spearmanr(v, t)
            rank[f"{task}|{d}"] = {"spearman_val_test_7": float(r.statistic), "p": float(r.pvalue)}
            winners[f"{task}|{d}"] = min(NEURAL, key=lambda m: table[f"{task}|{m}|{d}"]["test_MAE"])
    fill = {}
    for task in ("1h", "day"):
        for m in MODELS:
            o, b = table[f"{task}|{m}|observed"]["test_MAE"], table[f"{task}|{m}|recon_blockval"]["test_MAE"]
            fill[f"{task}|{m}"] = {"observed": o, "recon_blockval": b, "change_with_fills": b - o}
    out = {"designs": DESIGNS, "models": MODELS, "table": table, "rank_corr": rank, "winners": winners,
           "fill_effect_matched_validation": fill}
    check_and_write("sensitivity_summary.json", out)
    margins = []
    for d in DESIGNS:
        cf = table[f"1h|CLEAR-Former|{d}"]["test_MAE"]
        margins.append(min(table[f"1h|{m}|{d}"]["test_MAE"] for m in NEURAL if m != "CLEAR-Former") - cf)
    print("winners:", winners)
    print("one-hour margin of CLEAR-Former over the next neural model:", [round(x, 3) for x in margins])


if __name__ == "__main__":
    main()
