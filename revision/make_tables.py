"""
LaTeX tables (LNCS, \\hline style, no extra packages) generated from the
result files, so every number in them traces to results/.  Captions are
factual drafts for the authors to rewrite.  Output: tables/*.tex
"""
from __future__ import annotations

import json
import os

import common as C

OUT = os.path.join(C.HERE, "tables")
os.makedirs(OUT, exist_ok=True)
DISP = {"SeasonalNaive": "Seasonal-naive", "SARIMA": "SARIMA", "HoltWinters": "Holt--Winters", "Prophet": "Prophet",
        "SVR": "SVR", "RandomForest": "Random Forest", "XGBoost": "XGBoost", "MLP": "MLP", "LSTM": "LSTM",
        "Transformer": "Transformer", "CLEAR-Former": "CLEAR-Former"}
ORDER = list(DISP)


def j(name):
    return json.load(open(os.path.join(C.RESULTS, name)))


def write(name, lines):
    import re
    # true minus sign for negative numbers in table rows
    body = [re.sub(r"(?<![\w$])-(\d)", r"$-$\1", ln) if "&" in ln else ln for ln in lines]
    open(os.path.join(OUT, name), "w", encoding="utf-8").write("\n".join(body) + "\n")


def f2(x):
    return f"{x:.2f}"


def table_recon(R):
    meth = ["LOCF", "Linear", "HourMean", "PrevDay", "Kalman", "Ours"]
    head = ["LOCF", "Linear", "Hour mean", "Prev.\\ day", "Kalman", "Ours"]
    L = ["\\begin{table}[t]",
         "\\caption{Reconstruction error on artificial gaps cut into observed pre-test hours. MAE in kW, "
         "mean over gap placements (starts every 6 h); $n$ is the number of placements. Bold: lowest per row.}",
         "\\label{tab:recon}", "\\centering", "\\small", "\\begin{tabular}{rr" + "r" * len(meth) + "}", "\\hline",
         "Gap (h) & $n$ & " + " & ".join(head) + " \\\\", "\\hline"]
    for G in R["lengths"]:
        s = R["target"][str(G)]["summary"]
        best = min(meth, key=lambda m: s[m]["mae_mean"])
        cells = [("\\textbf{%s}" if m == best else "%s") % f2(s[m]["mae_mean"]) for m in meth]
        L.append(f"{G} & {s['Ours']['n_placements']} & " + " & ".join(cells) + " \\\\")
    L += ["\\hline", "\\end{tabular}", "\\end{table}"]
    write("tab_recon_validation.tex", L)


def table_main(S):
    m1, md = S["metrics"]["1h"], S["metrics"]["day"]
    b = {k: min(S["metrics"][t][m][k] for m in ORDER) for t in ("1h", "day") for k in ("MAE", "RMSE")}
    best = {(t, k): min(S["metrics"][t][m][k] for m in ORDER) for t in ("1h", "day") for k in ("MAE", "RMSE")}
    bR2 = {t: max(S["metrics"][t][m]["R2"] for m in ORDER) for t in ("1h", "day")}

    def cell(t, m, k):
        v = S["metrics"][t][m][k]
        isb = abs(v - (bR2[t] if k == "R2" else best[(t, k)])) < 1e-12
        s = f"{v:.3f}" if k == "R2" else f2(v)
        return f"\\textbf{{{s}}}" if isb else s

    L = ["\\begin{table}[t]",
         "\\caption{Test-block error (all inputs and targets observed). MAE and RMSE in kW. Neural models: "
         "5-seed ensemble of the validation-selected configuration. Origin MAE: median [min, max] over the 25 "
         "day-ahead origins. Bold: best per column.}",
         "\\label{tab:main}", "\\centering", "\\small", "\\begin{tabular}{lrrrrrrl}", "\\hline",
         "& \\multicolumn{3}{c}{1 h ahead} & \\multicolumn{4}{c}{24 h ahead} \\\\",
         "Method & MAE & RMSE & $R^2$ & MAE & RMSE & $R^2$ & Origin MAE \\\\", "\\hline"]
    for m in ORDER:
        r = md[m]
        L.append(f"{DISP[m]} & {cell('1h', m, 'MAE')} & {cell('1h', m, 'RMSE')} & {cell('1h', m, 'R2')} & "
                 f"{cell('day', m, 'MAE')} & {cell('day', m, 'RMSE')} & {cell('day', m, 'R2')} & "
                 f"{f2(r['origin_MAE_median'])} [{f2(r['origin_MAE_min'])}, {f2(r['origin_MAE_max'])}] \\\\")
    L += ["\\hline", "\\end{tabular}", "\\end{table}"]
    write("tab_main.tex", L)


def table_splits(S):
    """Windows per horizon, computed from the same split as the experiments (common.split)."""
    import common as C
    P = C.reconstruct()
    t0 = P["times"][0]

    def when(i):
        import datetime as dt
        return (t0 + dt.timedelta(hours=int(i))).strftime("%d %b %H:%M").lstrip("0")

    L = ["\\begin{table}[t]",
         "\\caption{Windows per horizon (24 h input, 1 h stride, split by origin). $n$: windows. Target hours: "
         "first and last target of the windows in each part (2026). Full: every input and target hour observed. "
         "Recon.: at least one reconstructed hour in the inputs or targets. Scored pairs: (window, target hour) pairs that enter the validation or test "
         "score; validation scores only the 26 observed hours from 26 Jun 18:00 at both horizons. Purged: "
         "training windows whose targets reach the first validation target. Excluded: windows whose targets "
         "cross the test start.}",
         "\\label{tab:splits}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
         "\\begin{tabular}{lrlrrrr}", "\\hline",
         "& & & & & \\multicolumn{2}{c}{Scored} \\\\",
         "Part & $n$ & Target hours & Full & Recon. & windows & pairs \\\\", "\\hline"]
    for H, lab in ((1, "1 h ahead"), (24, "24 h ahead")):
        ws = C.make_windows(P, H)
        sp = C.split(P, ws, H)
        pool_origins = {w["origin"] for w in sp["train"] + sp["val"]}
        head = [w for w in ws if w["tgt_max"] < C.TEST_START and w["origin"] < sp["val"][0]["origin"]]
        purged = [w for w in head if w["origin"] not in pool_origins]
        excluded = [w for w in ws if w["tgt_max"] >= C.TEST_START and w["origin"] + 1 < C.TEST_START]
        L.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{lab}}}}} \\\\")

        def row(name, part, scored=True):
            lo = min(w["origin"] + 1 for w in part)
            hi = max(w["tgt_max"] for w in part)
            full = sum(1 for w in part if w["in_obs"].all() and w["tg_obs"].all())
            sw = [w for w in part if w["tg_obs"].any()] if scored else []
            pairs = sum(int(w["tg_obs"].sum()) for w in sw)
            s_w, s_p = (str(len(sw)), str(pairs)) if scored else ("--", "--")
            return (f"{name} & {len(part)} & {span(lo, hi)} & {full} & {len(part) - full} & "
                    f"{s_w} & {s_p} \\\\")

        def span(lo, hi):
            return when(lo) + "\\textendash{}" + when(hi)
        L.append(row("training", sp["train"], scored=False))
        if purged:
            L.append(f"purged & {len(purged)} & "
                     f"{span(min(w['origin'] + 1 for w in purged), max(w['tgt_max'] for w in purged))} & & & & \\\\")
        L.append(row("validation", sp["val"]))
        if excluded:
            L.append(f"excluded & {len(excluded)} & "
                     f"{span(min(w['origin'] + 1 for w in excluded), max(w['tgt_max'] for w in excluded))} & & & & \\\\")
        L.append(row("test", sp["test"]))
    L += ["\\hline", "\\end{tabular}", "\\end{table}"]
    write("tab_splits.tex", L)


def table_params(S):
    p1 = S["params"]["1h|CLEAR-Former|selected"]
    pd = S["params"]["day|CLEAR-Former|selected"]
    names = {"hour_profile": "Climatology: hour profile $a$", "dow_profile": "Climatology: weekday profile $b$",
             "lin_trend": "Linear map, trend $W_{\\mathrm{tr}}$", "lin_seasonal": "Linear map, seasonal $W_{\\mathrm{se}}$",
             "feat_mlp": "Feature MLP $f_z$", "patch_embed": "Attention: patch embedding",
             "pos": "Attention: positional embedding", "hour_emb": "Attention: hour embedding",
             "dow_emb": "Attention: weekday embedding", "cov_gru": "Attention: covariate GRU",
             "cov_proj": "Attention: covariate projection", "gate": "Attention: token/covariate gate",
             "encoder": "Attention: encoder layer(s)", "norm": "Attention: final LayerNorm",
             "corr_head": "Attention: output head", "alpha": "Gate $\\alpha$"}
    def cfg_txt(c):
        n = {1: "one", 2: "two", 4: "four"}
        return (f"$d = {c['d_model']}$, {n[c['heads']]} head{'s' if c['heads'] > 1 else ''}, "
                f"{n[c['layers']]} encoder layer{'s' if c['layers'] > 1 else ''}, ff = {c['ff']}")
    L = ["\\begin{table}[t]",
         "\\caption{CLEAR-Former parameters per component for the validation-selected configurations "
         f"(1 h: {cfg_txt(p1['config'])}; 24 h: {cfg_txt(pd['config'])}).}}",
         "\\label{tab:params}", "\\centering", "\\small", "\\begin{tabular}{lrr}", "\\hline",
         "Component & 1 h & 24 h \\\\", "\\hline"]
    for k, lab in names.items():
        L.append(f"{lab} & {p1['components'].get(k, 0):,} & {pd['components'].get(k, 0):,} \\\\".replace(",", "{,}"))
    L += ["\\hline", f"Total & {p1['total']:,} & {pd['total']:,} \\\\".replace(",", "{,}"), "\\hline",
          "\\end{tabular}", "\\end{table}"]
    write("tab_params.tex", L)


def table_followup(F):
    import numpy as np

    def row(tag):
        r = F[tag]
        return r["MAE"], float(np.std(r["seed_MAE"], ddof=1)), r["R2"], float(np.mean(r["val_mae_masked_seeds"]))
    labels = {"Full": "Full model", "-Climatology": "$-$ climatology", "-Decomposition": "$-$ decomposition",
              "-Exogenous MLP": "$-$ feature MLP $f_z$", "-Attention branch": "$-$ attention branch $f_a$",
              "Linear backbone only": "Climatology + linear only",
              "-Covariate gate (in attention)": "$-$ covariate gate in $f_a$",
              "-Hour/day embeddings (in attention)": "$-$ hour/day embeddings in $f_a$"}
    L = ["\\begin{table}[t]",
         "\\caption{CLEAR-Former ablation (validation-selected configuration; exploratory, after selection). "
         "Test MAE in kW of the 5-seed ensemble $\\pm$ sample SD of the five single-seed MAEs; Val: mean "
         "single-seed validation MAE on observed hours.}",
         "\\label{tab:ablation}", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
         "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrrrrr}", "\\hline",
         "Variant & Val 1 h & Test 1 h & Val 24 h & Test 24 h & $R^2$ 24 h \\\\", "\\hline"]
    for v, lab in labels.items():
        a, sa, _, va = row(f"ablation|1h|{v}")
        b, sb, rb, vb = row(f"ablation|day|{v}")
        L.append(f"{lab} & {f2(va)} & {f2(a)} $\\pm$ {f2(sa)} & {f2(vb)} & {f2(b)} $\\pm$ {f2(sb)} & {rb:.3f} \\\\")
    L += ["\\hline", "\\end{tabular}}", "\\end{table}"]
    write("tab_ablation.tex", L)

    arms = [("observed", "No reconstructed hour anywhere"), ("short_fills", "Short fills only (47 h and 1 h)"),
            ("recon_all", "All fills, validation on every target"), ("recon_obs", "All fills, validation on observed hours")]
    L = ["\\begin{table}[t]",
         "\\caption{Where the reconstruction enters (CLEAR-Former, validation-selected configuration). "
         "MAE in kW on the observed test block; $n$: training windows.}",
         "\\label{tab:arms}", "\\centering", "\\small", "\\begin{tabular}{lrrrrr}", "\\hline",
         "Training data and validation & $n$ (1 h) & 1 h MAE & $n$ (24 h) & 24 h MAE & 24 h $R^2$ \\\\", "\\hline"]
    for a, lab in arms:
        t1 = f"arm|1h|{a}" if a != "recon_obs" else "ablation|1h|Full"
        td = f"arm|day|{a}" if a != "recon_obs" else "ablation|day|Full"
        L.append(f"{lab} & {F[t1]['n_train']} & {f2(F[t1]['MAE'])} & {F[td]['n_train']} & {f2(F[td]['MAE'])} & "
                 f"{F[td]['R2']:.3f} \\\\")
    L += ["\\hline", "\\end{tabular}", "\\end{table}"]
    write("tab_arms.tex", L)


def table_sensitivity(Z):
    designs = Z["designs"]
    head = {"recon_obs": "Main", "recon_all": "All targets", "recon_blockval": "Block val.",
            "short_fills": "Short fills", "observed": "No fills"}
    lab = {"MLP": "MLP", "LSTM": "LSTM", "Transformer": "Transformer", "CLEAR-Former": "CLEAR-Former",
           "CLEAR-Former Linear backbone only": "\\quad climatology + linear only",
           "CLEAR-Former -Exogenous MLP": "\\quad $-$ feature MLP $f_z$",
           "CLEAR-Former -Attention branch": "\\quad $-$ attention branch $f_a$"}
    for t, tl in (("1h", "1 h"), ("day", "24 h")):
        L = ["\\begin{table}[t]",
             f"\\caption{{Sensitivity of the {tl} test MAE (kW) to the training data and the validation design "
             "(post hoc). Main: all fills, validation on observed hours; All targets: all fills, validation on every "
             "target; Block val.: all fills, validation on fully observed windows before the 216 h gap; Short fills: "
             "no window touching the 216 h fill; No fills: fully observed windows only. $\\rho$: Spearman correlation "
             "of validation and test MAE over the seven rows.}",
             f"\\label{{tab:sens{t}}}", "\\centering", "\\small", "\\begin{tabular}{lrrrrr}", "\\hline",
             "Model & " + " & ".join(head[d] for d in designs) + " \\\\", "\\hline"]
        for m in Z["models"]:
            row = [Z["table"][f"{t}|{m}|{d}"]["test_MAE"] for d in designs]
            L.append(f"{lab[m]} & " + " & ".join(f2(v) for v in row) + " \\\\")
        L += ["\\hline", "$\\rho$ (validation, test) & " +
              " & ".join(f"{Z['rank_corr'][f'{t}|{d}']['spearman_val_test_7']:.2f}" for d in designs) + " \\\\",
              "\\hline", "\\end{tabular}", "\\end{table}"]
        write(f"tab_sensitivity_{t}.tex", L)


def main():
    R = j("recon_validation.json")
    if os.path.exists(os.path.join(C.RESULTS, "sensitivity_summary.json")):
        table_sensitivity(j("sensitivity_summary.json"))
    table_recon(R)
    if os.path.exists(os.path.join(C.RESULTS, "summary.json")):
        S = j("summary.json")
        table_main(S)
        table_splits(S)
        table_params(S)
    if os.path.exists(os.path.join(C.RESULTS, "followup.json")):
        table_followup(j("followup.json"))
    print("tables ->", OUT, sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
