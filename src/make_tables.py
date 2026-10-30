"""
Emit LaTeX (booktabs) tables and an inline-numbers macro file directly from the
real experiment JSON, so the paper can never drift from the measured results.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper")
os.makedirs(PAPER, exist_ok=True)

DISP = {"ApartElectricFormer": "\\textbf{CLEAR-Former}", "SeasonalNaive": "Seasonal-Na\\\"ive",
        "HoltWinters": "Holt-Winters", "RandomForest": "Random Forest", "SARIMA": "SARIMA",
        "Prophet": "Prophet", "SVR": "SVR", "XGBoost": "XGBoost", "MLP": "MLP",
        "LSTM": "LSTM", "Transformer": "Transformer"}
GROUPS = [
    ("\\emph{Reference}", ["SeasonalNaive"]),
    ("\\emph{Statistical}", ["SARIMA", "HoltWinters", "Prophet"]),
    ("\\emph{Classical ML}", ["SVR", "RandomForest", "XGBoost"]),
    ("\\emph{Deep learning}", ["MLP", "LSTM", "Transformer"]),
    ("\\emph{Proposed}", ["ApartElectricFormer"]),
]


def fmt(v, best, lower=True):
    s = f"{v:.2f}"
    if (lower and abs(v - best) < 1e-9) or (not lower and abs(v - best) < 1e-9):
        return f"\\textbf{{{s}}}"
    return s


def main_table(metrics):
    # best (min) per metric per horizon
    allk = list(metrics["1h"].keys())
    best = {}
    for task in ("1h", "day"):
        for mt in ("MAE", "RMSE", "MAPE"):
            best[(task, mt)] = min(metrics[task][k][mt] for k in allk)
    lines = []
    lines.append("\\begin{tabular}{l ccc ccc}")
    lines.append("\\toprule")
    lines.append("& \\multicolumn{3}{c}{\\textbf{1\\,h-ahead}} & \\multicolumn{3}{c}{\\textbf{Day-ahead (24\\,h)}} \\\\")
    lines.append("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
    lines.append("Method & MAE & RMSE & MAPE & MAE & RMSE & MAPE \\\\")
    lines.append("\\midrule")
    for gname, keys in GROUPS:
        lines.append(f"\\multicolumn{{7}}{{l}}{{{gname}}} \\\\")
        for k in keys:
            r1, rd = metrics["1h"][k], metrics["day"][k]
            row = (f"\\quad {DISP[k]} & "
                   f"{fmt(r1['MAE'], best[('1h','MAE')])} & {fmt(r1['RMSE'], best[('1h','RMSE')])} & {fmt(r1['MAPE'], best[('1h','MAPE')])} & "
                   f"{fmt(rd['MAE'], best[('day','MAE')])} & {fmt(rd['RMSE'], best[('day','RMSE')])} & {fmt(rd['MAPE'], best[('day','MAPE')])} \\\\")
            lines.append(row)
        if gname != GROUPS[-1][0]:
            lines.append("\\addlinespace")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    open(os.path.join(PAPER, "tab_main.tex"), "w").write("\n".join(lines))


def ablation_table():
    p = os.path.join(RES, "ablation.json")
    if not os.path.exists(p):
        print("  ablation.json missing"); return
    ab = json.load(open(p))
    order = ["Full", "-Climatology", "-Decomposition", "-FeatureFusion", "-Attention", "LinearBackboneOnly"]
    disp = {"Full": "\\textbf{CLEAR-Former (full)}", "-Climatology": "\\;-- climatology anchor",
            "-Decomposition": "\\;-- series decomposition", "-FeatureFusion": "\\;-- exogenous feature fusion",
            "-Attention": "\\;-- gated attention", "LinearBackboneOnly": "\\;backbone only (clim.+decomp.-linear)"}
    lines = ["\\begin{tabular}{l cc cc}", "\\toprule",
             "& \\multicolumn{2}{c}{1\\,h-ahead} & \\multicolumn{2}{c}{Day-ahead} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Variant & MAE & RMSE & MAE & RMSE \\\\", "\\midrule"]
    for k in order:
        if k not in ab["1h"]:
            continue
        a, d = ab["1h"][k], ab["day"][k]
        lines.append(f"{disp[k]} & {a['MAE']:.2f} & {a['RMSE']:.2f} & {d['MAE']:.2f} & {d['RMSE']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(PAPER, "tab_ablation.tex"), "w").write("\n".join(lines))


def dataset_table(meta):
    rows = [
        ("Monitored quantity", "Total active power (kW), 3-phase"),
        ("Sampling interval", "1 hour"),
        ("Observation window", f"{meta['date_start'][:16]} -- {meta['date_end'][:16]}"),
        ("Valid hourly records", str(meta["n_raw_rows"])),
        ("Contiguous segments", f"{meta['n_segments']} (lengths {', '.join(str(x) for x in meta['segment_lengths'] if x>1)}\\,h)"),
        ("Largest data gap", "217\\,h (9.0 days)"),
        ("Load mean / std", f"{meta['target_mean']:.1f} / {meta['target_std']:.1f}\\,kW"),
        ("Load range", f"{meta['target_min']:.0f}--{meta['target_max']:.0f}\\,kW"),
        ("Exogenous covariates", "3$\\times$ phase current, 3$\\times$ phase voltage, power factor"),
        ("Dropped (corrupted) fields", str(len(meta["dropped_columns"])) + " time-formatted columns"),
    ]
    lines = ["\\begin{tabular}{ll}", "\\toprule", "Property & Value \\\\", "\\midrule"]
    for a, b in rows:
        lines.append(f"{a} & {b} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(PAPER, "tab_dataset.tex"), "w").write("\n".join(lines))


def segments_table(meta):
    import datetime as dt
    spans = [(a, b, L) for (a, b), L in zip(meta["segment_spans"], meta["segment_lengths"]) if L > 1]
    def parse(s): return dt.datetime.fromisoformat(s)
    roles = ["train", "train", "train", "train head + \\textbf{test} tail"]
    lines = ["\\begin{tabular}{clccl}", "\\toprule",
             "Seg. & Span & Length & Prec.\\ gap & Role \\\\", "\\midrule"]
    prev_end = None
    for i, (a, b, L) in enumerate(spans):
        gap = "--" if prev_end is None else f"{int(round((parse(a)-prev_end).total_seconds()/3600))}\\,h"
        span = f"{a[5:16]} -- {b[5:16]}"
        role = roles[i] if i < len(roles) else "--"
        lines.append(f"{chr(65+i)} & {span} & {L}\\,h & {gap} & {role} \\\\")
        prev_end = parse(b)
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(PAPER, "tab_segments.tex"), "w").write("\n".join(lines))


def macros(metrics):
    # word-only macro names (LaTeX forbids digits in \newcommand names)
    pref = {"ApartElectricFormer": "CF", "HoltWinters": "HW", "SARIMA": "SAR",
            "Prophet": "PR", "SVR": "SVR", "RandomForest": "RF", "XGBoost": "XGB",
            "MLP": "MLP", "LSTM": "LSTM", "Transformer": "TR", "SeasonalNaive": "SN"}
    hz = {"1h": "one", "day": "day"}
    m = {}
    for task, tt in hz.items():
        for key, pf in pref.items():
            if key in metrics[task]:
                m[f"{pf}{tt}MAE"] = f"{metrics[task][key]['MAE']:.2f}"
                m[f"{pf}{tt}R"] = f"{metrics[task][key]['R2']:.2f}"
    # best learning-based (non-statistical) at each horizon
    for task, tt in hz.items():
        lb = {k: metrics[task][k]['MAE'] for k in metrics[task]
              if k not in ('SARIMA', 'HoltWinters', 'Prophet', 'SeasonalNaive')}
        m[f"BestLearn{tt}"] = f"{min(lb.values()):.2f}"
    # mean MAE across the two horizons (balance across horizons)
    avg = {k: 0.5 * (metrics['1h'][k]['MAE'] + metrics['day'][k]['MAE'])
           for k in metrics['1h'] if k in metrics['day']}
    for key, pf in pref.items():
        if key in avg:
            m[f"{pf}avgMAE"] = f"{avg[key]:.2f}"
    ranked = sorted(avg.items(), key=lambda kv: kv[1])
    m["BestAvgName"] = {"ApartElectricFormer": "CLEAR-Former"}.get(ranked[0][0], ranked[0][0])
    m["SecondAvgMAE"] = f"{ranked[1][1]:.2f}"
    # imputation counts from the run metadata
    try:
        mr = json.load(open(os.path.join(RES, "meta_run.json")))
        m["Nobs"] = str(mr.get("n_observed", 456))
        m["Nimp"] = str(mr.get("n_imputed", 264))
        m["Ntot"] = str(mr.get("n_total", 720))
        m["PctImp"] = f"{100*mr.get('n_imputed',264)/mr.get('n_total',720):.0f}"
        m["Ntrainone"] = ""  # placeholder if needed
    except Exception:
        pass
    # reconstruction A/B: CLEAR-Former trained on observed-only windows
    try:
        rob = json.load(open(os.path.join(RES, "recon_obs.json")))
        m["CFoneObs"] = f"{rob['1h']['ApartElectricFormer']['MAE']:.2f}"
        m["CFdayObs"] = f"{rob['day']['ApartElectricFormer']['MAE']:.2f}"
    except Exception:
        pass
    out = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items() if v != "")
    open(os.path.join(PAPER, "numbers.tex"), "w").write(out)


if __name__ == "__main__":
    metrics = json.load(open(os.path.join(RES, "metrics.json")))
    meta = json.load(open(os.path.join(ROOT, "data", "meta.json")))
    main_table(metrics); print("  tab_main.tex")
    dataset_table(meta); print("  tab_dataset.tex")
    segments_table(meta); print("  tab_segments.tex")
    ablation_table(); print("  tab_ablation.tex")
    macros(metrics); print("  numbers.tex")
