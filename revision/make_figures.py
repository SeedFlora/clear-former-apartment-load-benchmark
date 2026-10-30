"""Draw Fig. 1 (protocol timeline) and Fig. 3 (per-origin 24 h MAE) from figdata/.

Every plotted value and every number in a label is read from the CSV files that
export_figdata.py writes from results/, or computed from them (the window counts
of the purge and the exclusion are checked against protocol_windows.csv below).
Fig. 2 is a TikZ diagram in figures/fig2_architecture.tex; compile it before
running this script so that figures_manifest.json records its size.

    python make_figures.py          # writes figures/fig1_*.pdf, figures/fig3_*.pdf (+ PNG previews)
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDATA = os.path.join(HERE, "figdata")
OUT = os.path.join(HERE, "figures")

TEXT_W = 122 / 25.4          # LNCS text width, inches
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
RECON = "#eb6834"            # categorical slot 2: reconstructed hours, in every figure
RECON_TINT = "#fbe5dc"       # opaque ~17% tint of RECON (transparency drops out in halftone print)
RECON_HATCH = "#8a3a14"      # darker step of RECON: luma 78 against 122 (validation) and 174 (training)
SPLIT = {"train": "#86b6ef", "val": "#3987e5", "test": "#184f95"}   # blue ordinal ramp 250/400/600
# Fig. 3 skips slot 2 so that orange keeps one meaning (reconstructed) across the paper;
# slots 1, 3, 4, 5, 6 validated: worst adjacent CVD deltaE 9.1, normal-vision 19.6 on white
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
OTHER = "#c3c2b7"            # the six models Fig. 3 does not highlight

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5,
    "hatch.linewidth": 0.7, "pdf.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, linestyle="-")
        ax.set_axisbelow(True)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    pdf = os.path.join(OUT, name + ".pdf")
    fig.savefig(pdf, metadata={"CreationDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
    try:                                        # preview rendered from the PDF itself
        import fitz
        page = fitz.open(pdf)[0]
        page.get_pixmap(dpi=300).save(os.path.join(OUT, name + ".png"))
    except ImportError:
        pass
    return pdf


def _runs(flag):
    """(start, end) index pairs of consecutive True values."""
    idx = np.flatnonzero(flag)
    if idx.size == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cut + 1]]
    ends = np.r_[idx[cut], idx[-1]]
    return list(zip(starts, ends))


def fig1_protocol_timeline():
    g = pd.read_csv(os.path.join(FIGDATA, "grid_series.csv"), parse_dates=["time"])
    w = pd.read_csv(os.path.join(FIGDATA, "protocol_windows.csv"),
                    parse_dates=["first_origin", "last_origin"])
    w = w[w.scheme == "recon"].set_index(["task", "part"])
    t0 = g.time.iloc[0]
    hour = lambda ts: int(round((ts - t0) / pd.Timedelta(hours=1)))   # noqa: E731
    recon = g.is_reconstructed.to_numpy().astype(bool)
    obs = ~recon
    test_start = hour(g.time[g.in_test_block == 1].min())
    gaps = _runs(recon)

    # target-hour spans of each split (origin + 1 ... origin + H)
    spans = {}
    for task, H in (("1h", 1), ("day", 24)):
        for part in ("train", "val", "test"):
            r = w.loc[(task, part)]
            spans[(task, part)] = (hour(r.first_origin) + 1, hour(r.last_origin) + H)
    n_purged = int(w.loc[("day", "purged"), "n_windows"])
    # 24 h windows dropped because their targets straddle the test start
    first_test_origin_day = hour(w.loc[("day", "test"), "first_origin"])
    last_val_origin_day = hour(w.loc[("day", "val"), "last_origin"])
    excl = (last_val_origin_day + 1, first_test_origin_day - 1)
    n_excl = excl[1] - excl[0] + 1
    purge_origins = (hour(w.loc[("day", "train"), "last_origin"]) + 1,
                     hour(w.loc[("day", "val"), "first_origin"]) - 1)
    assert purge_origins[1] - purge_origins[0] + 1 == n_purged
    assert spans[("1h", "test")][0] == test_start == spans[("day", "test")][0]

    fig = plt.figure(figsize=(TEXT_W, 2.62))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.95, 1.1], hspace=0.10,
                          left=0.1, right=0.98, top=0.885, bottom=0.18)
    ax = fig.add_subplot(gs[0])
    bx = fig.add_subplot(gs[1], sharex=ax)
    x = np.arange(len(g))

    # (a) load record
    y_obs = np.where(obs, g.observed_kW, np.nan)
    y_rec = g.revised_series_kW.to_numpy().copy()
    keep = recon.copy()
    for s, e in gaps:                            # join the fill to its bounding observations
        keep[max(s - 1, 0)] = True
        keep[min(e + 1, len(g) - 1)] = True
    y_rec[~keep] = np.nan
    for s, e in gaps:
        ax.axvspan(s - 0.5, e + 0.5, color=RECON_TINT, linewidth=0, zorder=0)
    ax.plot(x, y_rec, color=RECON, linewidth=0.8, solid_joinstyle="round")
    ax.plot(x, y_obs, color=INK, linewidth=0.7, solid_joinstyle="round")
    ax.set_ylim(118, 266)
    ax.set_ylabel("Load (kW)")
    for s, e in gaps:
        n = e - s + 1
        xc = (s + e) / 2
        ax.text(xc, 253.5, f"{n} h gap", ha="center", va="baseline", fontsize=7, color=INK2)   # above the 250 gridline
        if n == 1:                                # too narrow to see: point at it
            ax.plot([xc, xc], [249, 240.5], color=INK2, linewidth=0.6)
    ax.axvline(test_start - 0.5, color=INK, linewidth=0.8)
    bx.plot([test_start - 0.5] * 2, [-0.28, 1.42], color=INK, linewidth=0.8, zorder=6)   # stops at the bar bottom
    first_test = g.time[test_start].strftime("%d %b %H:%M").lstrip("0")
    ax.annotate(f"first test target\n({first_test})", xy=(test_start - 0.5, 133), xytext=(test_start - 9, 133),
                ha="right", va="center", fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.5))
    _style(ax)
    ax.tick_params(axis="x", labelbottom=False)
    ax.text(-0.09, 1.0, "(a)", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    handles = [plt.Line2D([], [], color=INK, linewidth=1.2, label=f"observed ({int(obs.sum())} h)"),
               plt.Line2D([], [], color=RECON, linewidth=1.2, label=f"reconstructed ({int(recon.sum())} h)")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
              handlelength=1.6, borderaxespad=0.2, columnspacing=1.2)

    # (b) target hours per split
    rows = {"1h": 1.0, "day": 0.0}
    hgt = 0.56
    for task, yc in rows.items():
        for part in ("train", "val", "test"):
            a, b = spans[(task, part)]
            bx.add_patch(Rectangle((a - 0.5, yc - hgt / 2), b - a + 1, hgt, facecolor=SPLIT[part],
                                   edgecolor="none", zorder=2))
            for s, e in gaps:                     # reconstructed target hours in this span
                lo, hi = max(a, s), min(b, e)
                if lo <= hi:
                    bx.add_patch(Rectangle((lo - 0.5, yc - hgt / 2), hi - lo + 1, hgt, facecolor="none",
                                           edgecolor=RECON_HATCH, hatch="//////", linewidth=0, zorder=3))
        for part in ("val", "test"):              # 2 px surface gap at each boundary
            bx.plot([spans[(task, part)][0] - 0.5] * 2, [yc - hgt / 2, yc + hgt / 2], color=SURFACE,
                    linewidth=1.5, zorder=4, solid_capstyle="butt")
        bx.text(360, yc, "training", ha="center", va="center", fontsize=7, color=INK, zorder=5)
        a, b = spans[(task, "test")]
        bx.text((a + b) / 2, yc, "test", ha="center", va="center", fontsize=7, color=SURFACE, zorder=5)

    def bracket(lo, hi, y, label):
        bx.plot([lo - 0.5, lo - 0.5, hi + 0.5, hi + 0.5], [y + 0.07, y, y, y + 0.07], color=INK2,
                linewidth=0.6, clip_on=False, zorder=5)
        bx.text((lo + hi) / 2, y - 0.06, label, ha="center", va="top", fontsize=7, color=INK2,
                linespacing=1.0, clip_on=False)

    yb = rows["day"] - hgt / 2 - 0.10
    bracket(purge_origins[0] + 1, purge_origins[1] + 24, yb, f"{n_purged} purged\nwindows")
    bracket(excl[0] + 1, excl[1] + 24, yb, f"{n_excl} excluded\nwindows")
    bx.set_ylim(-1.28, 1.42)
    bx.set_yticks([rows["day"], rows["1h"]], ["24 h", "1 h"])
    bx.set_ylabel("Horizon", labelpad=4)
    bx.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        bx.spines[side].set_visible(False)
    bx.spines["bottom"].set_position(("data", -1.28))
    bx.text(-0.09, 1.02, "(b)", transform=bx.transAxes, fontsize=8, fontweight="bold", va="bottom")
    fig.align_ylabels([ax, bx])

    # calendar axis: Mondays labelled, days as minor ticks
    first_midnight = (24 - t0.hour) % 24
    days = np.arange(first_midnight, len(g), 24)
    mondays = [d for d in days if (t0 + pd.Timedelta(hours=int(d))).weekday() == 0]
    bx.set_xticks(mondays, [(t0 + pd.Timedelta(hours=int(d))).strftime("%d %b").lstrip("0") for d in mondays])
    bx.set_xticks(days, minor=True)
    bx.set_xlim(-0.5, len(g) - 0.5)

    key = [Patch(facecolor=SPLIT["train"], label="training targets"),
           Patch(facecolor=SPLIT["val"], label="validation targets"),
           Patch(facecolor=SPLIT["test"], label="test targets"),
           Patch(facecolor=RECON_TINT, edgecolor=RECON_HATCH, hatch="//////", linewidth=0,
                 label="reconstructed (never scored)")]
    fig.legend(handles=key, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=4, frameon=False,
               handlelength=1.3, handleheight=0.9, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.3)
    pdf = _save(fig, "fig1_protocol_timeline")
    return pdf, {"gaps": [(int(s), int(e), int(e - s + 1)) for s, e in gaps], "test_start": test_start,
                 "spans": {f"{k[0]}:{k[1]}": v for k, v in spans.items()},
                 "purge_target_span": (purge_origins[0] + 1, purge_origins[1] + 24),
                 "excluded_target_span": (excl[0] + 1, excl[1] + 24), "n_purged": n_purged, "n_excl": n_excl}


def _repel(ys, gap, lo, hi):
    """Spread label positions so neighbours sit at least `gap` apart (order kept)."""
    order = np.argsort(ys)
    pos = np.array(ys, dtype=float)[order]
    for _ in range(200):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < gap:
                mid = (pos[i] + pos[i - 1]) / 2
                pos[i - 1], pos[i] = mid - gap / 2, mid + gap / 2
                moved = True
        pos = np.clip(pos, lo, hi)
        if not moved:
            break
    out = np.empty_like(pos)
    out[order] = pos
    return out


def fig3_per_origin_mae():
    p = pd.read_csv(os.path.join(FIGDATA, "per_origin_mae_24h.csv"), parse_dates=["origin"])
    # highlighted: CLEAR-Former, the two most accurate day-ahead models (Table 5) and the most
    # accurate statistical models at 24 h (Prophet) and at 1 h (Holt-Winters); the other six in grey
    models = [("CLEAR-Former", "CLEAR-Former", "-", "o"),
              ("Transformer", "Transformer", (0, (4, 1.5)), "s"),
              ("MLP", "MLP", (0, (1, 1.2)), "^"),
              ("Prophet", "Prophet", (0, (5, 1.5, 1, 1.5)), "D"),
              ("HoltWinters", "Holt–Winters", (0, (10, 3)), "X")]
    others = [c for c in p.columns if c not in ("origin", *[m[0] for m in models])]
    x = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.2))
    fig.subplots_adjust(left=0.085, right=0.8, top=0.8, bottom=0.235)
    for i, col in enumerate(others):
        ax.plot(x, p[col].to_numpy(), color=OTHER, linewidth=0.7, zorder=1,
                label="other six models" if i == 0 else None)
    ends = []
    for k, ((col, label, ls, mk), colour) in enumerate(zip(models, SERIES)):
        y = p[col].to_numpy()
        ax.plot(x, y, color=colour, linestyle=ls, linewidth=1.4, marker=mk, markersize=4.2,
                markevery=list(range(k % 4, len(x) - 1, 4)), markerfacecolor=colour, markeredgecolor=SURFACE,
                markeredgewidth=0.6, label=label, solid_joinstyle="round", dash_joinstyle="round", zorder=3)
        ends.append((label, y[-1], colour))
    for xi in (0, len(x) - 1):                   # the only two origins with disjoint targets
        ax.axvline(xi, color=MUTED, linewidth=0.7, zorder=0)
    ax.set_xlim(-1.5, len(x) - 0.4)
    ax.set_ylim(0, 44)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylabel("MAE per origin (kW)")
    ax.set_xlabel("Forecast origin", labelpad=2)
    ticks = list(range(0, len(x), 6))
    ax.set_xticks(ticks, [p.origin[i].strftime("%a %d %b\n%H:%M").replace(" 0", " ") if i in (0, 24)
                          else p.origin[i].strftime("%H:%M") for i in ticks])
    ax.set_xticks(x, minor=True)
    _style(ax)
    # direct labels (names only) at the right end, with leader lines when values crowd
    ys = _repel([e[1] for e in ends], gap=3.6, lo=1.5, hi=42)    # 3.6 kW is about one 7 pt line here
    for (label, yv, colour), yl in zip(ends, ys):
        ax.plot([len(x) - 1, len(x) - 0.55, len(x) - 0.15], [yv, yv, yl], color=colour, linewidth=0.6,
                clip_on=False, zorder=2)
        ax.text(len(x) - 0.05, yl, label, ha="left", va="center", fontsize=7, color=INK, clip_on=False)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(m[1]) for m in models] + [labels.index("other six models")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
               bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, handlelength=3.2, columnspacing=1.2,
               handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)
    pdf = _save(fig, "fig3_per_origin_mae_24h")
    first, last = p.iloc[0], p.iloc[-1]
    all11 = [c for c in p.columns if c != "origin"]
    return pdf, {"origins": (str(p.origin.iloc[0]), str(p.origin.iloc[-1]), len(p)),
                 "first": {m[0]: round(float(first[m[0]]), 2) for m in models},
                 "last": {m[0]: round(float(last[m[0]]), 2) for m in models},
                 "grey": others,
                 "best_count_all11": p[all11].idxmin(axis=1).value_counts().to_dict(),
                 "best_count_highlighted5": p[[m[0] for m in models]].idxmin(axis=1).value_counts().to_dict()}


def manifest():
    """Measured page size and text sizes of each figure PDF (Fig. 2 must be compiled first)."""
    import json
    import fitz
    out = {}
    for name in ("fig1_protocol_timeline", "fig2_architecture", "fig3_per_origin_mae_24h"):
        pdf = os.path.join(OUT, name + ".pdf")
        if not os.path.exists(pdf):
            continue
        page = fitz.open(pdf)[0]
        sizes = sorted({round(s["size"], 1) for b in page.get_text("dict")["blocks"]
                        for ln in b.get("lines", []) for s in ln["spans"]})
        out[name] = {"width_mm": round(page.rect.width / 72 * 25.4, 1),
                     "height_mm": round(page.rect.height / 72 * 25.4, 1),
                     "text_pt": sizes, "fonts": sorted({f[3].split("+")[-1] for f in page.get_fonts()})}
    json.dump(out, open(os.path.join(OUT, "figures_manifest.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    for fn in (fig1_protocol_timeline, fig3_per_origin_mae):
        path, info = fn()
        print(os.path.relpath(path, HERE), info)
    print(manifest())
