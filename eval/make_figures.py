"""
Make PNG charts from results/summary.csv (run `python -m eval.make_report` first).
    python -m eval.make_figures

Saves to results/figures/:
- rmse_{stream}.png        mean RMSE per method (sudden, gradual, recurring, none)
- recovery_{stream}.png    recovery time per method (sudden, gradual, recurring)
- regret_none.png          no-drift regret per method (% vs P0 vanilla)
- rmse_{bike,pm25,energy}.png  mean RMSE per method on the real datasets (if present).
  Real data has no known drift point, so there is no recovery chart for it.

Error bars = 95% CI across seeds (rough with 3 seeds).
Blue = TabPFN context policy, orange = baseline model.
Recovery: hatched bar + text = at least one drift was NOT recovered; the bar length is
then the cap (blocks until the next drift / end of stream), not a measured time.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

# Colours: reference palette, categorical slots 1-2 (validated: CVD + contrast pass)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
KIND_COLOR = {"TabPFN policy": "#2a78d6", "baseline": "#eb6834"}

# Fixed method order (same in every chart) and readable names
ORDER = ["P0_vanilla", "P1_window", "P2_recency", "P3_drift", "P4_valguided",
         "kNN_context", "XGBoost", "MLP", "AdaptiveForest"]
LABEL = {"P0_vanilla": "P0 vanilla", "P1_window": "P1 sliding window",
         "P2_recency": "P2 recency", "P3_drift": "P3 ADWIN", "P4_valguided": "P4 validation",
         "kNN_context": "kNN context", "XGBoost": "XGBoost", "MLP": "MLP",
         "AdaptiveForest": "Adaptive forest"}
NOTE = "Synthetic Friedman-1 streams, {n} seeds. Error bars: 95% CI across seeds (rough with few seeds)."
REAL_NOTE = "Real data, {n}. Error bars: 95% CI across seeds. No known drift point."
REAL = {"bike": ("Bike Sharing", "bikes per hour"),
        "pm25": ("Beijing PM2.5", "PM2.5 in ug/m3"),
        "energy": ("Appliances Energy", "Wh per 10 min")}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK, "font.size": 10,
})


def base_axes(ax, xlabel):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)


def hbars(ax, d, value, ci, hatch_mask=None):
    """Horizontal bars in ORDER (top to bottom), coloured by kind, with CI whiskers."""
    d = d.set_index("method").reindex([m for m in ORDER if m in set(d.method)])
    y = range(len(d))[::-1]
    colors = [KIND_COLOR[k] for k in d["kind"]]
    hatches = ["////" if (hatch_mask is not None and hatch_mask.get(m, False)) else "" for m in d.index]
    bars = ax.barh(list(y), d[value], height=0.62, color=colors, edgecolor=SURFACE, linewidth=2,
                   hatch=None)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
            bar.set_edgecolor(SURFACE)
            bar.set_alpha(0.55)
    ax.errorbar(d[value], list(y), xerr=d[ci].fillna(0), fmt="none",
                ecolor=INK2, elinewidth=1.2, capsize=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([LABEL[m] for m in d.index])
    return d, list(y)


def legend(fig, extra=None):
    """One legend row under the title, so it never covers bars."""
    handles = [Patch(color=c, label="TabPFN context policy" if k == "TabPFN policy" else "Baseline (retrained model)")
               for k, c in KIND_COLOR.items()]
    if extra:
        handles.append(extra)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.935), ncol=len(handles),
               frameon=False, fontsize=9)


def finish(fig, title, n, path, note=NOTE):
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.015, note.format(n=n), fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.88))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def rmse_chart(s, stream, n, out):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    hbars(ax, s[s.stream == stream], "rmse_mean", "rmse_ci95")
    base_axes(ax, "mean RMSE over blocks 1-59 (lower is better)")
    if stream == "none":
        ax.set_xlim(0, None)
    legend(fig)
    finish(fig, f"Mean RMSE per method - {stream} stream", n, out / f"rmse_{stream}.png")


def real_rmse_chart(s, stream, out):
    d = s[s.stream == stream]
    name, units = REAL[stream]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    dd, y = hbars(ax, d, "rmse_mean", "rmse_ci95")
    for yi, (m, row) in zip(y, dd.iterrows()):      # one seed -> no CI possible: say so on the bar
        if row.n_seeds < 2:
            ax.text(row.rmse_mean, yi, "  1 seed, no CI", va="center", fontsize=8.5, color=INK2)
    base_axes(ax, f"mean block RMSE, {units} (lower is better)")
    ax.set_xlim(0, d.rmse_mean.max() * 1.25)
    legend(fig)
    missing = "" if (d.kind == "TabPFN policy").any() else ", baselines only so far"
    def count(g):   # "3 seeds", "1 seed", or "1-3 seeds" when methods differ
        lo, hi = int(g.n_seeds.min()), int(g.n_seeds.max())
        return f"{lo} seed" if hi == 1 else f"{hi} seeds" if lo == hi else f"{lo}-{hi} seeds"
    seeds = ", ".join(f"{k}s: {count(g)}" for k, g in d.groupby("kind"))   # e.g. "TabPFN policys: 3 seeds"
    finish(fig, f"Mean RMSE - {name} (real data){missing}", seeds.replace("policys", "policies"),
           out / f"rmse_{stream}.png", note=REAL_NOTE)


def recovery_chart(s, stream, n, out):
    d = s[s.stream == stream].copy()
    got, total = zip(*(map(int, e.split("/")) for e in d.recovered_events))
    d["got"], d["total"] = got, total
    not_all = dict(zip(d.method, d.got < d.total))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    dd, y = hbars(ax, d, "recovery_mean", "recovery_ci95", hatch_mask=not_all)
    xmax = (dd.recovery_mean + dd.recovery_ci95.fillna(0)).max()
    for m, yy in zip(dd.index, y):
        if not_all[m]:
            g, t = int(dd.loc[m, "got"]), int(dd.loc[m, "total"])
            txt = "NOT recovered (capped)" if g == 0 else f"recovered {g}/{t} drifts (rest capped)"
            ax.text(dd.loc[m, "recovery_mean"] + dd.loc[m, "recovery_ci95"] + xmax * 0.02, yy, txt,
                    va="center", fontsize=8.5, color=INK, fontweight="bold" if g == 0 else "normal")
    ax.set_xlim(0, xmax * 1.55)
    base_axes(ax, "blocks after the drift until RMSE is back within 10% (lower is better)")
    legend(fig, Patch(facecolor="white", edgecolor=INK2, hatch="////", label="not recovered every time (capped)"))
    start = " (counted from block 40, end of the transition)" if stream == "gradual" else ""
    finish(fig, f"Recovery time per method - {stream} stream{start}", n, out / f"recovery_{stream}.png")


def regret_chart(s, n, out):
    d = s[(s.stream == "none")].copy()
    pol = d[d.kind == "TabPFN policy"]
    base = d[d.kind == "baseline"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    for ax, part, title in ((a1, pol, "TabPFN context policies"), (a2, base, "Baselines (note: different scale)")):
        dd, y = hbars(ax, part, "regret_pct_mean", "regret_pct_ci95")
        ax.axvline(0, color=INK2, linewidth=1)
        base_axes(ax, "% RMSE vs P0 vanilla on the no-drift stream")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        for m, yy in zip(dd.index, y):
            if m == "P0_vanilla":
                ax.text(0, yy, "  reference (0 by definition)", va="center", fontsize=8.5, color=INK2)
            elif abs(dd.loc[m, "regret_pct_mean"]) < 1e-9 and dd.loc[m, "regret_pct_ci95"] < 1e-9:
                ax.text(0, yy, "  0.0 (identical to vanilla in every seed)", va="center", fontsize=8.5, color=INK2)
    lim = max(abs(pol.regret_pct_mean) + pol.regret_pct_ci95.fillna(0)) * 1.3
    a1.set_xlim(-lim, lim)
    a2.set_xlim(0, (base.regret_pct_mean + base.regret_pct_ci95).max() * 1.1)
    legend(fig)
    finish(fig, "No-drift regret: cost of the method when nothing changes (lower is better)", n,
           out / "regret_none.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/summary.csv")
    p.add_argument("--out", default="results/figures")
    a = p.parse_args()
    s = pd.read_csv(a.summary)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = int(s.n_seeds.max())
    for stream in ("sudden", "gradual", "recurring", "none"):
        rmse_chart(s, stream, n, out)
    for stream in ("sudden", "gradual", "recurring"):
        recovery_chart(s, stream, n, out)
    regret_chart(s, n, out)
    for stream in REAL:
        if (s.stream == stream).any():
            real_rmse_chart(s, stream, out)


if __name__ == "__main__":
    main()
