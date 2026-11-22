"""
ADWIN sensitivity ablation: how does ADWIN's delta change the number of drift flags and the error?
    python -m eval.make_ablation
    python -m eval.make_ablation --ablation DIR --main DIR --fig PATH

delta is ADWIN's significance level: smaller delta = more evidence needed = fewer drift flags.

Reads:
- results_ablation/  ADWIN runs with other deltas, method names like P3_drift_d1e-4
                     (made by: run_experiments.py --methods adwin --adwin-delta 1e-4 --out results_ablation)
- results/           the default ADWIN runs (delta 0.002, method P3_drift) and P0_vanilla, unchanged
Prints one table per dataset and saves:
- results_ablation/ablation_summary.csv
- results_ablation/figures/adwin_delta.png
It never writes to results/ and does not touch summary.csv or the main report.

Per dataset and delta (per seed first, then mean +/- 95% CI across seeds, like make_report):
- mean RMSE         : average block RMSE of the run
- drift flags       : number of blocks where ADWIN flagged drift in the run
- RMSE vs vanilla % : 100 * (ADWIN RMSE - vanilla RMSE) / vanilla RMSE, paired by seed
Real datasets use seed 0 only in this plan: 1 seed = no confidence interval.
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.contract import RESULT_COLUMNS, ALIASES, VANILLA_NAME
from data.streams import drift_blocks_for
from eval.make_report import ci95
from eval.make_figures import SURFACE, INK, INK2, GRID, KIND_COLOR

PLAN = {"sudden": [0, 1, 2], "gradual": [0, 1, 2], "bike": [0], "energy": [0]}   # dataset: seeds
DEFAULT_DELTA = 0.002
DELTAS = [0.002, 1e-4, 1e-6, 1e-10]      # planned; any other delta found in the folder is added
ADWIN = "P3_drift"
ABL_NAME = re.compile(r"^P3_drift_d(?P<delta>[0-9.e+-]+)$")
COLOR = KIND_COLOR["TabPFN policy"]


def read_folder(folder):
    """All per-block result CSVs in a folder (other CSVs are skipped). Marks stand-in runs."""
    files = sorted(Path(folder).glob("*.csv"))
    frames = [d for d in (pd.read_csv(f) for f in files) if list(d.columns) == RESULT_COLUMNS]
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS + ["standin"])
    df = pd.concat(frames, ignore_index=True)
    df["standin"] = df.method.str.endswith("-standin")
    df["method"] = df.method.str.replace(r"-standin$", "", regex=True).replace(ALIASES)
    return df


def collect(ablation_dir, main_dir):
    """One table of runs with a 'delta' column: main ADWIN = 0.002, ablation names parsed."""
    main, abl = read_folder(main_dir), read_folder(ablation_dir)
    van = main[main.method == VANILLA_NAME]
    default = main[main.method == ADWIN].assign(delta=DEFAULT_DELTA)
    abl = abl[abl.method.str.match(ABL_NAME)].copy()
    abl["delta"] = abl.method.str.extract(ABL_NAME)["delta"].astype(float)
    runs = pd.concat([default, abl], ignore_index=True)
    keep = lambda d: d[[s in PLAN and seed in PLAN[s] for s, seed in zip(d.stream, d.seed)]]
    return keep(runs), keep(van)


def summarise(runs, van):
    """One row per (dataset, delta): means and 95% CIs across the seeds that exist."""
    rows = []
    van_rmse = van.groupby(["stream", "seed"]).rmse.mean()
    deltas = sorted(set(DELTAS) | set(runs.delta), reverse=True)     # default first, most conservative last
    for ds, seeds in PLAN.items():
        for delta in deltas:
            d = runs[(runs.stream == ds) & np.isclose(runs.delta, delta, rtol=1e-6, atol=0)]
            per = d.groupby("seed").agg(rmse=("rmse", "mean"), n_flags=("drift_flag", "sum"),
                                        blocks=("block", "nunique"))
            per["vs_van"] = [100 * (r - van_rmse[(ds, s)]) / van_rmse[(ds, s)] if (ds, s) in van_rmse.index
                             else np.nan for s, r in per.rmse.items()]
            rm, rc = ci95(per.rmse)
            fm, fc = ci95(per.n_flags)
            vm, vc = ci95(per.vs_van)
            rows.append(dict(dataset=ds, delta=delta, default=delta == DEFAULT_DELTA,
                             seeds_run=len(per), seeds_planned=len(seeds),
                             blocks=int(per.blocks.max()) if len(per) else np.nan,
                             rmse_mean=rm, rmse_ci95=rc, flags_mean=fm, flags_ci95=fc,
                             vs_vanilla_pct=vm, vs_vanilla_ci95=vc,
                             vanilla_rmse=np.mean([van_rmse.get((ds, s), np.nan) for s in per.index])
                             if len(per) else np.nan))
    return pd.DataFrame(rows)


def tag(delta):
    """0.0001 -> '1e-4' (same spelling as the file names, see run_experiments.delta_tag)."""
    mant, exp = f"{delta:.0e}".split("e")
    return f"{mant}e{int(exp)}"


def cell(m, c, n, digits=2):
    if pd.isna(m):
        return "not run yet"
    if n < 2 or pd.isna(c):
        return f"{m:.{digits}f} (1 seed, no CI)" if n == 1 else f"{m:.{digits}f}"
    return f"{m:.{digits}f} +/- {c:.{digits}f}"


def print_tables(summ, standin):
    if standin:
        print("!!! STAND-IN MODEL (ExtraTrees), NOT TabPFN: these numbers only test the pipeline !!!\n")
    for ds, g in summ.groupby("dataset", sort=False):
        true = drift_blocks_for(ds)
        truth = f"true drift starts at {true}" if ds in ("sudden", "gradual", "recurring", "none") \
            else "real data: true drift points unknown"
        van = g.vanilla_rmse.dropna()
        print(f"=== {ds}  ({truth}; seeds planned: {PLAN[ds]}; vanilla mean RMSE: "
              f"{van.iloc[0]:.2f})" if len(van) else f"=== {ds}  ({truth}; no vanilla results)")
        t = pd.DataFrame({
            "delta": [tag(d) +(" (default)" if dflt else "") for d, dflt in zip(g.delta, g.default)],
            "seeds": [f"{r}/{p}" for r, p in zip(g.seeds_run, g.seeds_planned)],
            "mean RMSE (95% CI)": [cell(m, c, n) for m, c, n in zip(g.rmse_mean, g.rmse_ci95, g.seeds_run)],
            "drift flags per run (95% CI)": [cell(m, c, n, 1) for m, c, n in zip(g.flags_mean, g.flags_ci95, g.seeds_run)],
            "RMSE vs vanilla % (95% CI)": [cell(m, c, n, 1) for m, c, n in zip(g.vs_vanilla_pct, g.vs_vanilla_ci95, g.seeds_run)],
        })
        print(t.to_string(index=False), "\n")


def chart(summ, path, standin):
    """Top row: drift flags per run; bottom row: RMSE vs vanilla (%). One column per dataset.
    One series per panel (no legend needed); the reference lines are explained in the titles."""
    datasets = list(PLAN)
    deltas = sorted(summ.delta.unique(), reverse=True)
    x = np.arange(len(deltas))
    labels = [tag(d) + ("\n(default)" if d == DEFAULT_DELTA else "") for d in deltas]
    fig, axes = plt.subplots(2, len(datasets), figsize=(3.4 * len(datasets), 6.8))
    rows = [("flags_mean", "flags_ci95", "drift flags per run"),
            ("vs_vanilla_pct", "vs_vanilla_ci95", "RMSE vs vanilla (%)\n(0 = same as vanilla)")]
    for j, ds in enumerate(datasets):
        g = summ[summ.dataset == ds].set_index("delta").reindex(deltas)
        n = int(g.seeds_run.max()) if g.seeds_run.notna().any() else 0
        synthetic = ds in ("sudden", "gradual")
        for i, (col, ci, ylabel) in enumerate(rows):
            ax = axes[i, j]
            y, e = g[col].to_numpy(float), g[ci].fillna(0).to_numpy(float)
            ok = ~np.isnan(y)
            lo = np.minimum(e, y) if i == 0 else e          # a flag count can't go below 0
            ax.errorbar(x[ok], y[ok], yerr=[lo[ok], e[ok]], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3)
            ax.plot(x[ok], y[ok], color=COLOR, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, zorder=3)
            for xi in x[~ok]:   # x in data units, y in axes units: always inside the panel
                ax.text(xi, 0.04, "not run", ha="center", va="bottom", fontsize=8, color=INK2,
                        transform=ax.get_xaxis_transform())
            if i == 0:
                if synthetic:
                    k = len(drift_blocks_for(ds))
                    ax.axhline(k, color=INK2, ls="--", lw=1)
                    extra = f"\ndashed line = true drifts ({k})"
                else:
                    extra = "\ntrue drifts unknown"
                ax.set_ylim(bottom=0)
                ax.set_title(f"{ds} ({'synthetic' if synthetic else 'real'}), {n} seed{'s' if n != 1 else ''}"
                             + (", no CI" if n == 1 else "") + extra, fontsize=10, color=INK)
            else:
                ax.axhline(0, color=INK2, lw=1)
                ax.set_xlabel("ADWIN delta (smaller = stricter)", fontsize=9)
            ax.set_xticks(x, labels if i == 1 else [""] * len(x), fontsize=8)
            ax.set_xlim(-0.5, len(deltas) - 0.5)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(axis="y", color=GRID)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    title = "ADWIN sensitivity: drift flags and error vs delta (P3, TabPFN)"
    if standin:
        title = "STAND-IN MODEL, NOT TabPFN (pipeline test) - " + title.replace(", TabPFN", "")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Error bars: 95% CI across seeds (rough with 3 seeds; none with 1 seed). "
             "Real data: true drift points unknown, so no true-drift line.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", default="results_ablation")
    p.add_argument("--main", default="results", help="default-delta ADWIN + vanilla runs (read only)")
    p.add_argument("--fig", default=None, help="default: <ablation>/figures/adwin_delta.png")
    a = p.parse_args()

    runs, van = collect(a.ablation, a.main)
    if runs.empty:
        raise SystemExit("no ADWIN runs found for the planned datasets/seeds")
    kinds = set(runs.standin) | set(van.standin)
    if len(kinds) > 1:
        raise SystemExit("mixed stand-in and TabPFN results: they cannot be compared. Use matching folders.")
    standin = kinds == {True}

    summ = summarise(runs, van)
    print_tables(summ, standin)
    out = Path(a.ablation)
    summ.assign(model="STAND-IN (not TabPFN)" if standin else "TabPFN").to_csv(out / "ablation_summary.csv", index=False)
    print("saved", out / "ablation_summary.csv")
    chart(summ, a.fig or out / "figures" / "adwin_delta.png", standin)


if __name__ == "__main__":
    main()
