"""
Build the results report from the per-block CSVs.
    python -m eval.make_report                  (reads results/)
    python -m eval.make_report --results DIR    (any other folder)

Writes:
- results/summary.csv     one row per (stream, method): mean +/- 95% CI across seeds
- site_data/results.json  the same summary + per-seed table + per-block curves for plots

How the numbers are made (all per seed first, then averaged over seeds):
- mean RMSE      : average block RMSE over all scored blocks (1..59) of one run.
- recovery time  : eval/metrics.py recovery rule, averaged over the drifts in that run.
                   If a policy never recovers, the value is CAPPED (blocks until the next
                   drift / end of stream) and counted in recovered_events as a failure.
- no-drift regret: only on the 'none' stream. 100 * (method RMSE - vanilla RMSE) / vanilla RMSE,
                   paired by seed (same data). Vanilla is the reference (no value).
- 95% CI         : t-interval, mean +/- t(0.975, n-1) * sd / sqrt(n). With 3 seeds this
                   interval is wide and only rough: no confidence claims from 3 seeds.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from common.contract import VANILLA_NAME, ALIASES
from data.streams import drift_blocks_for
from eval.metrics import load_results, _recovery_for_run

TABPFN_POLICIES = set(ALIASES.values())     # P0..P4 + kNN_context; the rest are baselines


def ci95(values):
    """Mean and 95% CI half-width of a list of per-seed values (NaNs dropped)."""
    v = np.asarray([x for x in values if not pd.isna(x)], dtype=float)
    if len(v) == 0:
        return np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), np.nan
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return float(v.mean()), float(half)


def per_seed_table(res):
    """One row per run (stream, method, seed)."""
    rows = []
    for (stream, method, seed), run in res.groupby(["stream", "method", "seed"]):
        rec = _recovery_for_run(run, stream)           # list of (blocks, recovered?) per drift
        rows.append(dict(
            stream=stream, method=method, seed=int(seed),
            mean_rmse=run["rmse"].mean(),
            recovery_blocks=np.mean([r[0] for r in rec]) if rec else np.nan,
            recovered=sum(r[1] for r in rec), n_drifts=len(rec),
            drift_flags=int(run["drift_flag"].sum()),
        ))
    seeds = pd.DataFrame(rows)

    # Paired no-drift regret: compare with vanilla on the same seed of the 'none' stream.
    van = seeds[(seeds.stream == "none") & (seeds.method == VANILLA_NAME)].set_index("seed")["mean_rmse"]
    seeds["regret_pct"] = np.nan
    m = (seeds.stream == "none") & (seeds.method != VANILLA_NAME) & seeds.seed.isin(van.index)
    base = seeds.loc[m, "seed"].map(van)
    seeds.loc[m, "regret_pct"] = 100 * (seeds.loc[m, "mean_rmse"] - base) / base
    return seeds


def summary(seeds):
    rows = []
    for (stream, method), g in seeds.groupby(["stream", "method"]):
        rmse, rmse_ci = ci95(g.mean_rmse)
        rec, rec_ci = ci95(g.recovery_blocks)
        reg, reg_ci = ci95(g.regret_pct)
        rows.append(dict(
            stream=stream, method=method,
            kind="TabPFN policy" if method in TABPFN_POLICIES else "baseline",
            n_seeds=g.seed.nunique(),
            rmse_mean=rmse, rmse_ci95=rmse_ci,
            recovery_mean=rec, recovery_ci95=rec_ci,
            recovered_events=f"{int(g.recovered.sum())}/{int(g.n_drifts.sum())}" if stream != "none" else "",
            regret_pct_mean=reg, regret_pct_ci95=reg_ci,
            drift_flags_mean=g.drift_flags.mean(),
        ))
    return pd.DataFrame(rows).sort_values(["stream", "rmse_mean"]).reset_index(drop=True)


def curves(res):
    """Per-block RMSE per (stream, method): each seed + mean + 95% CI band."""
    out = {}
    for (stream, method), g in res.groupby(["stream", "method"]):
        wide = g.pivot(index="block", columns="seed", values="rmse").sort_index()
        mean, half = zip(*(ci95(row) for row in wide.to_numpy()))
        flags = g[g.drift_flag == 1].groupby("seed")["block"].apply(list)
        out.setdefault(stream, {})[method] = dict(
            blocks=wide.index.astype(int).tolist(),
            rmse_mean=list(mean),
            rmse_ci95=list(half),
            rmse_by_seed={str(s): wide[s].tolist() for s in wide.columns},
            drift_flag_blocks={str(s): [int(b) for b in v] for s, v in flags.items()},
        )
    return out


def clean(x):
    """Make data JSON-safe: NaN -> null, numpy numbers -> plain Python numbers."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if math.isnan(x) else round(float(x), 5)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--summary-out", default="results/summary.csv")
    p.add_argument("--json-out", default="site_data/results.json")
    a = p.parse_args()

    res = load_results(a.results)                 # also maps teammate names via ALIASES
    seeds = per_seed_table(res)
    summ = summary(seeds)

    Path(a.summary_out).parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(a.summary_out, index=False)

    report = {
        "meta": {
            "source": a.results,
            "data": "synthetic Friedman-1 drift streams (not real-world data)",
            "seeds": sorted(int(s) for s in res.seed.unique()),
            "ci": "95% t-interval across seeds; with 3 seeds it is rough, not a confidence claim",
            "recovery_note": "capped at blocks-until-next-drift when not recovered; see recovered_events",
            "regret_note": "only on 'none' stream, % vs P0_vanilla, paired by seed",
            "drift_blocks": {s: drift_blocks_for(s) for s in sorted(res.stream.unique())},
        },
        "summary": summ.to_dict(orient="records"),
        "per_seed": seeds.to_dict(orient="records"),
        "curves": curves(res),
    }
    Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.json_out, "w", encoding="utf-8") as f:
        json.dump(clean(report), f, indent=1, allow_nan=False)

    pd.set_option("display.width", 200)
    print(summ.drop(columns=["kind"]).round(3).to_string(index=False))
    print(f"\nsaved {a.summary_out} and {a.json_out}")


if __name__ == "__main__":
    main()
