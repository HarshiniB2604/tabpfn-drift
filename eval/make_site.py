"""
Generate the website's data file:  python -m eval.make_site
    (run python -m eval.make_report first, so results.json and summary.csv are current)

Reads  site_data/results.json   per-block curves (+ meta), written by make_report
       results/summary.csv      one row per (stream, method), written by make_report
       results_ablation/ablation_summary.csv   only if it exists (ADWIN sensitivity block)
Writes site/data.js             window.RESULTS = {...}

Why a .js file and not .json: the page must work when index.html is opened by double-click
(file://). Browsers block fetch() of local files there, but a <script src="data.js"> loads fine.

Everything the page shows comes from here. Statistics (means, CIs, ratios, rolling RMSE,
findings) are computed in Python so the page only formats and draws; it never invents numbers.
"""
import argparse
import inspect
import json
import math
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from common.contract import BLOCK_SIZE, C_MAX, N_BLOCKS, VANILLA_NAME, N_FEATURES
from data.streams import drift_blocks_for, GRADUAL_START, GRADUAL_END, RECUR_PERIOD
from eval.make_report import ci95
from eval.metrics import PRE_WINDOW, TOL
from eval import findings as F

ROLL = 5   # real data: RMSE over the last 5 blocks, same as the app

# Display metadata (names and units; no results). Order = order on the page.
METHODS = [  # key, label, short label (narrow screens), kind, dash (SVG stroke-dasharray; colour comes from kind)
    ("P0_vanilla", "Vanilla", "Vanilla", "tabpfn", ""),
    ("P1_window", "Sliding window", "Window", "tabpfn", "8 4"),
    ("P2_recency", "Recency", "Recency", "tabpfn", "2 3"),
    ("P3_drift", "ADWIN", "ADWIN", "tabpfn", "12 3 2 3"),
    ("P4_valguided", "Validation", "Valid.", "tabpfn", "5 2 1 2"),
    ("kNN_context", "kNN", "kNN", "tabpfn", "1 2"),
    ("XGBoost", "XGBoost", "XGB", "baseline", ""),
    ("MLP", "MLP", "MLP", "baseline", "8 4"),
    ("AdaptiveForest", "Adaptive forest", "ARF", "baseline", "2 3"),
]
DATASETS = [  # key, label, kind, unit
    ("sudden", "Sudden drift", "synthetic", ""),
    ("gradual", "Gradual drift", "synthetic", ""),
    ("recurring", "Recurring drift", "synthetic", ""),
    ("none", "No drift (control)", "synthetic", ""),
    ("bike", "Bike Sharing", "real", "bikes per hour"),
    ("energy", "Appliances Energy", "real", "Wh per 10 min"),
    ("pm25", "Beijing PM2.5", "real", "µg/m³"),
]


def synthetic_story(key):
    """One-line description built from the stream definitions in data/streams.py."""
    if key == "sudden":
        return f"The rule linking inputs to the target changes all at once at block {drift_blocks_for(key)[0]}."
    if key == "gradual":
        return f"Between blocks {GRADUAL_START} and {GRADUAL_END} the old rule slowly fades into a new one."
    if key == "recurring":
        b = drift_blocks_for(key)
        return f"The rule flips every {RECUR_PERIOD} blocks (at {', '.join(map(str, b))}), so old concepts come back."
    return "The rule never changes: a control to measure the cost of each policy when nothing drifts."


def real_info(key):
    """(description, number of input features) from the loader in data/real.py."""
    try:
        from data.real import load_real
        s = load_real(key)
        return s.description, len(s.feature_cols)
    except (FileNotFoundError, OSError):   # data/raw/ missing: still build the site
        return "", None


def candidate_label(name):
    """'window3' -> 'last 3 blocks', 'recency2' -> 'recency, half-life 2 blocks', 'all' -> 'all history'."""
    m = re.fullmatch(r"(window|recency)(\d+)", name)
    if name == "all":
        return "all history"
    if m and m.group(1) == "window":
        return f"last {m.group(2)} blocks"
    if m:
        return f"recency, half-life {m.group(2)} blocks"
    return name


def policy_settings():
    """The policies' settings, read from the code in methods/policies.py (not typed here)."""
    from methods.policies import SlidingWindow, Recency, AdwinPolicy, ValidationPolicy, CANDIDATES
    delta = inspect.signature(AdwinPolicy.__init__).parameters["delta"].default
    start = ValidationPolicy(model=None).current          # the rule validation starts with
    return {
        "P0_vanilla": {},
        "P1_window": dict(window=SlidingWindow.window),
        "P2_recency": dict(half_life=Recency.half_life),
        "P3_drift": dict(delta=delta),
        "P4_valguided": dict(n_val_blocks=ValidationPolicy.n_val_blocks,
                             min_gain_pct=ValidationPolicy.min_gain * 100,
                             candidates=[candidate_label(c) for c in CANDIDATES],
                             start=candidate_label(start)),
        "kNN_context": {},
    }


def baseline_settings():
    """Retraining window of XGBoost / MLP, read from methods/baselines.py."""
    from methods.baselines import _WindowedRefit
    return dict(refit_blocks=inspect.signature(_WindowedRefit.__init__).parameters["n_blocks"].default)


def clean(x):
    """JSON-safe: NaN -> None, numpy -> Python, floats rounded to 5 digits."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return None if math.isnan(x) else round(float(x), 5)
    if isinstance(x, np.integer):
        return int(x)
    return x


def curve_for(c, real):
    """Per-block curve for one (stream, method): mean + 95% CI across seeds, per seed, flags.
    Real data: rolling RMSE over the last ROLL blocks, per seed first, then mean + CI."""
    blocks = c["blocks"]
    by_seed = pd.DataFrame({s: v for s, v in c["rmse_by_seed"].items()}, index=blocks, dtype=float)
    if real:
        by_seed = np.sqrt((by_seed ** 2).rolling(ROLL).mean())   # exact RMSE of the last ROLL*200 rows
    mean, half = zip(*(ci95(row) for row in by_seed.to_numpy()))
    return dict(blocks=blocks, mean=list(mean), ci95=list(half), n_seeds=by_seed.shape[1],
                flags={s: v for s, v in c["drift_flag_blocks"].items()},
                kind="rolling" if real else "block")


def problem_data(curves, summ):
    """Data for 'The problem' section: vanilla on the sudden stream, block by block."""
    c = curves["sudden"][VANILLA_NAME]
    drift = drift_blocks_for("sudden")[0]
    s = pd.Series(c["rmse_mean"], index=c["blocks"])
    pre_blocks = list(range(max(1, drift - PRE_WINDOW), drift))
    pre = float(s.loc[pre_blocks].mean())
    row = F.get(summ, "sudden", VANILLA_NAME)
    return dict(stream="sudden", method=VANILLA_NAME, drift_block=drift, pre_blocks=pre_blocks,
                pre_mean=pre, tolerance_pct=TOL * 100, peak=float(s.max()), peak_block=int(s.idxmax()),
                # ratio of the SHOWN values (2 decimals), so the text ratio matches the numbers beside it
                ratio_to_pre=[round(v, 2) / round(pre, 2) for v in s], recovered_events=row.recovered_events,
                n_seeds=int(row.n_seeds),
                # vanilla samples from ALL history (blocks 0..b-1 when predicting block b), so the share of
                # sampled history from before the drift is drift/b. Shown in the slider text, as a percent.
                old_rule_share_pct=[100 * drift / b if b >= drift else 100.0 for b in c["blocks"]])


def limitations(summ, seeds, problem, ab, blocks):
    """The Limitations list, every number taken from the data or the code (same points as the app)."""
    from methods.policies import ValidationPolicy
    ps, bs = policy_settings(), baseline_settings()
    out = []
    syn = summ[summ.stream.isin(["sudden", "gradual", "recurring", "none"])]
    out.append(dict(title=f"Few seeds on synthetic data",
                    text=f"Every synthetic result is the mean of {int(syn.n_seeds.max())} seeds (seeds {', '.join(map(str, seeds))}). "
                         "The 95% intervals are wide and rough, small differences are within noise, and we make no "
                         "confidence claims from so few seeds."))
    parts = []
    for ds in F.REAL_SETS:
        g = summ[(summ.stream == ds) & summ.method.isin(F.TABPFN)]
        if g.empty:
            parts.append(f"{ds}: no TabPFN results yet")
            continue
        most = int(g.n_seeds.mode().iloc[0])
        odd = [f"{dict((k, l) for k, l, *_ in METHODS)[r.method]} {int(r.n_seeds)} seed{'s' if r.n_seeds > 1 else ''}"
               for r in g.itertuples() if r.n_seeds != most]
        parts.append(f"{ds}: {most} seed{'s' if most > 1 else ''} per policy" + (f" ({', '.join(odd)})" if odd else ""))
    zero = sorted({r.method for r in summ[summ.stream.isin(F.REAL_SETS)].itertuples()
                   if r.n_seeds > 1 and r.rmse_ci95 == 0})
    out.append(dict(title="Few seeds for TabPFN on real data",
                    text="TabPFN policies on real data: " + "; ".join(parts) + ". A method run once has no interval at all."
                         + (f" {', '.join(zero)} gave exactly the same result on every seed, so an interval of 0 there means "
                            "'deterministic', not 'precise'." if zero else "")))
    out.append(dict(title="Untuned baselines with a short window",
                    text=f"XGBoost and the MLP are retrained before every block on only the last {bs['refit_blocks']} blocks "
                         f"({bs['refit_blocks'] * BLOCK_SIZE} rows); TabPFN policies may use up to {C_MAX} rows. The adaptive "
                         "forest learns row by row. None of them was tuned: they are reference points, not the best possible "
                         "versions of these models."))
    adw = []
    for ds in F.REAL_SETS:
        r = F.get(summ, ds, F.ADW)
        if r is not None:
            adw.append(f"{ds} {r.drift_flags_mean:.0f} of {blocks[ds]} blocks")
    worst = [ds for ds in F.REAL_SETS if not summ[(summ.stream == ds) & summ.method.isin(F.TABPFN)].empty
             and summ[(summ.stream == ds) & summ.method.isin(F.TABPFN)].sort_values("rmse_mean").method.iloc[-1] == F.ADW]
    out.append(dict(title="ADWIN over-triggers on real data",
                    text=f"At its default setting ADWIN flagged drift in many blocks ({', '.join(adw)}; mean over seeds). Each "
                         "flag throws away older history." + (f" It has the highest mean RMSE of the TabPFN policies on "
                         f"{', '.join(worst)}." if worst else "") + " " + F.adwin_limitation(summ, ab)))
    out.append(dict(title="Synthetic drift is not real drift",
                    text=f"On the synthetic sudden stream we create a clean change: vanilla's error jumps to "
                         f"{F.r2(problem['peak']) / F.r2(problem['pre_mean']):.1f}× its pre-drift level ({F.r2(problem['pre_mean']):.2f} "
                         f"→ {F.r2(problem['peak']):.2f}). In the real data the true drift points are unknown, so we can only "
                         "compare error over time, not recovery time. Results on synthetic streams do not prove anything "
                         "about real data."))
    out.append(dict(title="Policy settings were not tuned",
                    text=f"Window {ps['P1_window']['window']} blocks, recency half-life {ps['P2_recency']['half_life']} blocks, "
                         f"validation scored on the last {ps['P4_valguided']['n_val_blocks']} blocks with a "
                         f"{ps['P4_valguided']['min_gain_pct']:.0f}% switching margin, ADWIN δ = {ps['P3_drift']['delta']} "
                         "(see the ADWIN sensitivity check for other values)."))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="site_data/results.json")
    p.add_argument("--summary", default="results/summary.csv")
    p.add_argument("--ablation", default="results_ablation/ablation_summary.csv")
    p.add_argument("--out", default="site/data.js")
    a = p.parse_args()

    rj = json.load(open(a.json, encoding="utf-8"))
    summ = pd.read_csv(a.summary)
    curves_in = rj["curves"]

    # safety: summary.csv and results.json must come from the same make_report run
    js = pd.DataFrame(rj["summary"])[["stream", "method", "rmse_mean"]]
    m = summ.merge(js, on=["stream", "method"], how="outer", suffixes=("", "_json"))
    if m.isna()[["rmse_mean", "rmse_mean_json"]].any().any() or not np.allclose(m.rmse_mean, m.rmse_mean_json, atol=1e-4):
        raise SystemExit("summary.csv and results.json disagree: run python -m eval.make_report first")

    have = set(summ.stream)
    datasets = []
    for key, label, kind, unit in DATASETS:
        if key not in have:
            continue
        real = kind == "real"
        n_blocks = len(next(iter(curves_in[key].values()))["blocks"])
        datasets.append(dict(key=key, label=label, kind=kind, unit=unit,
                             story=real_info(key)[0] if real else synthetic_story(key),
                             n_features=real_info(key)[1] if real else N_FEATURES,
                             drift_blocks=drift_blocks_for(key), scored_blocks=n_blocks,
                             gradual=[GRADUAL_START, GRADUAL_END] if key == "gradual" else None,
                             methods=sorted(set(summ.method[summ.stream == key]))))

    curves = {ds: {m: curve_for(c, ds in F.REAL_SETS) for m, c in by_m.items()} for ds, by_m in curves_in.items()}
    blocks = {ds["key"]: ds["scored_blocks"] for ds in datasets}

    ab = pd.read_csv(a.ablation) if Path(a.ablation).exists() else None   # made by eval.make_ablation
    ablation = ab.to_dict(orient="records") if ab is not None else None
    sensitivity = F.adwin_sensitivity(ab, summ)          # None if missing or stand-in: the block is hidden

    data = dict(
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        sources=dict(results_json=a.json, summary_csv=a.summary,
                     ablation_csv=a.ablation if ablation is not None else None),
        settings=dict(block_size=BLOCK_SIZE, c_max=C_MAX, n_blocks=N_BLOCKS, rolling_window=ROLL,
                      pre_window=PRE_WINDOW, tolerance_pct=TOL * 100, ci=rj["meta"]["ci"],
                      seeds=rj["meta"]["seeds"]),
        methods=[dict(key=k, label=l, short=sh, kind=kd, dash=d) for k, l, sh, kd, d in METHODS],
        policy_settings=policy_settings(),
        baseline_settings=baseline_settings(),
        warmup_blocks=int(min(len(range(c["blocks"][0])) for by_m in curves_in.values() for c in by_m.values())),
        datasets=datasets,
        summary=summ.to_dict(orient="records"),
        curves=curves,
        problem=problem_data(curves_in, summ),
        limitations=limitations(summ, rj["meta"]["seeds"], problem_data(curves_in, summ), ab, blocks),
        findings=F.all_findings(summ, blocks, ablation_done=sensitivity is not None),
        robustness=F.robustness(summ),
        robustness_takeaway=F.robustness_takeaway(F.robustness(summ)),
        synthetic_winners=F.synthetic_winners(summ),
        ablation=ablation,
        adwin_sensitivity=sensitivity,
        adwin_limitation=F.adwin_limitation(summ, ab),
        counts=dict(synthetic_drift=sum(1 for d in datasets if d["kind"] == "synthetic" and d["drift_blocks"]),
                    synthetic_control=sum(1 for d in datasets if d["kind"] == "synthetic" and not d["drift_blocks"]),
                    real=sum(1 for d in datasets if d["kind"] == "real"),
                    tabpfn_policies=int(summ.method.isin(F.TABPFN).groupby(summ.method).any().sum()),
                    baselines=int((~summ.method.isin(F.TABPFN)).groupby(summ.method).any().sum())),
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean(data), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    out.write_text("/* Generated by python -m eval.make_site. Do not edit by hand. */\n"
                   f"window.RESULTS = {text};\n", encoding="utf-8")
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB): {len(datasets)} datasets, "
          f"{len(data['findings'])} findings, robustness for {list(data['robustness'])}, "
          f"ADWIN sensitivity block: {'yes' if sensitivity is not None else 'hidden (no ablation)'}")


if __name__ == "__main__":
    main()
