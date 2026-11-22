"""
The three key findings and the robustness table, computed from the summary table
(results/summary.csv) and the per-block curves (site_data/results.json).

Same logic as the "Key findings" panel in app/app.py: every number comes from the data, and
each headline is only worded as a claim if the numbers support it (otherwise it says so).
Used by eval/make_site.py. Returns plain Python data (dicts, lists, strings).
"""
import pandas as pd
from data.streams import drift_blocks_for

DRIFT_STREAMS = ("sudden", "gradual", "recurring")
REAL_SETS = ("bike", "energy", "pm25")
VAN, VAL, ADW = "P0_vanilla", "P4_valguided", "P3_drift"
TABPFN = ("P0_vanilla", "P1_window", "P2_recency", "P3_drift", "P4_valguided", "kNN_context")


def get(s, stream, method):
    """One summary row, or None if that method has no results on that stream."""
    r = s[(s.stream == stream) & (s.method == method)]
    return None if r.empty else r.iloc[0]


def fmt(mean, ci, n, digits=2):
    """'50.22 +/- 1.30', or '50.22 (1 seed, no CI)'."""
    if pd.isna(mean):
        return "n/a"
    if n < 2 or pd.isna(ci):
        return f"{mean:.{digits}f} (1 seed, no CI)"
    return f"{mean:.{digits}f} ± {ci:.{digits}f}"


def r2(x):
    """The value as it is shown (2 decimals). Ratios and % are computed from these shown values,
    so a reader who divides the numbers on the page gets the same ratio."""
    return round(float(x), 2)


def pct(a, b):
    """% difference of the SHOWN (2-decimal) values."""
    return 100 * (r2(a) - r2(b)) / r2(b)


def evidence(rows, what):
    """'Synthetic streams: 3 seeds per method, with 95% CI (rough with 3 seeds).'"""
    n = sorted({int(r.n_seeds) for r in rows})
    seeds = " or ".join(map(str, n)) + (" seed" if n == [1] else " seeds")
    has = [pd.notna(r.rmse_ci95) for r in rows]
    if all(has):
        ci = "with 95% CI (rough with 3 seeds)"
    elif not any(has):
        ci = "no confidence interval (a single run)"
    else:
        ci = "95% CI where there are 3 seeds; 1-seed values have no CI"
    return dict(text=f"{what}: {seeds} per method, {ci}.", seeds=n, has_ci=all(has))


def compare(a, b):
    """Plain verdict for a vs b (lower RMSE is better), based on the 95% CIs."""
    if pd.isna(a.rmse_ci95) or pd.isna(b.rmse_ci95):
        return "no conclusion: a single run has no CI"
    if a.rmse_mean + a.rmse_ci95 < b.rmse_mean - b.rmse_ci95:
        return "lower, CIs do not overlap"
    if a.rmse_mean - a.rmse_ci95 > b.rmse_mean + b.rmse_ci95:
        return "higher, CIs do not overlap"
    return "CIs overlap: no clear difference"


def events(rows):
    """'3/3' strings -> (recovered, total) summed over the rows."""
    pairs = [tuple(int(x) for x in str(r.recovered_events).split("/")) for r in rows]
    return sum(p[0] for p in pairs), sum(p[1] for p in pairs)


def finding_vanilla(s):
    base, drift = get(s, "none", VAN), [get(s, k, VAN) for k in DRIFT_STREAMS]
    if base is None or any(r is None for r in drift):
        return None
    # supported only if every drift stream's CI lies fully above the no-drift CI
    separated = bool(min(r.rmse_mean - r.rmse_ci95 for r in drift) > base.rmse_mean + base.rmse_ci95)
    ratio = [r2(r.rmse_mean) / r2(base.rmse_mean) for r in drift]   # from shown values
    got, tot = events(drift)
    rmses = ", ".join(f"{fmt(r.rmse_mean, r.rmse_ci95, r.n_seeds)} ({k})" for r, k in zip(drift, DRIFT_STREAMS))
    per = ", ".join(f"{k} {r.recovered_events}" for r, k in zip(drift, DRIFT_STREAMS))
    return dict(
        id="vanilla", supported=separated,
        headline=("Vanilla TabPFN degrades under drift." if separated
                  else "Vanilla TabPFN: no clear degradation under drift in these results."),
        body=(f"Mean RMSE is {fmt(base.rmse_mean, base.rmse_ci95, base.n_seeds)} with no drift, but {rmses}: "
              f"{min(ratio):.1f}× to {max(ratio):.1f}× the no-drift error"
              f"{' (the 95% CIs do not overlap)' if separated else ''}. It returned to its pre-drift error "
              f"after only {got} of {tot} drifts ({per})."),
        evidence=[evidence(drift + [base], "Synthetic streams")],
        values=dict(ratio_min=min(ratio), ratio_max=max(ratio), recovered=got, drifts=tot,
                    none_rmse=float(base.rmse_mean), n_seeds=int(base.n_seeds)),
    )


def finding_validation(s):
    drift, reg = [get(s, k, VAL) for k in DRIFT_STREAMS], get(s, "none", VAL)
    if reg is None or any(r is None for r in drift):
        return None
    got, tot = events(drift)
    lo, hi = reg.regret_pct_mean - reg.regret_pct_ci95, reg.regret_pct_mean + reg.regret_pct_ci95
    rec = "recovers after every synthetic drift" if got == tot else f"recovers after {got} of {tot} synthetic drifts"
    if hi < 0:
        cost, cost_head = "lower error than vanilla", "and is better than vanilla when there is no drift"
    elif lo <= 0:
        cost, cost_head = "the 95% CI includes 0, so no measurable cost", "without a measurable cost when there is no drift"
    else:
        cost, cost_head = "a real cost", f"but costs {reg.regret_pct_mean:.1f}% when there is no drift"
    times = ", ".join(f"{k} {fmt(r.recovery_mean, r.recovery_ci95, r.n_seeds, 1)}" for r, k in zip(drift, DRIFT_STREAMS))
    body = (f"Recovered {got} of {tot} drifts; mean recovery time in blocks: {times}. With no drift its RMSE "
            f"differs from vanilla by {fmt(reg.regret_pct_mean, reg.regret_pct_ci95, reg.n_seeds, 1)}% ({cost}).")
    ev = [evidence(drift + [reg], "Synthetic streams")]
    real = [(ds, get(s, ds, VAL), get(s, ds, VAN)) for ds in REAL_SETS]
    real = [(ds, v, p) for ds, v, p in real if v is not None and p is not None]
    real_text = ""
    if real:
        real_text = ("On real data, validation vs vanilla mean RMSE: " + "; ".join(
            f"{ds} {fmt(v.rmse_mean, v.rmse_ci95, v.n_seeds)} vs {fmt(p.rmse_mean, p.rmse_ci95, p.n_seeds)} "
            f"({pct(v.rmse_mean, p.rmse_mean):+.1f}%, {compare(v, p)})" for ds, v, p in real) + ".")
        ev.append(evidence([r for _, v, p in real for r in (v, p)], "Real data"))
    return dict(
        id="validation", supported=got == tot and lo <= 0,
        headline=f"Validation-guided selection {rec} {cost_head}.",
        body=body, real=real_text,
        caveat=("Low-drift cases are not shown by our data: real drift points are unknown, so we cannot "
                "say which real dataset has little drift."),
        evidence=ev,
        values=dict(recovered=got, drifts=tot, regret=float(reg.regret_pct_mean), regret_ci=float(reg.regret_pct_ci95)),
    )


def finding_adwin(s, blocks, ablation_done=False):
    """blocks: {dataset: number of scored blocks} (from the curves)."""
    real = [(ds, get(s, ds, ADW), get(s, ds, VAN)) for ds in REAL_SETS]
    real = [(ds, a, p) for ds, a, p in real if a is not None and p is not None]
    syn = [(k, get(s, k, ADW)) for k in DRIFT_STREAMS + ("none",)]
    syn = [(k, r) for k, r in syn if r is not None]
    if not real or not syn:
        return None
    # supported only if real data has far more flags than any synthetic stream AND a higher error
    over = min(a.drift_flags_mean for _, a, _ in real) > 3 * max(r.drift_flags_mean for _, r in syn)
    worse = [ds for ds, a, p in real if a.rmse_mean > p.rmse_mean]
    flags_real = ", ".join(f"{a.drift_flags_mean:.0f} times in {blocks[ds]} blocks ({ds})" for ds, a, _ in real)
    n_true = {k: len(drift_blocks_for(k)) for k, _ in syn}
    flags_syn = ", ".join(f"{r.drift_flags_mean:.1f} ({k}, {n_true[k]} true drift{'s' if n_true[k] != 1 else ''})"
                          for k, r in syn)
    rmse = "; ".join(f"{ds} {fmt(a.rmse_mean, a.rmse_ci95, a.n_seeds)} vs {fmt(p.rmse_mean, p.rmse_ci95, p.n_seeds)} "
                     f"({pct(a.rmse_mean, p.rmse_mean):+.1f}%, {compare(a, p)})" for ds, a, p in real)
    n_syn = int(min(r.n_seeds for _, r in syn))
    return dict(
        id="adwin", supported=bool(over and worse),
        headline=(("ADWIN over-triggers on real data at its default setting (see the sensitivity check)."
                   if ablation_done else
                   "ADWIN over-triggers on real data (an observation; the delta ablation is still to come).")
                  if over and worse else "ADWIN on real data: no clear over-triggering in these results."),
        body=(f"It flagged drift on average {flags_real}, against {flags_syn} flags per run on the synthetic "
              f"streams. Each flag throws away older history. Its mean RMSE vs vanilla: {rmse}."),
        caveat=("The true drift points in real data are unknown, so 'too many flags' is judged from this "
                "higher error, not from a known answer."),
        evidence=[evidence([a for _, a, _ in real], "Real data"),
                  dict(text=f"Synthetic flag counts: mean over {n_syn} seeds. Flag counts have no confidence interval.",
                       seeds=[n_syn], has_ci=False)],
        values={ds: dict(flags=float(a.drift_flags_mean), blocks=int(blocks[ds]),
                         vs_vanilla_pct=float(pct(a.rmse_mean, p.rmse_mean))) for ds, a, p in real},
    )


def robustness(s):
    """For each real dataset: every TabPFN policy's % gap to the best TabPFN policy there."""
    out = {}
    for ds in REAL_SETS:
        g = s[(s.stream == ds) & s.method.isin(TABPFN)]
        if g.empty:
            continue
        best = g.loc[g.rmse_mean.idxmin()]
        out[ds] = [dict(method=r.method, rmse_mean=float(r.rmse_mean),
                        rmse_ci95=None if pd.isna(r.rmse_ci95) else float(r.rmse_ci95),
                        n_seeds=int(r.n_seeds), gap_pct=float(pct(r.rmse_mean, best.rmse_mean)),
                        best=r.method == best.method)
                   for _, r in g.sort_values("rmse_mean").iterrows()]
    return out


def all_findings(s, blocks, ablation_done=False):
    return [f for f in (finding_vanilla(s), finding_validation(s), finding_adwin(s, blocks, ablation_done)) if f]


# ---------------------------------------------------------------- ADWIN sensitivity (delta ablation)
def delta_tag(delta):
    """0.0001 -> '1e-4' (same spelling as the ablation file names)."""
    mant, exp = f"{delta:.0e}".split("e")
    return f"{mant}e{int(exp)}"


def adwin_sensitivity(ab, s):
    """ADWIN sensitivity block, from results_ablation/ablation_summary.csv (ab) and summary.csv (s).
    Every sentence is built from the numbers; no 'best' delta is named. None if there is no ablation
    or if it was made with the stand-in model (never shown as a result)."""
    if ab is None or ab.empty or (ab.model != "TabPFN").any():
        return None
    ab = ab.sort_values(["dataset", "delta"], ascending=[True, False])
    default = ab[ab["default"]].set_index("dataset")
    d_default = float(default.delta.iloc[0])
    show = lambda d: f"{d:g} (default)" if d == d_default else delta_tag(d)
    tags = lambda g: ", ".join(show(d) for d in g)
    summary, over_sets = [], []
    points, table = [], []

    for ds, g in ab.groupby("dataset", sort=False):
        for r in g.itertuples():
            table.append(dict(dataset=ds, delta=show(r.delta), default=bool(r.default),
                              seeds=int(r.seeds_run), rmse=float(r.rmse_mean),
                              rmse_ci95=None if pd.isna(r.rmse_ci95) else float(r.rmse_ci95),
                              flags=float(r.flags_mean), blocks=int(r.blocks),
                              vanilla=float(r.vanilla_rmse),
                              # paired by seed, as in make_ablation (mean of per-seed %), with its CI
                              vs_vanilla_pct=float(r.vs_vanilla_pct),
                              vs_vanilla_ci95=None if pd.isna(r.vs_vanilla_ci95) else float(r.vs_vanilla_ci95)))

    # real data: does the default over-trigger, and what does a stricter delta do?
    syn_flags = max(default.loc[k, "flags_mean"] for k in default.index if k not in REAL_SETS) if         any(k not in REAL_SETS for k in default.index) else None
    for ds in [d for d in REAL_SETS if d in default.index]:
        g = ab[ab.dataset == ds]
        d0 = default.loc[ds]
        stricter = g[~g["default"]]
        flags = " → ".join(f"{f:.0f}" for f in g.flags_mean)
        rmses = " → ".join(f"{r2(x):.2f}" for x in g.rmse_mean)
        below = g[[r2(a) < r2(b) for a, b in zip(g.rmse_mean, g.vanilla_rmse)]]
        n = int(g.seeds_run.max())
        head = (f"{ds} ({n} seed{'s' if n > 1 else ''}, {'no CI' if n == 1 else '95% CI'}): drift flags "
                f"{flags} in {int(d0.blocks)} blocks and mean RMSE {rmses} as δ goes {tags(g.delta)}; "
                f"vanilla on the same seed {r2(d0.vanilla_rmse):.2f}.")
        over = syn_flags is not None and d0.flags_mean > 3 * syn_flags
        helps = len(stricter) > 0 and all(r2(x) < r2(d0.rmse_mean) for x in stricter.rmse_mean)
        if over:
            head += f" At the default δ it flags far more often than on any synthetic stream (at most {syn_flags:.1f} per run)."
        if helps:
            head += " Every stricter δ had a lower error than the default."
        if below.empty:
            p = [pct(a, b) for a, b in zip(g.rmse_mean, g.vanilla_rmse)]
            head += f" It is still worse than vanilla at every δ ({min(p):+.1f}% to {max(p):+.1f}%)."
            summary.append(f"on {ds} it is still worse than vanilla at every δ")
        else:
            head += f" It is below vanilla only at δ = {tags(below.delta)}."
            if helps:
                summary.append(f"a stricter δ helps on {ds} (below vanilla at δ = {tags(below.delta)})")
        if over:
            over_sets.append(ds)
        points.append(head)
    if over_sets:
        summary.insert(0, "the default over-triggers on real data (" + ", ".join(over_sets) + ")")

    # synthetic streams: same idea, with 95% CIs
    for ds in [d for d in ("sudden", "gradual") if d in default.index]:
        g = ab[ab.dataset == ds]
        n = int(g.seeds_run.max())
        rm = ", ".join(fmt(m, c, n) for m, c in zip(g.rmse_mean, g.rmse_ci95))
        fl = ", ".join(f"{f:.1f}" for f in g.flags_mean)
        k = len(drift_blocks_for(ds))
        better = all(r2(a) < r2(b) for a, b in zip(g.rmse_mean, g.vanilla_rmse))
        points.append(f"{ds} (synthetic, {n} seeds, 95% CI): mean RMSE {rm} and {fl} flags per run "
                      f"({k} true drift{'s' if k != 1 else ''}) as δ goes {tags(g.delta)}"
                      f"{'; below vanilla (' + f'{r2(g.vanilla_rmse.iloc[0]):.2f}' + ') at every δ' if better else ''}.")

    return dict(
        title="ADWIN sensitivity: how strict should the drift detector be?",
        intro=("δ is ADWIN's significance level: a smaller δ needs more evidence before it flags a drift, "
               f"so it flags less often. Default δ = {d_default:g}."),
        summary=("In short: " + "; ".join(summary) + ".") if summary else "",
        points=points,
        caveat=("The δ values were chosen after looking at the same test streams, and the real datasets were run "
                "with 1 seed (no confidence interval). This is a sensitivity result, not a tuned method, and we do not "
                "name a best δ."),
        table=table,
    )


def adwin_limitation(s, ab):
    """Limitations line about ADWIN, from the data (replaces 'weak on gradual drift, unexplained')."""
    parts = []
    g, v = get(s, "gradual", ADW), get(s, "gradual", VAN)
    if g is not None and v is not None:
        k = len(drift_blocks_for("gradual"))
        parts.append(f"On gradual drift ADWIN flags {g.drift_flags_mean:.1f} times per run for {k} true drift "
                     f"and still {'beats' if r2(g.rmse_mean) < r2(v.rmse_mean) else 'does not beat'} vanilla "
                     f"(mean RMSE {fmt(g.rmse_mean, g.rmse_ci95, g.n_seeds)} vs {fmt(v.rmse_mean, v.rmse_ci95, v.n_seeds)}).")
    if ab is not None and not ab.empty and (ab.model == "TabPFN").all():
        for ds in [d for d in REAL_SETS if d in set(ab.dataset)]:
            e = ab[ab.dataset == ds]
            if all(r2(a) >= r2(b) for a, b in zip(e.rmse_mean, e.vanilla_rmse)):
                parts.append(f"On {ds} it is worse than vanilla at every δ tested ({len(e)} values, 1 seed); "
                             "the cause is not explained.")
    return " ".join(parts)
