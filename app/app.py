"""
Presentation app: TabPFN under concept drift.
    python app/app.py        then open http://127.0.0.1:7860

Everything in the main tabs is a REPLAY of saved results in results/ (nothing is
computed live). Mock (FAKE) data from results/mock/ is only shown in the developer
section at the bottom, with a warning in every title.

Tabs:
1. What is this?  plain-English explainer + diagram
2. Examples       6 presets: error over time + results table for one stream/dataset
3. Results        summary tables (mean +/- 95% CI) + the PNG charts from results/figures/
4. Limitations    what the results can and cannot show

Developer-only MOCK section: hidden unless the environment variable SHOW_MOCK=1 is set.
"""
import os
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # so `common`, `data`, `eval` can be imported

import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from common.contract import N_BLOCKS
from data.streams import drift_blocks_for, GRADUAL_START, GRADUAL_END
from eval.metrics import load_results
from eval.make_report import per_seed_table, summary
from eval import findings as FINDINGS

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MOCK = RESULTS / "mock"
ROLL = 5  # real data: rolling RMSE over the last 5 blocks (= 1000 rows)
SHOW_MOCK = os.environ.get("SHOW_MOCK") == "1"   # developer-only mock section

# ---------------------------------------------------------------- names and colours
# Fixed order and colour per method, the same in every view ("colour follows the method").
# TabPFN policies: reference palette slots 1-6. Baselines: grey, told apart by dash style.
# Every line also has its own dash style and a name label at its end, so colour is never
# the only way to tell lines apart.
METHODS = {
    # key in results     label                 colour     dash
    "P0_vanilla":     ("P0 vanilla",          "#2a78d6", "solid"),
    "P1_window":      ("P1 sliding window",   "#eb6834", "solid"),
    "P2_recency":     ("P2 recency",          "#1baf7a", "dash"),
    "P3_drift":       ("P3 ADWIN",            "#eda100", "solid"),
    "P4_valguided":   ("P4 validation",       "#6250d6", "dot"),
    "kNN_context":    ("kNN context",         "#008300", "dashdot"),
    "XGBoost":        ("XGBoost",             "#52514e", "dash"),
    "MLP":            ("MLP",                 "#8a8984", "dot"),
    "AdaptiveForest": ("Adaptive forest",     "#52514e", "longdashdot"),
}
TABPFN = ["P0_vanilla", "P1_window", "P2_recency", "P3_drift", "P4_valguided", "kNN_context"]
BASELINES = ["XGBoost", "MLP", "AdaptiveForest"]
LABEL_TO_KEY = {v[0]: k for k, v in METHODS.items()}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# --------------------------------------------------------------------------- presets
PRESETS = {
    "Sudden drift (synthetic)": dict(
        stream="sudden", real=False, unit="",
        story="At block 30 the rule that links the inputs to the target changes all at once."),
    "Gradual drift (synthetic)": dict(
        stream="gradual", real=False, unit="",
        story="Between blocks 20 and 40 the old rule slowly fades into a new one."),
    "Recurring drift (synthetic)": dict(
        stream="recurring", real=False, unit="",
        story="The rule flips every 15 blocks (15, 30, 45), so old concepts come back."),
    "Bike Sharing (real)": dict(
        stream="bike", real=True, unit="bikes per hour",
        story="Hourly bike rentals in Washington D.C. (2011-2012), predicted from calendar and weather."),
    "Appliances Energy (real)": dict(
        stream="energy", real=True, unit="Wh per 10 min",
        story="Appliance energy use of one house in Belgium (Jan-May 2016), from room and weather sensors."),
    "Beijing PM2.5 (real)": dict(
        stream="pm25", real=True, unit="ug/m3",
        story="Hourly air pollution in Beijing (2010-2014), predicted from the weather."),
}
DEFAULT_METHODS = ["P0 vanilla", "P1 sliding window", "P3 ADWIN", "P4 validation", "XGBoost", "Adaptive forest"]
REPLAY = "**REPLAY of saved results** (read from `results/`, nothing is computed live)."


# ---------------------------------------------------------------------- loading data
def load_all(folder):
    """Per-block results + per-seed table + summary (mean +/- 95% CI), same code as make_report."""
    res = load_results(folder)
    seeds = per_seed_table(res)
    return res, seeds, summary(seeds)


try:
    RES, SEEDS, SUMMARY = load_all(RESULTS)
    LOAD_ERROR = ""
except FileNotFoundError as e:   # no results at all: the app still starts and says so
    RES = SEEDS = SUMMARY = pd.DataFrame(columns=["stream", "method", "seed", "block", "rmse",
                                                  "drift_flag", "kind", "n_seeds"])
    LOAD_ERROR = f"No results found: {e}"


def seed_count(stream, keys):
    """Number of seeds per method, e.g. {'P0_vanilla': 1, 'XGBoost': 3}."""
    d = RES[(RES.stream == stream) & RES.method.isin(keys)]
    return d.groupby("method").seed.nunique().to_dict()


def seeds_text(stream):
    """One line saying how many seeds each group of methods has. Warns about 1 seed."""
    parts = []
    for name, keys in (("TabPFN policies", TABPFN), ("baselines", BASELINES)):
        n = sorted(set(seed_count(stream, keys).values()))
        if not n:
            parts.append(f"{name}: **no results yet**")
        else:
            txt = " or ".join(str(x) for x in n)
            parts.append(f"{name}: {txt} seed{'s' if n != [1] else ''}")
    line = "Seeds: " + "; ".join(parts) + "."
    if 1 in set(seed_count(stream, TABPFN + BASELINES).values()):
        line += (" **Methods with 1 seed are a single run: no confidence interval, "
                 "so no claim that they are better or worse.**")
    return line


# ---------------------------------------------------------------------- Examples tab
def error_curve(d, real):
    """Error over time for one method: mean over seeds of the per-block RMSE.
    Real data: rolling RMSE over the last ROLL blocks, per seed first, then averaged."""
    wide = d.pivot(index="block", columns="seed", values="rmse").sort_index()
    if real:
        # exact RMSE over the last ROLL blocks (all blocks have 200 rows): sqrt(mean of rmse^2)
        wide = np.sqrt((wide ** 2).rolling(ROLL).mean())
    return wide.mean(axis=1).dropna()


PLOT_PX, MARGIN_T, MARGIN_B = 340, 60, 170   # plot area height and margins, in pixels
LABEL_PX = 15                                 # height of one end-of-line label, in pixels


def example_plot(stream, real, unit, keys, show_flags):
    fig = go.Figure()
    ends = []   # (y, x, label) of each line's last point, for the direct labels
    ytop = 0.0  # highest point drawn, to size the label spacing
    for k in keys:
        d = RES[(RES.stream == stream) & (RES.method == k)]
        if d.empty:
            continue
        label, colour, dash = METHODS[k]
        n = d.seed.nunique()
        curve = error_curve(d, real)
        ytop = max(ytop, curve.max())
        name = f"{label} ({n} seed{'s' if n > 1 else ''})"
        fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=name,
                                 line=dict(color=colour, dash=dash, width=2),
                                 hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>"))
        ends.append([curve.values[-1], curve.index[-1], label, curve.values[-1]])  # label y, x, text, line y
        if show_flags and k == "P3_drift":
            fl = sorted(set(d.loc[d.drift_flag == 1, "block"]) & set(curve.index))
            if fl:
                fig.add_trace(go.Scatter(
                    x=fl, y=curve.loc[fl], mode="markers", name="P3 ADWIN drift flag (any seed)",
                    marker=dict(symbol="triangle-down", size=10, color=colour,
                                line=dict(color=SURFACE, width=2)),
                    hovertemplate="ADWIN flagged drift at block %{x}<extra></extra>"))

    # Direct label at the end of each line (text in ink, not the line colour).
    # Labels are spread so they never overlap: the y-axis range and the plot height are
    # fixed, so we know how many data units one text line (LABEL_PX pixels) takes.
    ymax = 1.08 * ytop if ytop > 0 else 1.0
    if ends:
        gap = LABEL_PX / PLOT_PX * ymax              # one text line, in data units
        ends.sort()
        for i in range(1, len(ends)):                # pass 1: push labels up
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        ends[-1][0] = min(ends[-1][0], ymax - gap / 2)
        for i in range(len(ends) - 2, -1, -1):       # pass 2: keep the top one inside the plot
            ends[i][0] = min(ends[i][0], ends[i + 1][0] - gap)
        for y, x, label, y_line in ends:
            # the arrow is a thin leader line from the label to the end of its own line
            fig.add_annotation(x=x, y=y_line, text=label, xanchor="left", showarrow=True, arrowhead=0,
                               arrowwidth=1, arrowcolor="#b5b4ae", ax=28,
                               ay=-(y - y_line) / ymax * PLOT_PX, font=dict(size=11, color=INK2))

    # True drift points: only for synthetic streams, where we made the drift ourselves.
    if not real:
        if stream == "gradual":
            fig.add_vrect(x0=GRADUAL_START, x1=GRADUAL_END, fillcolor=GRID, opacity=0.5, line_width=0, layer="below",
                          annotation_text="gradual change (true)", annotation_position="top left")
        else:
            for b in drift_blocks_for(stream):
                fig.add_vline(x=b, line_dash="dash", line_color=INK2,
                              annotation_text=f"true drift ({b})", annotation_position="top")

    if real:
        title = f"Rolling RMSE over the last {ROLL} blocks - no drift line: real drift points are unknown"
        ytitle = f"rolling RMSE ({unit})"
    else:
        title = "Block RMSE (mean over seeds) - dashed lines / shading = true drift"
        ytitle = "RMSE per block"
    fig.update_layout(
        title=dict(text=f"REPLAY - {title}", font=dict(size=14, color=INK)),
        xaxis=dict(title="block (200 rows each)", gridcolor=GRID, zeroline=False),
        yaxis=dict(title=ytitle, gridcolor=GRID, zeroline=False, range=[0, ymax]),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK2),
        hovermode="x unified", height=PLOT_PX + MARGIN_T + MARGIN_B,
        margin=dict(r=160, t=MARGIN_T, b=MARGIN_B),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
    )
    return fig


def fmt(mean, ci, n, digits=2):
    """'50.22 +/- 1.30', or '50.22 (1 seed, no CI)', or '' when there is no value."""
    if pd.isna(mean):
        return ""
    if n < 2 or pd.isna(ci):
        return f"{mean:.{digits}f} (1 seed, no CI)"
    return f"{mean:.{digits}f} +/- {ci:.{digits}f}"


def results_table(stream, real):
    """One row per method for this stream, best mean RMSE first."""
    s = SUMMARY[SUMMARY.stream == stream]
    rows = []
    for _, r in s.iterrows():
        row = {"Method": METHODS[r.method][0] if r.method in METHODS else r.method,
               "Kind": "TabPFN policy" if r.method in TABPFN else "baseline",
               "Seeds": int(r.n_seeds),
               "Mean RMSE (95% CI)": fmt(r.rmse_mean, r.rmse_ci95, r.n_seeds)}
        if not real:
            rec = fmt(r.recovery_mean, r.recovery_ci95, r.n_seeds, 1)
            got, total = str(r.recovered_events).split("/") if "/" in str(r.recovered_events) else (0, 0)
            if got != total:   # some drifts never recovered: the number is the cap, not a real time
                rec += " CAPPED (not all recovered)"
            row["Recovery time, blocks (95% CI)"] = rec
            row["Drifts recovered"] = r.recovered_events
        if r.method == "P3_drift" or r.drift_flags_mean > 0:
            row["Drift flags per run"] = f"{r.drift_flags_mean:.1f}"
        else:
            row["Drift flags per run"] = "-"
        rows.append(row)
    return pd.DataFrame(rows)


def show_example(preset, method_labels, show_flags):
    p = PRESETS[preset]
    stream, real = p["stream"], p["real"]
    keys = [LABEL_TO_KEY[m] for m in method_labels]
    have = set(RES.method[RES.stream == stream])

    notes = []
    if not have & set(TABPFN):
        notes.append("**TabPFN results for this dataset are not available yet: showing baselines only.**")
    missing = [METHODS[k][0] for k in keys if k not in have]
    if missing and have & set(TABPFN):
        notes.append(f"No saved results for: {', '.join(missing)}.")
    if not [k for k in keys if k in have]:
        notes.append("None of the ticked methods has results here: tick another method.")
    if real:
        notes.append(f"Real data: the line is RMSE over the last {ROLL} blocks, so it starts at block {ROLL}. "
                     "Recovery time is not computed because the true drift points are unknown.")
    else:
        notes.append("Synthetic data: we created the drift, so the true drift points are shown.")
    kind = "real dataset" if real else "synthetic stream"
    header = "\n\n".join([f"### {preset}", f"*{p['story']}* ({kind})", REPLAY, seeds_text(stream)] + notes)
    return header, example_plot(stream, real, p["unit"], keys, show_flags), results_table(stream, real)


# ----------------------------------------------------------------------- Results tab
RESULT_STREAMS = {
    "Sudden drift (synthetic)": "sudden", "Gradual drift (synthetic)": "gradual",
    "Recurring drift (synthetic)": "recurring", "No drift (synthetic)": "none",
    "Bike Sharing (real)": "bike", "Appliances Energy (real)": "energy", "Beijing PM2.5 (real)": "pm25",
}


def figure_path(name):
    f = FIGURES / name
    return str(f) if f.exists() else None


def show_results(choice):
    stream = RESULT_STREAMS[choice]
    real = stream in ("bike", "energy", "pm25")
    table = results_table(stream, real)
    if stream == "none":   # no drift: show the regret vs vanilla instead of recovery
        s = SUMMARY[SUMMARY.stream == "none"].set_index("method")
        table["No-drift regret vs vanilla, % (95% CI)"] = [
            fmt(s.loc[LABEL_TO_KEY.get(m, m), "regret_pct_mean"], s.loc[LABEL_TO_KEY.get(m, m), "regret_pct_ci95"],
                s.loc[LABEL_TO_KEY.get(m, m), "n_seeds"], 1) or "reference" for m in table.Method]
        table = table.drop(columns=["Recovery time, blocks (95% CI)", "Drifts recovered"])
    second = (figure_path("regret_none.png") if stream == "none"
              else None if real else figure_path(f"recovery_{stream}.png"))
    text = [REPLAY, seeds_text(stream)]
    if table.empty:
        text.append("**No results for this stream yet.**")
    if real and not (set(SUMMARY.method[SUMMARY.stream == stream]) & set(TABPFN)):
        text.append("**TabPFN results for this dataset are not available yet: baselines only.**")
    text.append("95% CI = t-interval across seeds. With 3 seeds it is wide and rough; with 1 seed there is none.")
    return ("\n\n".join(text), table, figure_path(f"rmse_{stream}.png"),
            gr.update(value=second, visible=second is not None))


# ------------------------------------------------------------------- Limitations tab
def adwin_flags_text():
    """'bike 38 of 85 blocks, energy 56 of 97 blocks' from the saved results (not typed in)."""
    out = []
    for ds in ("bike", "energy", "pm25"):
        d = RES[(RES.stream == ds) & (RES.method == "P3_drift")]
        if not d.empty:
            per_seed = d.groupby("seed").drift_flag.sum()
            out.append(f"{ds}: {per_seed.mean():.0f} of {d.groupby('seed').block.nunique().iloc[0]} blocks")
    return ", ".join(out) if out else "no real-data ADWIN results yet"


def real_seeds_text():
    """e.g. 'TabPFN policies: bike 3 seeds each; pm25 3 seeds (P4 validation: 1 seed).'"""
    out = []
    for ds in ("bike", "energy", "pm25"):
        n = seed_count(ds, TABPFN)
        if not n:
            out.append(f"{ds}: no TabPFN results yet (baselines only)")
            continue
        most = max(set(n.values()), key=list(n.values()).count)
        odd = [f"{METHODS[k][0]}: {v} seed{'s' if v > 1 else ''}" for k, v in n.items() if v != most]
        out.append(f"{ds}: {most} seed{'s' if most > 1 else ''} per policy" + (f" ({', '.join(odd)})" if odd else ""))
    return "TabPFN policies on real data: " + "; ".join(out) + "."


def adwin_rank_text():
    """Where ADWIN ranks among the TabPFN policies on each real dataset (mean RMSE)."""
    worst, other = [], []
    for ds in ("bike", "energy", "pm25"):
        s = SUMMARY[(SUMMARY.stream == ds) & SUMMARY.method.isin(TABPFN)]
        if "P3_drift" not in set(s.method):
            continue
        (worst if s.loc[s.rmse_mean.idxmax(), "method"] == "P3_drift" else other).append(ds)
    txt = f"It has the highest mean RMSE of the TabPFN policies on {', '.join(worst)}." if worst else ""
    return txt + (f" It is not the worst on {', '.join(other)}." if other else "")


ABLATION_CSV = ROOT / "results_ablation" / "ablation_summary.csv"


def read_ablation():
    """ADWIN delta ablation summary (made by python -m eval.make_ablation), or None if not run yet."""
    return pd.read_csv(ABLATION_CSV) if ABLATION_CSV.exists() else None


def adwin_limitation_text():
    """What the data shows about ADWIN on gradual drift and on energy (eval/findings.py)."""
    return FINDINGS.adwin_limitation(SUMMARY, read_ablation())


def adwin_sensitivity_md():
    """'ADWIN sensitivity' block: only from results_ablation/; empty string if there is none."""
    a = FINDINGS.adwin_sensitivity(read_ablation(), SUMMARY)
    if a is None:
        return ""
    rows = ["| Dataset | δ | Seeds | Mean RMSE | Drift flags per run | vs vanilla (same seeds) |",
            "|---|---|---|---|---|---|"]
    for r in a["table"]:
        vs = (f"{r['vs_vanilla_pct']:+.1f}% (1 seed, no CI)" if r["seeds"] < 2 or r["vs_vanilla_ci95"] is None
              else f"{r['vs_vanilla_pct']:+.1f}% +/- {r['vs_vanilla_ci95']:.1f}")
        rows.append(f"| {r['dataset']} | {r['delta']} | {r['seeds']} | "
                    f"{fmt(r['rmse'], r['rmse_ci95'], r['seeds'])} | {r['flags']:.1f} of {r['blocks']} blocks | {vs} |")
    return "\n\n".join([f"## {a['title']}", "*REPLAY of saved results from `results_ablation/`.*", a["intro"],
                        f"**{a['summary']}**", "\n".join("- " + p for p in a["points"]), "\n".join(rows),
                        f"*{a['caveat']}*"])


def limitations_md():
    return f"""
## Limitations: what these results can and cannot show

1. **Few seeds on synthetic data.** Every synthetic result is the mean of **3 seeds**. The 95% intervals
   are wide and rough; small differences (for example sliding window vs validation) are within noise.
   We make no confidence claims from 3 seeds.
2. **Few seeds for TabPFN on real data.** {real_seeds_text()} With 3 seeds the 95% intervals are rough;
   a method with 1 seed has no interval at all. XGBoost gives the same result on every seed (it uses no
   random sampling), so its interval of 0 means "deterministic", not "precise".
3. **Untuned baselines with a 5-block window.** XGBoost and MLP are refit before every block on only the
   **last 5 blocks (1000 rows)**, with default-like settings; TabPFN policies may use up to 2000 rows. The
   adaptive forest learns row by row. None of them was tuned, so they are reference points, not the best
   possible versions of these models.
4. **ADWIN over-triggers on real data.** ADWIN flagged drift in many blocks ({adwin_flags_text()}).
   Each flag throws away older history, so it ends up with little context. {adwin_rank_text()}
   {adwin_limitation_text()}
5. **Synthetic drift is stronger than real drift.** In the synthetic streams the whole rule changes at
   once and the error jumps about 6x (vanilla ~1.0 -> ~6.5 at block 30). In the real data the changes are
   smaller and mixed with normal day/season patterns, and the true drift points are unknown, so we can only
   compare error over time, not recovery time. Results on synthetic streams do not prove anything about
   real data.
6. **Policy settings were not tuned** (window 5 blocks, half-life 5, validation on the last 2 blocks).
"""


# ------------------------------------------------------------- Results: key findings
# Three claims. Every number comes from results/summary.csv (written by make_report); the
# wording of each headline depends on the numbers, so a claim is only made if the data
# supports it. Each claim says how many seeds it rests on and whether it has a 95% CI.
SUMMARY_CSV = RESULTS / "summary.csv"
DRIFT_STREAMS = ("sudden", "gradual", "recurring")
REAL_SETS = ("bike", "energy", "pm25")
VAN, VAL, ADW = "P0_vanilla", "P4_valguided", "P3_drift"


def read_summary_csv():
    """summary.csv + a warning if it is missing or older than the result CSVs."""
    if not SUMMARY_CSV.exists():
        return None, "**results/summary.csv is missing: run `python -m eval.make_report`.**"
    s = pd.read_csv(SUMMARY_CSV)
    m = s.merge(SUMMARY[["stream", "method", "n_seeds", "rmse_mean"]], on=["stream", "method"],
                how="outer", suffixes=("", "_now"))
    stale = (m[["rmse_mean", "rmse_mean_now"]].isna().any().any()
             or not np.allclose(m.rmse_mean, m.rmse_mean_now) or not (m.n_seeds == m.n_seeds_now).all())
    warn = "**Warning: results/summary.csv is out of date: run `python -m eval.make_report`.**" if stale else ""
    return s, warn


def get(s, stream, method):
    """One summary row, or None if that method has no results on that stream."""
    r = s[(s.stream == stream) & (s.method == method)]
    return None if r.empty else r.iloc[0]


def evidence(rows, what):
    """e.g. '*Synthetic streams: 3 seeds per method, with 95% CI (rough with 3 seeds).*'"""
    n = sorted({int(r.n_seeds) for r in rows})
    seeds = " or ".join(map(str, n)) + (" seed" if n == [1] else " seeds")
    has = [pd.notna(r.rmse_ci95) for r in rows]
    if all(has):
        ci = "with 95% CI (rough with 3 seeds)"
    elif not any(has):
        ci = "no confidence interval (a single run)"
    else:
        ci = "95% CI where there are 3 seeds; 1-seed values have no CI"
    return f"*{what}: {seeds} per method, {ci}.*"


def events(rows):
    """'3/3' strings -> (recovered, total) summed over the rows."""
    pairs = [tuple(int(x) for x in str(r.recovered_events).split("/")) for r in rows]
    return sum(p[0] for p in pairs), sum(p[1] for p in pairs)


def pct(a, b):
    return 100 * (a - b) / b


def compare(a, b):
    """Plain verdict for method a vs method b (lower RMSE is better), based on the 95% CIs."""
    if pd.isna(a.rmse_ci95) or pd.isna(b.rmse_ci95):
        return "no conclusion: a single run has no CI"
    if a.rmse_mean + a.rmse_ci95 < b.rmse_mean - b.rmse_ci95:
        return "lower, CIs do not overlap"
    if a.rmse_mean - a.rmse_ci95 > b.rmse_mean + b.rmse_ci95:
        return "higher, CIs do not overlap"
    return "CIs overlap: no clear difference"


def finding_vanilla(s):
    base, drift = get(s, "none", VAN), [get(s, k, VAN) for k in DRIFT_STREAMS]
    if base is None or any(r is None for r in drift):
        return "**1. Vanilla under drift:** not enough synthetic results yet."
    # supported only if every drift stream's CI lies fully above the no-drift CI
    separated = min(r.rmse_mean - r.rmse_ci95 for r in drift) > base.rmse_mean + base.rmse_ci95
    ratio = [r.rmse_mean / base.rmse_mean for r in drift]
    got, tot = events(drift)
    head = ("Vanilla (P0) degrades under drift." if separated
            else "Vanilla (P0): no clear degradation under drift in these results.")
    rmses = ", ".join(f"{fmt(r.rmse_mean, r.rmse_ci95, r.n_seeds)} ({k})" for r, k in zip(drift, DRIFT_STREAMS))
    per = ", ".join(f"{k} {r.recovered_events}" for r, k in zip(drift, DRIFT_STREAMS))
    return (f"**1. {head}** Mean RMSE is {fmt(base.rmse_mean, base.rmse_ci95, base.n_seeds)} with no drift, "
            f"but {rmses}: {min(ratio):.1f}x to {max(ratio):.1f}x higher"
            f"{' (the 95% CIs do not overlap)' if separated else ''}. "
            f"It returned to its pre-drift error after only {got} of {tot} drifts ({per}).  \n"
            + evidence(drift + [base], "Synthetic streams"))


def finding_validation(s):
    drift, reg = [get(s, k, VAL) for k in DRIFT_STREAMS], get(s, "none", VAL)
    if reg is None or any(r is None for r in drift):
        return "**2. Validation-guided selection:** not enough synthetic results yet."
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
    text = (f"**2. Validation-guided selection (P4) {rec} {cost_head}.** Recovered {got} of {tot} drifts; "
            f"mean recovery time in blocks: {times}. With no drift its RMSE differs from vanilla by "
            f"{fmt(reg.regret_pct_mean, reg.regret_pct_ci95, reg.n_seeds, 1)}% ({cost}).  \n"
            + evidence(drift + [reg], "Synthetic streams"))
    real = [(ds, get(s, ds, VAL), get(s, ds, VAN)) for ds in REAL_SETS]
    real = [(ds, v, p) for ds, v, p in real if v is not None and p is not None]
    if real:
        diffs = "; ".join(f"{ds} {fmt(v.rmse_mean, v.rmse_ci95, v.n_seeds)} vs "
                          f"{fmt(p.rmse_mean, p.rmse_ci95, p.n_seeds)} ({pct(v.rmse_mean, p.rmse_mean):+.1f}%, "
                          f"{compare(v, p)})" for ds, v, p in real)
        text += (f"  \nOn real data, P4 vs vanilla mean RMSE: {diffs}. *The true drift points in the real data are unknown, so we cannot "
                 "tell whether a real dataset has drift at all, or how much.*  \n"
                 + evidence([r for _, v, p in real for r in (v, p)], "Real data"))
    return text


def finding_adwin(s):
    real = [(ds, get(s, ds, ADW), get(s, ds, VAN)) for ds in REAL_SETS]
    real = [(ds, a, p) for ds, a, p in real if a is not None and p is not None]
    syn = [(k, get(s, k, ADW)) for k in DRIFT_STREAMS + ("none",)]
    syn = [(k, r) for k, r in syn if r is not None]
    if not real or not syn:
        return "**3. ADWIN on real data:** no real-data ADWIN results yet."
    # supported only if real data has far more flags than any synthetic stream AND a higher error
    over = min(a.drift_flags_mean for _, a, _ in real) > 3 * max(r.drift_flags_mean for _, r in syn)
    worse = [ds for ds, a, p in real if a.rmse_mean > p.rmse_mean]
    blocks = {ds: RES[(RES.stream == ds) & (RES.method == ADW)].groupby("seed").block.nunique().iloc[0]
              for ds, _, _ in real}
    flags_real = ", ".join(f"{a.drift_flags_mean:.0f} times in {blocks[ds]} blocks ({ds})" for ds, a, _ in real)
    flags_real = "on average " + flags_real
    n_true = {k: len(drift_blocks_for(k)) for k, _ in syn}
    flags_syn = ", ".join(f"{r.drift_flags_mean:.1f} ({k}, {n_true[k]} true drift{'s' if n_true[k] != 1 else ''})"
                          for k, r in syn)
    rmse = "; ".join(f"{ds} {fmt(a.rmse_mean, a.rmse_ci95, a.n_seeds)} vs {fmt(p.rmse_mean, p.rmse_ci95, p.n_seeds)} "
                     f"({pct(a.rmse_mean, p.rmse_mean):+.1f}%, {compare(a, p)})" for ds, a, p in real)
    head = (("ADWIN (P3) over-triggers on real data at its default setting (see ADWIN sensitivity below)."
             if read_ablation() is not None else "ADWIN (P3) over-triggers on real data.") if over and worse
            else "ADWIN (P3) on real data: no clear over-triggering in these results.")
    return (f"**3. {head}** It flagged drift {flags_real}, against {flags_syn} flags per run on the "
            f"synthetic streams. Each flag throws away older history. Its mean RMSE vs vanilla: {rmse}"
            f". *The true drift points in real data "
            "are unknown, so 'too many flags' is judged from this higher error, not from a known answer.*  \n"
            + evidence([a for _, a, _ in real], "Real data")
            + f" *Synthetic flag counts: mean over {int(min(r.n_seeds for _, r in syn))} seeds. "
              "Flag counts have no confidence interval.*")


def key_findings_md():
    s, warn = read_summary_csv()
    if s is None:
        return f"## Key findings\n\n{warn}"
    parts = ["## Key findings",
             "*Computed from `results/summary.csv` when the app starts; no number is typed in. "
             "REPLAY of saved results.*"]
    if warn:
        parts.append(warn)
    parts += [finding_vanilla(s), finding_validation(s), finding_adwin(s)]
    return "\n\n".join(parts)


# ------------------------------------------------------------------ What is this? tab
EXPLAINER = """
## What is this project?

**TabPFN** is a pretrained model for tables. It is never trained on our data: instead, we give it a
*context* of labelled rows (up to 2000 here) and it predicts the new rows directly from them.

**Concept drift** means the relationship between inputs and the target changes over time, so old rows
can teach the model the wrong rule.

**Our question:** when data drifts, *which* past rows should TabPFN see? We compare six simple,
training-free ways to choose the context (each only uses data from the past):

| Policy | What it does |
|---|---|
| **P0 vanilla** | A random sample of up to 2000 rows from all of history. |
| **P1 sliding window** | Only the last 5 blocks (the most recent 1000 rows). |
| **P2 recency** | A random sample where newer blocks are more likely (weight halves every 5 blocks). |
| **P3 ADWIN** | Watches the prediction error; when it suddenly rises, forgets history before that point. |
| **P4 validation** | Tries 6 rules on the latest labelled blocks and switches to the best one if it is >5% better. |
| **kNN context** | Picks the 2000 past rows whose inputs look most like the new block. |

We compare them with three normal models that are retrained as data arrives: **XGBoost**, **MLP**,
and an **adaptive random forest**.

**How we test (test-then-train):** data arrives in blocks of 200 rows. For each block we first predict
it (using only earlier rows), measure the error (RMSE), and only then add its labels to the history.
"""


def box(text, sub, bg, fg="#0b0b0b"):
    return (f'<div style="background:{bg};color:{fg};border-radius:10px;padding:14px 16px;'
            f'min-width:150px;text-align:center;font-family:sans-serif">'
            f'<div style="font-weight:700;font-size:15px">{text}</div>'
            f'<div style="font-size:12px;margin-top:4px">{sub}</div></div>')


ARROW = '<div style="font-size:26px;color:#52514e;padding:0 8px">&rarr;</div>'
# Colours are fixed (not taken from the Gradio theme), so the contrast is the same in light
# and dark mode. Text contrast: white on #184f95 is 8.1:1; dark ink on the light boxes >= 14:1;
# the caption (#52514e on #fcfcfb) 7.7:1. The caption sits INSIDE the light panel on purpose.
DIAGRAM = (
    '<div style="padding:16px;background:#fcfcfb;border:1px solid #e4e3df;border-radius:12px">'
    '<div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px">'
    + box("Past labelled rows", "everything seen so far", "#e4e3df")
    + ARROW + box("Context policy", "picks up to 2000 rows<br>(P0-P4, kNN)", "#cde2fb")
    + ARROW + box("TabPFN", "pretrained, no training<br>+ inputs of the new block", "#184f95", "#ffffff")
    + ARROW + box("Prediction", "for the new block<br>-> error (RMSE)", "#e4e3df")
    + '</div><div style="font-family:sans-serif;font-size:13px;color:#52514e;margin-top:10px">'
    'After predicting, the true labels of the block are added to the past rows (test-then-train).'
    '</div></div>'
)


# ---------------------------------------------------------------- developer: mock data
def mock_plot(stream):
    """FAKE numbers from mock/make_mock_results.py, only for testing the app layout."""
    try:
        m = load_results(MOCK)
    except FileNotFoundError:
        raise gr.Error("No mock files in results/mock/ (run: python -m mock.make_mock_results)")
    fig = go.Figure()
    for k, g in m[m.stream == stream].groupby("method"):
        c = g.groupby("block").rmse.mean()
        label, colour, dash = METHODS.get(k, (k, INK2, "solid"))
        fig.add_trace(go.Scatter(x=c.index, y=c.values, name=f"{label} [MOCK]",
                                 line=dict(color=colour, dash=dash, width=2)))
    fig.update_layout(title="[MOCK - FAKE DATA, NOT RESULTS] invented numbers for testing the app",
                      height=380, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE)
    return fig


# ------------------------------------------------------------------------ the layout
with gr.Blocks(title="TabPFN under concept drift") as demo:
    gr.Markdown("# TabPFN under concept drift: which past rows should the model see?")
    if LOAD_ERROR:
        gr.Markdown(f"**{LOAD_ERROR}**")

    with gr.Tab("What is this?"):
        gr.Markdown(EXPLAINER)
        gr.HTML(DIAGRAM)

    with gr.Tab("Examples"):
        with gr.Row():
            preset = gr.Dropdown(list(PRESETS), value="Sudden drift (synthetic)", label="Example")
            flags = gr.Checkbox(value=True, label="Mark ADWIN drift flags")
        methods = gr.CheckboxGroup([v[0] for v in METHODS.values()], value=DEFAULT_METHODS,
                                   label="Methods to plot")
        ex_header = gr.Markdown()
        ex_plot = gr.Plot()
        gr.Markdown("**Results for this example** (all methods, best mean RMSE first)")
        ex_table = gr.Dataframe(interactive=False, wrap=True)
        for c in (preset, methods, flags):
            c.change(show_example, [preset, methods, flags], [ex_header, ex_plot, ex_table])

    with gr.Tab("Results"):
        gr.Markdown(key_findings_md())
        if read_ablation() is not None:          # shown only once the ablation has been run
            gr.Markdown(adwin_sensitivity_md())
        r_choice = gr.Dropdown(list(RESULT_STREAMS), value="Sudden drift (synthetic)", label="Stream / dataset")
        r_text = gr.Markdown()
        r_table = gr.Dataframe(interactive=False, wrap=True)
        with gr.Row():
            r_img1 = gr.Image(label="Mean RMSE per method", type="filepath", interactive=False)
            r_img2 = gr.Image(label="Recovery time / no-drift regret", type="filepath", interactive=False)
        r_choice.change(show_results, r_choice, [r_text, r_table, r_img1, r_img2])

    with gr.Tab("Limitations"):
        gr.Markdown(limitations_md())

    if SHOW_MOCK:   # hidden unless started with SHOW_MOCK=1
        with gr.Accordion("Developer only: MOCK data (FAKE numbers, never present these)", open=False):
            gr.Markdown("Invented numbers from `mock/make_mock_results.py`, used only to test the app layout.")
            m_stream = gr.Dropdown(["sudden", "gradual", "recurring", "none"], value="sudden", label="Stream")
            m_plot = gr.Plot()
            gr.Button("Show MOCK plot").click(mock_plot, m_stream, m_plot)

    # fill the first views when the page opens
    demo.load(show_example, [preset, methods, flags], [ex_header, ex_plot, ex_table])
    demo.load(show_results, r_choice, [r_text, r_table, r_img1, r_img2])


if __name__ == "__main__":
    demo.launch()
