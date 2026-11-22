"""
Site number check: every number in the site's computed text must come from the data.
    python tests/test_site_numbers.py        (exit code 0 = pass)

1. site/data.js is up to date: it equals a fresh `python -m eval.make_site` (except the build time).
2. Every number in the findings, the synthetic winners sentence, the robustness takeaway, the ADWIN
   sensitivity block and the limitations is either
   - a value from results/summary.csv or results_ablation/ablation_summary.csv, rounded as shown,
   - a ratio or % difference between two such shown values (how the site computes them),
   - a count or setting that make_site exports (blocks, seeds, deltas, ...), or a small count (0-10).
   Anything else is reported as an unexplained number and the test fails.
(The browser check tests/site_check.js then confirms the page shows exactly this text.)
"""
import itertools
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_js(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text[text.index("{"): text.rstrip().rstrip(";").rindex("}") + 1])


def fresh_build():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "data.js"
        subprocess.run([sys.executable, "-m", "eval.make_site", "--out", str(out)], cwd=ROOT, check=True,
                       capture_output=True)
        return load_js(out)


def shown(x, digits):
    return f"{round(float(x), digits):.{digits}f}"


def allowed_numbers(R, summ, ab):
    ok = {str(i) for i in range(0, 11)}                  # small counts: "3 seeds", "1 true drift", "4 values"
    vals2 = []                                            # shown 2-decimal values, for ratios and %
    for r in summ.itertuples():
        for col, digits in (("rmse_mean", 2), ("rmse_ci95", 2), ("recovery_mean", 1), ("recovery_ci95", 1),
                            ("regret_pct_mean", 1), ("regret_pct_ci95", 1), ("drift_flags_mean", 0), ("drift_flags_mean", 1)):
            v = getattr(r, col)
            if pd.notna(v):
                ok.add(shown(v, digits)); ok.add(shown(abs(v), digits))
        ok.add(str(int(r.n_seeds)))
        if isinstance(r.recovered_events, str) and "/" in r.recovered_events:
            ok.update(r.recovered_events.split("/"))
        vals2.append(round(float(r.rmse_mean), 2))
    if ab is not None:
        for r in ab.itertuples():
            for col, digits in (("rmse_mean", 2), ("rmse_ci95", 2), ("flags_mean", 0), ("flags_mean", 1),
                                ("vs_vanilla_pct", 1), ("vs_vanilla_ci95", 1), ("vanilla_rmse", 2)):
                v = getattr(r, col)
                if pd.notna(v):
                    ok.add(shown(v, digits)); ok.add(shown(abs(v), digits))
            vals2 += [round(float(r.rmse_mean), 2), round(float(r.vanilla_rmse), 2)]
            ok.add(str(int(r.blocks)))
    # ratios and % differences of shown values (the site computes them this way)
    for a, b in itertools.permutations(set(vals2), 2):
        if b:
            ok.add(shown(a / b, 1)); ok.add(shown(abs(100 * (a - b) / b), 1))
    # sums of recovered/total drift counts per method (e.g. "3 of 15 drifts")
    for m, g in summ[summ.recovered_events.astype(str).str.contains("/")].groupby("method"):
        parts = [tuple(map(int, e.split("/"))) for e in g.recovered_events]
        ok.update({str(sum(p[0] for p in parts)), str(sum(p[1] for p in parts))})
    # settings and counts exported by make_site
    s = R["settings"]
    ok.update(str(x) for x in (s["block_size"], s["c_max"], s["n_blocks"], s["rolling_window"], s["pre_window"],
                               int(s["tolerance_pct"]), *s["seeds"]))
    ok.update(re.findall(r"\d+", s["ci"]))                 # the CI level ("95% t-interval") from the report metadata
    ok.update(str(d["scored_blocks"]) for d in R["datasets"])
    ok.update(str(b) for d in R["datasets"] for b in d["drift_blocks"])
    for k, v in R["policy_settings"].items():
        for x in v.values():
            if isinstance(x, (int, float)):
                ok.update({str(x), shown(x, 0)})
    bs = R["baseline_settings"]["refit_blocks"]
    ok.update({str(bs), str(bs * s["block_size"])})
    P = R["problem"]
    ok.update({shown(P["pre_mean"], 2), shown(P["peak"], 2), str(P["drift_block"]), str(P["drift_block"] + 1),
               shown(round(P["peak"], 2) / round(P["pre_mean"], 2), 1)})
    for v in R["robustness"].values():
        ok.update(shown(r["gap_pct"], 1) for r in v)
    return ok


def texts(R):
    out = []
    for f in R["findings"]:
        out += [f["headline"], f["body"], f.get("real", ""), f.get("caveat", "")] + [e["text"] for e in f["evidence"]]
    if R.get("synthetic_winners"):
        out.append(R["synthetic_winners"]["text"])
    out.append(R.get("robustness_takeaway", ""))
    A = R.get("adwin_sensitivity")
    if A:
        out += [A["intro"], A["summary"], A["caveat"]] + A["points"]
    out += [l["text"] for l in R["limitations"]]
    return [t for t in out if t]


def main():
    committed = load_js(ROOT / "site" / "data.js")
    fresh = fresh_build()
    committed.pop("generated"); fresh.pop("generated")
    failures = []
    if committed != fresh:
        failures.append("site/data.js is out of date: run python -m eval.make_site")

    summ = pd.read_csv(ROOT / "results" / "summary.csv")
    abp = ROOT / "results_ablation" / "ablation_summary.csv"
    ab = pd.read_csv(abp) if abp.exists() else None
    ok = allowed_numbers(fresh, summ, ab)
    deltas = re.compile(r"\d+(?:\.\d+)?e-\d+|0\.002")          # delta values like 1e-4 are settings
    checked = 0
    for t in texts(fresh):
        for num in re.findall(r"\d+(?:\.\d+)?", deltas.sub(" ", t.replace("PM2.5", "PM").replace("pm25", "pm"))):
            checked += 1
            if num not in ok:
                failures.append(f"unexplained number {num!r} in: {t[:110]}...")
    print(f"checked {checked} numbers in {len(texts(fresh))} computed texts")
    if failures:
        print("FAIL"); print("\n".join(failures)); sys.exit(1)
    print("PASS: data.js is current and every number traces to summary.csv / the ablation summary / exported settings")


if __name__ == "__main__":
    main()
