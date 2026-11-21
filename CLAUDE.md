# CLAUDE.md - TabPFN under concept drift (context-selection policies)

## What this project is
3rd-year CSE course project (AIML). Question: when data drifts, which labelled rows
should TabPFN (pretrained tabular model, no training, predicts from a "context" of rows)
see? We compare cheap, training-free context-selection policies on regression streams.
I (the user) now finish it ALONE and must present a demo plus results. Keep explanations
simple; I must be able to explain every line in a viva. Prefer short replies.

## Layout (repo root)
- common/contract.py  shared contract: constants, RESULT_COLUMNS, Method interface, ALIASES
- data/streams.py     synthetic Friedman-1 drift streams (sudden, gradual, recurring, none)
- eval/runner.py      run_prequential (test-then-train, resumable, atomic CSV writes)
- eval/metrics.py     load_results, recovery_time, summary_table
- methods/baselines.py  XGBoost, MLP, River AdaptiveForest (windowed refit)
- eval/run_baselines.py, app/app.py (Gradio), mock/ (FAKE data, never show as real)
- results/*.csv       {stream}_{seed}_{method}.csv, 108 files, 4 streams x 9 methods x 3 seeds
  TabPFN policies in results came from my ex-teammate (names: vanilla, sliding_window,
  recency, adwin, validation, knn). Baselines: XGBoost, MLP, AdaptiveForest.
- tabpfn-drift-results/ is a DUPLICATE of the teammate CSVs -> remove it.

## MISSING: the TabPFN engine (teammate's zip is NOT available - rebuild it)
Needed: methods/tabpfn_wrapper.py, policies P0-P4 + kNN (methods/policies.py), run_experiments.py
(CLI: --streams --seeds --methods --c-max --out), smoke_test.py (fast stand-in model, no GPU),
time_tabpfn.py, plot_check.py, notebooks/colab_run.ipynb (a copy of the teammate's runner is in incoming/).
TabPFN needs a GPU (Colab T4) and a license token (TABPFN_TOKEN or HF_TOKEN; never commit tokens).
CPU refuses >1000 context rows by default. River ADWIN API (0.26.1): update(x), drift_detected.

Policy spec (from the teammate's design; defaults, untuned):
- P0 vanilla: random sample of up to C_MAX rows from all history.
- P1 sliding_window: last 5 blocks (cap C_MAX).
- P2 recency: sample up to C_MAX rows, weight 0.5 ** (age_in_blocks / 5), no replacement.
- P3 adwin: feed ADWIN each row's absolute error; on a detected INCREASE set drift_flag=1 for
  that block and drop history before the detected point; ignore error decreases.
- P4 validation: candidates e.g. P0, P1 with 3/5/10-block windows, P2 half-lives 2 and 5; score on the
  last 2 blocks (history excluding them); switch only if RMSE improves >5%; else keep current.
- kNN_context: nearest C_MAX history rows to the incoming block in feature space.
Each uses only past labelled data (no future leakage; test-then-train).
VALIDATION OF THE REBUILD: re-run sudden seed 0 and compare against results/sudden_0_*.csv.
vanilla and sliding_window should match closely (same shape: ~1.0 -> ~6.5 at block 30). Report
differences honestly; do not tune to force a match.

## Facts to respect
- 60 blocks x 200 rows, block 0 = warm-up (unscored). Drift at 30 (sudden), 20->40 (gradual),
  15/30/45 (recurring). C_MAX = 2000 context rows for every policy.
- Metrics: RMSE/MAE/R2 per block; recovery time (baseline = mean RMSE of 5 pre-drift blocks,
  10% tolerance, 2 consecutive blocks, capped + flagged "not recovered"); no-drift regret vs vanilla.
- Known results (3 seeds, synthetic): vanilla RMSE ~1.0 -> ~6.5 at drift; sliding window, ADWIN,
  validation recover to ~1.0; kNN no help (only the rule changes, not inputs); ADWIN weak on
  gradual drift (unexplained); sliding vs validation differ by noise only.
- Do NOT change the CSV schema or Method interface casually.
- Honesty rules: results from a stand-in model or mock data must be labelled as such. Report
  negative findings. Synthetic != real. No confidence claims from 3 seeds.

## TODO (in order, one at a time, run + verify each before the next)
1. Rebuild the engine per the spec above, keeping contract.py/ALIASES intact. Make smoke_test.py
   pass locally, delete the duplicate results folder, then give me a Colab notebook to run the
   validation on sudden seed 0. Explain each file to me in simple words.
2. eval/make_report.py: write results/summary.csv and site_data/results.json (mean +/- 95% CI
   per method/stream, per-seed table, curves for plotting). Run: python -m eval.make_report
3. Real datasets in data/raw/ (Bike Sharing hour.csv, Beijing PM2.5, Appliances Energy):
   loaders in data/real.py (time-ordered, numeric features, blocks), generalise the runner to
   arbitrary feature columns, run baselines locally and TabPFN policies via Colab.
   No known drift point on real data -> show rolling RMSE, no hidden "true drift" line.
4. App: Examples tab with 3 synthetic presets (sudden/gradual/recurring) + 3 real-dataset
   presets; each shows plot, vanilla vs best policy vs baselines, recovery/regret table.
   Clear labels: REPLAY of saved results vs LIVE computation. Summary tab with bar charts + CIs.
5. Fix weak points: ADWIN on gradual drift, fairer baseline window, 10 seeds, recency tuning.
6. Bonus last: P5 = concept-consistency filtering (keep old blocks that match the current
   concept; helps recurring drift). Otherwise remove P5 from the synopsis.
7. README with reproduction steps; keep repo private; .gitignore for data/raw, .env, tokens.
