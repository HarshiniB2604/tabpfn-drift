"""
Pipeline self-test. Uses the fast StandInModel, NOT TabPFN, and needs no GPU.
It checks the plumbing (streams, no leakage, context cap, runner, policies, metrics),
not how good TabPFN is. Must end with: ALL SMOKE TESTS PASSED
Run: python smoke_test.py   (about a minute)
"""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from common.contract import RESULT_COLUMNS, FEATURE_COLS, N_BLOCKS, BLOCK_SIZE
from data.streams import make_stream
from eval.runner import run_prequential
from eval.metrics import load_results, summary_table
from methods.policies import POLICIES, make_policy, pick_random, pick_window, pick_recency
from methods.tabpfn_wrapper import StandInModel


def ok(msg):
    print("  OK  ", msg)


print("1. streams")
df, drifts = make_stream("sudden", 0)
assert df.shape == (N_BLOCKS * BLOCK_SIZE, 13) and drifts == [30]
a, _ = make_stream("sudden", 0)
assert a.equals(df), "streams must be reproducible"
ok(f"sudden stream {df.shape}, drift at {drifts}, reproducible")

print("2. selection helpers")
rng = np.random.default_rng(0)
blocks = np.repeat(np.arange(20), 200)                       # 20 blocks of history
assert len(pick_random(len(blocks), 2000, rng)) == 2000
assert len(np.unique(pick_random(len(blocks), 2000, rng))) == 2000   # no repeats
assert set(blocks[pick_window(blocks, 5, 2000)]) == {15, 16, 17, 18, 19}
rec = blocks[pick_recency(blocks, 5, 2000, rng)]
assert rec.mean() > blocks.mean() + 2, "recency should favour new blocks"
ok("cap respected, window = last 5 blocks, recency favours new rows")

print("3. no future leakage + context cap, every policy (short stream, c_max=500)")
short, _ = make_stream("sudden", 1, n_blocks=8)
for name in POLICIES:
    pol = make_policy(name, StandInModel(seed=0), c_max=500, seed=0)
    for b, blk in short.groupby("block"):
        X, y = blk[FEATURE_COLS].to_numpy(), blk["y"].to_numpy()
        if b > 0:
            pol.predict(X)
            assert pol.last_context_blocks.max() < b, f"{name} used block {b} or later!"
            assert pol.info()["n_context"] <= 500, f"{name} broke the context cap"
        pol.update(X, y)
ok("all 6 policies only ever see past blocks and stay within c_max")

print("4. full sudden stream, seed 0, all policies through the runner (stand-in model)")
out = Path(tempfile.mkdtemp())
res = {}
for name in POLICIES:
    pol = make_policy(name, StandInModel(seed=0), seed=0)
    path = run_prequential(pol, df, "sudden", 0, out_dir=out)
    r = pd.read_csv(path)
    assert list(r.columns) == RESULT_COLUMNS and len(r) == N_BLOCKS - 1
    assert r.n_context.max() <= 2000
    res[name] = r.set_index("block")

pre = lambda m: res[m].rmse.loc[25:29].mean()
late = lambda m: res[m].rmse.loc[45:59].mean()
assert res["vanilla"].rmse.loc[30] > 2 * pre("vanilla"), "no error jump at the drift"
ok(f"vanilla RMSE jumps at drift: {pre('vanilla'):.2f} -> {res['vanilla'].rmse.loc[30]:.2f}")
assert late("sliding_window") < late("vanilla")
ok(f"sliding window recovers better than vanilla (blocks 45-59: {late('sliding_window'):.2f} vs {late('vanilla'):.2f})")
flags = res["adwin"].index[res["adwin"].drift_flag == 1].tolist()
assert flags and all(30 <= f <= 33 for f in flags[:1]) and not any(f < 30 for f in flags)
ok(f"ADWIN flags at blocks {flags} (drift at 30; flag shows on the block after detection)")
assert res["validation"].n_context.loc[35:].min() < 2000, "validation never switched after drift"
ok(f"validation switched context after drift (n_context at 35: {res['validation'].n_context.loc[35]:.0f})")
for m in POLICIES:
    print(f"        {m:15s} pre {pre(m):.2f}  block30 {res[m].rmse.loc[30]:.2f}  late {late(m):.2f}")

print("5. resumable runner")
p = out / "sudden_0_vanilla.csv"
t = p.stat().st_mtime_ns
run_prequential(make_policy("vanilla", StandInModel(), seed=0), df, "sudden", 0, out_dir=out)
assert p.stat().st_mtime_ns == t, "existing file should be skipped"
ok("existing result file is skipped, not recomputed")

print("6. metrics read the files")
tab = summary_table(load_results(out))
assert set(tab.method) == {"P0_vanilla", "P1_window", "P2_recency", "P3_drift", "P4_valguided", "kNN_context"}
ok("load_results maps names via ALIASES; summary_table works")
print(tab.round(2).to_string(index=False))

print("7. real-data loaders (skipped if data/raw/ is missing)")
from data.real import LOADERS, load_real, RAW
if not any(RAW.glob("*.csv")):
    print("  SKIP  no files in data/raw/")
else:
    for name in LOADERS:
        s = load_real(name)
        d = s.df
        assert d["time"].is_monotonic_increasing, f"{name} not in time order"
        assert d[s.feature_cols + ["y"]].notna().all().all(), f"{name} has missing values"
        assert (d.groupby("block").size() == BLOCK_SIZE).all() and d.block.nunique() >= 30
        assert "y" not in s.feature_cols and "time" not in s.feature_cols
        ok(f"{name}: {d.block.nunique()} blocks x {BLOCK_SIZE} rows, {len(s.feature_cols)} features, time-ordered, no NaN")
    s = load_real("bike")
    small = s.df[s.df.block < 4]
    path = run_prequential(make_policy("vanilla", StandInModel(seed=0), seed=0), small, "bike", 0,
                           out_dir=out, feature_cols=s.feature_cols)
    assert list(pd.read_csv(path).columns) == RESULT_COLUMNS
    ok("runner works with real feature columns, same CSV schema")

print("\nALL SMOKE TESTS PASSED  (stand-in model, NOT TabPFN)")
