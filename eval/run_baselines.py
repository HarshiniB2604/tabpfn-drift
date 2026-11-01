from data.streams import make_stream
from eval.runner import run_prequential
from methods.baselines import XGBBaseline, MLPBaseline, AdaptiveForestBaseline

SEEDS = [0, 1, 2]
STREAMS = ["sudden", "gradual", "recurring", "none"]

for kind in STREAMS:
    for seed in SEEDS:
        df, _ = make_stream(kind, seed)
        for m in [
            XGBBaseline(seed=seed),
            MLPBaseline(seed=seed),
            AdaptiveForestBaseline(seed=seed),
        ]:
            print(f"{kind} seed={seed} {m.name} ...", flush=True)
            print("  ->", run_prequential(m, df, kind, seed, out_dir="results"))
