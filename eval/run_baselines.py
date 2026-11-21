"""
Run the baselines (XGBoost, MLP, AdaptiveForest) and save one CSV per run:
    {out}/{stream}_{seed}_{method}.csv

Synthetic streams (default, all four):   python -m eval.run_baselines
Real datasets (data/raw/ needed):        python -m eval.run_baselines --dataset bike pm25 energy
Resumable: finished CSVs are skipped.
"""
import argparse
import time
from common.contract import STREAM_KINDS, REAL_DATASETS, FEATURE_COLS
from data.streams import make_stream
from eval.runner import run_prequential
from methods.baselines import XGBBaseline, MLPBaseline, AdaptiveForestBaseline


def streams_to_run(a):
    """Yield (name, df, feature_cols) for every stream/dataset requested."""
    if a.dataset:
        from data.real import load_real
        for name in a.dataset:
            s = load_real(name)
            yield name, (lambda seed, s=s: s.df), s.feature_cols   # real data is the same for every seed
    else:
        for kind in a.streams:
            yield kind, (lambda seed, kind=kind: make_stream(kind, seed)[0]), FEATURE_COLS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--streams", nargs="+", default=list(STREAM_KINDS), choices=STREAM_KINDS)
    p.add_argument("--dataset", nargs="+", choices=REAL_DATASETS, help="real datasets instead of synthetic streams")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out", default="results")
    a = p.parse_args()

    for name, get_df, cols in streams_to_run(a):
        for seed in a.seeds:
            df = get_df(seed)
            for m in [XGBBaseline(seed=seed), MLPBaseline(seed=seed), AdaptiveForestBaseline(seed=seed)]:
                t0 = time.time()
                path = run_prequential(m, df, name, seed, out_dir=a.out, feature_cols=cols)
                print(f"{name} seed={seed} {m.name}: {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
