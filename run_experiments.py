"""
Run TabPFN context policies on synthetic streams or real datasets and save one CSV per run:
    {out}/{stream}_{seed}_{method}.csv   (schema: common/contract.py RESULT_COLUMNS)

Real TabPFN (Colab GPU):
    python run_experiments.py --streams sudden --seeds 0 --methods vanilla sliding_window --c-max 2000 --out /content/drive/MyDrive/tabpfn-validation
Real datasets (data/raw/ needed) instead of synthetic streams:
    python run_experiments.py --dataset bike pm25 energy --seeds 0 1 2 --methods vanilla --out OUT
ADWIN sensitivity ablation (only --methods adwin, never into results/; files are named
like sudden_0_P3_drift_d1e-4.csv):
    python run_experiments.py --streams sudden --seeds 0 1 2 --methods adwin --adwin-delta 1e-4 --out results_ablation
Local pipeline test with the stand-in model (NOT TabPFN, files are named "...-standin"):
    python run_experiments.py --streams sudden --seeds 0 --methods vanilla --model standin --out results/standin

Resumable: finished CSVs are skipped, so after a Colab disconnect just run it again.
"""
import argparse
import time
from pathlib import Path
from common.contract import C_MAX, STREAM_KINDS, REAL_DATASETS, FEATURE_COLS
from data.streams import make_stream
from data.real import load_real
from eval.runner import run_prequential
from methods.policies import POLICIES, make_policy
from methods.tabpfn_wrapper import make_model

MAIN_RESULTS = (Path(__file__).resolve().parent / "results").resolve()


def delta_tag(delta):
    """0.0001 -> '1e-4', 0.002 -> '2e-3', 1e-10 -> '1e-10' (used in the method name)."""
    mant, exp = f"{delta:.0e}".split("e")
    return f"{mant}e{int(exp)}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--streams", nargs="+", default=list(STREAM_KINDS), choices=STREAM_KINDS)
    p.add_argument("--dataset", nargs="+", choices=REAL_DATASETS, help="real datasets instead of synthetic streams")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--methods", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    p.add_argument("--c-max", type=int, default=C_MAX)
    p.add_argument("--out", required=True, help="output folder (never the teammate's results/ by accident)")
    p.add_argument("--model", default="tabpfn", choices=["tabpfn", "standin"])
    p.add_argument("--adwin-delta", type=float, default=None,
                   help="ablation only: ADWIN significance level (default policy uses 0.002). "
                        "Needs --methods adwin and an --out folder other than results/.")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    if a.adwin_delta is not None:
        # The ablation must never mix into the main results or change other methods.
        if a.methods != ["adwin"]:
            p.error("--adwin-delta only works with --methods adwin")
        if Path(a.out).resolve() == MAIN_RESULTS:
            p.error("--adwin-delta runs must not write to results/ (use e.g. --out results_ablation)")
        if not 0 < a.adwin_delta < 1:
            p.error("--adwin-delta must be between 0 and 1")

    for seed in a.seeds:
        model = make_model(a.model, seed)          # loaded once per seed, reused by all runs
        for stream in (a.dataset or a.streams):
            if a.dataset:
                s = load_real(stream)                       # same data for every seed
                df, cols = s.df, s.feature_cols
            else:
                df, cols = make_stream(stream, seed)[0], FEATURE_COLS
            for name in a.methods:
                if a.adwin_delta is None:
                    policy = make_policy(name, model, c_max=a.c_max, seed=seed)   # unchanged default
                else:
                    policy = make_policy(name, model, c_max=a.c_max, seed=seed, delta=a.adwin_delta)
                    policy.name = f"P3_drift_d{delta_tag(a.adwin_delta)}"   # delta in the file name
                if a.model == "standin":
                    policy.name = f"{policy.name}-standin"  # label fake-model results clearly
                t0 = time.time()
                path = run_prequential(policy, df, stream, seed, out_dir=a.out, overwrite=a.overwrite,
                                       feature_cols=cols)
                print(f"{stream} seed={seed} {policy.name}: {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
