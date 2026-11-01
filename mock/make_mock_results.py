"""MOCK DATA ONLY: invented numbers in the real results schema. Output: results/mock/"""

from pathlib import Path
import zlib
import numpy as np
import pandas as pd
from common.contract import STREAM_KINDS, N_BLOCKS, C_MAX, RESULT_COLUMNS, BLOCK_SIZE
from data.streams import drift_blocks_for, GRADUAL_START, GRADUAL_END

# name: (base_rmse, spike_amplitude, recovery_tau_blocks, has_detector, uses_context)
PROFILES = {
    "P0_vanilla": (1.30, 1.60, 22.0, False, True),
    "P1_window": (1.36, 1.20, 5.0, False, True),
    "P2_recency": (1.34, 1.30, 8.0, False, True),
    "P3_drift": (1.38, 1.20, 4.0, True, True),
    "P4_valguided": (1.33, 1.00, 4.5, False, True),
    "XGBoost": (1.45, 1.10, 6.0, False, False),
    "MLP": (1.70, 1.30, 7.0, False, False),
    "AdaptiveForest": (1.60, 0.90, 5.0, True, False),
}


def _penalty(kind, t, amp, tau):
    if kind == "none":
        return 0.0
    starts = drift_blocks_for(kind)
    if kind == "gradual":
        if t < GRADUAL_START:
            return 0.0
        ramp = min(1.0, (t - GRADUAL_START) / (GRADUAL_END - GRADUAL_START))
        return amp * ramp * np.exp(-max(0, t - GRADUAL_END) / tau)
    past = [d for d in starts if d <= t]
    return amp * np.exp(-(t - past[-1]) / tau) if past else 0.0


def make_mock(seeds=(0, 1, 2), out_dir="results/mock"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for kind in STREAM_KINDS:
        starts = drift_blocks_for(kind)
        for seed in seeds:
            for method, (base, amp, tau, has_det, uses_ctx) in PROFILES.items():
                rng = np.random.default_rng(
                    zlib.crc32(f"{kind}-{seed}-{method}".encode())
                )
                rows = []
                for b in range(1, N_BLOCKS):
                    rmse = (
                        base
                        * (1 + _penalty(kind, b, amp, tau))
                        * rng.lognormal(0, 0.04)
                    )
                    flag = int(has_det and any(b == d + 1 for d in starts))
                    n_ctx = float(min(C_MAX, b * BLOCK_SIZE)) if uses_ctx else np.nan
                    rows.append(
                        dict(
                            stream=kind,
                            seed=seed,
                            method=method,
                            block=b,
                            rmse=rmse,
                            mae=rmse * 0.8,
                            r2=max(0.0, 1 - (rmse / 5) ** 2),
                            drift_flag=flag,
                            n_context=n_ctx,
                        )
                    )
                pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
                    out / f"{kind}_{seed}_{method}.csv", index=False
                )
    print(f"MOCK results written to {out}/ (fake numbers, do not present as real)")


if __name__ == "__main__":
    make_mock()
