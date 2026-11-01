"""
Prequential (test-then-train) runner. Block 0 is warm-up (not scored).
Resumable: skips a run if its CSV already exists (Colab disconnects!).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from common.contract import Method, FEATURE_COLS, RESULT_COLUMNS


def _scores(y, pred):
    err = y - pred
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return (
        float(np.sqrt(np.mean(err**2))),
        float(np.mean(np.abs(err))),
        1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    )


def run_prequential(method, df, stream, seed, out_dir="results", overwrite=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stream}_{seed}_{method.name}.csv"
    if path.exists() and not overwrite:
        return path

    rows = []
    for b, blk in df.groupby("block", sort=True):
        X = blk[FEATURE_COLS].to_numpy()
        y = blk["y"].to_numpy()
        if b > 0:
            pred = np.asarray(method.predict(X)).ravel()
            info = method.info()
            rmse, mae, r2 = _scores(y, pred)
            rows.append(
                dict(
                    stream=stream,
                    seed=seed,
                    method=method.name,
                    block=int(b),
                    rmse=rmse,
                    mae=mae,
                    r2=r2,
                    drift_flag=int(info.get("drift_flag", 0)),
                    n_context=info.get("n_context", np.nan),
                )
            )
        method.update(X, y)  # labels revealed only now

    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(tmp, index=False)
    tmp.replace(path)  # atomic: never leaves a half-written CSV
    return path
