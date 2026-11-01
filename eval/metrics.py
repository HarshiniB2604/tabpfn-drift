"""
recovery_time: pre-drift level = mean RMSE of the 5 blocks before the change starts.
  Recovery = first block where RMSE stays within 10% of that level for 2 consecutive
  blocks, counted from the drift block (from block 40 for gradual drift). Capped and
  flagged "not recovered" if it never happens before the next drift / end of stream.
no_drift_regret: (method RMSE - vanilla RMSE) / vanilla RMSE on the 'none' stream.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from common.contract import VANILLA_NAME, N_BLOCKS
from data.streams import drift_blocks_for, GRADUAL_END
from common.contract import VANILLA_NAME, N_BLOCKS, ALIASES

PRE_WINDOW = 5
TOL = 0.10


def load_results(results_dir="results"):
    files = sorted(Path(results_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSVs in {results_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["method"] = df["method"].replace(ALIASES)
    return df


def recovery_time(rmse_by_block, change_start, search_start, horizon_end):
    pre = rmse_by_block.loc[max(1, change_start - PRE_WINDOW) : change_start - 1]
    if len(pre) == 0:
        return np.nan, False
    threshold = pre.mean() * (1 + TOL)
    for k in range(search_start, horizon_end):
        if k + 1 > rmse_by_block.index.max():
            break
        if rmse_by_block.loc[k] <= threshold and rmse_by_block.loc[k + 1] <= threshold:
            return float(k - search_start), True
    return float(horizon_end - search_start), False


def _recovery_for_run(run, kind):
    s = run.set_index("block")["rmse"]
    starts = drift_blocks_for(kind)
    out = []
    for i, d in enumerate(starts):
        search_start = GRADUAL_END if kind == "gradual" else d
        horizon_end = starts[i + 1] if i + 1 < len(starts) else N_BLOCKS
        out.append(recovery_time(s, d, search_start, horizon_end))
    return out


def summary_table(res):
    rows = []
    for (stream, method), g in res.groupby(["stream", "method"]):
        recs = []
        for _, run in g.groupby("seed"):
            recs += _recovery_for_run(run, stream)
        vals = [r[0] for r in recs if not np.isnan(r[0])]
        flags = [r[1] for r in recs if not np.isnan(r[0])]
        rows.append(
            dict(
                stream=stream,
                method=method,
                mean_rmse=g["rmse"].mean(),
                recovery_blocks=np.mean(vals) if vals else np.nan,
                pct_recovered=100 * np.mean(flags) if flags else np.nan,
                n_seeds=g["seed"].nunique(),
            )
        )
    out = pd.DataFrame(rows)
    van = out[(out.stream == "none") & (out.method == VANILLA_NAME)]["mean_rmse"]
    out["no_drift_regret_pct"] = np.nan
    if len(van):
        base = van.iloc[0]
        m = out.stream == "none"
        out.loc[m, "no_drift_regret_pct"] = (
            100 * (out.loc[m, "mean_rmse"] - base) / base
        )
    return out.sort_values(["stream", "mean_rmse"]).reset_index(drop=True)
