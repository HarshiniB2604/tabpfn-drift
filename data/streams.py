"""
Synthetic drift streams. 10 features ~ U[0,1], target = Friedman-1 + noise.
Concept A: Friedman-1 on f0..f4.  Concept B: same function on f5..f9.
`concept` column is for plotting/debugging only; methods must never use it.
"""
import numpy as np
import pandas as pd
from common.contract import N_FEATURES, N_BLOCKS, BLOCK_SIZE, FEATURE_COLS, STREAM_KINDS

NOISE_SD = 1.0
PERM_B = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
GRADUAL_START, GRADUAL_END = 20, 40
RECUR_PERIOD = 15


def friedman1(X):
    return (10 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20 * (X[:, 2] - 0.5) ** 2
            + 10 * X[:, 3]
            + 5 * X[:, 4])


def _prob_new_concept(kind, b, n_blocks):
    if kind == "none":
        return 0.0
    if kind == "sudden":
        return 0.0 if b < n_blocks // 2 else 1.0
    if kind == "gradual":
        return float(np.clip((b - GRADUAL_START) / (GRADUAL_END - GRADUAL_START), 0, 1))
    if kind == "recurring":
        return float((b // RECUR_PERIOD) % 2)
    raise ValueError(f"unknown stream kind: {kind}")


def drift_blocks_for(kind, n_blocks=N_BLOCKS):
    if kind == "none":
        return []
    if kind == "sudden":
        return [n_blocks // 2]
    if kind == "gradual":
        return [GRADUAL_START]
    if kind == "recurring":
        return list(range(RECUR_PERIOD, n_blocks, RECUR_PERIOD))
    raise ValueError(kind)


def make_stream(kind, seed, n_blocks=N_BLOCKS, block_size=BLOCK_SIZE):
    assert kind in STREAM_KINDS, f"kind must be one of {STREAM_KINDS}"
    rng = np.random.default_rng(seed)
    frames = []
    for b in range(n_blocks):
        X = rng.random((block_size, N_FEATURES))
        p_new = _prob_new_concept(kind, b, n_blocks)
        is_new = rng.random(block_size) < p_new
        y = np.where(is_new, friedman1(X[:, PERM_B]), friedman1(X))
        y = y + rng.normal(0.0, NOISE_SD, block_size)
        blk = pd.DataFrame(X, columns=FEATURE_COLS)
        blk["y"] = y
        blk["block"] = b
        blk["concept"] = is_new.astype(int)
        frames.append(blk)
    return pd.concat(frames, ignore_index=True), drift_blocks_for(kind, n_blocks)