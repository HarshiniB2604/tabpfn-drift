"""
THE CONTRACT. Both members code against this file.
Do not change anything here without messaging the other person first.
"""
from abc import ABC, abstractmethod
import numpy as np

N_FEATURES = 10
N_BLOCKS = 60
BLOCK_SIZE = 200
STREAM_KINDS = ("sudden", "gradual", "recurring", "none")
FEATURE_COLS = [f"f{i}" for i in range(N_FEATURES)]

C_MAX = 2000   # context cap shared by all TabPFN policies

# One row per scored block. File name: results/{stream}_{seed}_{method}.csv
RESULT_COLUMNS = [
    "stream", "seed", "method", "block",
    "rmse", "mae", "r2",
    "drift_flag",   # 1 if the method's drift detector fired on this block
    "n_context",    # context rows used (NaN for non-TabPFN methods)
]

VANILLA_NAME = "P0_vanilla"
METHOD_NAMES = [
    "P0_vanilla", "P1_window", "P2_recency", "P3_drift", "P4_valguided",
    "XGBoost", "MLP", "AdaptiveForest",
]


class Method(ABC):
    """Every model, baseline and TabPFN policy implements this."""
    name: str = "unnamed"

    @abstractmethod
    def predict(self, X_block: np.ndarray) -> np.ndarray:
        """Predict a block. May ONLY use labelled data seen via update()."""

    @abstractmethod
    def update(self, X_block: np.ndarray, y_block: np.ndarray) -> None:
        """Called AFTER the block was predicted and its labels are revealed."""

    def info(self) -> dict:
        return {"drift_flag": 0, "n_context": np.nan}