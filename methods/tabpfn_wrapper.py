"""
Model wrappers. A "model" here is anything with fit(X, y) and predict(X).
The policies (methods/policies.py) only choose the context rows; they hand those rows
to one of these models.

- TabPFNModel  : the real TabPFN. Needs a GPU (Colab T4) and a licence token
                 (TABPFN_TOKEN or HF_TOKEN environment variable). Never commit tokens.
- StandInModel : a fast tree model used ONLY to test the pipeline locally.
                 It is NOT TabPFN. Results from it must be labelled "standin".
"""
import numpy as np


class TabPFNModel:
    """TabPFN does not train weights: fit() just stores the context rows,
    predict() compares new rows to that context. Checked against tabpfn 10.2.0."""

    def __init__(self, seed=0, device="auto"):
        # Imported here so the rest of the repo works without tabpfn installed.
        import torch
        from tabpfn import TabPFNRegressor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            # On CPU TabPFN refuses more than 1000 context rows by default
            # (and is very slow). We keep that safety check on purpose.
            print("WARNING: TabPFN on CPU. Use a GPU (Colab T4) for real runs.")
        self.device = device
        self.model = TabPFNRegressor(device=device, random_state=seed)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X)).ravel()


class StandInModel:
    """Small, fast random forest. Stand-in for pipeline tests only - NOT TabPFN."""

    def __init__(self, seed=0):
        from sklearn.ensemble import ExtraTreesRegressor

        self.model = ExtraTreesRegressor(
            n_estimators=30, min_samples_leaf=3, random_state=seed, n_jobs=-1
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def make_model(kind, seed=0):
    """kind = 'tabpfn' (real) or 'standin' (fast fake for tests)."""
    if kind == "tabpfn":
        return TabPFNModel(seed=seed)
    if kind == "standin":
        return StandInModel(seed=seed)
    raise ValueError(f"unknown model kind: {kind}")
