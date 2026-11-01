import numpy as np
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from common.contract import Method


class _WindowedRefit(Method):
    """Keeps the last n_blocks blocks and refits a fresh model before each prediction."""

    def __init__(self, n_blocks=5, seed=0):
        self.n_blocks = n_blocks
        self.seed = seed
        self._X, self._y = [], []

    def _make_model(self):
        raise NotImplementedError

    def predict(self, X_block):
        Xh = np.vstack(self._X[-self.n_blocks :])
        yh = np.concatenate(self._y[-self.n_blocks :])
        model = self._make_model()
        model.fit(Xh, yh)
        return model.predict(X_block)

    def update(self, X_block, y_block):
        self._X.append(np.asarray(X_block))
        self._y.append(np.asarray(y_block))


class XGBBaseline(_WindowedRefit):
    name = "XGBoost"

    def _make_model(self):
        return XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=self.seed,
            n_jobs=-1,
        )


class MLPBaseline(_WindowedRefit):
    name = "MLP"

    def _make_model(self):
        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 128),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            early_stopping=True,
            max_iter=300,
            random_state=self.seed,
        )
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), mlp), transformer=StandardScaler()
        )


class AdaptiveForestBaseline(Method):
    name = "AdaptiveForest"

    def __init__(self, seed=0):
        from river import forest

        self.model = forest.ARFRegressor(seed=seed)  # API from memory, check river docs

    @staticmethod
    def _row(r):
        return {f"f{i}": float(v) for i, v in enumerate(r)}

    def predict(self, X_block):
        return np.array([self.model.predict_one(self._row(r)) for r in X_block])

    def update(self, X_block, y_block):
        for r, y in zip(X_block, y_block):
            self.model.learn_one(self._row(r), float(y))
