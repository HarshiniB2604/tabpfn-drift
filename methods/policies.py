"""
Context-selection policies (the thing this project compares).

Before predicting each block, a policy decides WHICH past labelled rows the model sees
as its context. The model itself is the same for every policy; only the context differs.

Rules every policy follows:
- It only uses rows it was given through update(), i.e. blocks whose labels are already
  revealed (test-then-train, no future leakage).
- It never gives the model more than c_max rows (C_MAX = 2000 for every policy).

Names match the result files: vanilla, sliding_window, recency, adwin, validation, knn
(common/contract.py ALIASES maps them to P0..P4 / kNN_context).
"""
import numpy as np
from river import drift
from scipy.spatial.distance import cdist
from common.contract import Method, C_MAX


# ---------------------------------------------------------------------------
# Selection helpers. `blocks` = block number of every history row (oldest first).
# Each returns the indices (positions in history) of the rows to use as context.
# ---------------------------------------------------------------------------
def pick_random(n_rows, c_max, rng):
    """Uniform random sample of up to c_max rows from all history."""
    if n_rows <= c_max:
        return np.arange(n_rows)
    return np.sort(rng.choice(n_rows, size=c_max, replace=False))


def pick_window(blocks, n_blocks, c_max):
    """Rows from the newest n_blocks blocks (newest c_max rows if that is too many)."""
    idx = np.flatnonzero(blocks > blocks.max() - n_blocks)
    return idx[-c_max:]


def pick_recency(blocks, half_life, c_max, rng):
    """Random sample without replacement; a row's chance halves every half_life blocks of age."""
    if len(blocks) <= c_max:
        return np.arange(len(blocks))
    age = blocks.max() - blocks          # newest block has age 0
    w = 0.5 ** (age / half_life)
    return np.sort(rng.choice(len(blocks), size=c_max, replace=False, p=w / w.sum()))


# ---------------------------------------------------------------------------
# Base class: stores history, builds the context, calls the model.
# ---------------------------------------------------------------------------
class ContextPolicy(Method):
    name = "base"

    def __init__(self, model, c_max=C_MAX, seed=0):
        self.model = model
        self.c_max = c_max
        self.rng = np.random.default_rng(seed)
        self.X = None                      # history features
        self.y = None                      # history targets
        self.blocks = None                 # block number of each history row
        self.n_seen = 0                    # blocks seen so far = number of the next block
        self.last_pred = None              # predictions for the current block
        self.last_context_blocks = None    # used by smoke_test.py to check for leakage
        self._n_context = 0

    def select(self, X_block):
        """Return indices of history rows to use as context. Subclasses decide."""
        raise NotImplementedError

    def predict(self, X_block):
        idx = self.select(X_block)
        assert len(idx) <= self.c_max, "context cap broken"
        self.last_context_blocks = self.blocks[idx]
        self._n_context = len(idx)
        self.model.fit(self.X[idx], self.y[idx])
        self.last_pred = np.asarray(self.model.predict(X_block)).ravel()
        return self.last_pred

    def update(self, X_block, y_block):
        X_block, y_block = np.asarray(X_block), np.asarray(y_block)
        b = np.full(len(y_block), self.n_seen)
        if self.X is None:
            self.X, self.y, self.blocks = X_block, y_block, b
        else:
            self.X = np.vstack([self.X, X_block])
            self.y = np.concatenate([self.y, y_block])
            self.blocks = np.concatenate([self.blocks, b])
        self.n_seen += 1
        self.last_pred = None

    def info(self):
        return {"drift_flag": 0, "n_context": self._n_context}


# ---------------------------------------------------------------------------
# P0 - P2: fixed rules
# ---------------------------------------------------------------------------
class Vanilla(ContextPolicy):
    """P0: random sample of up to C_MAX rows from ALL history (the baseline)."""
    name = "vanilla"

    def select(self, X_block):
        return pick_random(len(self.y), self.c_max, self.rng)


class SlidingWindow(ContextPolicy):
    """P1: only the last 5 blocks (old data is forgotten)."""
    name = "sliding_window"
    window = 5

    def select(self, X_block):
        return pick_window(self.blocks, self.window, self.c_max)


class Recency(ContextPolicy):
    """P2: sample from all history, newer rows more likely (half-life 5 blocks)."""
    name = "recency"
    half_life = 5

    def select(self, X_block):
        return pick_recency(self.blocks, self.half_life, self.c_max, self.rng)


# ---------------------------------------------------------------------------
# P3: ADWIN drift detector on the model's errors
# ---------------------------------------------------------------------------
class AdwinPolicy(ContextPolicy):
    """P3: like vanilla, but when ADWIN sees the absolute error go UP,
    history before the detected change point is deleted."""
    name = "adwin"

    def __init__(self, model, c_max=C_MAX, seed=0, delta=0.002):
        super().__init__(model, c_max, seed)
        self.detector = drift.ADWIN(delta=delta)
        self.flag = 0

    def select(self, X_block):
        return pick_random(len(self.y), self.c_max, self.rng)

    def update(self, X_block, y_block):
        self.flag = 0
        error_went_up = False
        if self.last_pred is not None:          # block 0 is never predicted
            for e in np.abs(np.asarray(y_block) - self.last_pred):
                before = self.detector.estimation
                self.detector.update(float(e))
                # Only react to an INCREASE in error. A decrease (e.g. after we
                # pruned and the model got better again) is ignored.
                if self.detector.drift_detected and self.detector.estimation > before:
                    error_went_up = True
        super().update(X_block, y_block)
        if error_went_up:
            # ADWIN's window now only holds errors from after the change point,
            # so its width = how many of the newest rows to keep.
            keep = max(1, int(self.detector.width))
            self.X, self.y, self.blocks = self.X[-keep:], self.y[-keep:], self.blocks[-keep:]
            self.flag = 1

    def info(self):
        # The runner calls info() after predict() of the NEXT block, so the flag shows
        # on the first block predicted with the pruned context (detection block + 1).
        return {"drift_flag": self.flag, "n_context": self._n_context}


# ---------------------------------------------------------------------------
# P4: validation-guided choice between candidate rules
# ---------------------------------------------------------------------------
CANDIDATES = {
    "all":       lambda blocks, c, rng: pick_random(len(blocks), c, rng),   # = P0
    "window3":   lambda blocks, c, rng: pick_window(blocks, 3, c),
    "window5":   lambda blocks, c, rng: pick_window(blocks, 5, c),          # = P1
    "window10":  lambda blocks, c, rng: pick_window(blocks, 10, c),
    "recency2":  lambda blocks, c, rng: pick_recency(blocks, 2, c, rng),
    "recency5":  lambda blocks, c, rng: pick_recency(blocks, 5, c, rng),    # = P2
}


class ValidationPolicy(ContextPolicy):
    """P4: every time a block's labels arrive, test each candidate rule on that block
    (using only the history before it). Keep the scores of the last 2 blocks.
    Switch to the best candidate only if it beats the current one by more than 5%."""
    name = "validation"
    n_val_blocks = 2
    min_gain = 0.05

    def __init__(self, model, c_max=C_MAX, seed=0):
        super().__init__(model, c_max, seed)
        self.current = "all"                         # start like vanilla
        self.sq_errors = {k: [] for k in CANDIDATES}  # per candidate: [sum sq err, n] per block

    def select(self, X_block):
        return CANDIDATES[self.current](self.blocks, self.c_max, self.rng)

    def update(self, X_block, y_block):
        X_block, y_block = np.asarray(X_block), np.asarray(y_block)
        if self.y is not None:
            # Score each candidate on this newly labelled block. The context only
            # comes from history BEFORE this block, so this is a fair test.
            for name, rule in CANDIDATES.items():
                idx = rule(self.blocks, self.c_max, self.rng)
                self.model.fit(self.X[idx], self.y[idx])
                pred = np.asarray(self.model.predict(X_block)).ravel()
                self.sq_errors[name].append([np.sum((y_block - pred) ** 2), len(y_block)])
        super().update(X_block, y_block)

        if len(self.sq_errors["all"]) >= self.n_val_blocks:
            rmse = {}
            for name, rows in self.sq_errors.items():
                recent = np.array(rows[-self.n_val_blocks:])
                rmse[name] = np.sqrt(recent[:, 0].sum() / recent[:, 1].sum())
            best = min(rmse, key=rmse.get)
            if rmse[best] < (1 - self.min_gain) * rmse[self.current]:
                self.current = best


# ---------------------------------------------------------------------------
# kNN context: rows that look most like the incoming block
# ---------------------------------------------------------------------------
class KNNPolicy(ContextPolicy):
    """kNN_context: the C_MAX history rows closest (in feature space) to the incoming
    block. For each history row we take its distance to the nearest incoming row.
    Uses only the incoming FEATURES (known before prediction), never its labels."""
    name = "knn"

    def select(self, X_block):
        if len(self.y) <= self.c_max:
            return np.arange(len(self.y))
        dist = cdist(self.X, X_block).min(axis=1)
        return np.sort(np.argsort(dist)[: self.c_max])


POLICIES = {
    "vanilla": Vanilla,
    "sliding_window": SlidingWindow,
    "recency": Recency,
    "adwin": AdwinPolicy,
    "validation": ValidationPolicy,
    "knn": KNNPolicy,
}


def make_policy(name, model, c_max=C_MAX, seed=0):
    return POLICIES[name](model, c_max=c_max, seed=seed)
