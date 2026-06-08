"""
ensemble.py - Ensemble methods built on DecisionTreeClassifier.

Implements three ensemble strategies, all streaming-compatible via
partial_fit():

Classes
-------
RandomForestClassifier  : Bagging of decision trees with feature sub-sampling.
BaggingClassifier       : Generic bagging over any number of trees.
AdaBoostClassifier      : Discrete AdaBoost (SAMME) with streaming support.
EnsembleClassifier      : Unified wrapper - selects method via ``method`` param.
"""

import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# RandomForestClassifier
# ---------------------------------------------------------------------------

class RandomForestClassifier:
    """
    Random Forest: bagging of decision trees with random feature sub-sampling.

    Each tree is trained on a bootstrap sample of the accumulated data.
    partial_fit() appends the new chunk to the accumulated dataset and
    retrains all trees on fresh bootstrap samples.

    Parameters
    ----------
    n_estimators      : int    Number of trees (default 10).
    max_depth         : int    Maximum tree depth (default 5).
    min_samples_split : int    Minimum samples to split a node (default 2).
    max_features      : str    Features per split: 'sqrt' (default).
    criterion         : str    'gini' or 'entropy' (default 'gini').
    random_state      : int or None

    Attributes
    ----------
    estimators_ : list[DecisionTreeClassifier]
    classes_    : np.ndarray
    """

    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2,
                 max_features="sqrt", criterion="gini", random_state=None,
                 max_buffer=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.max_buffer = max_buffer

        self.estimators_ = []
        self.classes_ = None
        self._X_acc = None
        self._y_acc = None
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    def _make_tree(self, seed):
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            criterion=self.criterion,
            max_features=self.max_features,
            random_state=int(seed),
        )

    def fit(self, X, y):
        """Fit from scratch on (X, y)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._X_acc = X.copy()
        self._y_acc = y.copy()
        self.classes_ = np.unique(y)
        self._train_all()
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """Append chunk and refit all estimators on the retained buffer (bootstrap sampled)."""
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        if self._X_acc is None:
            self._X_acc = X_chunk
            self._y_acc = y_chunk
        else:
            self._X_acc = np.vstack([self._X_acc, X_chunk])
            self._y_acc = np.concatenate([self._y_acc, y_chunk])

        if self.max_buffer is not None and len(self._y_acc) > self.max_buffer:
            self._X_acc = self._X_acc[-self.max_buffer:]
            self._y_acc = self._y_acc[-self.max_buffer:]

        self.classes_ = np.unique(self._y_acc)
        self._train_all()
        return self

    def _train_all(self):
        """Build all trees using bootstrap sampling."""
        n = len(self._y_acc)
        seeds = self._rng.integers(0, 2 ** 31, size=self.n_estimators)
        self.estimators_ = []
        for seed in seeds:
            rng_local = np.random.default_rng(int(seed))
            idx = rng_local.choice(n, size=n, replace=True)
            tree = self._make_tree(int(seed))
            tree.fit(self._X_acc[idx], self._y_acc[idx])
            self.estimators_.append(tree)

    def predict(self, X):
        """Majority vote across all trees."""
        if not self.estimators_:
            raise RuntimeError("Call fit() or partial_fit() before predict().")
        X = np.asarray(X, dtype=float)
        # Collect votes: shape (n_estimators, n_samples)
        votes = np.array([tree.predict(X) for tree in self.estimators_])
        # Majority vote per sample
        predictions = []
        for col in votes.T:
            vals, counts = np.unique(col, return_counts=True)
            predictions.append(vals[np.argmax(counts)])
        return np.array(predictions)

    def predict_proba(self, X):
        """Average predicted probabilities across all trees."""
        if not self.estimators_:
            raise RuntimeError("Call fit() or partial_fit() before predict_proba().")
        X = np.asarray(X, dtype=float)
        # Align all trees to the global class set
        n_classes = len(self.classes_)
        proba_sum = np.zeros((len(X), n_classes))
        for tree in self.estimators_:
            tree_proba = tree.predict_proba(X)
            # Map tree classes to global classes
            for j, cls in enumerate(tree.classes_):
                global_j = np.searchsorted(self.classes_, cls)
                if global_j < n_classes:
                    proba_sum[:, global_j] += tree_proba[:, j]
        return proba_sum / len(self.estimators_)


# ---------------------------------------------------------------------------
# BaggingClassifier
# ---------------------------------------------------------------------------

class BaggingClassifier:
    """
    Generic bagging ensemble of decision trees.

    Similar to RandomForestClassifier but without mandatory feature
    sub-sampling (uses all features by default).

    Parameters
    ----------
    n_estimators      : int   Number of trees (default 10).
    max_depth         : int   Maximum depth (default 5).
    max_features      : value Passed to DecisionTreeClassifier (default None).
    random_state      : int or None
    """

    def __init__(self, n_estimators=10, max_depth=5,
                 max_features=None, random_state=None, max_buffer=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.max_buffer = max_buffer

        self.estimators_ = []
        self.classes_ = None
        self._X_acc = None
        self._y_acc = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._X_acc = X.copy()
        self._y_acc = y.copy()
        self.classes_ = np.unique(y)
        self._train_all()
        return self

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)
        if self._X_acc is None:
            self._X_acc = X_chunk
            self._y_acc = y_chunk
        else:
            self._X_acc = np.vstack([self._X_acc, X_chunk])
            self._y_acc = np.concatenate([self._y_acc, y_chunk])

        if self.max_buffer is not None and len(self._y_acc) > self.max_buffer:
            self._X_acc = self._X_acc[-self.max_buffer:]
            self._y_acc = self._y_acc[-self.max_buffer:]

        self.classes_ = np.unique(self._y_acc)
        self._train_all()
        return self

    def _train_all(self):
        n = len(self._y_acc)
        seeds = self._rng.integers(0, 2 ** 31, size=self.n_estimators)
        self.estimators_ = []
        for seed in seeds:
            rng_local = np.random.default_rng(int(seed))
            idx = rng_local.choice(n, size=n, replace=True)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=int(seed),
            )
            tree.fit(self._X_acc[idx], self._y_acc[idx])
            self.estimators_.append(tree)

    def predict(self, X):
        if not self.estimators_:
            raise RuntimeError("Call fit() or partial_fit() before predict().")
        X = np.asarray(X, dtype=float)
        votes = np.array([tree.predict(X) for tree in self.estimators_])
        predictions = []
        for col in votes.T:
            vals, counts = np.unique(col, return_counts=True)
            predictions.append(vals[np.argmax(counts)])
        return np.array(predictions)

    def predict_proba(self, X):
        if not self.estimators_:
            raise RuntimeError("Call fit() or partial_fit() before predict_proba().")
        X = np.asarray(X, dtype=float)
        n_classes = len(self.classes_)
        proba_sum = np.zeros((len(X), n_classes))
        for tree in self.estimators_:
            tp = tree.predict_proba(X)
            for j, cls in enumerate(tree.classes_):
                gi = np.searchsorted(self.classes_, cls)
                if gi < n_classes:
                    proba_sum[:, gi] += tp[:, j]
        return proba_sum / len(self.estimators_)


# ---------------------------------------------------------------------------
# AdaBoostClassifier  (SAMME – multi-class AdaBoost)
# ---------------------------------------------------------------------------

class AdaBoostClassifier:
    """
    Discrete AdaBoost (SAMME algorithm) with streaming support.

    Each call to partial_fit() appends data and re-runs the full boosting
    procedure on the accumulated dataset.

    Parameters
    ----------
    n_estimators : int   Number of boosting rounds (default 10).
    max_depth    : int   Max depth of each weak learner (default 1 = stump).
    learning_rate: float Shrinks each estimator's contribution (default 1.0).
    random_state : int or None

    Attributes
    ----------
    estimators_       : list[DecisionTreeClassifier]
    estimator_weights_: np.ndarray  Alpha weights for each estimator.
    classes_          : np.ndarray
    """

    def __init__(self, n_estimators=10, max_depth=1,
                 learning_rate=1.0, random_state=None, max_buffer=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.max_buffer = max_buffer

        self.estimators_ = []
        self.estimator_weights_ = []
        self.classes_ = None
        self._X_acc = None
        self._y_acc = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._X_acc = X.copy()
        self._y_acc = y.copy()
        self._boost()
        return self

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)
        if self._X_acc is None:
            self._X_acc = X_chunk
            self._y_acc = y_chunk
        else:
            self._X_acc = np.vstack([self._X_acc, X_chunk])
            self._y_acc = np.concatenate([self._y_acc, y_chunk])

        if self.max_buffer is not None and len(self._y_acc) > self.max_buffer:
            self._X_acc = self._X_acc[-self.max_buffer:]
            self._y_acc = self._y_acc[-self.max_buffer:]

        self._boost()
        return self

    def _boost(self):
        """Run the SAMME boosting loop on the full accumulated dataset."""
        X = self._X_acc
        y = self._y_acc
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n = len(y)
        w = np.full(n, 1.0 / n)

        self.estimators_ = []
        self.estimator_weights_ = []
        seeds = self._rng.integers(0, 2 ** 31, size=self.n_estimators)

        for seed in seeds:
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, random_state=int(seed)
            )
            tree.fit(X, y)
            y_pred = tree.predict(X)
            incorrect = (y_pred != y).astype(float)
            err = np.dot(w, incorrect)

            # Clip error to avoid division by zero or log(0)
            err = np.clip(err, 1e-10, 1 - 1e-10)

            # SAMME alpha
            alpha = self.learning_rate * (
                np.log((1.0 - err) / err) + np.log(K - 1)
            )
            # Update weights
            w *= np.exp(alpha * incorrect)
            w /= w.sum()

            self.estimators_.append(tree)
            self.estimator_weights_.append(alpha)

    def predict(self, X):
        """Weighted majority vote using SAMME alpha weights."""
        if not self.estimators_:
            raise RuntimeError("Call fit() or partial_fit() before predict().")
        X = np.asarray(X, dtype=float)
        K = len(self.classes_)
        n = len(X)
        class_scores = np.zeros((n, K))

        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            y_pred = tree.predict(X)
            for i, label in enumerate(y_pred):
                k = np.searchsorted(self.classes_, label)
                if k < K:
                    class_scores[i, k] += alpha

        return self.classes_[np.argmax(class_scores, axis=1)]

    def predict_proba(self, X):
        """Softmax-normalised class scores."""
        if not self.estimators_:
            raise RuntimeError("Call fit() or partial_fit() before predict_proba().")
        X = np.asarray(X, dtype=float)
        K = len(self.classes_)
        n = len(X)
        class_scores = np.zeros((n, K))

        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            y_pred = tree.predict(X)
            for i, label in enumerate(y_pred):
                k = np.searchsorted(self.classes_, label)
                if k < K:
                    class_scores[i, k] += alpha

        # Softmax normalisation
        shifted = class_scores - class_scores.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        return exp_s / exp_s.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# EnsembleClassifier  (unified wrapper)
# ---------------------------------------------------------------------------

class EnsembleClassifier:
    """
    Unified interface for all ensemble methods.

    Selects the underlying implementation via the ``method`` parameter
    and forwards all calls transparently.

    Parameters
    ----------
    method       : str   'random_forest' | 'bagging' | 'adaboost'
                         (default 'random_forest').
    n_estimators : int   Number of base estimators (default 10).
    max_depth    : int   Maximum tree depth (default 5).
    **kwargs           Additional keyword arguments forwarded to the
                         chosen ensemble class.

    Examples
    --------
    >>> clf = EnsembleClassifier(method='random_forest', n_estimators=20)
    >>> clf.partial_fit(X_chunk, y_chunk)
    >>> clf.predict(X_test)
    """

    _METHOD_MAP = {
        "random_forest": RandomForestClassifier,
        "bagging": BaggingClassifier,
        "adaboost": AdaBoostClassifier,
    }

    def __init__(self, method="random_forest", n_estimators=10,
                 max_depth=5, max_buffer=None, **kwargs):
        if method not in self._METHOD_MAP:
            raise ValueError(
                f"method must be one of {list(self._METHOD_MAP)}. Got '{method}'."
            )
        self.method = method
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_buffer = max_buffer
        self._model = self._METHOD_MAP[method](
            n_estimators=n_estimators, max_depth=max_depth,
            max_buffer=max_buffer, **kwargs
        )

    @property
    def classes_(self):
        return getattr(self._model, "classes_", None)

    @property
    def estimators_(self):
        return getattr(self._model, "estimators_", [])

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        self._model.partial_fit(X_chunk, y_chunk)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def __repr__(self):
        return (f"EnsembleClassifier(method={self.method!r}, "
                f"n_estimators={self.n_estimators}, "
                f"max_depth={self.max_depth})")
