"""
tree.py - Decision Tree Classifier with streaming support.

Implements a depth-limited CART decision tree using Gini impurity or
information entropy as the split criterion.  partial_fit() supports
chunk-wise refit on the retained buffer (optionally capped via max_buffer).

Classes
-------
DecisionTreeClassifier : Depth-limited decision tree; supports partial_fit().
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal node representation
# ---------------------------------------------------------------------------

class _Node:
    """A single node in the decision tree."""

    __slots__ = (
        "feature", "threshold", "left", "right",
        "is_leaf", "class_counts", "prediction",
    )

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.class_counts = None   # np.ndarray of counts per class
        self.prediction = None     # majority class at this node


# ---------------------------------------------------------------------------
# DecisionTreeClassifier
# ---------------------------------------------------------------------------

class DecisionTreeClassifier:
    """
    Binary/multi-class decision tree classifier.

    Uses vectorised NumPy operations for split search.  partial_fit()
    accumulates data and performs a chunk-wise refit on the retained buffer.

    Parameters
    ----------
    max_depth         : int   Maximum tree depth (default 5).
    min_samples_split : int   Minimum samples required to attempt a split
                              (default 2).
    criterion         : str   Split quality measure: 'gini' or 'entropy'
                              (default 'gini').
    max_features      : int, float, str, or None
                        Number of features to consider at each split.
                        - int   : use that many features.
                        - float : use that fraction of features.
                        - 'sqrt': use sqrt(n_features).
                        - 'log2': use log2(n_features).
                        - None  : use all features.
    random_state      : int or None  Seed for feature sub-sampling.

    Attributes
    ----------
    root_       : _Node or None  Root of the fitted tree.
    classes_    : np.ndarray     Sorted unique class labels seen so far.
    n_features_ : int            Number of features seen during fit.
    """

    def __init__(self, max_depth=5, min_samples_split=2,
                 criterion="gini", max_features=None, random_state=None,
                 max_buffer=None):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.max_buffer = max_buffer

        self.root_ = None
        self.classes_ = None
        self.n_features_ = None

        # Accumulated data for partial_fit
        self._X_acc = None
        self._y_acc = None
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the tree from scratch on (X, y).

        Parameters
        ----------
        X : np.ndarray  Shape (n_samples, n_features).
        y : np.ndarray  Shape (n_samples,).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self._X_acc = X.copy()
        self._y_acc = y.copy()
        self.root_ = self._build(X, y, depth=0)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """
        Append chunk and refit on the retained buffer.

        Appends the chunk to the accumulated dataset and rebuilds the
        tree, keeping full compatibility with prior observations.

        Parameters
        ----------
        X_chunk : np.ndarray  Shape (n_samples, n_features).
        y_chunk : np.ndarray  Shape (n_samples,).

        Returns
        -------
        self
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        if self._X_acc is None:
            self._X_acc = X_chunk
            self._y_acc = y_chunk
        else:
            self._X_acc = np.vstack([self._X_acc, X_chunk])
            self._y_acc = np.concatenate([self._y_acc, y_chunk])

        # Cap buffer size if max_buffer is set (sliding-window streaming)
        if self.max_buffer is not None and len(self._y_acc) > self.max_buffer:
            self._X_acc = self._X_acc[-self.max_buffer:]
            self._y_acc = self._y_acc[-self.max_buffer:]

        self.classes_ = np.unique(self._y_acc)
        self.n_features_ = self._X_acc.shape[1]
        self.root_ = self._build(self._X_acc, self._y_acc, depth=0)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray  Shape (n_samples, n_features).

        Returns
        -------
        np.ndarray  Shape (n_samples,).

        Raises
        ------
        RuntimeError  If called before fit / partial_fit.
        """
        if self.root_ is None:
            raise RuntimeError("Call fit() or partial_fit() before predict().")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self.root_) for row in X])

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Returns
        -------
        np.ndarray  Shape (n_samples, n_classes).
        """
        if self.root_ is None:
            raise RuntimeError("Call fit() or partial_fit() before predict_proba().")
        X = np.asarray(X, dtype=float)
        n_classes = len(self.classes_)
        proba = np.zeros((len(X), n_classes))
        for i, row in enumerate(X):
            node = self._traverse(row, self.root_)
            total = node.class_counts.sum()
            proba[i] = node.class_counts / total if total > 0 else np.ones(n_classes) / n_classes
        return proba

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build(self, X, y, depth):
        node = _Node()
        classes, counts = np.unique(y, return_counts=True)
        node.class_counts = np.zeros(len(self.classes_), dtype=int)
        for c, cnt in zip(classes, counts):
            idx = np.searchsorted(self.classes_, c)
            node.class_counts[idx] = cnt
        node.prediction = self.classes_[np.argmax(node.class_counts)]

        # Stopping conditions
        if (depth >= self.max_depth
                or len(y) < self.min_samples_split
                or len(np.unique(y)) == 1):
            node.is_leaf = True
            return node

        feat, thr = self._best_split(X, y)
        if feat is None:
            node.is_leaf = True
            return node

        node.feature = feat
        node.threshold = thr
        left_mask = X[:, feat] <= thr
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(self, X, y):
        """Find the feature and threshold that minimise the impurity."""
        n_samples, n_feats = X.shape
        n_use = self._n_features_to_use(n_feats)
        feat_indices = self._rng.choice(n_feats, size=n_use, replace=False)

        best_gain = -np.inf
        best_feat = None
        best_thr = None
        parent_impurity = self._impurity(y)

        for feat in feat_indices:
            col = X[:, feat]
            thresholds = np.unique(col)
            if len(thresholds) < 2:
                continue
            # Mid-points between consecutive unique values
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2

            for thr in thresholds:
                left_mask = col <= thr
                n_left = left_mask.sum()
                n_right = n_samples - n_left
                if n_left == 0 or n_right == 0:
                    continue
                gain = parent_impurity - (
                    n_left / n_samples * self._impurity(y[left_mask])
                    + n_right / n_samples * self._impurity(y[~left_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = thr

        return best_feat, best_thr

    def _impurity(self, y):
        """Compute Gini or entropy impurity for label array y."""
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2)
        else:  # entropy
            # Clip to avoid log(0)
            return -np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0)))

    def _n_features_to_use(self, n_feats):
        mf = self.max_features
        if mf is None:
            return n_feats
        if isinstance(mf, int):
            return min(mf, n_feats)
        if isinstance(mf, float):
            return max(1, int(mf * n_feats))
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_feats)))
        if mf == "log2":
            return max(1, int(np.log2(n_feats)))
        raise ValueError(f"Unknown max_features value: {mf!r}")

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _traverse(self, row, node):
        """Walk the tree for a single sample; return leaf node."""
        if node.is_leaf:
            return node
        if row[node.feature] <= node.threshold:
            return self._traverse(row, node.left)
        return self._traverse(row, node.right)

    def _predict_row(self, row, node):
        return self._traverse(row, node).prediction
