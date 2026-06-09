import numpy as np

class _Node:

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
        self.class_counts = None
        self.prediction = None

class DecisionTreeClassifier:

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

        self._X_acc = None
        self._y_acc = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self._X_acc = X.copy()
        self._y_acc = y.copy()
        self.root_ = self._build(X, y, depth=0)
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
        self.n_features_ = self._X_acc.shape[1]
        self.root_ = self._build(self._X_acc, self._y_acc, depth=0)
        return self

    def predict(self, X):
        if self.root_ is None:
            raise RuntimeError("Call fit() or partial_fit() before predict().")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self.root_) for row in X])

    def predict_proba(self, X):
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

    def _build(self, X, y, depth):
        node = _Node()
        classes, counts = np.unique(y, return_counts=True)
        node.class_counts = np.zeros(len(self.classes_), dtype=int)
        for c, cnt in zip(classes, counts):
            idx = np.searchsorted(self.classes_, c)
            node.class_counts[idx] = cnt
        node.prediction = self.classes_[np.argmax(node.class_counts)]

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
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2)
        else:
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

    def _traverse(self, row, node):
        if node.is_leaf:
            return node
        if row[node.feature] <= node.threshold:
            return self._traverse(row, node.left)
        return self._traverse(row, node.right)

    def _predict_row(self, row, node):
        return self._traverse(row, node).prediction
