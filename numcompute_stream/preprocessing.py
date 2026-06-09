import numpy as np

class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.n_seen_ = None

    def partial_fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        valid = ~np.isnan(X)
        n = valid.sum(axis=0).astype(float)

        if self.mean_ is None:
            self.mean_ = np.zeros(X.shape[1])
            self.var_ = np.zeros(X.shape[1])
            self.n_seen_ = np.zeros(X.shape[1])

        with np.errstate(invalid="ignore"):
            chunk_mean = np.nanmean(np.where(valid, X, np.nan), axis=0)
            chunk_var  = np.nanvar( np.where(valid, X, np.nan), axis=0)
        chunk_mean = np.where(n > 0, chunk_mean, 0.0)
        chunk_var  = np.where(n > 0, chunk_var,  0.0)

        n_prev  = self.n_seen_
        n_total = n_prev + n
        safe    = np.where(n_total > 0, n_total, 1.0)

        delta    = chunk_mean - self.mean_
        new_mean = (n_prev * self.mean_ + n * chunk_mean) / safe
        new_var  = (n_prev * self.var_ + n * chunk_var
                    + delta ** 2 * n_prev * n / safe) / safe

        self.mean_   = np.where(n > 0, new_mean, self.mean_)
        self.var_    = np.where(n > 0, new_var,  self.var_)
        self.n_seen_ = n_total
        return self

    def fit(self, X):
        self.mean_ = None
        self.var_ = None
        self.n_seen_ = None
        return self.partial_fit(X)

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("Call fit() or partial_fit() before transform().")
        X = np.asarray(X, dtype=float)
        std = np.sqrt(self.var_)
        std[std == 0] = 1.0
        return (X - self.mean_) / std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class MinMaxScaler:

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None

    def partial_fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)
        if self.min_ is None:
            self.min_ = chunk_min
            self.max_ = chunk_max
        else:
            self.min_ = np.minimum(self.min_, chunk_min)
            self.max_ = np.maximum(self.max_, chunk_max)
        return self

    def fit(self, X):
        self.min_ = None
        self.max_ = None
        return self.partial_fit(X)

    def transform(self, X):
        if self.min_ is None:
            raise RuntimeError("Call fit() or partial_fit() before transform().")
        X = np.asarray(X, dtype=float)
        scale = self.max_ - self.min_
        scale[scale == 0] = 1.0
        low, high = self.feature_range
        return (X - self.min_) / scale * (high - low) + low

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class SimpleImputer:

    def __init__(self, strategy="constant", fill_value=0.0):
        if strategy not in ("constant", "mean"):
            raise ValueError("strategy must be 'constant' or 'mean'.")
        self.strategy = strategy
        self.fill_value = fill_value
        self._running_mean = None
        self._n_seen = 0

    def partial_fit(self, X):
        if self.strategy == "mean":
            X = np.asarray(X, dtype=float)
            valid = ~np.isnan(X)
            n = valid.sum(axis=0).astype(float)
            with np.errstate(invalid="ignore"):
                chunk_mean = np.nanmean(np.where(valid, X, np.nan), axis=0)
            chunk_mean = np.where(n > 0, chunk_mean, 0.0)

            if self._running_mean is None:
                self._running_mean = chunk_mean.copy()
                self._n_seen = n.copy()
            else:
                n_prev  = self._n_seen
                n_total = n_prev + n
                safe    = np.where(n_total > 0, n_total, 1.0)
                new_mean = (n_prev * self._running_mean + n * chunk_mean) / safe
                self._running_mean = np.where(n > 0, new_mean, self._running_mean)
                self._n_seen = n_total
        return self

    def fit(self, X):
        self._running_mean = None
        self._n_seen = 0
        return self.partial_fit(X)

    def transform(self, X):
        X_out = np.array(X, dtype=float)
        if self.strategy == "constant":
            X_out[np.isnan(X_out)] = self.fill_value
        else:
            if self._running_mean is None:
                raise RuntimeError("Call fit() or partial_fit() before transform().")
            for col in range(X_out.shape[1]):
                mask = np.isnan(X_out[:, col])
                X_out[mask, col] = self._running_mean[col]
        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class OneHotEncoder:

    def __init__(self):
        self.categories_ = None

    def partial_fit(self, X):
        X = np.asarray(X).astype(int).flatten()
        new_cats = np.unique(X)
        if self.categories_ is None:
            self.categories_ = new_cats
        else:
            self.categories_ = np.unique(np.concatenate([self.categories_, new_cats]))
        return self

    def fit(self, X):
        self.categories_ = None
        return self.partial_fit(X)

    def transform(self, X):
        if self.categories_ is None:
            raise RuntimeError("Call fit() or partial_fit() before transform().")
        X = np.asarray(X).astype(int).flatten()
        return (X[:, None] == self.categories_[None, :]).astype(float)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
