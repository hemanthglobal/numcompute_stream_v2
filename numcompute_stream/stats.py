import numpy as np

class StreamingStats:

    def __init__(self, window_size=None):
        self.window_size = window_size
        self.mean_ = None
        self.var_ = None
        self.min_ = None
        self.max_ = None
        self.n_seen_ = 0
        self.history_ = []

    def update_stats(self, X_chunk):
        X = np.asarray(X_chunk, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        valid = ~np.isnan(X)
        n = valid.sum(axis=0).astype(float)
        with np.errstate(invalid="ignore"):
            chunk_mean = np.nanmean(np.where(valid, X, np.nan), axis=0)
            chunk_var  = np.nanvar( np.where(valid, X, np.nan), axis=0)
            chunk_min  = np.nanmin( np.where(valid, X, np.nan), axis=0)
            chunk_max  = np.nanmax( np.where(valid, X, np.nan), axis=0)
        chunk_mean = np.where(n > 0, chunk_mean, 0.0)
        chunk_var  = np.where(n > 0, chunk_var,  0.0)

        if self.mean_ is None:
            self.mean_ = chunk_mean.copy()
            self.var_  = chunk_var.copy()
            self.min_  = chunk_min.copy()
            self.max_  = chunk_max.copy()
            self.n_seen_ = n.copy()
        else:
            n_prev  = self.n_seen_
            n_total = n_prev + n
            safe    = np.where(n_total > 0, n_total, 1.0)
            delta    = chunk_mean - self.mean_
            new_mean = (n_prev * self.mean_ + n * chunk_mean) / safe
            new_var  = (n_prev * self.var_ + n * chunk_var
                        + delta ** 2 * n_prev * n / safe) / safe
            self.mean_   = np.where(n > 0, new_mean, self.mean_)
            self.var_    = np.where(n > 0, new_var,  self.var_)
            self.min_    = np.minimum(self.min_, chunk_min)
            self.max_    = np.maximum(self.max_, chunk_max)
            self.n_seen_ = n_total

        if self.window_size is not None:
            self.history_.append(X)
            total = sum(a.shape[0] for a in self.history_)
            while total - self.history_[0].shape[0] >= self.window_size:
                total -= self.history_[0].shape[0]
                self.history_.pop(0)

        return self

    def std_(self):
        if self.var_ is None:
            return None
        return np.sqrt(self.var_)

    def windowed_quantile(self, q, axis=0):
        if self.window_size is None:
            raise RuntimeError("window_size must be set to use windowed_quantile.")
        if not self.history_:
            raise ValueError("No data in window history yet.")
        data = np.concatenate(self.history_, axis=0)
        return np.nanpercentile(data, q, axis=axis)

    def windowed_histogram(self, col=0, bins=10):
        if self.window_size is None:
            raise RuntimeError("window_size must be set to use windowed_histogram.")
        if not self.history_:
            raise ValueError("No data in window history yet.")
        data = np.concatenate(self.history_, axis=0)[:, col]
        data = data[~np.isnan(data)]
        return np.histogram(data, bins=bins)

    def summary(self):
        return {
            "mean": self.mean_,
            "std": self.std_(),
            "min": self.min_,
            "max": self.max_,
            "n_seen": int(self.n_seen_.max()) if hasattr(self.n_seen_, "__len__") else self.n_seen_,
        }

    def reset(self):
        self.mean_ = None
        self.var_ = None
        self.min_ = None
        self.max_ = None
        self.n_seen_ = 0
        self.history_ = []
        return self

def mean(X, axis=None):
    return np.nanmean(X, axis=axis)

def median(X, axis=None):
    return np.nanmedian(X, axis=axis)

def std(X, axis=None):
    return np.nanstd(X, axis=axis)

def minimum(X, axis=None):
    return np.nanmin(X, axis=axis)

def maximum(X, axis=None):
    return np.nanmax(X, axis=axis)

def histogram(data, bins=10):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    return np.histogram(data, bins=bins)

def quantile(data, q, interpolation="linear"):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        raise ValueError("Cannot compute quantile of an all-NaN array.")
    return np.percentile(data, q * 100, method=interpolation)

def summary(X, axis=0):
    return {
        "mean": mean(X, axis=axis),
        "median": median(X, axis=axis),
        "std": std(X, axis=axis),
        "min": minimum(X, axis=axis),
        "max": maximum(X, axis=axis),
    }
