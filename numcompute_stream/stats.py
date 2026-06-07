"""
stats.py - Streaming-compatible descriptive statistics.

The StreamingStats class maintains running estimates of common statistics
(mean, variance, min, max, histogram) via incremental chunk updates.
Module-level functions operate on full arrays for batch use.

Classes
-------
StreamingStats : Maintains running stats across chunks via update_stats().

Functions
---------
mean, median, std, minimum, maximum : Batch NaN-aware statistics.
histogram    : Bin counts for a 1-D array.
quantile     : Percentile with NaN removal.
summary      : Dict of {mean, median, std, min, max}.
"""

import numpy as np


# ---------------------------------------------------------------------------
# StreamingStats
# ---------------------------------------------------------------------------

class StreamingStats:
    """
    Accumulate descriptive statistics chunk-by-chunk.

    Maintains per-column running estimates of mean, variance, min, and max
    using Welford's online algorithm.  An optional sliding window keeps a
    fixed-length history for windowed quantile estimates.

    Parameters
    ----------
    window_size : int or None
        If set, only the last ``window_size`` samples are stored for
        windowed operations (e.g. quantile).  If None, no history is kept.

    Attributes
    ----------
    mean_    : np.ndarray  Running column means.
    var_     : np.ndarray  Running column variances.
    min_     : np.ndarray  Running column minima.
    max_     : np.ndarray  Running column maxima.
    n_seen_  : int         Total samples processed.
    history_ : list        Stored chunks (only when window_size is set).
    """

    def __init__(self, window_size=None):
        self.window_size = window_size
        self.mean_ = None
        self.var_ = None
        self.min_ = None
        self.max_ = None
        self.n_seen_ = 0
        self.history_ = []

    # ------------------------------------------------------------------
    def update_stats(self, X_chunk):
        """
        Incorporate a new chunk into running statistics.

        Parameters
        ----------
        X_chunk : np.ndarray  Shape (n_samples, n_features) or (n_samples,).

        Returns
        -------
        self
        """
        X = np.asarray(X_chunk, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        valid = ~np.isnan(X)
        n = valid.sum(axis=0).astype(float)              # per-feature non-NaN counts
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

        # Sliding window history
        if self.window_size is not None:
            self.history_.append(X)
            total = sum(a.shape[0] for a in self.history_)
            while total - self.history_[0].shape[0] >= self.window_size:
                total -= self.history_[0].shape[0]
                self.history_.pop(0)

        return self

    def std_(self):
        """Return per-column running standard deviation."""
        if self.var_ is None:
            return None
        return np.sqrt(self.var_)

    def windowed_quantile(self, q, axis=0):
        """
        Compute quantile over the current sliding-window history.

        Parameters
        ----------
        q    : float  Quantile in [0, 100].
        axis : int    Axis along which to compute (default 0 = column-wise).

        Returns
        -------
        np.ndarray or float

        Raises
        ------
        RuntimeError  If no window_size was configured.
        ValueError    If the history is empty.
        """
        if self.window_size is None:
            raise RuntimeError("window_size must be set to use windowed_quantile.")
        if not self.history_:
            raise ValueError("No data in window history yet.")
        data = np.concatenate(self.history_, axis=0)
        return np.nanpercentile(data, q, axis=axis)

    def windowed_histogram(self, col=0, bins=10):
        """
        Compute a histogram over the sliding-window history for one column.

        Parameters
        ----------
        col  : int  Column index.
        bins : int  Number of histogram bins.

        Returns
        -------
        counts    : np.ndarray
        bin_edges : np.ndarray
        """
        if self.window_size is None:
            raise RuntimeError("window_size must be set to use windowed_histogram.")
        if not self.history_:
            raise ValueError("No data in window history yet.")
        data = np.concatenate(self.history_, axis=0)[:, col]
        data = data[~np.isnan(data)]
        return np.histogram(data, bins=bins)

    def summary(self):
        """Return a dict of current running statistics."""
        return {
            "mean": self.mean_,
            "std": self.std_(),
            "min": self.min_,
            "max": self.max_,
            "n_seen": int(self.n_seen_.max()) if hasattr(self.n_seen_, "__len__") else self.n_seen_,
        }

    def reset(self):
        """Reset all accumulated state."""
        self.mean_ = None
        self.var_ = None
        self.min_ = None
        self.max_ = None
        self.n_seen_ = 0
        self.history_ = []
        return self


# ---------------------------------------------------------------------------
# Batch helper functions (NaN-aware)
# ---------------------------------------------------------------------------

def mean(X, axis=None):
    """NaN-aware mean."""
    return np.nanmean(X, axis=axis)


def median(X, axis=None):
    """NaN-aware median."""
    return np.nanmedian(X, axis=axis)


def std(X, axis=None):
    """NaN-aware standard deviation."""
    return np.nanstd(X, axis=axis)


def minimum(X, axis=None):
    """NaN-aware minimum."""
    return np.nanmin(X, axis=axis)


def maximum(X, axis=None):
    """NaN-aware maximum."""
    return np.nanmax(X, axis=axis)


def histogram(data, bins=10):
    """
    Compute a histogram, ignoring NaNs.

    Parameters
    ----------
    data : array-like  1-D data.
    bins : int         Number of bins.

    Returns
    -------
    counts    : np.ndarray
    bin_edges : np.ndarray
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    return np.histogram(data, bins=bins)


def quantile(data, q, interpolation="linear"):
    """
    Compute quantile after removing NaNs.

    Parameters
    ----------
    data          : array-like  Input data.
    q             : float       Quantile in [0, 1].
    interpolation : str         NumPy percentile interpolation method.

    Returns
    -------
    float

    Raises
    ------
    ValueError  If the array is all-NaN.
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        raise ValueError("Cannot compute quantile of an all-NaN array.")
    return np.percentile(data, q * 100, method=interpolation)


def summary(X, axis=0):
    """
    Return a dict of common statistics computed along ``axis``.

    Keys: 'mean', 'median', 'std', 'min', 'max'.
    """
    return {
        "mean": mean(X, axis=axis),
        "median": median(X, axis=axis),
        "std": std(X, axis=axis),
        "min": minimum(X, axis=axis),
        "max": maximum(X, axis=axis),
    }
