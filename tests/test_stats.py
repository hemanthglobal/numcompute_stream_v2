"""Tests for streaming statistics."""
import numpy as np
import pytest
from numcompute_stream.stats import (
    StreamingStats, mean, median, std, histogram, quantile, summary
)


class TestStreamingStats:
    def test_update_mean(self):
        ss = StreamingStats()
        ss.update_stats(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert np.allclose(ss.mean_, [2.0, 3.0])

    def test_two_chunk_mean(self):
        ss = StreamingStats()
        ss.update_stats(np.array([[1.0], [3.0]]))
        ss.update_stats(np.array([[5.0], [7.0]]))
        # Overall mean of 1,3,5,7 = 4.0
        assert ss.mean_[0] == pytest.approx(4.0)

    def test_running_min_max(self):
        ss = StreamingStats()
        ss.update_stats(np.array([[2.0], [8.0]]))
        ss.update_stats(np.array([[1.0], [9.0]]))
        assert ss.min_[0] == pytest.approx(1.0)
        assert ss.max_[0] == pytest.approx(9.0)

    def test_n_seen(self):
        ss = StreamingStats()
        ss.update_stats(np.arange(10).reshape(5, 2))
        ss.update_stats(np.arange(10).reshape(5, 2))
        # n_seen_ is now a per-feature array after the NaN-weighting fix
        assert np.all(ss.n_seen_ == 10)

    def test_reset(self):
        ss = StreamingStats()
        ss.update_stats(np.array([[1.0, 2.0]]))
        ss.reset()
        assert ss.mean_ is None
        assert ss.n_seen_ == 0

    def test_std_property(self):
        ss = StreamingStats()
        ss.update_stats(np.array([[2.0], [4.0], [4.0], [4.0],
                                   [5.0], [5.0], [7.0], [9.0]]))
        assert ss.std_()[0] == pytest.approx(2.0, abs=1e-5)

    def test_windowed_quantile(self):
        ss = StreamingStats(window_size=20)
        ss.update_stats(np.arange(10).reshape(10, 1).astype(float))
        q = ss.windowed_quantile(50)
        assert q[0] == pytest.approx(4.5)

    def test_windowed_quantile_no_window_raises(self):
        ss = StreamingStats()
        ss.update_stats(np.array([[1.0]]))
        with pytest.raises(RuntimeError):
            ss.windowed_quantile(50)

    def test_windowed_histogram(self):
        ss = StreamingStats(window_size=100)
        ss.update_stats(np.arange(10).reshape(10, 1).astype(float))
        counts, edges = ss.windowed_histogram(col=0, bins=5)
        assert len(counts) == 5
        assert counts.sum() == 10

    def test_summary_keys(self):
        ss = StreamingStats()
        ss.update_stats(np.array([[1.0, 2.0], [3.0, 4.0]]))
        s = ss.summary()
        for key in ["mean", "std", "min", "max", "n_seen"]:
            assert key in s


class TestBatchStats:
    def test_mean_nan(self):
        assert mean(np.array([1.0, np.nan, 3.0])) == pytest.approx(2.0)

    def test_median(self):
        assert median(np.array([1.0, 3.0, 5.0])) == pytest.approx(3.0)

    def test_std(self):
        data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert std(data) == pytest.approx(2.0)

    def test_quantile(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert quantile(data, 0.5) == pytest.approx(3.0)

    def test_quantile_all_nan_raises(self):
        with pytest.raises(ValueError):
            quantile(np.array([np.nan, np.nan]), 0.5)

    def test_histogram_bins(self):
        counts, edges = histogram(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), bins=5)
        assert len(counts) == 5
        assert counts.sum() == 5

    def test_summary_keys(self):
        s = summary(np.array([[1.0, 2.0], [3.0, 4.0]]))
        for key in ["mean", "median", "std", "min", "max"]:
            assert key in s
