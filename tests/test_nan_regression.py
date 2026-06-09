import numpy as np
import pytest
from numcompute_stream.preprocessing import StandardScaler, SimpleImputer
from numcompute_stream.stats import StreamingStats

class TestNaNRegressionScaler:
    def test_partial_fit_nan_mean_matches_nanmean(self):
        X1 = np.array([[1.0,  np.nan],
                        [3.0,  np.nan],
                        [5.0,  np.nan]])

        X2 = np.array([[2.0, 10.0],
                        [4.0, 20.0],
                        [6.0, 30.0]])

        scaler = StandardScaler()
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)

        X_all = np.vstack([X1, X2])
        expected_mean = np.nanmean(X_all, axis=0)

        np.testing.assert_allclose(scaler.mean_, expected_mean, atol=1e-10,
            err_msg="Running mean does not match np.nanmean on combined data.")

    def test_partial_fit_nan_var_matches_nanvar(self):
        X1 = np.array([[1.0, np.nan],
                        [3.0, np.nan]])
        X2 = np.array([[5.0, 10.0],
                        [7.0, 20.0]])

        scaler = StandardScaler()
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)

        X_all = np.vstack([X1, X2])
        expected_var = np.nanvar(X_all, axis=0)

        np.testing.assert_allclose(scaler.var_, expected_var, atol=1e-8,
            err_msg="Running variance does not match np.nanvar on combined data.")

    def test_all_nan_column_stays_zero(self):
        X = np.array([[np.nan, 1.0],
                      [np.nan, 2.0],
                      [np.nan, 3.0]])
        scaler = StandardScaler()
        scaler.partial_fit(X)
        assert np.isfinite(scaler.mean_[0]), "mean_ has non-finite value for all-NaN column"
        assert np.isfinite(scaler.var_[0]),  "var_ has non-finite value for all-NaN column"

    def test_transform_no_nan_output(self):
        X_fit = np.array([[1.0, np.nan],
                          [np.nan, 2.0],
                          [3.0, 4.0]])
        X_transform = np.array([[2.0, 3.0]])
        scaler = StandardScaler()
        scaler.fit(X_fit)
        out = scaler.transform(X_transform)
        assert np.all(np.isfinite(out)), f"transform output contains non-finite: {out}"

    def test_n_seen_is_per_feature_array(self):
        X = np.array([[1.0, np.nan],
                      [2.0, 3.0]])
        scaler = StandardScaler()
        scaler.partial_fit(X)
        assert hasattr(scaler.n_seen_, "__len__"), "n_seen_ should be an array after fix"
        assert scaler.n_seen_[0] == 2
        assert scaler.n_seen_[1] == 1

class TestNaNRegressionStats:
    def test_streaming_mean_matches_nanmean(self):
        X1 = np.array([[1.0, np.nan],
                        [3.0, np.nan]])
        X2 = np.array([[5.0, 10.0],
                        [7.0, 20.0]])

        ss = StreamingStats()
        ss.update_stats(X1)
        ss.update_stats(X2)

        X_all = np.vstack([X1, X2])
        expected = np.nanmean(X_all, axis=0)
        np.testing.assert_allclose(ss.mean_, expected, atol=1e-10)

    def test_streaming_var_matches_nanvar(self):
        X1 = np.array([[2.0, np.nan],
                        [4.0, np.nan]])
        X2 = np.array([[6.0, 5.0],
                        [8.0, 15.0]])

        ss = StreamingStats()
        ss.update_stats(X1)
        ss.update_stats(X2)

        X_all = np.vstack([X1, X2])
        expected_var = np.nanvar(X_all, axis=0)
        np.testing.assert_allclose(ss.var_, expected_var, atol=1e-8)

    def test_all_nan_column_finite(self):
        X = np.array([[np.nan], [np.nan], [np.nan]])
        ss = StreamingStats()
        ss.update_stats(X)
        assert np.isfinite(ss.mean_[0])
        assert np.isfinite(ss.var_[0])

    def test_single_row_chunk(self):
        ss = StreamingStats()
        ss.update_stats(np.array([[5.0, 10.0]]))
        assert ss.mean_[0] == pytest.approx(5.0)
        assert ss.mean_[1] == pytest.approx(10.0)
        assert ss.n_seen_[0] == 1

    def test_reset_clears_state(self):
        ss = StreamingStats()
        ss.update_stats(np.array([[1.0, 2.0]]))
        ss.reset()
        assert ss.mean_ is None
        assert ss.n_seen_ == 0

class TestNaNRegressionImputer:
    def test_mean_strategy_nan_weighted(self):
        X1 = np.array([[1.0, np.nan],
                        [3.0, np.nan]])
        X2 = np.array([[5.0, 10.0]])

        imp = SimpleImputer(strategy="mean")
        imp.partial_fit(X1)
        imp.partial_fit(X2)

        assert imp._running_mean[0] == pytest.approx(3.0, abs=1e-8)
        assert imp._running_mean[1] == pytest.approx(10.0, abs=1e-8)
