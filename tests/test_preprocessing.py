import numpy as np
import pytest
from numcompute_stream.preprocessing import (
    StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
)

class TestStandardScaler:
    def test_fit_transform_mean_zero(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        assert np.allclose(np.mean(X_s, axis=0), 0.0, atol=1e-10)

    def test_fit_transform_std_one(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        assert np.allclose(np.std(X_s, axis=0), 1.0, atol=1e-10)

    def test_transform_before_fit_raises(self):
        scaler = StandardScaler()
        with pytest.raises(RuntimeError):
            scaler.transform(np.array([[1.0, 2.0]]))

    def test_partial_fit_two_chunks(self):
        X1 = np.array([[1.0], [2.0], [3.0]])
        X2 = np.array([[4.0], [5.0], [6.0]])
        scaler = StandardScaler()
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)
        assert abs(scaler.mean_[0] - 3.5) < 0.01

    def test_constant_column_no_divide_by_zero(self):
        X = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        assert not np.any(np.isnan(X_s))
        assert np.allclose(X_s[:, 0], 0.0)

class TestMinMaxScaler:
    def test_range_zero_one(self):
        X = np.array([[1.0], [2.0], [3.0]])
        scaler = MinMaxScaler()
        X_s = scaler.fit_transform(X)
        assert X_s.min() == pytest.approx(0.0)
        assert X_s.max() == pytest.approx(1.0)

    def test_custom_range(self):
        X = np.array([[0.0], [10.0]])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_s = scaler.fit_transform(X)
        assert X_s.min() == pytest.approx(-1.0)
        assert X_s.max() == pytest.approx(1.0)

    def test_partial_fit_updates_range(self):
        scaler = MinMaxScaler()
        scaler.partial_fit(np.array([[1.0], [5.0]]))
        scaler.partial_fit(np.array([[0.0], [10.0]]))
        assert scaler.min_[0] == pytest.approx(0.0)
        assert scaler.max_[0] == pytest.approx(10.0)

    def test_transform_before_fit_raises(self):
        scaler = MinMaxScaler()
        with pytest.raises(RuntimeError):
            scaler.transform(np.array([[1.0]]))

class TestSimpleImputer:
    def test_constant_fills_nan(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        imp = SimpleImputer(fill_value=0.0)
        X_out = imp.fit_transform(X)
        assert X_out[0, 1] == 0.0
        assert not np.any(np.isnan(X_out))

    def test_custom_fill_value(self):
        X = np.array([[np.nan]])
        imp = SimpleImputer(fill_value=-99.0)
        X_out = imp.fit_transform(X)
        assert X_out[0, 0] == -99.0

    def test_mean_strategy(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        imp = SimpleImputer(strategy="mean")
        imp.fit(X)
        X_out = imp.transform(X)
        assert X_out[0, 1] == pytest.approx(4.0)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            SimpleImputer(strategy="median")

    def test_partial_fit_updates_mean(self):
        imp = SimpleImputer(strategy="mean")
        imp.partial_fit(np.array([[2.0], [4.0]]))
        imp.partial_fit(np.array([[6.0], [8.0]]))
        assert imp._running_mean[0] == pytest.approx(5.0)

class TestOneHotEncoder:
    def test_shape(self):
        X = np.array([0, 1, 2])
        enc = OneHotEncoder()
        X_enc = enc.fit_transform(X)
        assert X_enc.shape == (3, 3)

    def test_values(self):
        X = np.array([0, 1, 2])
        enc = OneHotEncoder()
        X_enc = enc.fit_transform(X)
        assert X_enc[0, 0] == 1.0
        assert X_enc[1, 1] == 1.0
        assert X_enc[2, 2] == 1.0

    def test_partial_fit_expands_categories(self):
        enc = OneHotEncoder()
        enc.partial_fit(np.array([0, 1]))
        enc.partial_fit(np.array([2, 3]))
        assert len(enc.categories_) == 4

    def test_transform_before_fit_raises(self):
        enc = OneHotEncoder()
        with pytest.raises(RuntimeError):
            enc.transform(np.array([0, 1]))
