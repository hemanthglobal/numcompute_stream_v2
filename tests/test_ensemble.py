"""Tests for ensemble methods."""
import numpy as np
import pytest
from numcompute_stream.ensemble import (
    RandomForestClassifier, BaggingClassifier,
    AdaBoostClassifier, EnsembleClassifier
)


def make_data(n=150, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


class TestRandomForest:
    def test_fit_predict(self):
        X, y = make_data()
        rf = RandomForestClassifier(n_estimators=5, max_depth=3)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert preds.shape == (len(y),)

    def test_partial_fit(self):
        X, y = make_data(200)
        rf = RandomForestClassifier(n_estimators=5, max_depth=3)
        rf.partial_fit(X[:100], y[:100])
        rf.partial_fit(X[100:], y[100:])
        preds = rf.predict(X)
        assert preds.shape == (200,)

    def test_accuracy_better_than_chance(self):
        X, y = make_data(300)
        rf = RandomForestClassifier(n_estimators=10, max_depth=4)
        rf.fit(X, y)
        acc = np.mean(rf.predict(X) == y)
        assert acc > 0.6

    def test_predict_before_fit_raises(self):
        rf = RandomForestClassifier()
        with pytest.raises(RuntimeError):
            rf.predict(np.array([[1.0, 2.0, 3.0, 4.0]]))

    def test_predict_proba_shape(self):
        X, y = make_data()
        rf = RandomForestClassifier(n_estimators=5)
        rf.fit(X, y)
        proba = rf.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self):
        X, y = make_data()
        rf = RandomForestClassifier(n_estimators=5)
        rf.fit(X, y)
        proba = rf.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestBagging:
    def test_fit_predict(self):
        X, y = make_data()
        bag = BaggingClassifier(n_estimators=5, max_depth=3)
        bag.fit(X, y)
        assert bag.predict(X).shape == (len(y),)

    def test_partial_fit(self):
        X, y = make_data(200)
        bag = BaggingClassifier(n_estimators=5)
        bag.partial_fit(X[:100], y[:100])
        bag.partial_fit(X[100:], y[100:])
        assert bag.predict(X).shape == (200,)


class TestAdaBoost:
    def test_fit_predict(self):
        X, y = make_data()
        ada = AdaBoostClassifier(n_estimators=5, max_depth=1)
        ada.fit(X, y)
        assert ada.predict(X).shape == (len(y),)

    def test_partial_fit(self):
        X, y = make_data(200)
        ada = AdaBoostClassifier(n_estimators=5)
        ada.partial_fit(X[:100], y[:100])
        ada.partial_fit(X[100:], y[100:])
        assert ada.predict(X).shape == (200,)

    def test_predict_proba(self):
        X, y = make_data()
        ada = AdaBoostClassifier(n_estimators=5)
        ada.fit(X, y)
        proba = ada.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestEnsembleClassifier:
    def test_random_forest(self):
        X, y = make_data()
        clf = EnsembleClassifier(method="random_forest", n_estimators=5)
        clf.fit(X, y)
        assert clf.predict(X).shape == (len(y),)

    def test_bagging(self):
        X, y = make_data()
        clf = EnsembleClassifier(method="bagging", n_estimators=5)
        clf.fit(X, y)
        assert clf.predict(X).shape == (len(y),)

    def test_adaboost(self):
        X, y = make_data()
        clf = EnsembleClassifier(method="adaboost", n_estimators=5)
        clf.fit(X, y)
        assert clf.predict(X).shape == (len(y),)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            EnsembleClassifier(method="xgboost")

    def test_partial_fit_streaming(self):
        X, y = make_data(200)
        clf = EnsembleClassifier(method="random_forest", n_estimators=5)
        for i in range(0, 200, 50):
            clf.partial_fit(X[i:i+50], y[i:i+50])
        assert clf.predict(X).shape == (200,)

    def test_repr(self):
        clf = EnsembleClassifier(method="bagging", n_estimators=7)
        assert "bagging" in repr(clf)
