import numpy as np
import pytest
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier

def make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

class TestTreeBuffer:
    def test_max_buffer_caps_accumulation(self):
        X, y = make_data(200)
        tree = DecisionTreeClassifier(max_depth=3, max_buffer=50)
        for i in range(0, 200, 50):
            tree.partial_fit(X[i:i+50], y[i:i+50])
        assert len(tree._y_acc) <= 50

    def test_max_buffer_none_accumulates_all(self):
        X, y = make_data(200)
        tree = DecisionTreeClassifier(max_depth=3, max_buffer=None)
        for i in range(0, 200, 50):
            tree.partial_fit(X[i:i+50], y[i:i+50])
        assert len(tree._y_acc) == 200

    def test_max_buffer_predict_still_works(self):
        X, y = make_data(200)
        tree = DecisionTreeClassifier(max_depth=3, max_buffer=80)
        for i in range(0, 200, 40):
            tree.partial_fit(X[i:i+40], y[i:i+40])
        preds = tree.predict(X)
        assert preds.shape == (200,)
        assert set(preds).issubset({0, 1})

    def test_max_buffer_smaller_than_chunk_keeps_last_chunk(self):
        X, y = make_data(100)
        tree = DecisionTreeClassifier(max_depth=2, max_buffer=30)
        tree.partial_fit(X[:50], y[:50])
        tree.partial_fit(X[50:], y[50:])
        assert len(tree._y_acc) <= 30

class TestEnsembleBuffer:
    def test_rf_max_buffer_caps_buffer(self):
        X, y = make_data(200)
        clf = EnsembleClassifier(method="random_forest", n_estimators=5,
                                  max_depth=3, max_buffer=60)
        for i in range(0, 200, 50):
            clf.partial_fit(X[i:i+50], y[i:i+50])
        assert len(clf._model._y_acc) <= 60

    def test_bagging_max_buffer_caps_buffer(self):
        X, y = make_data(200)
        clf = EnsembleClassifier(method="bagging", n_estimators=5,
                                  max_depth=3, max_buffer=60)
        for i in range(0, 200, 50):
            clf.partial_fit(X[i:i+50], y[i:i+50])
        assert len(clf._model._y_acc) <= 60

    def test_adaboost_max_buffer_caps_buffer(self):
        X, y = make_data(200)
        clf = EnsembleClassifier(method="adaboost", n_estimators=5,
                                  max_depth=1, max_buffer=60)
        for i in range(0, 200, 50):
            clf.partial_fit(X[i:i+50], y[i:i+50])
        assert len(clf._model._y_acc) <= 60

    def test_ensemble_buffer_predict_correct_shape(self):
        X, y = make_data(200)
        clf = EnsembleClassifier(method="random_forest", n_estimators=5,
                                  max_buffer=80)
        for i in range(0, 200, 40):
            clf.partial_fit(X[i:i+40], y[i:i+40])
        assert clf.predict(X).shape == (200,)
