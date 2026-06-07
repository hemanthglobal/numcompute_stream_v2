"""Tests for DecisionTreeClassifier."""
import numpy as np
import pytest
from numcompute_stream.tree import DecisionTreeClassifier


def make_data(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


class TestDecisionTree:
    def test_fit_predict_shape(self):
        X, y = make_data()
        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(X, y)
        preds = tree.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_before_fit_raises(self):
        tree = DecisionTreeClassifier()
        with pytest.raises(RuntimeError):
            tree.predict(np.array([[1.0, 2.0]]))

    def test_fit_accuracy_linearly_separable(self):
        X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(X, y)
        assert np.all(tree.predict(X) == y)

    def test_gini_criterion(self):
        X, y = make_data()
        tree = DecisionTreeClassifier(criterion="gini")
        tree.fit(X, y)
        assert tree.root_ is not None

    def test_entropy_criterion(self):
        X, y = make_data()
        tree = DecisionTreeClassifier(criterion="entropy")
        tree.fit(X, y)
        assert tree.root_ is not None

    def test_invalid_criterion_raises(self):
        with pytest.raises(ValueError):
            DecisionTreeClassifier(criterion="log_loss")

    def test_partial_fit_two_chunks(self):
        X, y = make_data(200)
        tree = DecisionTreeClassifier(max_depth=3)
        tree.partial_fit(X[:100], y[:100])
        tree.partial_fit(X[100:], y[100:])
        preds = tree.predict(X)
        assert preds.shape == (200,)

    def test_predict_proba_shape(self):
        X, y = make_data()
        tree = DecisionTreeClassifier()
        tree.fit(X, y)
        proba = tree.predict_proba(X)
        assert proba.shape == (len(X), len(np.unique(y)))

    def test_predict_proba_sums_to_one(self):
        X, y = make_data()
        tree = DecisionTreeClassifier()
        tree.fit(X, y)
        proba = tree.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_max_features_sqrt(self):
        X, y = make_data()
        tree = DecisionTreeClassifier(max_features="sqrt")
        tree.fit(X, y)
        assert tree.root_ is not None

    def test_classes_attribute(self):
        X, y = make_data()
        tree = DecisionTreeClassifier()
        tree.fit(X, y)
        assert set(tree.classes_) == {0, 1}
