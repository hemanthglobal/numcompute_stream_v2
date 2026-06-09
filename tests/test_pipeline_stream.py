import numpy as np
import pytest
from numcompute_stream.pipeline import Pipeline, Transformer, Estimator
from numcompute_stream.preprocessing import StandardScaler, SimpleImputer
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.stream import StreamTrainer

class DoubleTransformer(Transformer):
    def fit(self, X): return self
    def partial_fit(self, X): return self
    def transform(self, X): return X * 2

class AddOneTransformer(Transformer):
    def fit(self, X): return self
    def partial_fit(self, X): return self
    def transform(self, X): return X + 1

def make_data(n=120, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

class TestPipeline:
    def test_single_step(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        pipe = Pipeline([("double", DoubleTransformer())])
        result = pipe.fit_transform(X)
        np.testing.assert_array_equal(result, X * 2)

    def test_two_steps_ordered(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        pipe = Pipeline([("double", DoubleTransformer()),
                         ("add", AddOneTransformer())])
        result = pipe.fit_transform(X)
        np.testing.assert_array_equal(result, X * 2 + 1)

    def test_empty_steps_raises(self):
        with pytest.raises(ValueError):
            Pipeline([])

    def test_invalid_step_format_raises(self):
        with pytest.raises(ValueError):
            Pipeline([DoubleTransformer()])

    def test_double_underscore_in_name_raises(self):
        with pytest.raises(ValueError):
            Pipeline([("bad__name", DoubleTransformer())])

    def test_get_step(self):
        t = DoubleTransformer()
        pipe = Pipeline([("d", t)])
        assert pipe.get_step("d") is t

    def test_get_step_missing_raises(self):
        pipe = Pipeline([("d", DoubleTransformer())])
        with pytest.raises(KeyError):
            pipe.get_step("missing")

    def test_original_data_not_modified(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_copy = X.copy()
        pipe = Pipeline([("double", DoubleTransformer())])
        pipe.fit_transform(X)
        np.testing.assert_array_equal(X, X_copy)

    def test_partial_fit_and_predict(self):
        X, y = make_data()
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", EnsembleClassifier(n_estimators=3, max_depth=3)),
        ])
        pipe.partial_fit(X[:60], y[:60])
        pipe.partial_fit(X[60:], y[60:])
        preds = pipe.predict(X)
        assert preds.shape == (len(y),)

    def test_repr_contains_names(self):
        pipe = Pipeline([("a", DoubleTransformer()), ("b", AddOneTransformer())])
        assert "a" in repr(pipe) and "b" in repr(pipe)

class TestStreamTrainer:
    def _make_pipe(self):
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", EnsembleClassifier(n_estimators=3, max_depth=3)),
        ])

    def test_fit_chunk_returns_entry(self):
        X, y = make_data()
        trainer = StreamTrainer(self._make_pipe(), verbose=False)
        entry = trainer.fit_chunk(X[:30], y[:30])
        assert "accuracy" in entry
        assert "cumulative_acc" in entry

    def test_log_grows_per_chunk(self):
        X, y = make_data()
        trainer = StreamTrainer(self._make_pipe(), verbose=False)
        for i in range(0, 120, 40):
            trainer.fit_chunk(X[i:i+40], y[i:i+40])
        assert len(trainer.log_) == 3

    def test_cumulative_acc_in_range(self):
        X, y = make_data()
        trainer = StreamTrainer(self._make_pipe(), verbose=False)
        for i in range(0, 120, 40):
            trainer.fit_chunk(X[i:i+40], y[i:i+40])
        cum = trainer.log_[-1]["cumulative_acc"]
        assert 0.0 <= cum <= 1.0

    def test_get_metric_history(self):
        X, y = make_data()
        trainer = StreamTrainer(self._make_pipe(), verbose=False)
        for i in range(0, 120, 40):
            trainer.fit_chunk(X[i:i+40], y[i:i+40])
        history = trainer.get_metric_history("accuracy")
        assert len(history) == 3

    def test_score_chunk(self):
        X, y = make_data()
        trainer = StreamTrainer(self._make_pipe(), verbose=False)
        trainer.fit_chunk(X[:80], y[:80])
        score = trainer.score_chunk(X[80:], y[80:])
        assert 0.0 <= score <= 1.0

    def test_run_iterable(self):
        X, y = make_data()
        chunks = [(X[i:i+30], y[i:i+30]) for i in range(0, 120, 30)]
        trainer = StreamTrainer(self._make_pipe(), verbose=False)
        trainer.run(chunks)
        assert len(trainer.log_) == 4
