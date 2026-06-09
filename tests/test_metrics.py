import numpy as np
import pytest
from numcompute_stream.metrics import (
    StreamingMetrics, accuracy, precision, recall, f1, mse,
    confusion_matrix, roc_curve, auc
)

class TestStreamingMetrics:
    def test_update_and_accuracy(self):
        sm = StreamingMetrics()
        sm.update(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0]))
        assert sm.result()["accuracy"] == pytest.approx(1.0)

    def test_two_chunk_cumulative(self):
        sm = StreamingMetrics()
        sm.update(np.array([0, 1]), np.array([0, 1]))
        sm.update(np.array([0, 1]), np.array([1, 0]))
        assert sm.result()["accuracy"] == pytest.approx(0.5)

    def test_reset(self):
        sm = StreamingMetrics()
        sm.update(np.array([1, 0]), np.array([1, 0]))
        sm.reset()
        assert sm.result() == {}

    def test_precision_recall_f1(self):
        sm = StreamingMetrics()
        sm.update(np.array([1, 0, 1, 1, 0]),
                  np.array([1, 1, 1, 0, 0]))
        r = sm.result()
        assert 0.0 <= r["precision"] <= 1.0
        assert 0.0 <= r["recall"] <= 1.0
        assert 0.0 <= r["f1"] <= 1.0

    def test_confusion_matrix_shape(self):
        sm = StreamingMetrics()
        sm.update(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0]))
        cm = sm.result()["confusion_matrix"]
        assert cm.shape == (2, 2)

    def test_chunk_accuracies_logged(self):
        sm = StreamingMetrics()
        sm.update(np.array([1, 1]), np.array([1, 1]))
        sm.update(np.array([0, 0]), np.array([1, 1]))
        accs = sm.result()["chunk_accuracies"]
        assert len(accs) == 2
        assert accs[0] == pytest.approx(1.0)
        assert accs[1] == pytest.approx(0.0)

    def test_rolling_accuracy(self):
        sm = StreamingMetrics(window_size=4)
        sm.update(np.array([1, 1]), np.array([0, 0]))
        sm.update(np.array([1, 1]), np.array([1, 1]))
        r = sm.rolling_accuracy()
        assert 0.0 <= r <= 1.0

    def test_rolling_accuracy_no_window_raises(self):
        sm = StreamingMetrics()
        sm.update(np.array([1]), np.array([1]))
        with pytest.raises(RuntimeError):
            sm.rolling_accuracy()

class TestBatchMetrics:
    def test_accuracy_perfect(self):
        assert accuracy([1, 0, 1], [1, 0, 1]) == pytest.approx(1.0)

    def test_accuracy_zero(self):
        assert accuracy([0, 0], [1, 1]) == pytest.approx(0.0)

    def test_precision_zero_division(self):
        assert precision([1, 1], [0, 0], pos_label=1) == pytest.approx(0.0)

    def test_recall_zero_division(self):
        assert recall([0, 0], [0, 0], pos_label=1) == pytest.approx(0.0)

    def test_f1_perfect(self):
        assert f1([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)

    def test_mse_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mse(y, y) == pytest.approx(0.0)

    def test_mse_value(self):
        assert mse([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == pytest.approx(2 / 3)

    def test_confusion_matrix_shape(self):
        cm = confusion_matrix([0, 1, 0, 1], [0, 1, 1, 0])
        assert cm.shape == (2, 2)

    def test_auc_perfect(self):
        fpr = np.array([0.0, 0.0, 1.0])
        tpr = np.array([0.0, 1.0, 1.0])
        assert auc(fpr, tpr) == pytest.approx(1.0)
