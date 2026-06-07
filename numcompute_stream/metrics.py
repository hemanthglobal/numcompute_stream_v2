"""
metrics.py - Streaming-compatible classification and regression metrics.

The StreamingMetrics class accumulates a confusion matrix across chunks,
deriving accuracy, precision, recall, F1, and AUC on demand.
A rolling-window variant is also provided for recency-weighted evaluation.

Classes
-------
StreamingMetrics : Accumulates metrics via update(); supports reset/result.

Functions
---------
accuracy, precision, recall, f1, mse : Batch metric functions.
confusion_matrix : Batch confusion matrix.
roc_curve, auc   : ROC / AUC computation.
"""

import numpy as np


# ---------------------------------------------------------------------------
# StreamingMetrics
# ---------------------------------------------------------------------------

class StreamingMetrics:
    """
    Accumulate binary or multi-class classification metrics chunk-by-chunk.

    All counts are maintained as a running confusion matrix so that
    calling result() at any point yields up-to-date aggregate metrics.

    Parameters
    ----------
    n_classes    : int or None  Number of classes. Inferred from first chunk
                                if None.
    window_size  : int or None  If set, only the last ``window_size``
                                (y_true, y_pred) pairs are kept for rolling
                                metrics.

    Public methods
    --------------
    update(y_true_chunk, y_pred_chunk)  Incorporate a new chunk.
    result()                            Return dict of current metrics.
    reset()                             Clear all accumulated state.
    """

    def __init__(self, n_classes=None, window_size=None):
        self.n_classes = n_classes
        self.window_size = window_size
        self._cm = None                 # running confusion matrix
        self._classes = None
        # Rolling window storage
        self._win_true = []
        self._win_pred = []
        self._win_scores = []           # for AUC
        self._chunk_accuracies = []     # per-chunk accuracy log

    # ------------------------------------------------------------------
    def update(self, y_true_chunk, y_pred_chunk, y_scores_chunk=None):
        """
        Incorporate predictions from one chunk.

        Parameters
        ----------
        y_true_chunk  : array-like  Shape (n,).
        y_pred_chunk  : array-like  Shape (n,).
        y_scores_chunk: array-like or None  Predicted probabilities for AUC.

        Returns
        -------
        self
        """
        y_true = np.asarray(y_true_chunk).flatten()
        y_pred = np.asarray(y_pred_chunk).flatten()

        # Discover / extend class set
        all_labels = np.unique(np.concatenate([y_true, y_pred]))
        if self._classes is None:
            self._classes = all_labels
        else:
            self._classes = np.unique(np.concatenate([self._classes, all_labels]))

        n = len(self._classes)
        # Rebuild CM if classes expanded
        if self._cm is None or self._cm.shape[0] != n:
            new_cm = np.zeros((n, n), dtype=int)
            if self._cm is not None:
                old_n = self._cm.shape[0]
                new_cm[:old_n, :old_n] = self._cm
            self._cm = new_cm

        # Accumulate
        label_to_idx = {c: i for i, c in enumerate(self._classes)}
        for yt, yp in zip(y_true, y_pred):
            i = label_to_idx.get(yt)
            j = label_to_idx.get(yp)
            if i is not None and j is not None:
                self._cm[i, j] += 1

        # Per-chunk accuracy
        self._chunk_accuracies.append(float(np.mean(y_true == y_pred)))

        # Rolling window
        if self.window_size is not None:
            self._win_true.extend(y_true.tolist())
            self._win_pred.extend(y_pred.tolist())
            if y_scores_chunk is not None:
                self._win_scores.extend(np.asarray(y_scores_chunk).tolist())
            # Trim
            if len(self._win_true) > self.window_size:
                excess = len(self._win_true) - self.window_size
                self._win_true = self._win_true[excess:]
                self._win_pred = self._win_pred[excess:]
                if self._win_scores:
                    self._win_scores = self._win_scores[excess:]

        return self

    # ------------------------------------------------------------------
    def result(self):
        """
        Return a dict of current aggregate metrics.

        Keys: 'accuracy', 'precision', 'recall', 'f1',
              'confusion_matrix', 'chunk_accuracies'.
        """
        if self._cm is None:
            return {}
        cm = self._cm
        # Per-class TP, FP, FN
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        denom_p = tp + fp
        denom_r = tp + fn
        per_prec = np.where(denom_p > 0, tp / denom_p, 0.0)
        per_rec = np.where(denom_r > 0, tp / denom_r, 0.0)
        denom_f1 = per_prec + per_rec
        per_f1 = np.where(denom_f1 > 0, 2 * per_prec * per_rec / denom_f1, 0.0)

        total = cm.sum()
        acc = float(np.trace(cm) / total) if total > 0 else 0.0

        return {
            "accuracy": acc,
            "precision": float(np.mean(per_prec)),
            "recall": float(np.mean(per_rec)),
            "f1": float(np.mean(per_f1)),
            "confusion_matrix": cm.copy(),
            "chunk_accuracies": list(self._chunk_accuracies),
        }

    def rolling_accuracy(self):
        """Accuracy over the current rolling window (requires window_size)."""
        if self.window_size is None:
            raise RuntimeError("window_size must be set for rolling metrics.")
        if not self._win_true:
            return 0.0
        yt = np.array(self._win_true)
        yp = np.array(self._win_pred)
        return float(np.mean(yt == yp))

    def reset(self):
        """Clear all accumulated state."""
        self._cm = None
        self._classes = None
        self._win_true = []
        self._win_pred = []
        self._win_scores = []
        self._chunk_accuracies = []
        return self


# ---------------------------------------------------------------------------
# Batch metric functions
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred):
    """
    Compute a confusion matrix for multi-class predictions.

    Returns
    -------
    np.ndarray  Shape (n_classes, n_classes).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for yt, yp in zip(y_true, y_pred):
        cm[idx[yt], idx[yp]] += 1
    return cm


def precision(y_true, y_pred, pos_label=1):
    """Binary precision for ``pos_label``."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred, pos_label=1):
    """Binary recall for ``pos_label``."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1(y_true, y_pred, pos_label=1):
    """Binary F1 score for ``pos_label``."""
    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def mse(y_true, y_pred):
    """Mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def roc_curve(y_true, y_scores):
    """
    Compute ROC curve (FPR, TPR, thresholds).

    Parameters
    ----------
    y_true   : array-like  Binary ground-truth labels.
    y_scores : array-like  Predicted probability for the positive class.

    Returns
    -------
    fpr        : np.ndarray
    tpr        : np.ndarray
    thresholds : np.ndarray
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    thresholds = np.sort(np.unique(y_scores))[::-1]
    fprs, tprs = [], []
    for thr in thresholds:
        yp = (y_scores >= thr).astype(int)
        tp = np.sum((yp == 1) & (y_true == 1))
        fp = np.sum((yp == 1) & (y_true == 0))
        fn = np.sum((yp == 0) & (y_true == 1))
        tn = np.sum((yp == 0) & (y_true == 0))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    return np.array(fprs), np.array(tprs), thresholds


def auc(fpr, tpr):
    """Area under the ROC curve via the trapezoidal rule."""
    return float(np.trapezoid(np.asarray(tpr), np.asarray(fpr)))
