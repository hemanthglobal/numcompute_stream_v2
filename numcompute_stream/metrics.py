import numpy as np

class StreamingMetrics:

    def __init__(self, n_classes=None, window_size=None):
        self.n_classes = n_classes
        self.window_size = window_size
        self._cm = None
        self._classes = None
        self._win_true = []
        self._win_pred = []
        self._win_scores = []
        self._chunk_accuracies = []

    def update(self, y_true_chunk, y_pred_chunk, y_scores_chunk=None):
        y_true = np.asarray(y_true_chunk).flatten()
        y_pred = np.asarray(y_pred_chunk).flatten()

        all_labels = np.unique(np.concatenate([y_true, y_pred]))
        if self._classes is None:
            self._classes = all_labels
        else:
            self._classes = np.unique(np.concatenate([self._classes, all_labels]))

        n = len(self._classes)
        if self._cm is None or self._cm.shape[0] != n:
            new_cm = np.zeros((n, n), dtype=int)
            if self._cm is not None:
                old_n = self._cm.shape[0]
                new_cm[:old_n, :old_n] = self._cm
            self._cm = new_cm

        label_to_idx = {c: i for i, c in enumerate(self._classes)}
        for yt, yp in zip(y_true, y_pred):
            i = label_to_idx.get(yt)
            j = label_to_idx.get(yp)
            if i is not None and j is not None:
                self._cm[i, j] += 1

        self._chunk_accuracies.append(float(np.mean(y_true == y_pred)))

        if self.window_size is not None:
            self._win_true.extend(y_true.tolist())
            self._win_pred.extend(y_pred.tolist())
            if y_scores_chunk is not None:
                self._win_scores.extend(np.asarray(y_scores_chunk).tolist())
            if len(self._win_true) > self.window_size:
                excess = len(self._win_true) - self.window_size
                self._win_true = self._win_true[excess:]
                self._win_pred = self._win_pred[excess:]
                if self._win_scores:
                    self._win_scores = self._win_scores[excess:]

        return self

    def result(self):
        if self._cm is None:
            return {}
        cm = self._cm
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
        if self.window_size is None:
            raise RuntimeError("window_size must be set for rolling metrics.")
        if not self._win_true:
            return 0.0
        yt = np.array(self._win_true)
        yp = np.array(self._win_pred)
        return float(np.mean(yt == yp))

    def reset(self):
        self._cm = None
        self._classes = None
        self._win_true = []
        self._win_pred = []
        self._win_scores = []
        self._chunk_accuracies = []
        return self

def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

def confusion_matrix(y_true, y_pred):
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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred, pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

def f1(y_true, y_pred, pos_label=1):
    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))

def roc_curve(y_true, y_scores):
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
    return float(np.trapezoid(np.asarray(tpr), np.asarray(fpr)))
