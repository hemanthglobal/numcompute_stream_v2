"""
stream.py - StreamTrainer: orchestrates chunk-wise model training and logging.

StreamTrainer wraps a Pipeline (or any model with partial_fit / predict)
and provides a clean loop for processing streaming data.  It logs per-chunk
metrics, running accuracy, and memory footprint, making the full training
history available for downstream visualisation.

Classes
-------
StreamTrainer : Manages streaming fit / score / log lifecycle.
"""

import time
import sys
import numpy as np
from numcompute_stream.metrics import accuracy as batch_accuracy


class StreamTrainer:
    """
    Orchestrates incremental training over a sequence of data chunks.

    Wraps a pipeline or model and drives the partial_fit / predict /
    score loop.  All per-chunk results are recorded in ``log_``.

    Parameters
    ----------
    model : object
        Any object with partial_fit(X, y) and predict(X) methods.
        Typically a Pipeline or an EnsembleClassifier.
    verbose : bool
        If True, prints a summary line after each chunk (default True).

    Attributes
    ----------
    log_ : list of dict
        Each entry corresponds to one processed chunk and contains:
        - 'chunk'          : int    Chunk index (0-based).
        - 'n_samples'      : int    Number of samples in the chunk.
        - 'accuracy'       : float  Accuracy on that chunk after training.
        - 'cumulative_acc' : float  Accuracy over all samples seen so far.
        - 'fit_time_s'     : float  Seconds spent in partial_fit().
        - 'score_time_s'   : float  Seconds spent in predict().
        - 'memory_bytes'   : int    Rough memory of X_chunk + y_chunk.

    Examples
    --------
    >>> trainer = StreamTrainer(model=pipe)
    >>> for X_chunk, y_chunk in data_stream:
    ...     trainer.fit_chunk(X_chunk, y_chunk)
    >>> trainer.print_summary()
    """

    def __init__(self, model, verbose=True):
        self.model = model
        self.verbose = verbose
        self.log_ = []
        self._all_true = []
        self._all_pred = []

    # ------------------------------------------------------------------
    def fit_chunk(self, X_chunk, y_chunk):
        """
        Fit the model on one chunk and score it immediately after.

        Parameters
        ----------
        X_chunk : np.ndarray  Shape (n, n_features).
        y_chunk : np.ndarray  Shape (n,).

        Returns
        -------
        dict  The log entry for this chunk.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)
        chunk_idx = len(self.log_)
        n = len(y_chunk)

        # --- partial_fit ---
        t0 = time.perf_counter()
        self.model.partial_fit(X_chunk, y_chunk)
        fit_time = time.perf_counter() - t0

        # --- score ---
        t1 = time.perf_counter()
        y_pred = self.model.predict(X_chunk)
        score_time = time.perf_counter() - t1

        chunk_acc = float(batch_accuracy(y_chunk, y_pred))

        # Cumulative accuracy
        self._all_true.extend(y_chunk.tolist())
        self._all_pred.extend(y_pred.tolist())
        cum_acc = float(
            np.mean(np.array(self._all_true) == np.array(self._all_pred))
        )

        # Memory footprint: size of the retained training buffer, not just the chunk.
        # Climb with unbounded design; plateau when max_buffer is set.
        buf = getattr(self.model, "_X_acc", None)
        if buf is None:
            inner = getattr(self.model, "_model", None)   # EnsembleClassifier wrapper
            buf = getattr(inner, "_X_acc", None)
        mem = int(buf.nbytes) if buf is not None else int(X_chunk.nbytes + y_chunk.nbytes)

        entry = {
            "chunk": chunk_idx,
            "n_samples": n,
            "accuracy": chunk_acc,
            "cumulative_acc": cum_acc,
            "fit_time_s": round(fit_time, 6),
            "score_time_s": round(score_time, 6),
            "memory_bytes": mem,
        }
        self.log_.append(entry)

        if self.verbose:
            print(
                f"  Chunk {chunk_idx:>3} | n={n:>5} | "
                f"acc={chunk_acc:.4f} | cum_acc={cum_acc:.4f} | "
                f"fit={fit_time*1e3:.1f}ms"
            )

        return entry

    # ------------------------------------------------------------------
    def score_chunk(self, X_chunk, y_chunk):
        """
        Score the current model on a held-out chunk (no fitting).

        Parameters
        ----------
        X_chunk : np.ndarray
        y_chunk : np.ndarray

        Returns
        -------
        float  Accuracy on this chunk.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)
        y_pred = self.model.predict(X_chunk)
        return float(batch_accuracy(y_chunk, y_pred))

    # ------------------------------------------------------------------
    def run(self, data_stream):
        """
        Process an iterable of (X_chunk, y_chunk) pairs.

        Parameters
        ----------
        data_stream : iterable of (np.ndarray, np.ndarray) tuples.

        Returns
        -------
        self
        """
        for X_chunk, y_chunk in data_stream:
            self.fit_chunk(X_chunk, y_chunk)
        return self

    # ------------------------------------------------------------------
    def get_metric_history(self, key="accuracy"):
        """
        Extract a list of per-chunk values for a logged metric.

        Parameters
        ----------
        key : str  Any key present in log_ entries.

        Returns
        -------
        list
        """
        return [entry[key] for entry in self.log_]

    # ------------------------------------------------------------------
    def print_summary(self):
        """Print a formatted summary of the full training run."""
        if not self.log_:
            print("No chunks processed yet.")
            return
        n_chunks = len(self.log_)
        total_samples = sum(e["n_samples"] for e in self.log_)
        final_cum_acc = self.log_[-1]["cumulative_acc"]
        total_fit = sum(e["fit_time_s"] for e in self.log_)
        print("\n" + "=" * 56)
        print("  StreamTrainer Summary")
        print("=" * 56)
        print(f"  Chunks processed  : {n_chunks}")
        print(f"  Total samples     : {total_samples}")
        print(f"  Final cum. acc.   : {final_cum_acc:.4f}")
        print(f"  Total fit time    : {total_fit*1e3:.1f} ms")
        print("=" * 56)
