import time
import sys
import numpy as np
from numcompute_stream.metrics import accuracy as batch_accuracy

class StreamTrainer:

    def __init__(self, model, verbose=True):
        self.model = model
        self.verbose = verbose
        self.log_ = []
        self._all_true = []
        self._all_pred = []

    def fit_chunk(self, X_chunk, y_chunk):
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)
        chunk_idx = len(self.log_)
        n = len(y_chunk)

        t0 = time.perf_counter()
        self.model.partial_fit(X_chunk, y_chunk)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = self.model.predict(X_chunk)
        score_time = time.perf_counter() - t1

        chunk_acc = float(batch_accuracy(y_chunk, y_pred))

        self._all_true.extend(y_chunk.tolist())
        self._all_pred.extend(y_pred.tolist())
        cum_acc = float(
            np.mean(np.array(self._all_true) == np.array(self._all_pred))
        )

        buf = getattr(self.model, "_X_acc", None)
        if buf is None:
            inner = getattr(self.model, "_model", None)
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

    def score_chunk(self, X_chunk, y_chunk):
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)
        y_pred = self.model.predict(X_chunk)
        return float(batch_accuracy(y_chunk, y_pred))

    def run(self, data_stream):
        for X_chunk, y_chunk in data_stream:
            self.fit_chunk(X_chunk, y_chunk)
        return self

    def get_metric_history(self, key="accuracy"):
        return [entry[key] for entry in self.log_]

    def print_summary(self):
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
