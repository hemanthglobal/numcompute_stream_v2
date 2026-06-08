# NumCompute-Stream

**A Modularised Ensemble Tree-based Streaming Machine Learning Framework**

Built with plain Python and NumPy only (no scikit-learn, no pandas).

---

## Project Structure

```
numcompute_stream/
├── numcompute_stream/
│   ├── __init__.py
│   ├── io.py              # CSV loading & streaming generator
│   ├── preprocessing.py   # StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
│   ├── stats.py           # StreamingStats + batch helpers
│   ├── metrics.py         # StreamingMetrics + batch metric functions
│   ├── tree.py            # DecisionTreeClassifier (Gini / Entropy, partial_fit)
│   ├── ensemble.py        # RandomForest, Bagging, AdaBoost, EnsembleClassifier
│   ├── pipeline.py        # Streaming Pipeline (partial_fit support)
│   ├── stream.py          # StreamTrainer – orchestrates chunk-wise training
│   └── visualise.py       # matplotlib plotting functions
├── tests/                 # 35+ unit tests (pytest)
├── benchmark/
│   └── run_benchmarks.py  # Model & vectorisation benchmarks
├── demo/
│   └── stream_demo.ipynb  # Full streaming walkthrough notebook
├── data/
│   └── health_stream.csv  # Sample dataset (300 rows, 5 columns)
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
# No extra packages needed beyond NumPy and matplotlib
pip install numpy matplotlib
```

---

## Quick Start

```python
from numcompute_stream.io import load_csv_with_header
from numcompute_stream.preprocessing import StandardScaler, SimpleImputer
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer
import numpy as np

# Load data
headers, data = load_csv_with_header("data/health_stream.csv")
X, y = data[:, :-1], data[:, -1].astype(int)

# Build streaming pipeline
pipe = Pipeline([
    ("impute", SimpleImputer(fill_value=0.0)),
    ("scale",  StandardScaler()),
    ("model",  EnsembleClassifier(method="random_forest", n_estimators=10)),
])

# Stream in 50-row chunks
trainer = StreamTrainer(pipe, verbose=True)
chunk_size = 50
for i in range(0, len(X), chunk_size):
    trainer.fit_chunk(X[i:i+chunk_size], y[i:i+chunk_size])

trainer.print_summary()
```

---

## Streaming API

Every component exposes `partial_fit()` for incremental updates:

| Class | Key method |
|---|---|
| `StandardScaler` | `partial_fit(X_chunk)` — Welford running mean/var |
| `SimpleImputer` | `partial_fit(X_chunk)` — running column means |
| `OneHotEncoder` | `partial_fit(X_chunk)` — expands category set |
| `StreamingStats` | `update_stats(X_chunk)` — running min/max/mean/var |
| `StreamingMetrics` | `update(y_true, y_pred)` — running confusion matrix |
| `DecisionTreeClassifier` | `partial_fit(X_chunk, y_chunk)` — online regrowth |
| `EnsembleClassifier` | `partial_fit(X_chunk, y_chunk)` — all three methods |
| `Pipeline` | `partial_fit(X_chunk, y_chunk)` — chains all steps |

---

## Ensemble Methods

```python
# Random Forest (Bagging + feature sub-sampling)
clf = EnsembleClassifier(method="random_forest", n_estimators=20, max_depth=5)

# Generic Bagging
clf = EnsembleClassifier(method="bagging", n_estimators=20)

# AdaBoost (SAMME)
clf = EnsembleClassifier(method="adaboost", n_estimators=20, max_depth=1)

clf.partial_fit(X_chunk, y_chunk)
clf.predict(X_test)
clf.predict_proba(X_test)
```

---

## Visualisation

```python
from numcompute_stream import visualise

# Single metric over chunks
visualise.plot_metric_over_time(trainer.get_metric_history("accuracy"),
                                title="Streaming Accuracy", ylabel="Accuracy")

# Compare two models
visualise.compare_models(rf_accs, ada_accs, labels=["Random Forest", "AdaBoost"])

# Latest chunk predictions
visualise.plot_predictions_vs_ground_truth(y_true, y_pred)

# Full dashboard
visualise.plot_streaming_dashboard(
    chunk_accuracies, cumulative_accuracies,
    fit_times, y_true_last, y_pred_last, cm=confusion_matrix
)
```

---

## Running Tests

```bash
# From project root
python -m pytest tests/ -v
```

Expected: **35+ tests**, all passing.

---

## Running Benchmarks

```bash
python benchmark/run_benchmarks.py
```

Compares single tree vs ensemble streaming accuracy and NumPy vs Python-loop speed.

---

## Design Decisions

1. **Welford's algorithm** for numerically stable running mean/variance in `StandardScaler` and `StreamingStats` — no need to store all past data.
2. **Accumulated data refit** in tree/ensemble `partial_fit` — avoids concept drift by incorporating all observations while remaining chunk-compatible.
3. **SAMME AdaBoost** supports multi-class targets; alpha weights correct for random guessing baseline with `log(K-1)` term.
4. **Pipeline chains** `partial_fit` across all steps so a single call updates the full preprocessing + model stack.
5. **`visualise.py`** uses a non-interactive `Agg` backend by default so it works in scripts and notebooks without a display.

---