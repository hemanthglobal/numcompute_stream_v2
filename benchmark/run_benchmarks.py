import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.pipeline import Pipeline

def timeit(fn, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.mean(times)

def print_table(results):
    print(f"\n{'Operation':<45} {'Time (s)':<12} {'Note'}")
    print("-" * 75)
    for name, t, note in results:
        print(f"{name:<45} {t:<12.4f} {note}")
    print()

rng = np.random.default_rng(0)
N = 2000
X_full = rng.standard_normal((N, 8))
y_full = (X_full[:, 0] + X_full[:, 1] > 0).astype(int)
CHUNK = 200
chunks = [(X_full[i:i+CHUNK], y_full[i:i+CHUNK]) for i in range(0, N, CHUNK)]

print("Running streaming benchmarks …\n")
results = []

def bench_tree():
    tree = DecisionTreeClassifier(max_depth=5)
    for X_c, y_c in chunks:
        tree.partial_fit(X_c, y_c)
    return tree.predict(X_full)

t_tree = timeit(bench_tree)
results.append(("Single DecisionTree (10 chunks, n=2000)", t_tree, "streaming"))

def bench_rf():
    rf = EnsembleClassifier(method="random_forest", n_estimators=10, max_depth=4)
    for X_c, y_c in chunks:
        rf.partial_fit(X_c, y_c)
    return rf.predict(X_full)

t_rf = timeit(bench_rf)
results.append(("RandomForest n=10 (10 chunks, n=2000)", t_rf, "streaming"))

def bench_ada():
    ada = EnsembleClassifier(method="adaboost", n_estimators=10, max_depth=1)
    for X_c, y_c in chunks:
        ada.partial_fit(X_c, y_c)
    return ada.predict(X_full)

t_ada = timeit(bench_ada)
results.append(("AdaBoost n=10 (10 chunks, n=2000)", t_ada, "streaming"))

def bench_pipe():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", EnsembleClassifier(n_estimators=5, max_depth=3)),
    ])
    for X_c, y_c in chunks:
        pipe.partial_fit(X_c, y_c)
    return pipe.predict(X_full)

t_pipe = timeit(bench_pipe)
results.append(("Pipeline (scaler + RF, 10 chunks)", t_pipe, "streaming"))

data_large = np.random.rand(100_000)

def numpy_mean(): return np.mean(data_large)
def loop_mean():
    s = 0.0
    for x in data_large: s += x
    return s / len(data_large)

t_np = timeit(numpy_mean, repeats=20)
t_lp = timeit(loop_mean, repeats=5)
results.append(("NumPy mean (n=100,000)", t_np, f"loop={t_lp:.4f}s  speedup={t_lp/t_np:.0f}x"))

print_table(results)

tree2 = DecisionTreeClassifier(max_depth=5)
rf2   = EnsembleClassifier(method="random_forest", n_estimators=10, max_depth=4)
for X_c, y_c in chunks:
    tree2.partial_fit(X_c, y_c)
    rf2.partial_fit(X_c, y_c)

acc_tree = np.mean(tree2.predict(X_full) == y_full)
acc_rf   = np.mean(rf2.predict(X_full) == y_full)
print(f"  Streaming accuracy — Single Tree: {acc_tree:.4f}  |  Random Forest: {acc_rf:.4f}")
print()
