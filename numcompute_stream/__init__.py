"""
NumCompute-Stream: A Modularised Ensemble Tree-based Streaming ML Framework.
Built on plain Python and NumPy only.
"""

from numcompute_stream.io import load_csv, load_csv_with_header
from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
from numcompute_stream.stats import StreamingStats
from numcompute_stream.metrics import StreamingMetrics
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer
from numcompute_stream import visualise

__all__ = [
    "load_csv", "load_csv_with_header",
    "StandardScaler", "MinMaxScaler", "SimpleImputer", "OneHotEncoder",
    "StreamingStats",
    "StreamingMetrics",
    "DecisionTreeClassifier",
    "EnsembleClassifier",
    "Pipeline",
    "StreamTrainer",
    "visualise",
]
