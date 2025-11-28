"""
Enhanced Evaluation Framework for AutoTrain Advanced
=====================================================

Provides comprehensive evaluation capabilities for model training.
"""

from .benchmark import Benchmark, BenchmarkConfig, BenchmarkResult, compare_models, run_benchmark
from .callbacks import (
    BestModelCallback,
    EarlyStoppingCallback,
    EvaluationCallback,
    MetricsLoggerCallback,
    PeriodicEvalCallback,
)
from .evaluator import EvaluationConfig, EvaluationResult, Evaluator, MetricType, evaluate_generation, evaluate_model
from .metrics import (
    AccuracyMetric,
    BERTScoreMetric,
    BLEUMetric,
    CustomMetric,
    ExactMatchMetric,
    F1Metric,
    METEORMetric,
    Metric,
    MetricCollection,
    PerplexityMetric,
    ROUGEMetric,
)


__all__ = [
    # Core evaluator
    "Evaluator",
    "EvaluationConfig",
    "EvaluationResult",
    "MetricType",
    "evaluate_model",
    "evaluate_generation",
    # Metrics
    "Metric",
    "PerplexityMetric",
    "BLEUMetric",
    "ROUGEMetric",
    "BERTScoreMetric",
    "AccuracyMetric",
    "F1Metric",
    "ExactMatchMetric",
    "METEORMetric",
    "CustomMetric",
    "MetricCollection",
    # Callbacks
    "EvaluationCallback",
    "PeriodicEvalCallback",
    "BestModelCallback",
    "EarlyStoppingCallback",
    "MetricsLoggerCallback",
    # Benchmarking
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmark",
    "compare_models",
]
