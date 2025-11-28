"""
Benchmarking utilities for model comparison
============================================

Compare multiple models across various benchmarks.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from autotrain import logger

from .evaluator import EvaluationConfig, Evaluator


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    # Benchmark datasets
    benchmarks: List[str] = field(default_factory=lambda: ["mmlu", "hellaswag", "arc"])

    # Evaluation settings
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1"])
    batch_size: int = 8
    max_samples_per_benchmark: Optional[int] = None

    # Output settings
    output_dir: str = "./benchmark_results"
    save_predictions: bool = False
    generate_report: bool = True

    # Comparison settings
    compare_models: bool = True
    baseline_model: Optional[str] = None

    # Device settings
    device: str = "auto"
    fp16: bool = True


@dataclass
class BenchmarkResult:
    """Results from benchmarking."""

    model_name: str
    benchmark_scores: Dict[str, Dict[str, float]]
    overall_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        for benchmark, metrics in self.benchmark_scores.items():
            for metric, score in metrics.items():
                rows.append({"model": self.model_name, "benchmark": benchmark, "metric": metric, "score": score})
        return pd.DataFrame(rows)

    def save(self, filepath: Optional[str] = None):
        """Save results to file."""
        if filepath is None:
            filepath = f"benchmark_{self.model_name}_{self.timestamp[:10]}.json"

        data = {
            "model_name": self.model_name,
            "benchmark_scores": self.benchmark_scores,
            "overall_score": self.overall_score,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        # Also save as CSV
        df = self.to_dataframe()
        csv_path = filepath.replace(".json", ".csv")
        df.to_csv(csv_path, index=False)

        logger.info(f"Benchmark results saved to {filepath}")

    def print_summary(self):
        """Print summary of results."""
        print(f"\n{'='*60}")
        print(f"Benchmark Results for {self.model_name}")
        print(f"{'='*60}")

        for benchmark, metrics in self.benchmark_scores.items():
            print(f"\n{benchmark}:")
            for metric, score in metrics.items():
                print(f"  {metric}: {score:.4f}")

        print(f"\nOverall Score: {self.overall_score:.4f}")
        print(f"{'='*60}\n")


class Benchmark:
    """Main benchmarking class."""

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.results = []

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Load benchmark datasets
        self.datasets = self._load_benchmarks()

    def _load_benchmarks(self) -> Dict[str, Any]:
        """Load benchmark datasets."""
        datasets = {}

        for benchmark_name in self.config.benchmarks:
            try:
                dataset = self._load_single_benchmark(benchmark_name)
                if dataset:
                    datasets[benchmark_name] = dataset
                    logger.info(f"Loaded benchmark: {benchmark_name}")
            except Exception as e:
                logger.warning(f"Failed to load benchmark {benchmark_name}: {e}")

        return datasets

    def _load_single_benchmark(self, name: str) -> Any:
        """Load a single benchmark dataset."""
        # This is a simplified version - real implementation would load actual benchmarks
        # from HuggingFace datasets or other sources

        if name == "mmlu":
            # Mock MMLU dataset
            return self._create_mock_qa_dataset(100, "multiple_choice")
        elif name == "hellaswag":
            # Mock HellaSwag dataset
            return self._create_mock_qa_dataset(100, "completion")
        elif name == "arc":
            # Mock ARC dataset
            return self._create_mock_qa_dataset(100, "multiple_choice")
        elif name == "truthfulqa":
            # Mock TruthfulQA dataset
            return self._create_mock_qa_dataset(100, "generation")
        else:
            # Try to load from HuggingFace datasets
            try:
                from datasets import load_dataset

                dataset = load_dataset(name, split="test")
                if self.config.max_samples_per_benchmark:
                    dataset = dataset.select(range(min(len(dataset), self.config.max_samples_per_benchmark)))
                return dataset
            except:
                logger.warning(f"Could not load benchmark {name}")
                return None

    def _create_mock_qa_dataset(self, size: int, task_type: str) -> List[Dict[str, Any]]:
        """Create mock QA dataset for testing."""
        dataset = []
        for i in range(size):
            if task_type == "multiple_choice":
                sample = {
                    "question": f"Question {i}?",
                    "choices": ["A", "B", "C", "D"],
                    "answer": np.random.choice(["A", "B", "C", "D"]),
                }
            elif task_type == "completion":
                sample = {
                    "context": f"Context {i}",
                    "completion": f"Completion {i}",
                }
            else:
                sample = {
                    "input": f"Input {i}",
                    "target": f"Target {i}",
                }
            dataset.append(sample)
        return dataset

    def run_benchmark(
        self,
        model: Union[str, Any],
        model_name: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Run benchmark on a model.

        Args:
            model: Model to benchmark
            model_name: Name for the model

        Returns:
            BenchmarkResult
        """
        if model_name is None:
            model_name = model if isinstance(model, str) else "model"

        logger.info(f"Starting benchmark for {model_name}")

        # Create evaluator
        eval_config = EvaluationConfig(
            metrics=self.config.metrics,
            batch_size=self.config.batch_size,
            save_predictions=self.config.save_predictions,
            device=self.config.device,
            fp16=self.config.fp16,
        )
        evaluator = Evaluator(model, config=eval_config)

        # Run benchmarks
        benchmark_scores = {}
        all_scores = []

        for benchmark_name, dataset in tqdm(self.datasets.items(), desc="Benchmarks"):
            logger.info(f"Running {benchmark_name}...")

            # Evaluate on benchmark
            result = evaluator.evaluate(dataset)

            # Store scores
            benchmark_scores[benchmark_name] = result.metrics
            all_scores.extend(result.metrics.values())

            logger.info(f"{benchmark_name} results: {result.metrics}")

        # Calculate overall score
        overall_score = np.mean([v for metrics in benchmark_scores.values() for v in metrics.values()])

        # Create result
        result = BenchmarkResult(
            model_name=model_name,
            benchmark_scores=benchmark_scores,
            overall_score=overall_score,
            metadata={
                "num_benchmarks": len(self.datasets),
                "config": self.config.__dict__,
            },
        )

        # Save result
        result.save(os.path.join(self.config.output_dir, f"{model_name}_benchmark.json"))
        result.print_summary()

        self.results.append(result)
        return result

    def compare_models(
        self,
        models: Dict[str, Union[str, Any]],
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            models: Dictionary of model name to model

        Returns:
            Comparison DataFrame
        """
        logger.info(f"Comparing {len(models)} models")

        # Run benchmarks for each model
        results = []
        for name, model in models.items():
            result = self.run_benchmark(model, name)
            results.append(result)

        # Create comparison DataFrame
        comparison_data = []
        for result in results:
            for benchmark, metrics in result.benchmark_scores.items():
                for metric, score in metrics.items():
                    comparison_data.append(
                        {"model": result.model_name, "benchmark": benchmark, "metric": metric, "score": score}
                    )

        df = pd.DataFrame(comparison_data)

        # Pivot for easier comparison
        pivot_df = df.pivot_table(index=["benchmark", "metric"], columns="model", values="score")

        # Save comparison
        comparison_path = os.path.join(self.config.output_dir, "model_comparison.csv")
        pivot_df.to_csv(comparison_path)
        logger.info(f"Comparison saved to {comparison_path}")

        # Print comparison
        print("\n" + "=" * 80)
        print("Model Comparison")
        print("=" * 80)
        print(pivot_df)
        print("=" * 80)

        # Generate report if requested
        if self.config.generate_report:
            self._generate_report(results, pivot_df)

        return pivot_df

    def _generate_report(self, results: List[BenchmarkResult], comparison_df: pd.DataFrame):
        """Generate HTML report of benchmark results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping report generation")
            return

        # Create report directory
        report_dir = os.path.join(self.config.output_dir, "report")
        os.makedirs(report_dir, exist_ok=True)

        # Generate plots
        # 1. Overall scores bar chart
        plt.figure(figsize=(10, 6))
        model_scores = {r.model_name: r.overall_score for r in results}
        plt.bar(model_scores.keys(), model_scores.values())
        plt.title("Overall Benchmark Scores")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "overall_scores.png"))
        plt.close()

        # 2. Heatmap of all scores
        plt.figure(figsize=(12, 8))
        sns.heatmap(comparison_df, annot=True, fmt=".3f", cmap="YlOrRd")
        plt.title("Benchmark Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "score_heatmap.png"))
        plt.close()

        # 3. Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Model Benchmark Report</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <h2>Overall Scores</h2>
            <img src="overall_scores.png" alt="Overall Scores">

            <h2>Detailed Comparison</h2>
            <img src="score_heatmap.png" alt="Score Heatmap">

            <h2>Numerical Results</h2>
            {comparison_df.to_html()}

            <h2>Model Rankings</h2>
            <ol>
            {"".join([f"<li>{r.model_name}: {r.overall_score:.4f}</li>"
                     for r in sorted(results, key=lambda x: x.overall_score, reverse=True)])}
            </ol>
        </body>
        </html>
        """

        report_path = os.path.join(report_dir, "index.html")
        with open(report_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report generated at {report_path}")


# Convenience functions
def run_benchmark(
    model: Union[str, Any], benchmarks: List[str] = None, metrics: List[str] = None, **kwargs
) -> BenchmarkResult:
    """
    Run benchmark on a single model.

    Args:
        model: Model to benchmark
        benchmarks: List of benchmark names
        metrics: List of metrics to compute

    Returns:
        BenchmarkResult
    """
    config = BenchmarkConfig(benchmarks=benchmarks or ["mmlu", "hellaswag"], metrics=metrics or ["accuracy"], **kwargs)

    benchmark = Benchmark(config)
    return benchmark.run_benchmark(model)


def compare_models(
    models: Dict[str, Union[str, Any]], benchmarks: List[str] = None, metrics: List[str] = None, **kwargs
) -> pd.DataFrame:
    """
    Compare multiple models on benchmarks.

    Args:
        models: Dictionary of model name to model
        benchmarks: List of benchmark names
        metrics: List of metrics to compute

    Returns:
        Comparison DataFrame
    """
    config = BenchmarkConfig(benchmarks=benchmarks or ["mmlu", "hellaswag"], metrics=metrics or ["accuracy"], **kwargs)

    benchmark = Benchmark(config)
    return benchmark.compare_models(models)
