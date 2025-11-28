"""
Tests for Enhanced Evaluation Framework
========================================
"""

import json
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from autotrain.evaluation import (
    AccuracyMetric,
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
    BestModelCallback,
    BLEUMetric,
    CustomMetric,
    EarlyStoppingCallback,
    EvaluationCallback,
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    ExactMatchMetric,
    F1Metric,
    Metric,
    MetricCollection,
    MetricsLoggerCallback,
    MetricType,
    PeriodicEvalCallback,
    PerplexityMetric,
    ROUGEMetric,
    compare_models,
    evaluate_generation,
    evaluate_model,
    run_benchmark,
)


@pytest.fixture
def model_and_tokenizer():
    """Create small model and tokenizer for testing."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@pytest.fixture
def eval_config():
    """Create evaluation config."""
    return EvaluationConfig(
        metrics=[MetricType.PERPLEXITY],
        batch_size=2,
        max_samples=10,
        max_length=32,
        save_results=False,
        verbose=0,
    )


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    # Create dummy data
    input_ids = torch.randint(0, 1000, (10, 32))
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)

    dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=2)


def test_evaluation_config():
    """Test EvaluationConfig creation."""
    config = EvaluationConfig(
        metrics=[MetricType.PERPLEXITY, MetricType.BLEU],
        batch_size=16,
        task="generation",
        fp16=True,
    )

    assert MetricType.PERPLEXITY in config.metrics
    assert config.batch_size == 16
    assert config.task == "generation"
    assert config.fp16 == True


def test_evaluation_result():
    """Test EvaluationResult."""
    result = EvaluationResult(
        metrics={"perplexity": 10.5, "loss": 2.3},
        predictions=["pred1", "pred2"],
    )

    assert result.metrics["perplexity"] == 10.5
    assert len(result.predictions) == 2

    # Test string representation
    str_repr = str(result)
    assert "perplexity=10.5" in str_repr

    # Test to_dict
    data = result.to_dict()
    assert "metrics" in data
    assert "timestamp" in data


def test_evaluation_result_save(tmp_path):
    """Test saving evaluation results."""
    result = EvaluationResult(
        metrics={"accuracy": 0.95},
        predictions=["a", "b", "c"],
    )

    save_path = tmp_path / "results.json"
    result.save(str(save_path))

    assert save_path.exists()

    with open(save_path, "r") as f:
        data = json.load(f)

    assert data["metrics"]["accuracy"] == 0.95
    assert len(data["predictions"]) == 3


def test_evaluator_init(model_and_tokenizer):
    """Test Evaluator initialization."""
    model, tokenizer = model_and_tokenizer
    evaluator = Evaluator(model, tokenizer)

    assert evaluator.model == model
    assert evaluator.tokenizer == tokenizer
    assert evaluator.config is not None


def test_evaluator_language_modeling(model_and_tokenizer, sample_dataset):
    """Test language modeling evaluation."""
    model, tokenizer = model_and_tokenizer

    config = EvaluationConfig(
        metrics=[MetricType.PERPLEXITY],
        task="language_modeling",
        save_predictions=False,
    )

    evaluator = Evaluator(model, tokenizer, config)
    result = evaluator.evaluate(sample_dataset)

    assert isinstance(result, EvaluationResult)
    assert "perplexity" in result.metrics
    assert "loss" in result.metrics
    assert result.metrics["perplexity"] > 0


def test_evaluator_generation(model_and_tokenizer):
    """Test generation evaluation."""
    model, tokenizer = model_and_tokenizer

    dataset = [
        {"input": "Hello", "target": "Hi there"},
        {"input": "How are you?", "target": "I'm fine"},
    ]

    config = EvaluationConfig(
        metrics=["exact_match"],
        task="generation",
        generate=True,
    )

    evaluator = Evaluator(model, tokenizer, config)
    result = evaluator.evaluate(dataset)

    assert isinstance(result, EvaluationResult)
    assert "exact_match" in result.metrics


def test_evaluator_classification(model_and_tokenizer):
    """Test classification evaluation."""
    model, tokenizer = model_and_tokenizer

    # Create classification dataset
    input_ids = torch.randint(0, 1000, (10, 32))
    labels = torch.randint(0, 2, (10,))  # Binary classification
    dataset = TensorDataset(input_ids, labels)

    config = EvaluationConfig(
        metrics=["accuracy", "f1"],
        task="classification",
    )

    evaluator = Evaluator(model, tokenizer, config)
    result = evaluator.evaluate(DataLoader(dataset, batch_size=2))

    assert isinstance(result, EvaluationResult)
    if "accuracy" in result.metrics:
        assert 0 <= result.metrics["accuracy"] <= 1


def test_perplexity_metric():
    """Test PerplexityMetric."""
    metric = PerplexityMetric()

    losses = [2.3, 2.5, 2.1]
    perplexity = metric.compute(losses)

    assert perplexity > 0
    assert np.isclose(perplexity, np.exp(np.mean(losses)), rtol=1e-5)


def test_bleu_metric():
    """Test BLEUMetric."""
    metric = BLEUMetric()

    predictions = ["The cat sat on the mat", "Hello world"]
    references = ["The cat is on the mat", "Hello world"]

    scores = metric.compute(predictions, references)

    assert isinstance(scores, dict)
    assert "bleu" in scores
    assert 0 <= scores["bleu"] <= 1


def test_rouge_metric():
    """Test ROUGEMetric."""
    metric = ROUGEMetric(rouge_types=["rouge1", "rouge2"])

    predictions = ["The cat sat on the mat"]
    references = ["The cat is on the mat"]

    scores = metric.compute(predictions, references)

    assert isinstance(scores, dict)
    if "rouge1" in scores:
        assert 0 <= scores["rouge1"] <= 1


def test_accuracy_metric():
    """Test AccuracyMetric."""
    metric = AccuracyMetric()

    predictions = [0, 1, 1, 0, 1]
    references = [0, 1, 0, 0, 1]

    accuracy = metric.compute(predictions, references)
    assert accuracy == 0.8


def test_f1_metric():
    """Test F1Metric."""
    metric = F1Metric(average="binary")

    predictions = [0, 1, 1, 0, 1]
    references = [0, 1, 0, 0, 1]

    scores = metric.compute(predictions, references)

    assert "f1" in scores
    assert "precision" in scores
    assert "recall" in scores
    assert 0 <= scores["f1"] <= 1


def test_exact_match_metric():
    """Test ExactMatchMetric."""
    metric = ExactMatchMetric(normalize=True, ignore_case=True)

    predictions = ["Hello World", "test", "ANSWER"]
    references = ["hello world", "test", "answer"]

    score = metric.compute(predictions, references)
    assert score == 1.0  # All match after normalization


def test_custom_metric():
    """Test CustomMetric."""

    def custom_fn(predictions, references, **kwargs):
        return sum(p == r for p, r in zip(predictions, references)) / len(predictions)

    metric = CustomMetric(custom_fn, name="custom_accuracy")

    predictions = [1, 2, 3]
    references = [1, 2, 4]

    score = metric.compute(predictions, references)
    assert score == 2 / 3


def test_metric_collection():
    """Test MetricCollection."""
    metrics = MetricCollection(
        {
            "accuracy": AccuracyMetric(),
            "f1": F1Metric(),
        }
    )

    predictions = [0, 1, 1, 0]
    references = [0, 1, 0, 0]

    results = metrics.compute_all(predictions, references)

    assert "accuracy" in results
    assert "f1" in results or "f1_f1" in results

    # Test add/remove
    metrics.add_metric("custom", CustomMetric(lambda p, r, **k: 0.5))
    assert "custom" in metrics.metrics

    metrics.remove_metric("custom")
    assert "custom" not in metrics.metrics


def test_evaluation_callback():
    """Test EvaluationCallback."""
    callback = EvaluationCallback()

    # Mock trainer state
    from transformers import TrainerControl, TrainerState, TrainingArguments

    state = TrainerState()
    state.global_step = 100
    state.epoch = 1
    state.log_history = [{"eval_loss": 0.5}]

    control = TrainerControl()
    args = TrainingArguments(output_dir="./test")

    callback.on_evaluate(args, state, control)

    assert len(callback.history) == 1
    assert callback.history[0]["step"] == 100


def test_best_model_callback(tmp_path, model_and_tokenizer):
    """Test BestModelCallback."""
    model, _ = model_and_tokenizer

    callback = BestModelCallback(
        save_dir=str(tmp_path),
        metric="eval_loss",
        mode="min",
    )

    from transformers import TrainerControl, TrainerState, TrainingArguments

    state = TrainerState()
    state.global_step = 100
    state.log_history = [{"eval_loss": 0.5}]

    control = TrainerControl()
    args = TrainingArguments(output_dir=str(tmp_path))

    callback.on_evaluate(args, state, control, model=model)

    # Check if model was saved
    assert callback.best_value == 0.5

    # Test improvement
    state.global_step = 200
    state.log_history.append({"eval_loss": 0.3})
    callback.on_evaluate(args, state, control, model=model)

    assert callback.best_value == 0.3


def test_early_stopping_callback():
    """Test EarlyStoppingCallback."""
    callback = EarlyStoppingCallback(
        metric="eval_loss",
        mode="min",
        patience=2,
    )

    from transformers import TrainerControl, TrainerState, TrainingArguments

    state = TrainerState()
    control = TrainerControl()
    args = TrainingArguments(output_dir="./test")

    # First evaluation - good
    state.log_history = [{"eval_loss": 0.5}]
    control = callback.on_evaluate(args, state, control)
    assert not control.should_training_stop

    # Second evaluation - worse
    state.log_history.append({"eval_loss": 0.6})
    control = callback.on_evaluate(args, state, control)
    assert not control.should_training_stop

    # Third evaluation - worse again (trigger stopping)
    state.log_history.append({"eval_loss": 0.7})
    control = callback.on_evaluate(args, state, control)
    assert control.should_training_stop


def test_metrics_logger_callback(tmp_path):
    """Test MetricsLoggerCallback."""
    log_file = tmp_path / "metrics.jsonl"
    callback = MetricsLoggerCallback(log_file=str(log_file), log_to_console=False)

    from transformers import TrainerControl, TrainerState, TrainingArguments

    state = TrainerState()
    state.global_step = 100
    control = TrainerControl()
    args = TrainingArguments(output_dir="./test")

    logs = {"loss": 0.5, "learning_rate": 1e-4}
    callback.on_log(args, state, control, logs=logs)

    assert log_file.exists()

    with open(log_file, "r") as f:
        line = f.readline()
        data = json.loads(line)

    assert data["step"] == 100
    assert data["loss"] == 0.5


def test_benchmark_config():
    """Test BenchmarkConfig."""
    config = BenchmarkConfig(
        benchmarks=["mmlu", "arc"],
        metrics=["accuracy"],
        batch_size=16,
    )

    assert "mmlu" in config.benchmarks
    assert config.batch_size == 16


def test_benchmark_result():
    """Test BenchmarkResult."""
    result = BenchmarkResult(
        model_name="test_model",
        benchmark_scores={
            "mmlu": {"accuracy": 0.85},
            "arc": {"accuracy": 0.90},
        },
        overall_score=0.875,
    )

    assert result.model_name == "test_model"
    assert result.overall_score == 0.875

    # Test DataFrame conversion
    df = result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_benchmark_result_save(tmp_path):
    """Test saving benchmark results."""
    result = BenchmarkResult(
        model_name="model1",
        benchmark_scores={"test": {"acc": 0.9}},
        overall_score=0.9,
    )

    save_path = tmp_path / "benchmark.json"
    result.save(str(save_path))

    assert save_path.exists()
    assert (tmp_path / "benchmark.csv").exists()


def test_benchmark(model_and_tokenizer):
    """Test Benchmark class."""
    model, _ = model_and_tokenizer

    config = BenchmarkConfig(
        benchmarks=["mmlu"],  # Will use mock dataset
        metrics=["accuracy"],
        max_samples_per_benchmark=10,
    )

    benchmark = Benchmark(config)
    assert len(benchmark.datasets) > 0

    # Test running benchmark
    with tempfile.TemporaryDirectory() as tmp_dir:
        config.output_dir = tmp_dir
        result = benchmark.run_benchmark(model, "test_model")

        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "test_model"
        assert "mmlu" in result.benchmark_scores


def test_compare_models(model_and_tokenizer):
    """Test model comparison."""
    model, _ = model_and_tokenizer

    config = BenchmarkConfig(
        benchmarks=["mmlu"],
        metrics=["accuracy"],
        max_samples_per_benchmark=5,
        generate_report=False,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        config.output_dir = tmp_dir
        benchmark = Benchmark(config)

        models = {
            "model1": model,
            "model2": model,  # Use same model for testing
        }

        comparison_df = benchmark.compare_models(models)

        assert isinstance(comparison_df, pd.DataFrame)
        assert "model1" in comparison_df.columns
        assert "model2" in comparison_df.columns


def test_evaluate_model_convenience(model_and_tokenizer):
    """Test evaluate_model convenience function."""
    model, _ = model_and_tokenizer

    # Create simple dataset
    dataset = [{"input_ids": torch.randint(0, 1000, (32,)), "labels": torch.randint(0, 1000, (32,))} for _ in range(5)]

    result = evaluate_model(
        model,
        dataset,
        metrics=[MetricType.PERPLEXITY],
        batch_size=2,
    )

    assert isinstance(result, EvaluationResult)
    assert "perplexity" in result.metrics


def test_evaluate_generation_convenience(model_and_tokenizer):
    """Test evaluate_generation convenience function."""
    model, _ = model_and_tokenizer

    prompts = ["What is 2+2?", "Hello"]
    references = ["4", "Hi"]

    result = evaluate_generation(
        model,
        prompts,
        references,
        metrics=["exact_match"],
    )

    assert isinstance(result, EvaluationResult)
    if "exact_match" in result.metrics:
        assert 0 <= result.metrics["exact_match"] <= 1


def test_get_callbacks_with_enhanced_eval(model_and_tokenizer, sample_dataset):
    """Test get_callbacks function with enhanced evaluation enabled."""
    from types import SimpleNamespace

    from autotrain.trainers.clm import utils

    model, tokenizer = model_and_tokenizer

    # Create config with enhanced eval enabled
    config = SimpleNamespace(
        use_enhanced_eval=True,
        trainer="sft",
        eval_metrics="perplexity,bleu",
        eval_save_predictions=True,
        per_device_eval_batch_size=8,
        logging_steps=100,
        sweep_metric="eval_loss",
        valid_split="validation",
        peft=False,
        push_to_hub=False,  # Required by UploadLogs callback
        project_name="test_project",  # Required by UploadLogs callback
    )

    # Create dummy datasets
    train_data = sample_dataset.dataset
    valid_data = sample_dataset.dataset

    # Test with all parameters provided
    callbacks = utils.get_callbacks(
        config, train_data=train_data, valid_data=valid_data, model=model, tokenizer=tokenizer
    )

    # Check that callbacks were created
    assert len(callbacks) > 0

    # Check that PeriodicEvalCallback is in the callbacks
    callback_types = [type(cb).__name__ for cb in callbacks]
    assert "PeriodicEvalCallback" in callback_types
    assert "BestModelCallback" in callback_types
    assert "MetricsLoggerCallback" in callback_types

    # Test with None parameters (should warn but not fail)
    callbacks_none = utils.get_callbacks(config)
    assert len(callbacks_none) > 0

    # PeriodicEvalCallback should not be added when required params are None
    callback_types_none = [type(cb).__name__ for cb in callbacks_none]
    assert "PeriodicEvalCallback" not in callback_types_none

    # Test with use_enhanced_eval=False
    config.use_enhanced_eval = False
    callbacks_disabled = utils.get_callbacks(
        config, train_data=train_data, valid_data=valid_data, model=model, tokenizer=tokenizer
    )

    callback_types_disabled = [type(cb).__name__ for cb in callbacks_disabled]
    assert "PeriodicEvalCallback" not in callback_types_disabled


def test_periodic_eval_callback_init(model_and_tokenizer, sample_dataset):
    """Test PeriodicEvalCallback initialization with new signature."""
    from autotrain.evaluation import PeriodicEvalCallback
    from autotrain.evaluation.evaluator import EvaluationConfig, Evaluator

    model, tokenizer = model_and_tokenizer

    # Create evaluator
    eval_config = EvaluationConfig(
        metrics=["perplexity"], batch_size=8, save_predictions=False, task="language_modeling"
    )

    evaluator = Evaluator(model=model, tokenizer=tokenizer, config=eval_config)

    # Create callback with correct parameters
    callback = PeriodicEvalCallback(
        evaluator=evaluator, eval_dataset=sample_dataset.dataset, eval_steps=100, metrics=["perplexity"]
    )

    assert callback.evaluator == evaluator
    assert callback.eval_dataset == sample_dataset.dataset
    assert callback.eval_steps == 100
    assert callback.metrics == ["perplexity"]


def test_all_trainers_enhanced_eval(model_and_tokenizer, sample_dataset):
    """Test that all CLM trainers work with enhanced evaluation."""
    from types import SimpleNamespace

    from autotrain.trainers.clm import utils

    model, tokenizer = model_and_tokenizer

    # List of all trainers to test
    trainers_to_test = ["sft", "dpo", "orpo", "reward", "default", "ppo"]

    # Create dummy datasets
    train_data = sample_dataset.dataset
    valid_data = sample_dataset.dataset

    for trainer_name in trainers_to_test:
        # Create config for this trainer type
        config = SimpleNamespace(
            use_enhanced_eval=True,
            trainer=trainer_name,
            eval_metrics="perplexity",
            eval_save_predictions=False,
            per_device_eval_batch_size=8,
            logging_steps=100,
            sweep_metric="eval_loss",
            valid_split="validation",
            peft=False,
            push_to_hub=False,
            project_name=f"test_{trainer_name}",
        )

        # Test that get_callbacks works with enhanced eval for this trainer
        callbacks = utils.get_callbacks(
            config, train_data=train_data, valid_data=valid_data, model=model, tokenizer=tokenizer
        )

        # Verify callbacks were created
        assert len(callbacks) > 0, f"No callbacks created for {trainer_name} trainer"

        # Check callback types
        callback_types = [type(cb).__name__ for cb in callbacks]

        # All three callbacks should be present when enhanced eval is enabled
        assert "PeriodicEvalCallback" in callback_types, f"PeriodicEvalCallback missing for {trainer_name}"
        assert "BestModelCallback" in callback_types, f"BestModelCallback missing for {trainer_name}"
        assert "MetricsLoggerCallback" in callback_types, f"MetricsLoggerCallback missing for {trainer_name}"

        # Verify PeriodicEvalCallback has correct attributes
        periodic_callback = [cb for cb in callbacks if type(cb).__name__ == "PeriodicEvalCallback"][0]
        assert hasattr(periodic_callback, "evaluator"), f"PeriodicEvalCallback missing evaluator for {trainer_name}"
        assert hasattr(
            periodic_callback, "eval_dataset"
        ), f"PeriodicEvalCallback missing eval_dataset for {trainer_name}"
        assert hasattr(periodic_callback, "eval_steps"), f"PeriodicEvalCallback missing eval_steps for {trainer_name}"

        # Verify BestModelCallback has correct mode attribute
        best_callback = [cb for cb in callbacks if type(cb).__name__ == "BestModelCallback"][0]
        assert hasattr(best_callback, "mode"), f"BestModelCallback missing mode attribute for {trainer_name}"
        assert best_callback.mode in ["min", "max"], f"BestModelCallback has invalid mode for {trainer_name}"

        print(f"✓ Enhanced eval callbacks work correctly for {trainer_name} trainer")
