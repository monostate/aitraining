"""
Main Evaluator Class for Model Evaluation
==========================================

Provides comprehensive evaluation of language models.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from autotrain import logger
from autotrain.generation import CompletionConfig
from autotrain.utils import get_model_loading_kwargs, maybe_move_to_mps


class MetricType(Enum):
    """Types of evaluation metrics."""

    PERPLEXITY = "perplexity"
    BLEU = "bleu"
    ROUGE = "rouge"
    BERT_SCORE = "bert_score"
    ACCURACY = "accuracy"
    F1 = "f1"
    EXACT_MATCH = "exact_match"
    METEOR = "meteor"
    CUSTOM = "custom"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    # Metrics to compute
    metrics: List[Union[MetricType, str]] = field(default_factory=lambda: [MetricType.PERPLEXITY])

    # Data settings
    batch_size: int = 8
    max_samples: Optional[int] = None
    max_length: int = 512

    # Generation settings (for generation metrics)
    generate: bool = False
    generation_config: Optional[CompletionConfig] = None

    # Task-specific settings
    task: str = "language_modeling"  # "language_modeling", "generation", "classification"
    target_key: str = "labels"
    prediction_key: str = "predictions"

    # Output settings
    save_results: bool = True
    output_dir: str = "./eval_results"
    save_predictions: bool = False

    # Device settings
    device: str = "auto"  # "auto", "cuda", "cpu"
    fp16: bool = False

    # Callback settings
    callbacks: List[Any] = field(default_factory=list)
    verbose: int = 1


@dataclass
class EvaluationResult:
    """Results from evaluation."""

    metrics: Dict[str, float]
    predictions: Optional[List[Any]] = None
    targets: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __str__(self):
        """String representation."""
        metric_str = ", ".join([f"{k}={v:.4f}" for k, v in self.metrics.items()])
        return f"EvaluationResult({metric_str})"

    def save(self, filepath: Optional[str] = None):
        """Save results to file."""
        if filepath is None:
            filepath = "eval_results.json"

        data = {
            "metrics": self.metrics,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

        if self.predictions and len(self.predictions) < 1000:  # Don't save huge lists
            data["predictions"] = self.predictions
        if self.targets and len(self.targets) < 1000:
            data["targets"] = self.targets

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Evaluation results saved to {filepath}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class Evaluator:
    """Main evaluator class for model evaluation."""

    def __init__(
        self,
        model: Union[str, AutoModelForCausalLM],
        tokenizer: Union[str, AutoTokenizer, None] = None,
        config: Optional[EvaluationConfig] = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: Model or model name
            tokenizer: Tokenizer or tokenizer name
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()

        # Load model if string
        if isinstance(model, str):
            # Build consistent loading kwargs; respect fp16 preference when CUDA
            prefer_fp16 = bool(self.config.fp16)
            model_kwargs = get_model_loading_kwargs(fp16_if_cuda=prefer_fp16, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
            self.model = maybe_move_to_mps(self.model, model_kwargs)
        else:
            self.model = model

        # Load tokenizer
        if tokenizer is None:
            if hasattr(self.model, "config") and hasattr(self.model.config, "name_or_path"):
                self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.name_or_path)
            else:
                raise ValueError("Tokenizer must be provided")
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize metrics
        self.metrics = self._initialize_metrics()

        # Create output directory
        if self.config.save_results:
            os.makedirs(self.config.output_dir, exist_ok=True)

    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize metric calculators."""
        from .metrics import (
            AccuracyMetric,
            BERTScoreMetric,
            BLEUMetric,
            ExactMatchMetric,
            F1Metric,
            PerplexityMetric,
            ROUGEMetric,
        )

        metrics = {}
        for metric_type in self.config.metrics:
            if isinstance(metric_type, str):
                metric_type = MetricType(metric_type)

            if metric_type == MetricType.PERPLEXITY:
                metrics["perplexity"] = PerplexityMetric()
            elif metric_type == MetricType.BLEU:
                metrics["bleu"] = BLEUMetric()
            elif metric_type == MetricType.ROUGE:
                metrics["rouge"] = ROUGEMetric()
            elif metric_type == MetricType.BERT_SCORE:
                metrics["bert_score"] = BERTScoreMetric()
            elif metric_type == MetricType.ACCURACY:
                metrics["accuracy"] = AccuracyMetric()
            elif metric_type == MetricType.F1:
                metrics["f1"] = F1Metric()
            elif metric_type == MetricType.EXACT_MATCH:
                metrics["exact_match"] = ExactMatchMetric()

        return metrics

    def evaluate(
        self,
        dataset: Union[DataLoader, List[Dict[str, Any]]],
        metric_names: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate model on dataset.

        Args:
            dataset: Evaluation dataset
            metric_names: Specific metrics to compute

        Returns:
            EvaluationResult
        """
        logger.info(f"Starting evaluation on {len(dataset)} samples")

        # Select metrics
        if metric_names:
            active_metrics = {k: v for k, v in self.metrics.items() if k in metric_names}
        else:
            active_metrics = self.metrics

        # Run evaluation based on task
        if self.config.task == "language_modeling":
            return self._evaluate_language_modeling(dataset, active_metrics)
        elif self.config.task == "generation":
            return self._evaluate_generation(dataset, active_metrics)
        elif self.config.task == "classification":
            return self._evaluate_classification(dataset, active_metrics)
        else:
            raise ValueError(f"Unknown task: {self.config.task}")

    def _evaluate_language_modeling(
        self,
        dataset: Union[DataLoader, List[Dict[str, Any]]],
        metrics: Dict[str, Any],
    ) -> EvaluationResult:
        """Evaluate language modeling task."""
        self.model.eval()

        total_loss = 0
        total_tokens = 0
        all_predictions = []
        all_targets = []

        # Create DataLoader if needed
        if not isinstance(dataset, DataLoader):
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )
        else:
            dataloader = dataset

        # Progress bar
        if self.config.verbose:
            dataloader = tqdm(dataloader, desc="Evaluating")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if self.config.max_samples and batch_idx * self.config.batch_size >= self.config.max_samples:
                    break

                # Move to device
                if isinstance(batch, dict):
                    # Check if already tokenized
                    if "input_ids" in batch:
                        input_ids = batch["input_ids"].to(self.model.device)
                        labels = batch.get("labels", input_ids).to(self.model.device)
                        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)
                    else:
                        # Need to tokenize raw text data
                        # Extract text from various possible keys
                        # Note: DataLoader collation may create lists/tuples of values
                        texts = []

                        # Handle batched or single samples
                        if "text" in batch:
                            texts = batch["text"] if isinstance(batch["text"], (list, tuple)) else [batch["text"]]
                            len(texts)
                        elif "input" in batch:
                            texts = batch["input"] if isinstance(batch["input"], (list, tuple)) else [batch["input"]]
                            len(texts)
                        elif "question" in batch:
                            # For QA datasets, combine question with choices if available
                            questions = (
                                batch["question"]
                                if isinstance(batch["question"], (list, tuple))
                                else [batch["question"]]
                            )
                            len(questions)

                            if "choices" in batch:
                                choices_batch = batch["choices"]
                                # Handle nested structure from DataLoader collation
                                if isinstance(choices_batch, (list, tuple)) and len(choices_batch) > 0:
                                    for i, q in enumerate(questions):
                                        if i < len(choices_batch) and isinstance(choices_batch[i], (list, tuple)):
                                            choice_str = ", ".join(str(c) for c in choices_batch[i])
                                            texts.append(f"{q} Options: {choice_str}")
                                        else:
                                            texts.append(q)
                                else:
                                    texts = questions
                            else:
                                texts = questions
                        elif "context" in batch:
                            texts = (
                                batch["context"] if isinstance(batch["context"], (list, tuple)) else [batch["context"]]
                            )
                            len(texts)
                        else:
                            # Try to find any string value
                            for v in batch.values():
                                if isinstance(v, str):
                                    texts = [v]
                                    break
                                elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str):
                                    texts = v
                                    len(texts)
                                    break

                        if not texts:
                            raise ValueError(f"Could not extract text from batch: {batch.keys()}")

                        # Tokenize
                        encoded = self.tokenizer(
                            texts,
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                            max_length=self.config.max_length,
                        )
                        input_ids = encoded["input_ids"].to(self.model.device)
                        attention_mask = encoded["attention_mask"].to(self.model.device)
                        labels = input_ids
                elif isinstance(batch, (list, tuple)):
                    # Handle tuple/list from TensorDataset (input_ids, attention_mask, labels)
                    if len(batch) == 3:
                        input_ids, attention_mask, labels = batch
                        input_ids = input_ids.to(self.model.device)
                        attention_mask = attention_mask.to(self.model.device)
                        labels = labels.to(self.model.device)
                    elif len(batch) == 2:
                        input_ids, attention_mask = batch
                        input_ids = input_ids.to(self.model.device)
                        attention_mask = attention_mask.to(self.model.device)
                        labels = input_ids
                    else:
                        input_ids = batch[0].to(self.model.device)
                        labels = input_ids
                        attention_mask = torch.ones_like(input_ids)
                else:
                    # Assume batch is input_ids tensor
                    input_ids = batch.to(self.model.device)
                    labels = input_ids
                    attention_mask = torch.ones_like(input_ids)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # Accumulate loss
                loss = outputs.loss
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

                # Store predictions if needed
                if self.config.save_predictions:
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    all_predictions.extend(predictions.cpu().tolist())
                    all_targets.extend(labels.cpu().tolist())

                # Run callbacks
                for callback in self.config.callbacks:
                    if hasattr(callback, "on_batch_end"):
                        callback.on_batch_end(batch_idx, batch, outputs)

        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = np.exp(avg_loss) if avg_loss < 100 else float("inf")

        # Compile results
        result_metrics = {"loss": avg_loss, "perplexity": perplexity}

        # Calculate additional metrics if we have predictions
        if all_predictions and all_targets:
            for metric_name, metric_fn in metrics.items():
                if metric_name not in ["perplexity", "loss"]:
                    try:
                        result_metrics[metric_name] = metric_fn.compute(all_predictions, all_targets)
                    except:
                        pass

        return EvaluationResult(
            metrics=result_metrics,
            predictions=all_predictions if self.config.save_predictions else None,
            targets=all_targets if self.config.save_predictions else None,
            metadata={
                "task": "language_modeling",
                "num_samples": len(dataloader.dataset) if hasattr(dataloader, "dataset") else len(dataset),
                "model": self.model.config.name_or_path if hasattr(self.model.config, "name_or_path") else "unknown",
            },
        )

    def _evaluate_generation(
        self,
        dataset: Union[DataLoader, List[Dict[str, Any]]],
        metrics: Dict[str, Any],
    ) -> EvaluationResult:
        """Evaluate generation task."""
        from autotrain.generation import TokenCompleter

        # Create completer
        completer = TokenCompleter(self.model, self.tokenizer, self.config.generation_config or CompletionConfig())

        all_predictions = []
        all_references = []

        # Process dataset
        if isinstance(dataset, DataLoader):
            samples = []
            for batch in dataset:
                if isinstance(batch, dict):
                    samples.extend([{k: v[i] for k, v in batch.items()} for i in range(len(batch["input"]))])
                else:
                    samples.extend(batch)
        else:
            samples = dataset

        # Limit samples
        if self.config.max_samples:
            samples = samples[: self.config.max_samples]

        # Generate predictions
        if self.config.verbose:
            samples = tqdm(samples, desc="Generating")

        for sample in samples:
            # Get input and reference
            if isinstance(sample, dict):
                input_text = sample.get("input", sample.get("text", ""))
                reference = sample.get("target", sample.get("output", ""))
            else:
                input_text = sample
                reference = ""

            # Generate
            result = completer.complete(input_text)
            prediction = result.text

            all_predictions.append(prediction)
            all_references.append(reference)

            # Run callbacks
            for callback in self.config.callbacks:
                if hasattr(callback, "on_generation"):
                    callback.on_generation(input_text, prediction, reference)

        # Calculate metrics
        result_metrics = {}
        for metric_name, metric_fn in metrics.items():
            try:
                if metric_name in ["bleu", "rouge", "meteor", "bert_score"]:
                    result_metrics[metric_name] = metric_fn.compute(all_predictions, all_references)
                elif metric_name == "exact_match":
                    result_metrics[metric_name] = metric_fn.compute(all_predictions, all_references)
            except Exception as e:
                logger.warning(f"Failed to compute {metric_name}: {e}")

        return EvaluationResult(
            metrics=result_metrics,
            predictions=all_predictions if self.config.save_predictions else None,
            targets=all_references if self.config.save_predictions else None,
            metadata={
                "task": "generation",
                "num_samples": len(all_predictions),
                "model": self.model.config.name_or_path if hasattr(self.model.config, "name_or_path") else "unknown",
            },
        )

    def _evaluate_classification(
        self,
        dataset: Union[DataLoader, List[Dict[str, Any]]],
        metrics: Dict[str, Any],
    ) -> EvaluationResult:
        """Evaluate classification task."""
        self.model.eval()

        all_predictions = []
        all_targets = []

        # Create DataLoader if needed
        if not isinstance(dataset, DataLoader):
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )
        else:
            dataloader = dataset

        # Progress bar
        if self.config.verbose:
            dataloader = tqdm(dataloader, desc="Evaluating")

        with torch.no_grad():
            for batch in dataloader:
                # Get inputs and labels
                if isinstance(batch, dict):
                    inputs = batch["input_ids"].to(self.model.device)
                    labels = batch["labels"].to(self.model.device)
                    attention_mask = batch.get("attention_mask", torch.ones_like(inputs)).to(self.model.device)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(self.model.device)
                    labels = labels.to(self.model.device)
                    attention_mask = torch.ones_like(inputs)

                # Forward pass
                outputs = self.model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                )

                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().tolist())
                all_targets.extend(labels.cpu().tolist())

        # Calculate metrics
        result_metrics = {}
        for metric_name, metric_fn in metrics.items():
            try:
                result_metrics[metric_name] = metric_fn.compute(all_predictions, all_targets)
            except Exception as e:
                logger.warning(f"Failed to compute {metric_name}: {e}")

        return EvaluationResult(
            metrics=result_metrics,
            predictions=all_predictions if self.config.save_predictions else None,
            targets=all_targets if self.config.save_predictions else None,
            metadata={
                "task": "classification",
                "num_samples": len(all_predictions),
                "model": self.model.config.name_or_path if hasattr(self.model.config, "name_or_path") else "unknown",
            },
        )

    def compare_with_baseline(
        self,
        dataset: Any,
        baseline_model: Union[str, AutoModelForCausalLM],
    ) -> Dict[str, EvaluationResult]:
        """
        Compare current model with baseline.

        Args:
            dataset: Evaluation dataset
            baseline_model: Baseline model to compare with

        Returns:
            Dictionary with results for both models
        """
        # Evaluate current model
        current_results = self.evaluate(dataset)

        # Create evaluator for baseline
        baseline_evaluator = Evaluator(baseline_model, self.tokenizer, self.config)

        # Evaluate baseline
        baseline_results = baseline_evaluator.evaluate(dataset)

        # Compare and log
        logger.info("\n=== Comparison Results ===")
        logger.info(f"Current Model: {current_results}")
        logger.info(f"Baseline Model: {baseline_results}")

        # Calculate improvements
        improvements = {}
        for metric in current_results.metrics:
            if metric in baseline_results.metrics:
                current_val = current_results.metrics[metric]
                baseline_val = baseline_results.metrics[metric]

                if metric in ["perplexity", "loss"]:  # Lower is better
                    improvement = (baseline_val - current_val) / baseline_val * 100
                else:  # Higher is better
                    improvement = (current_val - baseline_val) / baseline_val * 100

                improvements[metric] = improvement
                logger.info(f"{metric} improvement: {improvement:.2f}%")

        return {"current": current_results, "baseline": baseline_results, "improvements": improvements}


# Convenience functions
def evaluate_model(
    model: Union[str, AutoModelForCausalLM],
    dataset: Any,
    metrics: List[Union[str, MetricType]] = None,
    task: str = "language_modeling",
    **kwargs,
) -> EvaluationResult:
    """
    Evaluate a model on a dataset.

    Args:
        model: Model to evaluate
        dataset: Evaluation dataset
        metrics: Metrics to compute
        task: Task type

    Returns:
        EvaluationResult
    """
    config = EvaluationConfig(metrics=metrics or [MetricType.PERPLEXITY], task=task, **kwargs)

    evaluator = Evaluator(model, config=config)
    return evaluator.evaluate(dataset)


def evaluate_generation(
    model: Union[str, AutoModelForCausalLM],
    prompts: List[str],
    references: List[str],
    metrics: List[str] = None,
    **kwargs,
) -> EvaluationResult:
    """
    Evaluate generation quality.

    Args:
        model: Model to evaluate
        prompts: Input prompts
        references: Reference outputs
        metrics: Metrics to compute

    Returns:
        EvaluationResult
    """
    # Create dataset
    dataset = [{"input": p, "target": r} for p, r in zip(prompts, references)]

    config = EvaluationConfig(metrics=metrics or ["bleu", "rouge"], task="generation", generate=True, **kwargs)

    evaluator = Evaluator(model, config=config)
    return evaluator.evaluate(dataset)
