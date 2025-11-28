"""
Evaluation Callbacks for Training
==================================

Callbacks for periodic evaluation during training.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from autotrain import logger


class EvaluationCallback(TrainerCallback):
    """Base class for evaluation callbacks."""

    def __init__(self, evaluator=None, eval_dataset=None):
        """
        Initialize evaluation callback.

        Args:
            evaluator: Evaluator instance
            eval_dataset: Evaluation dataset
        """
        self.evaluator = evaluator
        self.eval_dataset = eval_dataset
        self.history = []

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called after evaluation."""
        # Store evaluation results
        if state.log_history:
            latest_metrics = state.log_history[-1]
            self.history.append(
                {
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "metrics": latest_metrics,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def get_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history."""
        return self.history

    def save_history(self, filepath: str):
        """Save evaluation history to file."""
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Evaluation history saved to {filepath}")


class PeriodicEvalCallback(EvaluationCallback):
    """Callback for periodic evaluation during training."""

    def __init__(self, evaluator, eval_dataset, eval_steps: int = 100, metrics: Optional[List[str]] = None):
        """
        Initialize periodic evaluation.

        Args:
            evaluator: Evaluator instance
            eval_dataset: Evaluation dataset
            eval_steps: Evaluate every N steps
            metrics: Specific metrics to compute
        """
        super().__init__(evaluator, eval_dataset)
        self.eval_steps = eval_steps
        self.metrics = metrics
        self.last_eval_step = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Check if evaluation should be performed."""
        if state.global_step - self.last_eval_step >= self.eval_steps:
            self.last_eval_step = state.global_step

            # Run evaluation
            logger.info(f"Running periodic evaluation at step {state.global_step}")
            result = self.evaluator.evaluate(self.eval_dataset, self.metrics)

            # Log metrics
            for metric_name, value in result.metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")

            # Store in history
            self.history.append(
                {
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "metrics": result.metrics,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Update state with metrics
            state.log_history.append(
                {"step": state.global_step, **{f"eval_{k}": v for k, v in result.metrics.items()}}
            )


class BestModelCallback(EvaluationCallback):
    """Callback to save the best model based on evaluation metric."""

    def __init__(
        self,
        save_dir: str = "./best_model",
        metric: str = "eval_loss",
        mode: str = "min",
        save_top_k: int = 1,
        min_delta: float = 0.001,
    ):
        """
        Initialize best model callback.

        Args:
            save_dir: Directory to save best model
            metric: Metric to track
            mode: "min" or "max"
            save_top_k: Number of best models to keep
            min_delta: Minimum change to consider improvement
        """
        super().__init__()
        self.save_dir = save_dir
        self.metric = metric
        self.mode = mode
        self.save_top_k = save_top_k
        self.min_delta = min_delta
        self.best_metrics = []
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Check if current model is best and save if needed."""
        # Get metric value
        metric_value = None
        for log in reversed(state.log_history):
            if self.metric in log:
                metric_value = log[self.metric]
                break

        if metric_value is None:
            return

        # Check if improvement
        is_better = False
        if self.mode == "min":
            is_better = metric_value < self.best_value - self.min_delta
        else:
            is_better = metric_value > self.best_value + self.min_delta

        if is_better:
            self.best_value = metric_value
            logger.info(f"New best {self.metric}: {metric_value:.4f}")

            # Save model
            if model is not None:
                save_path = os.path.join(self.save_dir, f"step_{state.global_step}")
                os.makedirs(save_path, exist_ok=True)

                model.save_pretrained(save_path)
                if hasattr(model, "tokenizer"):
                    model.tokenizer.save_pretrained(save_path)

                # Save metric info
                with open(os.path.join(save_path, "metric_info.json"), "w") as f:
                    json.dump(
                        {
                            "metric": self.metric,
                            "value": metric_value,
                            "step": state.global_step,
                            "epoch": state.epoch,
                        },
                        f,
                        indent=2,
                    )

                logger.info(f"Best model saved to {save_path}")

                # Update best metrics list
                self.best_metrics.append({"path": save_path, "metric": metric_value, "step": state.global_step})

                # Remove old checkpoints if needed
                if len(self.best_metrics) > self.save_top_k:
                    # Sort by metric value
                    self.best_metrics.sort(key=lambda x: x["metric"], reverse=(self.mode == "max"))
                    # Remove worst
                    to_remove = self.best_metrics[self.save_top_k :]
                    for item in to_remove:
                        if os.path.exists(item["path"]):
                            import shutil

                            shutil.rmtree(item["path"])
                            logger.info(f"Removed checkpoint {item['path']}")
                    self.best_metrics = self.best_metrics[: self.save_top_k]


class EarlyStoppingCallback(EvaluationCallback):
    """Callback for early stopping based on evaluation metric."""

    def __init__(self, metric: str = "eval_loss", mode: str = "min", patience: int = 3, min_delta: float = 0.001):
        """
        Initialize early stopping callback.

        Args:
            metric: Metric to monitor
            mode: "min" or "max"
            patience: Number of evaluations to wait
            min_delta: Minimum change to consider improvement
        """
        super().__init__()
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.patience_counter = 0

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Check for early stopping condition."""
        # Get metric value
        metric_value = None
        for log in reversed(state.log_history):
            if self.metric in log:
                metric_value = log[self.metric]
                break

        if metric_value is None:
            return control

        # Check if improvement
        is_better = False
        if self.mode == "min":
            is_better = metric_value < self.best_value - self.min_delta
        else:
            is_better = metric_value > self.best_value + self.min_delta

        if is_better:
            self.best_value = metric_value
            self.patience_counter = 0
            logger.info(f"Improvement in {self.metric}: {metric_value:.4f}")
        else:
            self.patience_counter += 1
            logger.info(f"No improvement for {self.patience_counter} evaluations")

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered! Best {self.metric}: {self.best_value:.4f}")
                control.should_training_stop = True

        return control


class MetricsLoggerCallback(EvaluationCallback):
    """Callback to log metrics to file during training."""

    def __init__(self, log_file: str = "./metrics_log.jsonl", log_to_console: bool = True):
        """
        Initialize metrics logger.

        Args:
            log_file: Path to log file
            log_to_console: Whether to also log to console
        """
        super().__init__()
        self.log_file = log_file
        self.log_to_console = log_to_console

        # Create directory if needed
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Log metrics."""
        if logs is None:
            return

        # Prepare log entry
        log_entry = {
            "step": state.global_step,
            "epoch": state.epoch if state.epoch else 0,
            "timestamp": datetime.now().isoformat(),
            **logs,
        }

        # Write to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Log to console if requested
        if self.log_to_console:
            metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items()])
            logger.info(f"Step {state.global_step}: {metrics_str}")


class ComparisonCallback(EvaluationCallback):
    """Callback to compare model with baseline during training."""

    def __init__(
        self, evaluator, eval_dataset, baseline_model, eval_steps: int = 500, metrics: Optional[List[str]] = None
    ):
        """
        Initialize comparison callback.

        Args:
            evaluator: Evaluator for current model
            eval_dataset: Evaluation dataset
            baseline_model: Baseline model to compare
            eval_steps: Steps between comparisons
            metrics: Metrics to compare
        """
        super().__init__(evaluator, eval_dataset)
        self.baseline_model = baseline_model
        self.eval_steps = eval_steps
        self.metrics = metrics
        self.last_eval_step = 0
        self.comparison_history = []

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Run comparison if needed."""
        if state.global_step - self.last_eval_step >= self.eval_steps:
            self.last_eval_step = state.global_step

            logger.info(f"Running model comparison at step {state.global_step}")

            # Compare models
            comparison = self.evaluator.compare_with_baseline(self.eval_dataset, self.baseline_model)

            # Store comparison
            self.comparison_history.append({"step": state.global_step, "comparison": comparison})

            # Log improvements
            for metric, improvement in comparison["improvements"].items():
                logger.info(f"  {metric} improvement: {improvement:.2f}%")
