"""Common metrics utilities for all trainers."""

from typing import Any, Callable, Dict, List, Optional, Union

from autotrain import logger


# Cache loaded metrics to avoid re-loading
_METRIC_CACHE = {}

# Custom metric implementations
_CUSTOM_METRICS = {}


import math

from transformers import TrainerCallback


class PerplexityCallback(TrainerCallback):
    """Automatically add perplexity to logs based on loss values."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Add eval_perplexity when eval_loss is present
            if "eval_loss" in logs and "eval_perplexity" not in logs:
                logs["eval_perplexity"] = math.exp(min(logs["eval_loss"], 100))
            # Add training perplexity when loss is present
            if "loss" in logs and "perplexity" not in logs:
                logs["perplexity"] = math.exp(min(logs["loss"], 100))


class CustomMetricsCallback(TrainerCallback):
    """Generic callback for computing custom metrics during training.

    This callback can compute ANY metric based on what's available in logs or state.
    Users can register custom metric functions that receive the logs/state and compute values.
    """

    def __init__(self, metric_names, tokenizer=None):
        """
        Args:
            metric_names: List of metric names to compute
            tokenizer: Optional tokenizer for metrics that need text decoding
        """
        self.metric_names = metric_names
        self.tokenizer = tokenizer
        self.metrics = {}

        # Load the requested metrics
        for name in metric_names:
            # Check if it's a special metric that needs logs access
            if name == "perplexity":
                # Perplexity is computed from loss
                self.metrics[name] = lambda logs: {
                    "eval_perplexity": math.exp(min(logs.get("eval_loss", 100), 100)) if "eval_loss" in logs else None,
                    "perplexity": math.exp(min(logs.get("loss", 100), 100)) if "loss" in logs else None,
                }
            elif name in ["reward_mean", "reward_std", "advantage_mean"]:
                # PPO-specific metrics computed from logs
                self.metrics[name] = self._get_ppo_metric_func(name)
            elif name in _CUSTOM_METRICS:
                # User-registered custom metrics
                self.metrics[name] = _CUSTOM_METRICS[name]
            else:
                # Try to load from evaluate library (for post-evaluation)
                try:
                    metric = get_metric(name)
                    self.metrics[name] = metric
                except:
                    logger.warning(f"Could not load metric {name}")

    def _get_ppo_metric_func(self, name):
        """Get PPO-specific metric functions."""
        if name == "reward_mean":
            return lambda logs: {"reward_mean": logs.get("reward_mean")} if "reward_mean" in logs else {}
        elif name == "reward_std":
            return lambda logs: {"reward_std": logs.get("reward_std")} if "reward_std" in logs else {}
        elif name == "advantage_mean":
            return lambda logs: {"advantage_mean": logs.get("advantage_mean")} if "advantage_mean" in logs else {}
        return lambda logs: {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Compute metrics when logs are available."""
        if logs and self.metrics:
            for name, metric_func in self.metrics.items():
                if callable(metric_func):
                    try:
                        # For log-based metrics (like perplexity)
                        if name in ["perplexity", "reward_mean", "reward_std", "advantage_mean"]:
                            result = metric_func(logs)
                            if result:
                                # Filter out None values
                                result = {k: v for k, v in result.items() if v is not None}
                                logs.update(result)
                    except Exception as e:
                        logger.debug(f"Could not compute metric {name} from logs: {e}")


def register_custom_metric(name: str, func: Callable):
    """
    Register a custom metric function.

    Args:
        name: Name of the metric
        func: Callable that computes the metric
    """
    _CUSTOM_METRICS[name] = func
    logger.info(f"Registered custom metric: {name}")


def get_metric(metric_name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """
    Load a metric from evaluate library or custom implementations.

    Args:
        metric_name: Name of the metric to load (e.g., 'accuracy', 'bleu', 'rouge')
        config: Optional configuration for the metric
        **kwargs: Additional arguments passed to metric loading

    Returns:
        The loaded metric object

    Examples:
        >>> accuracy_metric = get_metric('accuracy')
        >>> bleu_metric = get_metric('bleu')
        >>> custom_metric = get_metric('my_custom_metric')  # if registered
    """
    # Check if it's a custom metric
    if metric_name in _CUSTOM_METRICS:
        # Return a wrapper that looks like an evaluate metric
        class CustomMetricWrapper:
            def __init__(self, compute_fn):
                self._compute_fn = compute_fn

            def compute(self, predictions=None, references=None, **kwargs):
                return self._compute_fn(predictions, references)

        return CustomMetricWrapper(_CUSTOM_METRICS[metric_name])

    # Check cache
    cache_key = f"{metric_name}_{str(config)}_{str(kwargs)}"
    if cache_key in _METRIC_CACHE:
        return _METRIC_CACHE[cache_key]

    try:
        # Try to load from evaluate library
        import evaluate

        metric = evaluate.load(metric_name, config, **kwargs)
        _METRIC_CACHE[cache_key] = metric
        logger.info(f"Loaded metric: {metric_name}")
        return metric
    except Exception as e:
        # Try old datasets.load_metric for compatibility
        try:
            from datasets import load_metric

            metric = load_metric(metric_name, config, **kwargs)
            _METRIC_CACHE[cache_key] = metric
            logger.info(f"Loaded metric (legacy): {metric_name}")
            return metric
        except Exception:
            logger.error(f"Failed to load metric {metric_name}: {e}")
            raise ValueError(
                f"Could not load metric '{metric_name}'. "
                f"Make sure it's available in 'evaluate' library or registered as custom metric."
            )


def get_compute_metrics_func(
    metric_names: Union[str, List[str]],
    tokenizer: Optional[Any] = None,
    is_regression: bool = False,
    is_multilabel: bool = False,
) -> Callable:
    """
    Generate a compute_metrics function with selected metrics.

    Args:
        metric_names: Single metric name or list of metric names
        tokenizer: Optional tokenizer for metrics that need it (e.g., BLEU)
        is_regression: Whether this is a regression task
        is_multilabel: Whether this is a multilabel classification task

    Returns:
        A compute_metrics function compatible with Transformers Trainer

    Examples:
        >>> compute_metrics = get_compute_metrics_func(['accuracy', 'f1'])
        >>> trainer = Trainer(compute_metrics=compute_metrics, ...)
    """
    if isinstance(metric_names, str):
        metric_names = [metric_names]

    # Load all metrics
    metrics = {}
    for name in metric_names:
        try:
            metrics[name] = get_metric(name)
        except Exception as e:
            logger.warning(f"Could not load metric {name}: {e}")

    if not metrics:
        logger.warning("No metrics loaded, returning None")
        return None

    def compute_metrics(eval_pred):
        """Compute metrics from predictions and labels."""
        predictions, labels = eval_pred

        # Handle different prediction formats
        if not is_regression:
            # Classification: predictions are logits
            import numpy as np

            if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
                # Multi-class or multi-label
                if is_multilabel:
                    predictions = (predictions > 0).astype(int)
                else:
                    predictions = np.argmax(predictions, axis=-1)
            else:
                # Binary classification with single output
                predictions = (predictions > 0).astype(int).squeeze()

        # Compute each metric
        results = {}
        for name, metric in metrics.items():
            try:
                # Some metrics need special handling for text generation
                if name in ["bleu", "sacrebleu", "rouge", "meteor", "bertscore", "bleurt"] and tokenizer:
                    # These metrics expect strings, not token IDs
                    if hasattr(tokenizer, "batch_decode"):
                        # For generation tasks, predictions might be logits or token IDs
                        import numpy as np

                        if len(predictions.shape) == 3:  # (batch, seq_len, vocab_size) - logits
                            # Convert logits to token IDs
                            pred_ids = np.argmax(predictions, axis=-1)
                        elif len(predictions.shape) == 2:  # (batch, seq_len) - already token IDs
                            pred_ids = predictions
                        else:
                            # Unexpected shape, try to use as-is
                            pred_ids = predictions

                        # Decode to text
                        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

                        # Handle labels that might have -100 for padding
                        import numpy as np

                        if isinstance(labels, np.ndarray):
                            # Replace -100 with pad token id for decoding
                            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

                        # Handle special formatting for specific metrics
                        if name in ["sacrebleu", "bleu"]:
                            # BLEU/SacreBLEU expect references as list of lists
                            label_str = [[ref] for ref in label_str]

                        result = metric.compute(predictions=pred_str, references=label_str)
                    else:
                        result = metric.compute(predictions=predictions, references=labels)
                else:
                    result = metric.compute(predictions=predictions, references=labels)

                # Handle different result formats
                if isinstance(result, dict):
                    # Add prefix to avoid key collisions
                    for k, v in result.items():
                        results[f"{name}_{k}" if k != name else name] = v
                else:
                    results[name] = result

            except Exception as e:
                logger.warning(f"Error computing metric {name}: {e}")
                # Provide more helpful error message
                if name in ["bleu", "sacrebleu", "rouge"] and not tokenizer:
                    logger.warning(
                        f"Metric {name} requires tokenizer for text decoding. Pass tokenizer to get_compute_metrics_func()"
                    )
                results[name] = -999  # Error value to indicate computation failed

        return results

    return compute_metrics


def get_sft_metrics(metric_names: Optional[List[str]] = None, tokenizer: Optional[Any] = None) -> Optional[Callable]:
    """
    Get compute_metrics function for SFT (supervised fine-tuning) tasks.

    Args:
        metric_names: List of metrics to compute. Defaults to None (no custom metrics)
        tokenizer: Tokenizer for decoding predictions

    Returns:
        compute_metrics function or None

    Note:
        During training, SFT outputs logits, not generated text. Text generation metrics
        like BLEU/ROUGE are better suited for evaluation with actual generation, not training.
        Consider using perplexity or accuracy-based metrics for training evaluation.
    """
    if metric_names is None:
        metric_names = []  # SFT typically uses loss/perplexity, additional metrics optional

    if not metric_names:
        return None

    # Warn if text generation metrics are requested
    gen_metrics = [m for m in metric_names if m in ["bleu", "sacrebleu", "rouge", "meteor"]]
    if gen_metrics:
        logger.info(
            f"Note: Text generation metrics {gen_metrics} require actual text generation. "
            f"During training, only logits are available. Consider using these metrics "
            f"with a separate generation evaluation step."
        )

    return get_compute_metrics_func(metric_names, tokenizer=tokenizer)


def get_rl_metrics(metric_names: Optional[List[str]] = None) -> Optional[Callable]:
    """
    Get compute_metrics function for RL tasks.

    Args:
        metric_names: List of metrics to compute

    Returns:
        compute_metrics function or None
    """
    if metric_names is None:
        metric_names = []  # RL typically uses rewards, additional metrics optional

    if not metric_names:
        return None

    return get_compute_metrics_func(metric_names)
