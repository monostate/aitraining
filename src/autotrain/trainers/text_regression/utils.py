import os

import numpy as np
from sklearn import metrics

from autotrain.trainers.common_metrics import get_compute_metrics_func


SINGLE_COLUMN_REGRESSION_EVAL_METRICS = (
    "eval_loss",
    "eval_mse",
    "eval_mae",
    "eval_r2",
    "eval_rmse",
    "eval_explained_variance",
)


MODEL_CARD = """
---
tags:
- autotrain
- text-regression{base_model}
widget:
- text: "I love AutoTrain"{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Text Regression

## Validation Metrics
{validation_metrics}
"""


def single_column_regression_metrics(pred):
    """
    Computes various regression metrics for a single column of predictions.

    Args:
        pred (tuple): A tuple containing raw predictions and true labels.
                      The first element is an array-like of raw predictions,
                      and the second element is an array-like of true labels.

    Returns:
        dict: A dictionary containing the computed regression metrics:
            - "mse": Mean Squared Error
            - "mae": Mean Absolute Error
            - "r2": R-squared Score
            - "rmse": Root Mean Squared Error
            - "explained_variance": Explained Variance Score

    Notes:
        If any metric computation fails, the function will return a default value of -999 for that metric.
    """
    raw_predictions, labels = pred

    def safe_compute(metric_func, default=-999):
        try:
            return metric_func(labels, raw_predictions)
        except Exception:
            return default

    pred_dict = {
        "mse": safe_compute(lambda labels, predictions: metrics.mean_squared_error(labels, predictions)),
        "mae": safe_compute(lambda labels, predictions: metrics.mean_absolute_error(labels, predictions)),
        "r2": safe_compute(lambda labels, predictions: metrics.r2_score(labels, predictions)),
        "rmse": safe_compute(lambda labels, predictions: np.sqrt(metrics.mean_squared_error(labels, predictions))),
        "explained_variance": safe_compute(
            lambda labels, predictions: metrics.explained_variance_score(labels, predictions)
        ),
    }

    for key, value in pred_dict.items():
        pred_dict[key] = float(value)
    return pred_dict


def get_text_regression_metrics(custom_metrics=None):
    """
    Get a compute_metrics function that combines standard and custom metrics.

    Args:
        custom_metrics (list): Optional list of custom metric names to add

    Returns:
        Callable: A compute_metrics function for use with Trainer
    """
    # If no custom metrics, just return the standard function
    if not custom_metrics:
        return single_column_regression_metrics

    # Create a combined function
    def combined_metrics(pred):
        # Get standard metrics
        results = single_column_regression_metrics(pred)

        # Add custom metrics if specified
        if custom_metrics:
            custom_fn = get_compute_metrics_func(custom_metrics, is_regression=True)
            if custom_fn:
                custom_results = custom_fn(pred)
                # Merge results, custom metrics override if there's a conflict
                results.update(custom_results)

        return results

    return combined_metrics


def create_model_card(config, trainer):
    """
    Generates a model card string based on the provided configuration and trainer.

    Args:
        config (object): Configuration object containing the following attributes:
            - valid_split (optional): Validation split to evaluate the model.
            - data_path (str): Path to the dataset.
            - project_name (str): Name of the project.
            - model (str): Path or identifier of the model.
        trainer (object): Trainer object used to evaluate the model.

    Returns:
        str: A formatted model card string containing dataset information, validation metrics, and base model details.
    """
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
        eval_scores = [
            f"{k[len('eval_'):]}: {v}" for k, v in eval_scores.items() if k in SINGLE_COLUMN_REGRESSION_EVAL_METRICS
        ]
        eval_scores = "\n\n".join(eval_scores)

    else:
        eval_scores = "No validation metrics available"

    if config.data_path == f"{config.project_name}/autotrain-data" or os.path.isdir(config.data_path):
        dataset_tag = ""
    else:
        dataset_tag = f"\ndatasets:\n- {config.data_path}"

    if os.path.isdir(config.model):
        base_model = ""
    else:
        base_model = f"\nbase_model: {config.model}"

    model_card = MODEL_CARD.format(
        dataset_tag=dataset_tag,
        validation_metrics=eval_scores,
        base_model=base_model,
    )
    return model_card
