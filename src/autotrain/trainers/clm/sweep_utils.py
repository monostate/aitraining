"""
Shared sweep utilities for CLM trainers
"""

import json
from typing import Any, Callable

from autotrain import logger
from autotrain.trainers.clm.params import LLMTrainingParams


def run_with_sweep(config: LLMTrainingParams, train_func: Callable) -> Any:
    """
    Run hyperparameter sweep for any CLM trainer.

    Args:
        config: Training configuration with sweep parameters
        train_func: The trainer-specific train function to call

    Returns:
        The final trained model with best hyperparameters
    """
    from autotrain.utils import run_autotrain_sweep

    logger.info(f"Running hyperparameter sweep with {config.sweep_backend} backend")

    # Parse sweep parameters from JSON
    if config.sweep_params:
        sweep_params_dict = json.loads(config.sweep_params)
    else:
        # Default sweep parameters
        sweep_params_dict = {
            "lr": (1e-5, 1e-3, "log_uniform"),
            "batch_size": [2, 4, 8, 16],
            "warmup_ratio": (0.0, 0.2, "float"),
        }

    # Create train function for sweep
    def train_for_sweep(params):
        # Create new config with sweep params
        trial_config = LLMTrainingParams(**{**vars(config), **params})
        trial_config.use_sweep = False  # Prevent recursive sweep

        # Run training
        trainer = train_func(trial_config)

        # Extract metric from trainer
        if hasattr(trainer, "state") and hasattr(trainer.state, "log_history") and trainer.state.log_history:
            metric_name = config.sweep_metric or "eval_loss"
            for log in reversed(trainer.state.log_history):
                if metric_name in log:
                    return log[metric_name]

        # Return worst possible value if metric not found
        return float("inf") if config.sweep_direction == "minimize" else float("-inf")

    # Run sweep using existing implementation
    result = run_autotrain_sweep(
        model_config=vars(config),
        sweep_parameters=sweep_params_dict,
        train_function=train_for_sweep,
        metric=config.sweep_metric or "eval_loss",
        direction=config.sweep_direction or "minimize",
        n_trials=config.sweep_n_trials,
        backend=config.sweep_backend or "optuna",
        output_dir=f"{config.project_name}/sweep",
    )

    logger.info(f"Sweep completed! Best params: {result.best_params}")
    logger.info(f"Best {config.sweep_metric}: {result.best_value}")

    # Train final model with best parameters
    logger.info("Training final model with best parameters...")
    final_config = LLMTrainingParams(**{**vars(config), **result.best_params})
    final_config.use_sweep = False

    return train_func(final_config)


def with_sweep(train_func: Callable) -> Callable:
    """
    Decorator to add sweep functionality to any trainer.

    Usage:
        @with_sweep
        def train(config):
            # your training logic
            return trainer
    """

    def wrapper(config):
        if hasattr(config, "use_sweep") and config.use_sweep:
            return run_with_sweep(config, train_func)
        return train_func(config)

    return wrapper
