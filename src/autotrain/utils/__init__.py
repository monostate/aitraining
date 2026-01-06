"""
AutoTrain utils package (consolidated)
======================================

This package provides consolidated utilities that were previously spread across
`autotrain.utils` module and `autotrain.utils.sweep`. It also serves as a
compatibility layer for legacy imports under `autotrain.utils.sweep`.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

# Device and model-loading utilities (centralized)
import torch

from autotrain import logger


def get_model_loading_kwargs(
    token: Optional[str] = None,
    fp16_if_cuda: bool = True,
    trust_remote_code: bool = True,
    extra_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Build consistent kwargs for AutoModel.from_pretrained across codepaths.

    - Uses device_map="auto" on CUDA
    - Prefers float16 on CUDA when fp16_if_cuda=True
    - Uses float32 on MPS and CPU
    - Adds token and trust_remote_code if provided
    """
    kwargs: Dict = {}
    if token is not None:
        kwargs["token"] = token
    if trust_remote_code is not None:
        kwargs["trust_remote_code"] = trust_remote_code

    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16 if fp16_if_cuda else torch.float32
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # MPS prefers float32; placement handled after load
        kwargs["torch_dtype"] = torch.float32
    else:
        kwargs["torch_dtype"] = torch.float32

    if extra_kwargs:
        kwargs.update(extra_kwargs)

    return kwargs


def maybe_move_to_mps(model, model_kwargs: Dict):
    """
    If MPS is available and no device_map is set (i.e., CPU placement), move to MPS.
    Returns the (possibly moved) model.
    """
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        and "device_map" not in model_kwargs
    ):
        return model.to("mps")
    return model


# Sweep functionality (consolidated)
class SweepBackend(Enum):
    """Available hyperparameter sweep backends."""

    OPTUNA = "optuna"
    RAY_TUNE = "ray_tune"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"


@dataclass
class ParameterRange:
    """Defines a range for hyperparameter sweep."""

    low: float = None
    high: float = None
    distribution: str = "uniform"  # uniform, log_uniform, int_uniform
    name: str = None
    param_type: str = None  # categorical, float, int
    choices: List[Any] = None
    step: float = None

    def sample(self, trial=None, backend=None):
        """Sample a value from this range."""
        import random

        # Handle categorical parameters
        if self.param_type == "categorical" and self.choices:
            if trial:
                return trial.suggest_categorical(self.name or "param", self.choices)
            return random.choice(self.choices)

        # Handle numeric parameters with step
        if self.step is not None:
            import numpy as np

            values = np.arange(self.low, self.high + self.step, self.step)
            if trial:
                return trial.suggest_categorical(self.name or "param", values.tolist())
            return random.choice(values)

        # Handle regular distributions
        if self.distribution == "uniform" or self.param_type == "float":
            if trial:
                return trial.suggest_float(self.name or "param", self.low, self.high)
            return random.uniform(self.low, self.high)
        elif self.distribution == "log_uniform":
            if trial:
                return trial.suggest_float(self.name or "param", self.low, self.high, log=True)
            import math

            return math.exp(random.uniform(math.log(self.low), math.log(self.high)))
        elif self.distribution == "int_uniform" or self.param_type == "int":
            if trial:
                return trial.suggest_int(self.name or "param", int(self.low), int(self.high))
            return random.randint(int(self.low), int(self.high))


@dataclass
class SweepConfig:
    """Configuration for hyperparameter sweep."""

    backend: SweepBackend = SweepBackend.OPTUNA
    n_trials: int = 10
    direction: str = "minimize"
    parameters: Dict[str, Union[ParameterRange, List]] = field(default_factory=dict)
    metric: str = "eval_loss"
    timeout: Optional[int] = None
    output_dir: Optional[str] = None
    patience: Optional[int] = None  # For early stopping
    min_delta: Optional[float] = None  # For early stopping
    sampler: Optional[str] = None  # For Optuna
    n_jobs: Optional[int] = None  # For parallel execution
    # W&B native sweep integration
    wandb_sweep: bool = False  # Enable W&B native sweep dashboard
    wandb_project: Optional[str] = None  # W&B project name for sweep
    wandb_entity: Optional[str] = None  # W&B entity (team/username)
    wandb_sweep_id: Optional[str] = None  # Existing sweep ID to continue


class SweepResult:
    """Results from hyperparameter sweep."""

    def __init__(self, config=None, best_params=None, best_value=None, trials=None, backend=None, study=None):
        """Initialize SweepResult with flexible parameters."""
        if config is not None and best_params is None:
            # Initialize from config only (for compatibility with tests)
            self.config = config
            self.best_params = {}
            self.best_value = None
            self.trials = []
            backend_value = getattr(config, "backend", "unknown") if hasattr(config, "backend") else "unknown"
            # Handle enum values
            if hasattr(backend_value, "value"):
                self.backend = backend_value.value
            else:
                self.backend = backend_value
            self.direction = getattr(config, "direction", "minimize")
            self.study = None
            self.best_trial_id = None
        else:
            # Initialize with explicit parameters
            self.config = config
            self.best_params = best_params or {}
            self.best_value = best_value
            self.trials = trials or []
            self.backend = backend or "unknown"
            self.direction = getattr(config, "direction", "minimize") if config else "minimize"
            self.study = study
            self.best_trial_id = None

    def add_trial(self, trial_id, params, value):
        """Add a trial to the results."""
        trial = {"id": trial_id, "params": params, "value": value}
        self.trials.append(trial)

        # Update best values
        if self.best_value is None:
            self.best_value = value
            self.best_params = params
            self.best_trial_id = trial_id
        else:
            is_better = (value < self.best_value) if self.direction == "minimize" else (value > self.best_value)
            if is_better:
                self.best_value = value
                self.best_params = params
                self.best_trial_id = trial_id

    def to_dataframe(self):
        """Convert trials to pandas DataFrame."""
        import pandas as pd

        if not self.trials:
            return pd.DataFrame()

        # Flatten trial data
        rows = []
        for i, trial in enumerate(self.trials):
            if "id" in trial:
                row = {"trial_id": trial["id"], "value": trial["value"]}
            else:
                row = {"trial_id": i, "value": trial.get("value")}
            if "params" in trial:
                row.update(trial["params"])
            rows.append(row)

        return pd.DataFrame(rows)

    def save(self, path):
        """Save results to JSON and CSV."""
        import json
        from pathlib import Path

        path = Path(path)

        # Save JSON
        data = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "trials": self.trials,
            "backend": self.backend,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        # Save CSV
        csv_path = path.parent / path.name.replace(".json", ".csv")
        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(csv_path, index=False)

    def plot_optimization_history(self, path=None):
        """Plot optimization history (placeholder for tests)."""
        # Placeholder implementation for tests
        # Real implementation would create matplotlib plot

    def plot_parallel_coordinates(self, path=None):
        """Plot parallel coordinates (placeholder for tests)."""
        # Placeholder implementation for tests
        # Real implementation would use plotly


class HyperparameterSweep:
    """Manager for hyperparameter sweeps."""

    def __init__(self, config: SweepConfig, train_function: Callable = None):
        self.config = config
        self.train_function = train_function
        self.results = []
        self.best_value_history = []  # Track best value over time for early stopping

    def run(self, train_function: Callable = None) -> SweepResult:
        """Run the hyperparameter sweep."""
        # Use provided train_function or the one from init
        train_fn = train_function or self.train_function
        if not train_fn:
            raise ValueError("No train_function provided")

        if self.config.backend == SweepBackend.OPTUNA:
            result = self._run_optuna(train_fn)
        elif self.config.backend == SweepBackend.RANDOM_SEARCH:
            result = self._run_random_search(train_fn)
        elif self.config.backend == SweepBackend.GRID_SEARCH:
            result = self._run_grid_search(train_fn)
        else:
            raise NotImplementedError(f"Backend {self.config.backend} not implemented")

        # Save results if output_dir is specified
        if self.config.output_dir:
            import os

            os.makedirs(self.config.output_dir, exist_ok=True)
            result_path = os.path.join(self.config.output_dir, "sweep_results.json")
            result.save(result_path)

        return result

    def _should_stop_early(self, current_best: float) -> bool:
        """Check if we should stop early based on patience and min_delta."""
        if not self.config.patience:
            return False

        self.best_value_history.append(current_best)

        if len(self.best_value_history) <= self.config.patience:
            return False

        # Get the best value from patience trials ago
        old_best = self.best_value_history[-(self.config.patience + 1)]

        # Calculate improvement
        if self.config.direction == "minimize":
            improvement = old_best - current_best
        else:
            improvement = current_best - old_best

        # Check if improvement is less than min_delta
        min_delta = self.config.min_delta or 0
        if improvement < min_delta:
            from autotrain import logger

            logger.info(f"Early stopping: No improvement of at least {min_delta} for {self.config.patience} trials")
            return True

        return False

    def _create_wandb_sweep(self) -> Optional[str]:
        """Create a W&B sweep and return the sweep_id, or use existing one."""
        if not self.config.wandb_sweep:
            return None

        # If existing sweep_id provided, use it instead of creating new
        if self.config.wandb_sweep_id:
            logger.info(f"Continuing existing W&B sweep: {self.config.wandb_sweep_id}")
            return self.config.wandb_sweep_id

        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            return None

        # Build W&B sweep config from our parameters
        wandb_params = {}
        if isinstance(self.config.parameters, list):
            for spec in self.config.parameters:
                if spec.choices:
                    wandb_params[spec.name] = {"values": spec.choices}
                elif spec.distribution == "log_uniform":
                    wandb_params[spec.name] = {
                        "distribution": "log_uniform_values",
                        "min": spec.low,
                        "max": spec.high,
                    }
                elif spec.distribution == "int_uniform":
                    wandb_params[spec.name] = {
                        "distribution": "int_uniform",
                        "min": int(spec.low),
                        "max": int(spec.high),
                    }
                else:
                    wandb_params[spec.name] = {
                        "distribution": "uniform",
                        "min": spec.low,
                        "max": spec.high,
                    }
        else:
            for name, spec in self.config.parameters.items():
                if isinstance(spec, list):
                    wandb_params[name] = {"values": spec}
                elif isinstance(spec, ParameterRange):
                    if spec.distribution == "log_uniform":
                        wandb_params[name] = {
                            "distribution": "log_uniform_values",
                            "min": spec.low,
                            "max": spec.high,
                        }
                    elif spec.distribution == "int_uniform":
                        wandb_params[name] = {
                            "distribution": "int_uniform",
                            "min": int(spec.low),
                            "max": int(spec.high),
                        }
                    else:
                        wandb_params[name] = {
                            "distribution": "uniform",
                            "min": spec.low,
                            "max": spec.high,
                        }
                elif isinstance(spec, tuple) and len(spec) == 3:
                    low, high, dist = spec
                    if dist == "log_uniform":
                        wandb_params[name] = {
                            "distribution": "log_uniform_values",
                            "min": low,
                            "max": high,
                        }
                    else:
                        wandb_params[name] = {
                            "distribution": "uniform",
                            "min": low,
                            "max": high,
                        }

        # Map our backend to W&B method
        method_map = {
            SweepBackend.OPTUNA: "bayes",
            SweepBackend.RANDOM_SEARCH: "random",
            SweepBackend.GRID_SEARCH: "grid",
        }
        wandb_method = method_map.get(self.config.backend, "bayes")

        sweep_config = {
            "method": wandb_method,
            "metric": {
                "name": self.config.metric,
                "goal": self.config.direction,
            },
            "parameters": wandb_params,
            "run_cap": self.config.n_trials,
        }

        project = self.config.wandb_project or "aitraining-sweep"
        entity = self.config.wandb_entity

        try:
            sweep_id = wandb.sweep(
                sweep=sweep_config,
                project=project,
                entity=entity,
            )
            logger.info(f"Created W&B sweep: {sweep_id} in project '{project}'")
            logger.info(
                f"View sweep dashboard at: https://wandb.ai/{entity or 'your-entity'}/{project}/sweeps/{sweep_id}"
            )
            return sweep_id
        except Exception as e:
            logger.warning(f"Failed to create W&B sweep: {e}")
            return None

    def _run_optuna(self, train_function):
        """Run sweep using Optuna."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter sweeps. Install with: pip install optuna")

        # Create W&B sweep if enabled
        wandb_sweep_id = self._create_wandb_sweep()
        wandb_project = self.config.wandb_project or "aitraining-sweep"
        wandb_entity = self.config.wandb_entity

        def objective(trial):
            params = {}

            # Handle both list and dict parameter formats
            if isinstance(self.config.parameters, list):
                # List of ParameterRange objects
                for spec in self.config.parameters:
                    if spec.choices:
                        params[spec.name] = trial.suggest_categorical(spec.name, spec.choices)
                    elif spec.param_type == "int" or spec.distribution == "int_uniform":
                        params[spec.name] = trial.suggest_int(spec.name, int(spec.low), int(spec.high))
                    elif spec.distribution == "log_uniform":
                        params[spec.name] = trial.suggest_float(spec.name, spec.low, spec.high, log=True)
                    else:
                        params[spec.name] = trial.suggest_float(spec.name, spec.low, spec.high)
            else:
                # Dict format
                for name, spec in self.config.parameters.items():
                    if isinstance(spec, ParameterRange):
                        if spec.distribution == "uniform":
                            params[name] = trial.suggest_float(name, spec.low, spec.high)
                        elif spec.distribution == "log_uniform":
                            params[name] = trial.suggest_float(name, spec.low, spec.high, log=True)
                        elif spec.distribution == "int_uniform":
                            params[name] = trial.suggest_int(name, int(spec.low), int(spec.high))
                    elif isinstance(spec, list):
                        params[name] = trial.suggest_categorical(name, spec)
                    elif isinstance(spec, tuple) and len(spec) == 3:
                        low, high, dist = spec
                        if dist == "log_uniform":
                            params[name] = trial.suggest_float(name, low, high, log=True)
                        elif dist == "float":
                            params[name] = trial.suggest_float(name, low, high)
                        else:
                            params[name] = trial.suggest_float(name, low, high)

            # Initialize W&B run linked to sweep if enabled
            if wandb_sweep_id:
                try:
                    import wandb

                    # Set WANDB_PROJECT/ENTITY env vars so trainer's internal wandb.init() uses correct project
                    # This prevents the trainer from logging to 'huggingface' default project
                    os.environ["WANDB_PROJECT"] = wandb_project
                    if wandb_entity:
                        os.environ["WANDB_ENTITY"] = wandb_entity

                    # Use group to link runs to the sweep for aggregated views
                    wandb.init(
                        project=wandb_project,
                        entity=wandb_entity,
                        group=wandb_sweep_id,
                        name=f"trial-{trial.number}",
                        config=params,
                        reinit=True,
                    )

                    # Set env vars so trainer's internal wandb.init() resumes this run
                    # instead of creating a duplicate
                    if wandb.run is not None:
                        os.environ["WANDB_RUN_ID"] = wandb.run.id
                        os.environ["WANDB_RESUME"] = "allow"
                except Exception as e:
                    logger.warning(f"Failed to init W&B run for trial {trial.number}: {e}")

            try:
                result = train_function(params)
            finally:
                # Finish W&B run if we started one
                if wandb_sweep_id:
                    try:
                        import wandb

                        if wandb.run is not None:
                            wandb.finish()
                    except Exception:
                        pass

            return result

        study = optuna.create_study(direction=self.config.direction)
        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)

        sweep_result = SweepResult(
            best_params=study.best_params,
            best_value=study.best_value,
            trials=[{"params": t.params, "value": t.value} for t in study.trials],
            backend="optuna",
            study=study,
        )

        # Add W&B sweep info to result
        if wandb_sweep_id:
            sweep_result.wandb_sweep_id = wandb_sweep_id
            sweep_result.wandb_project = wandb_project
            logger.info(
                f"W&B sweep completed. View results at: https://wandb.ai/{wandb_entity or 'your-entity'}/{wandb_project}/sweeps/{wandb_sweep_id}"
            )

        return sweep_result

    def _run_random_search(self, train_function):
        """Run random search."""
        import random

        trials = []
        best_value = float("inf") if self.config.direction == "minimize" else float("-inf")
        best_params = None

        for _ in range(self.config.n_trials):
            params = {}

            # Handle both list and dict parameter formats
            if isinstance(self.config.parameters, list):
                # List of ParameterRange objects
                for spec in self.config.parameters:
                    params[spec.name] = spec.sample()
            else:
                # Dict format
                for name, spec in self.config.parameters.items():
                    if isinstance(spec, ParameterRange):
                        if spec.distribution == "int_uniform":
                            params[name] = random.randint(int(spec.low), int(spec.high))
                        elif spec.distribution == "log_uniform":
                            import math

                            params[name] = math.exp(random.uniform(math.log(spec.low), math.log(spec.high)))
                        else:
                            params[name] = random.uniform(spec.low, spec.high)
                    elif isinstance(spec, list):
                        params[name] = random.choice(spec)

            try:
                value = train_function(params)
                trials.append({"params": params, "value": value})

                if self.config.direction == "minimize":
                    if value < best_value:
                        best_value = value
                        best_params = params
                else:
                    if value > best_value:
                        best_value = value
                        best_params = params

                # Check for early stopping
                if self._should_stop_early(best_value):
                    break
            except Exception as e:
                # Log the error but continue with other trials
                from autotrain import logger

                logger.error(f"Trial failed with params {params}: {str(e)}")
                # Optionally add failed trial with None value
                trials.append({"params": params, "value": None, "error": str(e)})

        return SweepResult(best_params=best_params, best_value=best_value, trials=trials, backend="random_search")

    def _run_grid_search(self, train_function):
        """Run grid search."""
        import itertools

        # Create parameter grid
        param_lists = []
        param_names = []

        # Handle both list and dict parameter formats
        if isinstance(self.config.parameters, list):
            # List of ParameterRange objects
            for spec in self.config.parameters:
                param_names.append(spec.name)
                if spec.choices:
                    param_lists.append(spec.choices)
                elif spec.param_type == "int" or spec.distribution == "int_uniform":
                    values = list(range(int(spec.low), int(spec.high) + 1))
                    param_lists.append(values)
                else:
                    # Sample points for continuous parameters
                    values = [spec.low + (spec.high - spec.low) * i / 4 for i in range(5)]
                    param_lists.append(values)
        else:
            # Dict format
            for name, spec in self.config.parameters.items():
                param_names.append(name)
                if isinstance(spec, list):
                    param_lists.append(spec)
                elif isinstance(spec, ParameterRange):
                    # Convert range to discrete values for grid search
                    if spec.distribution == "int_uniform":
                        values = list(range(int(spec.low), int(spec.high) + 1))
                    else:
                        # Sample points for continuous parameters
                        values = [spec.low + (spec.high - spec.low) * i / 4 for i in range(5)]
                    param_lists.append(values)

        trials = []
        best_value = float("inf") if self.config.direction == "minimize" else float("-inf")
        best_params = None

        for param_values in itertools.product(*param_lists):
            params = dict(zip(param_names, param_values))
            try:
                value = train_function(params)
                trials.append({"params": params, "value": value})

                if self.config.direction == "minimize":
                    if value < best_value:
                        best_value = value
                        best_params = params
                else:
                    if value > best_value:
                        best_value = value
                        best_params = params

                # Check for early stopping
                if self._should_stop_early(best_value):
                    break
            except Exception as e:
                # Log the error but continue with other trials
                from autotrain import logger

                logger.error(f"Trial failed with params {params}: {str(e)}")
                # Optionally add failed trial with None value
                trials.append({"params": params, "value": None, "error": str(e)})

        return SweepResult(best_params=best_params, best_value=best_value, trials=trials, backend="grid_search")


def run_autotrain_sweep(
    model_config: Dict,
    sweep_parameters: Dict,
    train_function: Callable,
    metric: str = "eval_loss",
    direction: str = "minimize",
    n_trials: int = 10,
    backend: str = "optuna",
    output_dir: Optional[str] = None,
    wandb_sweep: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_sweep_id: Optional[str] = None,
) -> SweepResult:
    """
    Convenience function to run hyperparameter sweep for AutoTrain.

    Args:
        model_config: Base configuration dict
        sweep_parameters: Parameters to sweep with their ranges
        train_function: Function that takes params and returns metric value
        metric: Metric to optimize
        direction: "minimize" or "maximize"
        n_trials: Number of trials to run
        backend: Sweep backend to use
        output_dir: Directory to save results

    Returns:
        SweepResult with best parameters and trial history
    """
    # Convert sweep parameters to ParameterRange objects
    processed_params = {}
    for name, spec in sweep_parameters.items():
        if isinstance(spec, tuple) and len(spec) == 3:
            low, high, dist = spec
            processed_params[name] = ParameterRange(low, high, dist)
        elif isinstance(spec, list):
            processed_params[name] = spec
        elif isinstance(spec, dict) and "type" in spec:
            # Handle dict format: {"type": "categorical", "values": [...]}
            # or {"type": "loguniform", "low": x, "high": y}
            param_type = spec.get("type", "").lower()
            if param_type == "categorical":
                # Categorical: extract values list
                processed_params[name] = spec.get("values", [])
            elif param_type in ("loguniform", "log_uniform"):
                # Log-uniform: convert to ParameterRange
                processed_params[name] = ParameterRange(spec.get("low", 1e-5), spec.get("high", 1e-3), "log_uniform")
            elif param_type == "uniform":
                # Uniform: convert to ParameterRange
                processed_params[name] = ParameterRange(spec.get("low", 0.0), spec.get("high", 1.0), "uniform")
            elif param_type in ("int", "int_uniform"):
                # Integer uniform: convert to ParameterRange
                processed_params[name] = ParameterRange(spec.get("low", 1), spec.get("high", 10), "int_uniform")
            else:
                # Unknown type, pass through
                processed_params[name] = spec
        else:
            processed_params[name] = spec

    # Handle backend aliases for backward compatibility
    backend_aliases = {
        "random": "random_search",
        "grid": "grid_search",
        "ray": "ray_tune",
    }
    backend_normalized = backend.lower()
    backend_normalized = backend_aliases.get(backend_normalized, backend_normalized)

    config = SweepConfig(
        backend=SweepBackend(backend_normalized),
        n_trials=n_trials,
        direction=direction,
        parameters=processed_params,
        metric=metric,
        wandb_sweep=wandb_sweep,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_sweep_id=wandb_sweep_id,
    )

    sweep = HyperparameterSweep(config)
    result = sweep.run(train_function)

    # Save results if output_dir provided
    if output_dir:
        import json
        import os

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "sweep_results.json"), "w") as f:
            json.dump(
                {
                    "best_params": result.best_params,
                    "best_value": result.best_value,
                    "trials": result.trials,
                },
                f,
                indent=2,
            )

    return result


__all__ = [
    "get_model_loading_kwargs",
    "maybe_move_to_mps",
    "SweepBackend",
    "ParameterRange",
    "SweepConfig",
    "SweepResult",
    "HyperparameterSweep",
    "run_autotrain_sweep",
]


# Back-compat: run_training utility (used by CLI/backends/endpoints)
import json
import os
import subprocess
import sys
import threading
from typing import Optional


def _terminate_process(proc: Optional[subprocess.Popen]) -> None:
    """Best-effort termination for spawned subprocesses."""
    if not proc:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_training(params, task_id, local: bool = False, wait: bool = False) -> int:
    """
    Launch a training subprocess based on the provided parameters and task ID.

    Mirrors the original implementation from `autotrain.utils` (module form),
    kept here for compatibility after consolidation to a package.
    """
    # Lazy imports to avoid heavy module loading at package import time
    from autotrain.commands import launch_command
    from autotrain.trainers.clm.params import LLMTrainingParams
    from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
    from autotrain.trainers.generic.params import GenericParams
    from autotrain.trainers.image_classification.params import ImageClassificationParams
    from autotrain.trainers.image_regression.params import ImageRegressionParams
    from autotrain.trainers.object_detection.params import ObjectDetectionParams
    from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
    from autotrain.trainers.seq2seq.params import Seq2SeqParams
    from autotrain.trainers.tabular.params import TabularParams
    from autotrain.trainers.text_classification.params import TextClassificationParams
    from autotrain.trainers.text_regression.params import TextRegressionParams
    from autotrain.trainers.token_classification.params import TokenClassificationParams
    from autotrain.trainers.vlm.params import VLMTrainingParams

    # Parse params
    params = json.loads(params)
    if isinstance(params, str):
        params = json.loads(params)
    if task_id == 9:
        params = LLMTrainingParams(**params)
    elif task_id == 28:
        params = Seq2SeqParams(**params)
    elif task_id in (1, 2):
        params = TextClassificationParams(**params)
    elif task_id in (13, 14, 15, 16, 26):
        params = TabularParams(**params)
    elif task_id == 27:
        params = GenericParams(**params)
    elif task_id == 18:
        params = ImageClassificationParams(**params)
    elif task_id == 4:
        params = TokenClassificationParams(**params)
    elif task_id == 10:
        params = TextRegressionParams(**params)
    elif task_id == 29:
        params = ObjectDetectionParams(**params)
    elif task_id == 30:
        params = SentenceTransformersParams(**params)
    elif task_id == 24:
        params = ImageRegressionParams(**params)
    elif task_id == 31:
        params = VLMTrainingParams(**params)
    elif task_id == 5:
        params = ExtractiveQuestionAnsweringParams(**params)
    else:
        raise NotImplementedError

    params.save(output_dir=params.project_name)

    # Resolve absolute project directory for consistent logging + WANDB_DIR
    project_run_dir = os.path.abspath(params.project_name)

    # Ensure project log exists BEFORE building command to capture early hangs
    os.makedirs(project_run_dir, exist_ok=True)
    log_path = os.path.join(project_run_dir, "autotrain.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== Preparing training at {os.getcwd()} ===\n")
        f.write(f"Python path: {sys.executable}\n")
        f.write("Will compute launch command next...\n")
        f.flush()

    # Set GPU count override BEFORE building command to avoid torch/CUDA init in parent
    # But also check for MPS and quantization compatibility issues
    if "AUTOTRAIN_FORCE_NUM_GPUS" not in os.environ:
        import torch

        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()

        if cuda_available:
            forced = (
                "1"
                if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1")
                else str(torch.cuda.device_count())
            )
        elif mps_available:
            # Check if we should disable MPS due to compatibility issues
            should_disable_mps = False
            if hasattr(params, "quantization") and params.quantization and params.quantization != "none":
                logger.warning(
                    f"Quantization ({params.quantization}) is not fully compatible with MPS. "
                    "Falling back to CPU training. Set AUTOTRAIN_ENABLE_MPS=1 to force MPS anyway."
                )
                should_disable_mps = True

            force_mps = os.environ.get("AUTOTRAIN_ENABLE_MPS", "0") == "1"
            force_cpu = os.environ.get("AUTOTRAIN_DISABLE_MPS", "0") == "1"

            if force_cpu or (should_disable_mps and not force_mps):
                # Disable MPS completely by setting env var that PyTorch checks
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                os.environ["AUTOTRAIN_DISABLE_MPS"] = "1"
                # This is the key one - it actually disables MPS in PyTorch
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
                forced = "0"
            else:
                forced = "1"
        else:
            forced = "0"

        os.environ["AUTOTRAIN_FORCE_NUM_GPUS"] = forced

    # Build command - this may set additional environment variables (e.g., for MPS handling)
    cmd = [str(c) for c in launch_command(params=params)]

    # Copy environment AFTER launch_command so we capture any env vars it sets
    env = os.environ.copy()
    # Ensure all W&B write paths land under the project run directory
    env["WANDB_DIR"] = project_run_dir
    env["WANDB_CACHE_DIR"] = project_run_dir
    env["WANDB_DATA_DIR"] = project_run_dir

    # Set W&B API key if provided
    if hasattr(params, "wandb_token") and params.wandb_token:
        env["WANDB_API_KEY"] = params.wandb_token

    # Track wandb command hint if enabled
    wandb_logging_enabled = getattr(params, "log", "none") == "wandb"
    wandb_hint_command = (
        f'WANDB_DIR="{project_run_dir}" wandb beta leet "{project_run_dir}"' if wandb_logging_enabled else None
    )

    # Check if we're in an interactive terminal BEFORE opening log file (which might redirect stdout)
    # Use stdin.isatty() as a more reliable check since stdout might be redirected
    is_interactive_terminal = sys.stdin.isatty() or (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())

    # Log the environment for debugging
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"PATH: {env.get('PATH', 'Not set')}\n")
        f.write(f"PYTHONPATH: {env.get('PYTHONPATH', 'Not set')}\n")
        f.write(f"AUTOTRAIN_DISABLE_MPS: {env.get('AUTOTRAIN_DISABLE_MPS', 'Not set')}\n")
        if wandb_logging_enabled and wandb_hint_command:
            f.write(
                "[W&B] Offline metrics stored under this run directory. Reopen LEET anytime with:\n"
                f"      {wandb_hint_command}\n"
            )
        f.flush()
    if wandb_logging_enabled and wandb_hint_command:
        logger.info(f"[W&B] Visualizer command: {wandb_hint_command}")

    # Try to locate accelerate in typical locations if needed
    import shutil

    venv_bin = "/app/.venv/bin"
    if os.path.exists(venv_bin) and venv_bin not in env.get("PATH", ""):
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '/usr/local/bin:/usr/bin:/bin')}"
    accelerate_path = shutil.which("accelerate", path=env.get("PATH"))
    if not accelerate_path:
        for path in (
            "/app/.venv/bin/accelerate",
            "/home/bentoml/.local/bin/accelerate",
            "/usr/local/bin/accelerate",
            os.path.expanduser("~/.local/bin/accelerate"),
        ):
            if os.path.exists(path):
                accelerate_path = path
                break
    if accelerate_path and cmd and cmd[0] == "accelerate":
        cmd[0] = accelerate_path

    # Ensure unbuffered Python output
    env.setdefault("PYTHONUNBUFFERED", "1")

    # Validate training config presence
    training_config_path = os.path.join(project_run_dir, "training_params.json")
    if not os.path.isfile(training_config_path):
        raise FileNotFoundError(f"training_params.json not found at {training_config_path}")

    # Open log file for subprocess I/O
    log_fh = open(log_path, "a", encoding="utf-8")

    # Optionally force Python module launch instead of accelerate
    force_python = os.environ.get("AUTOTRAIN_FORCE_PYTHON_LAUNCH", "false").lower() in ("1", "true", "yes")
    if force_python and "-m" in cmd:
        try:
            m_index = cmd.index("-m")
            module = cmd[m_index + 1]
            module_args = cmd[m_index + 2 :]
            cmd = [sys.executable, "-m", module] + module_args
        except Exception:
            pass

    process: Optional[subprocess.Popen] = None
    leet_process: Optional[subprocess.Popen] = None
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )

        wandb_visualizer_enabled = bool(getattr(params, "wandb_visualizer", False))
        log_fh.write(
            f"[W&B] wandb_visualizer_enabled={wandb_visualizer_enabled}, is_interactive_terminal={is_interactive_terminal}, log={getattr(params, 'log', 'none')}\n"
        )
        log_fh.flush()
        # Use a lock file to prevent duplicate LEET launches
        leet_lock_file = os.path.join(project_run_dir, ".leet_launched")
        leet_already_launched = os.path.exists(leet_lock_file)

        if wandb_visualizer_enabled and is_interactive_terminal and not leet_already_launched:
            try:
                # Try to launch LEET in a new terminal window (macOS/Linux)
                # This allows LEET to display its TUI properly
                import platform

                leet_cmd = [sys.executable, "-m", "wandb", "beta", "leet", project_run_dir]

                if wandb_hint_command:
                    log_fh.write(f"[W&B] Launching LEET with: {wandb_hint_command}\n")
                    log_fh.flush()

                # Try to open LEET in a new terminal window (cross-platform)
                # LEET will stay open after training finishes to show metrics
                system = platform.system()

                # Create a wrapper script that waits for wandb .wandb file before launching LEET
                wait_script_content = f"""#!/bin/bash
cd {os.getcwd()}
export WANDB_DIR="{project_run_dir}"
echo "Waiting for W&B training to start..."
# Wait for wandb .wandb file to appear (wandb.init() creates it inside run-* directories)
# LEET needs the .wandb file, not just the directory structure
max_wait=60
wait_count=0
wandb_run_dir=""
while [ $wait_count -lt $max_wait ]; do
    # Check for .wandb file in any run-* directory
    wandb_run_dir=$(find "{project_run_dir}/wandb" -name "run-*.wandb" -type f 2>/dev/null | head -1)
    if [ -n "$wandb_run_dir" ]; then
        # Extract the directory containing the .wandb file
        wandb_run_dir=$(dirname "$wandb_run_dir")
        echo "W&B run detected in: $wandb_run_dir"
        echo "Launching LEET panel..."
        break
    fi
    sleep 1
    wait_count=$((wait_count + 1))
done
# Launch LEET with the run directory (or project root if run not found yet)
# LEET will display metrics and stay open even after training finishes
if [ -n "$wandb_run_dir" ]; then
    {sys.executable} -m wandb beta leet "$wandb_run_dir"
else
    # Fallback: try project root (LEET should find latest run)
    echo "Warning: Run directory not found, trying project root..."
    {sys.executable} -m wandb beta leet "{project_run_dir}"
fi
# Keep terminal open after LEET exits (so user can see metrics/history)
echo ""
echo "LEET panel closed. Press Enter to close this window..."
read
"""

                try:
                    import shutil
                    import tempfile

                    if system == "Darwin":
                        # macOS: Use osascript to open Terminal.app
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                            f.write(wait_script_content)
                            temp_script = f.name
                        os.chmod(temp_script, 0o755)

                        script = f"""
                        tell application "Terminal"
                            activate
                            do script "{temp_script}"
                        end tell
                        """
                        subprocess.run(["osascript", "-e", script], check=False, timeout=5)
                        log_fh.write("[W&B] ✓ Opening LEET in a new Terminal window (macOS)...\n")

                    elif system == "Windows":
                        # Windows: Use start command or PowerShell
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".bat", delete=False) as f:
                            # Convert bash script to Windows batch/PowerShell
                            f.write(
                                f"""@echo off
cd /d {os.getcwd()}
set WANDB_DIR={project_run_dir}
echo Waiting for W&B training to start...
timeout /t 3 /nobreak >nul
echo Launching W&B LEET panel...
{sys.executable} -m wandb beta leet "{project_run_dir}"
echo.
echo LEET panel closed. Press any key to close this window...
pause >nul
"""
                            )
                            temp_script = f.name

                        # Try to open in new window using start command
                        subprocess.Popen(
                            ["cmd", "/c", "start", "cmd", "/k", temp_script],
                            shell=False,
                            creationflags=(
                                subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, "CREATE_NEW_CONSOLE") else 0
                            ),
                        )
                        log_fh.write("[W&B] ✓ Opening LEET in a new Command Prompt window (Windows)...\n")

                    else:
                        # Linux: Try various terminal emulators
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                            f.write(wait_script_content)
                            temp_script = f.name
                        os.chmod(temp_script, 0o755)

                        # Try common Linux terminal emulators
                        terminal_launched = False
                        for term_cmd in [
                            ["gnome-terminal", "--", "bash", temp_script],
                            ["xterm", "-e", "bash", temp_script],
                            ["konsole", "-e", "bash", temp_script],
                            ["x-terminal-emulator", "-e", "bash", temp_script],
                            ["terminator", "-e", f"bash {temp_script}"],
                        ]:
                            if shutil.which(term_cmd[0]):
                                try:
                                    subprocess.Popen(term_cmd, check=False)
                                    log_fh.write(f"[W&B] ✓ Opening LEET in a new {term_cmd[0]} window (Linux)...\n")
                                    terminal_launched = True
                                    break
                                except Exception:
                                    continue

                        if not terminal_launched:
                            raise Exception("No suitable terminal emulator found")

                    # Create lock file to prevent duplicate launches
                    try:
                        with open(leet_lock_file, "w") as f:
                            f.write(str(os.getpid()))
                    except Exception:
                        pass

                    log_fh.write("[W&B] LEET will wait for training to start, then display metrics.\n")
                    log_fh.write("[W&B] LEET will stay open after training finishes to show metrics/history.\n")
                    log_fh.flush()
                    # Don't create a background process, we opened it in a new window
                    leet_process = None

                except Exception as e:
                    # Fallback: try background launch (may not display TUI)
                    log_fh.write(f"[W&B] ⚠ Could not open new terminal window: {e}\n")
                    log_fh.write("[W&B] Attempting background launch (TUI may not display)...\n")
                    leet_process = subprocess.Popen(
                        leet_cmd,
                        env=env,
                        stdout=None,  # Don't redirect - let LEET try to display
                        stderr=subprocess.DEVNULL,  # Still suppress stderr errors
                        start_new_session=True,
                    )
                    if leet_process.poll() is None:
                        log_fh.write(f"[W&B] ✓ LEET process started (PID: {leet_process.pid})\n")
                        log_fh.write("[W&B] Note: LEET TUI may not display when training output is redirected.\n")
                        log_fh.write(f"[W&B] Run manually in another terminal: {wandb_hint_command}\n")
                    else:
                        log_fh.write(
                            f"[W&B] ⚠ LEET process exited immediately (exit code: {leet_process.returncode})\n"
                        )
                        log_fh.write(
                            f"[W&B] This is normal - LEET will start once wandb.init() runs during training\n"
                        )
                        log_fh.write(f"[W&B] Run manually: {wandb_hint_command}\n")
                    log_fh.flush()
            except Exception as e:
                logger.warning(f"Failed to launch W&B visualizer: {e}")
                if wandb_hint_command:
                    log_fh.write(f"[W&B] ❌ Failed to auto-launch LEET: {e}\n")
                    log_fh.write(f"[W&B] You can still run it manually with:\n      {wandb_hint_command}\n")
                    log_fh.flush()
        elif wandb_visualizer_enabled and wandb_hint_command:
            log_fh.write("[W&B] ⚠ Visualizer requires an interactive terminal. Run manually with:\n")
            log_fh.write(f"      {wandb_hint_command}\n")
            log_fh.flush()

        # Reap in the background when wait=False
        def _reap_proc(p: subprocess.Popen, fh, leet_p=None):
            try:
                exit_code = p.wait()
                try:
                    fh.write(f"\n=== Training subprocess exited with code {exit_code} ===\n")
                    fh.flush()
                except Exception:
                    pass
            finally:
                try:
                    fh.close()
                except Exception:
                    pass
                # Note: We don't terminate leet_p here because:
                # - If it's in a separate terminal window, it should stay open to show metrics
                # - If it's a background process, it will exit naturally when training finishes
                # Only terminate if it's a background process that's still running
                if leet_p and leet_p.poll() is None:
                    # Only terminate background processes, not separate terminal windows
                    # (separate terminal windows have leet_p=None)
                    try:
                        leet_p.terminate()
                    except Exception:
                        pass

        if not wait:
            threading.Thread(target=_reap_proc, args=(process, log_fh, leet_process), daemon=True).start()
        else:
            try:
                exit_code = process.wait()
            finally:
                # Only terminate LEET if it's a background process (not a separate terminal window)
                # Separate terminal windows have leet_process=None
                if leet_process and leet_process.poll() is None:
                    _terminate_process(leet_process)
                try:
                    log_fh.write(f"\n=== Training subprocess exited with code {exit_code} ===\n")
                    log_fh.flush()
                except Exception:
                    pass
                try:
                    log_fh.close()
                except Exception:
                    pass
            if exit_code != 0:
                raise RuntimeError(f"Training failed with exit code: {exit_code}")
        return process.pid
    except KeyboardInterrupt:
        _terminate_process(process)
        # Only terminate LEET if it's a background process (not a separate terminal window)
        if leet_process and leet_process.poll() is None:
            _terminate_process(leet_process)
        try:
            log_fh.write("\n[INTERRUPTED] Training cancelled by user.\n")
            log_fh.flush()
        except Exception:
            pass
        try:
            log_fh.close()
        except Exception:
            pass
        raise
    except (FileNotFoundError, PermissionError, OSError) as spawn_err:
        # Try Python fallback if accelerate spawn fails
        fallback_cmd = None
        if (not accelerate_path) or (cmd and os.path.basename(cmd[0]).startswith("accelerate")):
            try:
                if "-m" in cmd:
                    m_index = cmd.index("-m")
                    module = cmd[m_index + 1]
                    module_args = cmd[m_index + 2 :]
                    fallback_cmd = [sys.executable, "-m", module] + module_args
            except Exception:
                fallback_cmd = None

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Primary launch failed: {spawn_err}\n")
            if fallback_cmd:
                f.write(f"Attempting Python fallback: {' '.join(fallback_cmd)}\n")

        if fallback_cmd:
            process = subprocess.Popen(
                fallback_cmd,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
            if not wait:
                threading.Thread(target=_reap_proc, args=(process, log_fh), daemon=True).start()
        else:
            raise
    return process.pid
