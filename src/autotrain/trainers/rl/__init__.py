"""
AutoTrain Advanced Reinforcement Learning Module
================================================

Implements advanced training methods inspired by Tinker's approaches:
- PPO (Proximal Policy Optimization) for LLMs
- DPO (Direct Preference Optimization)
- RLHF (Reinforcement Learning from Human Feedback)
- Custom reward modeling
- Async forward-backward training pipeline
"""

from .dpo import DPOConfig, DPOTrainer
from .environments import (
    MultiObjectiveRewardEnv,
    PreferenceComparisonEnv,
    TextGenerationEnv,
    create_code_generation_env,
    create_math_problem_env,
)
from .forward_backward import AsyncTrainingFuture, ForwardBackwardOutput, ForwardBackwardPipeline, OptimStepOutput
from .ppo import PPOConfig, PPOTrainer
from .reward_model import (
    MultiObjectiveRewardModel,
    PairwiseRewardModel,
    RewardModel,
    RewardModelConfig,
    RewardModelTrainer,
)


__all__ = [
    # PPO
    "PPOTrainer",
    "PPOConfig",
    # DPO
    "DPOTrainer",
    "DPOConfig",
    # Reward Models
    "RewardModel",
    "RewardModelConfig",
    "PairwiseRewardModel",
    "MultiObjectiveRewardModel",
    "RewardModelTrainer",
    # Environments
    "TextGenerationEnv",
    "MultiObjectiveRewardEnv",
    "PreferenceComparisonEnv",
    "create_math_problem_env",
    "create_code_generation_env",
    # Forward-Backward Pipeline
    "ForwardBackwardPipeline",
    "AsyncTrainingFuture",
    "ForwardBackwardOutput",
    "OptimStepOutput",
]
