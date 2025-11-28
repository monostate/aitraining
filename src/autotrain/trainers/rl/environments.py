"""
RL Environments for Text Generation
====================================

Implements environments for reinforcement learning with LLMs,
inspired by Tinker's approach to RL training.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """Represents an observation in the RL environment."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result from environment step."""

    reward: float
    done: bool
    next_observation: Optional[Observation] = None
    info: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Trajectory:
    """A sequence of observations and actions from an episode."""

    observations: List[Observation]
    actions: List[torch.Tensor]
    rewards: List[float]
    logprobs: List[torch.Tensor]
    done: bool
    total_reward: float
    metrics: Dict[str, Any] = field(default_factory=dict)


class RLEnvironment(ABC):
    """Base class for RL environments."""

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment and return initial observation."""

    @abstractmethod
    def step(self, action: torch.Tensor) -> StepResult:
        """Take an action and return the result."""

    @abstractmethod
    def render(self) -> str:
        """Render the current state as a string."""


class TextGenerationEnv(RLEnvironment):
    """
    Environment for text generation tasks with RL.

    This environment supports various text generation scenarios where
    the model generates text and receives rewards based on quality metrics.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        max_length: int = 512,
        reward_fn: Optional[Callable] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: float = 1.0,
    ):
        """
        Initialize the text generation environment.

        Args:
            tokenizer: Tokenizer for encoding/decoding text
            prompts: List of prompts to generate from
            max_length: Maximum sequence length
            reward_fn: Function to compute rewards from generated text
            stop_sequences: Sequences that end generation
            temperature: Temperature for sampling
        """
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_length = max_length
        self.reward_fn = reward_fn or self._default_reward
        self.stop_sequences = stop_sequences or []
        self.temperature = temperature

        self.current_prompt_idx = 0
        self.current_prompt = None
        self.current_tokens = []
        self.current_text = ""
        self.steps = 0

    def reset(self) -> Observation:
        """Reset the environment with a new prompt."""
        self.current_prompt = self.prompts[self.current_prompt_idx]
        self.current_prompt_idx = (self.current_prompt_idx + 1) % len(self.prompts)
        self.current_tokens = []
        self.current_text = self.current_prompt
        self.steps = 0

        # Encode the prompt
        encoded = self.tokenizer(
            self.current_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        return Observation(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            prompt=self.current_prompt,
            metadata={"prompt_idx": self.current_prompt_idx - 1},
        )

    def step(self, action: torch.Tensor) -> StepResult:
        """Take a step in the environment."""
        self.steps += 1

        # Decode the action (token) to text
        token_text = self.tokenizer.decode(action, skip_special_tokens=True)
        self.current_tokens.append(action)
        self.current_text += token_text

        # Check if we should stop
        done = False
        if self.steps >= self.max_length:
            done = True
        elif any(seq in self.current_text for seq in self.stop_sequences):
            done = True
        elif action.item() == self.tokenizer.eos_token_id:
            done = True

        # Compute reward
        if done:
            generated_text = self.current_text[len(self.current_prompt) :]
            reward = self.reward_fn(prompt=self.current_prompt, generated=generated_text, full_text=self.current_text)
        else:
            reward = 0.0  # No intermediate rewards by default

        # Create next observation if not done
        next_obs = None
        if not done:
            encoded = self.tokenizer(
                self.current_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            next_obs = Observation(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                prompt=self.current_prompt,
                metadata={"steps": self.steps},
            )

        return StepResult(
            reward=reward,
            done=done,
            next_observation=next_obs,
            info={
                "generated_text": self.current_text[len(self.current_prompt) :],
                "steps": self.steps,
            },
            metrics={
                "reward": reward,
                "length": len(self.current_tokens),
            },
        )

    def render(self) -> str:
        """Render the current state."""
        return f"Prompt: {self.current_prompt}\nGenerated: {self.current_text[len(self.current_prompt):]}\nSteps: {self.steps}"

    def _default_reward(self, prompt: str, generated: str, full_text: str) -> float:
        """Default reward function based on text length."""
        # Simple length-based reward
        return len(generated.split()) / 100.0


class MultiObjectiveRewardEnv(TextGenerationEnv):
    """
    Environment with multiple reward objectives.

    This environment supports complex reward functions with multiple
    objectives, similar to Tinker's approach for combining correctness,
    formatting, and other quality metrics.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        reward_components: Dict[str, Callable],
        reward_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Initialize multi-objective environment.

        Args:
            tokenizer: Tokenizer for encoding/decoding
            prompts: List of prompts
            reward_components: Dictionary of reward component functions
            reward_weights: Weights for each reward component
            **kwargs: Additional arguments for TextGenerationEnv
        """
        super().__init__(tokenizer, prompts, **kwargs)
        self.reward_components = reward_components
        self.reward_weights = reward_weights or {k: 1.0 for k in reward_components}

    def compute_multi_objective_reward(
        self, prompt: str, generated: str, full_text: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward from multiple objectives.

        Returns:
            Total reward and individual component scores
        """
        component_rewards = {}
        total_reward = 0.0

        for name, reward_fn in self.reward_components.items():
            score = reward_fn(prompt, generated, full_text)
            component_rewards[name] = score
            weight = self.reward_weights.get(name, 1.0)
            total_reward += weight * score

        return total_reward, component_rewards

    def step(self, action: torch.Tensor) -> StepResult:
        """Take a step with multi-objective rewards."""
        result = super().step(action)

        # If episode is done, compute multi-objective rewards
        if result.done:
            generated_text = self.current_text[len(self.current_prompt) :]
            total_reward, component_rewards = self.compute_multi_objective_reward(
                self.current_prompt, generated_text, self.current_text
            )

            result.reward = total_reward
            result.metrics.update({f"reward_{k}": v for k, v in component_rewards.items()})
            result.metrics["reward_total"] = total_reward

        return result


class PreferenceComparisonEnv(RLEnvironment):
    """
    Environment for preference learning and comparison.

    This environment generates pairs of responses and learns from
    preference comparisons, useful for RLHF and DPO training.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        preference_model: Optional[PreTrainedModel] = None,
        human_feedback_fn: Optional[Callable] = None,
        max_length: int = 512,
    ):
        """
        Initialize preference comparison environment.

        Args:
            tokenizer: Tokenizer for encoding/decoding
            prompts: List of prompts
            preference_model: Model for computing preferences
            human_feedback_fn: Function for human preference feedback
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.preference_model = preference_model
        self.human_feedback_fn = human_feedback_fn
        self.max_length = max_length

        self.current_prompt_idx = 0
        self.current_prompt = None
        self.responses = []

    def reset(self) -> Observation:
        """Reset with a new prompt."""
        self.current_prompt = self.prompts[self.current_prompt_idx]
        self.current_prompt_idx = (self.current_prompt_idx + 1) % len(self.prompts)
        self.responses = []

        encoded = self.tokenizer(
            self.current_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        return Observation(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            prompt=self.current_prompt,
            metadata={"comparison_round": 0},
        )

    def step(self, action: torch.Tensor) -> StepResult:
        """Generate response and compute preference-based reward."""
        # Store the generated response
        response = self.tokenizer.decode(action, skip_special_tokens=True)
        self.responses.append(response)

        # If we have two responses, compute preference
        if len(self.responses) == 2:
            if self.preference_model:
                reward = self._compute_model_preference()
            elif self.human_feedback_fn:
                reward = self.human_feedback_fn(self.current_prompt, self.responses[0], self.responses[1])
            else:
                reward = 0.0

            done = True
        else:
            # Need another response
            reward = 0.0
            done = False

        return StepResult(
            reward=reward,
            done=done,
            info={"responses": self.responses},
            metrics={"num_responses": len(self.responses)},
        )

    def _compute_model_preference(self) -> float:
        """Compute preference using the preference model."""
        # This is a placeholder - actual implementation would use
        # the preference model to score the responses
        return np.random.randn()  # Random preference for now

    def render(self) -> str:
        """Render current state."""
        output = f"Prompt: {self.current_prompt}\n"
        for i, resp in enumerate(self.responses):
            output += f"Response {i+1}: {resp}\n"
        return output


# Factory functions for creating environments
def create_math_problem_env(tokenizer: PreTrainedTokenizer) -> MultiObjectiveRewardEnv:
    """
    Create an environment for math problem solving.

    Similar to Tinker's GSM8K example with correctness and formatting rewards.
    """

    def correctness_reward(prompt, generated, full_text):
        # Check if answer is correct (placeholder logic)
        return 1.0 if "correct" in generated.lower() else 0.0

    def formatting_reward(prompt, generated, full_text):
        # Check if properly formatted
        return 0.1 if "Answer:" in generated else -0.1

    return MultiObjectiveRewardEnv(
        tokenizer=tokenizer,
        prompts=["Solve: 2 + 2 = ?", "What is 5 * 3?"],  # Example prompts
        reward_components={
            "correctness": correctness_reward,
            "formatting": formatting_reward,
        },
        reward_weights={"correctness": 1.0, "formatting": 0.1},
    )


def create_code_generation_env(tokenizer: PreTrainedTokenizer) -> MultiObjectiveRewardEnv:
    """
    Create an environment for code generation tasks.
    """

    def syntax_reward(prompt, generated, full_text):
        # Check syntax validity (placeholder)
        try:
            compile(generated, "<string>", "exec")
            return 1.0
        except:
            return -0.5

    def style_reward(prompt, generated, full_text):
        # Check code style (placeholder)
        return 0.1 if "def " in generated else 0.0

    return MultiObjectiveRewardEnv(
        tokenizer=tokenizer,
        prompts=["Write a Python function to compute factorial"],
        reward_components={
            "syntax": syntax_reward,
            "style": style_reward,
        },
    )
