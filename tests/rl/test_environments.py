"""
Tests for RL Environments
==========================
"""

import os

# Import our modules
import sys

import pytest
import torch
from transformers import AutoTokenizer


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from autotrain.trainers.rl.environments import (
    MultiObjectiveRewardEnv,
    Observation,
    PreferenceComparisonEnv,
    StepResult,
    TextGenerationEnv,
    create_code_generation_env,
    create_math_problem_env,
)


@pytest.fixture
def tokenizer():
    """Create a tokenizer for testing."""
    # Use a small model for testing
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    # Set pad_token to eos_token for GPT2
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "What is 2 + 2?",
        "Write a Python function to add two numbers.",
        "Explain machine learning in simple terms.",
    ]


class TestTextGenerationEnv:
    """Test TextGenerationEnv class."""

    def test_initialization(self, tokenizer, sample_prompts):
        """Test environment initialization."""
        env = TextGenerationEnv(
            tokenizer=tokenizer,
            prompts=sample_prompts,
            max_length=128,
        )

        assert env.tokenizer == tokenizer
        assert env.prompts == sample_prompts
        assert env.max_length == 128
        assert env.current_prompt_idx == 0

    def test_reset(self, tokenizer, sample_prompts):
        """Test environment reset."""
        env = TextGenerationEnv(
            tokenizer=tokenizer,
            prompts=sample_prompts,
        )

        obs = env.reset()

        assert isinstance(obs, Observation)
        assert obs.prompt == sample_prompts[0]
        assert obs.input_ids is not None
        assert obs.attention_mask is not None
        assert env.current_text == sample_prompts[0]
        assert env.steps == 0

    def test_step(self, tokenizer, sample_prompts):
        """Test environment step."""
        env = TextGenerationEnv(
            tokenizer=tokenizer,
            prompts=sample_prompts,
            max_length=10,
        )

        obs = env.reset()

        # Simulate a step with a token
        action = torch.tensor([50])  # Random token ID
        result = env.step(action)

        assert isinstance(result, StepResult)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert env.steps == 1

        # Check that text was updated
        assert len(env.current_text) > len(sample_prompts[0])

    def test_max_length_termination(self, tokenizer):
        """Test that environment terminates at max_length."""
        env = TextGenerationEnv(
            tokenizer=tokenizer,
            prompts=["Test"],
            max_length=5,
        )

        env.reset()

        # Take steps until done
        for i in range(10):
            action = torch.tensor([50])
            result = env.step(action)
            if result.done:
                break

        assert result.done
        assert env.steps <= 5

    def test_custom_reward_function(self, tokenizer):
        """Test custom reward function."""

        def custom_reward(prompt, generated, full_text):
            return len(generated) * 0.5

        env = TextGenerationEnv(
            tokenizer=tokenizer,
            prompts=["Test"],
            reward_fn=custom_reward,
        )

        env.reset()

        # Generate some text
        for _ in range(3):
            env.step(torch.tensor([50]))

        # Final step should trigger reward
        result = env.step(torch.tensor([tokenizer.eos_token_id]))

        assert result.done
        assert result.reward > 0


class TestMultiObjectiveRewardEnv:
    """Test MultiObjectiveRewardEnv class."""

    def test_initialization(self, tokenizer, sample_prompts):
        """Test multi-objective environment initialization."""

        def reward1(p, g, f):
            return 1.0

        def reward2(p, g, f):
            return 0.5

        env = MultiObjectiveRewardEnv(
            tokenizer=tokenizer,
            prompts=sample_prompts,
            reward_components={
                "reward1": reward1,
                "reward2": reward2,
            },
            reward_weights={"reward1": 1.0, "reward2": 2.0},
        )

        assert len(env.reward_components) == 2
        assert env.reward_weights["reward2"] == 2.0

    def test_multi_objective_reward_computation(self, tokenizer):
        """Test multi-objective reward computation."""

        def correctness_reward(p, g, f):
            return 1.0 if "4" in g else 0.0

        def length_reward(p, g, f):
            return min(len(g.split()) / 10, 1.0)

        env = MultiObjectiveRewardEnv(
            tokenizer=tokenizer,
            prompts=["What is 2 + 2?"],
            reward_components={
                "correctness": correctness_reward,
                "length": length_reward,
            },
            reward_weights={"correctness": 1.0, "length": 0.1},
        )

        total, components = env.compute_multi_objective_reward(
            "What is 2 + 2?", "The answer is 4", "What is 2 + 2? The answer is 4"
        )

        assert components["correctness"] == 1.0
        assert components["length"] > 0
        assert total == 1.0 + 0.1 * components["length"]


class TestPreferenceComparisonEnv:
    """Test PreferenceComparisonEnv class."""

    def test_initialization(self, tokenizer, sample_prompts):
        """Test preference environment initialization."""
        env = PreferenceComparisonEnv(
            tokenizer=tokenizer,
            prompts=sample_prompts,
            max_length=128,
        )

        assert env.prompts == sample_prompts
        assert env.responses == []

    def test_reset(self, tokenizer, sample_prompts):
        """Test preference environment reset."""
        env = PreferenceComparisonEnv(
            tokenizer=tokenizer,
            prompts=sample_prompts,
        )

        obs = env.reset()

        assert isinstance(obs, Observation)
        assert env.responses == []
        assert obs.metadata["comparison_round"] == 0

    def test_two_response_comparison(self, tokenizer):
        """Test that environment compares two responses."""

        def mock_feedback(prompt, resp1, resp2):
            return 1.0 if len(resp1) > len(resp2) else -1.0

        env = PreferenceComparisonEnv(
            tokenizer=tokenizer,
            prompts=["Test"],
            human_feedback_fn=mock_feedback,
        )

        env.reset()

        # First response
        result1 = env.step(torch.tensor([50, 51, 52]))
        assert not result1.done
        assert len(env.responses) == 1

        # Second response triggers comparison
        result2 = env.step(torch.tensor([50]))
        assert result2.done
        assert len(env.responses) == 2
        assert result2.reward != 0


class TestFactoryFunctions:
    """Test factory functions for creating environments."""

    def test_create_math_problem_env(self, tokenizer):
        """Test math problem environment creation."""
        env = create_math_problem_env(tokenizer)

        assert isinstance(env, MultiObjectiveRewardEnv)
        assert "correctness" in env.reward_components
        assert "formatting" in env.reward_components
        assert env.reward_weights["correctness"] == 1.0
        assert env.reward_weights["formatting"] == 0.1

    def test_create_code_generation_env(self, tokenizer):
        """Test code generation environment creation."""
        env = create_code_generation_env(tokenizer)

        assert isinstance(env, MultiObjectiveRewardEnv)
        assert "syntax" in env.reward_components
        assert "style" in env.reward_components


class TestObservationAndStepResult:
    """Test data classes."""

    def test_observation_creation(self):
        """Test Observation creation."""
        obs = Observation(
            input_ids=torch.tensor([1, 2, 3]),
            attention_mask=torch.tensor([1, 1, 1]),
            prompt="Test prompt",
            metadata={"key": "value"},
        )

        assert obs.prompt == "Test prompt"
        assert obs.metadata["key"] == "value"
        assert obs.input_ids.shape[0] == 3

    def test_step_result_creation(self):
        """Test StepResult creation."""
        result = StepResult(reward=1.5, done=True, info={"test": "info"}, metrics={"metric": 0.8})

        assert result.reward == 1.5
        assert result.done
        assert result.info["test"] == "info"
        assert result.metrics["metric"] == 0.8


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
