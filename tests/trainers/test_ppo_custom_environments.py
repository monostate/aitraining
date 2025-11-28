#!/usr/bin/env python3
"""Tests for custom RL environments integration with PPO trainer - LIGHTWEIGHT."""

import json
import os
import sys

import pytest
import torch


# Add src to path if needed
if os.path.exists("/code/src"):
    sys.path.insert(0, "/code/src")

# Import required libraries
from transformers import AutoTokenizer

from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.clm.train_clm_ppo import create_env_aware_reward_fn, create_rl_environment


class TestCustomEnvironments:
    """Test suite for custom RL environment integration - lightweight, no training."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test environment with minimal resources."""
        print("\n[SETUP] Creating test environment...")
        self.temp_dir = str(tmp_path)
        self.model_name = "sshleifer/tiny-gpt2"
        self.tokenizer = None

        # Load tokenizer (lightweight)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[SETUP] ✓ Loaded tokenizer")
        except Exception as e:
            pytest.skip(f"Failed to load tokenizer: {e}")

        yield

    def test_text_generation_env_factory(self):
        """Test text_generation environment factory."""
        print("\nTEST: Create text_generation environment")

        config = LLMTrainingParams(
            trainer="ppo",
            model=self.model_name,
            data_path="/tmp",
            project_name="/tmp/test",
            rl_reward_model_path="/tmp/reward",
            rl_env_type="text_generation",
            rl_env_config='{"stop_sequences": ["\\n", "END"]}',
            epochs=1,
        )

        prompts = ["Test prompt 1", "Test prompt 2"]
        env = create_rl_environment(config, self.tokenizer, prompts)

        assert env is not None
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "render")
        print("[TEST] ✓ text_generation environment created successfully")

    def test_multi_objective_env_factory(self):
        """Test multi_objective environment factory."""
        print("\nTEST: Create multi_objective environment")

        env_config = {
            "reward_components": {
                "correctness": {"type": "keyword", "keywords": ["correct", "yes"]},
                "formatting": {"type": "keyword", "keywords": ["Answer:"]},
            },
            "reward_weights": {"correctness": 1.0, "formatting": 0.1},
        }

        config = LLMTrainingParams(
            trainer="ppo",
            model=self.model_name,
            data_path="/tmp",
            project_name="/tmp/test",
            rl_reward_model_path="/tmp/reward",
            rl_env_type="multi_objective",
            rl_multi_objective=True,
            rl_env_config=json.dumps(env_config),
            epochs=1,
        )

        prompts = ["Math problem 1", "Math problem 2"]
        env = create_rl_environment(config, self.tokenizer, prompts)

        assert env is not None
        assert hasattr(env, "compute_multi_objective_reward")
        print("[TEST] ✓ multi_objective environment created successfully")

    def test_preference_comparison_env_factory(self):
        """Test preference_comparison environment factory."""
        print("\nTEST: Create preference_comparison environment")

        config = LLMTrainingParams(
            trainer="ppo",
            model=self.model_name,
            data_path="/tmp",
            project_name="/tmp/test",
            rl_reward_model_path="/tmp/reward",
            rl_env_type="preference_comparison",
            rl_env_config="{}",
            epochs=1,
        )

        prompts = ["Preference prompt 1"]
        env = create_rl_environment(config, self.tokenizer, prompts)

        assert env is not None
        print("[TEST] ✓ preference_comparison environment created successfully")

    def test_env_config_json_parsing(self):
        """Test environment config parsing from JSON string."""
        print("\nTEST: JSON config parsing")

        config = LLMTrainingParams(
            trainer="ppo",
            model=self.model_name,
            data_path="/tmp",
            project_name="/tmp/test",
            rl_reward_model_path="/tmp/reward",
            rl_env_type="text_generation",
            rl_env_config='{"stop_sequences": ["END"]}',
            epochs=1,
        )

        prompts = ["Test"]
        env = create_rl_environment(config, self.tokenizer, prompts)

        assert env is not None
        print("[TEST] ✓ JSON string config parsed correctly")

    def test_invalid_env_type(self):
        """Test invalid environment type raises ValueError."""
        print("\nTEST: Invalid environment type")

        config = LLMTrainingParams(
            trainer="ppo",
            model=self.model_name,
            data_path="/tmp",
            project_name="/tmp/test",
            rl_reward_model_path="/tmp/reward",
            rl_env_type="invalid_type",
            epochs=1,
        )

        prompts = ["Test"]

        with pytest.raises(ValueError, match="Invalid rl_env_type"):
            create_rl_environment(config, self.tokenizer, prompts)
        print("[TEST] ✓ Correctly raised ValueError for invalid env type")

    def test_malformed_env_config(self):
        """Test malformed env config JSON raises ValueError."""
        print("\nTEST: Malformed JSON config")

        config = LLMTrainingParams(
            trainer="ppo",
            model=self.model_name,
            data_path="/tmp",
            project_name="/tmp/test",
            rl_reward_model_path="/tmp/reward",
            rl_env_type="text_generation",
            rl_env_config='{"invalid json',  # Malformed
            epochs=1,
        )

        prompts = ["Test"]

        with pytest.raises(ValueError, match="Failed to parse rl_env_config"):
            create_rl_environment(config, self.tokenizer, prompts)
        print("[TEST] ✓ Correctly raised ValueError for malformed JSON")

    def test_env_factory_no_env_type(self):
        """Test factory returns None when no env_type specified."""
        print("\nTEST: Factory with no env_type")

        config = LLMTrainingParams(
            trainer="ppo",
            model=self.model_name,
            data_path="/tmp",
            project_name="/tmp/test",
            rl_reward_model_path="/tmp/reward",
            # No rl_env_type
            epochs=1,
        )

        prompts = ["Test"]
        env = create_rl_environment(config, self.tokenizer, prompts)

        assert env is None
        print("[TEST] ✓ Factory correctly returns None when no env_type")

    def test_multi_objective_missing_components(self):
        """Test multi_objective env requires reward_components."""
        print("\nTEST: Multi-objective without reward_components")

        config = LLMTrainingParams(
            trainer="ppo",
            model=self.model_name,
            data_path="/tmp",
            project_name="/tmp/test",
            rl_reward_model_path="/tmp/reward",
            rl_env_type="multi_objective",
            rl_env_config="{}",  # Empty - missing reward_components
            epochs=1,
        )

        prompts = ["Test"]

        with pytest.raises(ValueError, match="reward_components"):
            create_rl_environment(config, self.tokenizer, prompts)
        print("[TEST] ✓ Correctly raised ValueError for missing reward_components")

    def test_reward_weights_parsing(self):
        """Test reward weights can be specified via rl_reward_weights."""
        print("\nTEST: Reward weights via rl_reward_weights")

        env_config = {
            "reward_components": {"comp1": {"type": "length"}, "comp2": {"type": "keyword", "keywords": ["test"]}}
        }

        config = LLMTrainingParams(
            trainer="ppo",
            model=self.model_name,
            data_path="/tmp",
            project_name="/tmp/test",
            rl_reward_model_path="/tmp/reward",
            rl_env_type="multi_objective",
            rl_multi_objective=True,
            rl_env_config=json.dumps(env_config),
            rl_reward_weights='{"comp1": 0.7, "comp2": 0.3}',
            epochs=1,
        )

        prompts = ["Test"]
        env = create_rl_environment(config, self.tokenizer, prompts)

        assert env is not None
        print("[TEST] ✓ Reward weights parsed correctly")

    def test_reward_fn_with_env(self):
        """Test reward function creation with environment."""
        print("\nTEST: Reward function with environment")

        config = LLMTrainingParams(
            trainer="ppo",
            model=self.model_name,
            data_path="/tmp",
            project_name="/tmp/test",
            rl_reward_model_path="/tmp/reward",
            rl_env_type="text_generation",
            rl_env_config="{}",
            epochs=1,
        )

        prompts = ["Test prompt"]
        env = create_rl_environment(config, self.tokenizer, prompts)
        reward_fn = create_env_aware_reward_fn(env, None, self.tokenizer)

        assert reward_fn is not None
        assert callable(reward_fn)

        # Test the reward function
        test_text = "Test response"
        test_tokens = self.tokenizer.encode(test_text, return_tensors="pt")[0]
        rewards = reward_fn([test_tokens])

        assert isinstance(rewards, list)
        assert len(rewards) == 1
        assert isinstance(rewards[0], (int, float))
        print("[TEST] ✓ Reward function with environment works correctly")

    def test_reward_fn_without_env(self):
        """Test reward function creation without environment (default)."""
        print("\nTEST: Default reward function without environment")

        reward_fn = create_env_aware_reward_fn(None, None, self.tokenizer)

        assert reward_fn is not None
        assert callable(reward_fn)

        # Test the reward function
        test_text = "Test response with punctuation."
        test_tokens = self.tokenizer.encode(test_text, return_tensors="pt")[0]
        rewards = reward_fn([test_tokens])

        assert isinstance(rewards, list)
        assert len(rewards) == 1
        print("[TEST] ✓ Default reward function works correctly")
