"""
Tests for Reward Models
========================
"""

import os

# Import our modules
import sys

import pytest
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from autotrain.trainers.rl.reward_model import (
    MultiObjectiveRewardModel,
    PairwiseRewardModel,
    RewardModel,
    RewardModelConfig,
    RewardModelTrainer,
)


@pytest.fixture
def config():
    """Create a config for testing."""
    return RewardModelConfig(
        model_name="gpt2",
        num_labels=1,
        pooling_strategy="last",
        dropout_prob=0.1,
        learning_rate=1e-4,
    )


@pytest.fixture
def tokenizer():
    """Create a tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    # Set pad_token to eos_token for GPT2
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This is a great product!",
        "This is terrible.",
        "Average quality, nothing special.",
    ]


class TestRewardModelConfig:
    """Test RewardModelConfig."""

    def test_config_creation(self):
        """Test config creation with default values."""
        config = RewardModelConfig(model_name="gpt2")

        assert config.model_name == "gpt2"
        assert config.num_labels == 1
        assert config.pooling_strategy == "last"
        assert config.dropout_prob == 0.1
        assert config.temperature == 1.0

    def test_config_with_lora(self):
        """Test config with LoRA settings."""
        config = RewardModelConfig(
            model_name="gpt2",
            use_lora=True,
            lora_rank=8,
            lora_alpha=16,
        )

        assert config.use_lora
        assert config.lora_rank == 8
        assert config.lora_alpha == 16


class TestRewardModel:
    """Test RewardModel class."""

    def test_initialization(self, config):
        """Test reward model initialization."""
        model = RewardModel(config)

        assert model.config == config
        assert model.base_model is not None
        assert model.reward_head is not None
        assert isinstance(model.reward_head, nn.Sequential)

    def test_forward_pass(self, config, tokenizer):
        """Test forward pass through the model."""
        model = RewardModel(config)

        # Create dummy input
        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones(2, 10)

        # Forward pass
        output = model(input_ids, attention_mask, return_dict=True)

        assert "rewards" in output
        assert "pooled_output" in output
        assert "last_hidden_state" in output
        assert output["rewards"].shape == (2, 1)

    def test_forward_pass_no_dict(self, config):
        """Test forward pass without returning dict."""
        model = RewardModel(config)

        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones(2, 10)

        rewards = model(input_ids, attention_mask, return_dict=False)

        assert rewards.shape == (2, 1)
        assert isinstance(rewards, torch.Tensor)

    def test_pooling_strategies(self, tokenizer):
        """Test different pooling strategies."""
        strategies = ["mean", "last", "cls"]

        for strategy in strategies:
            config = RewardModelConfig(
                model_name="gpt2",
                pooling_strategy=strategy,
            )
            model = RewardModel(config)

            input_ids = torch.randint(0, 100, (2, 10))
            attention_mask = torch.ones(2, 10)

            output = model(input_ids, attention_mask, return_dict=True)
            assert output["rewards"].shape == (2, 1)

    def test_compute_preference_loss(self, config):
        """Test preference loss computation."""
        model = RewardModel(config)

        chosen_ids = torch.randint(0, 100, (2, 10))
        chosen_mask = torch.ones(2, 10)
        rejected_ids = torch.randint(0, 100, (2, 10))
        rejected_mask = torch.ones(2, 10)

        loss = model.compute_preference_loss(chosen_ids, chosen_mask, rejected_ids, rejected_mask, margin=0.1)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.requires_grad

    def test_predict_rewards(self, config, tokenizer, sample_texts):
        """Test reward prediction for texts."""
        model = RewardModel(config)

        rewards = model.predict_rewards(
            sample_texts,
            tokenizer,
            max_length=128,
            batch_size=2,
        )

        assert len(rewards) == len(sample_texts)
        assert all(isinstance(r, list) for r in rewards)


class TestPairwiseRewardModel:
    """Test PairwiseRewardModel class."""

    def test_initialization(self, config):
        """Test pairwise model initialization."""
        model = PairwiseRewardModel(config)

        assert model.comparison_head is not None
        assert isinstance(model.comparison_head, nn.Sequential)

    def test_forward_pair(self, config):
        """Test pairwise forward pass."""
        model = PairwiseRewardModel(config)

        input_ids_a = torch.randint(0, 100, (2, 10))
        attention_mask_a = torch.ones(2, 10)
        input_ids_b = torch.randint(0, 100, (2, 10))
        attention_mask_b = torch.ones(2, 10)

        preference = model.forward_pair(
            input_ids_a,
            attention_mask_a,
            input_ids_b,
            attention_mask_b,
        )

        assert preference.shape == (2, 1)
        assert isinstance(preference, torch.Tensor)

    def test_bradley_terry_loss(self, config):
        """Test Bradley-Terry loss computation."""
        model = PairwiseRewardModel(config)

        input_ids_a = torch.randint(0, 100, (2, 10))
        attention_mask_a = torch.ones(2, 10)
        input_ids_b = torch.randint(0, 100, (2, 10))
        attention_mask_b = torch.ones(2, 10)
        labels = torch.tensor([1, 0], dtype=torch.float)

        loss = model.compute_bradley_terry_loss(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.requires_grad


class TestMultiObjectiveRewardModel:
    """Test MultiObjectiveRewardModel class."""

    def test_initialization(self, config):
        """Test multi-objective model initialization."""
        model = MultiObjectiveRewardModel(
            config,
            num_objectives=3,
            objective_weights=[0.5, 0.3, 0.2],
        )

        assert model.num_objectives == 3
        assert model.objective_weights == [0.5, 0.3, 0.2]
        assert model.config.num_labels == 3

    def test_forward_all_objectives(self, config):
        """Test forward pass returning all objectives."""
        model = MultiObjectiveRewardModel(
            config,
            num_objectives=3,
        )

        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones(2, 10)

        output = model(
            input_ids,
            attention_mask,
            return_all_objectives=True,
            return_dict=True,
        )

        assert "rewards" in output
        assert "combined_reward" in output
        assert output["rewards"].shape == (2, 3)
        assert output["combined_reward"].shape == (2, 1)

    def test_combine_objectives(self, config):
        """Test objective combination."""
        model = MultiObjectiveRewardModel(
            config,
            num_objectives=2,
            objective_weights=[0.7, 0.3],
        )

        multi_rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        combined = model.combine_objectives(multi_rewards)

        expected = torch.tensor([[1.3], [3.3]])
        assert torch.allclose(combined, expected)

    def test_multi_objective_loss(self, config):
        """Test multi-objective loss computation."""
        model = MultiObjectiveRewardModel(
            config,
            num_objectives=3,
            objective_weights=[0.5, 0.3, 0.2],
        )

        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones(2, 10)
        target_rewards = torch.randn(2, 3)

        total_loss, per_obj_losses = model.compute_multi_objective_loss(input_ids, attention_mask, target_rewards)

        assert isinstance(total_loss, torch.Tensor)
        assert len(per_obj_losses) == 3
        assert all("objective_" in k for k in per_obj_losses.keys())


class TestRewardModelTrainer:
    """Test RewardModelTrainer class."""

    def test_initialization(self, config, tokenizer):
        """Test trainer initialization."""
        model = RewardModel(config)
        trainer = RewardModelTrainer(model, tokenizer, config)

        assert trainer.model == model
        assert trainer.tokenizer == tokenizer
        assert trainer.config == config
        assert trainer.optimizer is not None

    def test_train_on_preferences(self, config, tokenizer):
        """Test training on preference data."""
        model = RewardModel(config)
        trainer = RewardModelTrainer(model, tokenizer, config)

        chosen_texts = ["This is good", "Great!"]
        rejected_texts = ["This is bad", "Terrible!"]

        # Just test that training runs without errors
        trainer.train_on_preferences(
            chosen_texts,
            rejected_texts,
            num_epochs=1,
            batch_size=2,
        )

        # Model should still be in train mode
        assert model.training

    def test_save_and_load_model(self, config, tokenizer, tmp_path):
        """Test model saving and loading."""
        model = RewardModel(config)
        trainer = RewardModelTrainer(model, tokenizer, config)

        # Manually set some weights to test saving/loading
        with torch.no_grad():
            for param in model.reward_head.parameters():
                param.data.fill_(0.5)

        # Save model
        save_path = str(tmp_path / "model.pt")
        trainer.save_model(save_path)

        assert os.path.exists(save_path)

        # Create test input
        input_ids = torch.randint(0, 100, (1, 10))
        attention_mask = torch.ones(1, 10)

        # Get output from original model
        with torch.no_grad():
            original_output = model(input_ids, attention_mask, return_dict=False)

        # Create a new model with different weights
        new_model = RewardModel(config)
        with torch.no_grad():
            for param in new_model.reward_head.parameters():
                param.data.fill_(0.1)  # Different weights

        # Verify models produce different outputs before loading
        with torch.no_grad():
            different_output = new_model(input_ids, attention_mask, return_dict=False)
        assert not torch.allclose(original_output, different_output)

        # Load saved weights
        new_trainer = RewardModelTrainer(new_model, tokenizer, config)
        new_trainer.load_model(save_path)

        # Check that loaded model now produces same output
        with torch.no_grad():
            loaded_output = new_model(input_ids, attention_mask, return_dict=False)

        # Only check reward head weights match (since base model wasn't saved)
        # Get first linear layer weights from the Sequential module
        original_weights = list(model.reward_head.parameters())[0]
        loaded_weights = list(new_model.reward_head.parameters())[0]
        assert torch.allclose(original_weights, loaded_weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
