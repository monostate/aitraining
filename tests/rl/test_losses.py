"""
Tests for Custom Loss Functions
================================
"""

import os

# Import our modules
import sys

import pytest
import torch
import torch.nn.functional as F


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from autotrain.trainers.losses.custom_loss import (
    AdaptiveLoss,
    CompositeLoss,
    ContrastiveLoss,
    CustomLoss,
    CustomLossConfig,
    FocalLoss,
    TokenLevelLoss,
)
from autotrain.trainers.losses.importance_sampling import ImportanceSamplingLoss
from autotrain.trainers.losses.kl_loss import KLDivergenceLoss, compute_kl_penalty
from autotrain.trainers.losses.ppo_loss import PPOLoss, compute_ppo_loss
from autotrain.trainers.losses.variance_loss import VarianceLoss, variance_regularization


class TestCustomLossConfig:
    """Test CustomLossConfig."""

    def test_config_creation(self):
        """Test config creation with defaults."""
        config = CustomLossConfig(name="test")

        assert config.name == "test"
        assert config.weight == 1.0
        assert config.reduction == "mean"
        assert config.normalize is False
        assert config.clip_value is None
        assert config.temperature == 1.0
        assert config.epsilon == 1e-8

    def test_config_with_custom_values(self):
        """Test config with custom values."""
        config = CustomLossConfig(
            name="custom",
            weight=2.0,
            reduction="sum",
            normalize=True,
            clip_value=5.0,
        )

        assert config.weight == 2.0
        assert config.reduction == "sum"
        assert config.normalize
        assert config.clip_value == 5.0


class SimpleLoss(CustomLoss):
    """Simple loss for testing."""

    def compute_loss(self, predictions, targets, mask=None, **kwargs):
        return F.mse_loss(predictions, targets, reduction="none")


class TestCustomLoss:
    """Test base CustomLoss functionality."""

    def test_forward_mean_reduction(self):
        """Test forward pass with mean reduction."""
        config = CustomLossConfig(name="simple", reduction="mean")
        loss = SimpleLoss(config)

        predictions = torch.randn(4, 10)
        targets = torch.randn(4, 10)

        result = loss(predictions, targets)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Scalar

    def test_forward_sum_reduction(self):
        """Test forward pass with sum reduction."""
        config = CustomLossConfig(name="simple", reduction="sum")
        loss = SimpleLoss(config)

        predictions = torch.randn(4, 10)
        targets = torch.randn(4, 10)

        result = loss(predictions, targets)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

    def test_forward_with_mask(self):
        """Test forward pass with masking."""
        config = CustomLossConfig(name="simple", reduction="mean")
        loss = SimpleLoss(config)

        predictions = torch.randn(4, 10, requires_grad=True)
        targets = torch.randn(4, 10)
        mask = torch.ones(4, 10)
        mask[:, 5:] = 0  # Mask out half

        result = loss(predictions, targets, mask=mask)

        assert isinstance(result, torch.Tensor)
        assert result.requires_grad

    def test_forward_return_dict(self):
        """Test forward pass returning dictionary."""
        config = CustomLossConfig(name="simple", weight=2.0)
        loss = SimpleLoss(config)

        predictions = torch.randn(4, 10)
        targets = torch.randn(4, 10)

        result = loss(predictions, targets, return_dict=True)

        assert isinstance(result, dict)
        assert "loss" in result
        assert "weight" in result
        assert "name" in result
        assert result["weight"] == 2.0
        assert result["name"] == "simple"


class TestCompositeLoss:
    """Test CompositeLoss."""

    def test_composite_initialization(self):
        """Test composite loss initialization."""
        loss1 = SimpleLoss(CustomLossConfig(name="loss1"))
        loss2 = SimpleLoss(CustomLossConfig(name="loss2"))

        composite = CompositeLoss([loss1, loss2], weights=[1.0, 0.5])

        assert len(composite.losses) == 2
        assert composite.weights == [1.0, 0.5]

    def test_composite_forward(self):
        """Test composite loss forward pass."""
        loss1 = SimpleLoss(CustomLossConfig(name="loss1"))
        loss2 = SimpleLoss(CustomLossConfig(name="loss2"))

        composite = CompositeLoss([loss1, loss2], weights=[1.0, 2.0])

        predictions = torch.randn(4, 10)
        targets = torch.randn(4, 10)

        result = composite(predictions, targets)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

    def test_composite_with_components(self):
        """Test composite loss with component breakdown."""
        loss1 = SimpleLoss(CustomLossConfig(name="loss1"))
        loss2 = SimpleLoss(CustomLossConfig(name="loss2"))

        composite = CompositeLoss([loss1, loss2])

        predictions = torch.randn(4, 10)
        targets = torch.randn(4, 10)

        result = composite(predictions, targets, return_components=True)

        assert isinstance(result, dict)
        assert "loss" in result
        assert "components" in result
        assert len(result["components"]) == 2


class TestAdaptiveLoss:
    """Test AdaptiveLoss."""

    def test_adaptive_initialization(self):
        """Test adaptive loss initialization."""
        base_loss = SimpleLoss(CustomLossConfig(name="base"))
        schedule_fn = lambda step: 1.0 + step * 0.1

        adaptive = AdaptiveLoss(base_loss, schedule_fn)

        assert adaptive.base_loss == base_loss
        assert adaptive.current_step == 0

    def test_adaptive_weight_schedule(self):
        """Test adaptive weight scheduling."""
        base_loss = SimpleLoss(CustomLossConfig(name="base"))
        schedule_fn = lambda step: 1.0 + step * 0.5

        adaptive = AdaptiveLoss(base_loss, schedule_fn)

        predictions = torch.randn(4, 10)
        targets = torch.randn(4, 10)

        # First step - weight = 1.0
        loss1 = adaptive(predictions, targets)
        adaptive.step()

        # Second step - weight = 1.5
        loss2 = adaptive(predictions, targets)

        # Loss should increase due to weight increase
        # (This is a rough check as losses are on random data)
        assert adaptive.current_step == 1


class TestTokenLevelLoss:
    """Test TokenLevelLoss."""

    def test_initialization(self):
        """Test token-level loss initialization."""
        loss = TokenLevelLoss(ignore_index=-100)

        assert loss.ignore_index == -100

    def test_compute_loss(self):
        """Test token-level loss computation."""
        loss = TokenLevelLoss()

        predictions = torch.randn(2, 10, 100)  # batch, seq_len, vocab_size
        targets = torch.randint(0, 100, (2, 10))

        result = loss.compute_loss(predictions, targets)

        assert result.shape == (2, 10)

    def test_per_token_weights(self):
        """Test per-token weighting."""
        loss = TokenLevelLoss()

        predictions = torch.randn(2, 10, 100)
        targets = torch.randint(0, 100, (2, 10))
        weights = torch.randn(2, 10).abs()  # Positive weights

        result = loss.compute_loss(predictions, targets, per_token_weights=weights)

        assert result.shape == (2, 10)


class TestContrastiveLoss:
    """Test ContrastiveLoss."""

    def test_triplet_loss(self):
        """Test triplet loss computation."""
        loss = ContrastiveLoss(margin=1.0)

        anchor = torch.randn(4, 128)
        positive = torch.randn(4, 128)
        negative = torch.randn(4, 128)

        result = loss.compute_loss(anchor, positive, negative)

        assert result.shape == (4,)

    def test_pairwise_loss(self):
        """Test pairwise contrastive loss."""
        loss = ContrastiveLoss(margin=1.0)

        x1 = torch.randn(4, 128)
        x2 = torch.randn(4, 128)
        labels = torch.tensor([1, 0, 1, 0], dtype=torch.float)

        result = loss.compute_loss(x1, x2, labels=labels)

        assert result.shape == (4,)


class TestFocalLoss:
    """Test FocalLoss."""

    def test_focal_loss(self):
        """Test focal loss computation."""
        loss = FocalLoss(alpha=0.25, gamma=2.0)

        predictions = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4, 10), dtype=torch.float)

        result = loss.compute_loss(predictions, targets)

        assert result.shape == (4, 10)


class TestVarianceLoss:
    """Test VarianceLoss."""

    def test_variance_loss(self):
        """Test variance loss computation."""
        loss = VarianceLoss(target_variance=1.0, beta=0.1)

        predictions = torch.randn(4, 10)

        result = loss.compute_loss(predictions)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

    def test_variance_regularization(self):
        """Test variance regularization function."""
        outputs = torch.randn(4, 10)

        loss = variance_regularization(outputs, target_std=1.0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


class TestKLDivergenceLoss:
    """Test KLDivergenceLoss."""

    def test_kl_loss(self):
        """Test KL divergence loss."""
        loss = KLDivergenceLoss(target_kl=0.01, kl_coef=0.1)

        log_probs = torch.randn(4, 10)
        ref_log_probs = torch.randn(4, 10)

        result = loss.compute_loss(log_probs, ref_log_probs)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

    def test_compute_kl_penalty(self):
        """Test KL penalty computation."""
        logits = torch.randn(4, 10, 100)
        ref_logits = torch.randn(4, 10, 100)

        penalty = compute_kl_penalty(logits, ref_logits)

        assert isinstance(penalty, torch.Tensor)
        assert penalty.dim() == 0
        assert penalty >= 0  # KL divergence is non-negative


class TestImportanceSamplingLoss:
    """Test ImportanceSamplingLoss."""

    def test_importance_sampling_loss(self):
        """Test importance sampling loss."""
        loss = ImportanceSamplingLoss(clip_ratio=10.0)

        log_probs = torch.randn(4, 10)
        old_log_probs = torch.randn(4, 10)
        advantages = torch.randn(4, 10)

        result = loss.compute_loss(log_probs, old_log_probs, advantages)

        assert result.shape == (4, 10)


class TestPPOLoss:
    """Test PPOLoss."""

    def test_ppo_loss_basic(self):
        """Test basic PPO loss."""
        loss = PPOLoss(clip_param=0.2)

        log_probs = torch.randn(4, 10)
        old_log_probs = torch.randn(4, 10)
        advantages = torch.randn(4, 10)

        result = loss.compute_loss(log_probs, old_log_probs, advantages)

        assert result.shape == (4, 10)

    def test_ppo_loss_with_value(self):
        """Test PPO loss with value function."""
        loss = PPOLoss(clip_param=0.2, value_loss_coef=0.5)

        log_probs = torch.randn(4, 10)
        old_log_probs = torch.randn(4, 10)
        advantages = torch.randn(4, 10)
        values = torch.randn(4, 10)
        returns = torch.randn(4, 10)

        result = loss.compute_loss(log_probs, old_log_probs, advantages, values=values, returns=returns)

        assert result.shape == (4, 10)

    def test_compute_ppo_loss_function(self):
        """Test compute_ppo_loss function."""
        log_probs = torch.randn(4, 10)
        old_log_probs = torch.randn(4, 10)
        advantages = torch.randn(4, 10)

        loss, metrics = compute_ppo_loss(log_probs, old_log_probs, advantages, clip_param=0.2)

        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)
        assert "ppo_loss" in metrics
        assert "clip_fraction" in metrics
        assert "mean_ratio" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
