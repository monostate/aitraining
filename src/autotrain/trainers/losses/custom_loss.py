"""
Base Custom Loss Framework
===========================

Flexible loss function framework inspired by Tinker's approach
to custom loss functions.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


@dataclass
class CustomLossConfig:
    """Configuration for custom loss functions."""

    name: str
    weight: float = 1.0
    reduction: str = "mean"  # "mean", "sum", "none"
    normalize: bool = False
    clip_value: Optional[float] = None
    temperature: float = 1.0
    epsilon: float = 1e-8


class CustomLoss(nn.Module, ABC):
    """
    Base class for custom loss functions.

    This provides a flexible framework for implementing custom losses
    similar to Tinker's approach where arbitrary differentiable functions
    can be used as losses.
    """

    def __init__(self, config: CustomLossConfig):
        """
        Initialize the custom loss.

        Args:
            config: Loss configuration
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the loss value.

        Args:
            predictions: Model predictions
            targets: Target values
            mask: Optional mask for valid positions
            **kwargs: Additional arguments

        Returns:
            Loss tensor
        """

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the loss function.

        Args:
            predictions: Model predictions
            targets: Target values
            mask: Optional mask
            return_dict: Whether to return dictionary
            **kwargs: Additional arguments

        Returns:
            Loss value or dictionary with loss and metrics
        """
        # Compute raw loss
        loss = self.compute_loss(predictions, targets, mask, **kwargs)

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask

        # Apply reduction
        if self.config.reduction == "mean":
            if mask is not None:
                loss = loss.sum() / mask.sum().clamp(min=1)
            else:
                loss = loss.mean()
        elif self.config.reduction == "sum":
            loss = loss.sum()
        # else "none" - return per-element losses

        # Apply weight
        loss = loss * self.config.weight

        # Optional clipping
        if self.config.clip_value is not None:
            loss = torch.clamp(loss, max=self.config.clip_value)

        if return_dict:
            return {
                "loss": loss,
                "weight": self.config.weight,
                "name": self.config.name,
            }
        else:
            return loss


class CompositeLoss(CustomLoss):
    """
    Combines multiple loss functions with different weights.

    This enables complex multi-objective training similar to Tinker's
    approach of combining multiple loss components.
    """

    def __init__(
        self,
        losses: List[CustomLoss],
        weights: Optional[List[float]] = None,
        config: Optional[CustomLossConfig] = None,
    ):
        """
        Initialize composite loss.

        Args:
            losses: List of loss functions to combine
            weights: Weights for each loss
            config: Overall configuration
        """
        if config is None:
            config = CustomLossConfig(name="composite")
        super().__init__(config)

        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute combined loss from all components."""
        total_loss = 0.0

        for loss_fn, weight in zip(self.losses, self.weights):
            component_loss = loss_fn(predictions, targets, mask, **kwargs)
            total_loss += weight * component_loss

        return total_loss

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_components: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with optional component breakdown.

        Args:
            predictions: Model predictions
            targets: Target values
            mask: Optional mask
            return_components: Whether to return individual components
            **kwargs: Additional arguments

        Returns:
            Total loss or dictionary with components
        """
        if not return_components:
            return super().forward(predictions, targets, mask, **kwargs)

        # Compute each component
        component_losses = {}
        total_loss = 0.0

        for i, (loss_fn, weight) in enumerate(zip(self.losses, self.weights)):
            component = loss_fn(predictions, targets, mask, return_dict=True, **kwargs)
            component_name = component.get("name", f"component_{i}")
            component_losses[component_name] = component["loss"]
            total_loss += weight * component["loss"]

        return {
            "loss": total_loss,
            "components": component_losses,
            "weights": self.weights,
        }


class AdaptiveLoss(CustomLoss):
    """
    Loss function with adaptive weighting based on training progress.

    This implements dynamic loss weighting strategies that can change
    during training, similar to curriculum learning approaches.
    """

    def __init__(
        self,
        base_loss: CustomLoss,
        schedule_fn: Callable[[int], float],
        config: Optional[CustomLossConfig] = None,
    ):
        """
        Initialize adaptive loss.

        Args:
            base_loss: Underlying loss function
            schedule_fn: Function mapping step to weight
            config: Configuration
        """
        if config is None:
            config = CustomLossConfig(name="adaptive")
        super().__init__(config)

        self.base_loss = base_loss
        self.schedule_fn = schedule_fn
        self.current_step = 0

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss with adaptive weight."""
        # Get current weight from schedule
        weight = self.schedule_fn(self.current_step)

        # Compute base loss
        loss = self.base_loss.compute_loss(predictions, targets, mask, **kwargs)

        # Apply adaptive weight
        return loss * weight

    def step(self):
        """Increment the training step counter."""
        self.current_step += 1

    def reset(self):
        """Reset the step counter."""
        self.current_step = 0


class TokenLevelLoss(CustomLoss):
    """
    Loss computed at token level with fine-grained control.

    Similar to Tinker's token-level loss application for
    precise control over gradient flow.
    """

    def __init__(
        self,
        config: Optional[CustomLossConfig] = None,
        ignore_index: int = -100,
    ):
        """
        Initialize token-level loss.

        Args:
            config: Configuration
            ignore_index: Index to ignore in loss computation
        """
        if config is None:
            config = CustomLossConfig(name="token_level")
        super().__init__(config)
        self.ignore_index = ignore_index

    def compute_loss(
        self,
        predictions: torch.Tensor,  # (batch, seq_len, vocab_size)
        targets: torch.Tensor,  # (batch, seq_len)
        mask: Optional[torch.Tensor] = None,
        per_token_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute token-level loss.

        Args:
            predictions: Logits from model
            targets: Target token IDs
            mask: Attention mask
            per_token_weights: Per-token loss weights
            **kwargs: Additional arguments

        Returns:
            Token-level loss
        """
        # Reshape for cross entropy
        batch_size, seq_len, vocab_size = predictions.shape
        predictions = predictions.reshape(-1, vocab_size)
        targets = targets.reshape(-1)

        # Compute cross entropy
        loss = F.cross_entropy(
            predictions,
            targets,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # Reshape back
        loss = loss.reshape(batch_size, seq_len)

        # Apply per-token weights if provided
        if per_token_weights is not None:
            loss = loss * per_token_weights

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask

        return loss


class ContrastiveLoss(CustomLoss):
    """
    Contrastive loss for representation learning.

    Useful for learning discriminative representations
    in preference learning and RLHF scenarios.
    """

    def __init__(
        self,
        config: Optional[CustomLossConfig] = None,
        margin: float = 1.0,
        distance_metric: str = "cosine",  # "cosine", "euclidean"
    ):
        """
        Initialize contrastive loss.

        Args:
            config: Configuration
            margin: Margin for contrastive loss
            distance_metric: Distance metric to use
        """
        if config is None:
            config = CustomLossConfig(name="contrastive")
        super().__init__(config)
        self.margin = margin
        self.distance_metric = distance_metric

    def compute_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings
            negative: Negative embeddings (for triplet loss)
            labels: Binary labels (for pairwise loss)
            **kwargs: Additional arguments

        Returns:
            Contrastive loss
        """
        if negative is not None:
            # Triplet loss
            return self._triplet_loss(anchor, positive, negative)
        elif labels is not None:
            # Pairwise contrastive loss
            return self._pairwise_loss(anchor, positive, labels)
        else:
            raise ValueError("Either negative samples or labels must be provided")

    def _triplet_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet loss."""
        if self.distance_metric == "cosine":
            pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=-1)
            neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=-1)
        else:  # euclidean
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss

    def _pairwise_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise contrastive loss."""
        if self.distance_metric == "cosine":
            dist = 1 - F.cosine_similarity(x1, x2, dim=-1)
        else:  # euclidean
            dist = F.pairwise_distance(x1, x2)

        # Loss for similar pairs (label=1) and dissimilar pairs (label=0)
        loss = labels * dist.pow(2) + (1 - labels) * F.relu(self.margin - dist).pow(2)
        return loss


class FocalLoss(CustomLoss):
    """
    Focal loss for addressing class imbalance.

    Useful for scenarios with imbalanced rewards or preferences.
    """

    def __init__(
        self,
        config: Optional[CustomLossConfig] = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        """
        Initialize focal loss.

        Args:
            config: Configuration
            alpha: Weighting factor for class imbalance
            gamma: Focusing parameter
        """
        if config is None:
            config = CustomLossConfig(name="focal")
        super().__init__(config)
        self.alpha = alpha
        self.gamma = gamma

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            predictions: Model predictions (logits)
            targets: Target labels
            mask: Optional mask
            **kwargs: Additional arguments

        Returns:
            Focal loss
        """
        # Get probabilities
        probs = torch.sigmoid(predictions)

        # Compute focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt).pow(self.gamma)

        # Compute cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets.float(), reduction="none")

        # Apply focal weight and alpha
        loss = self.alpha * focal_weight * ce_loss

        return loss
