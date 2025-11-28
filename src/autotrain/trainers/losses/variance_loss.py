"""
Variance-based Loss Functions
==============================

Implements variance regularization and exploration-encouraging losses.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .custom_loss import CustomLoss, CustomLossConfig


class VarianceLoss(CustomLoss):
    """
    Loss function that encourages variance in outputs.

    Useful for exploration in RL and preventing mode collapse.
    """

    def __init__(
        self,
        config: Optional[CustomLossConfig] = None,
        target_variance: float = 1.0,
        beta: float = 0.1,
    ):
        """
        Initialize variance loss.

        Args:
            config: Configuration
            target_variance: Target variance to encourage
            beta: Weight for variance term
        """
        if config is None:
            config = CustomLossConfig(name="variance")
        super().__init__(config)
        self.target_variance = target_variance
        self.beta = beta

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute variance-based loss.

        Args:
            predictions: Model predictions
            targets: Optional targets (not used for pure variance loss)
            mask: Optional mask
            **kwargs: Additional arguments

        Returns:
            Variance loss
        """
        # Compute variance across batch dimension
        variance = torch.var(predictions, dim=0)

        # Penalize deviation from target variance
        variance_loss = F.mse_loss(variance, torch.full_like(variance, self.target_variance))

        # Optional: Add entropy regularization for discrete distributions
        if predictions.dim() == 2:  # Assume logits
            probs = F.softmax(predictions, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            entropy_loss = -entropy.mean()  # Negative because we want to maximize entropy
            variance_loss = variance_loss + self.beta * entropy_loss

        return variance_loss


def variance_regularization(
    outputs: torch.Tensor,
    target_std: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute variance regularization term.

    Args:
        outputs: Model outputs
        target_std: Target standard deviation
        reduction: Reduction method

    Returns:
        Variance regularization loss
    """
    std = torch.std(outputs, dim=0)
    loss = F.mse_loss(std, torch.full_like(std, target_std), reduction="none")

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
