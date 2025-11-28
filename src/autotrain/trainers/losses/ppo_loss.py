"""
PPO Loss Functions
==================

Implements Proximal Policy Optimization loss.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .custom_loss import CustomLoss, CustomLossConfig


class PPOLoss(CustomLoss):
    """
    Proximal Policy Optimization loss.

    Implements the clipped surrogate objective from PPO.
    """

    def __init__(
        self,
        config: Optional[CustomLossConfig] = None,
        clip_param: float = 0.2,
        value_clip: Optional[float] = None,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
    ):
        """
        Initialize PPO loss.

        Args:
            config: Configuration
            clip_param: PPO clipping parameter
            value_clip: Optional value function clipping
            entropy_coef: Entropy bonus coefficient
            value_loss_coef: Value loss coefficient
        """
        if config is None:
            config = CustomLossConfig(name="ppo")
        super().__init__(config)
        self.clip_param = clip_param
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        old_values: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute PPO loss.

        Args:
            log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from old policy
            advantages: Advantage estimates
            values: Value predictions (optional)
            old_values: Old value predictions (optional)
            returns: Return targets (optional)
            mask: Optional mask
            **kwargs: Additional arguments

        Returns:
            PPO loss
        """
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Compute clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2)

        # Optional value loss
        value_loss = 0
        if values is not None and returns is not None:
            if self.value_clip is not None and old_values is not None:
                # Clipped value loss
                value_pred_clipped = old_values + torch.clamp(values - old_values, -self.value_clip, self.value_clip)
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
            else:
                # Standard value loss
                value_loss = 0.5 * F.mse_loss(values, returns, reduction="none")

            value_loss = self.value_loss_coef * value_loss

        # Compute total loss
        total_loss = policy_loss + value_loss

        # Apply mask if provided
        if mask is not None:
            total_loss = total_loss * mask

        return total_loss


def compute_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_param: float = 0.2,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute PPO clipped objective.

    Args:
        log_probs: Current policy log probabilities
        old_log_probs: Old policy log probabilities
        advantages: Advantage estimates
        clip_param: Clipping parameter
        mask: Optional mask

    Returns:
        Loss value and metrics dictionary
    """
    # Compute probability ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    loss = -torch.min(surr1, surr2)

    # Apply mask
    if mask is not None:
        loss = loss * mask
        loss = loss.sum() / mask.sum().clamp(min=1)
    else:
        loss = loss.mean()

    # Compute metrics
    with torch.no_grad():
        clip_fraction = ((ratio - 1).abs() > clip_param).float()
        if mask is not None:
            clip_fraction = (clip_fraction * mask).sum() / mask.sum()
        else:
            clip_fraction = clip_fraction.mean()

    metrics = {
        "ppo_loss": loss.item(),
        "clip_fraction": clip_fraction.item(),
        "mean_ratio": ratio.mean().item(),
    }

    return loss, metrics
