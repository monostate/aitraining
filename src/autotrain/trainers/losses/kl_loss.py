"""
KL Divergence Loss Functions
=============================

Implements KL divergence losses for policy learning and regularization.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .custom_loss import CustomLoss, CustomLossConfig


class KLDivergenceLoss(CustomLoss):
    """
    KL divergence loss for policy learning.

    Used in PPO and other policy gradient methods to constrain
    policy updates.
    """

    def __init__(
        self,
        config: Optional[CustomLossConfig] = None,
        target_kl: float = 0.01,
        kl_coef: float = 0.1,
    ):
        """
        Initialize KL divergence loss.

        Args:
            config: Configuration
            target_kl: Target KL divergence
            kl_coef: Coefficient for KL penalty
        """
        if config is None:
            config = CustomLossConfig(name="kl_divergence")
        super().__init__(config)
        self.target_kl = target_kl
        self.kl_coef = kl_coef

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute KL divergence loss.

        Args:
            log_probs: Log probabilities from current policy
            ref_log_probs: Log probabilities from reference policy
            mask: Optional mask for valid positions
            rewards: Optional rewards for adaptive KL penalty
            **kwargs: Additional arguments

        Returns:
            KL divergence loss
        """
        # Compute KL divergence
        kl = log_probs - ref_log_probs

        if mask is not None:
            kl = kl * mask
            kl_mean = kl.sum() / mask.sum().clamp(min=1)
        else:
            kl_mean = kl.mean()

        # Adaptive KL penalty based on deviation from target
        kl_penalty = self.kl_coef * torch.abs(kl_mean - self.target_kl)

        return kl_penalty


def compute_kl_penalty(
    logits: torch.Tensor,
    ref_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute KL penalty between two distributions.

    Args:
        logits: Logits from current model
        ref_logits: Logits from reference model
        mask: Optional mask
        temperature: Temperature for softmax

    Returns:
        KL penalty value
    """
    # Compute log probabilities
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits / temperature, dim=-1)

    # Compute KL divergence
    kl = F.kl_div(
        log_probs,
        ref_log_probs.exp(),
        reduction="none",
        log_target=False,
    ).sum(-1)

    if mask is not None:
        kl = kl * mask
        return kl.sum() / mask.sum().clamp(min=1)
    else:
        return kl.mean()
