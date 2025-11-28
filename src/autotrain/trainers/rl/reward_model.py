"""
Reward Modeling for RLHF
=========================

Implements reward models for reinforcement learning from human feedback,
inspired by Tinker's approach to preference learning and reward modeling.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel, PreTrainedTokenizer


logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig:
    """Configuration for reward models."""

    model_name: str
    num_labels: int = 1
    pooling_strategy: str = "last"  # "mean", "last", "cls"
    dropout_prob: float = 0.1
    temperature: float = 1.0
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1


class RewardModel(nn.Module):
    """
    Reward model for RLHF training.

    This model takes text inputs and outputs scalar rewards,
    which can be used to train language models via reinforcement learning.
    """

    def __init__(
        self,
        config: RewardModelConfig,
        base_model: Optional[PreTrainedModel] = None,
    ):
        """
        Initialize the reward model.

        Args:
            config: Configuration for the reward model
            base_model: Optional pre-trained model to use as base
        """
        super().__init__()
        self.config = config

        # Load base model if not provided
        if base_model is None:
            self.base_model = AutoModel.from_pretrained(config.model_name)
        else:
            self.base_model = base_model

        # Get hidden size from model config
        self.hidden_size = self.base_model.config.hidden_size

        # Add reward head
        self.reward_head = nn.Sequential(
            nn.Dropout(config.dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(self.hidden_size, config.num_labels),
        )

        # Optional LoRA for efficient fine-tuning
        if config.use_lora:
            self._apply_lora()

    def _apply_lora(self):
        """Apply LoRA to the base model."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj"],  # Customize based on model
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
        except ImportError:
            logger.warning("PEFT not installed. Skipping LoRA application.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the reward model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_dict: Whether to return a dictionary

        Returns:
            Rewards or dictionary with rewards and hidden states
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Pool hidden states based on strategy
        if self.config.pooling_strategy == "mean":
            # Mean pooling over sequence
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
        elif self.config.pooling_strategy == "last":
            # Use last token
            hidden_states = outputs.last_hidden_state
            sequence_lengths = (attention_mask.sum(dim=1) - 1).long()
            batch_size = hidden_states.size(0)
            pooled = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths,
            ]
        else:  # cls
            # Use CLS token (first token)
            pooled = outputs.last_hidden_state[:, 0]

        # Pass through reward head
        rewards = self.reward_head(pooled)

        # Apply temperature scaling
        rewards = rewards / self.config.temperature

        if return_dict:
            return {
                "rewards": rewards,
                "pooled_output": pooled,
                "last_hidden_state": outputs.last_hidden_state,
            }
        else:
            return rewards

    def compute_preference_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
        margin: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute preference learning loss.

        Args:
            chosen_ids: Token IDs for chosen responses
            chosen_mask: Attention mask for chosen responses
            rejected_ids: Token IDs for rejected responses
            rejected_mask: Attention mask for rejected responses
            margin: Margin for ranking loss

        Returns:
            Preference learning loss
        """
        # Get rewards for chosen and rejected
        chosen_rewards = self.forward(chosen_ids, chosen_mask, return_dict=False)
        rejected_rewards = self.forward(rejected_ids, rejected_mask, return_dict=False)

        # Compute ranking loss with margin
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards - margin))
        return loss.mean()

    def predict_rewards(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> List[float]:
        """
        Predict rewards for a list of texts.

        Args:
            texts: List of text strings
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            batch_size: Batch size for inference

        Returns:
            List of reward scores
        """
        self.eval()
        rewards = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Tokenize batch
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

                # Move to device
                device = next(self.parameters()).device
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                # Get rewards
                batch_rewards = self.forward(input_ids, attention_mask, return_dict=False)
                rewards.extend(batch_rewards.cpu().numpy().tolist())

        return rewards


class PairwiseRewardModel(RewardModel):
    """
    Reward model that directly compares pairs of responses.

    This is useful for preference learning where we have
    pairwise comparisons rather than absolute rewards.
    """

    def __init__(self, config: RewardModelConfig, **kwargs):
        """Initialize pairwise reward model."""
        super().__init__(config, **kwargs)

        # Additional layers for pairwise comparison
        self.comparison_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(self.hidden_size, 1),
        )

    def forward_pair(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for comparing two inputs.

        Args:
            input_ids_a: Token IDs for first input
            attention_mask_a: Attention mask for first input
            input_ids_b: Token IDs for second input
            attention_mask_b: Attention mask for second input

        Returns:
            Preference score (positive means A is preferred)
        """
        # Get representations for both inputs
        outputs_a = self.forward(input_ids_a, attention_mask_a, return_dict=True)
        outputs_b = self.forward(input_ids_b, attention_mask_b, return_dict=True)

        # Concatenate pooled outputs
        combined = torch.cat([outputs_a["pooled_output"], outputs_b["pooled_output"]], dim=-1)

        # Get preference score
        preference = self.comparison_head(combined)
        return preference

    def compute_bradley_terry_loss(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
        labels: torch.Tensor,  # 1 if A is preferred, 0 if B is preferred
    ) -> torch.Tensor:
        """
        Compute Bradley-Terry model loss for preference learning.

        Args:
            input_ids_a: Token IDs for first input
            attention_mask_a: Attention mask for first input
            input_ids_b: Token IDs for second input
            attention_mask_b: Attention mask for second input
            labels: Binary labels indicating preference

        Returns:
            Bradley-Terry loss
        """
        # Get preference scores
        logits = self.forward_pair(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)

        # Compute Bradley-Terry loss
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels.float())
        return loss


class MultiObjectiveRewardModel(RewardModel):
    """
    Reward model with multiple objectives.

    This model outputs multiple reward signals that can be
    combined with different weights for various objectives.
    """

    def __init__(
        self,
        config: RewardModelConfig,
        num_objectives: int = 3,
        objective_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Initialize multi-objective reward model.

        Args:
            config: Model configuration
            num_objectives: Number of reward objectives
            objective_weights: Weights for combining objectives
            **kwargs: Additional arguments
        """
        # Update config for multiple objectives
        config.num_labels = num_objectives
        super().__init__(config, **kwargs)

        self.num_objectives = num_objectives
        self.objective_weights = objective_weights if objective_weights else [1.0 / num_objectives] * num_objectives

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_all_objectives: bool = False,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with multi-objective rewards.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_all_objectives: Whether to return all objectives
            return_dict: Whether to return dictionary

        Returns:
            Combined reward or all objectives
        """
        outputs = super().forward(input_ids, attention_mask, return_dict=True)
        multi_rewards = outputs["rewards"]  # Shape: (batch_size, num_objectives)

        if return_all_objectives:
            if return_dict:
                return {
                    "rewards": multi_rewards,
                    "combined_reward": self.combine_objectives(multi_rewards),
                    "pooled_output": outputs["pooled_output"],
                }
            else:
                return multi_rewards
        else:
            # Return combined reward
            combined = self.combine_objectives(multi_rewards)
            if return_dict:
                return {"rewards": combined, "pooled_output": outputs["pooled_output"]}
            else:
                return combined

    def combine_objectives(self, multi_rewards: torch.Tensor) -> torch.Tensor:
        """
        Combine multiple objectives into a single reward.

        Args:
            multi_rewards: Tensor of shape (batch_size, num_objectives)

        Returns:
            Combined reward tensor of shape (batch_size, 1)
        """
        weights = torch.tensor(
            self.objective_weights, device=multi_rewards.device, dtype=multi_rewards.dtype
        ).unsqueeze(0)
        return torch.sum(multi_rewards * weights, dim=-1, keepdim=True)

    def compute_multi_objective_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_rewards: torch.Tensor,
        objective_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss for multi-objective training.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            target_rewards: Target rewards for each objective
            objective_mask: Mask for which objectives to train

        Returns:
            Total loss and per-objective losses
        """
        # Get predicted rewards
        outputs = self.forward(input_ids, attention_mask, return_all_objectives=True, return_dict=True)
        predicted_rewards = outputs["rewards"]

        # Compute per-objective losses
        per_objective_losses = {}
        total_loss = 0.0

        for i in range(self.num_objectives):
            if objective_mask is None or objective_mask[i]:
                loss = F.mse_loss(predicted_rewards[:, i], target_rewards[:, i])
                per_objective_losses[f"objective_{i}_loss"] = loss
                total_loss += self.objective_weights[i] * loss

        return total_loss, per_objective_losses


# Training utilities for reward models
class RewardModelTrainer:
    """Trainer for reward models."""

    def __init__(
        self,
        model: RewardModel,
        tokenizer: PreTrainedTokenizer,
        config: RewardModelConfig,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the reward model trainer.

        Args:
            model: Reward model to train
            tokenizer: Tokenizer for encoding
            config: Training configuration
            device: Device to use for training
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

    def train_on_preferences(
        self,
        chosen_texts: List[str],
        rejected_texts: List[str],
        num_epochs: int = 3,
        batch_size: int = 8,
    ):
        """
        Train the reward model on preference data.

        Args:
            chosen_texts: List of chosen responses
            rejected_texts: List of rejected responses
            num_epochs: Number of training epochs
            batch_size: Training batch size
        """
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            for i in range(0, len(chosen_texts), batch_size):
                # Get batch
                batch_chosen = chosen_texts[i : i + batch_size]
                batch_rejected = rejected_texts[i : i + batch_size]

                # Tokenize
                chosen_encoded = self.tokenizer(
                    batch_chosen,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                rejected_encoded = self.tokenizer(
                    batch_rejected,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                # Forward pass
                loss = self.model.compute_preference_loss(
                    chosen_encoded["input_ids"],
                    chosen_encoded["attention_mask"],
                    rejected_encoded["input_ids"],
                    rejected_encoded["attention_mask"],
                )

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (i // batch_size + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def save_model(self, path: str):
        """Save the trained reward model."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Reward model saved to {path}")

    def load_model(self, path: str):
        """Load a trained reward model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Reward model loaded from {path}")
