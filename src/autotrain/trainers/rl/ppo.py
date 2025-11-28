"""
Proximal Policy Optimization (PPO) for Large Language Models
============================================================

Implements PPO algorithm for fine-tuning LLMs with:
- KL penalty for preventing model drift
- Advantage estimation
- Value function training
- Multi-objective rewards
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .forward_backward import AsyncTrainingClient


logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    # Model configuration
    model_name: str
    learning_rate: float = 1e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 1

    # PPO specific parameters
    ppo_epochs: int = 4
    gamma: float = 0.99  # Discount factor
    lam: float = 0.95  # GAE lambda
    clip_ratio: float = 0.2  # PPO clip ratio
    value_clip: float = 0.2  # Value function clip ratio
    max_grad_norm: float = 1.0

    # KL penalty
    kl_penalty_coef: float = 0.01
    kl_target: float = 0.01
    kl_horizon: int = 10000

    # Entropy regularization
    entropy_coef: float = 0.01

    # Value function
    value_coef: float = 0.5

    # Generation parameters
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

    # Training parameters
    num_iterations: int = 100
    save_every: int = 10
    eval_every: int = 5

    # Device
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class ValueHead(nn.Module):
    """Value head for PPO value function."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = self.dense(hidden_states)
        x = self.activation(x)
        return self.output(x)


class PPOModel(nn.Module):
    """PPO model wrapper with policy and value heads."""

    def __init__(self, base_model: PreTrainedModel):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.value_head = ValueHead(self.hidden_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_value: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass through policy and optionally value head."""

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        result = {"logits": outputs.logits}

        if return_value:
            # Get last hidden state
            last_hidden_state = outputs.hidden_states[-1]
            # Average pooling
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)
                count = mask_expanded.sum(dim=1)
                pooled = sum_hidden / count
            else:
                pooled = last_hidden_state.mean(dim=1)

            # Pass through value head
            values = self.value_head(pooled).squeeze(-1)
            result["values"] = values

        return result


class PPOTrainer:
    """
    PPO trainer for LLMs inspired by Tinker's RL approach.

    Key features:
    - Async forward-backward training
    - KL penalty to prevent drift from reference model
    - GAE for advantage estimation
    - Multi-objective reward support
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        reward_fn: Optional[Callable] = None,
    ):
        """
        Initialize PPO trainer.

        Args:
            config: PPO configuration
            tokenizer: Tokenizer for the model
            reward_fn: Custom reward function
        """
        self.config = config
        self.device = torch.device(config.device)

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        # Initialize models
        logger.info(f"Loading model: {config.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        )

        self.model = PPOModel(base_model).to(self.device)
        self.reference_model = PPOModel(
            AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
            )
        ).to(self.device)
        self.reference_model.eval()  # Reference model is frozen

        # Initialize async training client
        self.training_client = AsyncTrainingClient(
            model=self.model,
            reference_model=self.reference_model,
            device=self.device,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Initialize optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.model.base_model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        self.value_optimizer = torch.optim.AdamW(
            self.model.value_head.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Reward function
        self.reward_fn = reward_fn or self._default_reward_fn

        # KL controller
        self.kl_ctl = AdaptiveKLController(
            init_kl_coef=config.kl_penalty_coef,
            target=config.kl_target,
            horizon=config.kl_horizon,
        )

        # Metrics
        self.metrics = {}

    def _default_reward_fn(
        self,
        prompts: List[str],
        responses: List[str],
        metadata: Optional[Dict] = None,
    ) -> List[float]:
        """Default reward function (length-based)."""
        rewards = []
        for response in responses:
            # Simple reward based on response length (normalized)
            length = len(response.split())
            reward = min(length / 50.0, 1.0)  # Normalize to [0, 1]
            rewards.append(reward)
        return rewards

    def generate_responses(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, Tensor]]:
        """
        Generate responses for given prompts.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum new tokens to generate

        Returns:
            Tuple of (responses, generation_info)
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate with model
        with torch.no_grad():
            outputs = self.model.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode responses
        response_ids = outputs.sequences[:, inputs.input_ids.shape[1] :]
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # Get generation info
        generation_info = {
            "sequences": outputs.sequences,
            "response_ids": response_ids,
            "prompt_ids": inputs.input_ids,
        }

        return responses, generation_info

    def compute_advantages(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute advantages using GAE.

        Args:
            rewards: Reward tensor [batch_size, seq_len]
            values: Value estimates [batch_size, seq_len]
            dones: Episode done flags [batch_size, seq_len]

        Returns:
            Tuple of (advantages, returns)
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Compute GAE
        lastgaelam = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_values = 0
            else:
                next_values = values[:, t + 1]

            delta = rewards[:, t] + self.config.gamma * next_values * (1 - dones[:, t]) - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * (1 - dones[:, t]) * lastgaelam
            advantages[:, t] = lastgaelam
            returns[:, t] = advantages[:, t] + values[:, t]

        return advantages, returns

    def compute_kl_penalty(
        self,
        logprobs: Tensor,
        ref_logprobs: Tensor,
    ) -> Tuple[Tensor, float]:
        """
        Compute KL divergence penalty.

        Args:
            logprobs: Log probabilities from current policy
            ref_logprobs: Log probabilities from reference policy

        Returns:
            Tuple of (kl_penalty, mean_kl)
        """
        kl = logprobs - ref_logprobs
        mean_kl = kl.mean().item()

        # Adaptive KL coefficient
        kl_penalty = self.kl_ctl.value * kl

        return kl_penalty, mean_kl

    def train_step(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Single PPO training step.

        Args:
            batch: Training batch containing trajectories

        Returns:
            Dictionary of metrics
        """
        # Extract batch data
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        response_ids = batch["response_ids"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        old_logprobs = batch["old_logprobs"].to(self.device)
        old_values = batch["old_values"].to(self.device)
        ref_logprobs = batch["ref_logprobs"].to(self.device)

        # Compute advantages
        advantages, returns = self.compute_advantages(
            rewards, old_values, torch.zeros_like(rewards)  # No early stopping for now
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO training epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_kl = 0
        total_entropy = 0

        for _ in range(self.config.ppo_epochs):
            # Forward pass through model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_value=True,
            )

            logits = outputs["logits"]
            values = outputs["values"]

            # Compute log probabilities
            logprobs = F.log_softmax(logits, dim=-1)

            # Get logprobs for actual responses
            batch_size, seq_len = response_ids.shape
            logprobs_flat = logprobs.view(-1, logprobs.size(-1))
            response_ids_flat = response_ids.view(-1)
            current_logprobs = logprobs_flat[torch.arange(response_ids_flat.size(0)), response_ids_flat].view(
                batch_size, seq_len
            )

            # Compute ratios
            ratio = torch.exp(current_logprobs - old_logprobs)

            # Policy loss (PPO objective)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.config.value_clip,
                self.config.value_clip,
            )
            value_loss1 = F.mse_loss(values, returns)
            value_loss2 = F.mse_loss(values_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)

            # Entropy bonus
            entropy = -(logprobs * torch.exp(logprobs)).sum(dim=-1).mean()

            # KL penalty
            kl_penalty, mean_kl = self.compute_kl_penalty(current_logprobs, ref_logprobs)

            # Total loss
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
                + kl_penalty.mean()
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            # Optimizer steps
            self.policy_optimizer.step()
            self.value_optimizer.step()
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            # Accumulate metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_kl += mean_kl
            total_entropy += entropy.item()

        # Update KL controller
        self.kl_ctl.update(total_kl / self.config.ppo_epochs)

        metrics = {
            "policy_loss": total_policy_loss / self.config.ppo_epochs,
            "value_loss": total_value_loss / self.config.ppo_epochs,
            "kl_divergence": total_kl / self.config.ppo_epochs,
            "entropy": total_entropy / self.config.ppo_epochs,
            "kl_coef": self.kl_ctl.value,
        }

        return metrics

    def train(
        self,
        prompts: List[str],
        num_iterations: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            prompts: List of training prompts
            num_iterations: Number of training iterations

        Returns:
            Dictionary of training metrics
        """
        num_iterations = num_iterations or self.config.num_iterations
        all_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "kl_divergence": [],
            "entropy": [],
            "rewards": [],
        }

        for iteration in range(num_iterations):
            logger.info(f"Starting iteration {iteration + 1}/{num_iterations}")

            # Generate responses
            responses, generation_info = self.generate_responses(prompts)

            # Compute rewards
            rewards = self.reward_fn(prompts, responses)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)

            # Get old policy outputs
            with torch.no_grad():
                # Current policy
                old_outputs = self.model(
                    generation_info["sequences"],
                    return_value=True,
                )
                old_logits = old_outputs["logits"]
                old_values = old_outputs["values"]
                old_logprobs = F.log_softmax(old_logits, dim=-1)

                # Reference policy
                ref_outputs = self.reference_model(
                    generation_info["sequences"],
                    return_value=False,
                )
                ref_logits = ref_outputs["logits"]
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)

            # Prepare batch
            batch = {
                "input_ids": generation_info["sequences"],
                "attention_mask": torch.ones_like(generation_info["sequences"]),
                "response_ids": generation_info["response_ids"],
                "rewards": rewards_tensor.unsqueeze(-1).expand_as(generation_info["response_ids"]),
                "old_logprobs": old_logprobs,
                "old_values": old_values,
                "ref_logprobs": ref_logprobs,
            }

            # Training step
            metrics = self.train_step(batch)
            metrics["rewards"] = np.mean(rewards)

            # Log metrics
            for key, value in metrics.items():
                all_metrics.setdefault(key, []).append(value)

            logger.info(f"Iteration {iteration + 1} metrics: {metrics}")

            # Save checkpoint
            if (iteration + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"ppo_checkpoint_{iteration + 1}")

        return all_metrics

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "value_optimizer_state_dict": self.value_optimizer.state_dict(),
                "kl_ctl_value": self.kl_ctl.value,
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")


class AdaptiveKLController:
    """Adaptive KL coefficient controller."""

    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float):
        """Update KL coefficient based on current KL divergence."""
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        self.value *= 1 + proportional_error * 1 / self.horizon
