"""
Direct Preference Optimization (DPO) for LLMs
==============================================

Implements DPO algorithm for training LLMs from preference data without
requiring a separate reward model, inspired by Tinker's approach.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .forward_backward import AsyncTrainingClient


logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    # Model configuration
    model_name: str
    learning_rate: float = 1e-6
    batch_size: int = 8
    gradient_accumulation_steps: int = 2

    # DPO specific parameters
    beta: float = 0.1  # Temperature parameter for DPO loss
    label_smoothing: float = 0.0  # Label smoothing for robustness
    reference_free: bool = False  # Whether to use reference-free DPO

    # Training parameters
    num_epochs: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # Generation parameters (for evaluation)
    max_length: int = 512
    max_prompt_length: int = 256

    # Device
    device: Optional[str] = None

    # Logging
    eval_every: int = 100
    save_every: int = 500

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class PreferenceDataset(Dataset):
    """Dataset for preference pairs."""

    def __init__(
        self,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_prompt_length: int = 256,
    ):
        """
        Initialize preference dataset.

        Args:
            prompts: List of prompts
            chosen: List of preferred responses
            rejected: List of rejected responses
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            max_prompt_length: Maximum prompt length
        """
        self.prompts = prompts
        self.chosen = chosen
        self.rejected = rejected
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        prompt = self.prompts[idx]
        chosen = self.chosen[idx]
        rejected = self.rejected[idx]

        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )

        # Tokenize full sequences
        chosen_tokens = self.tokenizer(
            prompt + chosen,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        rejected_tokens = self.tokenizer(
            prompt + rejected,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "prompt_input_ids": prompt_tokens["input_ids"].squeeze(),
            "prompt_attention_mask": prompt_tokens["attention_mask"].squeeze(),
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(),
            "prompt_length": prompt_tokens["input_ids"].shape[1],
        }


class DPOTrainer:
    """
    DPO trainer for LLMs inspired by Tinker's implementation.

    DPO directly optimizes the policy to match human preferences without
    requiring a separate reward model training step.
    """

    def __init__(
        self,
        config: DPOConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        """
        Initialize DPO trainer.

        Args:
            config: DPO configuration
            tokenizer: Tokenizer for the model
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
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        ).to(self.device)

        # Reference model (frozen copy of the initial model)
        if not config.reference_free:
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
            ).to(self.device)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        else:
            self.reference_model = None

        # Initialize async training client
        self.training_client = AsyncTrainingClient(
            model=self.model,
            reference_model=self.reference_model,
            device=self.device,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Metrics
        self.metrics_history = []

    def compute_logprobs(
        self,
        model: PreTrainedModel,
        input_ids: Tensor,
        attention_mask: Tensor,
        prompt_length: int,
    ) -> Tensor:
        """
        Compute log probabilities for the response tokens.

        Args:
            model: Model to use for computation
            input_ids: Input token IDs
            attention_mask: Attention mask
            prompt_length: Length of the prompt (to mask prompt tokens)

        Returns:
            Log probabilities for response tokens
        """
        with torch.no_grad() if model == self.reference_model else torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

        # Compute log probabilities
        logprobs = F.log_softmax(logits, dim=-1)

        # Get logprobs for actual tokens (shift by 1)
        input_ids.shape[0]
        input_ids.shape[1]

        # Gather logprobs for actual tokens
        token_logprobs = torch.gather(
            logprobs[:, :-1, :],
            dim=2,
            index=input_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)

        # Mask prompt tokens (we only want response logprobs)
        response_mask = torch.zeros_like(token_logprobs)
        response_mask[:, prompt_length - 1 :] = 1.0

        # Apply mask and sum
        masked_logprobs = token_logprobs * response_mask
        sequence_logprobs = masked_logprobs.sum(dim=1)

        return sequence_logprobs

    def compute_dpo_loss(
        self,
        batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute DPO loss for a batch.

        Args:
            batch: Batch containing chosen and rejected examples

        Returns:
            Tuple of (loss, metrics)
        """
        # Get batch data
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
        prompt_length = batch["prompt_length"][0].item()  # Assuming same prompt length in batch

        # Compute log probabilities for chosen responses
        chosen_logprobs = self.compute_logprobs(
            self.model,
            chosen_input_ids,
            chosen_attention_mask,
            prompt_length,
        )

        # Compute log probabilities for rejected responses
        rejected_logprobs = self.compute_logprobs(
            self.model,
            rejected_input_ids,
            rejected_attention_mask,
            prompt_length,
        )

        # Compute reference log probabilities if not reference-free
        if self.reference_model is not None:
            with torch.no_grad():
                ref_chosen_logprobs = self.compute_logprobs(
                    self.reference_model,
                    chosen_input_ids,
                    chosen_attention_mask,
                    prompt_length,
                )
                ref_rejected_logprobs = self.compute_logprobs(
                    self.reference_model,
                    rejected_input_ids,
                    rejected_attention_mask,
                    prompt_length,
                )
        else:
            # Reference-free DPO: use zero reference logprobs
            ref_chosen_logprobs = torch.zeros_like(chosen_logprobs)
            ref_rejected_logprobs = torch.zeros_like(rejected_logprobs)

        # Compute log ratios
        chosen_log_ratios = chosen_logprobs - ref_chosen_logprobs
        rejected_log_ratios = rejected_logprobs - ref_rejected_logprobs

        # DPO loss
        if self.config.label_smoothing > 0:
            # Apply label smoothing
            chosen_rewards = self.config.beta * chosen_log_ratios
            rejected_rewards = self.config.beta * rejected_log_ratios

            # Smooth the binary classification
            chosen_label = 1 - self.config.label_smoothing
            rejected_label = self.config.label_smoothing

            loss = (
                -chosen_label * F.logsigmoid(chosen_rewards - rejected_rewards)
                - rejected_label * F.logsigmoid(rejected_rewards - chosen_rewards)
            ).mean()
        else:
            # Standard DPO loss
            loss = -F.logsigmoid(self.config.beta * (chosen_log_ratios - rejected_log_ratios)).mean()

        # Compute metrics
        with torch.no_grad():
            accuracy = (chosen_log_ratios > rejected_log_ratios).float().mean().item()
            chosen_rewards = self.config.beta * chosen_log_ratios
            rejected_rewards = self.config.beta * rejected_log_ratios
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "reward_margin": reward_margin,
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
        }

        return loss, metrics

    def train_step(
        self,
        batch: Dict[str, Tensor],
    ) -> Dict[str, float]:
        """
        Single DPO training step using async forward-backward.

        Args:
            batch: Training batch

        Returns:
            Dictionary of metrics
        """
        # Use async training client for forward-backward
        loss_fn = lambda outputs, labels, **kwargs: self.compute_dpo_loss(batch)

        # Queue forward-backward pass
        fwd_bwd_future = self.training_client.forward_backward(
            batch=batch,
            loss_fn=loss_fn,
        )

        # Queue optimizer step
        optim_future = self.training_client.optim_step(
            optimizer=self.optimizer,
            max_grad_norm=self.config.max_grad_norm,
        )

        # Wait for results
        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        return fwd_bwd_result.metrics

    def train(
        self,
        dataset: PreferenceDataset,
        eval_dataset: Optional[PreferenceDataset] = None,
    ) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            dataset: Training dataset
            eval_dataset: Optional evaluation dataset

        Returns:
            Dictionary of training metrics
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        total_steps = len(dataloader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )

        all_metrics = {
            "loss": [],
            "accuracy": [],
            "reward_margin": [],
        }

        global_step = 0

        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

            for batch_idx, batch in enumerate(dataloader):
                # Training step
                metrics = self.train_step(batch)

                # Update scheduler
                if global_step < warmup_steps:
                    scheduler.step()

                # Log metrics
                for key, value in metrics.items():
                    all_metrics.setdefault(key, []).append(value)

                if global_step % 10 == 0:
                    logger.info(
                        f"Step {global_step}: loss={metrics['loss']:.4f}, "
                        f"accuracy={metrics['accuracy']:.3f}, "
                        f"margin={metrics['reward_margin']:.3f}"
                    )

                # Evaluation
                if eval_dataset is not None and global_step % self.config.eval_every == 0:
                    eval_metrics = self.evaluate(eval_dataset)
                    logger.info(f"Eval at step {global_step}: {eval_metrics}")

                # Save checkpoint
                if global_step % self.config.save_every == 0:
                    self.save_checkpoint(f"dpo_checkpoint_{global_step}")

                global_step += 1

        # Shutdown async training
        self.training_client.shutdown()

        return all_metrics

    def evaluate(
        self,
        dataset: PreferenceDataset,
    ) -> Dict[str, float]:
        """
        Evaluate on a dataset.

        Args:
            dataset: Evaluation dataset

        Returns:
            Dictionary of evaluation metrics
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        total_loss = 0
        total_accuracy = 0
        total_margin = 0
        num_batches = 0

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                loss, metrics = self.compute_dpo_loss(batch)
                total_loss += metrics["loss"]
                total_accuracy += metrics["accuracy"]
                total_margin += metrics["reward_margin"]
                num_batches += 1

        self.model.train()

        return {
            "eval_loss": total_loss / num_batches,
            "eval_accuracy": total_accuracy / num_batches,
            "eval_margin": total_margin / num_batches,
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")
