"""
Forward-Backward Training Pipeline
===================================

Implements Tinker-inspired forward-backward training with:
- Async/non-blocking gradient computation
- Custom loss function support
- Gradient accumulation
- Future-based API for parallel training
"""

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel


logger = logging.getLogger(__name__)


@dataclass
class ForwardBackwardOutput:
    """Output from forward-backward pass."""

    loss: float
    logits: Optional[Tensor] = None
    logprobs: Optional[Tensor] = None
    metrics: Dict[str, float] = None
    gradients: Optional[Dict[str, Tensor]] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class OptimStepOutput:
    """Output from optimizer step."""

    step: int
    learning_rate: float
    grad_norm: float
    metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class AsyncTrainingFuture:
    """Future wrapper for async training operations."""

    def __init__(self, future: Future):
        self._future = future
        self._start_time = time.time()

    def result(self, timeout: Optional[float] = None) -> Any:
        """Get the result of the async operation."""
        result = self._future.result(timeout=timeout)
        elapsed = time.time() - self._start_time
        if hasattr(result, "metrics"):
            result.metrics["execution_time"] = elapsed
        return result

    def done(self) -> bool:
        """Check if the operation is complete."""
        return self._future.done()

    def cancel(self) -> bool:
        """Cancel the operation."""
        return self._future.cancel()


class ForwardBackwardPipeline:
    """
    Implements forward-backward training pipeline with async support.

    This is inspired by Tinker's approach where forward and backward passes
    can be queued and executed asynchronously.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        device: Optional[torch.device] = None,
        max_workers: int = 2,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Initialize the forward-backward pipeline.

        Args:
            model: The model to train
            device: Device to run training on
            max_workers: Number of worker threads for async operations
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._accumulated_steps = 0

        # Track gradients for accumulation
        self._accumulated_gradients = {}

    def forward_backward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        loss_fn: Union[str, Callable] = "cross_entropy",
        loss_fn_kwargs: Optional[Dict] = None,
    ) -> AsyncTrainingFuture:
        """
        Queue a forward-backward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for loss computation
            loss_fn: Loss function to use (string name or callable)
            loss_fn_kwargs: Additional kwargs for custom loss function

        Returns:
            AsyncTrainingFuture that will contain ForwardBackwardOutput
        """
        future = self.executor.submit(
            self._forward_backward_sync, input_ids, attention_mask, labels, loss_fn, loss_fn_kwargs or {}
        )
        return AsyncTrainingFuture(future)

    def _forward_backward_sync(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        labels: Optional[Tensor],
        loss_fn: Union[str, Callable],
        loss_fn_kwargs: Dict,
    ) -> ForwardBackwardOutput:
        """Synchronous forward-backward implementation."""

        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # Forward pass
        with torch.set_grad_enabled(True):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels if isinstance(loss_fn, str) else None,
            )

            # Compute loss
            if isinstance(loss_fn, str):
                loss = self._compute_builtin_loss(loss_fn, outputs, labels, **loss_fn_kwargs)
            else:
                # Custom loss function
                loss, metrics = loss_fn(outputs, labels, **loss_fn_kwargs)

            # Normalize loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Accumulate gradients
            self._accumulate_gradients()

            # Compute logprobs if available
            logprobs = None
            if hasattr(outputs, "logits"):
                with torch.no_grad():
                    logprobs = F.log_softmax(outputs.logits, dim=-1)
                    if labels is not None:
                        # Get logprobs for the target tokens
                        batch_size, seq_len = labels.shape
                        logprobs = logprobs.view(-1, logprobs.size(-1))
                        labels_flat = labels.view(-1)
                        logprobs = logprobs[torch.arange(labels_flat.size(0)), labels_flat]
                        logprobs = logprobs.view(batch_size, seq_len)

        return ForwardBackwardOutput(
            loss=loss.item() * self.gradient_accumulation_steps,
            logits=outputs.logits if hasattr(outputs, "logits") else None,
            logprobs=logprobs,
            metrics=metrics if not isinstance(loss_fn, str) else {},
        )

    def _compute_builtin_loss(self, loss_fn: str, outputs: Any, labels: Optional[Tensor], **kwargs) -> Tensor:
        """Compute loss using built-in loss functions."""

        if loss_fn == "cross_entropy":
            if hasattr(outputs, "loss") and outputs.loss is not None:
                return outputs.loss

            if not hasattr(outputs, "logits"):
                raise ValueError("Model outputs do not contain logits for cross_entropy loss")

            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean", **kwargs
            )
            return loss

        elif loss_fn == "importance_sampling":
            # Implement importance sampling loss (for RL)
            if "old_logprobs" not in kwargs:
                raise ValueError("importance_sampling loss requires old_logprobs")

            old_logprobs = kwargs["old_logprobs"]
            advantages = kwargs.get("advantages", torch.ones_like(labels, dtype=torch.float))

            # Compute current logprobs
            logits = outputs.logits
            logprobs = F.log_softmax(logits, dim=-1)

            # Importance sampling ratio
            ratio = torch.exp(logprobs - old_logprobs)

            # Compute loss
            loss = -(ratio * advantages).mean()
            return loss

        elif loss_fn == "ppo":
            # PPO loss (Proximal Policy Optimization)
            from ..losses.ppo_loss import PPOLoss

            if "old_log_probs" not in kwargs:
                raise ValueError("ppo loss requires old_log_probs")

            # Get required inputs
            old_log_probs = kwargs["old_log_probs"]
            advantages = kwargs["advantages"]

            # Compute current log probs
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)

            # Optional inputs
            values = kwargs.get("values")
            old_values = kwargs.get("old_values")
            returns = kwargs.get("returns")
            mask = kwargs.get("mask")

            # PPO hyperparameters
            clip_param = kwargs.get("clip_param", 0.2)
            value_clip = kwargs.get("value_clip", None)
            entropy_coef = kwargs.get("entropy_coef", 0.01)
            value_loss_coef = kwargs.get("value_loss_coef", 0.5)

            # Create PPO loss instance
            ppo_loss = PPOLoss(
                clip_param=clip_param,
                value_clip=value_clip,
                entropy_coef=entropy_coef,
                value_loss_coef=value_loss_coef,
            )

            # Compute PPO loss
            loss = ppo_loss.compute_loss(
                log_probs=log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages,
                values=values,
                old_values=old_values,
                returns=returns,
                mask=mask,
            ).mean()

            return loss

        else:
            raise ValueError(f"Unknown built-in loss function: {loss_fn}")

    def _accumulate_gradients(self):
        """Accumulate gradients across multiple forward-backward passes."""
        self._accumulated_steps += 1

        if self._accumulated_steps == 1:
            # First accumulation - store gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self._accumulated_gradients[name] = param.grad.clone()
        else:
            # Subsequent accumulations - add to stored gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name in self._accumulated_gradients:
                        self._accumulated_gradients[name] += param.grad
                    else:
                        self._accumulated_gradients[name] = param.grad.clone()

    def forward_backward_custom(
        self,
        input_ids: Tensor,
        custom_loss_fn: Callable,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> AsyncTrainingFuture:
        """
        Queue a forward-backward pass with a custom loss function.

        Inspired by Tinker's forward_backward_custom() for arbitrary differentiable losses.

        Args:
            input_ids: Input token IDs
            custom_loss_fn: Custom loss function that takes (model, inputs, outputs, **kwargs)
                           and returns a scalar loss tensor
            attention_mask: Attention mask
            **kwargs: Additional arguments passed to the custom loss function

        Returns:
            AsyncTrainingFuture containing ForwardBackwardOutput
        """
        future = self.executor.submit(
            self._forward_backward_custom_sync,
            input_ids,
            custom_loss_fn,
            attention_mask,
            kwargs,
        )
        return AsyncTrainingFuture(future)

    def _forward_backward_custom_sync(
        self,
        input_ids: Tensor,
        custom_loss_fn: Callable,
        attention_mask: Optional[Tensor],
        kwargs: Dict,
    ) -> ForwardBackwardOutput:
        """Synchronous forward-backward with custom loss implementation."""

        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass
        with torch.set_grad_enabled(True):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Prepare inputs dict for custom loss
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            # Compute custom loss
            # The custom loss function should return a scalar tensor
            loss = custom_loss_fn(self.model, inputs, outputs, **kwargs)

            if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
                raise ValueError(
                    f"Custom loss function must return a scalar tensor, got {type(loss)} with shape {loss.shape if hasattr(loss, 'shape') else 'N/A'}"
                )

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.gradient_accumulation_steps

        # Backward pass
        scaled_loss.backward()

        # Track gradients
        self._accumulate_gradients()

        # Extract logprobs if available
        logprobs = None
        if hasattr(outputs, "logits"):
            logprobs = F.log_softmax(outputs.logits, dim=-1)

        # Prepare output
        output = ForwardBackwardOutput(
            loss=loss.item(),
            logits=outputs.logits if hasattr(outputs, "logits") else None,
            logprobs=logprobs,
            metrics={
                "loss": loss.item(),
                "scaled_loss": scaled_loss.item(),
                "accumulated_steps": self._accumulated_steps,
            },
        )

        # Clear gradients if accumulation complete
        if self._accumulated_steps >= self.gradient_accumulation_steps:
            self.model.zero_grad()
            self._accumulated_steps = 0

        return output

    def optim_step(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        max_grad_norm: Optional[float] = 1.0,
    ) -> AsyncTrainingFuture:
        """
        Queue an optimizer step.

        Args:
            optimizer: The optimizer to use
            scheduler: Optional learning rate scheduler
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            AsyncTrainingFuture that will contain OptimStepOutput
        """
        future = self.executor.submit(self._optim_step_sync, optimizer, scheduler, max_grad_norm)
        return AsyncTrainingFuture(future)

    def _optim_step_sync(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        max_grad_norm: Optional[float],
    ) -> OptimStepOutput:
        """Synchronous optimizer step implementation."""

        # Check if we've accumulated enough gradients
        if self._accumulated_steps < self.gradient_accumulation_steps:
            return OptimStepOutput(
                step=0, learning_rate=optimizer.param_groups[0]["lr"], grad_norm=0.0, metrics={"skipped": True}
            )

        # Apply accumulated gradients to model parameters
        for name, param in self.model.named_parameters():
            if name in self._accumulated_gradients:
                param.grad = self._accumulated_gradients[name]

        # Gradient clipping
        grad_norm = 0.0
        if max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm).item()

        # Optimizer step
        optimizer.step()

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        # Clear gradients
        optimizer.zero_grad()
        self._accumulated_gradients.clear()
        self._accumulated_steps = 0

        return OptimStepOutput(
            step=1,
            learning_rate=optimizer.param_groups[0]["lr"],
            grad_norm=grad_norm,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> AsyncTrainingFuture:
        """
        Queue a forward pass only (no gradients).

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            AsyncTrainingFuture containing model outputs
        """
        future = self.executor.submit(self._forward_sync, input_ids, attention_mask)
        return AsyncTrainingFuture(future)

    def _forward_sync(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        """Synchronous forward pass implementation."""

        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass without gradients
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Compute logprobs
            logprobs = F.log_softmax(outputs.logits, dim=-1)

        return {
            "logits": outputs.logits,
            "logprobs": logprobs,
        }

    def save_state(self, name: str) -> Dict[str, Any]:
        """
        Save model weights and optimizer state.

        Inspired by Tinker's save_state() for resumable training.

        Args:
            name: Checkpoint name

        Returns:
            Dict containing path and metadata
        """
        from pathlib import Path

        # Create checkpoint directory
        checkpoint_dir = Path(f"checkpoints/{name}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_path = checkpoint_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        # Save optimizer state if exists
        optimizer_path = None
        if hasattr(self, "optimizer") and self.optimizer is not None:
            optimizer_path = checkpoint_dir / "optimizer.pt"
            torch.save(self.optimizer.state_dict(), optimizer_path)

        # Save training state
        state_path = checkpoint_dir / "training_state.pt"
        state = {
            "step": getattr(self, "global_step", 0),
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "accumulated_gradients": self.accumulated_gradients,
        }
        torch.save(state, state_path)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

        return {
            "path": str(checkpoint_dir),
            "model_path": str(model_path),
            "optimizer_path": str(optimizer_path) if optimizer_path else None,
            "state_path": str(state_path),
        }

    def load_state(self, path: str) -> None:
        """
        Load model weights and optimizer state from checkpoint.

        Inspired by Tinker's load_state() for resuming training.

        Args:
            path: Path to checkpoint directory or state dict
        """
        from pathlib import Path

        checkpoint_dir = Path(path)

        # Load model state
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded model weights from {model_path}")

        # Load optimizer state if exists
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists() and hasattr(self, "optimizer") and self.optimizer is not None:
            optimizer_state = torch.load(optimizer_path, map_location=self.device)
            self.optimizer.load_state_dict(optimizer_state)
            logger.info(f"Loaded optimizer state from {optimizer_path}")

        # Load training state
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            if hasattr(self, "global_step"):
                self.global_step = state.get("step", 0)
            self.gradient_accumulation_steps = state.get("gradient_accumulation_steps", 1)
            self.accumulated_gradients = state.get("accumulated_gradients", 0)
            logger.info(f"Loaded training state from {state_path}")

        logger.info(f"Successfully loaded checkpoint from {checkpoint_dir}")

    def sample(
        self,
        prompt: Union[List[int], Tensor],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        stop: Optional[List[Union[str, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate samples from the model.

        Inspired by Tinker's sample() API.

        Args:
            prompt: Input token IDs or tensor
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            stop: Stop sequences (strings or token IDs)

        Returns:
            Dict containing generated tokens and logprobs
        """
        from ...generation.sampling import create_sampler

        # Convert prompt to tensor if needed
        if isinstance(prompt, list):
            prompt = torch.tensor(prompt, dtype=torch.long, device=self.device).unsqueeze(0)
        elif len(prompt.shape) == 1:
            prompt = prompt.unsqueeze(0)

        # Create sampler
        sampler = create_sampler(
            strategy="top_p" if top_p < 1.0 else ("top_k" if top_k > 0 else "greedy"),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Generate tokens
        self.model.eval()
        generated_tokens = []
        logprobs = []

        with torch.no_grad():
            input_ids = prompt
            for _ in range(max_tokens):
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]

                # Sample next token
                next_token, token_logprobs = sampler.sample(logits)
                generated_tokens.append(next_token.item())
                logprobs.append(token_logprobs.item())

                # Check stop conditions
                if stop:
                    should_stop = False
                    for stop_token in stop:
                        if isinstance(stop_token, int) and next_token.item() == stop_token:
                            should_stop = True
                            break
                    if should_stop:
                        break

                # Append to input
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        self.model.train()

        return {
            "tokens": generated_tokens,
            "logprobs": logprobs,
            "prompt": prompt.squeeze(0).tolist(),
        }

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class AsyncTrainingClient:
    """
    High-level async training client inspired by Tinker's approach.

    Provides a clean API for async training with support for:
    - Multiple models (for reference models in DPO/PPO)
    - Gradient accumulation
    - Custom loss functions
    """

    def __init__(
        self,
        model: PreTrainedModel,
        reference_model: Optional[PreTrainedModel] = None,
        device: Optional[torch.device] = None,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Initialize the async training client.

        Args:
            model: The main model to train
            reference_model: Optional reference model (for PPO/DPO)
            device: Device to run training on
            gradient_accumulation_steps: Number of gradient accumulation steps
        """
        self.pipeline = ForwardBackwardPipeline(
            model=model, device=device, gradient_accumulation_steps=gradient_accumulation_steps
        )

        self.reference_pipeline = None
        if reference_model is not None:
            self.reference_pipeline = ForwardBackwardPipeline(
                model=reference_model,
                device=device,
                gradient_accumulation_steps=1,  # No gradient accumulation for reference
            )

    def forward_backward(
        self,
        batch: Dict[str, Tensor],
        loss_fn: Union[str, Callable] = "cross_entropy",
    ) -> AsyncTrainingFuture:
        """
        Queue a forward-backward pass on the main model.

        Args:
            batch: Batch containing input_ids, attention_mask, labels
            loss_fn: Loss function to use

        Returns:
            AsyncTrainingFuture containing ForwardBackwardOutput
        """
        return self.pipeline.forward_backward(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
            loss_fn=loss_fn,
        )

    def forward(
        self,
        batch: Dict[str, Tensor],
        use_reference: bool = False,
    ) -> AsyncTrainingFuture:
        """
        Queue a forward pass (no gradients).

        Args:
            batch: Batch containing input_ids, attention_mask
            use_reference: Whether to use reference model

        Returns:
            AsyncTrainingFuture containing forward outputs
        """
        pipeline = self.reference_pipeline if use_reference else self.pipeline
        if pipeline is None:
            raise ValueError("Reference model not initialized")

        return pipeline.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )

    def optim_step(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        max_grad_norm: float = 1.0,
    ) -> AsyncTrainingFuture:
        """
        Queue an optimizer step on the main model.

        Args:
            optimizer: Optimizer to use
            scheduler: Optional learning rate scheduler
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            AsyncTrainingFuture containing OptimStepOutput
        """
        return self.pipeline.optim_step(
            optimizer=optimizer,
            scheduler=scheduler,
            max_grad_norm=max_grad_norm,
        )

    def shutdown(self):
        """Shutdown all pipelines."""
        self.pipeline.shutdown()
        if self.reference_pipeline is not None:
            self.reference_pipeline.shutdown()
