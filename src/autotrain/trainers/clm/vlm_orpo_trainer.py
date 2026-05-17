"""
VLMORPOTrainer: ORPOTrainer subclass with VLM (Vision-Language Model) support.

Ports the VLM plumbing from DPOTrainer to ORPOTrainer:
- Detects ProcessorMixin vs PreTrainedTokenizerBase
- Uses DataCollatorForVisionPreference for on-the-fly image processing
- Passes pixel_values/image_grid_thw to model forward
- Skips upfront tokenization for vision datasets
"""

import torch

from transformers import ProcessorMixin, PreTrainedTokenizerBase

from autotrain import logger

try:
    from trl import ORPOTrainer
except ImportError:
    from trl.experimental.orpo import ORPOTrainer

from trl.trainer.dpo_trainer import DataCollatorForVisionPreference


def _chunked_per_token_logps(logits: torch.Tensor, labels: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """Per-token log-probs at label positions, computed in seq-dim chunks.

    Avoids the ~11 GB fp32 logsumexp temp that ``selective_log_softmax`` produces over the full
    [seq, vocab=248K] tensor on long VLM sequences. Each chunk's intermediate stays bounded by
    ``chunk_size * vocab * 4`` bytes (~1 GB at chunk_size=1024, vocab=248K).
    """
    B, S, _ = logits.shape
    out = torch.empty(B, S, device=logits.device, dtype=logits.dtype)
    safe_idx = labels.clamp(min=0).unsqueeze(-1)
    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)
        chunk = logits[:, start:end, :]
        gathered = chunk.gather(-1, safe_idx[:, start:end, :]).squeeze(-1)
        lse = torch.logsumexp(chunk, dim=-1)
        out[:, start:end] = gathered - lse
    return out


class VLMORPOTrainer(ORPOTrainer):
    """ORPOTrainer with VLM support.

    When a ProcessorMixin (e.g. AutoProcessor) is passed as processing_class
    and the dataset contains 'image'/'images' keys, this trainer:
    1. Uses DataCollatorForVisionPreference for on-the-fly image processing
    2. Skips upfront dataset tokenization (images processed at batch time)
    3. Passes pixel_values/image_grid_thw to model in forward pass
    """

    def __init__(self, *args, **kwargs):
        processing_class = kwargs.get("processing_class")

        # Detect VLM mode
        if isinstance(processing_class, ProcessorMixin):
            self._is_vlm = True
            self._vlm_processor = processing_class
            tokenizer = processing_class.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Check for vision dataset
            train_dataset = kwargs.get("train_dataset") or (args[3] if len(args) > 3 else None)
            if train_dataset is not None:
                sample = next(iter(train_dataset))
                self._is_vision_dataset = "image" in sample or "images" in sample
            else:
                self._is_vision_dataset = False

            if self._is_vision_dataset:
                # Replace processing_class with the inner tokenizer for parent's pad_token_id access
                kwargs["processing_class"] = tokenizer

                # Set up VLM data collator
                kwargs["data_collator"] = DataCollatorForVisionPreference(
                    processor=self._vlm_processor,
                )

                # Remove tokenization from parent's __init__ by pre-processing datasets minimally
                # Parent calls .map(self.tokenize_row) which we need to skip for VLM
                self._skip_tokenization = True
            else:
                self._skip_tokenization = False
        else:
            self._is_vlm = False
            self._is_vision_dataset = False
            self._skip_tokenization = False
            self._vlm_processor = None

        # Store datasets before parent __init__ modifies them
        if self._skip_tokenization:
            train_data = kwargs.get("train_dataset")
            eval_data = kwargs.get("eval_dataset")

        super().__init__(*args, **kwargs)

        # For VLM: restore original datasets (parent tokenized them, but we need raw)
        if self._skip_tokenization:
            self.train_dataset = train_data
            self.eval_dataset = eval_data

    def _set_signature_columns_if_needed(self):
        """Override to preserve image columns for VLM datasets."""
        if self._is_vision_dataset:
            self._signature_columns = [
                "prompt", "chosen", "rejected",
                "image", "images",
            ]
        else:
            super()._set_signature_columns_if_needed()

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        """Override to handle VLM batch format from DataCollatorForVisionPreference.

        VLM batches have new-style format: input_ids, attention_mask, completion_mask, pixel_values.
        Text-only batches have old-style format: chosen_input_ids, rejected_input_ids, etc.
        """
        if not self._is_vision_dataset:
            return super().get_batch_loss_metrics(model, batch, train_eval)

        # VLM path: batch from DataCollatorForVisionPreference
        # Format: input_ids [2*B, seq_len], completion_mask [2*B, seq_len]
        # First B are chosen, last B are rejected
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        completion_mask = batch["completion_mask"]

        # DataCollatorForVisionPreference does not honor max_length / max_completion_length, so we
        # cap here. Without this, a single long image-prompt sequence drives the chunked logsumexp
        # (and the model forward) over a [2*B, seq, vocab] tensor that OOMs even after the
        # CrossEntropyLoss fix.
        max_len = getattr(self.args, "max_length", None) or 8192
        if input_ids.shape[1] > max_len:
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]
            completion_mask = completion_mask[:, :max_len]

        batch_size = input_ids.shape[0] // 2  # half chosen, half rejected

        # Build model kwargs with VLM keys
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        for key in ("pixel_values", "pixel_attention_mask", "image_grid_thw", "image_sizes", "token_type_ids"):
            if key in batch:
                model_kwargs[key] = batch[key]

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(**model_kwargs)
        all_logits = outputs.logits

        # Build labels: use input_ids where completion_mask is 1, else -100
        labels = input_ids.clone()
        labels[completion_mask == 0] = -100

        # Shift for causal LM: predict next token
        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()

        # Chunked log-softmax: TRL's selective_log_softmax materializes a full [seq, vocab=248K]
        # fp32 logsumexp temp (~11 GB), and CrossEntropyLoss allocates the same again as a grad
        # tensor. _chunked_per_token_logps avoids both by streaming the seq dimension.
        per_token_logps = _chunked_per_token_logps(shift_logits, shift_labels)
        per_token_logps = per_token_logps.masked_fill(shift_completion_mask == 0, 0.0)
        all_logps = per_token_logps.sum(dim=1)

        policy_chosen_logps = all_logps[:batch_size]
        policy_rejected_logps = all_logps[batch_size:]

        # NLL on chosen, derived from per_token_logps. Equivalent to
        # CrossEntropyLoss(ignore_index=-100) over the chosen completion tokens, but reuses the
        # already-computed log-probs instead of allocating a second [seq, vocab] gradient tensor.
        chosen_per_token_logps = per_token_logps[:batch_size]
        chosen_completion_mask = shift_completion_mask[:batch_size]
        num_chosen_tokens = chosen_completion_mask.sum().clamp(min=1)
        policy_nll_loss = -chosen_per_token_logps.sum() / num_chosen_tokens

        # Mean logits for metrics
        policy_chosen_logits = shift_logits[:batch_size].detach().mean()
        policy_rejected_logits = shift_logits[batch_size:].detach().mean()

        # ORPO odds ratio loss
        losses, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen = self.odds_ratio_loss(
            policy_chosen_logps, policy_rejected_logps
        )

        # Full ORPO loss
        loss = policy_nll_loss - losses.mean()

        # Metrics
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = self.accelerator.gather_for_metrics(
            chosen_rewards - rejected_rewards
        ).mean().item()
        metrics[f"{prefix}logps/chosen"] = self.accelerator.gather_for_metrics(policy_chosen_logps).detach().mean().item()
        metrics[f"{prefix}logps/rejected"] = self.accelerator.gather_for_metrics(policy_rejected_logps).detach().mean().item()
        metrics[f"{prefix}logits/chosen"] = self.accelerator.gather_for_metrics(policy_chosen_logits).mean().item()
        metrics[f"{prefix}logits/rejected"] = self.accelerator.gather_for_metrics(policy_rejected_logits).mean().item()
        metrics[f"{prefix}nll_loss"] = self.accelerator.gather_for_metrics(policy_nll_loss).detach().mean().item()
        metrics[f"{prefix}log_odds_ratio"] = self.accelerator.gather_for_metrics(log_odds_ratio).detach().mean().item()
        metrics[f"{prefix}log_odds_chosen"] = self.accelerator.gather_for_metrics(log_odds_chosen).detach().mean().item()

        if self.aux_loss_enabled:
            loss += self.aux_loss_coef * outputs.aux_loss

        return loss, metrics
