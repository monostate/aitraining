import json

import torch
import torch.nn.functional as F
from peft import LoraConfig
from transformers.trainer_callback import PrinterCallback
from trl import SFTConfig, SFTTrainer

from autotrain import logger
from autotrain.trainers.clm import utils
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.clm.sweep_utils import with_sweep


class GenerativeSFTTrainer(SFTTrainer):
    """SFT Trainer that generates text during evaluation for metrics like BLEU/ROUGE."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to generate text during evaluation."""
        # For training loss, use standard prediction
        if self.args.predict_with_generate and not prediction_loss_only:
            # Generate text for evaluation metrics
            has_labels = "labels" in inputs

            # Prepare generation inputs (remove labels)
            generation_inputs = {k: v for k, v in inputs.items() if k != "labels"}

            # Generate text
            with torch.no_grad():
                generated_tokens = model.generate(
                    **generation_inputs,
                    max_new_tokens=getattr(self.args, "generation_max_length", 128),
                    num_beams=getattr(self.args, "generation_num_beams", 1),
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Calculate loss if we have labels
            loss = None
            if has_labels:
                with torch.no_grad():
                    outputs = model(**inputs)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            # Return generated tokens for metrics computation
            labels = inputs.get("labels", None)
            return (loss, generated_tokens, labels)
        else:
            # Use standard prediction for loss-only evaluation
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


class CustomSFTTrainer(SFTTrainer):
    """Custom SFT Trainer with distillation and custom loss support."""

    def __init__(self, *args, custom_loss_fn=None, teacher_model=None, distill_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss_fn = custom_loss_fn
        self.teacher_model = teacher_model
        self.distill_config = distill_config or {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.teacher_model is not None:
            # Distillation loss
            outputs = model(**inputs)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)

            # Compute distillation loss (KL divergence + CE)
            temperature = self.distill_config.get("temperature", 3.0)
            alpha = self.distill_config.get("alpha", 0.7)

            # KL divergence loss between student and teacher
            kl_loss = F.kl_div(
                F.log_softmax(outputs.logits / temperature, dim=-1),
                F.softmax(teacher_outputs.logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)

            # Standard CE loss with ground truth
            ce_loss = (
                outputs.loss
                if hasattr(outputs, "loss")
                else F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), inputs["labels"].view(-1))
            )

            # Weighted combination
            loss = alpha * kl_loss + (1 - alpha) * ce_loss

            return (loss, outputs) if return_outputs else loss

        elif self.custom_loss_fn:
            outputs = model(**inputs)
            loss = self.custom_loss_fn(outputs, inputs)
            return (loss, outputs) if return_outputs else loss

        return super().compute_loss(model, inputs, return_outputs, **kwargs)


@with_sweep
def train(config):
    logger.info("Starting SFT training...")
    if isinstance(config, dict):
        config = LLMTrainingParams(**config)

    train_data, valid_data = utils.process_input_data(config)

    # Validate required columns
    utils.validate_required_columns(train_data, [config.text_column], "SFT", "training")
    if valid_data is not None:
        utils.validate_required_columns(valid_data, [config.text_column], "SFT", "validation")

    tokenizer = utils.get_tokenizer(config)
    train_data, valid_data = utils.process_data_with_chat_template(config, tokenizer, train_data, valid_data)

    logging_steps = utils.configure_logging_steps(config, train_data, valid_data)
    training_args = utils.configure_training_args(config, logging_steps)
    config = utils.configure_block_size(config, tokenizer)

    # SFTConfig-specific parameters
    # Use 'text' column if chat template was applied (creates 'text' from source column)
    # Otherwise use the configured text_column
    if config.chat_template and "text" in train_data.column_names:
        training_args["dataset_text_field"] = "text"
    else:
        training_args["dataset_text_field"] = config.text_column

    # Enable generation during evaluation if BLEU/ROUGE metrics are requested
    generation_metrics = []
    if hasattr(config, "custom_metrics") and config.custom_metrics:
        # Parse custom metrics list from config
        if isinstance(config.custom_metrics, str):
            metrics_list = json.loads(config.custom_metrics)
        else:
            metrics_list = config.custom_metrics

        # Check for generation metrics
        gen_metric_names = ["bleu", "sacrebleu", "rouge", "meteor", "bertscore", "bleurt"]
        generation_metrics = [m for m in metrics_list if m.lower() in gen_metric_names]

        if generation_metrics:
            logger.info(f"Generation metrics requested: {generation_metrics}. Enabling generation during evaluation.")
            # Note: We'll handle this with a custom trainer class below

    # TODO: REFACTOR configure_training_args() to accept trainer_type parameter
    # Currently it sets remove_unused_columns=False globally for DPO/ORPO compatibility,
    # but SFT needs remove_unused_columns=True to avoid collation errors when using dataset_text_field.
    # This override is a temporary fix until configure_training_args() is refactored to handle
    # trainer-specific settings properly instead of one-size-fits-all defaults.
    training_args["remove_unused_columns"] = True

    # Handle TRL API changes: detect which parameter SFTConfig supports
    import inspect

    sig = inspect.signature(SFTConfig.__init__)
    if "max_seq_length" in sig.parameters:
        training_args["max_seq_length"] = config.block_size
        logger.info("Using max_seq_length for SFTConfig (TRL <= 0.9.x)")
    elif "max_length" in sig.parameters:
        training_args["max_length"] = config.block_size
        logger.info("Using max_length for SFTConfig (TRL >= 0.10.x)")
    else:
        raise ValueError("SFTConfig doesn't support max_seq_length or max_length - incompatible TRL version")

    # Smart packing defaults based on hardware and config
    if hasattr(config, "packing") and config.packing is not None:
        # User explicitly set packing
        training_args["packing"] = config.packing
        logger.info(f"Packing explicitly set to: {config.packing}")
    else:
        # Smart defaults: enable packing for CUDA with flash attention, disable for MPS/CPU
        if torch.cuda.is_available() and config.use_flash_attention_2:
            training_args["packing"] = True
            logger.info("Packing enabled (CUDA with flash attention)")
        elif torch.backends.mps.is_available():
            # MPS doesn't support flash attention, packing would be slow
            training_args["packing"] = False
            logger.info("Packing disabled (MPS - no flash attention support)")
        else:
            # CPU or other devices - disable packing
            training_args["packing"] = False
            logger.info("Packing disabled (CPU/other device)")

    # ============================================================================
    # Response-Only Loss (SFT best practice)
    # ----------------------------------------------------------------------------
    # Only compute loss on assistant/model responses, not on prompts/instructions.
    # This prevents the model from overfitting to system prompts and echoing them.
    #
    # Two approaches supported:
    # 1. TRL native: assistant_only_loss=True (requires {% generation %} tags in chat template)
    # 2. Fallback: Add completion_mask to dataset (works with any model)
    # ============================================================================
    use_completion_mask = False  # Flag to add completion_mask during tokenization

    if getattr(config, "response_only_loss", True) and config.chat_template:
        # Check if chat template supports native generation tags
        if utils.has_generation_tags(tokenizer):
            training_args["assistant_only_loss"] = True
            logger.info("Response-only loss enabled via TRL native assistant_only_loss=True")
        else:
            # Fall back to completion_mask approach
            response_template = utils.get_response_template(tokenizer)
            if response_template:
                use_completion_mask = True
                logger.info("Response-only loss enabled via completion_mask")
                logger.info(f"Detected response template: {repr(response_template)}")
            else:
                logger.warning(
                    "Could not detect response template for this model. "
                    "Training will compute loss on entire sequence (including prompts). "
                    "Set response_only_loss=False to suppress this warning."
                )
    elif not getattr(config, "response_only_loss", True):
        logger.info("Response-only loss disabled by config (response_only_loss=False)")
    else:
        # No chat template - likely pretraining/completion style
        logger.info("No chat template - computing loss on entire sequence")

    # Add completion_mask to dataset if needed (for models without {% generation %} tags)
    # Note: process_data_with_chat_template already adds completion_mask when chat_template is set,
    # so we only need to add it here for non-chat-template cases
    if use_completion_mask and "completion_mask" not in train_data.column_names:
        response_template = utils.get_response_template(tokenizer)
        # Use "text" column which contains the formatted chat, not config.text_column which may be "messages"
        text_col = "text" if "text" in train_data.column_names else config.text_column
        train_data = utils.add_completion_mask(train_data, tokenizer, response_template, text_col)
        if valid_data is not None and "completion_mask" not in valid_data.column_names:
            valid_data = utils.add_completion_mask(valid_data, tokenizer, response_template, text_col)

    args = SFTConfig(**training_args)

    model = utils.get_model(config, tokenizer)

    if config.peft:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=utils.get_target_modules(config),
        )

    # ============================================================================
    # EXPERIMENTAL: Advanced Mode with ForwardBackwardPipeline
    # ----------------------------------------------------------------------------
    # Status: Working but experimental (SFT trainer only)
    # Security: Custom loss functions require AUTOTRAIN_ALLOW_CUSTOM_CODE=1
    # Features: Manual optimizer control, sampling during training, checkpointing
    # ============================================================================
    # Check if advanced mode is enabled
    if hasattr(config, "use_forward_backward") and config.use_forward_backward:
        logger.info("Advanced Mode: Using ForwardBackwardPipeline for low-level control")
        from autotrain.trainers.rl.forward_backward import ForwardBackwardPipeline

        # Create pipeline
        pipeline = ForwardBackwardPipeline(
            model=model,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Run custom training loop with direct API access
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        global_step = 0

        for epoch in range(int(config.num_train_epochs)):
            for batch_idx, batch in enumerate(train_data):
                global_step += 1

                # Use specified loss function
                loss_fn = config.forward_backward_loss_fn or "cross_entropy"

                # Execute forward-backward
                if loss_fn == "custom" and config.forward_backward_custom_fn:
                    import os

                    if not os.environ.get("AUTOTRAIN_ALLOW_CUSTOM_CODE"):
                        raise ValueError(
                            "Custom loss functions via eval() are disabled for security.\n\n"
                            "To enable (ONLY in trusted environments):\n"
                            "  export AUTOTRAIN_ALLOW_CUSTOM_CODE=1\n\n"
                            "WARNING: This allows arbitrary code execution!"
                        )
                    logger.warning("⚠️  SECURITY: Executing user-provided code via eval()")
                    custom_fn = eval(config.forward_backward_custom_fn)
                    result = pipeline.forward_backward_custom(
                        input_ids=batch["input_ids"],
                        custom_loss_fn=custom_fn,
                        attention_mask=batch.get("attention_mask"),
                    ).result()
                else:
                    result = pipeline.forward_backward(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch.get("labels", batch["input_ids"]),
                        loss_fn=loss_fn,
                    ).result()

                logger.info(f"Step {global_step} - Loss: {result.loss:.4f}")

                # Manual optimizer control
                if config.manual_optimizer_control:
                    if global_step % config.optimizer_step_frequency == 0:
                        pipeline.optim_step(optimizer).result()
                else:
                    pipeline.optim_step(optimizer).result()

                # Manual sampling
                if hasattr(config, "sample_every_n_steps") and config.sample_every_n_steps > 0:
                    if global_step % config.sample_every_n_steps == 0:
                        prompts = json.loads(config.sample_prompts or '["Hello"]')
                        for prompt_text in prompts[:3]:  # Sample first 3 prompts
                            prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").squeeze()
                            sample_result = pipeline.sample(
                                prompt=prompt_ids,
                                max_tokens=50,
                                temperature=config.sample_temperature,
                            )
                            logger.info(f"Sample: {tokenizer.decode(sample_result['tokens'])}")

                # Checkpointing
                if config.manual_checkpoint_control and config.save_state_every_n_steps > 0:
                    if global_step % config.save_state_every_n_steps == 0:
                        pipeline.save_state(f"step_{global_step}")

        utils.save_model(config, model, tokenizer)
        return pipeline  # Return pipeline instead of trainer

    logger.info("creating trainer")
    callbacks = utils.get_callbacks(
        config, train_data=train_data, valid_data=valid_data, model=model, tokenizer=tokenizer
    )

    # Set up compute_metrics if custom metrics are specified
    compute_metrics = None
    if hasattr(config, "custom_metrics") and config.custom_metrics:
        # Parse custom metrics list from config
        if isinstance(config.custom_metrics, str):
            custom_metrics_list = json.loads(config.custom_metrics)
        else:
            custom_metrics_list = config.custom_metrics

        # Separate callback-based metrics from compute_metrics-based ones
        callback_metrics = []
        compute_metrics_list = []

        for metric in custom_metrics_list:
            # These metrics need access to logs/state, so use callback
            if metric in ["perplexity", "entropy", "loss_variance"]:
                callback_metrics.append(metric)
            else:
                compute_metrics_list.append(metric)

        # Add callback for metrics that need log access
        if callback_metrics:
            from autotrain.trainers.common_metrics import CustomMetricsCallback

            metrics_callback = CustomMetricsCallback(callback_metrics, tokenizer=tokenizer)
            callbacks.append(metrics_callback)
            logger.info(f"Added CustomMetricsCallback for metrics: {callback_metrics}")

        # Get compute_metrics for remaining metrics
        if compute_metrics_list:
            compute_metrics = utils.get_sft_metrics(compute_metrics_list, tokenizer)
            logger.info(f"Using compute_metrics for SFT: {compute_metrics_list}")

    # Add predict_with_generate if we have generation metrics
    if generation_metrics and valid_data is not None:
        args.predict_with_generate = True
        logger.info("Enabled predict_with_generate for generation metrics")

    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )

    # Check if distillation or custom loss is specified
    if (hasattr(config, "use_distillation") and config.use_distillation) or (
        hasattr(config, "custom_loss") and config.custom_loss
    ):
        from autotrain.trainers.losses import CompositeLoss, KLDivergenceLoss

        # Initialize teacher model for distillation if needed
        teacher_model = None
        distill_config = None

        if hasattr(config, "use_distillation") and config.use_distillation:
            if hasattr(config, "teacher_model") and config.teacher_model:
                logger.info(f"Loading teacher model for distillation: {config.teacher_model}")
                from transformers import AutoModelForCausalLM

                from autotrain.utils import get_model_loading_kwargs, maybe_move_to_mps

                # Load teacher model with same device configuration
                teacher_kwargs = get_model_loading_kwargs(
                    token=config.token, fp16_if_cuda=True, trust_remote_code=True
                )
                teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model, **teacher_kwargs)
                teacher_model = maybe_move_to_mps(teacher_model, teacher_kwargs)
                teacher_model.eval()

                # Distillation config
                distill_config = {
                    "temperature": getattr(config, "distill_temperature", 3.0),
                    "alpha": getattr(config, "distill_alpha", 0.7),
                }
                logger.info(
                    f"Distillation config: temperature={distill_config['temperature']}, alpha={distill_config['alpha']}"
                )
            else:
                logger.warning(
                    "use_distillation is True but no teacher_model specified, falling back to standard training"
                )

        # Build custom loss if specified (and not using distillation)
        custom_loss = None
        if not teacher_model and hasattr(config, "custom_loss") and config.custom_loss:
            loss_type = config.custom_loss.lower()
            if loss_type == "composite":
                # Composite loss with weights
                weights = (
                    json.loads(config.custom_loss_weights)
                    if hasattr(config, "custom_loss_weights") and config.custom_loss_weights
                    else [1.0]
                )
                custom_loss = CompositeLoss(losses=[KLDivergenceLoss()], weights=weights)
            elif loss_type == "kl":
                custom_loss = KLDivergenceLoss()

        if teacher_model or custom_loss:
            trainer = CustomSFTTrainer(
                **trainer_args,
                train_dataset=train_data,
                eval_dataset=valid_data,
                peft_config=peft_config if config.peft else None,
                processing_class=tokenizer,
                teacher_model=teacher_model,
                distill_config=distill_config,
                custom_loss_fn=lambda outputs, inputs: (
                    custom_loss(outputs.logits, inputs.get("labels")) if custom_loss else None
                ),
            )
        elif generation_metrics and valid_data is not None:
            # Use GenerativeSFTTrainer for generation metrics
            trainer = GenerativeSFTTrainer(
                **trainer_args,
                train_dataset=train_data,
                eval_dataset=valid_data,
                peft_config=peft_config if config.peft else None,
                processing_class=tokenizer,
            )
        else:
            trainer = SFTTrainer(
                **trainer_args,
                train_dataset=train_data,
                eval_dataset=valid_data,
                peft_config=peft_config if config.peft else None,
                processing_class=tokenizer,
            )
    else:
        if generation_metrics and valid_data is not None:
            # Use GenerativeSFTTrainer for generation metrics
            trainer = GenerativeSFTTrainer(
                **trainer_args,
                train_dataset=train_data,
                eval_dataset=valid_data,
                peft_config=peft_config if config.peft else None,
                processing_class=tokenizer,
            )
        else:
            trainer = SFTTrainer(
                **trainer_args,
                train_dataset=train_data,
                eval_dataset=valid_data if config.valid_split is not None else None,
                peft_config=peft_config if config.peft else None,
                processing_class=tokenizer,
            )

    trainer.remove_callback(PrinterCallback)
    trainer.train(resume_from_checkpoint=utils.get_resume_checkpoint(config))
    utils.post_training_steps(config, trainer)
    return trainer
