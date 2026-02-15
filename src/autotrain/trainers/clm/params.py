from typing import Dict, List, Optional, Union

from pydantic import Field, model_validator

from autotrain import logger
from autotrain.trainers.common import AutoTrainParams


class LLMTrainingParams(AutoTrainParams):
    """
    LLMTrainingParams: Parameters for training a language model using the autotrain library.

    Attributes:
        model (str): Model name to be used for training. Default is "google/gemma-3-270m".
        project_name (str): Name of the project and output directory. Default is "project-name".

        data_path (str): Path to the dataset. Default is "data".
        train_split (str): Configuration for the training data split. Default is "train".
        valid_split (Optional[str]): Configuration for the validation data split. Default is None.
        max_samples (Optional[int]): Maximum number of samples to use from dataset (for testing/debugging). Default is None.
        add_eos_token (bool): Whether to add an EOS token at the end of sequences. Default is True.
        block_size (Union[int, List[int]]): Size of the blocks for training, can be a single integer or a list of integers. Default is -1.
        model_max_length (int): Maximum length of the model input. Auto-detected from model config if left at default (2048).
        padding (Optional[str]): Side on which to pad sequences (left or right). Default is "right".

        trainer (str): Type of trainer to use. Default is "default".
        use_flash_attention_2 (bool): Whether to use flash attention version 2. Default is False.
        attn_implementation (Optional[str]): Attention implementation to use (e.g., 'eager', 'sdpa', 'flash_attention_2'). Default is None.
        packing (Optional[bool]): Pack multiple short sequences into single sequences for efficiency (requires flash_attention_2). Default is None.
        log (str): Logging method for experiment tracking. Default is "wandb".
        disable_gradient_checkpointing (bool): Whether to disable gradient checkpointing. Default is False.
        logging_steps (int): Number of steps between logging events. Default is -1.
        eval_strategy (str): Strategy for evaluation (e.g., 'epoch'). Default is "epoch".
        save_strategy (str): Strategy for saving checkpoints ('epoch', 'steps', or 'no'). Default is "epoch".
        save_steps (int): Number of steps between checkpoint saves (when save_strategy='steps'). Default is 500.
        save_total_limit (int): Maximum number of checkpoints to keep. Default is 1.
        auto_find_batch_size (bool): Whether to automatically find the optimal batch size. Default is False.
        mixed_precision (Optional[str]): Type of mixed precision to use (e.g., 'fp16', 'bf16', or None). Default is None.
        lr (float): Learning rate for training. Default is 3e-5.
        epochs (int): Number of training epochs. Default is 1.
        batch_size (int): Batch size for training. Default is 2.
        warmup_ratio (float): Proportion of training to perform learning rate warmup. Default is 0.1.
        gradient_accumulation (int): Number of steps to accumulate gradients before updating. Default is 4.
        optimizer (str): Optimizer to use for training. Default is "adamw_torch".
        scheduler (str): Learning rate scheduler to use. Default is "linear".
        weight_decay (float): Weight decay to apply to the optimizer. Default is 0.0.
        max_grad_norm (float): Maximum norm for gradient clipping. Default is 1.0.
        seed (int): Random seed for reproducibility. Default is 42.
        chat_template (Optional[str]): Template for chat-based models, options include: None, zephyr, chatml, or tokenizer. Default is None.

        quantization (Optional[str]): Quantization method to use (e.g., 'int4', 'int8', or None). Default is None.
        target_modules (Optional[str]): Target modules for quantization or fine-tuning. Default is "all-linear".
        merge_adapter (bool): Whether to merge PEFT adapters and save full model. Default is True.
        peft (bool): Whether to use Parameter-Efficient Fine-Tuning (PEFT). Default is False.
        lora_r (int): Rank of the LoRA matrices. Default is 16.
        lora_alpha (int): Alpha parameter for LoRA. Default is 32.
        lora_dropout (float): Dropout rate for LoRA. Default is 0.05.

        model_ref (Optional[str]): Reference model for DPO trainer. Default is None.
        dpo_beta (float): Beta parameter for DPO trainer. Default is 0.1.

        max_prompt_length (int): Maximum length of the prompt. Default is 128.
        max_completion_length (Optional[int]): Maximum length of the completion. Default is None.

        prompt_text_column (Optional[str]): Column name for the prompt text. Default is None.
        text_column (str): Column name for the text data. Default is "text".
        rejected_text_column (Optional[str]): Column name for the rejected text data. Default is None.

        save_processed_data (str): Save processed training data: 'auto' (local + hub if from hub, hub-only if >5GB),
            'local', 'hub', 'both', or 'none'. Default is "auto".

        push_to_hub (bool): Whether to push the model to the Hugging Face Hub. Default is False.
        username (Optional[str]): Hugging Face username for authentication. Default is None.
        token (Optional[str]): Hugging Face token for authentication. Default is None.

        unsloth (bool): Whether to use the unsloth library. Default is False.
        distributed_backend (Optional[str]): Backend to use for distributed training. Default is None.
    """

    model: str = Field("google/gemma-3-270m", title="Model name to be used for training")
    project_name: str = Field("project-name", title="Name of the project and output directory")

    # data params
    data_path: str = Field("data", title="Path to the dataset")
    train_split: str = Field("train", title="Configuration for the training data split")
    valid_split: Optional[str] = Field(None, title="Configuration for the validation data split")
    max_samples: Optional[int] = Field(
        None, title="Maximum number of samples to use from dataset (for testing/debugging)"
    )
    add_eos_token: bool = Field(True, title="Whether to add an EOS token at the end of sequences")
    block_size: Union[int, List[int]] = Field(
        -1, title="Size of the blocks for training, can be a single integer or a list of integers"
    )
    model_max_length: int = Field(
        2048,
        title="Maximum length of the model input. Auto-detected from model config if not specified. "
        "Set explicitly to override (e.g., 4096 for longer context).",
    )
    padding: Optional[str] = Field("right", title="Side on which to pad sequences (left or right)")

    # trainer params
    trainer: str = Field("default", title="Type of trainer to use")
    use_flash_attention_2: bool = Field(False, title="Whether to use flash attention version 2")
    attn_implementation: Optional[str] = Field(
        None, title="Attention implementation to use (e.g., 'eager', 'sdpa', 'flash_attention_2')"
    )
    packing: Optional[bool] = Field(
        None, title="Pack multiple short sequences into single sequences for efficiency (requires flash_attention_2)"
    )
    log: str = Field("wandb", title="Logging method for experiment tracking")
    disable_gradient_checkpointing: bool = Field(False, title="Whether to disable gradient checkpointing")
    logging_steps: int = Field(-1, title="Number of steps between logging events")
    eval_strategy: str = Field("epoch", title="Strategy for evaluation (e.g., 'epoch')")
    save_strategy: str = Field("epoch", title="Strategy for saving checkpoints ('epoch', 'steps', or 'no')")
    save_steps: int = Field(500, title="Number of steps between checkpoint saves (when save_strategy='steps')")
    save_total_limit: int = Field(1, title="Maximum number of checkpoints to keep")
    auto_find_batch_size: bool = Field(False, title="Whether to automatically find the optimal batch size")
    mixed_precision: Optional[str] = Field(
        None, title="Type of mixed precision to use (e.g., 'fp16', 'bf16', or None)"
    )
    lr: float = Field(3e-5, title="Learning rate for training")
    epochs: int = Field(1, title="Number of training epochs")
    batch_size: int = Field(2, title="Batch size for training")
    warmup_ratio: float = Field(0.1, title="Proportion of training to perform learning rate warmup")
    gradient_accumulation: int = Field(
        4, title="Number of steps to accumulate gradients before updating (for standard training)"
    )
    optimizer: str = Field("adamw_torch", title="Optimizer to use for training")
    scheduler: str = Field("linear", title="Learning rate scheduler to use")
    weight_decay: float = Field(0.0, title="Weight decay to apply to the optimizer")
    max_grad_norm: float = Field(1.0, title="Maximum norm for gradient clipping")
    seed: int = Field(42, title="Random seed for reproducibility")
    chat_template: Optional[str] = Field(
        None,
        title="Template for chat-based models, options include: None, zephyr, chatml, or tokenizer (default: auto-selected based on trainer type)",
    )
    response_only_loss: bool = Field(
        True,
        title="Compute loss only on assistant/model responses, not on prompts/instructions. "
        "This is the recommended practice for SFT to prevent overfitting to system prompts.",
    )

    # peft
    quantization: Optional[str] = Field(None, title="Quantization method to use (e.g., 'int4', 'int8', or None)")
    target_modules: Optional[str] = Field("all-linear", title="Target modules for quantization or fine-tuning")
    merge_adapter: bool = Field(
        True,
        title="Whether to merge PEFT adapters and save full model (True = easier inference, False = smaller size)",
    )
    peft: bool = Field(False, title="Whether to use Parameter-Efficient Fine-Tuning (PEFT)")
    lora_r: int = Field(16, title="Rank of the LoRA matrices")
    lora_alpha: int = Field(32, title="Alpha parameter for LoRA")
    lora_dropout: float = Field(0.05, title="Dropout rate for LoRA")

    # dpo
    model_ref: Optional[str] = Field(None, title="Reference model for DPO trainer")
    dpo_beta: float = Field(0.1, title="Beta parameter for DPO trainer")

    # orpo + dpo
    max_prompt_length: int = Field(128, title="Maximum length of the prompt")
    max_completion_length: Optional[int] = Field(None, title="Maximum length of the completion")

    # column mappings
    prompt_text_column: Optional[str] = Field(None, title="Column name for the prompt text")
    text_column: str = Field("text", title="Column name for the text data")
    rejected_text_column: Optional[str] = Field(None, title="Column name for the rejected text data")

    # dataset conversion
    auto_convert_dataset: bool = Field(False, title="Automatically detect and convert dataset format to messages")
    use_sharegpt_mapping: bool = Field(
        False, title="Use Unsloth's ShareGPT mapping instead of converting (preserves original format)"
    )
    sharegpt_mapping_enabled: bool = Field(False, title="Internal flag for ShareGPT mapping")
    suggested_chat_template: Optional[str] = Field(None, title="Suggested chat template based on model")
    column_mapping: Optional[Dict[str, str]] = Field(
        None, title="Manual column mapping (e.g., {'user_col': 'instruction', 'assistant_col': 'response'})"
    )
    runtime_mapping: Optional[Dict[str, str]] = Field(
        None, title="Runtime mapping for chat template (e.g., {'role': 'sender', 'content': 'text'})"
    )
    map_eos_token: bool = Field(
        False, title="Map chat template end tokens to EOS token (useful for models that don't know when to stop)"
    )
    conversation_extension: int = Field(
        1, title="Merge N single-turn examples into multi-turn conversations (1 = no merging)", ge=1, le=10
    )
    apply_chat_template: bool = Field(True, title="Apply chat template during dataset conversion")
    save_processed_data: str = Field(
        "auto",
        title="Save processed training data: 'auto' (local + hub if from hub, hub-only if >5GB), "
        "'local', 'hub', 'both', or 'none'",
    )

    # push to hub
    push_to_hub: bool = Field(False, title="Whether to push the model to the Hugging Face Hub")
    username: Optional[str] = Field(None, title="Hugging Face username for authentication")
    token: Optional[str] = Field(None, title="Hugging Face token for authentication")
    repo_id: Optional[str] = Field(
        None,
        title="Full Hugging Face repo ID for push_to_hub (e.g., 'my-org/my-model' or 'username/model-name'). "
        "If not set, defaults to '{username}/{project_name}'. Use this to push to an organization or use a "
        "different repo name than your local project_name.",
    )

    # unsloth
    unsloth: bool = Field(False, title="Whether to use the unsloth library")
    distributed_backend: Optional[str] = Field(None, title="Backend to use for distributed training")

    # ========================================
    # Tinker-Inspired Features
    # ========================================

    # Prompt Distillation
    use_distillation: bool = Field(False, title="Enable prompt distillation training")
    teacher_model: Optional[str] = Field(None, title="Teacher model name or path (required if use_distillation=True)")
    teacher_prompt_template: Optional[str] = Field(
        None, title="Teacher prompt template with {input} placeholder (e.g., 'Answer the following: {input}')"
    )
    student_prompt_template: Optional[str] = Field(
        "{input}", title="Student prompt template (default: just input, set to empty string for no prompt)"
    )
    distill_temperature: float = Field(3.0, title="Distillation temperature for softening distributions (2.0-4.0)")
    distill_alpha: float = Field(0.7, title="Weight for KL loss vs CE loss (0.0-1.0, higher = more KL)")
    distill_max_teacher_length: int = Field(512, title="Maximum length for teacher outputs")

    # Hyperparameter Sweep
    use_sweep: bool = Field(False, title="Enable hyperparameter sweep")
    sweep_backend: Optional[str] = Field("optuna", title="Sweep backend: optuna, ray, grid, or random")
    sweep_n_trials: int = Field(10, title="Number of trials for hyperparameter sweep")
    sweep_metric: Optional[str] = Field("eval_loss", title="Metric to optimize during sweep")
    sweep_direction: Optional[str] = Field("minimize", title="Optimization direction: minimize or maximize")
    sweep_params: Optional[str] = Field(
        None,
        title='Sweep parameters as JSON (e.g., \'{"lr": {"low": 1e-5, "high": 1e-3, "type": "float"}}\')',
    )
    post_trial_script: Optional[str] = Field(
        None,
        title="Shell script/command to run after each sweep trial (receives TRIAL_NUMBER, TRIAL_METRIC_VALUE, TRIAL_IS_BEST env vars)",
    )
    # W&B Native Sweep Integration
    wandb_sweep: bool = Field(False, title="Enable W&B native sweep dashboard (creates aggregated sweep view)")
    wandb_sweep_project: Optional[str] = Field(None, title="W&B project name for sweep (defaults to project_name)")
    wandb_sweep_entity: Optional[str] = Field(None, title="W&B entity (team/username) for sweep")
    wandb_sweep_id: Optional[str] = Field(None, title="Existing W&B sweep ID to continue (skips creating new sweep)")
    wandb_run_id: Optional[str] = Field(
        None, title="W&B run ID to resume (prevents trainer from creating duplicate run)"
    )

    # Enhanced Evaluation
    use_enhanced_eval: bool = Field(False, title="Enable enhanced evaluation metrics")
    eval_metrics: Optional[str] = Field(
        "perplexity",
        title="Comma-separated metrics: perplexity,bleu,rouge,bertscore,accuracy,f1,exact_match,meteor",
    )
    eval_dataset_path: Optional[str] = Field(None, title="Path to evaluation dataset (if different from validation)")
    eval_batch_size: int = Field(8, title="Batch size for evaluation")
    eval_save_predictions: bool = Field(False, title="Save predictions during evaluation")
    eval_benchmark: Optional[str] = Field(None, title="Run standard benchmark: mmlu, hellaswag, arc, truthfulqa")

    # Message Rendering (extending existing chat_template)
    chat_format: Optional[str] = Field(
        None,
        title="Chat format for message rendering: chatml, alpaca, llama, vicuna, zephyr, mistral (overrides chat_template)",
    )
    token_weights: Optional[str] = Field(
        None,
        title='Token-level weights as JSON (e.g., \'{"system": 0.5, "user": 0.0, "assistant": 1.0}\')',
    )

    # ========================================
    # RL Training (PPO)
    # ========================================
    rl_gamma: float = Field(0.99, title="Discount factor for RL (0.9-0.99)")
    rl_gae_lambda: float = Field(0.95, title="GAE lambda for advantage estimation (0.9-0.99)")
    rl_kl_coef: float = Field(0.1, title="KL divergence coefficient (0.01-0.5)")
    rl_value_loss_coef: float = Field(1.0, title="Value loss coefficient (0.5-2.0)")
    rl_clip_range: float = Field(0.2, title="PPO clipping range (0.1-0.3)")
    rl_reward_fn: Optional[str] = Field(None, title="Reward function: default, length_penalty, correctness, custom")
    rl_multi_objective: bool = Field(
        False,
        title="Multi-Objective RL",
        description="Enable multi-objective rewards. When True with rl_env_type='multi_objective', tracks and combines multiple reward components (e.g., correctness + formatting).",
    )
    rl_reward_weights: Optional[str] = Field(
        None,
        title="Reward Weights",
        description="JSON dict of weights for multi-objective rewards. Example: '{\"correctness\": 1.0, \"formatting\": 0.1}'. Used with rl_env_type='multi_objective'.",
    )
    rl_env_type: Optional[str] = Field(
        None,
        title="RL Environment Type",
        description="Type of RL environment: 'text_generation', 'multi_objective', or 'preference_comparison'. Defaults to standard TRL behavior if not specified.",
    )
    rl_env_config: Optional[str] = Field(
        None,
        title="RL Environment Config",
        description="JSON config for RL environment. Required keys vary by env type. For multi_objective: {'reward_components': {...}, 'reward_weights': {...}}. For text_generation: {'stop_sequences': [...]}.",
    )
    rl_reward_model_path: Optional[str] = Field(None, title="Path to reward model for PPO training")
    rl_num_ppo_epochs: int = Field(4, title="Number of PPO epochs per batch")
    rl_chunk_size: int = Field(128, title="PPO training chunk size")
    rl_mini_batch_size: int = Field(8, title="PPO mini-batch size")
    rl_optimize_device_cache: bool = Field(True, title="Optimize PPO device memory cache")
    rl_value_clip_range: float = Field(0.2, title="PPO value function clipping range")
    rl_max_new_tokens: int = Field(128, title="Maximum new tokens to generate during PPO")
    rl_top_k: int = Field(50, title="Top-k sampling for PPO generation")
    rl_top_p: float = Field(1.0, title="Top-p (nucleus) sampling for PPO generation")
    rl_temperature: float = Field(1.0, title="Temperature for PPO generation")

    # ========================================
    # RL Training (GRPO)
    # ========================================
    rl_env_module: Optional[str] = Field(
        None,
        title="Python module path for RL environment (e.g., 'my_envs.hotel_env')",
    )
    rl_env_class: Optional[str] = Field(
        None,
        title="Class name in rl_env_module (e.g., 'HotelEnv')",
    )
    rl_num_generations: int = Field(4, title="Number of completions per prompt for GRPO")
    use_vllm: bool = Field(False, title="Use vLLM for faster generation in GRPO")
    vllm_mode: str = Field("colocate", title="vLLM mode: 'server' or 'colocate'")
    vllm_gpu_memory_utilization: float = Field(0.3, title="Fraction of GPU memory for vLLM (colocate mode)")

    # Custom Losses
    custom_loss: Optional[str] = Field(
        None, title="Custom loss function: composite, kl, ppo, variance, or JSON config"
    )
    custom_loss_weights: Optional[str] = Field(None, title="JSON array of weights for composite loss")

    # Custom Metrics
    custom_metrics: Optional[str] = Field(
        None, title='JSON array of custom metric names to compute (e.g., \'["bleu", "rouge"]\')'
    )

    # Advanced Research Mode
    # advanced_mode removed - redundant with use_forward_backward

    # Forward-Backward Control
    use_forward_backward: bool = Field(False, title="Use manual forward-backward control (advanced)")
    forward_backward_loss_fn: Optional[str] = Field(
        None, title="Loss function for forward_backward: cross_entropy, importance_sampling, ppo, or custom"
    )
    forward_backward_custom_fn: Optional[str] = Field(
        None, title="Python code for custom loss function (if loss_fn=custom)"
    )
    gradient_accumulation_steps: int = Field(
        1,
        title="[ADVANCED] Gradient accumulation for manual forward/backward control (use gradient_accumulation for normal training)",
    )

    # Optimizer Control
    manual_optimizer_control: bool = Field(False, title="Manual control over optimizer steps (advanced)")
    optimizer_step_frequency: int = Field(1, title="Run optimizer every N forward-backward steps")
    grad_clip_value: Optional[float] = Field(1.0, title="Gradient clipping value")

    # Sampling Control
    # manual_sampling removed - redundant, just use sample_every_n_steps > 0
    sample_every_n_steps: int = Field(0, title="Sample from model every N steps (0=disabled)")
    sample_prompts: Optional[str] = Field(None, title="JSON array of prompts for sampling")
    sample_temperature: float = Field(1.0, title="Temperature for manual sampling")
    sample_top_k: int = Field(50, title="Top-k for manual sampling")
    sample_top_p: float = Field(1.0, title="Top-p for manual sampling")

    # Checkpoint Control
    manual_checkpoint_control: bool = Field(False, title="Manual checkpoint save/load control (advanced)")
    save_state_every_n_steps: int = Field(0, title="Save state every N steps (0=disabled)")
    load_state_from: Optional[str] = Field(None, title="Path to load state from")

    # Inference Parameters
    inference_prompts: Optional[str] = Field(None, title="Prompts for inference (file path or comma-separated)")
    inference_max_tokens: int = Field(256, title="Maximum tokens to generate during inference")
    inference_temperature: float = Field(1.0, title="Temperature for inference generation")
    inference_top_p: float = Field(1.0, title="Top-p (nucleus) sampling for inference")
    inference_top_k: int = Field(50, title="Top-k sampling for inference")
    inference_output: Optional[str] = Field(None, title="Output file path for inference results")

    # ========================================
    # Model Validators
    # ========================================

    @model_validator(mode="before")
    @classmethod
    def set_chat_template_defaults(cls, data):
        """Set smart defaults for chat_template based on trainer type."""
        # Check if chat_template was explicitly provided in the input
        if "chat_template" in data:
            # User explicitly set it (even if None or "none"), respect their choice
            # Convert string "none" to None for consistency
            if data["chat_template"] == "none":
                data["chat_template"] = None
            return data

        # chat_template was not provided, auto-select based on trainer
        trainer = data.get("trainer", "default")

        if trainer in ["sft", "dpo", "orpo", "reward"]:
            # These trainers typically work with instruction-tuned models
            # Default to "tokenizer" to use model's built-in chat template
            data["chat_template"] = "tokenizer"
            logger.info(f"Auto-selecting chat_template='tokenizer' for trainer='{trainer}'")
        elif trainer == "default":
            # Default trainer is for continued pretraining/completion
            # No chat template needed
            data["chat_template"] = None
            logger.info(f"No chat_template for trainer='default' (pretraining/completion)")
        # PPO trainer will use reward model's format, so leave as None (don't set it)

        return data

    @model_validator(mode="after")
    def validate_ppo_params(self):
        """Validate PPO-specific parameters are only used with PPO trainer."""
        rl_params = [
            "rl_gamma",
            "rl_gae_lambda",
            "rl_kl_coef",
            "rl_value_loss_coef",
            "rl_clip_range",
            "rl_reward_fn",
            "rl_multi_objective",
            "rl_reward_weights",
            "rl_env_type",
            "rl_env_config",
            "rl_num_ppo_epochs",
            "rl_chunk_size",
            "rl_mini_batch_size",
            "rl_optimize_device_cache",
            "rl_value_clip_range",
            "rl_max_new_tokens",
            "rl_top_k",
            "rl_top_p",
            "rl_temperature",
        ]

        if self.trainer not in ("ppo", "grpo"):
            # Check if any RL params are explicitly set (not default)
            for param in rl_params:
                if hasattr(self, param):
                    value = getattr(self, param)
                    field_info = self.model_fields[param]
                    default = field_info.default
                    # Warn if non-default value is set for non-PPO trainer
                    if value != default and value is not None:
                        logger.warning(
                            f"Parameter '{param}={value}' is only used with trainer=ppo, "
                            f"but trainer={self.trainer}. This parameter will be ignored."
                        )
        return self

    @model_validator(mode="after")
    def validate_grpo_params(self):
        """Validate GRPO-specific parameters."""
        if self.trainer == "grpo":
            if not self.rl_env_module:
                raise ValueError("GRPO trainer requires --rl-env-module (Python module path for the environment)")
            if not self.rl_env_class:
                raise ValueError("GRPO trainer requires --rl-env-class (class name in the environment module)")
        return self

    @model_validator(mode="after")
    def validate_distillation_params(self):
        """Validate distillation parameters are only used when use_distillation=True."""
        distill_params = [
            "teacher_model",
            "teacher_prompt_template",
            "student_prompt_template",
            "distill_temperature",
            "distill_alpha",
            "distill_max_teacher_length",
        ]

        if not self.use_distillation:
            for param in distill_params:
                if hasattr(self, param) and param != "teacher_model":
                    value = getattr(self, param)
                    field_info = self.model_fields[param]
                    default = field_info.default
                    if value != default and value is not None:
                        logger.warning(
                            f"Parameter '{param}={value}' is only used when use_distillation=True. "
                            f"This parameter will be ignored."
                        )

        # If use_distillation=True, teacher_model is required
        if self.use_distillation and not self.teacher_model:
            raise ValueError("When use_distillation=True, you must specify teacher_model")

        return self

    @model_validator(mode="after")
    def validate_dpo_params(self):
        """Validate DPO-specific parameters."""
        if self.trainer in ["dpo", "orpo"]:
            # Validate required columns for DPO/ORPO
            if not self.prompt_text_column:
                raise ValueError(
                    f"{self.trainer.upper()} training requires prompt_text_column to be specified. "
                    "This column should contain the prompt/instruction text."
                )
            if not self.rejected_text_column:
                raise ValueError(
                    f"{self.trainer.upper()} training requires rejected_text_column to be specified. "
                    "This column should contain the rejected response text."
                )
            # text_column is implicitly required as it contains the chosen response
        elif self.trainer not in ["orpo"]:
            # Warn if DPO params set for non-DPO/ORPO trainers
            if self.model_ref is not None:
                logger.warning(
                    f"Parameter 'model_ref' is primarily for DPO/ORPO trainers, "
                    f"but trainer={self.trainer}. This may be ignored."
                )
        return self

    @model_validator(mode="after")
    def validate_ppo_requirements(self):
        """Validate PPO has required parameters."""
        if self.trainer == "ppo":
            if not self.rl_reward_model_path and not self.model_ref:
                raise ValueError(
                    "\n" + "=" * 60 + "\n"
                    "PPO training requires a reward model!\n\n"
                    "You must specify --rl-reward-model-path pointing to a trained reward model.\n\n"
                    "To train a reward model first:\n"
                    "  aitraining llm --trainer reward --data-path preference_data\n\n"
                    "Then use it with PPO:\n"
                    "  aitraining llm --trainer ppo --rl-reward-model-path path/to/reward/model\n" + "=" * 60
                )

            # Validate reward model path exists if provided
            if self.rl_reward_model_path:
                import os

                # Allow either local paths or HuggingFace Hub model IDs (contain "/")
                is_local_path = os.path.exists(self.rl_reward_model_path)
                is_hub_model = "/" in self.rl_reward_model_path
                if not (is_local_path or is_hub_model):
                    raise ValueError(
                        f"Reward model path does not exist: {self.rl_reward_model_path}\n"
                        f"Please train a reward model first or provide a valid path."
                    )

        return self
