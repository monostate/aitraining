"""
PPO Training for AutoTrain Advanced
====================================

Proximal Policy Optimization (PPO) trainer integration for CLI.
Uses TRL's PPOTrainer for consistency with other trainers.

Requirements:
- A trained reward model (via --rl-reward-model-path or --model-ref)
- Reward model validation happens at config time, not training time
"""

import json

import torch
from peft import LoraConfig
from transformers.trainer_callback import PrinterCallback
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from autotrain import logger
from autotrain.trainers.clm import utils
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.clm.sweep_utils import with_sweep


def create_rl_environment(config, tokenizer, prompts):
    """
    Create an RL environment based on config.rl_env_type.

    Args:
        config: LLMTrainingParams with rl_env_type and rl_env_config
        tokenizer: Tokenizer for the environment
        prompts: List of prompts for the environment

    Returns:
        RLEnvironment instance or None if no environment requested

    Raises:
        ValueError: If rl_env_type is invalid or rl_env_config is malformed
    """
    if not config.rl_env_type:
        return None

    # Import environment classes
    from autotrain.trainers.rl.environments import MultiObjectiveRewardEnv, PreferenceComparisonEnv, TextGenerationEnv

    # Parse environment config
    env_config = {}
    if config.rl_env_config:
        try:
            # Handle both JSON string and dict
            if isinstance(config.rl_env_config, str):
                env_config = json.loads(config.rl_env_config)
            elif isinstance(config.rl_env_config, dict):
                env_config = config.rl_env_config
            else:
                raise ValueError(f"rl_env_config must be a JSON string or dict, got {type(config.rl_env_config)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse rl_env_config as JSON: {e}")

    # Map env type to class
    env_type_map = {
        "text_generation": TextGenerationEnv,
        "multi_objective": MultiObjectiveRewardEnv,
        "preference_comparison": PreferenceComparisonEnv,
    }

    env_type = config.rl_env_type.lower()
    if env_type not in env_type_map:
        raise ValueError(f"Invalid rl_env_type: {config.rl_env_type}. " f"Must be one of: {list(env_type_map.keys())}")

    env_class = env_type_map[env_type]

    # Build common args (without temperature - added per-env)
    common_args = {
        "tokenizer": tokenizer,
        "prompts": prompts or ["Generate text:"],  # Default prompt if none provided
        "max_length": getattr(config, "rl_max_new_tokens", 256),
    }

    # Build environment based on type
    if env_type == "text_generation":
        # Basic text generation environment
        logger.info("Creating TextGenerationEnv")

        # Extract reward function if specified
        reward_fn = env_config.get("reward_fn")
        if reward_fn:
            # If it's a string, try to import it
            if isinstance(reward_fn, str):
                logger.warning(
                    f"Custom reward function '{reward_fn}' as string not yet supported. "
                    "Using default reward function."
                )
                reward_fn = None

        # TextGenerationEnv supports temperature
        env_args = common_args.copy()
        if hasattr(config, "rl_temperature"):
            env_args["temperature"] = config.rl_temperature

        env = env_class(reward_fn=reward_fn, stop_sequences=env_config.get("stop_sequences", []), **env_args)

    elif env_type == "multi_objective":
        # Multi-objective reward environment
        logger.info("Creating MultiObjectiveRewardEnv")

        # Parse reward components
        reward_components_config = env_config.get("reward_components", {})
        if not reward_components_config:
            raise ValueError(
                "multi_objective environment requires 'reward_components' in rl_env_config. "
                'Example: {"reward_components": {"correctness": {...}, "formatting": {...}}}'
            )

        # Build reward component functions
        reward_components = {}
        for name, component_config in reward_components_config.items():
            # For now, use simple placeholder reward functions
            # In the future, this could load custom functions
            def make_reward_fn(component_name, component_cfg):
                def reward_fn(prompt, generated, full_text):
                    # Placeholder logic - could be extended
                    if component_cfg.get("type") == "length":
                        return len(generated.split()) / 100.0
                    elif component_cfg.get("type") == "keyword":
                        keywords = component_cfg.get("keywords", [])
                        return sum(1.0 for kw in keywords if kw in generated)
                    else:
                        # Default simple reward
                        return 0.5 if len(generated) > 0 else 0.0

                return reward_fn

            reward_components[name] = make_reward_fn(name, component_config)

        # Parse reward weights
        reward_weights = env_config.get("reward_weights")
        if not reward_weights and config.rl_reward_weights:
            # Try to get from config.rl_reward_weights
            try:
                if isinstance(config.rl_reward_weights, str):
                    reward_weights = json.loads(config.rl_reward_weights)
                else:
                    reward_weights = config.rl_reward_weights
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse rl_reward_weights: {e}")

        # MultiObjectiveRewardEnv supports temperature
        env_args = common_args.copy()
        if hasattr(config, "rl_temperature"):
            env_args["temperature"] = config.rl_temperature

        env = env_class(reward_components=reward_components, reward_weights=reward_weights, **env_args)

    elif env_type == "preference_comparison":
        # Preference comparison environment
        logger.info("Creating PreferenceComparisonEnv")

        # This environment needs a preference model or human feedback function
        preference_model = env_config.get("preference_model")
        human_feedback_fn = env_config.get("human_feedback_fn")

        # PreferenceComparisonEnv does NOT support temperature - only use common_args
        env = env_class(preference_model=preference_model, human_feedback_fn=human_feedback_fn, **common_args)

    logger.info(f"Created {env_type} environment with config: {env_config}")
    return env


def create_env_aware_reward_fn(env, reward_model, tokenizer):
    """
    Create a reward function that integrates the custom environment.

    Args:
        env: RLEnvironment instance or None
        reward_model: Trained reward model or None
        tokenizer: Tokenizer instance

    Returns:
        Callable reward function for PPO
    """
    if env is None:
        # No custom environment, use standard reward model or default
        if reward_model is not None:

            def reward_fn(samples):
                with torch.no_grad():
                    inputs = tokenizer(
                        samples, padding=True, truncation=True, return_tensors="pt", pad_to_multiple_of=8
                    )
                    device = next(reward_model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = reward_model(**inputs)
                    logits = outputs.logits.squeeze()
                    if logits.dim() == 0:
                        return [logits.cpu().item()]
                    return logits.cpu().tolist()

            return reward_fn
        else:
            # Default simple reward
            def reward_fn(samples):
                rewards = []
                for sample in samples:
                    text = (
                        tokenizer.decode(sample, skip_special_tokens=True)
                        if isinstance(sample, torch.Tensor)
                        else sample
                    )
                    length_penalty = -0.01 * max(0, len(text.split()) - 50)
                    coherence_bonus = 0.5 if any(p in text for p in ".!?") else 0.0
                    rewards.append(length_penalty + coherence_bonus)
                return rewards

            return reward_fn

    # Use environment's reward function
    def env_reward_fn(samples):
        """
        Reward function that uses the custom environment.

        Note: In TRL's PPOTrainer, this function is called with completed generated text
        samples. The environment is used here primarily as a reward computation module
        rather than a stateful RL environment. We reset for each sample to get a fresh
        prompt, which is appropriate for this batch-based reward computation pattern.
        """
        rewards = []
        metrics = {}

        for sample in samples:
            # Decode sample to text
            text = tokenizer.decode(sample, skip_special_tokens=True) if isinstance(sample, torch.Tensor) else sample

            # Reset environment to get a prompt for this sample
            # Note: This is appropriate for TRL's batch reward computation pattern
            try:
                obs = env.reset()
                if hasattr(obs, "prompt"):
                    prompt = obs.prompt
                else:
                    # Fallback if observation doesn't have prompt attribute
                    prompt = ""
                    logger.warning("Environment observation missing 'prompt' attribute")
            except Exception as e:
                logger.warning(f"Environment reset failed: {e}. Using empty prompt.")
                prompt = ""

            # Extract generated portion (remove prompt)
            if text.startswith(prompt):
                generated = text[len(prompt) :]
            else:
                generated = text

            # Get reward from environment
            # For TextGenerationEnv and MultiObjectiveRewardEnv, we call their reward function
            if hasattr(env, "compute_multi_objective_reward"):
                # Multi-objective environment
                reward, component_rewards = env.compute_multi_objective_reward(prompt, generated, text)
                # Track component metrics
                for comp_name, comp_reward in component_rewards.items():
                    metrics.setdefault(f"env_reward_{comp_name}", []).append(comp_reward)
            elif hasattr(env, "reward_fn"):
                # Text generation environment
                reward = env.reward_fn(prompt=prompt, generated=generated, full_text=text)
            else:
                # Fallback
                reward = 0.0

            rewards.append(reward)

        # Log environment metrics if available
        if metrics:
            for metric_name, metric_values in metrics.items():
                avg_value = sum(metric_values) / len(metric_values)
                logger.debug(f"{metric_name}: {avg_value:.4f}")

        return rewards

    return env_reward_fn


@with_sweep
def train(config):
    """PPO training entry point for AutoTrain CLI."""
    logger.info("Starting PPO training...")

    if isinstance(config, dict):
        config = LLMTrainingParams(**config)

    # Note: Reward model validation now happens at config validation time
    # See params.py::validate_ppo_requirements()
    logger.info(
        "PPO training initialized. Using reward model from: %s", config.rl_reward_model_path or config.model_ref
    )

    # Log custom environment usage if configured
    if config.rl_env_type:
        logger.info(f"Custom RL environment requested: {config.rl_env_type}")
        if config.rl_multi_objective:
            logger.info("Multi-objective rewards enabled")
    else:
        logger.info("Using default TRL PPO behavior (no custom environment)")

    # Load data and tokenizer
    train_data, valid_data = utils.process_input_data(config)

    # Validate required columns
    utils.validate_required_columns(train_data, [config.text_column], "PPO", "training")
    # PPO doesn't use validation data but check if provided
    if valid_data is not None:
        logger.info("Note: PPO trainer doesn't use validation data, but it was provided")

    tokenizer = utils.get_tokenizer(config)
    train_data, valid_data = utils.process_data_with_chat_template(config, tokenizer, train_data, valid_data)

    # Extract prompts from training data for custom environment
    text_col = config.text_column if config.text_column in train_data.column_names else "text"
    prompts = train_data[text_col][:100] if len(train_data) > 0 else []  # Use first 100 prompts

    # Create custom RL environment if requested
    custom_env = None
    try:
        custom_env = create_rl_environment(config, tokenizer, prompts)
        if custom_env:
            logger.info(f"Successfully created custom environment: {type(custom_env).__name__}")
    except ValueError as e:
        logger.error(f"Failed to create custom environment: {e}")
        raise

    # Configure training args similar to other trainers
    logging_steps = utils.configure_logging_steps(config, train_data, valid_data)
    training_args = utils.configure_training_args(config, logging_steps)
    config = utils.configure_block_size(config, tokenizer)

    # PPO specific configuration
    training_args["learning_rate"] = config.lr
    training_args["batch_size"] = config.batch_size
    training_args["num_ppo_epochs"] = getattr(config, "rl_num_ppo_epochs", 4)
    training_args["mini_batch_size"] = getattr(config, "rl_mini_batch_size", 1)
    training_args["gradient_accumulation_steps"] = config.gradient_accumulation
    # Disable bf16 if not supported
    training_args["bf16"] = False
    training_args["fp16"] = False
    # PPO hyperparameters
    training_args["kl_coef"] = getattr(config, "rl_kl_coef", 0.1)
    training_args["gamma"] = getattr(config, "rl_gamma", 0.99)
    training_args["lam"] = getattr(config, "rl_gae_lambda", 0.95)
    training_args["cliprange"] = getattr(config, "rl_clip_range", 0.2)
    training_args["cliprange_value"] = getattr(config, "rl_value_clip_range", 0.2)
    training_args["vf_coef"] = getattr(config, "rl_value_loss_coef", 1.0)
    training_args["max_grad_norm"] = config.max_grad_norm

    # Generation parameters for PPO sampling
    generation_kwargs = {
        "min_length": -1,
        "max_new_tokens": getattr(config, "rl_max_new_tokens", 256),
        "top_k": getattr(config, "rl_top_k", 0),
        "top_p": getattr(config, "rl_top_p", 1.0),
        "do_sample": True,
        "temperature": getattr(config, "rl_temperature", 1.0),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Create PPO config
    # Disable evaluation if no validation data
    training_args["do_eval"] = valid_data is not None
    if valid_data is None:
        training_args["eval_strategy"] = "no"
        # Disable sample generation which requires eval_dataloader
        training_args["num_sample_generations"] = 0
    ppo_config = PPOConfig(
        **training_args,
        # log_with="tensorboard",  # Not supported in current version
        # project_kwargs={"logging_dir": config.project_name},
    )

    # Load models
    model = utils.get_model(config, tokenizer)

    # Configure PEFT if needed
    peft_config = None
    if config.peft:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=utils.get_target_modules(config),
        )

    # For PPO, we need a reference model
    # If using PEFT, ref_model should be None
    # Otherwise, we need a deep copy of the model
    if peft_config is not None:
        ref_model = None
    else:
        import copy

        ref_model = copy.deepcopy(model)

    # Prepare dataset for PPO (needs to return tokenized queries)
    # PPO expects tokenized prompts/queries to generate from
    def prepare_dataset(dataset):
        def tokenize_queries(examples):
            # Get the text column to use
            text_col = config.text_column if config.text_column in dataset.column_names else "text"
            # Truncate to half block size to leave room for generation
            queries = [text[: config.block_size // 2] for text in examples[text_col]]
            # Tokenize the queries
            return tokenizer(queries, truncation=True, max_length=config.block_size // 2, padding=False)

        return dataset.map(tokenize_queries, batched=True, remove_columns=dataset.column_names)

    train_dataset = prepare_dataset(train_data)

    # Load reward model if path is provided
    reward_model = None
    reward_model_path = config.rl_reward_model_path or config.model_ref

    if reward_model_path:
        from transformers import AutoConfig, AutoModelForSequenceClassification

        from autotrain.utils import get_model_loading_kwargs, maybe_move_to_mps

        logger.info(f"Loading reward model from {reward_model_path}")

        # Load reward model config first to ensure num_labels is set
        reward_config = AutoConfig.from_pretrained(reward_model_path, token=config.token)

        # Ensure pad_token is set in config to match tokenizer
        reward_config.pad_token_id = tokenizer.pad_token_id

        # Use existing utilities for device handling
        reward_kwargs = get_model_loading_kwargs(token=config.token, fp16_if_cuda=False, trust_remote_code=True)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path, config=reward_config, **reward_kwargs
        )
        reward_model = maybe_move_to_mps(reward_model, reward_kwargs)
        reward_model.eval()

    # Create environment-aware reward function
    # This integrates custom RL environments if specified, otherwise uses standard reward model
    reward_fn = create_env_aware_reward_fn(custom_env, reward_model, tokenizer)

    # If we have a custom environment, we DON'T pass reward_model to PPOTrainer
    # Instead, we'll set it to None and inject our custom reward function
    ppo_reward_model = None if custom_env else reward_model

    if custom_env:
        logger.info("Using custom environment reward function")
    elif reward_model:
        logger.info("Using trained reward model")
        ppo_reward_model = reward_model
    else:
        logger.info("Using default heuristic reward function")

    # Create PPO trainer with required arguments
    callbacks = utils.get_callbacks(
        config, train_data=train_data, valid_data=valid_data, model=model, tokenizer=tokenizer
    )

    # Set up custom metrics via callbacks if specified
    # PPO doesn't support compute_metrics directly, so we use callbacks
    if hasattr(config, "custom_metrics") and config.custom_metrics:
        # Parse custom metrics list from config
        if isinstance(config.custom_metrics, str):
            import json

            custom_metrics_list = json.loads(config.custom_metrics)
        else:
            custom_metrics_list = config.custom_metrics

        logger.info(f"Setting up custom metrics for PPO via callbacks: {custom_metrics_list}")

        # Use the generic CustomMetricsCallback that handles ANY metric
        from autotrain.trainers.common_metrics import CustomMetricsCallback

        metrics_callback = CustomMetricsCallback(custom_metrics_list, tokenizer=tokenizer)
        callbacks.append(metrics_callback)
        logger.info(f"Added CustomMetricsCallback for PPO with metrics: {custom_metrics_list}")

    # Create value model by wrapping the base model with a value head
    # This works with any causal LM model (GPT-2, LLaMA, etc.)
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model,
        token=config.token,
        trust_remote_code=getattr(config, "trust_remote_code", False),
    )

    # Move value model to same device as main model
    device = next(model.parameters()).device
    value_model = value_model.to(device)

    # Fix compatibility with PPOTrainer
    # PPOTrainer expects to access the base model through base_model_prefix attribute
    # Set base_model_prefix to 'pretrained_model' since that's where the actual model is
    value_model.base_model_prefix = "pretrained_model"

    # Also create a direct reference to pretrained_model's transformer if it exists
    # This handles cases where PPOTrainer might look for nested attributes
    if hasattr(value_model, "pretrained_model"):
        if hasattr(value_model.pretrained_model, "transformer"):
            value_model.transformer = value_model.pretrained_model.transformer
        elif hasattr(value_model.pretrained_model, "model"):
            value_model.model = value_model.pretrained_model.model

    # Add score method to value_model for PPOTrainer compatibility
    # PPOTrainer's get_reward function expects this method
    def value_score_method(self, hidden_states):
        """Score method expected by TRL's PPOTrainer for value model."""
        # AutoModelForCausalLMWithValueHead has a v_head that computes values
        if hasattr(self, "v_head"):
            return self.v_head(hidden_states)
        else:
            # Fallback - this shouldn't happen with AutoModelForCausalLMWithValueHead
            raise AttributeError("Value model doesn't have v_head for scoring")

    # Bind the score method to the value model instance
    value_model.score = value_score_method.__get__(value_model, type(value_model))

    # Prepare eval dataset if validation data exists
    eval_dataset = None
    if valid_data is not None:
        eval_dataset = prepare_dataset(valid_data)

    ppo_trainer = PPOTrainer(
        args=ppo_config,  # Changed from config to args
        model=model,
        ref_model=ref_model,
        reward_model=ppo_reward_model,  # Use None if custom_env, else use reward_model
        train_dataset=train_dataset,  # Now a required argument
        eval_dataset=eval_dataset,  # Only pass eval dataset if we have validation data
        value_model=value_model,  # Proper value model with value head
        processing_class=tokenizer,  # Changed from tokenizer to processing_class
        peft_config=peft_config,
        callbacks=callbacks,
        # PPOTrainer doesn't support compute_metrics - use callbacks instead
    )

    # If using custom environment, monkey-patch the compute_rewards method
    if custom_env:
        logger.info("Injecting custom environment reward function into PPOTrainer")
        # Store original method (if it exists)
        original_compute_rewards = getattr(
            ppo_trainer,
            "_original_compute_rewards",
            ppo_trainer.compute_rewards if hasattr(ppo_trainer, "compute_rewards") else None,
        )

        def custom_compute_rewards(queries, responses, *args, **kwargs):
            """Custom reward computation using our environment."""
            # Combine queries and responses into full text
            batch_rewards = []
            for query_tokens, response_tokens in zip(queries, responses):
                # Decode to text
                query_text = tokenizer.decode(query_tokens, skip_special_tokens=True)
                response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
                full_text = query_text + response_text

                # Get reward from our custom reward_fn
                rewards = reward_fn([full_text])
                batch_rewards.append(rewards[0] if isinstance(rewards, list) else rewards)

            return torch.tensor(batch_rewards, dtype=torch.float32)

        ppo_trainer.compute_rewards = custom_compute_rewards
        logger.info("Custom reward function successfully injected")

    ppo_trainer.remove_callback(PrinterCallback)

    # Training loop - Use TRL's train() method
    logger.info("Starting PPO training...")

    # In newer TRL versions, PPOTrainer uses the standard train() method
    ppo_trainer.train()

    # Save final model
    utils.post_training_steps(config, ppo_trainer)

    return ppo_trainer
