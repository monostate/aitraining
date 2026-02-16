import importlib
import json

import torch
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from autotrain import logger
from autotrain.trainers.clm import utils
from autotrain.trainers.clm.sweep_utils import with_sweep


@with_sweep
def train(config):
    logger.info("Starting GRPO training")

    # 1. Load env module dynamically
    module = importlib.import_module(config.rl_env_module)
    env_cls = getattr(module, config.rl_env_class)

    # 2. Instantiate env
    env_config = json.loads(config.rl_env_config) if config.rl_env_config else {}
    env = env_cls(**env_config)
    logger.info(f"Loaded environment: {config.rl_env_module}.{config.rl_env_class}")

    # 3. Load tokenizer (GRPOTrainer requires left padding)
    tokenizer = utils.get_tokenizer(config)
    tokenizer.padding_side = "left"

    # 4. Build dataset from env (must have "prompt" column)
    train_dataset = env.build_dataset(tokenizer)
    logger.info(f"Built dataset with {len(train_dataset)} examples, columns: {train_dataset.column_names}")

    # 5. Load model
    model = utils.get_model(config, tokenizer)

    # 6. PEFT config
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

    # 7. Build reward function wrapping env.score_episode()
    # GRPOTrainer calls: reward_fn(completions=..., prompts=..., **dataset_columns)
    def reward_fn(completions, prompts, **kwargs):
        rewards = []
        case_indices = kwargs.get("case_idx", list(range(len(completions))))
        for i, completion in enumerate(completions):
            case_idx = case_indices[i] if i < len(case_indices) else i
            with torch.no_grad():
                score = env.score_episode(
                    model=model,
                    tokenizer=tokenizer,
                    completion=completion,
                    case_idx=case_idx,
                )
            rewards.append(float(score))
        return rewards

    # 8. Configure GRPOConfig
    logging_steps = utils.configure_logging_steps(config, train_dataset, None)
    training_args = utils.configure_training_args(config, logging_steps)

    # GRPO-specific args
    training_args["num_generations"] = config.rl_num_generations
    training_args["max_completion_length"] = config.rl_max_new_tokens
    training_args["temperature"] = config.rl_temperature
    training_args["top_p"] = config.rl_top_p
    training_args["top_k"] = config.rl_top_k
    training_args["beta"] = config.rl_kl_coef
    training_args["epsilon"] = config.rl_clip_range
    training_args["loss_type"] = "grpo"

    if config.use_vllm:
        training_args["use_vllm"] = True
        training_args["vllm_mode"] = config.vllm_mode
        training_args["vllm_gpu_memory_utilization"] = config.vllm_gpu_memory_utilization
        if config.vllm_mode == "server" and config.vllm_server_url:
            training_args["vllm_server_url"] = config.vllm_server_url
        training_args["vllm_tensor_parallel_size"] = config.vllm_tensor_parallel_size

    grpo_config = GRPOConfig(**training_args)

    # 9. Create trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 10. Train
    trainer.train()
    utils.post_training_steps(config, trainer)
    return trainer
