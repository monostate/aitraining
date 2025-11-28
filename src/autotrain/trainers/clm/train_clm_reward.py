from functools import partial

import torch
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForSequenceClassification, BitsAndBytesConfig
from transformers.trainer_callback import PrinterCallback
from trl import RewardConfig, RewardTrainer

from autotrain import logger
from autotrain.trainers.clm import utils
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.clm.sweep_utils import with_sweep
from autotrain.trainers.common import ALLOW_REMOTE_CODE


@with_sweep
def train(config):
    """
    Train a reward model for RLHF/PPO training.

    ⚠️  IMPORTANT: This trainer produces a DIFFERENT model type than other LLM trainers:

    OUTPUT: AutoModelForSequenceClassification
    - Returns a scalar score for a given text (not text generation)
    - Used to provide rewards during PPO/RLHF training
    - Cannot be used as a normal LLM for text generation

    INPUT: Preference data with 'chosen' and 'rejected' text columns

    USAGE: After training, use this model with PPO via:
           --trainer ppo --rl-reward-model-path path/to/this/model

    This is Step 1 of a 2-step RLHF pipeline. Step 2 is PPO training.
    """
    logger.info("Starting Reward Model training...")
    logger.info("Note: This trains a scoring model for RLHF, not a text generation model.")

    if isinstance(config, dict):
        config = LLMTrainingParams(**config)
    train_data, valid_data = utils.process_input_data(config)

    # Validate required columns
    required_columns = [config.prompt_text_column, config.text_column, config.rejected_text_column]
    utils.validate_required_columns(train_data, required_columns, "Reward", "training")
    if valid_data is not None:
        utils.validate_required_columns(valid_data, required_columns, "Reward", "validation")

    tokenizer = utils.get_tokenizer(config)
    train_data, valid_data = utils.process_data_with_chat_template(config, tokenizer, train_data, valid_data)

    logging_steps = utils.configure_logging_steps(config, train_data, valid_data)
    training_args = utils.configure_training_args(config, logging_steps)
    config = utils.configure_block_size(config, tokenizer)
    training_args["max_length"] = config.block_size
    args = RewardConfig(**training_args)

    logger.info("loading model config...")
    model_config = AutoConfig.from_pretrained(
        config.model,
        token=config.token,
        trust_remote_code=ALLOW_REMOTE_CODE,
        use_cache=config.disable_gradient_checkpointing,
    )

    model_config.num_labels = 1
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.pad_token = tokenizer.pad_token

    logger.info("loading model...")
    if config.peft:
        if config.quantization == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
        elif config.quantization == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None

        model_kwargs = {
            "config": model_config,
            "token": config.token,
            "quantization_config": bnb_config,
            "trust_remote_code": ALLOW_REMOTE_CODE,
        }
        # Only pass use_flash_attention_2 if enabled (many models don't support it)
        if config.use_flash_attention_2:
            model_kwargs["use_flash_attention_2"] = True

        model = AutoModelForSequenceClassification.from_pretrained(config.model, **model_kwargs)
    else:
        model_kwargs = {
            "config": model_config,
            "token": config.token,
            "trust_remote_code": ALLOW_REMOTE_CODE,
        }
        # Only pass use_flash_attention_2 if enabled (many models don't support it)
        if config.use_flash_attention_2:
            model_kwargs["use_flash_attention_2"] = True

        model = AutoModelForSequenceClassification.from_pretrained(config.model, **model_kwargs)

    logger.info(f"model dtype: {model.dtype}")
    model.resize_token_embeddings(len(tokenizer))

    if config.peft:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=utils.get_target_modules(config),
            modules_to_save=["score"],  # Save classification head with adapter
        )

    reward_proc = partial(utils.preprocess_reward, tokenizer=tokenizer)
    train_data = train_data.map(
        reward_proc,
        batched=True,
        num_proc=4,
        desc="Running tokenizer on train dataset",
    )
    train_data = train_data.filter(
        lambda x: len(x["input_ids_chosen"]) <= config.block_size and len(x["input_ids_rejected"]) <= config.block_size
    )
    if config.valid_split is not None:
        valid_data = valid_data.map(
            reward_proc,
            batched=True,
            num_proc=4,
            desc="Running tokenizer on validation dataset",
        )
        valid_data = valid_data.filter(
            lambda x: len(x["input_ids_chosen"]) <= config.block_size
            and len(x["input_ids_rejected"]) <= config.block_size
        )

    logger.info("creating trainer")
    callbacks = utils.get_callbacks(
        config, train_data=train_data, valid_data=valid_data, model=model, tokenizer=tokenizer
    )
    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks,
    )
    trainer = RewardTrainer(
        **trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        peft_config=peft_config if config.peft else None,
        processing_class=tokenizer,
    )

    trainer.remove_callback(PrinterCallback)
    trainer.train()

    # Ensure config.use_cache is set before saving
    trainer.model.config.use_cache = True

    # post_training_steps will handle saving the model and config
    utils.post_training_steps(config, trainer)

    logger.info("=" * 60)
    logger.info("REWARD MODEL TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("⚠️  IMPORTANT: This model is a SCORER, not a text generator!")
    logger.info("Model type: AutoModelForSequenceClassification")
    logger.info("Output: Scalar reward score (not generated text)")
    logger.info("")
    logger.info("To use this reward model with PPO training:")
    logger.info(f"  autotrain llm --trainer ppo --rl-reward-model-path {config.project_name}")
    logger.info("=" * 60)

    return trainer
