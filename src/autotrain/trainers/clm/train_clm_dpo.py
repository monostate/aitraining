import torch
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_callback import PrinterCallback
from trl import DPOConfig, DPOTrainer

from autotrain import logger
from autotrain.trainers.clm import utils
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.clm.sweep_utils import with_sweep
from autotrain.trainers.common import ALLOW_REMOTE_CODE


@with_sweep
def train(config):
    logger.info("Starting DPO training...")
    if isinstance(config, dict):
        config = LLMTrainingParams(**config)

    # process_input_data() now handles column validation internally before renaming
    train_data, valid_data = utils.process_input_data(config)

    tokenizer = utils.get_tokenizer(config)
    train_data, valid_data = utils.process_data_with_chat_template(config, tokenizer, train_data, valid_data)

    logging_steps = utils.configure_logging_steps(config, train_data, valid_data)
    training_args = utils.configure_training_args(config, logging_steps)
    config = utils.configure_block_size(config, tokenizer)

    training_args["max_length"] = config.block_size
    training_args["beta"] = config.dpo_beta
    args = DPOConfig(**training_args)

    logger.info("loading model config...")
    model_config = AutoConfig.from_pretrained(
        config.model,
        token=config.token,
        trust_remote_code=ALLOW_REMOTE_CODE,
        use_cache=config.disable_gradient_checkpointing,
    )

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

        model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
        logger.info("Using PEFT, model_ref will be set to None")
        model_ref = None
    else:
        model_kwargs = {
            "config": model_config,
            "token": config.token,
            "trust_remote_code": ALLOW_REMOTE_CODE,
        }
        # Only pass use_flash_attention_2 if enabled (many models don't support it)
        if config.use_flash_attention_2:
            model_kwargs["use_flash_attention_2"] = True

        model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)

        if config.model_ref is not None:
            model_ref = AutoModelForCausalLM.from_pretrained(config.model_ref, **model_kwargs)
        else:
            model_ref = None

    logger.info(f"model dtype: {model.dtype}")
    model.resize_token_embeddings(len(tokenizer))

    if model_ref is not None:
        logger.info(f"model_ref dtype: {model_ref.dtype}")
        model_ref.resize_token_embeddings(len(tokenizer))

    if config.peft:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=utils.get_target_modules(config),
        )

    logger.info("creating trainer")
    callbacks = utils.get_callbacks(
        config, train_data=train_data, valid_data=valid_data, model=model, tokenizer=tokenizer
    )

    # Set up compute_metrics if custom metrics are specified
    compute_metrics = None
    if hasattr(config, "custom_metrics") and config.custom_metrics:
        # Parse custom metrics list from config
        if isinstance(config.custom_metrics, str):
            import json

            custom_metrics_list = json.loads(config.custom_metrics)
        else:
            custom_metrics_list = config.custom_metrics
        compute_metrics = utils.get_rl_metrics(custom_metrics_list)
        logger.info(f"Using custom metrics for DPO: {custom_metrics_list}")

    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )

    trainer = DPOTrainer(
        **trainer_args,
        ref_model=model_ref,
        train_dataset=train_data,
        eval_dataset=valid_data,
        processing_class=tokenizer,
        peft_config=peft_config if config.peft else None,
    )

    trainer.remove_callback(PrinterCallback)
    trainer.train(resume_from_checkpoint=utils.get_resume_checkpoint(config))
    utils.post_training_steps(config, trainer)
    return trainer
