from peft import LoraConfig
from transformers.trainer_callback import PrinterCallback
from trl import ORPOConfig, ORPOTrainer

from autotrain import logger
from autotrain.trainers.clm import utils
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.clm.sweep_utils import with_sweep


@with_sweep
def train(config):
    logger.info("Starting ORPO training...")
    if isinstance(config, dict):
        config = LLMTrainingParams(**config)

    # process_input_data() handles column validation internally before renaming
    train_data, valid_data = utils.process_input_data(config)

    tokenizer = utils.get_tokenizer(config)
    train_data, valid_data = utils.process_data_with_chat_template(config, tokenizer, train_data, valid_data)

    logging_steps = utils.configure_logging_steps(config, train_data, valid_data)
    training_args = utils.configure_training_args(config, logging_steps)
    config = utils.configure_block_size(config, tokenizer)

    training_args["max_length"] = config.block_size
    training_args["max_prompt_length"] = config.max_prompt_length
    training_args["max_completion_length"] = config.max_completion_length
    training_args["beta"] = config.dpo_beta
    args = ORPOConfig(**training_args)

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
        logger.info(f"Using custom metrics for ORPO: {custom_metrics_list}")

    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )

    trainer = ORPOTrainer(
        **trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        processing_class=tokenizer,
        peft_config=peft_config if config.peft else None,
    )

    trainer.remove_callback(PrinterCallback)
    trainer.train()
    utils.post_training_steps(config, trainer)
    return trainer
