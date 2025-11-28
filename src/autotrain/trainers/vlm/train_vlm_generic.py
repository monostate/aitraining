from functools import partial

from datasets import load_dataset, load_from_disk
from PIL import Image
from transformers import AutoProcessor, Trainer, TrainingArguments
from transformers.trainer_callback import PrinterCallback

from autotrain import logger
from autotrain.trainers.common import ALLOW_REMOTE_CODE
from autotrain.trainers.vlm import utils


def collate_fn(examples, config, processor):
    if config.trainer == "captioning":
        # For captioning, check if there's a prompt column, otherwise use a simple prompt
        if config.prompt_text_column and config.prompt_text_column in examples[0]:
            prompts = ["answer " + example[config.prompt_text_column] for example in examples]
        else:
            # No prompt column for captioning, use a simple instruction
            prompts = ["describe" for _ in examples]
    else:
        # For VQA and other tasks, use the prompt column
        prompts = ["answer " + example[config.prompt_text_column] for example in examples]

    labels = [example[config.text_column] for example in examples]

    # Handle both PIL Images and string paths
    images = []
    for example in examples:
        img = example[config.image_column]
        if isinstance(img, str):
            img = Image.open(img)
        images.append(img.convert("RGB"))

    tokens = processor(
        text=prompts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="longest",
        tokenize_newline_separately=False,
    )
    return tokens


def train(config):
    valid_data = None
    if config.data_path == f"{config.project_name}/autotrain-data":
        train_data = load_from_disk(config.data_path)[config.train_split]
    else:
        if ":" in config.train_split:
            dataset_config_name, split = config.train_split.split(":")
            train_data = load_dataset(
                config.data_path,
                name=dataset_config_name,
                split=split,
                token=config.token,
            )
        else:
            train_data = load_dataset(
                config.data_path,
                split=config.train_split,
                token=config.token,
            )

    if config.valid_split is not None:
        if config.data_path == f"{config.project_name}/autotrain-data":
            valid_data = load_from_disk(config.data_path)[config.valid_split]
        else:
            if ":" in config.valid_split:
                dataset_config_name, split = config.valid_split.split(":")
                valid_data = load_dataset(
                    config.data_path,
                    name=dataset_config_name,
                    split=split,
                    token=config.token,
                )
            else:
                valid_data = load_dataset(
                    config.data_path,
                    split=config.valid_split,
                    token=config.token,
                )

    # Apply max_samples to training data if specified (for testing/debugging)
    if hasattr(config, "max_samples") and config.max_samples is not None and config.max_samples > 0:
        original_size = len(train_data)

        # For VLM, ensure diverse vision-language patterns by taking evenly spaced samples
        step = max(1, original_size // config.max_samples)
        indices = list(range(0, original_size, step))[: config.max_samples]
        train_data = train_data.select(indices)
        logger.info(
            f"Limited training data from {original_size} to {len(train_data)} samples (max_samples={config.max_samples}, evenly spaced for vision-language diversity)"
        )

    # Apply max_samples to validation data if specified (proportionally)
    if (
        config.valid_split is not None
        and hasattr(config, "max_samples")
        and config.max_samples is not None
        and config.max_samples > 0
    ):
        # Use 20% of max_samples for validation or less if validation set is smaller
        valid_max_samples = max(1, int(config.max_samples * 0.2))
        if len(valid_data) > valid_max_samples:
            original_size = len(valid_data)
            valid_data = valid_data.select(range(min(valid_max_samples, len(valid_data))))
            logger.info(f"Limited validation data from {original_size} to {len(valid_data)} samples")

    logger.info(f"Train data: {train_data}")
    logger.info(f"Valid data: {valid_data}")

    processor = AutoProcessor.from_pretrained(config.model, token=config.token, trust_remote_code=ALLOW_REMOTE_CODE)

    logging_steps = utils.configure_logging_steps(config, train_data, valid_data)
    training_args = utils.configure_training_args(config, logging_steps)

    args = TrainingArguments(**training_args)
    model = utils.get_model(config)

    logger.info("creating trainer")
    callbacks = utils.get_callbacks(config)
    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks,
    )

    col_fn = partial(collate_fn, config=config, processor=processor)

    trainer = Trainer(
        **trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data if valid_data is not None else None,
        data_collator=col_fn,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.train()
    utils.post_training_steps(config, trainer, processor)
