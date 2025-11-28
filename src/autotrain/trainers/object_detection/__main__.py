import argparse
import json
from functools import partial

from accelerate.state import PartialState
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback

from autotrain import logger
from autotrain.trainers.common import (
    ALLOW_REMOTE_CODE,
    LossLoggingCallback,
    TrainStartCallback,
    UploadLogs,
    monitor,
    pause_space,
    remove_autotrain_data,
    save_training_params,
)
from autotrain.trainers.object_detection import utils
from autotrain.trainers.object_detection.params import ObjectDetectionParams


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


@monitor
def train(config):
    if isinstance(config, dict):
        config = ObjectDetectionParams(**config)

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
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
        else:
            train_data = load_dataset(
                config.data_path,
                split=config.train_split,
                token=config.token,
                trust_remote_code=ALLOW_REMOTE_CODE,
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
                    trust_remote_code=ALLOW_REMOTE_CODE,
                )
            else:
                valid_data = load_dataset(
                    config.data_path,
                    split=config.valid_split,
                    token=config.token,
                    trust_remote_code=ALLOW_REMOTE_CODE,
                )

    # Apply max_samples to training data if specified (for testing/debugging)
    if hasattr(config, "max_samples") and config.max_samples is not None and config.max_samples > 0:
        original_size = len(train_data)

        # For object detection, ensure diverse object patterns by taking evenly spaced samples
        step = max(1, original_size // config.max_samples)
        indices = list(range(0, original_size, step))[: config.max_samples]
        train_data = train_data.select(indices)
        logger.info(
            f"Limited training data from {original_size} to {len(train_data)} samples (max_samples={config.max_samples}, evenly spaced for object diversity)"
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

    # Parse JSON strings in objects column if they are strings
    import json

    def parse_objects_if_string(example):
        if isinstance(example[config.objects_column], str):
            example[config.objects_column] = json.loads(example[config.objects_column])
        return example

    # Apply parsing to both train and validation data
    train_data = train_data.map(parse_objects_if_string)
    if valid_data is not None:
        valid_data = valid_data.map(parse_objects_if_string)

    logger.info(f"Train data: {train_data}")
    logger.info(f"Valid data: {valid_data}")

    # Get categories from feature metadata if available, otherwise extract from data
    try:
        categories = train_data.features[config.objects_column].feature["category"].names
    except (AttributeError, KeyError, TypeError):
        # Fallback: extract unique categories from the data
        all_categories = set()
        for item in train_data[config.objects_column]:
            if isinstance(item, dict) and "category" in item:
                if isinstance(item["category"], list):
                    all_categories.update(item["category"])
                else:
                    all_categories.add(item["category"])
        categories = sorted(all_categories)

    id2label = dict(enumerate(categories))
    label2id = {v: k for k, v in id2label.items()}

    model_config = AutoConfig.from_pretrained(
        config.model,
        label2id=label2id,
        id2label=id2label,
        trust_remote_code=ALLOW_REMOTE_CODE,
        token=config.token,
    )
    try:
        model = AutoModelForObjectDetection.from_pretrained(
            config.model,
            config=model_config,
            ignore_mismatched_sizes=True,
            trust_remote_code=ALLOW_REMOTE_CODE,
            token=config.token,
        )
    except OSError:
        model = AutoModelForObjectDetection.from_pretrained(
            config.model,
            config=model_config,
            trust_remote_code=ALLOW_REMOTE_CODE,
            token=config.token,
            ignore_mismatched_sizes=True,
            from_tf=True,
        )
    image_processor = AutoImageProcessor.from_pretrained(
        config.model,
        token=config.token,
        do_pad=False,
        do_resize=False,
        size={"longest_edge": config.image_square_size},
        trust_remote_code=ALLOW_REMOTE_CODE,
    )
    train_data, valid_data = utils.process_data(train_data, valid_data, image_processor, config)

    if config.logging_steps == -1:
        if config.valid_split is not None:
            logging_steps = int(0.2 * len(valid_data) / config.batch_size)
        else:
            logging_steps = int(0.2 * len(train_data) / config.batch_size)
        if logging_steps == 0:
            logging_steps = 1
        if logging_steps > 25:
            logging_steps = 25
        config.logging_steps = logging_steps
    else:
        logging_steps = config.logging_steps

    logger.info(f"Logging steps: {logging_steps}")

    training_args = dict(
        output_dir=config.project_name,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=2 * config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        eval_strategy=config.eval_strategy if config.valid_split is not None else "no",
        logging_steps=logging_steps,
        save_total_limit=config.save_total_limit,
        save_strategy=config.eval_strategy if config.valid_split is not None else "no",
        gradient_accumulation_steps=config.gradient_accumulation,
        report_to=config.log,
        auto_find_batch_size=config.auto_find_batch_size,
        lr_scheduler_type=config.scheduler,
        optim=config.optimizer,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        push_to_hub=False,
        load_best_model_at_end=True if config.valid_split is not None else False,
        ddp_find_unused_parameters=False,
    )

    if config.mixed_precision == "fp16":
        training_args["fp16"] = True
    if config.mixed_precision == "bf16":
        training_args["bf16"] = True

    if config.valid_split is not None:
        training_args["eval_do_concat_batches"] = False
        early_stop = EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        )
        callbacks_to_use = [early_stop]
    else:
        callbacks_to_use = []

    callbacks_to_use.extend([UploadLogs(config=config), LossLoggingCallback(), TrainStartCallback()])

    # Only set compute_metrics if we have validation data
    # This avoids the pycocotools dependency when not doing evaluation
    if config.valid_split is not None:
        _compute_metrics_fn = partial(
            utils.object_detection_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
        )
    else:
        _compute_metrics_fn = None

    args = TrainingArguments(**training_args)
    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks_to_use,
        data_collator=utils.collate_fn,
        tokenizer=image_processor,
        compute_metrics=_compute_metrics_fn if _compute_metrics_fn is not None else None,
    )

    trainer = Trainer(
        **trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.train()

    logger.info("Finished training, saving model...")
    trainer.save_model(config.project_name)
    image_processor.save_pretrained(config.project_name)

    model_card = utils.create_model_card(config, trainer)

    # save model card to output directory as README.md
    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card)

    if config.push_to_hub:
        if PartialState().process_index == 0:
            remove_autotrain_data(config)
            save_training_params(config)
            logger.info("Pushing model to hub...")
            api = HfApi(token=config.token)
            api.create_repo(
                repo_id=f"{config.username}/{config.project_name}", repo_type="model", private=True, exist_ok=True
            )
            api.upload_folder(
                folder_path=config.project_name, repo_id=f"{config.username}/{config.project_name}", repo_type="model"
            )

    if PartialState().process_index == 0:
        pause_space(config)


if __name__ == "__main__":
    _args = parse_args()
    training_config = json.load(open(_args.training_config))
    _config = ObjectDetectionParams(**training_config)
    train(_config)
