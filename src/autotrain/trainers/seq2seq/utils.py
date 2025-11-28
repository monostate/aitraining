import os

import evaluate
import nltk
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from autotrain import logger


ROUGE_METRIC = evaluate.load("rouge")

MODEL_CARD = """
---
library_name: transformers
tags:
- autotrain
- text2text-generation{base_model}
widget:
- text: "I love AutoTrain"{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Seq2Seq

## Validation Metrics
{validation_metrics}
"""


def _seq2seq_metrics(pred, tokenizer):
    """
    Compute sequence-to-sequence metrics for predictions and labels.

    Args:
        pred (tuple): A tuple containing predictions and labels.
                      Predictions and labels are expected to be token IDs.
        tokenizer (PreTrainedTokenizer): The tokenizer used for decoding the predictions and labels.

    Returns:
        dict: A dictionary containing the computed ROUGE metrics and the average length of the generated sequences.
              The keys are the metric names and the values are the corresponding scores rounded to four decimal places.
    """
    predictions, labels = pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = ROUGE_METRIC.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def merge_adapter(base_model_path, target_model_path, adapter_path):
    """
    Merge PEFT adapter weights with base model and save the full model.

    Args:
        base_model_path: Path or name of the base model
        target_model_path: Directory to save the merged model
        adapter_path: Path to the PEFT adapter weights
    """
    logger.info("Loading base model for merging...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    # Resize embeddings if needed (for added tokens during training)
    model_vocab_size = model.get_input_embeddings().num_embeddings
    tokenizer_vocab_size = len(tokenizer)

    if model_vocab_size != tokenizer_vocab_size:
        logger.info(f"Resizing model embeddings from {model_vocab_size} to {tokenizer_vocab_size}")
        model.resize_token_embeddings(tokenizer_vocab_size)

    # Load and merge adapter
    logger.info("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    # Save merged model
    logger.info("Saving merged model...")
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)
    logger.info("Model merging complete!")


def create_model_card(config, trainer):
    """
    Generates a model card string based on the provided configuration and trainer.

    Args:
        config (object): Configuration object containing the following attributes:
            - valid_split (optional): If not None, the function will include evaluation scores.
            - data_path (str): Path to the dataset.
            - project_name (str): Name of the project.
            - model (str): Path or identifier of the model.
        trainer (object): Trainer object with an `evaluate` method that returns evaluation metrics.

    Returns:
        str: A formatted model card string containing dataset information, validation metrics, and base model details.
    """
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
        eval_scores = [f"{k[len('eval_'):]}: {v}" for k, v in eval_scores.items()]
        eval_scores = "\n\n".join(eval_scores)

    else:
        eval_scores = "No validation metrics available"

    if config.data_path == f"{config.project_name}/autotrain-data" or os.path.isdir(config.data_path):
        dataset_tag = ""
    else:
        dataset_tag = f"\ndatasets:\n- {config.data_path}"

    if os.path.isdir(config.model):
        base_model = ""
    else:
        base_model = f"\nbase_model: {config.model}"

    model_card = MODEL_CARD.format(
        dataset_tag=dataset_tag,
        validation_metrics=eval_scores,
        base_model=base_model,
    )
    return model_card
