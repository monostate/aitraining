"""
Copyright 2023 The HuggingFace Team
"""

import json
import os
from dataclasses import dataclass
from typing import Union


def _create_text_from_messages(example):
    """Helper to create text from messages, preserving tool_calls.

    Note: This is a fallback for when template application fails.
    It preserves tool role as-is since there's no tokenizer to check support.
    """
    messages = example.get("messages", [])
    if not messages:
        return {"text": ""}
    text_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        # Serialize tool_calls to content
        if msg.get("tool_calls"):
            tool_json = json.dumps(msg["tool_calls"], ensure_ascii=False)
            content = f"{content}\n[Tool Calls] {tool_json}" if content else f"[Tool Calls] {tool_json}"
        text_parts.append(f"{role}: {content}")
    return {"text": "\n".join(text_parts)}


from autotrain import logger
from autotrain.backends.base import AVAILABLE_HARDWARE
from autotrain.backends.endpoints import EndpointsRunner
from autotrain.backends.local import LocalRunner
from autotrain.backends.ngc import NGCRunner
from autotrain.backends.nvcf import NVCFRunner
from autotrain.backends.spaces import SpaceRunner
from autotrain.dataset import (
    AutoTrainDataset,
    AutoTrainImageClassificationDataset,
    AutoTrainImageRegressionDataset,
    AutoTrainObjectDetectionDataset,
    AutoTrainVLMDataset,
)
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_regression.params import ImageRegressionParams
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams
from autotrain.trainers.vlm.params import VLMTrainingParams


def tabular_munge_data(params, local):
    if isinstance(params.target_columns, str):
        col_map_label = [params.target_columns]
    else:
        col_map_label = params.target_columns
    task = params.task
    if task == "classification" and len(col_map_label) > 1:
        task = "tabular_multi_label_classification"
    elif task == "classification" and len(col_map_label) == 1:
        task = "tabular_multi_class_classification"
    elif task == "regression" and len(col_map_label) > 1:
        task = "tabular_multi_column_regression"
    elif task == "regression" and len(col_map_label) == 1:
        task = "tabular_single_column_regression"
    else:
        raise Exception("Please select a valid task.")

    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            task=task,
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={"id": params.id_column, "label": col_map_label},
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            percent_valid=None,  # TODO: add to UI
            local=local,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.id_column = "autotrain_id"
        if len(col_map_label) == 1:
            params.target_columns = ["autotrain_label"]
        else:
            params.target_columns = [f"autotrain_label_{i}" for i in range(len(col_map_label))]
    return params


def llm_munge_data(params, local):
    # Check if dataset conversion is requested
    auto_convert = getattr(params, "auto_convert_dataset", False)
    use_sharegpt_mapping = getattr(params, "use_sharegpt_mapping", False)
    conversation_extension = getattr(params, "conversation_extension", 1)

    # Handle ShareGPT mapping (no conversion, just use Unsloth's mapping)
    if use_sharegpt_mapping:
        # User chose to use Unsloth's mapping feature instead of conversion
        logger.info("Using Unsloth's ShareGPT mapping feature (no dataset conversion)")

        # Check if we have CUDA for Unsloth runtime mapping
        try:
            import torch

            if torch.cuda.is_available():
                # We can use runtime mapping with Unsloth
                import pandas as pd
                from datasets import load_dataset
                from transformers import AutoTokenizer

                from .preprocessor.llm import apply_chat_template_with_mapping

                # Load dataset
                if os.path.exists(params.data_path):
                    # Local dataset
                    for ext in ["csv", "jsonl", "json", "parquet"]:
                        files = [f for f in os.listdir(params.data_path) if f.endswith(f".{ext}")]
                        if files:
                            file_path = os.path.join(params.data_path, f"{params.train_split}.{ext}")
                            if os.path.exists(file_path):
                                if ext == "csv":
                                    dataset = pd.read_csv(file_path)
                                elif ext in ["json", "jsonl"]:
                                    dataset = pd.read_json(file_path, lines=(ext == "jsonl"))
                                elif ext == "parquet":
                                    dataset = pd.read_parquet(file_path)
                                break
                else:
                    # HuggingFace dataset
                    dataset = load_dataset(params.data_path, split=params.train_split, trust_remote_code=True)

                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(params.model, token=params.token, trust_remote_code=True)

                # Get runtime mapping and map_eos_token from params
                runtime_mapping = getattr(params, "runtime_mapping", None)
                map_eos_token = getattr(params, "map_eos_token", False)

                # Detect the conversations column
                conversations_column = "conversations"  # Default
                if hasattr(dataset, "column_names"):
                    columns = dataset.column_names
                elif hasattr(dataset, "columns"):
                    columns = list(dataset.columns)
                else:
                    columns = []

                # Look for conversation-like columns
                for col in ["conversations", "messages", "chats", "dialog", "dialogue"]:
                    if col in columns:
                        conversations_column = col
                        logger.info(f"Detected conversations column: {col}")
                        break

                # Apply template with runtime mapping
                dataset = apply_chat_template_with_mapping(
                    dataset,
                    tokenizer,
                    chat_template=params.chat_template,
                    conversations_column=conversations_column,
                    output_column="text",
                    runtime_mapping=runtime_mapping,
                    map_eos_token=map_eos_token,
                )

                # Save the processed dataset
                project_data_dir = os.path.join(params.project_name, "data_mapped")
                os.makedirs(project_data_dir, exist_ok=True)

                if isinstance(dataset, pd.DataFrame):
                    mapped_path = os.path.join(project_data_dir, f"{params.train_split}.jsonl")
                    dataset.to_json(mapped_path, orient="records", lines=True, force_ascii=False)
                else:
                    mapped_path = os.path.join(project_data_dir, f"{params.train_split}.jsonl")
                    dataset.to_json(mapped_path, lines=True, force_ascii=False)

                # Update data path to use the mapped dataset
                params.data_path = project_data_dir
                params.text_column = "text"
                logger.info(f"Applied runtime mapping and saved to {mapped_path}")
            else:
                logger.warning("CUDA not available, cannot use Unsloth runtime mapping. Falling back to conversion.")
                # Fall through to regular conversion
                use_sharegpt_mapping = False
        except ImportError:
            logger.warning("Unsloth not available, cannot use runtime mapping. Falling back to conversion.")
            use_sharegpt_mapping = False

        # If we successfully applied mapping, return
        if use_sharegpt_mapping:
            params.sharegpt_mapping_enabled = True
            return params

    # If auto_convert is enabled, try to convert the dataset first
    if auto_convert:
        try:
            import pandas as pd
            from datasets import load_dataset

            from .preprocessor.llm import (
                apply_chat_template,
                detect_dataset_format,
                extend_alpaca_conversations,
                standardize_dataset,
            )

            logger.info("Auto-converting dataset format...")

            # Load dataset to analyze and convert
            if os.path.exists(params.data_path):
                # Local dataset
                for ext in ["csv", "jsonl", "json", "parquet"]:
                    files = [f for f in os.listdir(params.data_path) if f.endswith(f".{ext}")]
                    if files:
                        file_path = os.path.join(params.data_path, f"{params.train_split}.{ext}")
                        if os.path.exists(file_path):
                            if ext == "csv":
                                dataset = pd.read_csv(file_path)
                            elif ext in ["json", "jsonl"]:
                                dataset = pd.read_json(file_path, lines=(ext == "jsonl"))
                            elif ext == "parquet":
                                dataset = pd.read_parquet(file_path)
                            break
            else:
                # HuggingFace dataset
                dataset = load_dataset(params.data_path, split=params.train_split, trust_remote_code=True)

            # Detect format
            format_type = detect_dataset_format(dataset, trainer_type=params.trainer)
            logger.info(f"Detected dataset format: {format_type}")

            # Convert if needed
            if format_type in ["alpaca", "sharegpt", "qa", "unknown"]:
                logger.info(f"Converting dataset from {format_type} format to messages format...")
                # Get column mapping if provided
                column_mapping = getattr(params, "column_mapping", None)
                if column_mapping:
                    logger.info(f"Using column mapping: {column_mapping}")

                # First standardize to messages format
                dataset = standardize_dataset(dataset, column_mapping=column_mapping)
                logger.info("✓ Dataset standardized to messages format")

                # Apply conversation extension if requested (for Alpaca)
                if format_type == "alpaca" and conversation_extension > 1:
                    dataset = extend_alpaca_conversations(
                        dataset, conversation_extension=conversation_extension, output_column_name="output"
                    )

                # Apply chat template if tokenizer is available
                # Always apply template after conversion (None means use tokenizer default)
                # CRITICAL: We must create a 'text' column for training to work
                text_column_created = False
                if hasattr(params, "model"):
                    try:
                        from transformers import AutoTokenizer

                        tokenizer = AutoTokenizer.from_pretrained(
                            params.model, token=params.token, trust_remote_code=True
                        )

                        # Get map_eos_token parameter
                        map_eos_token = getattr(params, "map_eos_token", False)

                        # Apply the chat template using our function
                        dataset = apply_chat_template(
                            dataset,
                            tokenizer,
                            chat_template=params.chat_template,
                            messages_column="messages",
                            map_eos_token=map_eos_token,
                        )

                        # Verify text column was created
                        if isinstance(dataset, pd.DataFrame):
                            text_column_created = "text" in dataset.columns
                        else:
                            text_column_created = (
                                "text" in dataset.column_names if hasattr(dataset, "column_names") else False
                            )

                        # Update column mapping to use the new 'text' column
                        params.text_column = "text"
                        logger.info("✓ Chat template applied successfully, 'text' column created")
                    except Exception as e:
                        logger.warning(f"Could not apply chat template: {e}")
                        logger.info("Falling back to simple message formatting...")

                        # Fallback: create text column from messages manually
                        if isinstance(dataset, pd.DataFrame):
                            dataset["text"] = dataset.apply(lambda x: _create_text_from_messages(x)["text"], axis=1)
                        else:
                            dataset = dataset.map(_create_text_from_messages)
                        params.text_column = "text"
                        text_column_created = True
                        logger.info("✓ Fallback text column created from messages")

                # Ensure text column exists - if not, create it from messages
                if not text_column_created:
                    logger.warning("No text column found after conversion, creating from messages...")

                    if isinstance(dataset, pd.DataFrame):
                        dataset["text"] = dataset.apply(lambda x: _create_text_from_messages(x)["text"], axis=1)
                    else:
                        dataset = dataset.map(_create_text_from_messages)
                    params.text_column = "text"
                    logger.info("✓ Text column created from messages as fallback")

                # Save converted dataset to project directory under data_converted/
                project_data_dir = os.path.join(params.project_name, "data_converted")
                os.makedirs(project_data_dir, exist_ok=True)

                # Convert to pandas DataFrame for consistent saving
                if isinstance(dataset, pd.DataFrame):
                    df = dataset
                else:
                    # HuggingFace Dataset
                    df = dataset.to_pandas() if hasattr(dataset, "to_pandas") else pd.DataFrame(dataset)

                # Verify text column exists before saving
                if "text" not in df.columns:
                    logger.error("CRITICAL: 'text' column missing after conversion! Cannot proceed.")
                    raise ValueError("Dataset conversion failed: 'text' column not created")

                converted_path = os.path.join(project_data_dir, f"{params.train_split}.jsonl")
                df.to_json(converted_path, orient="records", lines=True, force_ascii=False)
                logger.info(f"✓ Converted dataset saved to {converted_path}")

                # Also save validation split if it exists
                if params.valid_split:
                    logger.info("Converting validation split...")
                    try:
                        # Load validation split
                        if os.path.exists(params.data_path):
                            # Local dataset - find validation file
                            for ext in ["csv", "jsonl", "json", "parquet"]:
                                valid_file = os.path.join(params.data_path, f"{params.valid_split}.{ext}")
                                if os.path.exists(valid_file):
                                    if ext == "csv":
                                        valid_dataset = pd.read_csv(valid_file)
                                    elif ext in ["json", "jsonl"]:
                                        valid_dataset = pd.read_json(valid_file, lines=(ext == "jsonl"))
                                    elif ext == "parquet":
                                        valid_dataset = pd.read_parquet(valid_file)
                                    break
                        else:
                            # HuggingFace dataset
                            valid_dataset = load_dataset(
                                params.data_path, split=params.valid_split, trust_remote_code=True
                            )

                        # Convert validation dataset the same way
                        valid_format_type = detect_dataset_format(valid_dataset, trainer_type=params.trainer)
                        if valid_format_type in ["alpaca", "sharegpt", "qa", "unknown"]:
                            column_mapping = getattr(params, "column_mapping", None)
                            valid_dataset = standardize_dataset(valid_dataset, column_mapping=column_mapping)

                        # Apply chat template to validation data
                        if hasattr(params, "model"):
                            try:
                                from transformers import AutoTokenizer

                                tokenizer = AutoTokenizer.from_pretrained(
                                    params.model, token=params.token, trust_remote_code=True
                                )
                                map_eos_token = getattr(params, "map_eos_token", False)
                                valid_dataset = apply_chat_template(
                                    valid_dataset,
                                    tokenizer,
                                    chat_template=params.chat_template,
                                    messages_column="messages",
                                    map_eos_token=map_eos_token,
                                )
                            except Exception as e:
                                logger.warning(f"Could not apply chat template to validation: {e}")

                                # Fallback
                                if isinstance(valid_dataset, pd.DataFrame):
                                    valid_dataset["text"] = valid_dataset.apply(
                                        lambda x: _create_text_from_messages(x)["text"], axis=1
                                    )
                                else:
                                    valid_dataset = valid_dataset.map(_create_text_from_messages)

                        # Convert to DataFrame and save
                        if isinstance(valid_dataset, pd.DataFrame):
                            valid_df = valid_dataset
                        else:
                            valid_df = (
                                valid_dataset.to_pandas()
                                if hasattr(valid_dataset, "to_pandas")
                                else pd.DataFrame(valid_dataset)
                            )

                        # Ensure text column exists
                        if "text" not in valid_df.columns:
                            valid_df["text"] = valid_df.apply(lambda x: _create_text_from_messages(x)["text"], axis=1)

                        valid_path = os.path.join(project_data_dir, f"{params.valid_split}.jsonl")
                        valid_df.to_json(valid_path, orient="records", lines=True, force_ascii=False)
                        logger.info(f"✓ Validation dataset converted and saved to {valid_path}")
                    except Exception as e:
                        logger.warning(
                            f"Could not convert validation split: {e}. Continuing with training split only."
                        )

                # Update data path to use converted dataset
                params.data_path = project_data_dir
                logger.info(f"✓ Dataset converted and saved to {project_data_dir}")
                logger.info(f"✓ Updated data_path to: {project_data_dir}")
                logger.info(f"✓ Text column set to: {params.text_column}")
                logger.info(f"Dataset conversion completed successfully. New data_path: {params.data_path}")

                # Important: Return early since we've already converted the data
                # The converted files don't need further processing through AutoTrainDataset
                return params

        except Exception as e:
            logger.error(f"Dataset auto-conversion failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            logger.warning("Continuing with original dataset.")

    # Continue with original logic
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        col_map = {"text": params.text_column}
        if params.rejected_text_column is not None:
            col_map["rejected_text"] = params.rejected_text_column
        if params.prompt_text_column is not None:
            col_map["prompt"] = params.prompt_text_column
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            task="lm_training",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping=col_map,
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            percent_valid=None,  # TODO: add to UI
            local=local,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = None
        params.text_column = "autotrain_text"
        params.rejected_text_column = "autotrain_rejected_text"
        params.prompt_text_column = "autotrain_prompt"
    return params


def seq2seq_munge_data(params, local):
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            task="seq2seq",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={"text": params.text_column, "label": params.target_column},
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            percent_valid=None,  # TODO: add to UI
            local=local,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.text_column = "autotrain_text"
        params.target_column = "autotrain_label"
    return params


def text_clf_munge_data(params, local):
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            task="text_multi_class_classification",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={"text": params.text_column, "label": params.target_column},
            percent_valid=None,  # TODO: add to UI
            local=local,
            convert_to_class_label=True,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.text_column = "autotrain_text"
        params.target_column = "autotrain_label"
    return params


def text_reg_munge_data(params, local):
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            task="text_single_column_regression",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={"text": params.text_column, "label": params.target_column},
            percent_valid=None,  # TODO: add to UI
            local=local,
            convert_to_class_label=False,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.text_column = "autotrain_text"
        params.target_column = "autotrain_label"
    return params


def token_clf_munge_data(params, local):
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            task="text_token_classification",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={"text": params.tokens_column, "label": params.tags_column},
            percent_valid=None,  # TODO: add to UI
            local=local,
            convert_to_class_label=True,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.tokens_column = "autotrain_text"
        params.tags_column = "autotrain_label"
    return params


def img_clf_munge_data(params, local):
    train_data_path = f"{params.data_path}/{params.train_split}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}"
    else:
        valid_data_path = None
    if os.path.isdir(train_data_path):
        dset = AutoTrainImageClassificationDataset(
            train_data=train_data_path,
            valid_data=valid_data_path,
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            local=local,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.image_column = "autotrain_image"
        params.target_column = "autotrain_label"
    return params


def img_obj_detect_munge_data(params, local):
    train_data_path = f"{params.data_path}/{params.train_split}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}"
    else:
        valid_data_path = None
    if os.path.isdir(train_data_path):
        dset = AutoTrainObjectDetectionDataset(
            train_data=train_data_path,
            valid_data=valid_data_path,
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            local=local,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.image_column = "autotrain_image"
        params.objects_column = "autotrain_objects"
    return params


def sent_transformers_munge_data(params, local):
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            task="sentence_transformers",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={
                "sentence1": params.sentence1_column,
                "sentence2": params.sentence2_column,
                "sentence3": params.sentence3_column,
                "target": params.target_column,
            },
            percent_valid=None,  # TODO: add to UI
            local=local,
            convert_to_class_label=True if params.trainer == "pair_class" else False,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.sentence1_column = "autotrain_sentence1"
        params.sentence2_column = "autotrain_sentence2"
        params.sentence3_column = "autotrain_sentence3"
        params.target_column = "autotrain_target"
    return params


def img_reg_munge_data(params, local):
    train_data_path = f"{params.data_path}/{params.train_split}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}"
    else:
        valid_data_path = None
    if os.path.isdir(train_data_path):
        dset = AutoTrainImageRegressionDataset(
            train_data=train_data_path,
            valid_data=valid_data_path,
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            local=local,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.image_column = "autotrain_image"
        params.target_column = "autotrain_label"
    return params


def vlm_munge_data(params, local):
    train_data_path = f"{params.data_path}/{params.train_split}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        col_map = {"text": params.text_column}
        if params.prompt_text_column is not None:
            col_map["prompt"] = params.prompt_text_column
        dset = AutoTrainVLMDataset(
            train_data=train_data_path,
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping=col_map,
            valid_data=valid_data_path if valid_data_path is not None else None,
            percent_valid=None,  # TODO: add to UI
            local=local,
        )
        params.data_path = dset.prepare()
        params.text_column = "autotrain_text"
        params.image_column = "autotrain_image"
        params.prompt_text_column = "autotrain_prompt"
    return params


def ext_qa_munge_data(params, local):
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break

    train_data_path = f"{params.data_path}/{params.train_split}.{ext_to_use}"
    if params.valid_split is not None:
        valid_data_path = f"{params.data_path}/{params.valid_split}.{ext_to_use}"
    else:
        valid_data_path = None
    if os.path.exists(train_data_path):
        dset = AutoTrainDataset(
            train_data=[train_data_path],
            valid_data=[valid_data_path] if valid_data_path is not None else None,
            task="text_extractive_question_answering",
            token=params.token,
            project_name=params.project_name,
            username=params.username,
            column_mapping={
                "text": params.text_column,
                "question": params.question_column,
                "answer": params.answer_column,
            },
            percent_valid=None,  # TODO: add to UI
            local=local,
            convert_to_class_label=True,
            ext=ext_to_use,
        )
        params.data_path = dset.prepare()
        params.valid_split = "validation"
        params.text_column = "autotrain_text"
        params.question_column = "autotrain_question"
        params.answer_column = "autotrain_answer"
    return params


@dataclass
class AutoTrainProject:
    """
    A class to train an AutoTrain project

    Attributes
    ----------
    params : Union[
        LLMTrainingParams,
        TextClassificationParams,
        TabularParams,
        Seq2SeqParams,
        ImageClassificationParams,
        TextRegressionParams,
        ObjectDetectionParams,
        TokenClassificationParams,
        SentenceTransformersParams,
        ImageRegressionParams,
        ExtractiveQuestionAnsweringParams,
        VLMTrainingParams,
    ]
        The parameters for the AutoTrain project.
    backend : str
        The backend to be used for the AutoTrain project. It should be one of the following:
        - local
        - spaces-a10g-large
        - spaces-a10g-small
        - spaces-a100-large
        - spaces-t4-medium
        - spaces-t4-small
        - spaces-cpu-upgrade
        - spaces-cpu-basic
        - spaces-l4x1
        - spaces-l4x4
        - spaces-l40sx1
        - spaces-l40sx4
        - spaces-l40sx8
        - spaces-a10g-largex2
        - spaces-a10g-largex4
    process : bool
        Flag to indicate if the params and dataset should be processed. If your data format is not AutoTrain-readable, set it to True. Set it to True when in doubt. Defaults to False.

    Methods
    -------
    __post_init__():
        Validates the backend attribute.
    create():
        Creates a runner based on the backend and initializes the AutoTrain project.
    """

    params: Union[
        LLMTrainingParams,
        TextClassificationParams,
        TabularParams,
        Seq2SeqParams,
        ImageClassificationParams,
        TextRegressionParams,
        ObjectDetectionParams,
        TokenClassificationParams,
        SentenceTransformersParams,
        ImageRegressionParams,
        ExtractiveQuestionAnsweringParams,
        VLMTrainingParams,
    ]
    backend: str
    process: bool = False

    def __post_init__(self):
        self.local = self.backend.startswith("local")
        if self.backend not in AVAILABLE_HARDWARE:
            raise ValueError(f"Invalid backend: {self.backend}")

    def _process_params_data(self):
        if isinstance(self.params, LLMTrainingParams):
            return llm_munge_data(self.params, self.local)
        elif isinstance(self.params, ExtractiveQuestionAnsweringParams):
            return ext_qa_munge_data(self.params, self.local)
        elif isinstance(self.params, ImageClassificationParams):
            return img_clf_munge_data(self.params, self.local)
        elif isinstance(self.params, ImageRegressionParams):
            return img_reg_munge_data(self.params, self.local)
        elif isinstance(self.params, ObjectDetectionParams):
            return img_obj_detect_munge_data(self.params, self.local)
        elif isinstance(self.params, SentenceTransformersParams):
            return sent_transformers_munge_data(self.params, self.local)
        elif isinstance(self.params, Seq2SeqParams):
            return seq2seq_munge_data(self.params, self.local)
        elif isinstance(self.params, TabularParams):
            return tabular_munge_data(self.params, self.local)
        elif isinstance(self.params, TextClassificationParams):
            return text_clf_munge_data(self.params, self.local)
        elif isinstance(self.params, TextRegressionParams):
            return text_reg_munge_data(self.params, self.local)
        elif isinstance(self.params, TokenClassificationParams):
            return token_clf_munge_data(self.params, self.local)
        elif isinstance(self.params, VLMTrainingParams):
            return vlm_munge_data(self.params, self.local)
        else:
            raise Exception("Invalid params class")

    def create(self):
        if self.process:
            self.params = self._process_params_data()

        if self.backend.startswith("local"):
            runner = LocalRunner(params=self.params, backend=self.backend)
            return runner.create()
        elif self.backend.startswith("spaces-"):
            runner = SpaceRunner(params=self.params, backend=self.backend)
            return runner.create()
        elif self.backend.startswith("ep-"):
            runner = EndpointsRunner(params=self.params, backend=self.backend)
            return runner.create()
        elif self.backend.startswith("ngc-"):
            runner = NGCRunner(params=self.params, backend=self.backend)
            return runner.create()
        elif self.backend.startswith("nvcf-"):
            runner = NVCFRunner(params=self.params, backend=self.backend)
            return runner.create()
        else:
            raise NotImplementedError
