"""
LLM dataset format converter and preprocessor for AutoTrain.

This module handles conversion of various dataset formats to the standard
messages/conversations format expected by chat models.
"""

import logging
from typing import Any, Dict, Optional, Union

import pandas as pd
from datasets import Dataset

from autotrain.rendering.utils import safe_apply_chat_template


# No vendor packages needed

logger = logging.getLogger(__name__)


def detect_dataset_format(
    dataset: Union[Dataset, pd.DataFrame], sample_size: int = 5, trainer_type: str = "sft"
) -> str:
    """
    Detect the format of a dataset by analyzing its columns and content.

    Args:
        dataset: The dataset to analyze
        sample_size: Number of samples to examine
        trainer_type: Trainer type to help determine format (e.g., 'default' for plain text)

    Returns:
        Format name: 'messages', 'sharegpt', 'alpaca', 'plain_text', 'dpo', or 'unknown'
    """
    # Get columns
    if isinstance(dataset, pd.DataFrame):
        columns = dataset.columns.tolist()
    else:
        columns = dataset.column_names if hasattr(dataset, "column_names") else list(dataset.features.keys())

    # Get samples correctly for HF datasets
    try:
        if isinstance(dataset, pd.DataFrame):
            samples = dataset.head(sample_size).to_dict("records")
        else:
            # Properly handle HF Dataset indexing - avoid accessing beyond dataset length
            dataset_len = len(dataset) if hasattr(dataset, "__len__") else 0
            if dataset_len > 0:
                samples = [dataset[i] for i in range(min(sample_size, dataset_len))]
            else:
                # Handle streaming/iterable datasets
                samples = []
                for i, item in enumerate(dataset):
                    if i >= sample_size:
                        break
                    samples.append(item)
    except Exception as e:
        logger.warning(f"Error getting samples: {e}")
        samples = []

    # For pretraining/completion trainers, check plain text first
    if trainer_type in ["default", "completion", "clm", "pretraining"]:
        if "text" in columns:
            logger.info(f"Detected plain text format (trainer type: {trainer_type})")
            return "plain_text"

    # Check for DPO/ORPO format first (most specific)
    dpo_cols = {"prompt", "chosen", "rejected"}
    if dpo_cols.issubset(set(columns)):
        logger.info("Detected DPO/ORPO format (prompt/chosen/rejected)")
        return "dpo"

    # Check for Alpaca format
    if {"instruction", "output"}.issubset(set(columns)):
        logger.info("Detected Alpaca format (instruction/input/output)")
        return "alpaca"

    # Check for Q&A format and variations
    qa_patterns = [
        {"question", "answer"},
        {"query", "response"},
        {"prompt", "completion"},
        {"human", "bot"},
        {"input", "target"},
        {"user", "assistant"},
        {"instruction", "response"},
        {"input", "output"},  # Single input/output without instruction
    ]

    for pattern in qa_patterns:
        if pattern.issubset(set(columns)):
            logger.info(f"Detected Q&A-like format with columns: {pattern}")
            return "qa"

    # Check any column that looks like conversations/messages
    # Inspect the actual content structure by looking at the first element's keys
    conversation_columns = ["messages", "conversations", "chats", "dialog", "dialogue", "chat"]

    for col in conversation_columns:
        if col in columns and samples:
            # Check multiple samples to ensure consistency
            for sample in samples[: min(3, len(samples))]:
                col_data = sample.get(col)
                if isinstance(col_data, list) and col_data:
                    first_elem = col_data[0]
                    if isinstance(first_elem, dict):
                        # Determine format based on the keys in the first element
                        elem_keys = set(first_elem.keys())

                        # Check for standard messages format (role/content)
                        if {"role", "content"}.issubset(elem_keys):
                            logger.info(f"Detected messages format in column '{col}' (role/content)")
                            return "messages"
                        # Check for ShareGPT format (from/value)
                        elif {"from", "value"}.issubset(elem_keys):
                            logger.info(f"Detected ShareGPT format in column '{col}' (from/value)")
                            return "sharegpt"
                        # Alternative ShareGPT formats
                        elif {"sender", "message"}.issubset(elem_keys):
                            logger.info(f"Detected ShareGPT variant in column '{col}' (sender/message)")
                            return "sharegpt"
                        elif {"user", "text"}.issubset(elem_keys):
                            logger.info(f"Detected ShareGPT variant in column '{col}' (user/text)")
                            return "sharegpt"
                    # If we found valid data in this column, don't check other columns
                    break

    # Check for plain text
    if "text" in columns:
        logger.info("Detected plain text format")
        return "plain_text"

    logger.warning(f"Unknown dataset format with columns: {columns}")
    return "unknown"


def convert_alpaca_to_messages(dataset: Union[Dataset, pd.DataFrame]) -> Union[Dataset, pd.DataFrame]:
    """
    Convert Alpaca format (instruction/input/output) to messages format.

    Args:
        dataset: Dataset in Alpaca format

    Returns:
        Dataset with 'messages' column containing role/content pairs
    """

    def alpaca_to_messages(example):
        messages = []

        # Build the user message
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")

        if input_text:
            user_content = f"{instruction}\n\nInput: {input_text}"
        else:
            user_content = instruction

        messages.append({"role": "user", "content": user_content})

        # Add assistant response if present
        if "output" in example and example["output"]:
            messages.append({"role": "assistant", "content": example["output"]})

        return {"messages": messages}

    logger.info("Converting Alpaca format to messages format")

    if isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
        df["messages"] = df.apply(lambda x: alpaca_to_messages(x)["messages"], axis=1)
        return df
    else:
        return dataset.map(alpaca_to_messages)


def convert_with_column_mapping(
    dataset: Union[Dataset, pd.DataFrame], column_mapping: Dict[str, str]
) -> Union[Dataset, pd.DataFrame]:
    """
    Convert dataset to messages format using manual column mapping.

    Args:
        dataset: Dataset to convert
        column_mapping: Dict mapping role names to column names, e.g.:
            {
                'user_col': 'my_question_column',
                'assistant_col': 'my_answer_column',
                'system_col': 'my_system_column'  # optional
            }
            Or for multi-column scenarios:
            {
                'instruction_col': 'task',
                'input_col': 'context',
                'output_col': 'response'
            }

    Returns:
        Dataset with 'messages' column
    """

    def map_to_messages(example):
        messages = []

        # Handle system message if mapped
        if "system_col" in column_mapping:
            system_content = example.get(column_mapping["system_col"], "")
            if system_content:
                messages.append({"role": "system", "content": system_content})

        # Handle instruction+input pattern (Alpaca-like)
        if "instruction_col" in column_mapping:
            instruction = example.get(column_mapping["instruction_col"], "")
            input_text = example.get(column_mapping.get("input_col", ""), "")

            if input_text:
                user_content = f"{instruction}\n\nInput: {input_text}"
            else:
                user_content = instruction

            if user_content:
                messages.append({"role": "user", "content": user_content})

            # Get the output/response
            output_col = column_mapping.get("output_col") or column_mapping.get("assistant_col")
            if output_col:
                output = example.get(output_col, "")
                if output:
                    messages.append({"role": "assistant", "content": output})

        # Handle simple Q&A pattern
        elif "user_col" in column_mapping:
            user_content = example.get(column_mapping["user_col"], "")
            if user_content:
                messages.append({"role": "user", "content": user_content})

            if "assistant_col" in column_mapping:
                assistant_content = example.get(column_mapping["assistant_col"], "")
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})

        return {"messages": messages}

    logger.info(f"Converting dataset using column mapping: {column_mapping}")

    if isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
        df["messages"] = df.apply(lambda x: map_to_messages(x)["messages"], axis=1)
        return df
    else:
        return dataset.map(map_to_messages)


def convert_qa_to_messages(dataset: Union[Dataset, pd.DataFrame]) -> Union[Dataset, pd.DataFrame]:
    """
    Convert Q&A format to messages format.
    Handles various Q&A column naming patterns.

    Args:
        dataset: Dataset in Q&A format

    Returns:
        Dataset with 'messages' column containing role/content pairs
    """
    # Get columns
    if isinstance(dataset, pd.DataFrame):
        columns = dataset.columns.tolist()
    else:
        columns = dataset.column_names if hasattr(dataset, "column_names") else list(dataset.features.keys())

    # Find the Q&A columns by checking known patterns
    user_col = None
    assistant_col = None

    # Patterns for user/question columns
    user_patterns = ["question", "query", "prompt", "human", "input", "user", "instruction"]
    # Patterns for assistant/answer columns
    assistant_patterns = ["answer", "response", "completion", "bot", "target", "output", "assistant"]

    for col in columns:
        col_lower = col.lower()
        if not user_col and any(p in col_lower for p in user_patterns):
            user_col = col
        if not assistant_col and any(p in col_lower for p in assistant_patterns):
            assistant_col = col

    if not user_col or not assistant_col:
        logger.warning(f"Could not identify Q&A columns from: {columns}")
        # Fallback to first two columns
        if len(columns) >= 2:
            user_col = columns[0]
            assistant_col = columns[1]

    def qa_to_messages(example):
        messages = []

        # Add user question
        if user_col:
            question = example.get(user_col, "")
            if question:
                messages.append({"role": "user", "content": question})

        # Add assistant answer
        if assistant_col:
            answer = example.get(assistant_col, "")
            if answer:
                messages.append({"role": "assistant", "content": answer})

        return {"messages": messages}

    logger.info("Converting Q&A format to messages format")

    if isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
        df["messages"] = df.apply(lambda x: qa_to_messages(x)["messages"], axis=1)
        return df
    else:
        return dataset.map(qa_to_messages)


def convert_sharegpt_to_messages(
    dataset: Union[Dataset, pd.DataFrame], column_name: str = "conversations"
) -> Union[Dataset, pd.DataFrame]:
    """
    Convert ShareGPT format (from/value) to messages format (role/content).

    Using Unsloth's standardize_sharegpt when available for compatibility.

    Args:
        dataset: Dataset in ShareGPT format
        column_name: Name of the column containing conversations

    Returns:
        Dataset with 'messages' column containing role/content pairs
    """
    # Try to use Unsloth's standardize_sharegpt if available (CUDA systems only)
    try:
        import torch

        if torch.cuda.is_available():
            from unsloth.chat_templates import standardize_sharegpt

            logger.info("Using Unsloth's standardize_sharegpt for conversion")
            return standardize_sharegpt(dataset, column_name=column_name)
    except ImportError:
        logger.info("Unsloth not available, using fallback conversion")
    except Exception as e:
        logger.warning(f"Unsloth standardize_sharegpt failed: {e}, using fallback")

    # Fallback to our own implementation
    def sharegpt_to_messages(example):
        conversations = example.get(column_name, [])
        messages = []

        for conv in conversations:
            # Handle different ShareGPT key formats
            # Standard ShareGPT: from/value
            if "from" in conv and "value" in conv:
                role = conv["from"]
                content = conv["value"]
            # Variant: sender/message
            elif "sender" in conv and "message" in conv:
                role = conv["sender"]
                content = conv["message"]
            # Variant: user/text
            elif "user" in conv and "text" in conv:
                role = conv["user"]
                content = conv["text"]
            # Variant: role/text (sometimes used)
            elif "role" in conv and "text" in conv:
                role = conv["role"]
                content = conv["text"]
            else:
                logger.warning(f"Unknown conversation format: {conv.keys()}")
                continue

            # Map various role names to standard roles
            if role in ["human", "user", "usr", "USER"]:
                role = "user"
            elif role in ["gpt", "assistant", "ai", "model", "bot", "agent", "ASSISTANT", "chatbot"]:
                role = "assistant"
            elif role in ["system", "SYSTEM", "sys"]:
                role = "system"
            else:
                # Keep original role if not recognized
                logger.warning(f"Unknown role '{role}', keeping as-is")

            messages.append({"role": role, "content": content})

        return {"messages": messages}

    logger.info("Converting ShareGPT format to messages format (fallback)")

    if isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
        df["messages"] = df.apply(lambda x: sharegpt_to_messages(x)["messages"], axis=1)
        return df
    else:
        return dataset.map(sharegpt_to_messages)


def convert_dpo_to_messages(dataset: Union[Dataset, pd.DataFrame]) -> Union[Dataset, pd.DataFrame]:
    """
    Convert DPO format (prompt/chosen/rejected) to messages format.
    Creates two message sequences: one for chosen and one for rejected.

    Args:
        dataset: Dataset in DPO format

    Returns:
        Dataset with 'messages_chosen' and 'messages_rejected' columns
    """

    def dpo_to_messages(example):
        prompt = example.get("prompt", "")
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")

        # Handle if chosen/rejected are already lists (some DPO datasets have this format)
        if isinstance(chosen, list):
            # It's already in a conversation format, extract the assistant's response
            chosen = chosen[-1].get("content", "") if chosen and isinstance(chosen[-1], dict) else str(chosen)
        if isinstance(rejected, list):
            # It's already in a conversation format, extract the assistant's response
            rejected = (
                rejected[-1].get("content", "") if rejected and isinstance(rejected[-1], dict) else str(rejected)
            )

        # Ensure all are strings
        prompt = str(prompt) if prompt else ""
        chosen = str(chosen) if chosen else ""
        rejected = str(rejected) if rejected else ""

        messages_chosen = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]

        messages_rejected = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]

        return {
            "messages_chosen": messages_chosen,
            "messages_rejected": messages_rejected,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    logger.info("Converting DPO format to messages format")

    if isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
        result = df.apply(lambda x: dpo_to_messages(x), axis=1, result_type="expand")
        for col in ["messages_chosen", "messages_rejected", "prompt", "chosen", "rejected"]:
            if col in result.columns:
                df[col] = result[col]
        return df
    else:
        return dataset.map(dpo_to_messages)


def standardize_dataset(
    dataset: Union[Dataset, pd.DataFrame],
    target_format: str = "messages",
    auto_detect: bool = True,
    column_mapping: Optional[Dict[str, str]] = None,
) -> Union[Dataset, pd.DataFrame]:
    """
    Standardize a dataset to the target format.

    Args:
        dataset: The dataset to standardize
        target_format: Target format ('messages' for standard chat format)
        auto_detect: Whether to auto-detect the source format
        column_mapping: Manual column mapping to override auto-detection

    Returns:
        Standardized dataset
    """
    # If manual column mapping is provided, use it directly
    if column_mapping:
        logger.info(f"Using manual column mapping: {column_mapping}")
        return convert_with_column_mapping(dataset, column_mapping)
    if auto_detect:
        source_format = detect_dataset_format(dataset)
    else:
        source_format = "unknown"

    # If already in target format, return as-is
    if source_format == target_format:
        logger.info(f"Dataset already in {target_format} format")
        return dataset

    # Convert based on source format
    if source_format == "sharegpt":
        return convert_sharegpt_to_messages(dataset)
    elif source_format == "alpaca":
        return convert_alpaca_to_messages(dataset)
    elif source_format == "qa":
        return convert_qa_to_messages(dataset)
    elif source_format == "dpo":
        if target_format == "messages":
            return convert_dpo_to_messages(dataset)
        else:
            return dataset
    elif source_format == "plain_text":
        logger.info("Plain text format detected, no conversion needed")
        return dataset
    else:
        logger.warning(f"Cannot convert from {source_format} to {target_format}")
        return dataset


def apply_chat_template(
    dataset: Union[Dataset, pd.DataFrame],
    tokenizer,
    chat_template: Optional[str] = None,
    messages_column: str = "messages",
    output_column: str = "text",
    map_eos_token: bool = False,
) -> Union[Dataset, pd.DataFrame]:
    """
    Apply a chat template to a dataset with messages format.

    Args:
        dataset: Dataset with messages column
        tokenizer: Tokenizer with chat template
        chat_template: Optional specific template to use. Special values:
                      - None, "tokenizer", "none": Use tokenizer's built-in template
                      - Specific template name: Apply that template from Unsloth
        messages_column: Name of column containing messages
        output_column: Name of column to store formatted text
        map_eos_token: Whether to remap template end tokens to EOS (e.g., <|im_end|> → <eos>)

    Returns:
        Dataset with formatted text column
    """
    # Handle 'tokenizer' as meaning use the model's default template
    if chat_template == "tokenizer":
        chat_template = None

    # Try to apply chat template
    template_applied = False

    # Check if CUDA available and try Unsloth
    try:
        import torch

        if torch.cuda.is_available():
            try:
                from unsloth.chat_templates import get_chat_template

                tokenizer = get_chat_template(tokenizer, chat_template=chat_template, map_eos_token=map_eos_token)
                logger.info(f"Using Unsloth chat template: {chat_template or 'tokenizer default'}")
                if map_eos_token:
                    logger.info("Mapping template end tokens to EOS")
                template_applied = True
            except ImportError:
                logger.debug("Unsloth not available on CUDA system")
    except ImportError:
        pass

    # If not CUDA or Unsloth failed, just use tokenizer as-is
    if not template_applied:
        logger.info("Using tokenizer's built-in chat template")

    def format_messages(example):
        messages = example.get(messages_column, [])
        if not messages:
            return {output_column: ""}

        # Apply chat template with automatic tool role handling
        try:
            # Check if last message is from assistant to decide on generation prompt
            add_generation_prompt = True
            if messages and messages[-1].get("role") == "assistant":
                add_generation_prompt = False

            formatted = safe_apply_chat_template(
                tokenizer, messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            return {output_column: formatted}
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}")
            # Fallback to simple concatenation (preserving tool_calls, tool role as-is)
            import json

            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content") or ""
                # Serialize tool_calls to content
                if m.get("tool_calls"):
                    tool_json = json.dumps(m["tool_calls"], ensure_ascii=False)
                    content = f"{content}\n[Tool Calls] {tool_json}" if content else f"[Tool Calls] {tool_json}"
                parts.append(f"{role}: {content}")
            text = "\n".join(parts)
            return {output_column: text}

    logger.info(f"Applying chat template to {messages_column} column")

    if isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
        df[output_column] = df.apply(lambda x: format_messages(x)[output_column], axis=1)
        return df
    else:
        return dataset.map(format_messages)


def analyze_and_convert_dataset(
    dataset: Union[Dataset, pd.DataFrame],
    tokenizer=None,
    chat_template: Optional[str] = None,
    trainer_type: str = "sft",
    sample_size: int = 5,
    apply_template: bool = True,
    column_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Analyze a dataset and optionally convert it to the appropriate format.

    Args:
        dataset: The dataset to analyze
        tokenizer: Optional tokenizer for applying chat template
        chat_template: Optional chat template to use
        trainer_type: Type of trainer (sft, dpo, orpo, etc.)
        sample_size: Number of samples to analyze
        apply_template: Whether to apply chat template after conversion

    Returns:
        Dictionary with analysis results and converted dataset
    """
    # Detect format
    format_type = detect_dataset_format(dataset, sample_size)

    result = {
        "original_format": format_type,
        "format_detected": format_type,  # Alias for compatibility
        "columns": dataset.columns.tolist() if isinstance(dataset, pd.DataFrame) else list(dataset.features.keys()),
        "num_samples": len(dataset),
        "needs_conversion": False,
        "converted_dataset": dataset,
        "template_applied": False,
    }

    # Determine if conversion is needed
    if trainer_type in ["sft", "reward"] and format_type in ["alpaca", "sharegpt"]:
        result["needs_conversion"] = True
        result["target_format"] = "messages"
        result["converted_dataset"] = standardize_dataset(
            dataset, target_format="messages", column_mapping=column_mapping
        )

        # Apply chat template if tokenizer is provided
        if tokenizer and apply_template:
            if (
                "messages" in result["converted_dataset"].column_names
                if hasattr(result["converted_dataset"], "column_names")
                else result["converted_dataset"].columns
            ):
                result["converted_dataset"] = apply_chat_template(
                    result["converted_dataset"], tokenizer, chat_template=chat_template
                )
                result["template_applied"] = True

    elif trainer_type in ["dpo", "orpo"]:
        if format_type in ["alpaca", "sharegpt"]:
            # For DPO/ORPO, we need to convert to messages first, then user needs to create chosen/rejected
            result["needs_conversion"] = True
            result["target_format"] = "messages"
            result["converted_dataset"] = standardize_dataset(
                dataset, target_format="messages", column_mapping=column_mapping
            )
            result["warning"] = (
                "Dataset converted to messages format. You'll need to create chosen/rejected responses for DPO/ORPO training."
            )
        elif format_type == "dpo":
            # Already in DPO format, optionally apply templates to chosen/rejected
            result["needs_conversion"] = False
            result["target_format"] = "dpo"
            if tokenizer and apply_template:
                # Convert to messages format with templates applied
                result["converted_dataset"] = convert_dpo_to_messages(dataset)
                result["template_applied"] = True

    elif format_type == "plain_text":
        # Plain text doesn't need conversion
        result["needs_conversion"] = False
        result["target_format"] = "plain_text"

    # Add sample preview
    if len(dataset) > 0:
        if isinstance(dataset, pd.DataFrame):
            result["sample_preview"] = dataset.head(1).to_dict("records")[0]
        else:
            result["sample_preview"] = dataset[0]

    # Add converted sample preview if different
    if result["needs_conversion"] or result["template_applied"]:
        converted = result["converted_dataset"]
        if isinstance(converted, pd.DataFrame):
            result["converted_preview"] = converted.head(1).to_dict("records")[0]
        else:
            result["converted_preview"] = converted[0] if len(converted) > 0 else None

    return result


def apply_chat_template_with_mapping(
    dataset: Union[Dataset, pd.DataFrame],
    tokenizer,
    chat_template: Optional[str] = None,
    conversations_column: str = "conversations",
    output_column: str = "text",
    runtime_mapping: Optional[Dict[str, str]] = None,
    map_eos_token: bool = False,
) -> Union[Dataset, pd.DataFrame]:
    """
    Apply chat template directly using Unsloth's runtime mapping feature.

    This avoids converting to messages format first by mapping columns on the fly.

    Args:
        dataset: Dataset with conversation data
        tokenizer: Tokenizer with chat template support
        chat_template: Template name to use (None for tokenizer default)
        conversations_column: Column with conversation data
        output_column: Output column name for formatted text
        runtime_mapping: Custom mapping dict (e.g., {"role": "sender", "content": "text"})
                        If None, uses default ShareGPT mapping
        map_eos_token: Whether to remap template end tokens to EOS token

    Returns:
        Dataset with formatted text column
    """
    # Handle 'tokenizer' as meaning default
    if chat_template == "tokenizer":
        chat_template = None

    # Try Unsloth if CUDA available
    unsloth_available = False
    try:
        import torch

        if torch.cuda.is_available():
            from unsloth.chat_templates import get_chat_template

            # Use custom runtime mapping or default ShareGPT mapping
            if runtime_mapping:
                mapping = runtime_mapping
                logger.info(f"Using custom runtime mapping: {mapping}")
            else:
                # Default ShareGPT mapping for from/value -> role/content
                # Note: Unsloth expects strings, not lists for user/assistant
                mapping = {
                    "role": "from",
                    "content": "value",
                    "user": "human",  # Most common ShareGPT value
                    "assistant": "gpt",  # Most common ShareGPT value
                }
                logger.info("Using default ShareGPT runtime mapping")

            # Get tokenizer with mapping
            tokenizer = get_chat_template(
                tokenizer, chat_template=chat_template, mapping=mapping, map_eos_token=map_eos_token
            )
            logger.info(f"Using Unsloth with ShareGPT mapping for template: {chat_template or 'tokenizer default'}")
            unsloth_available = True
    except ImportError:
        pass

    if unsloth_available:

        def format_conversations(example):
            conversations = example.get(conversations_column, [])
            if not conversations:
                return {output_column: ""}

            try:
                # Unsloth's tokenizer can handle ShareGPT format directly with mapping
                add_generation_prompt = True
                if conversations and conversations[-1].get("from") in ["gpt", "assistant", "model", "bot"]:
                    add_generation_prompt = False

                # Note: ShareGPT format uses "from"/"value" keys, not standard "role"/"content"
                # safe_apply_chat_template handles standard messages format, so we use raw call here
                # ShareGPT typically doesn't have tool role, but Unsloth handles the mapping
                formatted = tokenizer.apply_chat_template(
                    conversations, tokenize=False, add_generation_prompt=add_generation_prompt
                )
                return {output_column: formatted}
            except Exception as e:
                logger.warning(f"Failed to apply template with mapping: {e}")
                # Fallback
                text = "\n".join([f"{c.get('from', 'user')}: {c.get('value', '')}" for c in conversations])
                return {output_column: text}

        if isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
            df[output_column] = df.apply(lambda x: format_conversations(x)[output_column], axis=1)
            return df
        else:
            return dataset.map(format_conversations)
    else:
        # Unsloth not available, fall back to conversion + template
        logger.warning("Unsloth not available, falling back to conversion + template")
        converted = convert_sharegpt_to_messages(dataset, conversations_column)
        return apply_chat_template(converted, tokenizer, chat_template, "messages", output_column)


def get_available_chat_templates():
    """
    Get list of available chat templates.

    First tries Unsloth if CUDA available, then falls back to standalone templates.

    Returns:
        List of template names, or None if not available
    """
    # Check if CUDA is available
    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    # Only try Unsloth if CUDA is available
    if cuda_available:
        # Try system Unsloth (if installed)
        try:
            from unsloth.chat_templates import CHAT_TEMPLATES

            logger.info("Using system Unsloth chat templates (CUDA available)")
            return list(CHAT_TEMPLATES.keys())
        except:
            pass

    # For CPU/MPS or if Unsloth failed, use standalone templates
    try:
        from .chat_templates_standalone import CHAT_TEMPLATES

        logger.info("Using standalone chat templates (CPU/MPS compatible)")
        return list(CHAT_TEMPLATES.keys())
    except ImportError:
        logger.warning("No chat templates available")
        return []


def extend_alpaca_conversations(
    dataset: Union[Dataset, pd.DataFrame],
    conversation_extension: int = 1,
    output_column_name: str = "output",
    seed: Optional[int] = None,
) -> Union[Dataset, pd.DataFrame]:
    """
    Extend single-turn Alpaca dataset to multi-turn conversations.

    This implements Unsloth's conversation_extension feature that merges
    multiple single-turn examples into multi-turn conversations.

    Args:
        dataset: Dataset in Alpaca format
        conversation_extension: Number of turns to merge (1 = no extension)
        output_column_name: Column containing the assistant responses
        seed: Random seed for reproducibility

    Returns:
        Dataset with extended conversations
    """
    if conversation_extension <= 1:
        return dataset

    logger.info(f"Extending conversations by merging {conversation_extension} single-turn examples")

    import random

    if seed is not None:
        random.seed(seed)

    def merge_examples(examples_batch):
        """Merge multiple examples into one conversation."""
        merged = []

        # Process in batches of conversation_extension
        for i in range(0, len(examples_batch), conversation_extension):
            batch = examples_batch[i : i + conversation_extension]
            if not batch:
                continue

            # Start with the first example
            first = batch[0]
            messages = []

            # Add each turn from the batch
            for example in batch:
                # Add user message
                instruction = example.get("instruction", "")
                input_text = example.get("input", "")

                if input_text:
                    user_content = f"{instruction}\n\nInput: {input_text}"
                else:
                    user_content = instruction

                messages.append({"role": "user", "content": user_content})

                # Add assistant response
                if output_column_name in example and example[output_column_name]:
                    messages.append({"role": "assistant", "content": example[output_column_name]})

            # Create merged example
            merged_example = {
                "messages": messages,
                "instruction": first.get("instruction", ""),  # Keep first instruction for reference
                "input": first.get("input", ""),
                "output": " ".join([e.get(output_column_name, "") for e in batch]),  # Concatenated output
            }
            merged.append(merged_example)

        return merged

    if isinstance(dataset, pd.DataFrame):
        # Shuffle and group DataFrame rows
        df = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
        examples = df.to_dict("records")
        merged = merge_examples(examples)
        return pd.DataFrame(merged)
    else:
        # For HF Dataset, we need to handle this differently
        # Shuffle indices
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        # Create batches
        merged_data = []
        for i in range(0, len(indices), conversation_extension):
            batch_indices = indices[i : i + conversation_extension]
            batch = [dataset[idx] for idx in batch_indices]

            if not batch:
                continue

            first = batch[0]
            messages = []

            for example in batch:
                instruction = example.get("instruction", "")
                input_text = example.get("input", "")

                if input_text:
                    user_content = f"{instruction}\n\nInput: {input_text}"
                else:
                    user_content = instruction

                messages.append({"role": "user", "content": user_content})

                if output_column_name in example and example[output_column_name]:
                    messages.append({"role": "assistant", "content": example[output_column_name]})

            merged_data.append(
                {
                    "messages": messages,
                    "instruction": first.get("instruction", ""),
                    "input": first.get("input", ""),
                    "output": " ".join([e.get(output_column_name, "") for e in batch]),
                }
            )

        # Create new dataset from merged data
        from datasets import Dataset as HFDataset

        return HFDataset.from_list(merged_data)


def formatting_prompts_func(
    dataset: Union[Dataset, pd.DataFrame], tokenizer, messages_column: str = "messages", batched: bool = True
) -> Union[Dataset, pd.DataFrame]:
    """
    Format dataset with chat template, following Unsloth's recommended approach.

    This is the standard formatting function that should be used after
    standardizing the dataset to messages format.

    Args:
        dataset: Dataset with messages column
        tokenizer: Tokenizer with chat template already applied via get_chat_template
        messages_column: Column containing messages
        batched: Whether to process in batches

    Returns:
        Dataset with 'text' column containing formatted prompts
    """

    def _format_batch(examples):
        """Format a batch of examples."""
        if messages_column in examples:
            conversations = examples[messages_column]
        else:
            # Handle both dict and list inputs
            conversations = examples.get("conversations", [])

        texts = []
        for convo in conversations:
            try:
                # Determine if we should add generation prompt
                add_generation_prompt = False
                if convo and isinstance(convo, list):
                    last_msg = convo[-1]
                    if isinstance(last_msg, dict) and last_msg.get("role") != "assistant":
                        add_generation_prompt = True

                text = safe_apply_chat_template(
                    tokenizer, convo, tokenize=False, add_generation_prompt=add_generation_prompt
                )
                texts.append(text)
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}")
                # Fallback formatting
                text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in convo])
                texts.append(text)

        return {"text": texts}

    def _format_single(example):
        """Format a single example."""
        convo = example.get(messages_column, example.get("conversations", []))

        try:
            add_generation_prompt = False
            if convo and isinstance(convo, list):
                last_msg = convo[-1]
                if isinstance(last_msg, dict) and last_msg.get("role") != "assistant":
                    add_generation_prompt = True

            text = safe_apply_chat_template(
                tokenizer, convo, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}")
            text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in convo])

        return {"text": text}

    logger.info("Applying chat template formatting to dataset")

    if isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
        if batched:
            # Process DataFrame in batches
            batch_size = 100
            texts = []
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i : i + batch_size]
                batch_dict = {messages_column: batch[messages_column].tolist()}
                result = _format_batch(batch_dict)
                texts.extend(result["text"])
            df["text"] = texts
        else:
            df["text"] = df.apply(lambda x: _format_single(x)["text"], axis=1)
        return df
    else:
        # HuggingFace Dataset
        return dataset.map(_format_batch if batched else _format_single, batched=batched)
