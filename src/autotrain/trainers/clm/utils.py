import ast
import gc
import os
from enum import Enum
from itertools import chain

import requests
import torch
from accelerate.state import PartialState
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from autotrain import is_unsloth_available, logger
from autotrain.rendering.utils import safe_apply_chat_template
from autotrain.trainers.clm.callbacks import LoadBestPeftModelCallback, SavePeftModelCallback
from autotrain.trainers.common import (
    ALLOW_REMOTE_CODE,
    LossLoggingCallback,
    TrainStartCallback,
    UploadLogs,
    pause_space,
    remove_autotrain_data,
    save_training_params,
)


def get_model_max_position_embeddings(model_name, token=None):
    """
    Auto-detect max_position_embeddings from model config.

    Handles both regular LLMs and VLMs (which have text_config).

    Args:
        model_name: Model name or path
        token: HuggingFace token for private models

    Returns:
        int or None: max_position_embeddings if found, None otherwise
    """
    try:
        model_config = AutoConfig.from_pretrained(model_name, token=token, trust_remote_code=ALLOW_REMOTE_CODE)

        # Try text_config first (VLMs like Gemma 3n, LLaVA, etc.)
        if hasattr(model_config, "text_config"):
            max_pos = getattr(model_config.text_config, "max_position_embeddings", None)
            if max_pos is not None:
                logger.info(f"Auto-detected max_position_embeddings={max_pos} from text_config")
                return max_pos

        # Regular LLMs
        max_pos = getattr(model_config, "max_position_embeddings", None)
        if max_pos is not None:
            logger.info(f"Auto-detected max_position_embeddings={max_pos} from model config")
            return max_pos

        # Some models use n_positions instead (e.g., GPT-2)
        max_pos = getattr(model_config, "n_positions", None)
        if max_pos is not None:
            logger.info(f"Auto-detected n_positions={max_pos} from model config")
            return max_pos

        logger.debug(f"Could not find max_position_embeddings in {model_name} config")
        return None

    except Exception as e:
        logger.debug(f"Failed to auto-detect max_position_embeddings: {e}")
        return None


def validate_required_columns(dataset, required_columns, trainer_name, data_type="training"):
    """
    Validate that required columns exist in dataset.

    Args:
        dataset: HF Dataset object
        required_columns: List of required column names
        trainer_name: Name of trainer (for error message)
        data_type: "training" or "validation"

    Raises:
        ValueError with helpful message if columns missing
    """
    missing = [col for col in required_columns if col not in dataset.column_names]
    if missing:
        available = list(dataset.column_names)

        # Build helpful hints for common column mapping issues
        hints = []
        if "text" in missing and "messages" in available:
            hints.append(
                "Hint: Your dataset has a 'messages' column. Use --text-column messages for chat format data."
            )

        hint_str = "\n\n" + "\n".join(hints) if hints else ""

        raise ValueError(
            f"{trainer_name} trainer requires {data_type} data to have columns: {required_columns}\n"
            f"Columns {missing} not found in dataset.\n"
            f"Available columns in your data: {available}\n\n"
            f"Please ensure your CSV/JSON has the required columns.{hint_str}"
        )


DEFAULT_CHAT_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
TARGET_MODULES = {
    "Salesforce/codegen25-7b-multi": "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
}

MODEL_CARD = """
---
tags:
- aitraining
- text-generation-inference
- text-generation{peft}
library_name: transformers{base_model}
widget:
  - messages:
      - role: user
        content: What is your favorite condiment?
license: other{dataset_tag}
---

# {project_name}

This model was trained using [AITraining](https://github.com/monostate/aitraining).

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | `{model_name}` |
| Trainer | `{trainer}` |
| Dataset | `{data_path}` |
| Epochs | {epochs} |
| Learning Rate | {lr} |
| Batch Size | {batch_size} |
| Block Size | {block_size} |
{extra_params}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "PATH_TO_THIS_REPO"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

messages = [
    {{"role": "user", "content": "hi"}}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('cuda'))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

print(response)
```

---
*Trained with [AITraining](https://github.com/monostate/aitraining)*
"""


class ZephyrSpecialTokens(str, Enum):
    USER = "<|user|>"
    ASSISTANT = "<|assistant|>"
    SYSTEM = "<|system|>"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    PAD_TOKEN = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    USER = "<|im_start|>user"
    ASSISTANT = "<|im_start|>assistant"
    SYSTEM = "<|im_start|>system"
    EOS_TOKEN = "<|im_end|>"
    BOS_TOKEN = "<s>"
    PAD_TOKEN = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


def preprocess_reward(examples, tokenizer):
    """
    Preprocesses the reward data by tokenizing the chosen and rejected examples.

    Args:
        examples (dict): A dictionary containing two keys, "chosen" and "rejected", each mapping to a list of text examples.
        tokenizer (PreTrainedTokenizer): A tokenizer instance from the Hugging Face library used to tokenize the text examples.

    Returns:
        dict: A dictionary with the following keys:
            - "input_ids_chosen": List of tokenized input IDs for the chosen examples.
            - "attention_mask_chosen": List of attention masks for the chosen examples.
            - "input_ids_rejected": List of tokenized input IDs for the rejected examples.
            - "attention_mask_rejected": List of attention masks for the rejected examples.
    """
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, truncation=True)
        tokenized_rejected = tokenizer(rejected, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


def get_target_modules(config):
    """
    Determines the target modules based on the provided configuration.

    Args:
        config (object): Configuration object that contains the following attributes:
            - target_modules (str or None): Specifies the target modules. It can be:
                - None: Returns the default target modules for the model specified in the config.
                - An empty string: Returns the default target modules for the model specified in the config.
                - "all-linear": Returns the string "all-linear".
                - A comma-separated string: Returns a list of target modules split by commas.

    Returns:
        list or str: A list of target modules or a specific string ("all-linear") based on the configuration.
    """
    if config.target_modules is None:
        return TARGET_MODULES.get(config.model)
    if config.target_modules.strip() == "":
        return TARGET_MODULES.get(config.model)
    if config.target_modules.strip().lower() == "all-linear":
        return "all-linear"
    return config.target_modules.split(",")


def group_texts(examples, config):
    """
    Groups texts into chunks of a specified block size.

    Args:
        examples (dict): A dictionary where keys are feature names and values are lists of lists containing text data.
        config (object): A configuration object that contains the block_size attribute.

    Returns:
        dict: A dictionary with the same keys as the input examples, where each value is a list of chunks of text data.
              Additionally, a "labels" key is added with the same value as the "input_ids" key.
    """
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= config.block_size:
        total_length = (total_length // config.block_size) * config.block_size
    else:
        # Pad short sequences to block_size for uniform batching
        pad_length = config.block_size - total_length
        for k in concatenated_examples.keys():
            concatenated_examples[k].extend([0] * pad_length)
        total_length = config.block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + config.block_size] for i in range(0, total_length, config.block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize(examples, tokenizer, config):
    """
    Tokenizes the input examples using the provided tokenizer and configuration.

    Args:
        examples (dict): A dictionary containing the input examples to be tokenized.
        tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenizing the examples.
        config (object): Configuration object that contains the text column name.

    Returns:
        dict: A dictionary containing the tokenized output.
    """
    output = tokenizer(examples[config.text_column])
    return output


def merge_adapter(base_model_path, target_model_path, adapter_path):
    """
    Merges an adapter into a base model and saves the resulting model and tokenizer.

    Args:
        base_model_path (str): Path to the base model directory.
        target_model_path (str): Path to the directory where the merged model and tokenizer will be saved.
        adapter_path (str): Path to the adapter model directory.

    Raises:
        RuntimeError: If resizing token embeddings fails without padding to a multiple of 8.
    """
    logger.info("Loading adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=ALLOW_REMOTE_CODE,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        target_model_path,
        trust_remote_code=ALLOW_REMOTE_CODE,
    )
    try:
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, adapter_path)
    except RuntimeError:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    logger.info("Saving target model...")
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)


def create_model_card(config):
    """
    Generates a model card string based on the provided configuration.

    Args:
        config (object): Configuration object with the following attributes:
            - peft (bool): Indicates if PEFT (Parameter-Efficient Fine-Tuning) is used.
            - data_path (str): Path to the dataset.
            - project_name (str): Name of the project.
            - model (str): Path or identifier of the model.
            - trainer (str): Trainer type (sft, dpo, orpo, etc.)
            - epochs (int): Number of training epochs.
            - lr (float): Learning rate.
            - batch_size (int): Batch size.
            - block_size (int): Block/sequence size.

    Returns:
        str: A formatted model card string.
    """
    if config.peft:
        peft = "\n- peft"
    else:
        peft = ""

    if config.data_path == f"{config.project_name}/autotrain-data" or os.path.isdir(config.data_path):
        dataset_tag = ""
        data_path_display = "local"
    else:
        dataset_tag = f"\ndatasets:\n- {config.data_path}"
        data_path_display = config.data_path

    if os.path.isdir(config.model):
        base_model = ""
        model_name = os.path.basename(config.model)
    else:
        base_model = f"\nbase_model: {config.model}"
        model_name = config.model

    # Build extra params section
    extra_params_list = []
    if config.peft:
        extra_params_list.append(f"| LoRA Rank | {getattr(config, 'lora_r', 'N/A')} |")
        extra_params_list.append(f"| LoRA Alpha | {getattr(config, 'lora_alpha', 'N/A')} |")
    if getattr(config, "quantization", None):
        extra_params_list.append(f"| Quantization | {config.quantization} |")
    if getattr(config, "chat_template", None):
        extra_params_list.append(f"| Chat Template | {config.chat_template} |")
    if getattr(config, "gradient_accumulation", None):
        extra_params_list.append(f"| Gradient Accumulation | {config.gradient_accumulation} |")

    extra_params = "\n".join(extra_params_list)

    # Get project name for title (use basename if path)
    project_name = os.path.basename(config.project_name) if config.project_name else "Fine-tuned Model"

    model_card = MODEL_CARD.format(
        dataset_tag=dataset_tag,
        peft=peft,
        base_model=base_model,
        project_name=project_name,
        model_name=model_name,
        trainer=getattr(config, "trainer", "sft"),
        data_path=data_path_display,
        epochs=getattr(config, "epochs", "N/A"),
        lr=getattr(config, "lr", "N/A"),
        batch_size=getattr(config, "batch_size", "N/A"),
        block_size=getattr(config, "block_size", "N/A"),
        extra_params=extra_params,
    )
    return model_card.strip()


def pause_endpoint(params):
    """
    Pauses a Hugging Face endpoint using the provided parameters.

    Args:
        params (object): An object containing the necessary parameters, including:
            - token (str): The authorization token to access the Hugging Face API.

    Returns:
        dict: The JSON response from the API call.

    Raises:
        KeyError: If the "ENDPOINT_ID" environment variable is not set.
        requests.exceptions.RequestException: If there is an issue with the API request.
    """
    endpoint_id = os.environ["ENDPOINT_ID"]
    username = endpoint_id.split("/")[0]
    project_name = endpoint_id.split("/")[1]
    api_url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{username}/{project_name}/pause"
    headers = {"Authorization": f"Bearer {params.token}"}
    r = requests.post(api_url, headers=headers, timeout=30)
    return r.json()


def _message_from_dict(m, Message):
    """
    Create a Message object from a dict, preserving tool_calls, tool_call_id, and reasoning_content.

    Args:
        m (dict): Message dict with role, content, and optionally tool_calls/tool_call_id/reasoning_content
        Message: The Message class to instantiate

    Returns:
        Message: A Message object with all relevant fields preserved
    """
    return Message(
        role=m["role"],
        content=m.get("content") or "",  # Handle None content
        tool_calls=m.get("tool_calls"),
        tool_call_id=m.get("tool_call_id"),
        reasoning_content=m.get("reasoning_content"),
    )


def strip_bos_token(text: str, tokenizer) -> str:
    """
    Strip BOS token from the beginning of text if present.

    This prevents double BOS tokens when the tokenizer adds BOS during training.
    Different tokenizers use different BOS tokens:
    - Gemma: <bos>
    - Llama 3: <|begin_of_text|>
    - Llama 2/Mistral: <s>
    - ChatML/Qwen: may use various tokens
    """
    if not text or not tokenizer:
        return text

    bos_token = getattr(tokenizer, "bos_token", None)
    if bos_token and text.startswith(bos_token):
        text = text[len(bos_token) :]
        # Also strip any leading newline that might follow BOS
        if text.startswith("\n"):
            text = text[1:]

    return text


def get_model_turn_markers(tokenizer):
    """
    Detect the turn markers used by the model's chat template.

    Returns a dict with:
    - start_turn: Token(s) that start a turn (e.g., '<start_of_turn>', '[INST]')
    - end_turn: Token(s) that end a turn (e.g., '<end_of_turn>', '[/INST]')
    - eos_token: The model's EOS token

    Different models use different conventions:
    - Gemma: <start_of_turn>role ... <end_of_turn>
    - Llama 2/Mistral: [INST] ... [/INST]
    - Llama 3: <|start_header_id|>role<|end_header_id|> ... <|eot_id|>
    - ChatML: <|im_start|>role ... <|im_end|>
    """
    markers = {
        "start_turn": None,
        "end_turn": None,
        "eos_token": getattr(tokenizer, "eos_token", None),
    }

    try:
        # Generate a sample to detect markers
        test_msgs = [{"role": "user", "content": "test"}]
        sample = tokenizer.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=False)

        # Detect turn markers based on common patterns
        if "<start_of_turn>" in sample and "<end_of_turn>" in sample:
            markers["start_turn"] = "<start_of_turn>"
            markers["end_turn"] = "<end_of_turn>"
        elif "<|start_header_id|>" in sample and "<|eot_id|>" in sample:
            markers["start_turn"] = "<|start_header_id|>"
            markers["end_turn"] = "<|eot_id|>"
        elif "[INST]" in sample and "[/INST]" in sample:
            markers["start_turn"] = "[INST]"
            markers["end_turn"] = "[/INST]"
        elif "<|im_start|>" in sample and "<|im_end|>" in sample:
            markers["start_turn"] = "<|im_start|>"
            markers["end_turn"] = "<|im_end|>"
        elif "<|user|>" in sample and "<|end|>" in sample:
            markers["start_turn"] = "<|user|>"
            markers["end_turn"] = "<|end|>"

    except Exception as e:
        logger.debug(f"Could not detect turn markers: {e}")

    return markers


def get_response_template(tokenizer):
    """
    Get the response template for DataCollatorForCompletionOnlyLM.

    This template marks where the assistant/model response begins in the chat format.
    Used to mask loss on user/system tokens so the model only learns to predict responses.

    Different models use different templates:
    - Gemma: '<start_of_turn>model\n'
    - Llama 3: '<|start_header_id|>assistant<|end_header_id|>\n\n'
    - ChatML/Qwen: '<|im_start|>assistant\n'
    - Llama 2/Mistral: '[/INST] '
    - Phi/Zephyr: '<|assistant|>\n'

    Returns:
        str or None: The response template string, or None if not detected
    """
    try:
        # Generate a sample with assistant response to detect the template
        test_msgs = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
        sample = tokenizer.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=False)

        # Detect response template based on common patterns
        # Order matters - check more specific patterns first

        # Gemma format
        if "<start_of_turn>model\n" in sample:
            return "<start_of_turn>model\n"

        # Llama 3 format
        if "<|start_header_id|>assistant<|end_header_id|>" in sample:
            # Find the exact format including newlines
            if "<|start_header_id|>assistant<|end_header_id|>\n\n" in sample:
                return "<|start_header_id|>assistant<|end_header_id|>\n\n"
            return "<|start_header_id|>assistant<|end_header_id|>\n"

        # ChatML/Qwen format - detect actual newline pattern
        if "<|im_start|>assistant" in sample:
            idx = sample.find("<|im_start|>assistant")
            after_assistant = sample[idx + len("<|im_start|>assistant") :]
            # Count leading newlines to match the actual template pattern
            newlines = ""
            for char in after_assistant:
                if char == "\n":
                    newlines += "\n"
                else:
                    break
            if newlines:
                return f"<|im_start|>assistant{newlines}"
            return "<|im_start|>assistant\n"

        # Llama 2/Mistral format - response comes after [/INST]
        if "[/INST]" in sample:
            return "[/INST] "

        # Phi/Zephyr format - detect actual newline pattern
        if "<|assistant|>" in sample:
            idx = sample.find("<|assistant|>")
            after_assistant = sample[idx + len("<|assistant|>") :]
            newlines = ""
            for char in after_assistant:
                if char == "\n":
                    newlines += "\n"
                else:
                    break
            if newlines:
                return f"<|assistant|>{newlines}"
            return "<|assistant|>\n"

        # Fallback: try to find "assistant" marker
        if "assistant" in sample.lower():
            logger.warning(
                "Could not detect exact response template, using generic 'assistant'. "
                "Consider setting response_template manually for best results."
            )

    except Exception as e:
        logger.debug(f"Could not detect response template: {e}")

    return None


def has_generation_tags(tokenizer):
    """
    Check if the tokenizer's chat template supports {% generation %} tags.

    TRL's assistant_only_loss=True requires chat templates with {% generation %} tags
    to automatically create assistant masks. Models without these tags need to use
    DataCollatorForCompletionOnlyLM instead.

    Returns:
        bool: True if chat template supports generation tags, False otherwise
    """
    try:
        template = getattr(tokenizer, "chat_template", None)
        if template is None:
            return False

        # Check for both variations of the generation tag
        has_tag = "{% generation %}" in template or "{%generation%}" in template
        has_endtag = "{% endgeneration %}" in template or "{%endgeneration%}" in template

        return has_tag and has_endtag

    except Exception as e:
        logger.debug(f"Could not check for generation tags: {e}")
        return False


def add_completion_mask(dataset, tokenizer, response_template, text_column="text"):
    """
    Add completion_mask to dataset for response-only training.

    The completion_mask marks which tokens are part of the assistant response (1)
    vs. which are part of prompts/instructions (0). TRL's DataCollatorForLanguageModeling
    uses this mask to set labels to -100 for non-completion tokens.

    Args:
        dataset: HuggingFace dataset
        tokenizer: The tokenizer
        response_template: String that marks start of assistant response (e.g., "<start_of_turn>model\n")
        text_column: Name of the text column

    Returns:
        Dataset with added completion_mask column
    """
    # Tokenize the response template to get its token IDs
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    template_len = len(response_template_ids)

    # For flexible matching: extract base template (without trailing newlines)
    # This handles cases where the template may have variable newlines (\n, \n\n, \n\n\n)
    base_template_str = response_template.rstrip("\n")
    base_template_ids = tokenizer.encode(base_template_str, add_special_tokens=False)
    base_template_len = len(base_template_ids)

    # Get newline tokens for this tokenizer (they vary by model)
    newline_tokens = set()
    for nl in ["\n", "\n\n", "\n\n\n"]:
        nl_ids = tokenizer.encode(nl, add_special_tokens=False)
        if len(nl_ids) == 1:
            newline_tokens.add(nl_ids[0])

    logger.info(f"Response template '{repr(response_template)}' -> token IDs: {response_template_ids}")
    logger.info(
        f"Base template '{repr(base_template_str)}' -> IDs: {base_template_ids}, newline tokens: {newline_tokens}"
    )

    def add_mask(example):
        text = example.get(text_column, "")
        if not text:
            return {"completion_mask": []}

        # Tokenize the full text
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Create mask: 0 for prompt, 1 for completion
        mask = [0] * len(tokens)

        # Get the end turn marker for this model
        markers = get_model_turn_markers(tokenizer)
        end_turn_ids = None
        if markers.get("end_turn"):
            end_turn_ids = tokenizer.encode(markers["end_turn"], add_special_tokens=False)

        # Find all occurrences of response template and mark everything after as completion
        # Use FLEXIBLE matching: match base template then skip any newline tokens
        # This handles variable newlines (\n, \n\n, \n\n\n) that different templates produce
        i = 0
        while i < len(tokens):
            # Check if we're at the start of a response template (flexible match on base pattern)
            if tokens[i : i + base_template_len] == base_template_ids:
                # Skip the base template
                i += base_template_len
                # Skip any newline tokens (handles \n, \n\n, \n\n\n variations)
                while i < len(tokens) and tokens[i] in newline_tokens:
                    i += 1
                # Mark everything until end_of_turn as completion
                if end_turn_ids:
                    while i < len(tokens):
                        if tokens[i : i + len(end_turn_ids)] == end_turn_ids:
                            # Include the end_of_turn token in completion
                            for j in range(len(end_turn_ids)):
                                if i + j < len(tokens):
                                    mask[i + j] = 1
                            i += len(end_turn_ids)
                            break
                        mask[i] = 1
                        i += 1
                else:
                    # No end turn marker - mark until end of sequence
                    while i < len(tokens):
                        mask[i] = 1
                        i += 1
            else:
                i += 1

        return {"completion_mask": mask}

    logger.info("Adding completion_mask to dataset for response-only training...")
    dataset = dataset.map(add_mask, desc="Adding completion_mask")

    # Log sample to verify
    if len(dataset) > 0:
        sample_mask = dataset[0].get("completion_mask", [])
        completion_ratio = sum(sample_mask) / len(sample_mask) if sample_mask else 0
        logger.info(f"Sample completion_mask: {completion_ratio:.1%} of tokens are completion")

    return dataset


def validate_and_fix_turn_markers(tokenizer, train_data, valid_data, config):
    """
    Validate and fix turn markers in the dataset to match the model's expected format.

    Some datasets may be processed with one model's tokenizer but trained on another.
    This function detects mismatches and fixes them.

    For example:
    - Dataset processed with google/gemma uses <eos> as EOS
    - But unsloth/gemma uses <end_of_turn> as EOS
    - This mismatch causes the model to not stop generating properly
    - We fix by ensuring turn markers match what the model expects
    """
    if len(train_data) == 0:
        return train_data, valid_data

    sample_text = train_data[0].get(config.text_column, "")
    if not isinstance(sample_text, str):
        return train_data, valid_data

    # Get the model's expected markers from its chat template
    model_markers = get_model_turn_markers(tokenizer)
    model_eos = model_markers.get("eos_token")
    model_end_turn = model_markers.get("end_turn")
    model_start_turn = model_markers.get("start_turn")

    if not model_end_turn:
        return train_data, valid_data

    # Detect what markers the dataset is using
    # Common patterns to check
    marker_patterns = {
        "gemma": {"start": "<start_of_turn>", "end": "<end_of_turn>"},
        "llama3": {"start": "<|start_header_id|>", "end": "<|eot_id|>"},
        "llama2": {"start": "[INST]", "end": "[/INST]"},
        "chatml": {"start": "<|im_start|>", "end": "<|im_end|>"},
        "zephyr": {"start": "<|user|>", "end": "<|end|>"},
    }

    dataset_format = None
    for fmt, markers in marker_patterns.items():
        if markers["end"] in sample_text:
            dataset_format = fmt
            break

    if not dataset_format:
        logger.debug("Could not detect dataset turn marker format")
        return train_data, valid_data

    dataset_markers = marker_patterns[dataset_format]

    # Detect model format
    model_format = None
    for fmt, markers in marker_patterns.items():
        if markers["end"] == model_end_turn:
            model_format = fmt
            break

    if not model_format:
        logger.debug(f"Could not detect model turn marker format (end_turn={model_end_turn})")
        return train_data, valid_data

    # If formats match, no fix needed
    if dataset_format == model_format:
        logger.debug(f"Dataset and model both use {dataset_format} format, no fix needed")
        return train_data, valid_data

    # Formats don't match - need to fix
    logger.warning(
        f"Turn marker mismatch detected: dataset uses {dataset_format} format, "
        f"model expects {model_format} format. Fixing..."
    )

    old_start = dataset_markers["start"]
    old_end = dataset_markers["end"]
    new_start = marker_patterns[model_format]["start"]
    new_end = marker_patterns[model_format]["end"]

    def fix_turn_markers(example):
        text = example.get(config.text_column, "")
        if isinstance(text, str):
            # Replace turn markers
            text = text.replace(old_start, new_start)
            text = text.replace(old_end, new_end)
            example[config.text_column] = text
        return example

    logger.info(f"Replacing {repr(old_start)} -> {repr(new_start)}, {repr(old_end)} -> {repr(new_end)}")

    train_data = train_data.map(fix_turn_markers, load_from_cache_file=False)
    if valid_data is not None:
        valid_data = valid_data.map(fix_turn_markers, load_from_cache_file=False)

    logger.info("Turn markers fixed successfully")
    return train_data, valid_data


def apply_chat_template_unified(
    example,
    renderer,
    config,
):
    """
    Applies chat template using the unified rendering system.

    Args:
        example (dict): The input example containing the text data to be processed.
        renderer (MessageRenderer): The renderer to use for formatting messages.
        config (object): Configuration object with trainer type and text column.

    Returns:
        dict: The modified example with the chat template applied.

    Raises:
        ValueError: If the required keys are not found in the example for specific trainers.
    """
    from autotrain.rendering import Conversation, Message

    if config.trainer in ("default", "sft"):
        # Check if 'text' column already exists and is formatted (from auto-conversion or previous processing)
        if "text" in example and isinstance(example["text"], str):
            # Common chat template tokens across different models:
            # Gemma: <bos>, <eos>, <start_of_turn>, <end_of_turn>
            # ChatML/Qwen: <|im_start|>, <|im_end|>, <|endoftext|>
            # Llama 2/Mistral: <s>, </s>, [INST], [/INST]
            # Llama 3: <|begin_of_text|>, <|start_header_id|>, <|eot_id|>, <|end_header_id|>
            # Zephyr/Phi-3: <|user|>, <|assistant|>, <|system|>, <|end|>
            template_tokens = [
                "<bos>",
                "<eos>",
                "<start_of_turn>",
                "<end_of_turn>",  # Gemma
                "<|im_start|>",
                "<|im_end|>",
                "<|endoftext|>",  # ChatML/Qwen
                "<s>",
                "</s>",
                "[INST]",
                "[/INST]",  # Llama 2/Mistral
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|eot_id|>",
                "<|end_header_id|>",  # Llama 3
                "<|user|>",
                "<|assistant|>",
                "<|system|>",
                "<|end|>",  # Zephyr/Phi-3
            ]
            if any(token in example["text"] for token in template_tokens):
                logger.debug("Skipping chat template - 'text' column already contains formatted text")
                return example

        text_value = example.get(config.text_column, "")

        # First, check if we have a messages column (preferred source for conversion)
        if "messages" in example and isinstance(example["messages"], list) and len(example["messages"]) > 0:
            # Use messages column - this is the source of truth
            messages = example["messages"]
        elif isinstance(text_value, str):
            # Check if text is already formatted (contains template tokens)
            # Common chat template tokens across different models:
            # Gemma: <bos>, <eos>, <start_of_turn>, <end_of_turn>
            # ChatML/Qwen: <|im_start|>, <|im_end|>, <|endoftext|>
            # Llama 2/Mistral: <s>, </s>, [INST], [/INST]
            # Llama 3: <|begin_of_text|>, <|start_header_id|>, <|eot_id|>, <|end_header_id|>
            # Zephyr/Phi-3: <|user|>, <|assistant|>, <|system|>, <|end|>
            template_tokens = [
                "<bos>",
                "<eos>",
                "<start_of_turn>",
                "<end_of_turn>",  # Gemma
                "<|im_start|>",
                "<|im_end|>",
                "<|endoftext|>",  # ChatML/Qwen
                "<s>",
                "</s>",
                "[INST]",
                "[/INST]",  # Llama 2/Mistral
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|eot_id|>",
                "<|end_header_id|>",  # Llama 3
                "<|user|>",
                "<|assistant|>",
                "<|system|>",
                "<|end|>",  # Zephyr/Phi-3
            ]
            if any(token in text_value for token in template_tokens):
                # Already formatted, skip processing
                logger.debug("Skipping chat template - text column already contains formatted text from conversion")
                return example

            # Try to parse as string representation of messages list
            try:
                parsed = ast.literal_eval(text_value)
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                    messages = parsed
                else:
                    # Not messages format, skip
                    logger.warning("Text column doesn't contain messages format, skipping chat template")
                    return example
            except (SyntaxError, ValueError):
                # Can't parse - might be plain text, skip processing
                logger.warning(f"Could not parse text column as messages, skipping chat template")
                return example
        elif isinstance(text_value, list):
            # Already a list of messages
            messages = text_value
        else:
            # Unexpected type
            logger.warning(f"Unexpected text column type: {type(text_value)}, skipping chat template")
            return example

        # Convert to Conversation object (preserving tool_calls if present)
        conversation = Conversation(messages=[_message_from_dict(m, Message) for m in messages])

        # Extract tools from example if present (for tool definitions injection)
        tools = example.get("tools", None)

        # Render conversation to 'text' column (preserve original messages)
        # Strip BOS token to prevent double BOS when tokenizer adds it during training
        rendered = renderer.render_conversation(conversation, tools=tools)
        example["text"] = strip_bos_token(rendered, renderer.tokenizer)

    elif config.trainer == "reward":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]

            # Try to parse as stringified list if they're strings
            if isinstance(chosen_messages, str):
                try:
                    chosen_messages = ast.literal_eval(chosen_messages)
                except (SyntaxError, ValueError):
                    # It's a plain string response - convert to conversation format
                    # For reward models, we typically don't have a separate prompt, so treat as assistant response
                    chosen_messages = [{"role": "assistant", "content": chosen_messages}]

            if isinstance(rejected_messages, str):
                try:
                    rejected_messages = ast.literal_eval(rejected_messages)
                except (SyntaxError, ValueError):
                    # It's a plain string response - convert to conversation format
                    rejected_messages = [{"role": "assistant", "content": rejected_messages}]

            # Extract tools from example if present (for tool definitions injection)
            tools = example.get("tools", None)

            # Convert and render chosen (preserving tool_calls if present)
            # Strip BOS to prevent double BOS when tokenizer adds it during training
            chosen_conv = Conversation(messages=[_message_from_dict(m, Message) for m in chosen_messages])
            example["chosen"] = strip_bos_token(
                renderer.render_conversation(chosen_conv, tools=tools), renderer.tokenizer
            )

            # Convert and render rejected (preserving tool_calls if present)
            rejected_conv = Conversation(messages=[_message_from_dict(m, Message) for m in rejected_messages])
            example["rejected"] = strip_bos_token(
                renderer.render_conversation(rejected_conv, tools=tools), renderer.tokenizer
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm/orpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )

    elif config.trainer in ("dpo", "orpo"):
        # Similar handling for DPO/ORPO
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]

            # Try to parse as stringified list if they're strings
            if isinstance(chosen_messages, str):
                try:
                    chosen_messages = ast.literal_eval(chosen_messages)
                except (SyntaxError, ValueError):
                    # It's a plain string response - convert to conversation format
                    prompt_text = example.get("prompt", "")
                    chosen_messages = [
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": chosen_messages},
                    ]

            if isinstance(rejected_messages, str):
                try:
                    rejected_messages = ast.literal_eval(rejected_messages)
                except (SyntaxError, ValueError):
                    # It's a plain string response - convert to conversation format
                    prompt_text = example.get("prompt", "")
                    rejected_messages = [
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": rejected_messages},
                    ]

            # Extract tools from example if present (for tool definitions injection)
            tools = example.get("tools", None)

            # Extract prompt (all messages except last, preserving tool_calls if present)
            # Strip BOS to prevent double BOS when tokenizer adds it during training
            prompt_messages = chosen_messages[:-1]
            prompt_conv = Conversation(messages=[_message_from_dict(m, Message) for m in prompt_messages])
            example["prompt"] = strip_bos_token(
                renderer.render_conversation(prompt_conv, tools=tools), renderer.tokenizer
            )

            # Render full chosen (preserving tool_calls if present)
            chosen_conv = Conversation(messages=[_message_from_dict(m, Message) for m in chosen_messages])
            example["chosen"] = strip_bos_token(
                renderer.render_conversation(chosen_conv, tools=tools), renderer.tokenizer
            )

            # Render full rejected (preserving tool_calls if present)
            rejected_conv = Conversation(messages=[_message_from_dict(m, Message) for m in rejected_messages])
            example["rejected"] = strip_bos_token(
                renderer.render_conversation(rejected_conv, tools=tools), renderer.tokenizer
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{config.trainer}` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )

    return example


def apply_chat_template(
    example,
    tokenizer,
    config,
):
    """
    Legacy chat template application for backward compatibility.

    Applies a chat template to the given example based on the specified configuration.

    Args:
        example (dict): The input example containing the text data to be processed.
        tokenizer (object): The tokenizer to be used for applying the chat template.
        config (object): Configuration object containing the following attributes:
            - trainer (str): Specifies the type of trainer. Can be "default", "sft", "reward", "dpo", or "orpo".
            - text_column (str): The key in the example dict that contains the text data.
            - chat_template (str): Specifies the chat template to be used. Relevant for "reward" and "dpo" trainers.

    Returns:
        dict: The modified example with the chat template applied.

    Raises:
        ValueError: If the required keys are not found in the example for "reward", "dpo", or "orpo" trainers.
    """
    # kudos to Hugging Face H4 Team for this snippet
    # Using safe_apply_chat_template to handle tool role for models that don't support it (e.g., Gemma)
    if config.trainer in ("default", "sft"):
        # Check if 'text' column already exists and is formatted
        if "text" in example and isinstance(example["text"], str):
            # Common chat template tokens across different models:
            # Gemma: <bos>, <eos>, <start_of_turn>, <end_of_turn>
            # ChatML/Qwen: <|im_start|>, <|im_end|>, <|endoftext|>
            # Llama 2/Mistral: <s>, </s>, [INST], [/INST]
            # Llama 3: <|begin_of_text|>, <|start_header_id|>, <|eot_id|>, <|end_header_id|>
            # Zephyr/Phi-3: <|user|>, <|assistant|>, <|system|>, <|end|>
            template_tokens = [
                "<bos>",
                "<eos>",
                "<start_of_turn>",
                "<end_of_turn>",  # Gemma
                "<|im_start|>",
                "<|im_end|>",
                "<|endoftext|>",  # ChatML/Qwen
                "<s>",
                "</s>",
                "[INST]",
                "[/INST]",  # Llama 2/Mistral
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|eot_id|>",
                "<|end_header_id|>",  # Llama 3
                "<|user|>",
                "<|assistant|>",
                "<|system|>",
                "<|end|>",  # Zephyr/Phi-3
            ]
            if any(token in example["text"] for token in template_tokens):
                return example

        messages = example[config.text_column]
        if isinstance(messages, str):
            messages = ast.literal_eval(messages)
        # Create 'text' column instead of overwriting text_column (preserve original messages)
        # Strip BOS token to prevent double BOS when tokenizer adds it during training
        rendered_text = safe_apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=False)
        example["text"] = strip_bos_token(rendered_text, tokenizer)

    elif config.trainer == "reward":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            if isinstance(chosen_messages, str):
                chosen_messages = ast.literal_eval(chosen_messages)
            if isinstance(rejected_messages, str):
                rejected_messages = ast.literal_eval(rejected_messages)

            if config.chat_template == "zephyr" and chosen_messages[0]["role"] != "system":
                chosen_messages.insert(0, {"role": "system", "content": ""})
            if config.chat_template == "zephyr" and rejected_messages[0]["role"] != "system":
                rejected_messages.insert(0, {"role": "system", "content": ""})

            # Strip BOS to prevent double BOS when tokenizer adds it during training
            example["chosen"] = strip_bos_token(
                safe_apply_chat_template(tokenizer, chosen_messages, tokenize=False), tokenizer
            )
            example["rejected"] = strip_bos_token(
                safe_apply_chat_template(tokenizer, rejected_messages, tokenize=False), tokenizer
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm/orpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif config.trainer in ("dpo", "orpo"):
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # For DPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if isinstance(example["chosen"], str):
                example["chosen"] = ast.literal_eval(example["chosen"])
            if isinstance(example["rejected"], str):
                example["rejected"] = ast.literal_eval(example["rejected"])
            prompt_messages = example["chosen"][:-1]
            if config.chat_template == "zephyr" and example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            chosen_messages = example["chosen"][-1:]
            rejected_messages = example["rejected"][-1:]
            # Strip BOS to prevent double BOS when tokenizer adds it during training
            example["chosen"] = strip_bos_token(
                safe_apply_chat_template(tokenizer, chosen_messages, tokenize=False), tokenizer
            )
            example["rejected"] = strip_bos_token(
                safe_apply_chat_template(tokenizer, rejected_messages, tokenize=False), tokenizer
            )
            example["prompt"] = strip_bos_token(
                safe_apply_chat_template(tokenizer, prompt_messages, tokenize=False), tokenizer
            )
    else:
        raise ValueError(
            f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example


def post_training_steps(config, trainer):
    """
    Perform post-training steps including saving the model, creating a model card, merging adapter weights,
    and optionally pushing the model to the Hugging Face Hub.

    Args:
        config (object): Configuration object containing various settings and parameters.
        trainer (object): Trainer object used for training the model.

    Steps:
        1. Save the trained model and set `use_cache` to True.
        2. Create a model card and save it as README.md in the output directory.
        3. If PEFT (Parameter-Efficient Fine-Tuning) and adapter merging are enabled:
            - Delete the trainer object and clear CUDA cache.
            - Merge adapter weights into the base model.
            - Remove adapter weight files from the output directory.
        4. If pushing to the Hugging Face Hub is enabled:
            - Remove training data folder.
            - Push the model to the Hugging Face Hub repository.
        5. Pause the space if the process index is 0.

    Raises:
        Exception: If merging adapter weights fails.
    """
    logger.info("Finished training, saving model...")
    trainer.model.config.use_cache = True

    # For PEFT adapters, save config.json BEFORE saving adapter to preserve modifications (e.g., num_labels for reward models)
    if config.peft and not getattr(config, "merge_adapter", False):
        os.makedirs(config.project_name, exist_ok=True)
        cfg_path = os.path.join(config.project_name, "config.json")
        logger.info("Saving model config to preserve modifications (e.g., num_labels for reward models)...")

        # Get config and ensure critical fields are preserved
        model_config = trainer.model.config

        # For reward models, explicitly ensure num_labels is set (PEFT models don't include it by default)
        if hasattr(model_config, "num_labels"):
            num_labels = model_config.num_labels
            logger.info(f"Explicitly setting num_labels={num_labels} in saved config")
            # Convert to dict, add num_labels, then save
            import json

            # Ensure we have a dict-like object that supports item assignment
            if hasattr(model_config, "to_dict"):
                config_dict = model_config.to_dict()
            elif isinstance(model_config, dict):
                config_dict = model_config.copy()
            else:
                # If it's a Mock or other object, try to get its attributes as a dict
                config_dict = {}

            # Only set num_labels if we have a mutable dict
            if isinstance(config_dict, dict):
                config_dict["num_labels"] = num_labels
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=2)
            else:
                # Fallback: try to save using to_json_string if available
                if hasattr(model_config, "to_json_string"):
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        f.write(model_config.to_json_string())
        else:
            # Try to save using to_json_string if available
            if hasattr(model_config, "to_json_string"):
                with open(cfg_path, "w", encoding="utf-8") as f:
                    f.write(model_config.to_json_string())
            else:
                # If it's a dict, save directly
                import json

                if isinstance(model_config, dict):
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        json.dump(model_config, f, indent=2)

        logger.info(f"Saved config.json before adapter save")

    trainer.save_model(config.project_name)

    model_card = create_model_card(config)

    # Ensure output directory exists before writing README
    os.makedirs(config.project_name, exist_ok=True)

    # save model card to output directory as README.md
    with open(f"{config.project_name}/README.md", "w", encoding="utf-8") as f:
        f.write(model_card)

    # Ensure tokenizer files are present in the output folder for hub usage
    try:
        tok_files = [
            os.path.join(config.project_name, "tokenizer.json"),
            os.path.join(config.project_name, "tokenizer_config.json"),
        ]
        if not all(os.path.exists(p) for p in tok_files):
            logger.info("Saving tokenizer to output directory...")
            tokenizer = getattr(trainer, "tokenizer", None) or getattr(trainer, "processing_class", None)
            if tokenizer is not None:
                tokenizer.save_pretrained(config.project_name)
            else:
                # Fallback: load from base model
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(
                    config.model, token=config.token, trust_remote_code=ALLOW_REMOTE_CODE
                )
                tok.save_pretrained(config.project_name)
            logger.info("Tokenizer saved to output directory")
    except Exception as e:
        logger.warning(f"Failed to save tokenizer: {e}")

    if config.peft and config.merge_adapter:
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Merging adapter weights...")
        try:
            merge_adapter(
                base_model_path=config.model,
                target_model_path=config.project_name,
                adapter_path=config.project_name,
            )
            # remove adapter weights: adapter_*
            for file in os.listdir(config.project_name):
                if file.startswith("adapter_"):
                    os.remove(f"{config.project_name}/{file}")
        except Exception as e:
            logger.warning(f"Failed to merge adapter weights: {e}")
            logger.warning("Skipping adapter merge. Only adapter weights will be saved.")

    # Show completion message with next steps
    logger.info("=" * 60)
    logger.info("✓ Training completed successfully!")
    logger.info("=" * 60)
    model_path = os.path.abspath(config.project_name)
    logger.info(f"Model saved to: {model_path}")

    # Suggest testing the model with chat UI
    logger.info("\n💡 Next steps:")
    logger.info(f"   Test your model with the Chat UI:")
    logger.info(f"   aitraining chat")
    logger.info(f"   Then select your model: {model_path}")
    logger.info(f"   Or visit: http://localhost:7860/inference")

    # Debug diagnostics prior to hub push
    try:
        logger.info(
            f"post_training_steps: push_to_hub={config.push_to_hub}, username={getattr(config, 'username', None)}, "
            f"token_present={bool(getattr(config, 'token', None))}, process_index={PartialState().process_index}"
        )
        if os.path.isdir(config.project_name):
            try:
                contents = os.listdir(config.project_name)
                logger.info(f"Output dir '{config.project_name}' has {len(contents)} items")
                # Log a few representative files
                for fname in sorted(contents)[:10]:
                    try:
                        fpath = os.path.join(config.project_name, fname)
                        fsize = os.path.getsize(fpath) if os.path.isfile(fpath) else -1
                        logger.info(f" - {fname} size={fsize}")
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Could not list output directory: {e}")
        else:
            logger.warning(f"Output dir '{config.project_name}' not found before push_to_hub")
    except Exception:
        pass

    # Config already saved before adapter (see above) - no need to save again

    # Save training params regardless of push_to_hub setting
    if PartialState().process_index == 0:
        config.save(config.project_name)
        save_training_params(config)

    if config.push_to_hub:
        if PartialState().process_index == 0:
            # remove data folder
            remove_autotrain_data(config)
            logger.info("Pushing model to hub...")
            try:
                api = HfApi(token=config.token)
                # Use repo_id if provided, otherwise fall back to username/basename(project_name)
                if getattr(config, "repo_id", None):
                    repo_id = config.repo_id
                else:
                    # Use basename to handle cases where project_name is a full path
                    project_basename = os.path.basename(config.project_name.rstrip("/"))
                    repo_id = f"{config.username}/{project_basename}"
                logger.info(f"Ensuring repo exists: {repo_id}")
                api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
                logger.info(f"Uploading folder '{config.project_name}' to {repo_id}...")
                api.upload_folder(
                    folder_path=config.project_name,
                    repo_id=repo_id,
                    repo_type="model",
                )
                logger.info("Upload to hub finished")
            except Exception as e:
                logger.error(f"Upload to hub failed: {e}")
                # Do not raise; allow pipeline to continue while logging the failure

    if PartialState().process_index == 0:
        pause_space(config)


def process_input_data(config):
    """
    Processes input data based on the provided configuration.

    Args:
        config (object): Configuration object containing the following attributes:
            - data_path (str): Path to the dataset.
            - project_name (str): Name of the project.
            - train_split (str): Split name for training data.
            - valid_split (str, optional): Split name for validation data.
            - token (str, optional): Token for accessing the dataset.
            - text_column (str): Name of the text column.
            - rejected_text_column (str): Name of the rejected text column.
            - prompt_text_column (str): Name of the prompt text column.
            - trainer (str): Type of trainer (e.g., "dpo", "reward", "orpo").

    Returns:
        tuple: A tuple containing:
            - train_data (Dataset): Processed training dataset.
            - valid_data (Dataset or None): Processed validation dataset if valid_split is provided, otherwise None.
    """
    # Check if this is a converted dataset (from auto_convert_dataset flow)
    if config.data_path.endswith("data_converted") or "data_converted" in config.data_path:
        logger.info(f"Loading converted dataset from {config.data_path}")
        # Load JSONL files directly from the converted directory
        train_file = os.path.join(config.data_path, f"{config.train_split}.jsonl")
        if os.path.exists(train_file):
            train_data = load_dataset(
                "json", data_files=train_file, split="train", trust_remote_code=ALLOW_REMOTE_CODE
            )
            logger.info(f"✓ Loaded converted training data from {train_file}")
        else:
            raise FileNotFoundError(f"Converted training file not found: {train_file}")
    elif config.data_path == f"{config.project_name}/autotrain-data":
        logger.info("loading dataset from disk")
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

    # Apply max_samples to training data if specified (for testing/debugging)
    if hasattr(config, "max_samples") and config.max_samples is not None and config.max_samples > 0:
        original_size = len(train_data)

        # For LLM tasks, ensure diverse instructions/prompts by taking evenly spaced samples
        # This helps get variety of instruction types rather than sequential similar ones
        step = max(1, original_size // config.max_samples)
        indices = list(range(0, original_size, step))[: config.max_samples]
        train_data = train_data.select(indices)
        logger.info(
            f"Limited training data from {original_size} to {len(train_data)} samples (max_samples={config.max_samples}, evenly spaced for instruction diversity)"
        )

    # Validate columns BEFORE renaming (so users see their original column names in errors)
    if config.trainer in ("dpo", "orpo"):
        required_columns = [config.prompt_text_column, config.text_column, config.rejected_text_column]
        validate_required_columns(train_data, required_columns, config.trainer.upper(), "training")
    elif config.trainer == "reward":
        required_columns = [config.text_column, config.rejected_text_column]
        validate_required_columns(train_data, required_columns, config.trainer.upper(), "training")

    # rename columns for reward trainer
    if config.trainer in ("dpo", "reward", "orpo"):
        if not (config.text_column == "chosen" and config.text_column in train_data.column_names):
            train_data = train_data.rename_column(config.text_column, "chosen")
        if not (config.rejected_text_column == "rejected" and config.rejected_text_column in train_data.column_names):
            train_data = train_data.rename_column(config.rejected_text_column, "rejected")
    if config.trainer in ("dpo", "orpo"):
        if not (config.prompt_text_column == "prompt" and config.prompt_text_column in train_data.column_names):
            train_data = train_data.rename_column(config.prompt_text_column, "prompt")

    if config.valid_split is not None:
        # Check if this is a converted dataset (from auto_convert_dataset flow)
        if config.data_path.endswith("data_converted") or "data_converted" in config.data_path:
            logger.info(f"Loading converted validation dataset from {config.data_path}")
            valid_file = os.path.join(config.data_path, f"{config.valid_split}.jsonl")
            if os.path.exists(valid_file):
                valid_data = load_dataset(
                    "json", data_files=valid_file, split="train", trust_remote_code=ALLOW_REMOTE_CODE
                )
                logger.info(f"✓ Loaded converted validation data from {valid_file}")
            else:
                logger.warning(f"Converted validation file not found: {valid_file}, skipping validation")
                valid_data = None
        elif config.data_path == f"{config.project_name}/autotrain-data":
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

        # Apply max_samples to validation data if specified (proportionally)
        if hasattr(config, "max_samples") and config.max_samples is not None and config.max_samples > 0:
            # Use 20% of max_samples for validation or less if validation set is smaller
            valid_max_samples = max(1, int(config.max_samples * 0.2))
            if len(valid_data) > valid_max_samples:
                original_size = len(valid_data)
                valid_data = valid_data.select(range(min(valid_max_samples, len(valid_data))))
                logger.info(f"Limited validation data from {original_size} to {len(valid_data)} samples")

        # Validate validation data columns BEFORE renaming
        if config.trainer in ("dpo", "orpo"):
            required_columns = [config.prompt_text_column, config.text_column, config.rejected_text_column]
            validate_required_columns(valid_data, required_columns, config.trainer.upper(), "validation")
        elif config.trainer == "reward":
            required_columns = [config.text_column, config.rejected_text_column]
            validate_required_columns(valid_data, required_columns, config.trainer.upper(), "validation")

        if config.trainer in ("dpo", "reward", "orpo"):
            if not (config.text_column == "chosen" and config.text_column in valid_data.column_names):
                valid_data = valid_data.rename_column(config.text_column, "chosen")
            if not (
                config.rejected_text_column == "rejected" and config.rejected_text_column in valid_data.column_names
            ):
                valid_data = valid_data.rename_column(config.rejected_text_column, "rejected")
        if config.trainer in ("dpo", "reward"):
            if not (config.prompt_text_column == "prompt" and config.prompt_text_column in valid_data.column_names):
                valid_data = valid_data.rename_column(config.prompt_text_column, "prompt")
    else:
        valid_data = None

    logger.info(f"Train data: {train_data}")
    logger.info(f"Valid data: {valid_data}")

    return train_data, valid_data


def get_tokenizer(config):
    """
    Initializes and returns a tokenizer based on the provided configuration.

    Args:
        config (object): Configuration object containing the following attributes:
            - chat_template (str): The chat template type, either "chatml" or "zephyr".
            - model (str): The model identifier to load the tokenizer from.
            - token (str): The token to use for the tokenizer.
            - model_max_length (int): The maximum length of the model.
            - padding (str): The padding side, either "left" or "right".

    Returns:
        tokenizer (PreTrainedTokenizer): The initialized tokenizer with the specified configuration.
    """
    special_tokens = None
    chat_template = None
    if config.chat_template == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = CHATML_CHAT_TEMPLATE
    elif config.chat_template == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            pad_token=special_tokens.PAD_TOKEN.value,
            bos_token=special_tokens.BOS_TOKEN.value,
            eos_token=special_tokens.EOS_TOKEN.value,
            additional_special_tokens=special_tokens.list(),
            token=config.token,
            trust_remote_code=ALLOW_REMOTE_CODE,
        )
        tokenizer.chat_template = chat_template
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model, token=config.token, trust_remote_code=ALLOW_REMOTE_CODE
        )
        if tokenizer.chat_template is None:
            tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # Determine the effective model_max_length
    # Priority: 1) User-specified value, 2) Auto-detected from model, 3) Fallback to 2048
    DEFAULT_MODEL_MAX_LENGTH = 2048

    if config.model_max_length != DEFAULT_MODEL_MAX_LENGTH:
        # User explicitly set model_max_length, use it
        effective_max_length = config.model_max_length
        logger.info(f"Using user-specified model_max_length={effective_max_length}")
    else:
        # Try to auto-detect from model config
        detected_max = get_model_max_position_embeddings(config.model, config.token)
        if detected_max is not None:
            effective_max_length = detected_max
            # Log if this differs significantly from the default
            if detected_max > DEFAULT_MODEL_MAX_LENGTH:
                logger.info(
                    f"Model supports {detected_max} tokens. Using auto-detected value instead of default {DEFAULT_MODEL_MAX_LENGTH}. "
                    f"Override with --model-max-length if needed."
                )
        else:
            # Fallback to default with warning
            effective_max_length = DEFAULT_MODEL_MAX_LENGTH
            logger.warning(
                f"Could not auto-detect model's max context length. Using default {DEFAULT_MODEL_MAX_LENGTH}. "
                f"Set --model-max-length explicitly if your model supports longer sequences."
            )

    # Only override tokenizer's model_max_length if we have a different value
    if tokenizer.model_max_length != effective_max_length:
        tokenizer.model_max_length = effective_max_length

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if config.padding in ("left", "right"):
        tokenizer.padding_side = config.padding

    # NOTE: We no longer modify add_bos_token here.
    # Instead, we strip BOS from the text column in process_data_with_chat_template.
    # This ensures the saved tokenizer retains its original settings for correct inference.

    return tokenizer


def process_data_with_chat_template(config, tokenizer, train_data, valid_data):
    """
    Processes training and validation data using a specified chat template.

    Args:
        config (object): Configuration object containing settings and parameters.
        tokenizer (object): Tokenizer object used for tokenizing the data.
        train_data (Dataset): Training dataset to be processed.
        valid_data (Dataset): Validation dataset to be processed.

    Returns:
        tuple: A tuple containing the processed training and validation datasets.

    Notes:
        - Uses the unified rendering system for consistent chat template application.
        - Supports multiple chat formats via the ChatFormat parameter.
        - For ORPO/DPO, the `prompt` will be extracted from chosen messages.
        - If `config.valid_split` is not None, the validation data will also be processed.
        - Skips processing if data is already formatted (from auto-conversion).
    """
    # Check if data is already formatted (from auto-conversion)
    # If text column contains template tokens, it's already been processed
    if len(train_data) > 0:
        sample_text = train_data[0].get(config.text_column, "")
        if isinstance(sample_text, str):
            # Common chat template tokens across different models:
            # Gemma: <bos>, <eos>, <start_of_turn>, <end_of_turn>
            # ChatML/Qwen: <|im_start|>, <|im_end|>, <|endoftext|>
            # Llama 2/Mistral: <s>, </s>, [INST], [/INST]
            # Llama 3: <|begin_of_text|>, <|start_header_id|>, <|eot_id|>, <|end_header_id|>
            # Zephyr/Phi-3: <|user|>, <|assistant|>, <|system|>, <|end|>
            template_tokens = [
                "<bos>",
                "<eos>",
                "<start_of_turn>",
                "<end_of_turn>",  # Gemma
                "<|im_start|>",
                "<|im_end|>",
                "<|endoftext|>",  # ChatML/Qwen
                "<s>",
                "</s>",
                "[INST]",
                "[/INST]",  # Llama 2/Mistral
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|eot_id|>",
                "<|end_header_id|>",  # Llama 3
                "<|user|>",
                "<|assistant|>",
                "<|system|>",
                "<|end|>",  # Zephyr/Phi-3
            ]
            if any(token in sample_text for token in template_tokens):
                logger.info(
                    "Dataset already has formatted text column (from auto-conversion), skipping chat template processing"
                )
                # Pre-tokenize the data to add 'input_ids' column
                # TRL 0.26+ uses 'input_ids' presence to detect already-processed data
                # Without this, TRL tries to apply chat template to 'messages' column
                # which fails for datasets with non-alternating roles (e.g., tool calls)
                logger.info("Pre-tokenizing already-formatted data to signal TRL that processing is complete")

                def tokenize_text(example, tokenizer):
                    # Always use 'text' column - this is where the formatted text is stored
                    text = example.get("text", "")
                    if text and isinstance(text, str):
                        tokenized = tokenizer(text, truncation=False, add_special_tokens=False)
                        return {"input_ids": tokenized["input_ids"]}
                    return {"input_ids": []}

                train_data = train_data.map(
                    tokenize_text,
                    fn_kwargs={"tokenizer": tokenizer},
                    load_from_cache_file=False,
                    desc="Pre-tokenizing training data",
                )
                if valid_data is not None:
                    valid_data = valid_data.map(
                        tokenize_text,
                        fn_kwargs={"tokenizer": tokenizer},
                        load_from_cache_file=False,
                        desc="Pre-tokenizing validation data",
                    )
                # Skip to end of function where BOS stripping and turn marker validation happens
                already_formatted = True
            else:
                already_formatted = False
        else:
            already_formatted = False
    else:
        already_formatted = False

    # Only apply chat template if not already formatted AND apply_chat_template is True
    # The apply_chat_template flag allows users to skip template application for pre-formatted data
    should_apply_template = getattr(config, "apply_chat_template", True)
    if not should_apply_template:
        logger.info("Skipping chat template application (apply_chat_template=False)")

    if not already_formatted and should_apply_template:
        # Map legacy chat template names to new ChatFormat
        chat_format_map = {
            "chatml": "chatml",
            "zephyr": "zephyr",
            "tokenizer": "native",  # Use tokenizer's native apply_chat_template
            "alpaca": "alpaca",
            "vicuna": "vicuna",
            "llama": "llama",
            "mistral": "mistral",
        }

        if config.chat_template and config.chat_template.lower() in chat_format_map:
            from autotrain.rendering import ChatFormat, get_renderer

            logger.info(f"Applying chat template: {config.chat_template}")
            logger.info("For ORPO/DPO, `prompt` will be extracted from chosen messages")

            # Get the appropriate renderer
            format_name = chat_format_map[config.chat_template.lower()]
            try:
                chat_format = ChatFormat[format_name.upper()]
            except KeyError:
                logger.warning(f"Unknown chat format {format_name}, defaulting to ChatML")
                chat_format = ChatFormat.CHATML

            renderer = get_renderer(chat_format, tokenizer)

            # Apply rendering to datasets (disable cache to ensure fresh processing after code changes)
            train_data = train_data.map(
                apply_chat_template_unified,
                fn_kwargs={
                    "renderer": renderer,
                    "config": config,
                },
                load_from_cache_file=False,
            )
            if config.valid_split is not None:
                valid_data = valid_data.map(
                    apply_chat_template_unified,
                    fn_kwargs={
                        "renderer": renderer,
                        "config": config,
                    },
                    load_from_cache_file=False,
                )
        elif config.chat_template:
            # Fall back to legacy implementation for custom templates
            logger.info("Using legacy chat template application")
            train_data = train_data.map(
                apply_chat_template,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "config": config,
                },
                load_from_cache_file=False,
            )
            if config.valid_split is not None:
                valid_data = valid_data.map(
                    apply_chat_template,
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "config": config,
                    },
                    load_from_cache_file=False,
                )

    # Pre-tokenize to add 'input_ids' column for TRL 0.26+ compatibility
    # TRL uses 'input_ids' presence to detect already-processed data
    # Without this, TRL tries to apply chat template to 'messages' column
    # which fails for datasets with non-alternating roles (e.g., tool calls)
    if config.chat_template and "text" in train_data.column_names:
        logger.info("Pre-tokenizing data to signal TRL that processing is complete")

        def tokenize_text_column(example, tokenizer):
            # Always use 'text' column - this is where the formatted text is stored
            text = example.get("text", "")
            if text and isinstance(text, str):
                tokenized = tokenizer(text, truncation=False, add_special_tokens=False)
                return {"input_ids": tokenized["input_ids"]}
            return {"input_ids": []}

        train_data = train_data.map(
            tokenize_text_column,
            fn_kwargs={"tokenizer": tokenizer},
            load_from_cache_file=False,
            desc="Pre-tokenizing training data",
        )
        if valid_data is not None:
            valid_data = valid_data.map(
                tokenize_text_column,
                fn_kwargs={"tokenizer": tokenizer},
                load_from_cache_file=False,
                desc="Pre-tokenizing validation data",
            )

    # Add completion_mask for response-only training (SFT label masking)
    # This marks which tokens are assistant responses (1) vs prompts (0)
    # Trigger if: chat_template is set OR data was already formatted (pre-processed externally)
    # This allows users with pre-formatted data to still get response-only loss
    should_add_mask = (
        (config.chat_template or already_formatted) and "text" in train_data.column_names and config.trainer == "sft"
    )
    if should_add_mask:
        response_template = get_response_template(tokenizer)
        if response_template:
            logger.info(f"Adding completion_mask for response-only training (template: {repr(response_template)})")
            train_data = add_completion_mask(train_data, tokenizer, response_template, "text")
            if valid_data is not None:
                valid_data = add_completion_mask(valid_data, tokenizer, response_template, "text")

    # Strip BOS from all processed data to prevent double BOS during training
    # The tokenizer will add BOS during tokenization (if add_bos_token=True)
    # This way the saved tokenizer retains original settings for correct inference
    bos = getattr(tokenizer, "bos_token", None)
    if bos and len(train_data) > 0:
        sample_text = train_data[0].get(config.text_column, "")
        if isinstance(sample_text, str) and sample_text.startswith(bos):
            logger.info(f"Stripping BOS token from processed data to prevent double BOS during training")

            def strip_bos_from_text(example):
                if config.text_column in example and isinstance(example[config.text_column], str):
                    example[config.text_column] = strip_bos_token(example[config.text_column], tokenizer)
                return example

            train_data = train_data.map(strip_bos_from_text, load_from_cache_file=False)
            if valid_data is not None:
                valid_data = valid_data.map(strip_bos_from_text, load_from_cache_file=False)

    # Validate and fix turn markers based on model's expected EOS token
    # This ensures the dataset uses the correct end-of-turn markers for the model
    train_data, valid_data = validate_and_fix_turn_markers(tokenizer, train_data, valid_data, config)

    # Save processed datasets if configured (Path 2 - normal training)
    save_mode = getattr(config, "save_processed_data", "auto")
    if save_mode != "none":
        try:
            from autotrain.data_utils import save_processed_datasets

            save_processed_datasets(
                train_data=train_data,
                valid_data=valid_data,
                project_name=config.project_name,
                train_split=config.train_split,
                valid_split=config.valid_split,
                source_path=config.data_path,
                username=getattr(config, "username", None),
                token=getattr(config, "token", None),
                save_mode=save_mode,
            )
        except Exception as e:
            logger.warning(f"Could not save processed datasets: {e}")

    return train_data, valid_data


def configure_logging_steps(config, train_data, valid_data):
    """
    Configures the logging steps for training based on the provided configuration and data.

    Parameters:
    config (object): Configuration object containing training parameters, including `logging_steps`, `valid_split`, and `batch_size`.
    train_data (iterable): Training dataset.
    valid_data (iterable): Validation dataset.

    Returns:
    int: The number of logging steps to be used during training.

    Notes:
    - If `config.logging_steps` is set to -1, the function calculates logging steps based on 20% of the length of the validation data (if `valid_split` is provided) or the training data.
    - The calculated logging steps are constrained to be between 1 and 25.
    - If `config.logging_steps` is not -1, the function uses the provided value.
    """
    logger.info("configuring logging steps")
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
    return logging_steps


def configure_training_args(config, logging_steps):
    """
    Configures the training arguments for a language model based on the provided configuration.

    TODO: REFACTOR to accept trainer_type parameter for trainer-specific settings.
    Currently uses one-size-fits-all defaults (e.g., remove_unused_columns=False for DPO/ORPO),
    requiring trainers like SFT to override incompatible settings. This is poor design and should
    be refactored to handle trainer-specific requirements properly.

    Args:
        config (object): Configuration object containing various training parameters.
        logging_steps (int): Number of steps between logging events.

    Returns:
        dict: A dictionary containing the configured training arguments.

    The configuration object `config` should have the following attributes:
        - project_name (str): The name of the project, used as the output directory.
        - batch_size (int): Batch size for both training and evaluation.
        - lr (float): Learning rate.
        - epochs (int): Number of training epochs.
        - eval_strategy (str): Evaluation strategy, e.g., "steps" or "epoch".
        - valid_split (float or None): Validation split ratio. If None, evaluation is disabled.
        - save_total_limit (int): Maximum number of checkpoints to save.
        - gradient_accumulation (int): Number of gradient accumulation steps.
        - log (str): Logging destination, e.g., "tensorboard".
        - auto_find_batch_size (bool): Whether to automatically find the optimal batch size.
        - scheduler (str): Learning rate scheduler type.
        - optimizer (str): Optimizer type.
        - warmup_ratio (float): Warmup ratio for learning rate scheduling.
        - weight_decay (float): Weight decay for the optimizer.
        - max_grad_norm (float): Maximum gradient norm for clipping.
        - disable_gradient_checkpointing (bool): Whether to disable gradient checkpointing.
        - peft (bool): Whether to use Parameter-Efficient Fine-Tuning (PEFT).
        - quantization (str): Quantization type, e.g., "int4" or "int8".
        - mixed_precision (str): Mixed precision type, e.g., "fp16" or "bf16".

    The function also sets additional training arguments based on the provided configuration,
    such as enabling gradient checkpointing and mixed precision training.
    """
    logger.info("configuring training args")
    training_args = dict(
        output_dir=config.project_name,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        eval_strategy=config.eval_strategy if config.valid_split is not None else "no",
        logging_steps=logging_steps,
        save_total_limit=config.save_total_limit,
        save_strategy=config.save_strategy,
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
        ddp_timeout=config.ddp_timeout,
        gradient_checkpointing=not config.disable_gradient_checkpointing,
        remove_unused_columns=False,
    )

    # Add save_steps when using step-based saving
    if config.save_strategy == "steps":
        training_args["save_steps"] = config.save_steps

    if not config.disable_gradient_checkpointing:
        if config.peft and config.quantization in ("int4", "int8"):
            training_args["gradient_checkpointing_kwargs"] = {"use_reentrant": True}
        else:
            training_args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    if config.mixed_precision == "fp16":
        training_args["fp16"] = True
    if config.mixed_precision == "bf16":
        training_args["bf16"] = True

    return training_args


def get_resume_checkpoint(config):
    """Resolve resume_from_checkpoint to a path or None."""
    resume = getattr(config, "resume_from_checkpoint", None)
    if resume is None:
        return None
    if resume.lower() in ("true", "auto", "latest"):
        import glob

        checkpoints = sorted(
            glob.glob(os.path.join(config.project_name, "checkpoint-*")),
            key=lambda x: int(x.rsplit("-", 1)[-1]) if x.rsplit("-", 1)[-1].isdigit() else 0,
        )
        if checkpoints:
            latest = checkpoints[-1]
            logger.info(f"Auto-detected checkpoint for resume: {latest}")
            return latest
        logger.warning(
            f"resume_from_checkpoint='auto' but no checkpoints found in {config.project_name}. Starting fresh."
        )
        return None
    elif os.path.isdir(resume):
        logger.info(f"Resuming from explicit checkpoint: {resume}")
        return resume
    else:
        logger.warning(f"resume_from_checkpoint path not found: {resume}. Starting fresh.")
        return None


def configure_block_size(config, tokenizer):
    """
    Configures the block size for the given configuration and tokenizer.

    This function sets the `block_size` attribute in the `config` object based on the `tokenizer`'s maximum model length.
    If `config.block_size` is -1, it is set to None. If `config.block_size` is None, it defaults to the tokenizer's
    `model_max_length` but not exceeding 1024. If `config.block_size` is specified and exceeds the tokenizer's
    `model_max_length`, a warning is logged and the block size is set to the tokenizer's `model_max_length`.

    Args:
        config (object): Configuration object that contains the `block_size` attribute.
        tokenizer (object): Tokenizer object that contains the `model_max_length` attribute.

    Returns:
        object: The updated configuration object with the `block_size` attribute set.
    """
    if config.block_size == -1:
        config.block_size = None

    if config.block_size is None:
        # Use getattr with fallback for tokenizers that don't have model_max_length
        block_size = getattr(tokenizer, "model_max_length", 1024)
        # Ensure block_size is a valid numeric value for comparison
        if isinstance(block_size, (int, float)) and block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
        elif not isinstance(block_size, (int, float)):
            # If block_size is not numeric, use default
            logger.debug(f"Invalid block_size type {type(block_size)}, using default 1024")
            block_size = 1024
    else:
        # Use getattr with fallback for tokenizers that don't have model_max_length
        tokenizer_max = getattr(tokenizer, "model_max_length", float("inf"))
        # Ensure both values are numeric before comparison
        if isinstance(config.block_size, (int, float)) and isinstance(tokenizer_max, (int, float)):
            if config.block_size > tokenizer_max:
                logger.warning(
                    f"The block_size passed ({config.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer_max}). Using block_size={tokenizer_max}."
                )
            block_size = min(config.block_size, tokenizer_max)
        else:
            # If types are incompatible, use config value or default
            block_size = config.block_size if isinstance(config.block_size, (int, float)) else 1024

    config.block_size = block_size

    logger.info(f"Using block size {block_size}")
    return config


def get_callbacks(config, train_data=None, valid_data=None, model=None, tokenizer=None):
    """
    Generate a list of callback instances based on the provided configuration.

    This function creates a list of callback instances that are used during the training process.
    It includes default callbacks for logging and training start, and conditionally adds callbacks
    for saving and loading PEFT models based on the configuration and environment settings.

    Args:
        config (object): Configuration object containing training settings and parameters.
        train_data (Dataset, optional): Training dataset for evaluation callbacks.
        valid_data (Dataset, optional): Validation dataset for evaluation callbacks.
        model (PreTrainedModel, optional): Model for evaluation callbacks.
        tokenizer (PreTrainedTokenizer, optional): Tokenizer for evaluation callbacks.

    Returns:
        list: A list of callback instances to be used during training.
    """
    is_deepspeed_enabled = os.environ.get("ACCELERATE_USE_DEEPSPEED", "False").lower() == "true"
    callbacks = [UploadLogs(config=config), LossLoggingCallback(), TrainStartCallback()]
    if config.peft and not is_deepspeed_enabled:
        callbacks.append(SavePeftModelCallback)
        if config.valid_split is not None:
            callbacks.append(LoadBestPeftModelCallback)

    # Add enhanced evaluation callbacks if enabled
    if hasattr(config, "use_enhanced_eval") and config.use_enhanced_eval:
        try:
            from autotrain.evaluation import BestModelCallback, MetricsLoggerCallback, PeriodicEvalCallback
            from autotrain.evaluation.evaluator import EvaluationConfig, Evaluator

            # Parse metrics
            metrics = (
                config.eval_metrics.split(",")
                if hasattr(config, "eval_metrics") and config.eval_metrics
                else ["perplexity"]
            )

            # Check if we have the required components to create evaluator
            if valid_data is not None and model is not None and tokenizer is not None:
                # Create evaluation config
                eval_config = EvaluationConfig(
                    metrics=metrics,
                    batch_size=(
                        config.per_device_eval_batch_size if hasattr(config, "per_device_eval_batch_size") else 8
                    ),
                    save_predictions=(
                        config.eval_save_predictions if hasattr(config, "eval_save_predictions") else False
                    ),
                    task="language_modeling",
                )

                # Create evaluator instance
                evaluator = Evaluator(model=model, tokenizer=tokenizer, config=eval_config)

                # Add periodic evaluation callback with correct parameters
                callbacks.append(
                    PeriodicEvalCallback(
                        evaluator=evaluator,
                        eval_dataset=valid_data,
                        eval_steps=(
                            config.logging_steps
                            if hasattr(config, "logging_steps") and config.logging_steps > 0
                            else 100
                        ),
                        metrics=metrics,
                    )
                )

                # Add best model tracking
                callbacks.append(
                    BestModelCallback(
                        metric=(
                            config.sweep_metric
                            if hasattr(config, "sweep_metric") and config.sweep_metric
                            else "eval_loss"
                        ),
                        mode="min" if "loss" in (config.sweep_metric or "eval_loss") else "max",
                    )
                )

                # Add metrics logger
                callbacks.append(MetricsLoggerCallback())

                logger.info(f"Enhanced evaluation enabled with metrics: {metrics}")
            else:
                logger.warning(
                    "Enhanced evaluation requested but required components not available (valid_data, model, or tokenizer)"
                )
        except ImportError:
            logger.warning("Enhanced evaluation requested but evaluation module not available")
        except Exception as e:
            logger.warning(f"Could not initialize enhanced evaluation: {e}")

    return callbacks


def get_model(config, tokenizer):
    """
    Loads and configures a language model based on the provided configuration and tokenizer.

    Args:
        config (Namespace): Configuration object containing model parameters and settings.
            - model (str): The model name or path.
            - token (str): Token for accessing the model.
            - unsloth (bool): Flag to determine if unsloth is used.
            - trainer (str): Type of trainer to use.
            - target_modules (str): Target modules for unsloth.
            - peft (bool): Flag to determine if PEFT (Parameter-Efficient Fine-Tuning) is used.
            - quantization (str): Quantization type, either "int4" or "int8".
            - mixed_precision (str): Mixed precision type, either "fp16" or "bf16".
            - block_size (int): Maximum sequence length.
            - lora_r (int): LoRA rank.
            - lora_alpha (int): LoRA alpha.
            - lora_dropout (float): LoRA dropout rate.
            - seed (int): Random seed.
            - disable_gradient_checkpointing (bool): Flag to disable gradient checkpointing.
            - use_flash_attention_2 (bool): Flag to use flash attention 2.
        tokenizer (PreTrainedTokenizer): Tokenizer to use with the model.

    Returns:
        PreTrainedModel: The configured language model.

    Raises:
        ImportError: If unsloth is not available when required.
    """
    model_config = AutoConfig.from_pretrained(
        config.model,
        token=config.token,
        trust_remote_code=ALLOW_REMOTE_CODE,
    )
    model_type = model_config.model_type
    unsloth_target_modules = None
    can_use_unloth = False

    if config.unsloth and is_unsloth_available() and config.trainer in ("default", "sft"):
        can_use_unloth = True

    if model_type in ("llama", "mistral", "gemma", "qwen2") and config.unsloth:
        if config.target_modules.strip().lower() == "all-linear":
            unsloth_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            unsloth_target_modules = get_target_modules(config)
    else:
        can_use_unloth = False

    logger.info(f"Can use unsloth: {can_use_unloth}")
    if can_use_unloth:
        from unsloth import FastLanguageModel

        load_in_4bit = False
        load_in_8bit = False
        if config.peft and config.quantization == "int4":
            load_in_4bit = True
        elif config.peft and config.quantization == "int8":
            load_in_8bit = True

        dtype = None
        if config.mixed_precision == "fp16":
            dtype = torch.float16
        elif config.mixed_precision == "bf16":
            dtype = torch.bfloat16

        model, _ = FastLanguageModel.from_pretrained(
            model_name=config.model,
            token=config.token,
            trust_remote_code=ALLOW_REMOTE_CODE,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            max_seq_length=config.block_size,
            dtype=dtype,
        )
        if config.peft:
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora_r,
                target_modules=unsloth_target_modules,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=config.seed,
                max_seq_length=config.block_size,
                use_rslora=False,
                loftq_config=None,
            )
        return model
    else:
        logger.warning("Unsloth not available, continuing without it...")

    logger.info("loading model config...")
    model_config = AutoConfig.from_pretrained(
        config.model,
        token=config.token,
        trust_remote_code=ALLOW_REMOTE_CODE,
        use_cache=config.disable_gradient_checkpointing,
    )

    logger.info("loading model...")
    if config.peft:
        # Check if CUDA is available for quantization
        cuda_available = torch.cuda.is_available()

        if config.quantization and not cuda_available:
            logger.warning(
                f"Quantization {config.quantization} requested but CUDA not available. Skipping quantization."
            )
            bnb_config = None
        elif config.quantization == "int4" and cuda_available:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
        elif config.quantization == "int8" and cuda_available:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None

        # Only pass use_flash_attention_2 if the model supports it
        model_kwargs = {
            "config": model_config,
            "token": config.token,
            "trust_remote_code": ALLOW_REMOTE_CODE,
        }

        # Only add quantization config if we have one
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        # Set device map and dtype for proper placement
        if torch.cuda.is_available():
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size > 1:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                model_kwargs["device_map"] = {"": local_rank}
            else:
                model_kwargs["device_map"] = "auto"
            if config.mixed_precision == "bf16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif config.mixed_precision == "fp16":
                model_kwargs["torch_dtype"] = torch.float16
        elif torch.backends.mps.is_available():
            # For MPS, we'll load to CPU first then move manually
            pass

        # Check if model supports flash attention (skip for Gemma3 and similar)
        if config.use_flash_attention_2 and "gemma" not in config.model.lower():
            model_kwargs["use_flash_attention_2"] = config.use_flash_attention_2

        # Handle attention implementation
        if hasattr(config, "attn_implementation") and config.attn_implementation:
            # User explicitly specified attention implementation
            model_kwargs["attn_implementation"] = config.attn_implementation
            logger.info(f"Using {config.attn_implementation} attention implementation (user specified)")
        elif "gemma" in config.model.lower():
            # Auto-detect Gemma and use eager for better MPS performance
            model_kwargs["attn_implementation"] = "eager"
            logger.info("Using eager attention implementation for Gemma model (auto-detected)")

        model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)

        # Move model to MPS if available and not using device_map
        # But respect environment variables that disable MPS
        mps_enabled = (
            torch.backends.mps.is_available()
            and "device_map" not in model_kwargs
            and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1"
            and os.environ.get("AUTOTRAIN_DISABLE_MPS", "0") != "1"
        )

        if mps_enabled:
            logger.info("Moving model to MPS device")
            model = model.to("mps")

            # Also convert to appropriate dtype based on mixed_precision setting
            if config.mixed_precision == "bf16":
                logger.info("Converting model to bfloat16")
                model = model.to(torch.bfloat16)
            elif config.mixed_precision == "fp16":
                logger.info("Converting model to float16")
                model = model.to(torch.float16)
    else:
        # Only pass use_flash_attention_2 if the model supports it
        model_kwargs = {
            "config": model_config,
            "token": config.token,
            "trust_remote_code": ALLOW_REMOTE_CODE,
        }

        # Set device map and dtype for proper placement
        if torch.cuda.is_available():
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size > 1:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                model_kwargs["device_map"] = {"": local_rank}
            else:
                model_kwargs["device_map"] = "auto"
            # Set dtype based on mixed_precision to avoid loading in float32
            if config.mixed_precision == "bf16":
                model_kwargs["torch_dtype"] = torch.bfloat16
                logger.info("Loading model in bfloat16 (mixed_precision=bf16)")
            elif config.mixed_precision == "fp16":
                model_kwargs["torch_dtype"] = torch.float16
                logger.info("Loading model in float16 (mixed_precision=fp16)")
        elif torch.backends.mps.is_available():
            # For MPS, we'll load to CPU first then move manually
            pass

        # Check if model supports flash attention (skip for Gemma3 and similar)
        if config.use_flash_attention_2 and "gemma" not in config.model.lower():
            model_kwargs["use_flash_attention_2"] = config.use_flash_attention_2

        # Handle attention implementation
        if hasattr(config, "attn_implementation") and config.attn_implementation:
            # User explicitly specified attention implementation
            model_kwargs["attn_implementation"] = config.attn_implementation
            logger.info(f"Using {config.attn_implementation} attention implementation (user specified)")
        elif "gemma" in config.model.lower():
            # Auto-detect Gemma and use eager for better MPS performance
            model_kwargs["attn_implementation"] = "eager"
            logger.info("Using eager attention implementation for Gemma model (auto-detected)")

        model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)

        # Move model to MPS if available and not using device_map
        # But respect environment variables that disable MPS
        mps_enabled = (
            torch.backends.mps.is_available()
            and "device_map" not in model_kwargs
            and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1"
            and os.environ.get("AUTOTRAIN_DISABLE_MPS", "0") != "1"
        )

        if mps_enabled:
            logger.info("Moving model to MPS device")
            model = model.to("mps")

            # Also convert to appropriate dtype based on mixed_precision setting
            if config.mixed_precision == "bf16":
                logger.info("Converting model to bfloat16")
                model = model.to(torch.bfloat16)
            elif config.mixed_precision == "fp16":
                logger.info("Converting model to float16")
                model = model.to(torch.float16)

    logger.info(f"model dtype: {model.dtype}")
    model.resize_token_embeddings(len(tokenizer))

    if config.trainer != "default":
        return model

    if config.peft:
        logger.info("preparing peft model...")
        if config.quantization is not None:
            gradient_checkpointing_kwargs = {}
            if not config.disable_gradient_checkpointing:
                if config.quantization in ("int4", "int8"):
                    gradient_checkpointing_kwargs = {"use_reentrant": True}
                else:
                    gradient_checkpointing_kwargs = {"use_reentrant": False}
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=not config.disable_gradient_checkpointing,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            )
        else:
            model.enable_input_require_grads()

        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=get_target_modules(config),
        )
        model = get_peft_model(model, peft_config)

    return model
