"""
Utility functions for message rendering
========================================

Helper functions for common rendering tasks.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from .message_renderer import ChatFormat, Conversation, Message, RenderConfig, TokenWeight, get_renderer


def build_generation_prompt(
    messages: Union[List[Dict[str, str]], Conversation],
    format: Union[ChatFormat, str] = ChatFormat.CHATML,
    tokenizer: Optional[AutoTokenizer] = None,
    add_generation_prompt: bool = True,
) -> str:
    """
    Build a generation prompt from messages.

    Args:
        messages: List of message dicts or Conversation object
        format: Chat format to use
        tokenizer: Optional tokenizer for format detection
        add_generation_prompt: Whether to add generation prompt

    Returns:
        Formatted prompt string
    """
    # Convert to Conversation if needed
    if isinstance(messages, list):
        conversation = Conversation(messages=[Message(role=m["role"], content=m["content"]) for m in messages])
    else:
        conversation = messages

    # Get appropriate renderer
    config = RenderConfig(
        format=format if isinstance(format, ChatFormat) else ChatFormat(format),
        add_generation_prompt=add_generation_prompt,
    )

    # Create dummy tokenizer if not provided
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    renderer = get_renderer(format, tokenizer, config)

    if add_generation_prompt:
        return renderer.build_generation_prompt(conversation)
    else:
        return renderer.render_conversation(conversation)


def parse_response(
    response: str,
    format: Union[ChatFormat, str] = ChatFormat.CHATML,
    tokenizer: Optional[AutoTokenizer] = None,
) -> str:
    """
    Parse a model response to extract content.

    Args:
        response: Raw model output
        format: Chat format used
        tokenizer: Optional tokenizer

    Returns:
        Parsed content string
    """
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    config = RenderConfig(format=format if isinstance(format, ChatFormat) else ChatFormat(format))
    renderer = get_renderer(format, tokenizer, config)

    return renderer.parse_response(response)


def get_stop_sequences(
    format: Union[ChatFormat, str] = ChatFormat.CHATML,
    tokenizer: Optional[AutoTokenizer] = None,
) -> List[str]:
    """
    Get stop sequences for a chat format.

    Args:
        format: Chat format
        tokenizer: Optional tokenizer

    Returns:
        List of stop sequences
    """
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    config = RenderConfig(format=format if isinstance(format, ChatFormat) else ChatFormat(format))
    renderer = get_renderer(format, tokenizer, config)

    return renderer.get_stop_sequences()


def build_supervised_example(
    messages: Union[List[Dict[str, str]], Conversation],
    tokenizer: AutoTokenizer,
    format: Union[ChatFormat, str] = ChatFormat.CHATML,
    max_length: int = 512,
    mask_system: bool = False,
    mask_user: bool = False,
    only_assistant: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Build a supervised training example with proper masking.

    Args:
        messages: Conversation messages
        tokenizer: Tokenizer to use
        format: Chat format
        max_length: Maximum sequence length
        mask_system: Whether to mask system messages
        mask_user: Whether to mask user messages
        only_assistant: Whether to only train on assistant messages

    Returns:
        Dictionary with input_ids, attention_mask, labels, and token_weights
    """
    # Convert to Conversation if needed
    if isinstance(messages, list):
        conversation = Conversation(messages=[Message(role=m["role"], content=m["content"]) for m in messages])
    else:
        conversation = messages

    # Configure rendering
    config = RenderConfig(
        format=format if isinstance(format, ChatFormat) else ChatFormat(format),
        max_length=max_length,
        mask_system=mask_system,
        mask_user=mask_user,
        only_assistant=only_assistant,
    )

    # Get renderer and build example
    renderer = get_renderer(format, tokenizer, config)
    return renderer.build_supervised_example(conversation, tokenizer)


def apply_token_weights(
    input_ids: torch.Tensor,
    weights: Union[List[TokenWeight], torch.Tensor],
    default_weight: float = 1.0,
) -> torch.Tensor:
    """
    Apply token-level weights to input IDs.

    Args:
        input_ids: Input token IDs
        weights: List of TokenWeight objects or weight tensor
        default_weight: Default weight for unspecified tokens

    Returns:
        Weight tensor matching input_ids shape
    """
    if isinstance(weights, torch.Tensor):
        return weights

    # Initialize weight tensor
    weight_tensor = torch.full_like(input_ids, default_weight, dtype=torch.float)

    # Apply individual weights
    for weight in weights:
        weight_tensor = weight.apply_to_tensor(weight_tensor)

    return weight_tensor


def detect_chat_format(
    model_name: str,
    tokenizer: Optional[AutoTokenizer] = None,
) -> ChatFormat:
    """
    Auto-detect chat format from model name or tokenizer.

    Args:
        model_name: Name of the model
        tokenizer: Optional tokenizer with chat template

    Returns:
        Detected ChatFormat
    """
    model_lower = model_name.lower()

    # Check model name patterns
    if "chatml" in model_lower or "qwen" in model_lower:
        return ChatFormat.CHATML
    elif "alpaca" in model_lower:
        return ChatFormat.ALPACA
    elif "vicuna" in model_lower:
        return ChatFormat.VICUNA
    elif "zephyr" in model_lower:
        return ChatFormat.ZEPHYR
    elif "llama" in model_lower:
        return ChatFormat.LLAMA
    elif "mistral" in model_lower:
        return ChatFormat.MISTRAL

    # Check tokenizer chat template if available
    if tokenizer and hasattr(tokenizer, "chat_template"):
        template = str(tokenizer.chat_template)
        if "im_start" in template:
            return ChatFormat.CHATML
        elif "INST" in template:
            return ChatFormat.LLAMA
        elif "### Instruction" in template:
            return ChatFormat.ALPACA

    # Default to ChatML
    return ChatFormat.CHATML


def convert_dataset_to_conversations(
    dataset: Any,
    text_column: str = "text",
    conversation_column: Optional[str] = "conversations",
) -> List[Conversation]:
    """
    Convert a dataset to list of Conversation objects.

    Args:
        dataset: HuggingFace dataset or list of examples
        text_column: Column containing text
        conversation_column: Column containing conversation data

    Returns:
        List of Conversation objects
    """
    conversations = []

    # Handle different dataset formats
    if hasattr(dataset, "column_names"):
        # HuggingFace dataset
        if conversation_column in dataset.column_names:
            # Already has conversations
            for example in dataset:
                conv_data = example[conversation_column]
                if isinstance(conv_data, str):
                    import json

                    conv_data = json.loads(conv_data)
                conversations.append(Conversation.from_dict(conv_data))
        elif text_column in dataset.column_names:
            # Plain text - convert to single turn
            for example in dataset:
                text = example[text_column]
                conv = Conversation(
                    messages=[Message(role="user", content=""), Message(role="assistant", content=text)]
                )
                conversations.append(conv)
    elif isinstance(dataset, list):
        # List of examples
        for example in dataset:
            if isinstance(example, dict):
                if "messages" in example:
                    conversations.append(Conversation.from_dict(example))
                elif "text" in example:
                    conv = Conversation(
                        messages=[Message(role="user", content=""), Message(role="assistant", content=example["text"])]
                    )
                    conversations.append(conv)
            elif isinstance(example, str):
                conv = Conversation(
                    messages=[Message(role="user", content=""), Message(role="assistant", content=example)]
                )
                conversations.append(conv)

    return conversations


def create_chat_template(
    format: Union[ChatFormat, str] = ChatFormat.CHATML,
    system_template: Optional[str] = None,
    user_template: Optional[str] = None,
    assistant_template: Optional[str] = None,
) -> str:
    """
    Create a Jinja2 chat template for a specific format.

    Args:
        format: Chat format to use
        system_template: Custom system template
        user_template: Custom user template
        assistant_template: Custom assistant template

    Returns:
        Jinja2 template string
    """
    format = format if isinstance(format, ChatFormat) else ChatFormat(format)

    if format == ChatFormat.CHATML:
        return """{% for message in messages %}
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|im_start|>assistant\n' }}
{% endif %}"""

    elif format == ChatFormat.ALPACA:
        return """{% if messages[0]['role'] == 'system' %}
{{ messages[0]['content'] + '\n\n' }}
{% endif %}
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '### Instruction:\n' + message['content'] + '\n\n' }}
{% elif message['role'] == 'assistant' %}
{{ '### Response:\n' + message['content'] + '\n\n' }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{ '### Response:\n' }}
{% endif %}"""

    elif format == ChatFormat.LLAMA:
        return """{% if messages[0]['role'] == 'system' %}
{{ '<s>[INST] <<SYS>>\n' + messages[0]['content'] + '\n<</SYS>>\n\n' }}
{% else %}
{{ '<s>[INST] ' }}
{% endif %}
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ message['content'] + ' [/INST]' }}
{% elif message['role'] == 'assistant' %}
{{ ' ' + message['content'] + ' </s>' }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{ ' ' }}
{% endif %}"""

    elif format == ChatFormat.CUSTOM:
        # Build custom template from provided templates
        system = system_template or "{content}"
        user = user_template or "User: {content}"
        assistant = assistant_template or "Assistant: {content}"

        # Build Jinja2 template with custom formats
        # Use Python string formatting to inject the template patterns
        template = "{% for message in messages %}\n"
        template += "{% if message['role'] == 'system' %}\n"
        template += "{{ '" + system.replace("'", "\\'") + "'.replace('{content}', message['content']) }}\n"
        template += "{% elif message['role'] == 'user' %}\n"
        template += "{{ '" + user.replace("'", "\\'") + "'.replace('{content}', message['content']) }}\n"
        template += "{% elif message['role'] == 'assistant' %}\n"
        template += "{{ '" + assistant.replace("'", "\\'") + "'.replace('{content}', message['content']) }}\n"
        template += "{% endif %}\n"
        template += "{% endfor %}"

        return template

    else:
        # Default to a simple format
        return """{% for message in messages %}
{{ message['role'] + ': ' + message['content'] + '\n' }}
{% endfor %}"""
