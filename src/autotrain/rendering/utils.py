"""
Utility functions for message rendering
========================================

Helper functions for common rendering tasks.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from .message_renderer import ChatFormat, Conversation, Message, RenderConfig, TokenWeight, get_renderer


# Cache for tool role support detection per tokenizer
_tool_role_support_cache: Dict[str, bool] = {}

# Cache for tool_calls field support detection per tokenizer
_tool_calls_support_cache: Dict[str, bool] = {}

# Cache for tools parameter support detection per tokenizer
_tools_support_cache: Dict[str, bool] = {}


def _get_tokenizer_id(tokenizer: AutoTokenizer) -> str:
    """Get a unique identifier for a tokenizer for caching."""
    if hasattr(tokenizer, "name_or_path"):
        return tokenizer.name_or_path
    return str(id(tokenizer))


def check_tool_role_support(tokenizer: AutoTokenizer) -> bool:
    """Check if a tokenizer's chat template supports the 'tool' role.

    Tests by attempting to render a minimal conversation with a tool message.
    The result is cached per tokenizer for performance.

    Args:
        tokenizer: The tokenizer to check

    Returns:
        True if tokenizer supports tool role, False otherwise
    """
    tokenizer_id = _get_tokenizer_id(tokenizer)

    if tokenizer_id in _tool_role_support_cache:
        return _tool_role_support_cache[tokenizer_id]

    # Test with a minimal tool conversation
    test_messages = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "test"},
        {"role": "tool", "content": "test"},
        {"role": "assistant", "content": "test"},
    ]

    try:
        tokenizer.apply_chat_template(test_messages, tokenize=False)
        result = True
    except Exception:
        result = False

    _tool_role_support_cache[tokenizer_id] = result
    return result


def check_tool_calls_support(tokenizer: AutoTokenizer) -> bool:
    """Check if a tokenizer's chat template supports the 'tool_calls' field.

    Tests by attempting to render a minimal conversation with an assistant message
    that has tool_calls. The result is cached per tokenizer for performance.

    Args:
        tokenizer: The tokenizer to check

    Returns:
        True if tokenizer supports tool_calls field, False otherwise
    """
    tokenizer_id = _get_tokenizer_id(tokenizer)

    if tokenizer_id in _tool_calls_support_cache:
        return _tool_calls_support_cache[tokenizer_id]

    # Test with a minimal tool_calls message
    test_messages = [
        {"role": "user", "content": "test"},
        {
            "role": "assistant",
            "content": None,  # Often None when tool_calls is present
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "test_func", "arguments": "{}"},
                }
            ],
        },
    ]

    try:
        result_text = tokenizer.apply_chat_template(test_messages, tokenize=False)
        # Check if the tool call information appears in output (model handled it)
        # If it just silently dropped it, we should serialize instead
        if "test_func" in result_text or "call_123" in result_text:
            result = True
        else:
            # Tokenizer didn't include tool_calls in output - doesn't support it
            result = False
    except Exception:
        result = False

    _tool_calls_support_cache[tokenizer_id] = result
    return result


def check_tools_support(tokenizer: AutoTokenizer) -> bool:
    """Check if a tokenizer's apply_chat_template supports the 'tools' parameter.

    Tests by attempting to call apply_chat_template with a tools parameter
    and verifying the tool information appears in the output. Some tokenizers
    (like Gemma) accept the parameter but silently ignore it.

    The result is cached per tokenizer for performance.

    Args:
        tokenizer: The tokenizer to check

    Returns:
        True if tokenizer supports and uses tools parameter, False otherwise
    """
    tokenizer_id = _get_tokenizer_id(tokenizer)

    if tokenizer_id in _tools_support_cache:
        return _tools_support_cache[tokenizer_id]

    # Test with a minimal tools definition
    test_messages = [{"role": "user", "content": "test"}]
    test_tools = [
        {
            "type": "function",
            "function": {
                "name": "unique_test_tool_xyz123",
                "description": "A test tool for checking support",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    try:
        output = tokenizer.apply_chat_template(test_messages, tools=test_tools, tokenize=False)
        # Check if the tool information actually appears in the output
        # Some tokenizers accept the parameter but silently ignore it (e.g., Gemma)
        if "unique_test_tool_xyz123" in output or "test_tool" in output:
            result = True
        else:
            # Tokenizer accepted the parameter but didn't use it
            result = False
    except (TypeError, Exception):
        # TypeError if tools parameter not accepted, other exceptions for other failures
        result = False

    _tools_support_cache[tokenizer_id] = result
    return result


def format_tools_as_text(tools: List[Dict]) -> str:
    """Format tool definitions as text for injection into system prompt.

    Converts OpenAI-style tool definitions to a readable text format that can
    be included in the system prompt for models that don't support native tools.

    Args:
        tools: List of tool definitions in OpenAI format

    Returns:
        Formatted string describing available tools
    """
    import json

    if not tools:
        return ""

    tool_descriptions = []
    for tool in tools:
        if isinstance(tool, dict):
            # Handle OpenAI format: {"type": "function", "function": {...}}
            if "function" in tool:
                func = tool["function"]
                name = func.get("name", "unknown")
                desc = func.get("description", "No description")
                params = func.get("parameters", {})

                # Format parameters
                props = params.get("properties", {}) or {}
                required = params.get("required", []) or []

                param_strs = []
                for param_name, param_info in props.items():
                    # Skip None values (some datasets have None for unused params)
                    if param_info is None:
                        continue
                    param_type = param_info.get("type", "any") if isinstance(param_info, dict) else "any"
                    param_desc = param_info.get("description", "") if isinstance(param_info, dict) else ""
                    req_marker = " (required)" if param_name in required else ""
                    param_strs.append(f"    - {param_name}: {param_type}{req_marker} - {param_desc}")

                params_text = "\n".join(param_strs) if param_strs else "    (no parameters)"

                tool_descriptions.append(f"- {name}: {desc}\n  Parameters:\n{params_text}")
            else:
                # Simple format: just the dict
                tool_descriptions.append(f"- {json.dumps(tool, ensure_ascii=False)}")

    header = "You have access to the following tools:\n\n"
    footer = '\n\nTo use a tool, respond with a tool call in this format: [Tool Call] {"tool": "tool_name", "arguments": {...}}'

    return header + "\n\n".join(tool_descriptions) + footer


def inject_tools_into_messages(
    messages: List[Dict[str, Any]],
    tools: List[Dict],
) -> List[Dict[str, Any]]:
    """Inject tool definitions into messages for models that don't support native tools.

    This function:
    1. Formats tool definitions as human-readable text
    2. Injects them into the system message (appended) or first user message (prepended)

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        tools: List of tool definitions in OpenAI format

    Returns:
        Messages with tool definitions injected
    """
    if not tools:
        return messages

    tools_text = format_tools_as_text(tools)
    if not tools_text:
        return messages

    # Make a copy to avoid mutating the original
    result = []
    tools_injected = False

    for msg in messages:
        msg_copy = msg.copy()
        content = msg_copy.get("content", "") or ""

        if not tools_injected:
            if msg_copy.get("role") == "system":
                # Append tools to system message
                msg_copy["content"] = f"{content}\n\n{tools_text}" if content else tools_text
                tools_injected = True
            elif msg_copy.get("role") == "user":
                # No system message found, prepend tools to first user message
                msg_copy["content"] = f"{tools_text}\n\n{content}" if content else tools_text
                tools_injected = True

        result.append(msg_copy)

    return result


def serialize_tool_calls_to_content(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Serialize tool_calls field into message content for models that don't support it natively.

    This function:
    1. Finds assistant messages with tool_calls field
    2. Appends the tool calls as JSON to the message content
    3. Returns messages without the tool_calls field (content contains the JSON)

    Args:
        messages: List of message dicts, potentially with tool_calls

    Returns:
        Messages with tool_calls serialized into content

    Example:
        Input:
        {
            "role": "assistant",
            "content": "Let me check that for you.",
            "tool_calls": [{"function": {"name": "search", "arguments": "{\"query\": \"weather\"}"}}]
        }

        Output:
        {
            "role": "assistant",
            "content": "Let me check that for you.\n[Tool Call] {\"tool\": \"search\", \"arguments\": {\"query\": \"weather\"}}"
        }
    """
    import json

    processed = []
    for msg in messages:
        msg_copy = dict(msg)
        tool_calls = msg_copy.pop("tool_calls", None)

        if tool_calls and msg_copy.get("role") == "assistant":
            content = msg_copy.get("content") or ""

            # Serialize each tool call
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")

                # Parse arguments if they're a string
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass  # Keep as string if can't parse

                tool_json = json.dumps({"tool": tool_name, "arguments": args}, ensure_ascii=False)
                content = f"{content}\n[Tool Call] {tool_json}" if content else f"[Tool Call] {tool_json}"

            msg_copy["content"] = content.strip()

        # Remove tool_call_id from tool response messages (handled separately by tool role conversion)
        msg_copy.pop("tool_call_id", None)

        processed.append(msg_copy)

    return processed


def preprocess_messages_for_tool_role(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Preprocess messages to handle 'tool' role for tokenizers that don't support it.

    This function:
    1. Converts 'tool' role to 'user' role with [Tool Result] prefix
    2. Merges consecutive same-role messages to maintain strict alternation

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Preprocessed messages compatible with strict-alternation tokenizers
    """
    if not messages:
        return messages

    processed = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""

        # Convert 'tool' or 'function' role to 'user' (tool responses are external input like user messages)
        # Note: 'function' is the older OpenAI format, 'tool' is the newer format
        if role in ("tool", "function"):
            role = "user"
            content = f"[Tool Result] {content}"

        # Check if we need to merge with previous message (same role after conversion)
        if processed and processed[-1]["role"] == role:
            # Merge consecutive same-role messages
            processed[-1]["content"] = f"{processed[-1]['content']}\n\n{content}"
        else:
            processed.append({"role": role, "content": content})

    return processed


def fix_message_alternation(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Fix messages to ensure proper user/assistant alternation.

    This handles:
    1. Consecutive same-role messages (merge them)
    2. System → assistant without user in between (insert placeholder user)
    3. Assistant at start without preceding user (insert placeholder user)

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Fixed messages with proper alternation
    """
    if not messages:
        return messages

    # Step 1: Merge consecutive same-role messages
    merged = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""

        if merged and merged[-1]["role"] == role:
            # Merge content with newline separator
            merged[-1]["content"] = f"{merged[-1]['content']}\n{content}"
        else:
            merged.append({"role": role, "content": content})

    # Step 2: Handle alternation issues
    result = []
    for msg in merged:
        role = msg["role"]

        if role == "system":
            result.append(msg)
        elif role == "assistant":
            # Check if we need to insert a user message before assistant
            if not result or result[-1]["role"] in ("system", "assistant"):
                # Assistant without preceding user - insert placeholder
                result.append({"role": "user", "content": "[Continued]"})
            result.append(msg)
        elif role == "user":
            # Check for consecutive users (shouldn't happen after merge, but safety)
            if result and result[-1]["role"] == "user":
                result[-1]["content"] = f"{result[-1]['content']}\n{msg['content']}"
            else:
                result.append(msg)
        else:
            # Unknown role - just append
            result.append(msg)

    return result


def safe_apply_chat_template(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, Any]],
    tokenize: bool = False,
    add_generation_prompt: bool = False,
    tools: Optional[List[Dict]] = None,
    **kwargs,
) -> str:
    """Safely apply chat template with automatic tool handling and alternation fixing.

    This function automatically:
    1. Detects if the tokenizer supports the 'tools' parameter and injects into messages if needed
    2. Detects if the tokenizer supports the 'tool_calls' field and serializes if needed
    3. Detects if the tokenizer supports the 'tool' role and converts if needed
    4. Fixes alternation issues (consecutive same-role, missing user before assistant)

    Use this instead of directly calling tokenizer.apply_chat_template() when
    messages may have tool_calls, tool role, tools definitions, or alternation issues.

    Args:
        tokenizer: The tokenizer to use
        messages: List of message dicts with 'role', 'content', and optionally 'tool_calls' keys
        tokenize: Whether to return token IDs (default False for string output)
        add_generation_prompt: Whether to add generation prompt
        tools: Optional list of tool definitions in OpenAI format. If provided and the tokenizer
               doesn't support native tools, they will be injected into the system prompt or
               first user message.
        **kwargs: Additional arguments passed to apply_chat_template

    Returns:
        Formatted conversation string (or token IDs if tokenize=True)

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "What's 2+2?"},
        ...     {"role": "assistant", "content": "Let me calculate", "tool_calls": [...]},
        ...     {"role": "tool", "content": "4"},
        ...     {"role": "assistant", "content": "The answer is 4"}
        ... ]
        >>> # For Gemma (no tool support): auto-converts tool_calls to JSON in content
        >>> # For Llama 3.1+ (has tool support): uses native format
        >>> result = safe_apply_chat_template(tokenizer, messages)
    """
    # Check if tokenizer supports native tools parameter
    tokenizer_supports_tools = check_tools_support(tokenizer) if tools and len(tools) > 0 else False

    # If tokenizer doesn't support native tools, inject them into messages
    if tools and len(tools) > 0 and not tokenizer_supports_tools:
        from autotrain import logger

        logger.debug("Tokenizer doesn't support 'tools' parameter, injecting into system prompt")
        messages = inject_tools_into_messages(messages, tools)

    # Check if messages have tool_calls field
    has_tool_calls = any(msg.get("tool_calls") for msg in messages)

    # Only serialize tool_calls if present AND tokenizer doesn't support them natively
    if has_tool_calls and not check_tool_calls_support(tokenizer):
        from autotrain import logger

        logger.debug("Tokenizer doesn't support 'tool_calls' field, serializing to content as JSON")
        messages = serialize_tool_calls_to_content(messages)

    # Check if messages have tool role
    has_tool = any(msg.get("role") == "tool" for msg in messages)

    # Only preprocess if we have tool messages AND tokenizer doesn't support them
    if has_tool and not check_tool_role_support(tokenizer):
        from autotrain import logger

        logger.debug("Tokenizer doesn't support 'tool' role, converting to 'user' with [Tool Result] prefix")
        messages = preprocess_messages_for_tool_role(messages)

    # Build kwargs for apply_chat_template
    template_kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
        **kwargs,
    }

    # Pass tools to native template if tokenizer supports it
    if tokenizer_supports_tools and tools:
        template_kwargs["tools"] = tools

    # Try to apply template
    try:
        return tokenizer.apply_chat_template(messages, **template_kwargs)
    except Exception as e:
        error_str = str(e).lower()

        # Check if it's an alternation error - if so, try to fix it
        if "alternate" in error_str or "must be" in error_str:
            from autotrain import logger

            logger.debug(f"Alternation error detected: {e}, attempting to fix message structure")
            messages = fix_message_alternation(messages)

            try:
                return tokenizer.apply_chat_template(messages, **template_kwargs)
            except Exception as e2:
                from autotrain import logger

                logger.warning(f"Failed to apply chat template even after fixing alternation: {e2}")
                raise e2

        # Different error, re-raise
        raise


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
    # Convert to Conversation if needed, preserving tool_calls
    if isinstance(messages, list):
        conversation = Conversation(
            messages=[
                Message(
                    role=m["role"],
                    content=m.get("content") or "",
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                )
                for m in messages
            ]
        )
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
    # Convert to Conversation if needed, preserving tool_calls
    if isinstance(messages, list):
        conversation = Conversation(
            messages=[
                Message(
                    role=m["role"],
                    content=m.get("content") or "",
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                )
                for m in messages
            ]
        )
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
