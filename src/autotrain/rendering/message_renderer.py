"""
Core Message Rendering System
==============================

Handles conversation-to-token conversion with fine-grained control.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer


class ChatFormat(Enum):
    """Supported chat formats."""

    CHATML = "chatml"
    ALPACA = "alpaca"
    VICUNA = "vicuna"
    ZEPHYR = "zephyr"
    LLAMA = "llama"
    MISTRAL = "mistral"
    CUSTOM = "custom"
    NATIVE = "native"  # Use tokenizer's native apply_chat_template


@dataclass
class Message:
    """Represents a single message in a conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Token-level weight for training
    tool_calls: Optional[List[Dict[str, Any]]] = None  # For assistant messages that call tools
    tool_call_id: Optional[str] = None  # For tool response messages


@dataclass
class Conversation:
    """Represents a full conversation."""

    messages: List[Message]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to the conversation."""
        self.messages.append(Message(role=role, content=content, **kwargs))

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary."""
        messages_list = []
        for m in self.messages:
            msg_dict = {"role": m.role, "content": m.content, "weight": m.weight}
            if m.tool_calls:
                msg_dict["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            messages_list.append(msg_dict)
        return {
            "messages": messages_list,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Create conversation from dictionary."""
        messages = [
            Message(
                role=m["role"],
                content=m["content"],
                weight=m.get("weight", 1.0),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
            for m in data["messages"]
        ]
        return cls(messages=messages, metadata=data.get("metadata", {}))


@dataclass
class TokenWeight:
    """Represents token-level weights for training."""

    start_idx: int
    end_idx: int
    weight: float

    def apply_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply weight to tensor slice."""
        weighted_tensor = tensor.clone()
        weighted_tensor[self.start_idx : self.end_idx] *= self.weight
        return weighted_tensor


@dataclass
class RenderConfig:
    """Configuration for message rendering."""

    format: ChatFormat = ChatFormat.CHATML
    add_generation_prompt: bool = False
    add_special_tokens: bool = True
    truncation: bool = True
    max_length: int = 512
    padding: str = "max_length"
    return_tensors: str = "pt"

    # Token weight settings
    mask_system: bool = False  # Don't train on system messages
    mask_user: bool = False  # Don't train on user messages
    only_assistant: bool = True  # Only train on assistant responses

    # Custom templates (for CUSTOM format)
    system_template: Optional[str] = None
    user_template: Optional[str] = None
    assistant_template: Optional[str] = None
    separator: str = "\n"


class MessageRenderer(ABC):
    """Abstract base class for message rendering."""

    def __init__(self, tokenizer: AutoTokenizer, config: RenderConfig):
        self.tokenizer = tokenizer
        self.config = config

    @abstractmethod
    def render_conversation(self, conversation: Conversation) -> str:
        """Render conversation to string format."""

    @abstractmethod
    def get_stop_sequences(self) -> List[str]:
        """Get stop sequences for this format."""

    @abstractmethod
    def parse_response(self, response: str) -> str:
        """Parse model response to extract content."""

    def build_generation_prompt(self, conversation: Conversation) -> str:
        """Build prompt for generation.

        For generation, we want to include the conversation context but remove
        any trailing assistant messages, then add an empty assistant marker to
        trigger generation.
        """
        # Find last non-assistant message index
        last_non_assistant_idx = -1
        for i, msg in enumerate(conversation.messages):
            if msg.role != "assistant":
                last_non_assistant_idx = i

        # Create conversation with messages up to and including last non-assistant message
        gen_conversation = Conversation(
            messages=conversation.messages[: last_non_assistant_idx + 1] if last_non_assistant_idx >= 0 else [],
            metadata=conversation.metadata,
        )

        # Add empty assistant message to trigger generation
        gen_conversation.add_message("assistant", "")

        # Render the conversation - this will end with the assistant marker
        rendered = self.render_conversation(gen_conversation)

        return rendered

    def tokenize_conversation(self, conversation: Conversation) -> Dict[str, torch.Tensor]:
        """Tokenize conversation with proper formatting."""

        rendered = self.render_conversation(conversation)

        # Tokenize
        encoded = self.tokenizer(
            rendered,
            truncation=self.config.truncation,
            max_length=self.config.max_length,
            padding=self.config.padding,
            return_tensors=self.config.return_tensors,
            add_special_tokens=self.config.add_special_tokens,
        )

        # Add labels for training
        encoded["labels"] = encoded["input_ids"].clone()

        # Apply token weights if configured
        if self.config.only_assistant or self.config.mask_system or self.config.mask_user:
            weights = self._compute_token_weights(conversation, rendered, encoded)
            encoded["token_weights"] = weights

            # Mask labels where weight is 0
            if "labels" in encoded:
                mask = weights == 0
                encoded["labels"][mask] = -100

        return encoded

    def _compute_token_weights(
        self, conversation: Conversation, rendered_text: str, encoded: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute token-level weights based on message roles."""

        # Initialize weights to 1
        weights = torch.ones_like(encoded["input_ids"], dtype=torch.float)

        # Find message boundaries in the rendered text
        current_pos = 0
        for message in conversation.messages:
            # Find where this message starts in the rendered text
            message_rendered = self._render_single_message(message)
            start_pos = rendered_text.find(message_rendered, current_pos)

            if start_pos == -1:
                continue

            end_pos = start_pos + len(message_rendered)

            # Convert character positions to token positions
            # This is approximate - for exact mapping, use tokenizer's offset mapping
            start_tokens = len(self.tokenizer.encode(rendered_text[:start_pos], add_special_tokens=False))
            end_tokens = len(self.tokenizer.encode(rendered_text[:end_pos], add_special_tokens=False))

            # Apply weights based on role and config
            if message.role == "system" and self.config.mask_system:
                weights[0, start_tokens:end_tokens] = 0
            elif message.role == "user" and self.config.mask_user:
                weights[0, start_tokens:end_tokens] = 0
            elif message.role == "assistant":
                if not self.config.only_assistant:
                    weights[0, start_tokens:end_tokens] = message.weight
            else:
                # For only_assistant mode, non-assistant messages get 0 weight
                if self.config.only_assistant:
                    weights[0, start_tokens:end_tokens] = 0

            current_pos = end_pos

        return weights

    @abstractmethod
    def _render_single_message(self, message: Message) -> str:
        """Render a single message (needed for weight computation)."""

    def build_supervised_example(
        self, conversation: Conversation, tokenizer: Optional[AutoTokenizer] = None
    ) -> Dict[str, Any]:
        """Build a supervised training example with proper masking."""

        if tokenizer:
            self.tokenizer = tokenizer

        # Tokenize with weights
        encoded = self.tokenize_conversation(conversation)

        # Add additional training information
        encoded["conversation_metadata"] = conversation.metadata

        return encoded


class ChatMLRenderer(MessageRenderer):
    """Renderer for ChatML format."""

    def render_conversation(self, conversation: Conversation) -> str:
        """Render conversation in ChatML format."""
        rendered = []

        for message in conversation.messages:
            rendered.append(self._render_single_message(message))

        return self.config.separator.join(rendered)

    def _render_single_message(self, message: Message) -> str:
        """Render a single message in ChatML format."""
        if message.content:
            return f"<|im_start|>{message.role}\n{message.content}<|im_end|>"
        else:
            # For generation prompts
            return f"<|im_start|>{message.role}\n"

    def get_stop_sequences(self) -> List[str]:
        """Get ChatML stop sequences."""
        return ["<|im_end|>", "<|im_start|>"]

    def parse_response(self, response: str) -> str:
        """Parse ChatML response."""
        # Remove special tokens
        response = response.replace("<|im_end|>", "")
        response = response.replace("<|im_start|>", "")

        # Extract content after role marker
        if "\n" in response:
            response = response.split("\n", 1)[1]

        return response.strip()


class AlpacaRenderer(MessageRenderer):
    """Renderer for Alpaca format."""

    def render_conversation(self, conversation: Conversation) -> str:
        """Render conversation in Alpaca format."""
        rendered = []

        # Extract system message if present
        system_msg = None
        messages = []
        for msg in conversation.messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                messages.append(msg)

        # Build instruction from system + user messages
        for i in range(0, len(messages), 2):
            if i < len(messages):
                user_msg = messages[i]
                instruction = user_msg.content

                if system_msg:
                    instruction = f"{system_msg}\n\n{instruction}"

                # Add response if present
                if i + 1 < len(messages):
                    assistant_msg = messages[i + 1]
                    rendered.append(f"### Instruction:\n{instruction}\n\n" f"### Response:\n{assistant_msg.content}")
                else:
                    # For generation
                    rendered.append(f"### Instruction:\n{instruction}\n\n" f"### Response:\n")

        return self.config.separator.join(rendered)

    def _render_single_message(self, message: Message) -> str:
        """Render single message in Alpaca format."""
        if message.role == "user":
            return f"### Instruction:\n{message.content}"
        elif message.role == "assistant":
            return f"### Response:\n{message.content}"
        else:
            return message.content

    def get_stop_sequences(self) -> List[str]:
        """Get Alpaca stop sequences."""
        return ["### Instruction:", "###", "\n\n\n"]

    def parse_response(self, response: str) -> str:
        """Parse Alpaca response."""
        # Remove format markers
        response = response.replace("### Response:", "")
        response = response.replace("### Instruction:", "")

        # Take content before next instruction
        if "###" in response:
            response = response.split("###")[0]

        return response.strip()


class CustomRenderer(MessageRenderer):
    """Renderer with custom templates."""

    def render_conversation(self, conversation: Conversation) -> str:
        """Render conversation with custom templates."""
        rendered = []

        for message in conversation.messages:
            rendered.append(self._render_single_message(message))

        return self.config.separator.join(rendered)

    def _render_single_message(self, message: Message) -> str:
        """Render message with custom template."""
        if message.role == "system" and self.config.system_template:
            return self.config.system_template.format(content=message.content)
        elif message.role == "user" and self.config.user_template:
            return self.config.user_template.format(content=message.content)
        elif message.role == "assistant" and self.config.assistant_template:
            return self.config.assistant_template.format(content=message.content)
        else:
            return message.content

    def get_stop_sequences(self) -> List[str]:
        """Get custom stop sequences."""
        # Extract prefixes from templates as stop sequences
        stops = []

        if self.config.user_template:
            # Extract prefix before {content}
            prefix = self.config.user_template.split("{content}")[0]
            if prefix:
                stops.append(prefix)

        return stops

    def parse_response(self, response: str) -> str:
        """Parse custom response."""
        # Remove template markers if present
        for stop in self.get_stop_sequences():
            response = response.replace(stop, "")

        return response.strip()


class TokenizerNativeRenderer(MessageRenderer):
    """Renderer that uses tokenizer's native apply_chat_template."""

    def __init__(self, tokenizer: AutoTokenizer, config: RenderConfig):
        super().__init__(tokenizer, config)
        self._supports_tool_role: Optional[bool] = None  # Cached result
        self._supports_tool_calls: Optional[bool] = None  # Cached result
        self._supports_tools: Optional[bool] = None  # Cached result for tools parameter

    def _check_tool_calls_support(self) -> bool:
        """Check if the tokenizer's chat template supports the 'tool_calls' field.

        Tests by attempting to render a message with tool_calls and checking
        if the function name appears in the output.

        Returns:
            True if tokenizer supports tool_calls field, False otherwise
        """
        if self._supports_tool_calls is not None:
            return self._supports_tool_calls

        # Test with a minimal tool_calls message
        test_messages = [
            {"role": "user", "content": "test"},
            {
                "role": "assistant",
                "content": None,
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
            result_text = self.tokenizer.apply_chat_template(test_messages, tokenize=False)
            # Check if the tool call information appears in output
            if "test_func" in result_text or "call_123" in result_text:
                self._supports_tool_calls = True
            else:
                self._supports_tool_calls = False
        except Exception:
            self._supports_tool_calls = False

        return self._supports_tool_calls

    def _check_tool_role_support(self) -> bool:
        """Check if the tokenizer's chat template supports the 'tool' role.

        Tests by attempting to render a minimal conversation with a tool message.
        The result is cached for performance.

        Returns:
            True if tokenizer supports tool role, False otherwise
        """
        if self._supports_tool_role is not None:
            return self._supports_tool_role

        # Test with a minimal tool conversation
        test_messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "test"},
            {"role": "tool", "content": "test"},
            {"role": "assistant", "content": "test"},
        ]

        try:
            self.tokenizer.apply_chat_template(test_messages, tokenize=False)
            self._supports_tool_role = True
        except Exception:
            self._supports_tool_role = False

        return self._supports_tool_role

    def _check_tools_support(self) -> bool:
        """Check if the tokenizer's apply_chat_template supports the 'tools' parameter.

        Tests by attempting to call apply_chat_template with a tools parameter
        and verifying the tool information appears in the output. Some tokenizers
        (like Gemma) accept the parameter but silently ignore it.

        The result is cached for performance.

        Returns:
            True if tokenizer supports and uses tools parameter, False otherwise
        """
        if self._supports_tools is not None:
            return self._supports_tools

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
            output = self.tokenizer.apply_chat_template(test_messages, tools=test_tools, tokenize=False)
            # Check if the tool information actually appears in the output
            # Some tokenizers accept the parameter but silently ignore it (e.g., Gemma)
            if "unique_test_tool_xyz123" in output or "test_tool" in output:
                self._supports_tools = True
            else:
                # Tokenizer accepted the parameter but didn't use it
                self._supports_tools = False
        except (TypeError, Exception):
            # TypeError if tools parameter not accepted, other exceptions for other failures
            self._supports_tools = False

        return self._supports_tools

    def _format_tools_as_text(self, tools: List[Dict]) -> str:
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
        return header + "\n\n".join(tool_descriptions)

    def _preprocess_messages_for_alternation(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Preprocess messages to handle roles not supported by strict-alternation tokenizers.

        Many tokenizers (e.g., Gemma) require strict user/assistant/user/assistant alternation
        and don't support 'tool' or other roles. This method:
        1. Converts 'tool' role to 'user' role with [Tool Result] prefix
        2. Merges consecutive same-role messages to maintain alternation

        This is ONLY called when the tokenizer doesn't support the tool role natively.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Preprocessed messages compatible with strict-alternation tokenizers
        """
        if not messages:
            return messages

        processed = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"] or ""

            # Convert 'tool' role to 'user' (tool responses are external input like user messages)
            if role == "tool":
                role = "user"
                content = f"[Tool Result] {content}"

            # Check if we need to merge with previous message (same role after conversion)
            if processed and processed[-1]["role"] == role:
                # Merge consecutive same-role messages
                processed[-1]["content"] = f"{processed[-1]['content']}\n\n{content}"
            else:
                processed.append({"role": role, "content": content})

        return processed

    def _has_tool_messages(self, messages: List[Dict[str, str]]) -> bool:
        """Check if any messages have the 'tool' role."""
        return any(msg.get("role") == "tool" for msg in messages)

    def _fix_alternation(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
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
            if merged and merged[-1]["role"] == msg["role"]:
                # Merge content with newline separator
                merged[-1]["content"] = f"{merged[-1]['content']}\n{msg['content']}"
            else:
                merged.append(msg.copy())

        # Step 2: Handle alternation issues
        result = []
        for i, msg in enumerate(merged):
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

    def render_conversation(self, conversation: Conversation, tools: Optional[List[Dict]] = None) -> str:
        """Render conversation using tokenizer's apply_chat_template.

        Args:
            conversation: The conversation to render
            tools: Optional list of tool definitions. If provided and the tokenizer
                   doesn't support native tools, they will be injected into the
                   system prompt or first user message.
        """
        import json

        # Check if any messages have tool_calls
        has_tool_calls = any(msg.tool_calls for msg in conversation.messages)

        # Only serialize tool_calls if present AND tokenizer doesn't support them natively
        should_serialize_tool_calls = has_tool_calls and not self._check_tool_calls_support()

        if should_serialize_tool_calls:
            from autotrain import logger

            logger.debug("Tokenizer doesn't support 'tool_calls' field, serializing to content as JSON")

        # Check if we need to inject tools into the conversation
        should_inject_tools = tools and len(tools) > 0 and not self._check_tools_support()

        if should_inject_tools:
            from autotrain import logger

            logger.debug("Tokenizer doesn't support 'tools' parameter, injecting into system prompt")

        # If we need to inject tools, format them as text first
        tools_text = ""
        if should_inject_tools:
            tools_text = self._format_tools_as_text(tools)

        # Convert to messages format
        messages = []
        tools_injected = False
        for msg in conversation.messages:
            content = msg.content or ""

            # Inject tools into system message or first user message if needed
            if should_inject_tools and tools_text and not tools_injected:
                if msg.role == "system":
                    # Append tools to system message
                    content = f"{content}\n\n{tools_text}" if content else tools_text
                    tools_injected = True
                elif msg.role == "user":
                    # No system message found, prepend tools to first user message
                    content = f"{tools_text}\n\n{content}" if content else tools_text
                    tools_injected = True

            # Only serialize tool_calls if tokenizer doesn't support them natively
            if msg.tool_calls and should_serialize_tool_calls:
                # Transform tool_calls to clean format (not raw OpenAI format with "function" key)
                for tc in msg.tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    tool_json = json.dumps({"tool": tool_name, "arguments": args}, ensure_ascii=False)
                    content = f"{content}\n{tool_json}" if content else tool_json
                messages.append({"role": msg.role, "content": content})
            elif msg.tool_calls:
                # Tokenizer supports tool_calls natively - pass them through
                messages.append({"role": msg.role, "content": content, "tool_calls": msg.tool_calls})
            else:
                messages.append({"role": msg.role, "content": content})

        # Only preprocess tool messages if tokenizer doesn't support them
        if self._has_tool_messages(messages) and not self._check_tool_role_support():
            from autotrain import logger

            logger.debug("Tokenizer doesn't support 'tool' role, converting to 'user' with [Tool Result] prefix")
            messages = self._preprocess_messages_for_alternation(messages)

        # Build kwargs for apply_chat_template
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": self.config.add_generation_prompt,
        }

        # Pass tools to native template if tokenizer supports it
        # (only if tools weren't injected into messages, meaning tokenizer supports native tools)
        if tools and not tools_injected and self._check_tools_support():
            template_kwargs["tools"] = tools

        # Try to apply template
        try:
            rendered = self.tokenizer.apply_chat_template(messages, **template_kwargs)
            return rendered
        except Exception as e:
            error_str = str(e).lower()

            # Check if it's an alternation error - if so, try to fix it
            if "alternate" in error_str or "must be" in error_str:
                from autotrain import logger

                logger.debug(f"Alternation error detected: {e}, attempting to fix message structure")
                messages = self._fix_alternation(messages)

                try:
                    rendered = self.tokenizer.apply_chat_template(messages, **template_kwargs)
                    return rendered
                except Exception as e2:
                    logger.warning(f"Failed to apply chat template even after fixing alternation: {e2}")

            # Fallback if apply_chat_template still fails
            from autotrain import logger

            logger.warning(f"Failed to apply tokenizer chat template: {e}, falling back to simple format")
            # Simple fallback
            parts = []
            for msg in messages:
                parts.append(f"{msg['role']}: {msg['content']}")
            return "\n".join(parts)

    def _render_single_message(self, message: Message) -> str:
        """Render single message (simplified for weight computation)."""
        return f"{message.role}: {message.content}"

    def get_stop_sequences(self) -> List[str]:
        """Get stop sequences from tokenizer template."""
        # Try to extract stop sequences from the tokenizer's template
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            template_str = str(self.tokenizer.chat_template).lower()

            # Common stop patterns
            if "start_of_turn" in template_str or "<start_of_turn>" in template_str:
                # Gemma format
                return ["<end_of_turn>", "<start_of_turn>"]
            elif "im_start" in template_str or "<|im_start|>" in template_str:
                # ChatML format
                return ["<|im_end|>", "<|im_start|>"]
            elif "[inst]" in template_str or "[/inst]" in template_str:
                # Llama format
                return ["</s>", "[INST]"]
            elif "start_header_id" in template_str or "<|start_header_id|>" in template_str:
                # Llama-3 format
                return ["<|end_header_id|>", "<|eot_id|>"]

        # Default: use EOS token if available
        if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token:
            return [self.tokenizer.eos_token]

        return []

    def parse_response(self, response: str) -> str:
        """Parse response by stripping stop sequences."""
        # Remove stop sequences
        for stop_seq in self.get_stop_sequences():
            response = response.replace(stop_seq, "")

        return response.strip()


# Registry of available renderers - base ones only, extended in __init__.py
RENDERER_REGISTRY = {
    ChatFormat.CHATML: ChatMLRenderer,
    ChatFormat.ALPACA: AlpacaRenderer,
    ChatFormat.CUSTOM: CustomRenderer,
    ChatFormat.NATIVE: TokenizerNativeRenderer,
}


def get_renderer(
    format: Union[ChatFormat, str], tokenizer: AutoTokenizer, config: Optional[RenderConfig] = None
) -> MessageRenderer:
    """Get a message renderer for the specified format."""

    if isinstance(format, str):
        format = ChatFormat(format)

    if config is None:
        config = RenderConfig(format=format)
    else:
        config.format = format

    renderer_class = RENDERER_REGISTRY.get(format)
    if renderer_class is None:
        raise ValueError(f"Unknown chat format: {format}")

    return renderer_class(tokenizer, config)
