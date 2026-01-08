"""
Message Rendering System for AutoTrain Advanced
================================================

Provides unified conversation-to-token conversion with support for:
- Multiple chat formats (ChatML, Alpaca, Vicuna, etc.)
- Token-level weight control for training
- Stop sequence detection
- Response parsing
"""

# Import format-specific renderers
from .formats import AVAILABLE_FORMATS, LlamaRenderer, MistralRenderer, VicunaRenderer, ZephyrRenderer

# Import base classes and renderers from message_renderer first
from .message_renderer import (
    RENDERER_REGISTRY,
    AlpacaRenderer,
    ChatFormat,
    ChatMLRenderer,
    Conversation,
    CustomRenderer,
    Message,
    MessageRenderer,
    RenderConfig,
    TokenizerNativeRenderer,
    TokenWeight,
    get_renderer,
)


# Extend the registry with format-specific renderers
RENDERER_REGISTRY[ChatFormat.VICUNA] = VicunaRenderer
RENDERER_REGISTRY[ChatFormat.ZEPHYR] = ZephyrRenderer
RENDERER_REGISTRY[ChatFormat.LLAMA] = LlamaRenderer
RENDERER_REGISTRY[ChatFormat.MISTRAL] = MistralRenderer

from .utils import (
    apply_token_weights,
    build_generation_prompt,
    build_supervised_example,
    check_tool_role_support,
    convert_dataset_to_conversations,
    create_chat_template,
    detect_chat_format,
    fix_message_alternation,
    get_stop_sequences,
    parse_response,
    preprocess_messages_for_tool_role,
    safe_apply_chat_template,
)


__all__ = [
    # Core classes
    "MessageRenderer",
    "Message",
    "Conversation",
    "RenderConfig",
    "TokenWeight",
    "ChatFormat",
    "get_renderer",
    "RENDERER_REGISTRY",
    # Format renderers
    "ChatMLRenderer",
    "AlpacaRenderer",
    "VicunaRenderer",
    "ZephyrRenderer",
    "LlamaRenderer",
    "MistralRenderer",
    "CustomRenderer",
    "TokenizerNativeRenderer",
    "AVAILABLE_FORMATS",
    # Utility functions
    "build_generation_prompt",
    "parse_response",
    "get_stop_sequences",
    "build_supervised_example",
    "apply_token_weights",
    "detect_chat_format",
    "convert_dataset_to_conversations",
    "create_chat_template",
    # Tool role and alternation handling
    "safe_apply_chat_template",
    "check_tool_role_support",
    "preprocess_messages_for_tool_role",
    "fix_message_alternation",
]
