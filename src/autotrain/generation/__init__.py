"""
Generation/Completers Interface for AutoTrain Advanced
========================================================

Provides clean abstractions for model generation at different levels:
- TokenCompleter: Low-level token generation (for RL)
- MessageCompleter: High-level message generation
- AsyncCompleter: Asynchronous generation support
"""

from .completers import (
    AsyncMessageCompleter,
    AsyncTokenCompleter,
    Completer,
    CompletionConfig,
    CompletionResult,
    MessageCompleter,
    MessageCompletionResult,
    TokenCompleter,
    TokenCompletionResult,
)
from .sampling import BeamSearchSampler, SamplingConfig, SamplingStrategy, TopKSampler, TopPSampler, TypicalSampler
from .utils import ChatSession, batch_complete, create_chat_session, create_completer, stream_messages, stream_tokens


__all__ = [
    # Core completers
    "Completer",
    "TokenCompleter",
    "MessageCompleter",
    "AsyncTokenCompleter",
    "AsyncMessageCompleter",
    # Configurations
    "CompletionConfig",
    "CompletionResult",
    "TokenCompletionResult",
    "MessageCompletionResult",
    # Sampling
    "SamplingConfig",
    "SamplingStrategy",
    "TopKSampler",
    "TopPSampler",
    "BeamSearchSampler",
    "TypicalSampler",
    # Utils
    "create_completer",
    "batch_complete",
    "stream_tokens",
    "stream_messages",
    "create_chat_session",
    "ChatSession",
]
