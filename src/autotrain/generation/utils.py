"""
Utility functions for generation/completion
============================================

Helper functions for creating and using completers.
"""

import asyncio
import json
import os
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from autotrain import logger
from autotrain.rendering import ChatFormat, Conversation
from autotrain.utils import get_model_loading_kwargs, maybe_move_to_mps

from .completers import (
    AsyncMessageCompleter,
    AsyncTokenCompleter,
    CompletionConfig,
    MessageCompleter,
    MessageCompletionResult,
    TokenCompleter,
    TokenCompletionResult,
)


def _detect_chat_format(tokenizer: AutoTokenizer) -> ChatFormat:
    """
    Auto-detect chat format from tokenizer's chat template.

    Args:
        tokenizer: The tokenizer to inspect

    Returns:
        Detected ChatFormat - uses NATIVE if tokenizer has chat_template
    """
    # Check if tokenizer has a chat template - if yes, use NATIVE renderer
    # which will dynamically use tokenizer.apply_chat_template()
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        template_str = str(tokenizer.chat_template)
        logger.info(
            f"Detected chat template in tokenizer, using NATIVE renderer. Template preview: {template_str[:100]}..."
        )
        return ChatFormat.NATIVE

    # No chat template - fallback to CHATML
    logger.warning("No chat template found in tokenizer, defaulting to CHATML format")
    return ChatFormat.CHATML


def create_completer(
    model: Union[str, AutoModelForCausalLM],
    tokenizer: Union[str, AutoTokenizer, None] = None,
    completer_type: str = "message",
    config: Optional[CompletionConfig] = None,
    chat_format: Optional[ChatFormat] = None,
    async_mode: bool = False,
) -> Union[TokenCompleter, MessageCompleter, AsyncTokenCompleter, AsyncMessageCompleter]:
    """
    Create a completer instance.

    Args:
        model: Model name or instance
        tokenizer: Tokenizer name or instance
        completer_type: "token" or "message"
        config: Completion configuration
        chat_format: Chat format for message completer
        async_mode: Whether to create async version

    Returns:
        Completer instance
    """
    # Load model if string
    if isinstance(model, str):
        model_path = model

        # Load tokenizer FIRST to check vocabulary size
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        # Set padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Check if this is a PEFT model (has adapter_config.json)
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        is_peft_model = os.path.exists(adapter_config_path)

        if is_peft_model:
            # This is a PEFT/LoRA model, use PEFT loader
            try:
                from peft import PeftModel

                # Read adapter config to get base model
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path")

                # Load base model
                model_kwargs = get_model_loading_kwargs(trust_remote_code=True)

                # Try to load from local cache first, then from hub
                try:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name, local_files_only=True, **model_kwargs  # Check local cache first
                    )
                except Exception:
                    # Fallback to downloading from hub if not in cache
                    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

                # Check if tokenizer was resized during training
                base_vocab_size = base_model.get_input_embeddings().num_embeddings
                tokenizer_vocab_size = len(tokenizer)

                # Resize base model if needed (when tokens were added during training)
                if tokenizer_vocab_size > base_vocab_size:
                    base_model.resize_token_embeddings(tokenizer_vocab_size)

                # Load PEFT adapters on top of the resized base model
                model = PeftModel.from_pretrained(base_model, model_path)
                model = maybe_move_to_mps(model, model_kwargs)
            except ImportError:
                raise ImportError("PEFT library is required to load adapter models. Install with: pip install peft")
        else:
            # Regular full fine-tuned model - load with resize handling
            model_kwargs = get_model_loading_kwargs(trust_remote_code=True)
            model_kwargs["ignore_mismatched_sizes"] = True  # Critical: allows loading with size mismatches

            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            model = maybe_move_to_mps(model, model_kwargs)

            # Now resize to match tokenizer if needed (only for non-PEFT models)
            if hasattr(model, "get_input_embeddings") and hasattr(model, "resize_token_embeddings"):
                model_vocab_size = model.get_input_embeddings().num_embeddings
                tokenizer_vocab_size = len(tokenizer)

                if model_vocab_size != tokenizer_vocab_size:
                    # Resize model to match tokenizer
                    model.resize_token_embeddings(tokenizer_vocab_size)
    else:
        # Model is already loaded, just handle tokenizer
        if tokenizer is None:
            if hasattr(model, "config") and hasattr(model.config, "name_or_path"):
                tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
            else:
                raise ValueError("Tokenizer must be provided if model doesn't have name_or_path")
        elif isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        # Set padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Create config if not provided
    if config is None:
        config = CompletionConfig()

    # Auto-detect chat format if not provided and we're creating a message completer
    if completer_type == "message" and chat_format is None:
        chat_format = _detect_chat_format(tokenizer)

    # Create completer
    if completer_type == "token":
        if async_mode:
            return AsyncTokenCompleter(model, tokenizer, config)
        else:
            return TokenCompleter(model, tokenizer, config)
    elif completer_type == "message":
        if async_mode:
            return AsyncMessageCompleter(model, tokenizer, config, chat_format)
        else:
            return MessageCompleter(model, tokenizer, config, chat_format)
    else:
        raise ValueError(f"Unknown completer type: {completer_type}")


def batch_complete(
    completer: Union[TokenCompleter, MessageCompleter],
    prompts: List[Any],
    batch_size: int = 8,
    show_progress: bool = True,
    **kwargs,
) -> List[Union[TokenCompletionResult, MessageCompletionResult]]:
    """
    Complete multiple prompts in batches.

    Args:
        completer: Completer to use
        prompts: List of prompts
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar

    Returns:
        List of completion results
    """
    results = []

    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(range(0, len(prompts), batch_size), desc="Completing")
    else:
        iterator = range(0, len(prompts), batch_size)

    for i in iterator:
        batch = prompts[i : i + batch_size]
        batch_results = completer.batch_complete(batch, **kwargs)
        results.extend(batch_results)

    return results


def stream_tokens(
    completer: TokenCompleter, prompt: Union[str, torch.Tensor], callback: Optional[Any] = None, **kwargs
) -> Iterator[torch.Tensor]:
    """
    Stream tokens with optional callback.

    Args:
        completer: Token completer
        prompt: Input prompt
        callback: Optional callback function for each token

    Yields:
        Token IDs
    """
    for token in completer.stream_tokens(prompt, **kwargs):
        if callback:
            callback(token)
        yield token


def stream_messages(
    completer: MessageCompleter,
    conversation: Union[Conversation, List[Dict[str, str]]],
    callback: Optional[Any] = None,
    **kwargs,
) -> Iterator[str]:
    """
    Stream message content as it's generated.

    This is a simplified version that generates the full message
    then yields it character by character for display.

    Args:
        completer: Message completer
        conversation: Conversation
        callback: Optional callback

    Yields:
        Message content chunks
    """
    # Generate full message
    result = completer.complete(conversation, **kwargs)
    content = result.content

    # Stream character by character
    for char in content:
        if callback:
            callback(char)
        yield char


async def async_batch_complete(
    completer: Union[AsyncTokenCompleter, AsyncMessageCompleter],
    prompts: List[Any],
    max_concurrent: int = 10,
    **kwargs,
) -> List[Union[TokenCompletionResult, MessageCompletionResult]]:
    """
    Async batch completion with concurrency limit.

    Args:
        completer: Async completer
        prompts: List of prompts
        max_concurrent: Maximum concurrent completions

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def complete_with_semaphore(prompt):
        async with semaphore:
            return await completer.complete_async(prompt, **kwargs)

    tasks = [complete_with_semaphore(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)

    return results


def create_chat_session(
    model: Union[str, AutoModelForCausalLM],
    tokenizer: Union[str, AutoTokenizer, None] = None,
    system_prompt: Optional[str] = None,
    chat_format: ChatFormat = ChatFormat.CHATML,
    config: Optional[CompletionConfig] = None,
) -> "ChatSession":
    """
    Create an interactive chat session.

    Args:
        model: Model to use
        tokenizer: Tokenizer
        system_prompt: System prompt
        chat_format: Chat format
        config: Completion config

    Returns:
        ChatSession instance
    """
    completer = create_completer(
        model=model,
        tokenizer=tokenizer,
        completer_type="message",
        config=config,
        chat_format=chat_format,
    )

    return ChatSession(completer, system_prompt)


class ChatSession:
    """Interactive chat session."""

    def __init__(
        self,
        completer: MessageCompleter,
        system_prompt: Optional[str] = None,
    ):
        self.completer = completer
        self.conversation = Conversation(messages=[])

        if system_prompt:
            self.conversation.add_message("system", system_prompt)

    def chat(self, user_input: str, **kwargs) -> str:
        """
        Send a message and get response.

        Args:
            user_input: User message

        Returns:
            Assistant response
        """
        result = self.completer.chat(user_input, conversation=self.conversation, **kwargs)

        # Update conversation
        self.conversation = result.conversation

        return result.content

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return [{"role": msg.role, "content": msg.content} for msg in self.conversation.messages]

    def clear_history(self, keep_system: bool = True):
        """Clear conversation history."""
        if keep_system:
            system_msgs = [m for m in self.conversation.messages if m.role == "system"]
            self.conversation = Conversation(messages=system_msgs)
        else:
            self.conversation = Conversation(messages=[])

    def save_history(self, filepath: str):
        """Save conversation to file."""
        import json

        with open(filepath, "w") as f:
            json.dump(self.conversation.to_dict(), f, indent=2)

    def load_history(self, filepath: str):
        """Load conversation from file."""
        import json

        with open(filepath, "r") as f:
            data = json.load(f)
        self.conversation = Conversation.from_dict(data)
