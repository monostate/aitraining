"""
Completer implementations for different generation levels
==========================================================

Provides TokenCompleter and MessageCompleter abstractions.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)

from autotrain.rendering import ChatFormat, Conversation, Message, RenderConfig, get_renderer


class MultiTokenStoppingCriteria(StoppingCriteria):
    """Stopping criteria that detects multi-token stop sequences during generation."""

    def __init__(self, stop_sequences: List[str], tokenizer: AutoTokenizer, input_length: int):
        """
        Initialize stopping criteria.

        Args:
            stop_sequences: List of strings to stop at (e.g., ['<end_of_turn>', '<|im_end|>'])
            tokenizer: The tokenizer to decode generated tokens
            input_length: Length of input prompt tokens (to skip checking prompt)
        """
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        self.input_length = input_length

        # Pre-tokenize stop sequences for faster matching
        self.stop_token_ids = []
        for seq in stop_sequences:
            # Get token IDs for each stop sequence
            tokens = tokenizer.encode(seq, add_special_tokens=False)
            self.stop_token_ids.append(tokens)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if any stop sequence appears in the generated tokens.

        Args:
            input_ids: Generated token IDs [batch_size, sequence_length]
            scores: Logits scores (unused but required by interface)

        Returns:
            True if generation should stop, False otherwise
        """
        # Only check the newly generated tokens (skip input prompt)
        generated_ids = input_ids[0, self.input_length :]

        # Check each stop sequence
        for stop_tokens in self.stop_token_ids:
            stop_len = len(stop_tokens)

            # Need at least enough tokens to match
            if len(generated_ids) < stop_len:
                continue

            # Check if the last N tokens match this stop sequence
            if torch.equal(generated_ids[-stop_len:], torch.tensor(stop_tokens, device=generated_ids.device)):
                return True

        return False


@dataclass
class CompletionConfig:
    """Configuration for text completion."""

    # Generation parameters
    max_new_tokens: int = 256
    min_new_tokens: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0

    # Sampling
    do_sample: bool = True
    num_beams: int = 1
    num_return_sequences: int = 1

    # Early stopping
    early_stopping: bool = False
    max_time: Optional[float] = None

    # Token control
    eos_token_id: Optional[Union[int, List[int]]] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None

    # Advanced
    use_cache: bool = True
    return_dict_in_generate: bool = True
    output_scores: bool = False
    output_logits: bool = False
    output_attentions: bool = False

    # Chat format (for MessageCompleter)
    chat_format: ChatFormat = ChatFormat.CHATML
    add_generation_prompt: bool = True

    # Custom processors
    logits_processor: Optional[LogitsProcessor] = None
    stopping_criteria: Optional[StoppingCriteriaList] = None

    def to_generation_config(self) -> GenerationConfig:
        """Convert to HuggingFace GenerationConfig."""
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
            early_stopping=self.early_stopping,
            max_time=self.max_time,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            use_cache=self.use_cache,
            return_dict_in_generate=self.return_dict_in_generate,
            output_scores=self.output_scores,
            output_attentions=self.output_attentions,
        )


@dataclass
class CompletionResult:
    """Base result class for completions."""

    text: str
    tokens: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenCompletionResult(CompletionResult):
    """Result from token-level completion."""

    token_ids: torch.Tensor = None
    token_scores: Optional[torch.Tensor] = None
    token_logprobs: Optional[torch.Tensor] = None

    @property
    def num_tokens(self) -> int:
        """Get number of generated tokens."""
        if self.token_ids is not None:
            return self.token_ids.shape[-1]
        return 0


@dataclass
class MessageCompletionResult(CompletionResult):
    """Result from message-level completion."""

    message: Message = None
    conversation: Optional[Conversation] = None
    stop_reason: Optional[str] = None

    @property
    def content(self) -> str:
        """Get message content."""
        return self.message.content if self.message else self.text


class Completer(ABC):
    """Abstract base class for text completers."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: Optional[CompletionConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or CompletionConfig()

        # Set special tokens if not configured
        if self.config.eos_token_id is None:
            self.config.eos_token_id = tokenizer.eos_token_id
        if self.config.pad_token_id is None:
            self.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if self.config.bos_token_id is None:
            self.config.bos_token_id = tokenizer.bos_token_id

    @abstractmethod
    def complete(self, prompt: Any, **kwargs) -> CompletionResult:
        """Generate completion for prompt."""

    @abstractmethod
    def batch_complete(self, prompts: List[Any], **kwargs) -> List[CompletionResult]:
        """Generate completions for multiple prompts."""


class TokenCompleter(Completer):
    """Low-level token generation for RL and fine-grained control."""

    def complete(
        self,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_logprobs: bool = False,
        **kwargs,
    ) -> TokenCompletionResult:
        """
        Generate tokens from prompt.

        Args:
            prompt: String or token tensor
            max_new_tokens: Override config setting
            temperature: Override config temperature
            return_logprobs: Whether to return log probabilities

        Returns:
            TokenCompletionResult with generated tokens
        """
        # Convert string to tokens if needed
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        else:
            input_ids = prompt

        # Move to model device
        input_ids = input_ids.to(self.model.device)

        # Update generation config
        gen_config = self.config.to_generation_config()
        if max_new_tokens is not None:
            gen_config.max_new_tokens = max_new_tokens
        if temperature is not None:
            gen_config.temperature = temperature

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                generation_config=gen_config,
                logits_processor=self.config.logits_processor,
                stopping_criteria=self.config.stopping_criteria,
                **kwargs,
            )

        # Extract generated tokens (remove prompt)
        generated_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
        generated_ids = generated_ids[:, input_ids.shape[1] :]

        # Decode text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Compute log probabilities if requested
        token_logprobs = None
        if return_logprobs and hasattr(outputs, "scores"):
            # Stack scores and compute log probs
            scores = torch.stack(outputs.scores, dim=1)
            token_logprobs = torch.log_softmax(scores, dim=-1)

            # Get log probs for generated tokens
            batch_size = generated_ids.shape[0]
            token_logprobs_selected = []
            for b in range(batch_size):
                logprobs_b = []
                for t in range(generated_ids.shape[1]):
                    token_id = generated_ids[b, t]
                    logprob = token_logprobs[b, t, token_id]
                    logprobs_b.append(logprob)
                token_logprobs_selected.append(torch.stack(logprobs_b))
            token_logprobs = torch.stack(token_logprobs_selected)

        return TokenCompletionResult(
            text=generated_text,
            tokens=generated_ids,
            token_ids=generated_ids[0],
            token_scores=outputs.scores if hasattr(outputs, "scores") else None,
            token_logprobs=token_logprobs,
            logits=outputs.logits if hasattr(outputs, "logits") else None,
            metadata={
                "num_tokens": generated_ids.shape[1],
                "model": self.model.config.name_or_path,
            },
        )

    def batch_complete(self, prompts: List[Union[str, torch.Tensor]], **kwargs) -> List[TokenCompletionResult]:
        """Generate completions for multiple prompts."""
        results = []

        # Process each prompt
        # TODO: Implement true batch processing for efficiency
        for prompt in prompts:
            result = self.complete(prompt, **kwargs)
            results.append(result)

        return results

    def stream_tokens(self, prompt: Union[str, torch.Tensor], **kwargs) -> Iterator[torch.Tensor]:
        """
        Stream tokens as they are generated.

        Yields:
            Individual token IDs
        """
        # This is a simplified version - real streaming would use
        # a custom generation loop
        result = self.complete(prompt, **kwargs)

        for i in range(result.num_tokens):
            yield result.token_ids[i]


class MessageCompleter(Completer):
    """High-level message generation with chat formatting."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: Optional[CompletionConfig] = None,
        chat_format: Optional[ChatFormat] = None,
    ):
        super().__init__(model, tokenizer, config)

        # Set chat format
        if chat_format:
            self.config.chat_format = chat_format

        # Create renderer
        render_config = RenderConfig(
            format=self.config.chat_format,
            add_generation_prompt=self.config.add_generation_prompt,
        )
        self.renderer = get_renderer(self.config.chat_format, tokenizer, render_config)

        # Get stop sequences
        self.stop_sequences = self.renderer.get_stop_sequences()

    def complete(
        self, conversation: Union[Conversation, List[Dict[str, str]]], system_prompt: Optional[str] = None, **kwargs
    ) -> MessageCompletionResult:
        """
        Generate a message response.

        Args:
            conversation: Conversation or list of message dicts
            system_prompt: Optional system prompt to prepend

        Returns:
            MessageCompletionResult with generated message
        """
        # Convert to Conversation if needed
        if isinstance(conversation, list):
            messages = []
            if system_prompt:
                messages.append(Message(role="system", content=system_prompt))
            messages.extend(
                [
                    Message(
                        role=m["role"],
                        content=m.get("content") or "",
                        tool_calls=m.get("tool_calls"),
                        tool_call_id=m.get("tool_call_id"),
                    )
                    for m in conversation
                ]
            )
            conversation = Conversation(messages=messages)
        elif system_prompt:
            # Prepend system message
            messages = [Message(role="system", content=system_prompt)] + conversation.messages
            conversation = Conversation(messages=messages)

        # Build generation prompt
        prompt = self.renderer.build_generation_prompt(conversation)

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        gen_config = self.config.to_generation_config()

        # Handle stop sequences with proper early stopping
        eos_token_id = self.config.eos_token_id
        if eos_token_id is None and hasattr(self.tokenizer, "eos_token_id"):
            eos_token_id = self.tokenizer.eos_token_id

        # Create stopping criteria for multi-token stop sequences
        stopping_criteria = None
        if self.stop_sequences:
            stopping_criteria = StoppingCriteriaList(
                [
                    MultiTokenStoppingCriteria(
                        stop_sequences=self.stop_sequences, tokenizer=self.tokenizer, input_length=input_ids.shape[1]
                    )
                ]
            )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                generation_config=gen_config,
                eos_token_id=eos_token_id,
                stopping_criteria=stopping_criteria,
                **kwargs,
            )

        # Extract generated text
        generated_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
        generated_ids = generated_ids[:, input_ids.shape[1] :]
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        # Truncate at stop sequences
        stop_reason = "eos"
        truncated_text = generated_text
        if self.stop_sequences:
            # Find the earliest stop sequence
            earliest_pos = len(generated_text)
            earliest_stop = None
            for stop_seq in self.stop_sequences:
                pos = generated_text.find(stop_seq)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    earliest_stop = stop_seq

            if earliest_stop:
                truncated_text = generated_text[:earliest_pos]
                stop_reason = f"stop_sequence:{earliest_stop}"

        # Parse response
        parsed_content = self.renderer.parse_response(truncated_text)

        # Create message
        generated_message = Message(
            role="assistant",
            content=parsed_content,
            metadata={
                "raw_text": truncated_text,
                "full_generated_text": generated_text,  # Keep original for debugging
                "num_tokens": generated_ids.shape[1],
                "stop_reason": stop_reason,
            },
        )

        # Update conversation
        updated_conversation = Conversation(
            messages=conversation.messages + [generated_message], metadata=conversation.metadata
        )

        return MessageCompletionResult(
            text=parsed_content,
            message=generated_message,
            conversation=updated_conversation,
            stop_reason=stop_reason,
            tokens=generated_ids,
            metadata={
                "model": self.model.config.name_or_path,
                "chat_format": self.config.chat_format.value,
            },
        )

    def batch_complete(
        self, conversations: List[Union[Conversation, List[Dict[str, str]]]], **kwargs
    ) -> List[MessageCompletionResult]:
        """Generate message completions for multiple conversations."""
        results = []

        for conv in conversations:
            result = self.complete(conv, **kwargs)
            results.append(result)

        return results

    def chat(
        self,
        user_input: str,
        conversation: Optional[Conversation] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> MessageCompletionResult:
        """
        Simple chat interface.

        Args:
            user_input: User message
            conversation: Optional existing conversation
            system_prompt: System prompt

        Returns:
            Assistant response
        """
        # Create or update conversation
        if conversation is None:
            messages = []
            if system_prompt:
                messages.append(Message(role="system", content=system_prompt))
            conversation = Conversation(messages=messages)

        # Add user message
        conversation.add_message("user", user_input)

        # Generate response
        return self.complete(conversation, **kwargs)


class AsyncTokenCompleter(TokenCompleter):
    """Async version of TokenCompleter."""

    async def complete_async(self, prompt: Union[str, torch.Tensor], **kwargs) -> TokenCompletionResult:
        """Async token generation."""
        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.complete, prompt, **kwargs)
        return result

    async def stream_tokens_async(self, prompt: Union[str, torch.Tensor], **kwargs) -> AsyncIterator[torch.Tensor]:
        """Async streaming token generation."""
        result = await self.complete_async(prompt, **kwargs)

        for i in range(result.num_tokens):
            yield result.token_ids[i]
            await asyncio.sleep(0)  # Allow other tasks


class AsyncMessageCompleter(MessageCompleter):
    """Async version of MessageCompleter."""

    async def complete_async(
        self, conversation: Union[Conversation, List[Dict[str, str]]], **kwargs
    ) -> MessageCompletionResult:
        """Async message generation."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.complete, conversation, **kwargs)
        return result

    async def chat_async(
        self, user_input: str, conversation: Optional[Conversation] = None, **kwargs
    ) -> MessageCompletionResult:
        """Async chat interface."""
        if conversation is None:
            conversation = Conversation(messages=[])

        conversation.add_message("user", user_input)
        return await self.complete_async(conversation, **kwargs)
