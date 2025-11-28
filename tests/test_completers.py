"""
Tests for Completers/Generation Interface
==========================================
"""

import asyncio

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from autotrain.generation import (
    AsyncMessageCompleter,
    AsyncTokenCompleter,
    BeamSearchSampler,
    CompletionConfig,
    CompletionResult,
    MessageCompleter,
    MessageCompletionResult,
    SamplingConfig,
    SamplingStrategy,
    TokenCompleter,
    TokenCompletionResult,
    TopKSampler,
    TopPSampler,
    TypicalSampler,
    batch_complete,
    create_chat_session,
    create_completer,
    stream_tokens,
)
from autotrain.rendering import ChatFormat, Conversation, Message


@pytest.fixture
def model_and_tokenizer():
    """Load small model for testing."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@pytest.fixture
def completion_config():
    """Create test completion config."""
    return CompletionConfig(
        max_new_tokens=10,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )


def test_completion_config():
    """Test CompletionConfig creation."""
    config = CompletionConfig(
        max_new_tokens=50,
        temperature=0.8,
        chat_format=ChatFormat.CHATML,
    )

    assert config.max_new_tokens == 50
    assert config.temperature == 0.8
    assert config.chat_format == ChatFormat.CHATML

    # Test conversion to GenerationConfig
    gen_config = config.to_generation_config()
    assert gen_config.max_new_tokens == 50
    assert gen_config.temperature == 0.8


def test_completion_results():
    """Test result dataclasses."""
    # Basic result
    result = CompletionResult(text="Generated text")
    assert result.text == "Generated text"

    # Token result
    token_result = TokenCompletionResult(text="Generated", token_ids=torch.tensor([1, 2, 3]))
    assert token_result.num_tokens == 3

    # Message result
    msg = Message(role="assistant", content="Response")
    msg_result = MessageCompletionResult(text="Response", message=msg, stop_reason="eos")
    assert msg_result.content == "Response"
    assert msg_result.stop_reason == "eos"


def test_token_completer(model_and_tokenizer):
    """Test TokenCompleter."""
    model, tokenizer = model_and_tokenizer
    config = CompletionConfig(max_new_tokens=5, do_sample=False)

    completer = TokenCompleter(model, tokenizer, config)

    # Test string prompt
    result = completer.complete("Hello, how are")
    assert isinstance(result, TokenCompletionResult)
    assert result.text
    assert result.token_ids is not None
    assert result.num_tokens <= 5

    # Test tensor prompt
    input_ids = tokenizer.encode("Hello", return_tensors="pt")
    result = completer.complete(input_ids)
    assert isinstance(result, TokenCompletionResult)


def test_token_completer_with_logprobs(model_and_tokenizer):
    """Test TokenCompleter with log probabilities."""
    model, tokenizer = model_and_tokenizer
    config = CompletionConfig(
        max_new_tokens=3,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )

    completer = TokenCompleter(model, tokenizer, config)

    result = completer.complete("The", return_logprobs=True)

    # Note: logprobs might be None if model doesn't support it
    # Just test that it doesn't crash
    assert isinstance(result, TokenCompletionResult)


def test_message_completer(model_and_tokenizer):
    """Test MessageCompleter."""
    model, tokenizer = model_and_tokenizer
    config = CompletionConfig(
        max_new_tokens=10,
        do_sample=False,
        chat_format=ChatFormat.CHATML,
    )

    completer = MessageCompleter(model, tokenizer, config)

    # Test with conversation
    conversation = Conversation(
        messages=[
            Message(role="user", content="Hello"),
        ]
    )

    result = completer.complete(conversation)
    assert isinstance(result, MessageCompletionResult)
    assert result.message.role == "assistant"
    assert result.conversation is not None
    assert len(result.conversation.messages) == 2  # User + Assistant

    # Test with message dicts
    messages = [{"role": "user", "content": "Hi there"}]
    result = completer.complete(messages)
    assert isinstance(result, MessageCompletionResult)


def test_message_completer_with_system(model_and_tokenizer):
    """Test MessageCompleter with system prompt."""
    model, tokenizer = model_and_tokenizer
    completer = MessageCompleter(model, tokenizer)

    messages = [{"role": "user", "content": "What is 2+2?"}]
    result = completer.complete(messages, system_prompt="You are a math tutor.")

    assert result.conversation.messages[0].role == "system"
    assert "math tutor" in result.conversation.messages[0].content


def test_chat_method(model_and_tokenizer):
    """Test MessageCompleter chat method."""
    model, tokenizer = model_and_tokenizer
    config = CompletionConfig(max_new_tokens=10, do_sample=False)
    completer = MessageCompleter(model, tokenizer, config)

    result = completer.chat("Hello", system_prompt="Be friendly")
    assert isinstance(result, MessageCompletionResult)
    assert result.conversation.messages[0].role == "system"
    assert result.conversation.messages[1].role == "user"
    assert result.conversation.messages[2].role == "assistant"


def test_batch_complete(model_and_tokenizer):
    """Test batch completion."""
    model, tokenizer = model_and_tokenizer
    config = CompletionConfig(max_new_tokens=5, do_sample=False)
    completer = TokenCompleter(model, tokenizer, config)

    prompts = ["Hello", "How", "What"]
    results = completer.batch_complete(prompts)

    assert len(results) == 3
    assert all(isinstance(r, TokenCompletionResult) for r in results)


def test_stream_tokens(model_and_tokenizer):
    """Test token streaming."""
    model, tokenizer = model_and_tokenizer
    config = CompletionConfig(max_new_tokens=5, do_sample=False)
    completer = TokenCompleter(model, tokenizer, config)

    tokens = list(completer.stream_tokens("Hello"))
    assert len(tokens) <= 5
    assert all(isinstance(t, torch.Tensor) for t in tokens)


@pytest.mark.asyncio
async def test_async_token_completer(model_and_tokenizer):
    """Test AsyncTokenCompleter."""
    model, tokenizer = model_and_tokenizer
    config = CompletionConfig(max_new_tokens=5, do_sample=False)
    completer = AsyncTokenCompleter(model, tokenizer, config)

    result = await completer.complete_async("Hello")
    assert isinstance(result, TokenCompletionResult)

    # Test async streaming
    tokens = []
    async for token in completer.stream_tokens_async("Test"):
        tokens.append(token)
    assert len(tokens) <= 5


@pytest.mark.asyncio
async def test_async_message_completer(model_and_tokenizer):
    """Test AsyncMessageCompleter."""
    model, tokenizer = model_and_tokenizer
    completer = AsyncMessageCompleter(model, tokenizer)

    result = await completer.chat_async("Hello")
    assert isinstance(result, MessageCompletionResult)


def test_sampling_config():
    """Test SamplingConfig."""
    config = SamplingConfig(
        strategy=SamplingStrategy.TOP_P,
        temperature=0.8,
        top_p=0.95,
    )

    assert config.strategy == SamplingStrategy.TOP_P
    assert config.temperature == 0.8
    assert config.top_p == 0.95


def test_top_k_sampler():
    """Test TopKSampler."""
    config = SamplingConfig(
        strategy=SamplingStrategy.TOP_K,
        top_k=10,
        temperature=1.0,
    )

    sampler = TopKSampler(config)

    # Test sampling
    logits = torch.randn(2, 100)  # Batch of 2, vocab size 100
    tokens, probs = sampler.sample(logits)

    assert tokens.shape == (2,)
    assert probs.shape == (2,)
    assert torch.all((tokens >= 0) & (tokens < 100))


def test_top_p_sampler():
    """Test TopPSampler."""
    config = SamplingConfig(
        strategy=SamplingStrategy.TOP_P,
        top_p=0.9,
        temperature=1.0,
    )

    sampler = TopPSampler(config)

    logits = torch.randn(2, 100)
    tokens, probs = sampler.sample(logits)

    assert tokens.shape == (2,)
    assert torch.all((probs >= 0) & (probs <= 1))


def test_typical_sampler():
    """Test TypicalSampler."""
    config = SamplingConfig(
        strategy=SamplingStrategy.TYPICAL,
        typical_p=0.95,
    )

    sampler = TypicalSampler(config)

    logits = torch.randn(2, 100)
    tokens, probs = sampler.sample(logits)

    assert tokens.shape == (2,)


def test_beam_search_sampler():
    """Test BeamSearchSampler."""
    config = SamplingConfig(
        strategy=SamplingStrategy.BEAM_SEARCH,
        num_beams=4,
    )

    sampler = BeamSearchSampler(config)

    logits = torch.randn(1, 100)  # Beam search typically with batch size 1
    tokens, probs = sampler.sample(logits)

    assert tokens.shape == (1,)

    # Test getting best sequence
    best = sampler.get_best_sequence()
    assert isinstance(best, torch.Tensor)


def test_repetition_penalty():
    """Test repetition penalty application."""
    config = SamplingConfig(repetition_penalty=1.5)
    sampler = TopKSampler(config)

    logits = torch.randn(1, 100)
    past_tokens = torch.tensor([[1, 2, 3]])

    # Apply repetition penalty
    penalized = sampler.apply_repetition_penalty(logits.clone(), past_tokens, 1.5)

    # Check that logits for past tokens are modified
    assert not torch.allclose(logits[0, 1], penalized[0, 1])


def test_create_completer():
    """Test completer factory function."""
    # Test token completer
    completer = create_completer(
        "gpt2",
        completer_type="token",
    )
    assert isinstance(completer, TokenCompleter)

    # Test message completer
    completer = create_completer(
        "gpt2",
        completer_type="message",
        chat_format=ChatFormat.ALPACA,
    )
    assert isinstance(completer, MessageCompleter)

    # Test async completer
    completer = create_completer(
        "gpt2",
        completer_type="token",
        async_mode=True,
    )
    assert isinstance(completer, AsyncTokenCompleter)


def test_batch_complete_utility(model_and_tokenizer):
    """Test batch_complete utility function."""
    model, tokenizer = model_and_tokenizer
    completer = TokenCompleter(model, tokenizer)

    prompts = ["Test 1", "Test 2", "Test 3"]
    results = batch_complete(
        completer,
        prompts,
        batch_size=2,
        show_progress=False,
    )

    assert len(results) == 3
    assert all(isinstance(r, TokenCompletionResult) for r in results)


def test_stream_tokens_utility(model_and_tokenizer):
    """Test stream_tokens utility function."""
    model, tokenizer = model_and_tokenizer
    completer = TokenCompleter(model, tokenizer)

    # Track tokens
    received_tokens = []

    def callback(token):
        received_tokens.append(token)

    tokens = list(stream_tokens(completer, "Hello", callback=callback))

    assert len(tokens) == len(received_tokens)
    assert all(isinstance(t, torch.Tensor) for t in tokens)


def test_chat_session(model_and_tokenizer):
    """Test ChatSession."""
    model, tokenizer = model_and_tokenizer

    session = create_chat_session(
        model,
        tokenizer,
        system_prompt="You are helpful.",
        chat_format=ChatFormat.CHATML,
    )

    # Test chat
    response = session.chat("Hello")
    assert isinstance(response, str)

    # Test history
    history = session.get_history()
    assert len(history) >= 2  # System + User + Assistant
    assert history[-2]["role"] == "user"
    assert history[-1]["role"] == "assistant"

    # Test clear history
    session.clear_history(keep_system=True)
    history = session.get_history()
    assert len(history) == 1  # Only system

    session.clear_history(keep_system=False)
    history = session.get_history()
    assert len(history) == 0


def test_chat_session_persistence(model_and_tokenizer, tmp_path):
    """Test ChatSession save/load."""
    model, tokenizer = model_and_tokenizer
    session = create_chat_session(model, tokenizer)

    # Have a conversation
    session.chat("Hello")
    session.chat("How are you?")

    # Save history
    save_path = tmp_path / "chat_history.json"
    session.save_history(str(save_path))
    assert save_path.exists()

    # Create new session and load
    new_session = create_chat_session(model, tokenizer)
    new_session.load_history(str(save_path))

    # Check history restored
    assert len(new_session.get_history()) == len(session.get_history())
