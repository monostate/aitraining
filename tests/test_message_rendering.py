"""
Tests for Message Rendering System
===================================
"""

import pytest
import torch
from transformers import AutoTokenizer

from autotrain.rendering import (
    AlpacaRenderer,
    ChatFormat,
    ChatMLRenderer,
    Conversation,
    LlamaRenderer,
    Message,
    MessageRenderer,
    MistralRenderer,
    RenderConfig,
    TokenWeight,
    VicunaRenderer,
    ZephyrRenderer,
    apply_token_weights,
    build_generation_prompt,
    build_supervised_example,
    convert_dataset_to_conversations,
    create_chat_template,
    detect_chat_format,
    get_renderer,
    get_stop_sequences,
    parse_response,
)


@pytest.fixture
def tokenizer():
    """Create tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def sample_conversation():
    """Create sample conversation."""
    return Conversation(
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="2+2 equals 4."),
        ]
    )


def test_message_creation():
    """Test Message creation."""
    msg = Message(role="user", content="Hello", weight=0.5)
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.weight == 0.5


def test_conversation_creation():
    """Test Conversation creation and methods."""
    conv = Conversation(messages=[])
    assert len(conv.messages) == 0

    conv.add_message("user", "Hello")
    assert len(conv.messages) == 1
    assert conv.messages[0].role == "user"

    # Test to_dict
    data = conv.to_dict()
    assert "messages" in data
    assert len(data["messages"]) == 1

    # Test from_dict
    conv2 = Conversation.from_dict(data)
    assert len(conv2.messages) == 1
    assert conv2.messages[0].content == "Hello"


def test_token_weight():
    """Test TokenWeight functionality."""
    weight = TokenWeight(start_idx=10, end_idx=20, weight=0.5)

    tensor = torch.ones(30)
    weighted = weight.apply_to_tensor(tensor)

    assert weighted[0] == 1.0  # Before range
    assert weighted[15] == 0.5  # In range
    assert weighted[25] == 1.0  # After range


def test_render_config():
    """Test RenderConfig."""
    config = RenderConfig(
        format=ChatFormat.CHATML,
        max_length=512,
        only_assistant=True,
    )

    assert config.format == ChatFormat.CHATML
    assert config.max_length == 512
    assert config.only_assistant == True


def test_chatml_renderer(tokenizer, sample_conversation):
    """Test ChatML rendering."""
    config = RenderConfig(format=ChatFormat.CHATML)
    renderer = ChatMLRenderer(tokenizer, config)

    # Test conversation rendering
    rendered = renderer.render_conversation(sample_conversation)
    assert "<|im_start|>" in rendered
    assert "<|im_end|>" in rendered
    assert "system" in rendered
    assert "You are a helpful assistant." in rendered

    # Test stop sequences
    stops = renderer.get_stop_sequences()
    assert "<|im_end|>" in stops

    # Test generation prompt
    prompt = renderer.build_generation_prompt(sample_conversation)
    assert prompt.endswith("<|im_start|>assistant\n")

    # Test response parsing
    response = "<|im_start|>assistant\nThis is a response<|im_end|>"
    parsed = renderer.parse_response(response)
    assert "This is a response" in parsed
    assert "<|im_start|>" not in parsed


def test_alpaca_renderer(tokenizer, sample_conversation):
    """Test Alpaca rendering."""
    config = RenderConfig(format=ChatFormat.ALPACA)
    renderer = AlpacaRenderer(tokenizer, config)

    rendered = renderer.render_conversation(sample_conversation)
    assert "### Instruction:" in rendered
    assert "### Response:" in rendered

    stops = renderer.get_stop_sequences()
    assert "### Instruction:" in stops


def test_vicuna_renderer(tokenizer, sample_conversation):
    """Test Vicuna rendering."""
    renderer = VicunaRenderer(tokenizer, RenderConfig())

    rendered = renderer.render_conversation(sample_conversation)
    assert "USER:" in rendered
    assert "ASSISTANT:" in rendered

    stops = renderer.get_stop_sequences()
    assert "USER:" in stops


def test_zephyr_renderer(tokenizer, sample_conversation):
    """Test Zephyr rendering."""
    renderer = ZephyrRenderer(tokenizer, RenderConfig())

    rendered = renderer.render_conversation(sample_conversation)
    assert "<|system|>" in rendered
    assert "<|user|>" in rendered
    assert "<|assistant|>" in rendered
    assert "</s>" in rendered


def test_llama_renderer(tokenizer):
    """Test Llama rendering."""
    renderer = LlamaRenderer(tokenizer, RenderConfig())

    conv = Conversation(
        messages=[
            Message(role="system", content="System prompt"),
            Message(role="user", content="Question"),
            Message(role="assistant", content="Answer"),
        ]
    )

    rendered = renderer.render_conversation(conv)
    assert "[INST]" in rendered
    assert "[/INST]" in rendered
    assert "<<SYS>>" in rendered
    assert "<</SYS>>" in rendered


def test_mistral_renderer(tokenizer):
    """Test Mistral rendering."""
    renderer = MistralRenderer(tokenizer, RenderConfig())

    conv = Conversation(
        messages=[
            Message(role="user", content="Question"),
            Message(role="assistant", content="Answer"),
        ]
    )

    rendered = renderer.render_conversation(conv)
    assert "[INST]" in rendered
    assert "[/INST]" in rendered


def test_get_renderer(tokenizer):
    """Test renderer factory."""
    renderer = get_renderer(ChatFormat.CHATML, tokenizer)
    assert isinstance(renderer, ChatMLRenderer)

    renderer = get_renderer("alpaca", tokenizer)
    assert isinstance(renderer, AlpacaRenderer)

    # Test with custom config
    config = RenderConfig(format=ChatFormat.VICUNA, max_length=256)
    renderer = get_renderer(ChatFormat.VICUNA, tokenizer, config)
    assert renderer.config.max_length == 256


def test_tokenize_conversation(tokenizer, sample_conversation):
    """Test conversation tokenization."""
    config = RenderConfig(
        format=ChatFormat.CHATML,
        max_length=128,
        only_assistant=True,
    )
    renderer = ChatMLRenderer(tokenizer, config)

    encoded = renderer.tokenize_conversation(sample_conversation)

    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert "labels" in encoded
    assert "token_weights" in encoded

    # Check shapes
    assert encoded["input_ids"].shape[-1] == 128  # max_length
    assert encoded["labels"].shape == encoded["input_ids"].shape


def test_token_weights_computation(tokenizer):
    """Test token weight computation for training."""
    config = RenderConfig(
        format=ChatFormat.CHATML,
        only_assistant=True,
        mask_user=True,
        mask_system=True,
    )
    renderer = ChatMLRenderer(tokenizer, config)

    conv = Conversation(
        messages=[
            Message(role="system", content="System"),
            Message(role="user", content="User"),
            Message(role="assistant", content="Assistant"),
        ]
    )

    encoded = renderer.tokenize_conversation(conv)
    weights = encoded["token_weights"]

    # Weights should be 0 for system/user, 1 for assistant
    assert weights.min() == 0  # Some tokens masked
    assert weights.max() == 1  # Some tokens not masked


def test_build_generation_prompt():
    """Test generation prompt building."""
    messages = [
        {"role": "user", "content": "Hello"},
    ]

    prompt = build_generation_prompt(messages, ChatFormat.CHATML)
    assert "<|im_start|>user" in prompt
    assert "Hello" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_parse_response():
    """Test response parsing."""
    response = "<|im_start|>assistant\nHello there!<|im_end|>"
    parsed = parse_response(response, ChatFormat.CHATML)
    assert parsed == "Hello there!"

    response = "### Response:\nHello there!\n### Instruction:"
    parsed = parse_response(response, ChatFormat.ALPACA)
    assert "Hello there!" in parsed
    assert "###" not in parsed


def test_get_stop_sequences():
    """Test stop sequence retrieval."""
    stops = get_stop_sequences(ChatFormat.CHATML)
    assert "<|im_end|>" in stops

    stops = get_stop_sequences(ChatFormat.ALPACA)
    assert "### Instruction:" in stops


def test_build_supervised_example(tokenizer):
    """Test supervised example building."""
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]

    example = build_supervised_example(
        messages,
        tokenizer,
        ChatFormat.CHATML,
        max_length=64,
        only_assistant=True,
    )

    assert "input_ids" in example
    assert "labels" in example
    assert "token_weights" in example
    assert example["input_ids"].shape[-1] == 64


def test_apply_token_weights():
    """Test token weight application."""
    input_ids = torch.ones(10)

    # Test with weight list
    weights = [
        TokenWeight(0, 3, 0.5),
        TokenWeight(7, 10, 0.0),
    ]

    weighted = apply_token_weights(input_ids, weights)
    assert weighted[1] == 0.5
    assert weighted[5] == 1.0
    assert weighted[8] == 0.0

    # Test with tensor weights
    weight_tensor = torch.tensor([0.5] * 10)
    weighted = apply_token_weights(input_ids, weight_tensor)
    assert torch.all(weighted == 0.5)


def test_detect_chat_format():
    """Test chat format detection."""
    fmt = detect_chat_format("qwen-chat")
    assert fmt == ChatFormat.CHATML

    fmt = detect_chat_format("alpaca-7b")
    assert fmt == ChatFormat.ALPACA

    fmt = detect_chat_format("llama-2-chat")
    assert fmt == ChatFormat.LLAMA

    fmt = detect_chat_format("unknown-model")
    assert fmt == ChatFormat.CHATML  # Default


def test_convert_dataset_to_conversations():
    """Test dataset conversion."""
    # Test with list of dicts
    dataset = [
        {"text": "This is text 1"},
        {"text": "This is text 2"},
    ]

    conversations = convert_dataset_to_conversations(dataset, text_column="text")
    assert len(conversations) == 2
    assert conversations[0].messages[1].content == "This is text 1"

    # Test with conversation format
    dataset = [
        {
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
            ]
        },
    ]

    conversations = convert_dataset_to_conversations(dataset)
    assert len(conversations) == 1
    assert len(conversations[0].messages) == 2


def test_create_chat_template():
    """Test Jinja2 template creation."""
    template = create_chat_template(ChatFormat.CHATML)
    assert "<|im_start|>" in template
    assert "message['role']" in template

    template = create_chat_template(ChatFormat.ALPACA)
    assert "### Instruction:" in template

    # Test custom template
    template = create_chat_template(
        ChatFormat.CUSTOM,
        user_template="USER: {content}",
        assistant_template="BOT: {content}",
    )
    assert "USER:" in template
    assert "BOT:" in template


def test_custom_renderer(tokenizer):
    """Test custom renderer with templates."""
    config = RenderConfig(
        format=ChatFormat.CUSTOM,
        system_template="System: {content}",
        user_template="Human: {content}",
        assistant_template="AI: {content}",
    )

    from autotrain.rendering.message_renderer import CustomRenderer

    renderer = CustomRenderer(tokenizer, config)

    conv = Conversation(
        messages=[
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
    )

    rendered = renderer.render_conversation(conv)
    assert "System: Be helpful" in rendered
    assert "Human: Hello" in rendered
    assert "AI: Hi there" in rendered
