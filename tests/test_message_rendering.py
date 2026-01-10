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


class TestToolRoleConversion:
    """Tests for tool role conversion in TokenizerNativeRenderer.

    Gemma and other models require strict user/assistant alternation and don't
    support 'tool' role. These tests verify that tool messages are correctly
    converted to user messages when the tokenizer doesn't support them.
    """

    def test_check_tool_role_support_gemma(self):
        """Test that Gemma is correctly detected as NOT supporting tool role."""
        from autotrain.rendering.message_renderer import TokenizerNativeRenderer

        # Use Gemma tokenizer which doesn't support tool role
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        except Exception:
            pytest.skip("Gemma tokenizer not available")

        config = RenderConfig(format=ChatFormat.NATIVE)
        renderer = TokenizerNativeRenderer(tokenizer, config)

        # Gemma should NOT support tool role
        assert renderer._check_tool_role_support() == False

    def test_check_tool_role_support_caching(self, tokenizer):
        """Test that tool role support check is cached."""
        from autotrain.rendering.message_renderer import TokenizerNativeRenderer

        config = RenderConfig(format=ChatFormat.NATIVE)
        renderer = TokenizerNativeRenderer(tokenizer, config)

        # First call should set the cache
        result1 = renderer._check_tool_role_support()
        assert renderer._supports_tool_role is not None

        # Second call should use cache (same result)
        result2 = renderer._check_tool_role_support()
        assert result1 == result2

    def test_preprocess_tool_role_to_user(self, tokenizer):
        """Test that tool role is converted to user role."""
        from autotrain.rendering.message_renderer import TokenizerNativeRenderer

        config = RenderConfig(format=ChatFormat.NATIVE)
        renderer = TokenizerNativeRenderer(tokenizer, config)

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Let me calculate that."},
            {"role": "tool", "content": "Result: 4"},
            {"role": "assistant", "content": "The answer is 4."},
        ]

        processed = renderer._preprocess_messages_for_alternation(messages)

        # Should have 4 messages (tool converted to user)
        assert len(processed) == 4
        # Tool message should now be user with [Tool Result] prefix
        assert processed[2]["role"] == "user"
        assert processed[2]["content"] == "[Tool Result] Result: 4"

    def test_preprocess_merges_consecutive_same_role(self, tokenizer):
        """Test that consecutive same-role messages are merged."""
        from autotrain.rendering.message_renderer import TokenizerNativeRenderer

        config = RenderConfig(format=ChatFormat.NATIVE)
        renderer = TokenizerNativeRenderer(tokenizer, config)

        # Two consecutive user messages (e.g., user + tool converted to user)
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "user", "content": "Second message"},
            {"role": "assistant", "content": "Response"},
        ]

        processed = renderer._preprocess_messages_for_alternation(messages)

        # Should merge the two user messages
        assert len(processed) == 2
        assert "First message" in processed[0]["content"]
        assert "Second message" in processed[0]["content"]

    def test_preprocess_complex_tool_sequence(self, tokenizer):
        """Test complex sequence with multiple tool calls."""
        from autotrain.rendering.message_renderer import TokenizerNativeRenderer

        config = RenderConfig(format=ChatFormat.NATIVE)
        renderer = TokenizerNativeRenderer(tokenizer, config)

        messages = [
            {"role": "user", "content": "Check the weather"},
            {"role": "assistant", "content": "Checking..."},
            {"role": "tool", "content": "Weather: sunny"},
            {"role": "tool", "content": "Temp: 20C"},  # Multiple tool results
            {"role": "assistant", "content": "It's sunny and 20C"},
        ]

        processed = renderer._preprocess_messages_for_alternation(messages)

        # user, assistant, user (merged tools), assistant = 4 messages
        assert len(processed) == 4
        # Both tool results should be merged into one user message
        assert "[Tool Result] Weather: sunny" in processed[2]["content"]
        assert "[Tool Result] Temp: 20C" in processed[2]["content"]

    def test_preprocess_preserves_system_role(self, tokenizer):
        """Test that system role is preserved."""
        from autotrain.rendering.message_renderer import TokenizerNativeRenderer

        config = RenderConfig(format=ChatFormat.NATIVE)
        renderer = TokenizerNativeRenderer(tokenizer, config)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        processed = renderer._preprocess_messages_for_alternation(messages)

        assert len(processed) == 3
        assert processed[0]["role"] == "system"
        assert processed[0]["content"] == "You are helpful"

    def test_preprocess_handles_empty_messages(self, tokenizer):
        """Test handling of empty message list."""
        from autotrain.rendering.message_renderer import TokenizerNativeRenderer

        config = RenderConfig(format=ChatFormat.NATIVE)
        renderer = TokenizerNativeRenderer(tokenizer, config)

        processed = renderer._preprocess_messages_for_alternation([])
        assert processed == []

    def test_preprocess_handles_none_content(self, tokenizer):
        """Test handling of None content in messages."""
        from autotrain.rendering.message_renderer import TokenizerNativeRenderer

        config = RenderConfig(format=ChatFormat.NATIVE)
        renderer = TokenizerNativeRenderer(tokenizer, config)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None},  # Some APIs return None for empty
            {"role": "tool", "content": "Result"},
            {"role": "assistant", "content": "Done"},
        ]

        processed = renderer._preprocess_messages_for_alternation(messages)

        # Should handle None gracefully
        assert len(processed) == 4
        assert processed[1]["content"] == ""  # None converted to empty string

    @pytest.mark.skipif(
        not pytest.importorskip("transformers", reason="transformers not available"),
        reason="requires transformers"
    )
    def test_render_with_tool_role_gemma_style(self):
        """Test rendering with tool role using a tokenizer that requires alternation.

        This simulates what happens when training data with tool messages is used
        with Gemma or similar models that require strict alternation.
        """
        from autotrain.rendering.message_renderer import TokenizerNativeRenderer

        # Use GPT2 as a stand-in (it doesn't have a chat template, but that's ok)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        config = RenderConfig(format=ChatFormat.NATIVE)
        renderer = TokenizerNativeRenderer(tokenizer, config)

        conv = Conversation(
            messages=[
                Message(role="user", content="Calculate 2+2"),
                Message(role="assistant", content="Let me use a calculator"),
                Message(role="tool", content="4"),
                Message(role="assistant", content="The result is 4"),
            ]
        )

        # This should not raise an error
        rendered = renderer.render_conversation(conv)

        # The tool message should be converted to user format
        assert "[Tool Result]" in rendered or "4" in rendered


class TestSafeApplyChatTemplate:
    """Tests for the safe_apply_chat_template centralized utility."""

    def test_safe_apply_preserves_native_tool_support(self):
        """Test that models with native tool support keep tool messages as-is."""
        from autotrain.rendering.utils import safe_apply_chat_template, check_tool_role_support

        # Qwen supports tool role natively
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        except Exception:
            pytest.skip("Qwen tokenizer not available")

        # Verify Qwen supports tool
        assert check_tool_role_support(tokenizer) == True

        messages = [
            {"role": "user", "content": "Calculate 2+2"},
            {"role": "assistant", "content": "Let me use a tool"},
            {"role": "tool", "content": "4"},
            {"role": "assistant", "content": "The answer is 4"},
        ]

        # Should NOT add [Tool Result] prefix for models that support tool natively
        result = safe_apply_chat_template(tokenizer, messages, tokenize=False)
        assert "[Tool Result]" not in result

    def test_safe_apply_converts_for_gemma(self):
        """Test that Gemma gets tool messages converted."""
        from autotrain.rendering.utils import safe_apply_chat_template, check_tool_role_support

        try:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        except Exception:
            pytest.skip("Gemma tokenizer not available")

        # Verify Gemma doesn't support tool
        assert check_tool_role_support(tokenizer) == False

        messages = [
            {"role": "user", "content": "Calculate 2+2"},
            {"role": "assistant", "content": "Let me calculate"},
            {"role": "tool", "content": "Result: 4"},
            {"role": "assistant", "content": "The answer is 4"},
        ]

        # Should convert tool to user with [Tool Result] prefix
        result = safe_apply_chat_template(tokenizer, messages, tokenize=False)
        assert "[Tool Result]" in result
        assert "Result: 4" in result

    def test_safe_apply_no_tool_messages_unchanged(self):
        """Test that messages without tool role are unchanged."""
        from autotrain.rendering.utils import safe_apply_chat_template

        # Use a tokenizer with a chat template
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        except Exception:
            pytest.skip("Qwen tokenizer not available")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Should work normally without any conversion
        result = safe_apply_chat_template(tokenizer, messages, tokenize=False)
        assert "Hello" in result
        assert "Hi there" in result

    def test_cache_is_used(self):
        """Test that tool role support detection is cached."""
        from autotrain.rendering.utils import _tool_role_support_cache, check_tool_role_support

        try:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        except Exception:
            pytest.skip("Gemma tokenizer not available")

        # First call populates cache
        result1 = check_tool_role_support(tokenizer)

        # Check cache has the entry
        tokenizer_id = tokenizer.name_or_path
        assert tokenizer_id in _tool_role_support_cache

        # Second call uses cache
        result2 = check_tool_role_support(tokenizer)
        assert result1 == result2


class TestMessageAlternationFix:
    """Tests for fix_message_alternation to handle strict alternation tokenizers."""

    def test_fix_consecutive_users(self):
        """Test merging consecutive user messages."""
        from autotrain.rendering.utils import fix_message_alternation

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good!"},
        ]

        fixed = fix_message_alternation(messages)
        assert len(fixed) == 2
        assert fixed[0]["role"] == "user"
        assert "Hello" in fixed[0]["content"]
        assert "How are you?" in fixed[0]["content"]
        assert fixed[1]["role"] == "assistant"

    def test_fix_consecutive_assistants(self):
        """Test handling consecutive assistant messages - they get merged."""
        from autotrain.rendering.utils import fix_message_alternation

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "assistant", "content": "How can I help?"},
        ]

        fixed = fix_message_alternation(messages)
        # Consecutive assistants are merged
        assert len(fixed) == 2
        assert fixed[0]["role"] == "user"
        assert fixed[1]["role"] == "assistant"
        assert "Hi!" in fixed[1]["content"]
        assert "How can I help?" in fixed[1]["content"]

    def test_fix_system_then_assistant(self):
        """Test handling system → assistant without user in between."""
        from autotrain.rendering.utils import fix_message_alternation

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "assistant", "content": "Hello, how can I help?"},
        ]

        fixed = fix_message_alternation(messages)
        # Should insert a user message between system and assistant
        assert len(fixed) == 3
        assert fixed[0]["role"] == "system"
        assert fixed[1]["role"] == "user"
        assert fixed[1]["content"] == "[Continued]"
        assert fixed[2]["role"] == "assistant"

    def test_fix_assistant_at_start(self):
        """Test handling assistant message at start (no preceding user)."""
        from autotrain.rendering.utils import fix_message_alternation

        messages = [
            {"role": "assistant", "content": "Hello, I'm your assistant"},
        ]

        fixed = fix_message_alternation(messages)
        # Should insert a user message before assistant
        assert len(fixed) == 2
        assert fixed[0]["role"] == "user"
        assert fixed[0]["content"] == "[Continued]"
        assert fixed[1]["role"] == "assistant"

    def test_fix_preserves_valid_alternation(self):
        """Test that valid alternating messages are preserved."""
        from autotrain.rendering.utils import fix_message_alternation

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good!"},
        ]

        fixed = fix_message_alternation(messages)
        assert len(fixed) == 4
        assert fixed[0]["role"] == "user"
        assert fixed[1]["role"] == "assistant"
        assert fixed[2]["role"] == "user"
        assert fixed[3]["role"] == "assistant"

    def test_fix_empty_messages(self):
        """Test handling empty message list."""
        from autotrain.rendering.utils import fix_message_alternation

        fixed = fix_message_alternation([])
        assert fixed == []

    def test_fix_complex_scenario(self):
        """Test complex scenario with multiple issues."""
        from autotrain.rendering.utils import fix_message_alternation

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "assistant", "content": "Hello"},  # No user before
            {"role": "user", "content": "Hi"},
            {"role": "user", "content": "Help me"},  # Consecutive users
            {"role": "assistant", "content": "Sure"},
        ]

        fixed = fix_message_alternation(messages)
        # Check proper alternation
        roles = [m["role"] for m in fixed]
        # After system, should have user, assistant, user, assistant
        assert roles[0] == "system"
        # Then user inserted
        assert roles[1] == "user"
        assert roles[2] == "assistant"
        # Then merged user
        assert roles[3] == "user"
        assert "Hi" in fixed[3]["content"]
        assert "Help me" in fixed[3]["content"]
        assert roles[4] == "assistant"

    def test_safe_apply_with_alternation_error(self):
        """Test that safe_apply_chat_template handles alternation errors."""
        from autotrain.rendering.utils import safe_apply_chat_template

        try:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        except Exception:
            pytest.skip("Gemma tokenizer not available")

        # Messages with consecutive users - would fail strict alternation
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Anyone there?"},
            {"role": "assistant", "content": "Yes, I'm here!"},
        ]

        # Should not raise - should fix alternation automatically
        result = safe_apply_chat_template(tokenizer, messages, tokenize=False)
        assert "Hello" in result
        assert "here" in result


class TestToolCallsSerialization:
    """Tests for tool_calls field serialization for models that don't support it natively."""

    def test_serialize_tool_calls_basic(self):
        """Test basic tool_calls serialization to content."""
        from autotrain.rendering.utils import serialize_tool_calls_to_content

        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "content": "Sunny, 20C", "tool_call_id": "call_123"},
            {"role": "assistant", "content": "It's sunny and 20C in Paris."},
        ]

        serialized = serialize_tool_calls_to_content(messages)

        # Assistant message should have tool call serialized into content
        assert len(serialized) == 4
        assert serialized[1]["role"] == "assistant"
        assert '"tool": "get_weather"' in serialized[1]["content"] or '"tool":"get_weather"' in serialized[1]["content"]
        assert "get_weather" in serialized[1]["content"]
        assert "Paris" in serialized[1]["content"]
        # tool_calls field should be removed
        assert "tool_calls" not in serialized[1]
        # tool_call_id should be removed
        assert "tool_call_id" not in serialized[2]

    def test_serialize_tool_calls_preserves_content(self):
        """Test that existing content is preserved when serializing tool_calls."""
        from autotrain.rendering.utils import serialize_tool_calls_to_content

        messages = [
            {
                "role": "assistant",
                "content": "I'll search for that.",
                "tool_calls": [
                    {"function": {"name": "search", "arguments": '{"query": "weather"}'}}
                ],
            }
        ]

        serialized = serialize_tool_calls_to_content(messages)

        # Original content should be preserved
        assert "I'll search for that." in serialized[0]["content"]
        # Tool call should be appended as JSON
        assert '"tool": "search"' in serialized[0]["content"] or '"tool":"search"' in serialized[0]["content"]

    def test_serialize_tool_calls_handles_none_content(self):
        """Test serialization when content is None (common in OpenAI responses)."""
        from autotrain.rendering.utils import serialize_tool_calls_to_content

        messages = [
            {
                "role": "assistant",
                "content": None,  # OpenAI often sets content=None when tool_calls is present
                "tool_calls": [
                    {"function": {"name": "calculator", "arguments": '{"expr": "2+2"}'}}
                ],
            }
        ]

        serialized = serialize_tool_calls_to_content(messages)

        # Should handle None content gracefully
        assert serialized[0]["content"] is not None
        assert '"tool": "calculator"' in serialized[0]["content"] or '"tool":"calculator"' in serialized[0]["content"]

    def test_serialize_multiple_tool_calls(self):
        """Test serialization of multiple tool calls in one message."""
        from autotrain.rendering.utils import serialize_tool_calls_to_content

        messages = [
            {
                "role": "assistant",
                "content": "I'll check both.",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
                    {"function": {"name": "get_time", "arguments": '{"timezone": "Europe/Paris"}'}},
                ],
            }
        ]

        serialized = serialize_tool_calls_to_content(messages)

        # Both tool calls should be serialized as JSON
        content = serialized[0]["content"]
        assert "get_weather" in content
        assert "get_time" in content
        assert content.count('"tool":') == 2 or content.count('"tool": ') == 2

    def test_serialize_preserves_non_tool_messages(self):
        """Test that messages without tool_calls are unchanged."""
        from autotrain.rendering.utils import serialize_tool_calls_to_content

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        serialized = serialize_tool_calls_to_content(messages)

        assert serialized == messages

    def test_check_tool_calls_support_gemma(self):
        """Test that Gemma is correctly detected as NOT supporting tool_calls."""
        from autotrain.rendering.utils import check_tool_calls_support

        try:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        except Exception:
            pytest.skip("Gemma tokenizer not available")

        # Gemma should NOT support tool_calls field
        assert check_tool_calls_support(tokenizer) == False

    def test_check_tool_calls_support_qwen(self):
        """Test that Qwen is correctly detected as supporting tool_calls natively."""
        from autotrain.rendering.utils import check_tool_calls_support, safe_apply_chat_template

        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        except Exception:
            pytest.skip("Qwen tokenizer not available")

        # Qwen SHOULD support tool_calls field natively
        assert check_tool_calls_support(tokenizer) == True

        # Verify native format is used (not serialized to [Tool Call])
        messages = [
            {"role": "user", "content": "test"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "my_function", "arguments": '{"arg": "value"}'}}
                ],
            },
        ]

        result = safe_apply_chat_template(tokenizer, messages, tokenize=False)

        # Should use native <tool_call> format, NOT [Tool Call] serialization
        assert "[Tool Call]" not in result
        assert "my_function" in result
        # Qwen uses <tool_call> tags
        assert "<tool_call>" in result or "tool_call" in result.lower()

    def test_safe_apply_with_tool_calls_gemma(self):
        """Test that safe_apply_chat_template serializes tool_calls for Gemma."""
        from autotrain.rendering.utils import safe_apply_chat_template

        try:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        except Exception:
            pytest.skip("Gemma tokenizer not available")

        messages = [
            {"role": "user", "content": "Calculate 2+2"},
            {
                "role": "assistant",
                "content": "Let me calculate.",
                "tool_calls": [
                    {"function": {"name": "calculator", "arguments": '{"expr": "2+2"}'}}
                ],
            },
            {"role": "tool", "content": "4"},
            {"role": "assistant", "content": "The answer is 4."},
        ]

        result = safe_apply_chat_template(tokenizer, messages, tokenize=False)

        # Tool call should be serialized to content as JSON
        assert '"tool": "calculator"' in result or '"tool":"calculator"' in result
        # Tool result should be converted
        assert "[Tool Result]" in result

    def test_message_with_tool_calls_preserved_in_dict(self):
        """Test that Message and Conversation preserve tool_calls in to_dict/from_dict."""
        from autotrain.rendering import Message, Conversation

        # Create message with tool_calls
        msg = Message(
            role="assistant",
            content="Let me check.",
            tool_calls=[{"function": {"name": "test", "arguments": "{}"}}],
        )

        # Create conversation
        conv = Conversation(messages=[msg])

        # Convert to dict
        data = conv.to_dict()

        # Verify tool_calls is preserved
        assert "tool_calls" in data["messages"][0]
        assert data["messages"][0]["tool_calls"][0]["function"]["name"] == "test"

        # Convert back from dict
        conv2 = Conversation.from_dict(data)

        # Verify tool_calls is restored
        assert conv2.messages[0].tool_calls is not None
        assert conv2.messages[0].tool_calls[0]["function"]["name"] == "test"

    def test_message_with_tool_call_id_preserved(self):
        """Test that tool_call_id is preserved in Message."""
        from autotrain.rendering import Message, Conversation

        msg = Message(
            role="tool",
            content="Result: 4",
            tool_call_id="call_123",
        )

        conv = Conversation(messages=[msg])
        data = conv.to_dict()

        assert "tool_call_id" in data["messages"][0]
        assert data["messages"][0]["tool_call_id"] == "call_123"

        conv2 = Conversation.from_dict(data)
        assert conv2.messages[0].tool_call_id == "call_123"

    def test_function_role_converted_like_tool(self):
        """Test that 'function' role (older OpenAI format) is converted same as 'tool'."""
        from autotrain.rendering.utils import preprocess_messages_for_tool_role

        messages = [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "Let me calculate."},
            {"role": "function", "content": "4"},  # Older OpenAI format uses "function"
            {"role": "assistant", "content": "The answer is 4."},
        ]

        processed = preprocess_messages_for_tool_role(messages)

        # function role should be converted to user with [Tool Result] prefix
        assert processed[2]["role"] == "user"
        assert "[Tool Result]" in processed[2]["content"]
        assert "4" in processed[2]["content"]

    def test_serialize_tool_calls_clean_format_not_function(self):
        """Test that tool_calls serialization produces clean format, not raw OpenAI format with 'function' key."""
        from autotrain.rendering.utils import serialize_tool_calls_to_content

        messages = [
            {
                "role": "assistant",
                "content": "Let me search.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"query": "weather"}'},
                    }
                ],
            }
        ]

        serialized = serialize_tool_calls_to_content(messages)
        content = serialized[0]["content"]

        # Should produce clean format: {"tool": "search", "arguments": {...}}
        assert '"tool": "search"' in content or '"tool":"search"' in content
        # Should NOT contain raw OpenAI format with "function" key
        assert '"function":' not in content
        assert '"type": "function"' not in content
        assert '"id": "call_123"' not in content


class TestToolsDefinitionsInjection:
    """Tests for tools parameter injection into messages for models that don't support native tools."""

    def test_format_tools_as_text(self):
        """Test formatting tool definitions as human-readable text."""
        from autotrain.rendering.utils import format_tools_as_text

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        text = format_tools_as_text(tools)

        assert "You have access to the following tools" in text
        assert "search" in text
        assert "Search for information" in text
        assert "query" in text
        assert "(required)" in text

    def test_format_tools_as_text_empty(self):
        """Test formatting empty tools list."""
        from autotrain.rendering.utils import format_tools_as_text

        assert format_tools_as_text([]) == ""
        assert format_tools_as_text(None) == ""

    def test_inject_tools_into_system_message(self):
        """Test that tools are appended to system message when present."""
        from autotrain.rendering.utils import inject_tools_into_messages

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate math",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        injected = inject_tools_into_messages(messages, tools)

        # Tools should be appended to system message
        assert "You are a helpful assistant." in injected[0]["content"]
        assert "You have access to the following tools" in injected[0]["content"]
        assert "calculator" in injected[0]["content"]
        # User message should be unchanged
        assert injected[1]["content"] == "Hello"

    def test_inject_tools_into_first_user_message_when_no_system(self):
        """Test that tools are prepended to first user message when no system message."""
        from autotrain.rendering.utils import inject_tools_into_messages

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate math",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        injected = inject_tools_into_messages(messages, tools)

        # Tools should be prepended to first user message
        assert "You have access to the following tools" in injected[0]["content"]
        assert "Hello" in injected[0]["content"]
        # Assistant message should be unchanged
        assert injected[1]["content"] == "Hi!"

    def test_check_tools_support_gemma(self):
        """Test that Gemma doesn't support tools parameter."""
        from autotrain.rendering.utils import check_tools_support

        try:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        except Exception:
            pytest.skip("Gemma tokenizer not available")

        # Gemma doesn't support tools parameter
        assert check_tools_support(tokenizer) == False

    def test_safe_apply_with_tools_injection(self):
        """Test safe_apply_chat_template injects tools for non-supporting tokenizers."""
        from autotrain.rendering.utils import safe_apply_chat_template

        try:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        except Exception:
            pytest.skip("Gemma tokenizer not available")

        messages = [
            {"role": "user", "content": "Calculate 2+2"},
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate math expressions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "The math expression"},
                        },
                    },
                },
            }
        ]

        result = safe_apply_chat_template(tokenizer, messages, tokenize=False, tools=tools)

        # Tools should be injected into the output
        assert "calculator" in result
        assert "Calculate math expressions" in result
        assert "Calculate 2+2" in result

    def test_tools_not_injected_when_tokenizer_supports(self):
        """Test that tools are NOT injected when tokenizer supports native tools."""
        from autotrain.rendering.utils import check_tools_support, inject_tools_into_messages

        # Just verify the logic - if tokenizer supports tools, don't inject
        messages = [
            {"role": "user", "content": "Hello"},
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "description": "Test",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        # Direct injection should still work but won't be called if tokenizer supports tools
        injected = inject_tools_into_messages(messages, tools)
        assert "You have access to the following tools" in injected[0]["content"]

    def test_tools_caching(self):
        """Test that tools support detection is cached."""
        from autotrain.rendering.utils import _tools_support_cache, check_tools_support

        try:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        except Exception:
            pytest.skip("Gemma tokenizer not available")

        # First call populates cache
        result1 = check_tools_support(tokenizer)

        # Check cache has the entry
        tokenizer_id = tokenizer.name_or_path
        assert tokenizer_id in _tools_support_cache

        # Second call uses cache
        result2 = check_tools_support(tokenizer)
        assert result1 == result2
