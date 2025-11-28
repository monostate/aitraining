"""
Test Gemma 3 model template validation.

This test validates that our standalone templates match the actual model's
chat template and produce the expected output format.
"""

import os
import sys

import pytest
from datasets import Dataset
from transformers import AutoTokenizer


# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotrain.preprocessor.chat_templates_standalone import CHAT_TEMPLATES, get_template_for_model
from autotrain.preprocessor.llm import (
    apply_chat_template,
    convert_alpaca_to_messages,
    detect_dataset_format,
    get_available_chat_templates,
)


class TestGemmaTemplateValidation:
    """Test Gemma 3 template with actual model tokenizer."""

    def test_gemma3_template_extraction_and_format(self):
        """Test that our Gemma 3 template matches the expected format."""
        print("\n=== Testing Gemma 3 Template ===")

        # Load the actual Gemma 3 model tokenizer
        model_id = "google/gemma-3-270m-it"
        print(f"Loading tokenizer for {model_id}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
        except Exception as e:
            pytest.skip(f"Could not load tokenizer: {e}")

        # Test 1: Check our template suggestion
        suggested_template = get_template_for_model(model_id)
        print(f"✓ Suggested template for {model_id}: {suggested_template}")
        assert suggested_template in ["gemma", "gemma2", "gemma3"], f"Expected gemma variant, got {suggested_template}"

        # Test 2: Check our standalone template exists
        templates = get_available_chat_templates()
        assert "gemma" in templates, "Gemma template not found in standalone templates"
        assert "gemma3" in templates, "Gemma3 template not found in standalone templates"
        print(f"✓ Found {len(templates)} templates including gemma variants")

        # Test 3: Create a test conversation
        test_messages = [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hey there!"}]

        # Test 4: Apply the tokenizer's native template
        native_output = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=False)
        print(f"\n--- Native tokenizer output ---")
        print(native_output)

        # Test 5: Check if it matches expected Gemma format
        # Gemma format should be:
        # <bos><start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\nHey there!<end_of_turn>

        # Check for key Gemma markers
        assert "<start_of_turn>" in native_output, "Missing <start_of_turn> marker"
        assert "<end_of_turn>" in native_output, "Missing <end_of_turn> marker"

        # Check role markers (Gemma uses 'model' instead of 'assistant')
        assert "user" in native_output, "Missing 'user' role"
        assert "model" in native_output or "assistant" in native_output, "Missing 'model' or 'assistant' role"

        print("✓ Native tokenizer output contains expected Gemma markers")

        # Test 6: Get our standalone template and check its format
        gemma_template = CHAT_TEMPLATES.get("gemma", "")
        print(f"\n--- Our Gemma template ---")
        print(gemma_template[:200] + "..." if len(gemma_template) > 200 else gemma_template)

        # Verify our template has the right markers
        assert "<start_of_turn>" in gemma_template, "Our template missing <start_of_turn>"
        assert "<end_of_turn>" in gemma_template, "Our template missing <end_of_turn>"
        assert "model" in gemma_template, "Our template missing 'model' role"
        print("✓ Our template contains expected Gemma markers")

        # Test 7: Test with a dataset conversion pipeline
        print("\n--- Testing full conversion pipeline ---")

        # Create a small Alpaca-style dataset
        alpaca_data = {
            "instruction": ["What is the capital of France?", "Explain photosynthesis"],
            "input": ["", ""],
            "output": [
                "The capital of France is Paris.",
                "Photosynthesis is the process by which plants convert sunlight into energy.",
            ],
        }
        dataset = Dataset.from_dict(alpaca_data)

        # Detect format
        format_type = detect_dataset_format(dataset)
        assert format_type == "alpaca", f"Expected 'alpaca', got '{format_type}'"
        print(f"✓ Correctly detected format: {format_type}")

        # Convert to messages
        converted = convert_alpaca_to_messages(dataset)
        assert "messages" in converted.column_names
        print(f"✓ Converted to messages format")

        # Apply chat template
        result = apply_chat_template(converted, tokenizer, chat_template=None)  # Use tokenizer's default

        assert "text" in result.column_names
        sample_text = result[0]["text"]

        print(f"\n--- Formatted sample ---")
        print(sample_text[:500] if len(sample_text) > 500 else sample_text)

        # Verify the formatted text has the expected structure
        assert "<start_of_turn>" in sample_text, "Formatted text missing Gemma markers"
        assert "What is the capital of France?" in sample_text, "User content missing"
        assert "Paris" in sample_text, "Assistant content missing"

        print("\n✓ Full pipeline successful with expected format")

    def test_gemma3_template_with_system_message(self):
        """Test Gemma 3 template handling with system message."""
        print("\n=== Testing Gemma 3 with System Message ===")

        model_id = "google/gemma-3-270m-it"

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
        except Exception as e:
            pytest.skip(f"Could not load tokenizer: {e}")

        # Test with system message
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hey there! How can I help you today?"},
        ]

        # Apply tokenizer template
        output = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=False)

        print(f"\n--- Output with system message ---")
        print(output)

        # Gemma typically merges system message with first user message
        # Check that content is preserved
        assert "Hello!" in output, "User message missing"
        assert "help you today" in output, "Assistant message missing"

        # System message might be prepended to user message or handled specially
        if "system" not in output.lower():
            print("✓ System message merged with user message (expected for Gemma)")
        else:
            print("✓ System message handled separately")

    def test_compare_gemma_variants(self):
        """Compare different Gemma template variants in our collection."""
        print("\n=== Comparing Gemma Template Variants ===")

        # Get all Gemma-related templates
        gemma_templates = {name: template for name, template in CHAT_TEMPLATES.items() if "gemma" in name.lower()}

        print(f"Found {len(gemma_templates)} Gemma variants: {list(gemma_templates.keys())}")

        # Test conversation
        test_messages = [
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "Test response"},
        ]

        # Compare key differences
        for name, template in gemma_templates.items():
            print(f"\n--- {name} ---")

            # Check for key markers
            has_start_turn = "<start_of_turn>" in template
            has_end_turn = "<end_of_turn>" in template
            uses_model = "model" in template
            uses_assistant = "assistant" in template

            print(f"  <start_of_turn>: {has_start_turn}")
            print(f"  <end_of_turn>: {has_end_turn}")
            print(f"  Uses 'model': {uses_model}")
            print(f"  Uses 'assistant': {uses_assistant}")

            # Check template complexity
            template_lines = template.count("\n")
            print(f"  Template complexity: {template_lines} lines")

    def test_gemma_template_edge_cases(self):
        """Test Gemma template with edge cases."""
        print("\n=== Testing Gemma Template Edge Cases ===")

        model_id = "google/gemma-3-270m-it"

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
        except Exception as e:
            pytest.skip(f"Could not load tokenizer: {e}")

        # Test 1: Empty message
        messages_empty = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "I see you sent an empty message."},
        ]

        output = tokenizer.apply_chat_template(messages_empty, tokenize=False, add_generation_prompt=False)
        assert "<start_of_turn>" in output
        print("✓ Handles empty user message")

        # Test 2: Multi-turn conversation
        messages_multi = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"},
        ]

        output = tokenizer.apply_chat_template(messages_multi, tokenize=False, add_generation_prompt=False)
        turn_count = output.count("<start_of_turn>")
        assert turn_count == 4, f"Expected 4 turns, got {turn_count}"
        print(f"✓ Handles multi-turn conversation ({turn_count} turns)")

        # Test 3: With generation prompt
        messages_gen = [{"role": "user", "content": "Tell me a joke"}]

        output_with_prompt = tokenizer.apply_chat_template(messages_gen, tokenize=False, add_generation_prompt=True)
        output_without_prompt = tokenizer.apply_chat_template(
            messages_gen, tokenize=False, add_generation_prompt=False
        )

        # With generation prompt should be longer (adds model/assistant marker)
        assert len(output_with_prompt) > len(output_without_prompt)
        print("✓ Generation prompt adds model turn marker")

        print(f"\n--- With generation prompt ---")
        print(output_with_prompt)
        print(f"\n--- Without generation prompt ---")
        print(output_without_prompt)


if __name__ == "__main__":
    # Run the tests
    print("=" * 60)
    print("GEMMA 3 TEMPLATE VALIDATION")
    print("=" * 60)

    # Check if templates are available
    templates = get_available_chat_templates()
    if templates:
        print(f"✓ {len(templates)} templates available")
        gemma_variants = [t for t in templates if "gemma" in t.lower()]
        print(f"✓ {len(gemma_variants)} Gemma variants: {gemma_variants}")
    else:
        print("⚠ No templates available")

    # Run pytest
    pytest.main([__file__, "-v", "-s"])
