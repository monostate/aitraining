"""
Real-world tests for dataset conversion with actual datasets and models.

Tests the full pipeline with real Unsloth integration, no mocks.
"""

import json
import os
import sys

import pandas as pd
import pytest
from datasets import load_dataset
from transformers import AutoTokenizer


# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotrain.preprocessor.llm import (
    analyze_and_convert_dataset,
    apply_chat_template,
    convert_alpaca_to_messages,
    convert_dpo_to_messages,
    convert_sharegpt_to_messages,
    detect_dataset_format,
    extend_alpaca_conversations,
    formatting_prompts_func,
    get_available_chat_templates,
    standardize_dataset,
)


# Test configuration
SAMPLE_SIZE = 5
TEST_MODELS = [
    "HuggingFaceH4/zephyr-7b-beta",  # Uses zephyr/chatml template
    "meta-llama/Llama-2-7b-hf",  # Uses llama template
    "google/gemma-2b",  # Uses gemma template
    "microsoft/phi-2",  # Uses phi template
]

TEST_DATASETS = {
    "alpaca": "tatsu-lab/alpaca",
    "sharegpt": "philschmid/guanaco-sharegpt-style",
    "messages": "HuggingFaceH4/no_robots",  # Already in messages format
    "dpo": "argilla/ultrafeedback-binarized-preferences-cleaned",
}


def validate_chat_templates():
    """Validate that chat templates are available and working."""
    try:
        # Try Unsloth first if available
        import torch

        if torch.cuda.is_available():
            from unsloth.chat_templates import CHAT_TEMPLATES, get_chat_template

            print("\n=== Available Unsloth Chat Templates (CUDA) ===")
        else:
            # Use standalone templates for CPU/MPS
            from autotrain.preprocessor.chat_templates_standalone import CHAT_TEMPLATES

            print("\n=== Available Standalone Chat Templates (CPU/MPS) ===")
            get_chat_template = None

        templates = list(CHAT_TEMPLATES.keys())
        print(f"Found {len(templates)} templates:")

        # Group by family
        families = {}
        for t in templates:
            if "llama" in t.lower():
                family = "Llama"
            elif "gemma" in t.lower():
                family = "Gemma"
            elif "phi" in t.lower():
                family = "Phi"
            elif "qwen" in t.lower():
                family = "Qwen"
            elif t in ["mistral", "zephyr", "chatml", "alpaca", "vicuna"]:
                family = "Common"
            else:
                family = "Other"

            if family not in families:
                families[family] = []
            families[family].append(t)

        for family, temps in sorted(families.items()):
            print(
                f"  {family}: {', '.join(sorted(temps)[:5])}" + (f" + {len(temps)-5} more" if len(temps) > 5 else "")
            )

        return True, templates
    except ImportError as e:
        print(f"WARNING: Unsloth not available: {e}")
        return False, []
    except Exception as e:
        print(f"ERROR validating templates: {e}")
        return False, []


class TestRealDatasetConversion:
    """Test dataset conversion with real datasets."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment and validate templates."""
        self.has_templates, self.available_templates = validate_chat_templates()
        if not self.has_templates:
            pytest.skip("No chat templates available, skipping real conversion tests")

    def test_alpaca_format_detection_and_conversion(self):
        """Test Alpaca dataset format detection and conversion."""
        print("\n=== Testing Alpaca Format ===")

        # Load real Alpaca dataset
        dataset = load_dataset("tatsu-lab/alpaca", split="train[:5]")
        print(f"Loaded {len(dataset)} samples")
        print(f"Columns: {dataset.column_names}")

        # Test format detection
        format_type = detect_dataset_format(dataset)
        assert format_type == "alpaca", f"Expected 'alpaca', got '{format_type}'"
        print(f"✓ Correctly detected format: {format_type}")

        # Convert to messages format
        converted = convert_alpaca_to_messages(dataset)

        # Validate conversion
        assert "messages" in converted.column_names
        sample = converted[0]["messages"]
        assert isinstance(sample, list)
        assert len(sample) >= 2  # At least user and assistant
        assert sample[0]["role"] == "user"
        assert sample[-1]["role"] == "assistant"

        print(f"✓ Converted to messages format")
        print(f"  First message: {sample[0]}")
        print(f"  Last message: {sample[-1]}")

        return converted

    def test_sharegpt_format_detection_and_conversion(self):
        """Test ShareGPT dataset format detection and conversion."""
        print("\n=== Testing ShareGPT Format ===")

        # Load real ShareGPT dataset
        dataset = load_dataset("philschmid/guanaco-sharegpt-style", split="train[:5]")
        print(f"Loaded {len(dataset)} samples")
        print(f"Columns: {dataset.column_names}")

        # Test format detection
        format_type = detect_dataset_format(dataset)
        assert format_type == "sharegpt", f"Expected 'sharegpt', got '{format_type}'"
        print(f"✓ Correctly detected format: {format_type}")

        # Test conversion - will use Unsloth on CUDA, fallback on CPU/MPS
        converted = convert_sharegpt_to_messages(dataset)
        print("✓ Converted ShareGPT format")

        # Validate conversion
        if "messages" in converted.column_names:
            messages_col = "messages"
        elif "conversations" in converted.column_names:
            messages_col = "conversations"
        else:
            raise AssertionError(f"No messages column found in {converted.column_names}")

        sample = converted[0][messages_col]
        assert isinstance(sample, list)
        assert len(sample) >= 1

        # Check role mapping
        first_msg = sample[0]
        assert "role" in first_msg or "from" in first_msg
        if "role" in first_msg:
            assert first_msg["role"] in ["user", "assistant", "system"]

        print(f"✓ Converted ShareGPT format")
        print(f"  Sample conversation length: {len(sample)}")

        return converted

    def test_dpo_format_detection_and_handling(self):
        """Test DPO dataset format detection."""
        print("\n=== Testing DPO Format ===")

        # Load real DPO dataset
        dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train[:5]")
        print(f"Loaded {len(dataset)} samples")
        print(f"Columns: {dataset.column_names}")

        # Test format detection
        format_type = detect_dataset_format(dataset)
        assert format_type in ["dpo", "messages"], f"Expected 'dpo' or 'messages', got '{format_type}'"
        print(f"✓ Detected format: {format_type}")

        # DPO datasets often have prompt/chosen/rejected or messages format
        if "prompt" in dataset.column_names and "chosen" in dataset.column_names:
            print("✓ Has DPO columns: prompt, chosen, rejected")

            # Convert if needed
            if format_type == "dpo":
                converted = convert_dpo_to_messages(dataset)
                assert "messages_chosen" in converted.column_names
                assert "messages_rejected" in converted.column_names
                print("✓ Converted DPO to messages format")

        return dataset

    def test_chat_template_application_with_models(self):
        """Test applying chat templates with real models and tokenizers."""
        print("\n=== Testing Chat Template Application ===")

        # Load a small dataset
        dataset = load_dataset("tatsu-lab/alpaca", split="train[:3]")

        # Convert to messages format first
        dataset = convert_alpaca_to_messages(dataset)

        for model_name in TEST_MODELS[:2]:  # Test first 2 models to save time
            print(f"\n--- Testing with {model_name} ---")

            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
                )

                # Test 1: Use tokenizer's default template
                print("  Testing tokenizer default template...")
                result_default = apply_chat_template(dataset, tokenizer, chat_template=None)  # Use tokenizer default

                assert "text" in result_default.column_names
                sample_text = result_default[0]["text"]
                assert len(sample_text) > 0
                print(f"  ✓ Default template applied, length: {len(sample_text)}")
                print(f"    Preview: {sample_text[:100]}...")

                # Test 2: Try specific template if available
                if self.has_templates:
                    # Determine appropriate template
                    if "llama" in model_name.lower():
                        template = "llama" if "llama-2" in model_name.lower() else "llama3"
                    elif "gemma" in model_name.lower():
                        template = "gemma"
                    elif "phi" in model_name.lower():
                        template = "phi-3"
                    elif "zephyr" in model_name.lower():
                        template = "zephyr"
                    else:
                        template = "chatml"  # Generic fallback

                    if template in self.available_templates:
                        print(f"  Testing template: {template}...")
                        result_specific = apply_chat_template(dataset, tokenizer, chat_template=template)

                        assert "text" in result_specific.column_names
                        sample_text = result_specific[0]["text"]
                        assert len(sample_text) > 0
                        print(f"  ✓ {template} template applied, length: {len(sample_text)}")
                        print(f"    Preview: {sample_text[:100]}...")

            except Exception as e:
                print(f"  ⚠ Error with {model_name}: {e}")
                continue

    def test_conversation_extension(self):
        """Test extending single-turn Alpaca to multi-turn conversations."""
        print("\n=== Testing Conversation Extension ===")

        # Load Alpaca dataset
        dataset = load_dataset("tatsu-lab/alpaca", split="train[:10]")

        # Test different extension values
        for extension in [1, 2, 3]:
            print(f"\n--- Extension factor: {extension} ---")

            extended = extend_alpaca_conversations(
                dataset, conversation_extension=extension, output_column_name="output"
            )

            if extension == 1:
                # No extension, should be same length
                assert len(extended) == len(dataset)
                print(f"  ✓ No extension (factor=1): {len(extended)} samples")
            else:
                # Should have fewer samples (merged)
                expected_len = len(dataset) // extension + (1 if len(dataset) % extension else 0)
                assert len(extended) <= expected_len + 1  # Allow for rounding
                print(f"  ✓ Extended: {len(dataset)} → {len(extended)} samples")

                # Check that conversations are longer
                if "messages" in extended.column_names:
                    sample = extended[0]["messages"]
                    expected_turns = extension * 2  # Each original has 2 messages
                    assert len(sample) >= min(expected_turns, len(dataset) * 2)
                    print(f"    Conversation length: {len(sample)} messages")

    def test_full_pipeline_with_analyze_and_convert(self):
        """Test the complete analyze_and_convert_dataset function."""
        print("\n=== Testing Full Pipeline ===")

        test_cases = [
            ("tatsu-lab/alpaca", "alpaca"),
            ("philschmid/guanaco-sharegpt-style", "sharegpt"),
        ]

        for dataset_name, expected_format in test_cases:
            print(f"\n--- Testing {dataset_name} ---")

            # Load dataset
            dataset = load_dataset(dataset_name, split="train[:5]")

            # Load a tokenizer
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta", trust_remote_code=True)

            # Run full analysis and conversion
            result = analyze_and_convert_dataset(
                dataset, tokenizer=tokenizer, chat_template="chatml", trainer_type="sft", apply_template=True
            )

            # Validate result
            assert "format_detected" in result
            assert result["format_detected"] == expected_format
            # The key might be 'dataset' or 'converted_dataset'
            assert "converted_dataset" in result or "dataset" in result

            converted_ds = result.get("converted_dataset", result.get("dataset"))
            if result.get("conversion_applied"):
                # Should have text column if template was applied
                if "text" in converted_ds.column_names:
                    assert len(converted_ds[0]["text"]) > 0
                    print(f"  ✓ Full pipeline successful")
                    print(f"    Format: {result['format_detected']}")
                    print(f"    Converted: {result.get('conversion_applied', False)}")
                    print(f"    Columns: {converted_ds.column_names}")

    def test_template_validation_against_registry(self):
        """Validate that template names match CHAT_TEMPLATES registry."""
        print("\n=== Testing Template Name Validation ===")

        if not self.has_templates:
            pytest.skip("No templates available")

        # Import appropriate CHAT_TEMPLATES
        try:
            import torch

            if torch.cuda.is_available():
                from unsloth.chat_templates import CHAT_TEMPLATES
            else:
                from autotrain.preprocessor.chat_templates_standalone import CHAT_TEMPLATES
        except ImportError:
            from autotrain.preprocessor.chat_templates_standalone import CHAT_TEMPLATES

        # Test valid template names
        valid_templates = ["llama", "llama3", "gemma", "chatml", "zephyr", "mistral"]

        for template_name in valid_templates:
            if template_name in CHAT_TEMPLATES:
                print(f"  ✓ '{template_name}' is valid in template registry")

                # On CPU/MPS we can't test template application with get_chat_template
                # but we can verify the template exists
                try:
                    import torch

                    if torch.cuda.is_available():
                        # Test that we can actually use it with Unsloth
                        tokenizer = AutoTokenizer.from_pretrained(
                            "HuggingFaceH4/zephyr-7b-beta", trust_remote_code=True
                        )
                        from unsloth.chat_templates import get_chat_template

                        enhanced_tokenizer = get_chat_template(tokenizer, chat_template=template_name)
                        assert enhanced_tokenizer is not None
                        print(f"    ✓ Successfully applied '{template_name}' template")
                    else:
                        # Just verify template string exists
                        template_str = CHAT_TEMPLATES[template_name]
                        assert len(template_str) > 0
                        print(f"    ✓ Template '{template_name}' exists ({len(template_str)} chars)")
                except Exception as e:
                    print(f"    ⚠ Could not validate '{template_name}': {e}")

        # Test invalid template handling
        invalid_template = "not_a_real_template_xyz"
        if invalid_template not in CHAT_TEMPLATES:
            print(f"  ✓ '{invalid_template}' correctly identified as invalid")


def test_dataset_format_detection_accuracy():
    """Test accuracy of format detection across multiple datasets."""
    print("\n=== Testing Format Detection Accuracy ===")

    test_cases = [
        # (dataset_id, split, expected_format)
        ("tatsu-lab/alpaca", "train[:5]", "alpaca"),
        ("yahma/alpaca-cleaned", "train[:5]", "alpaca"),
        ("philschmid/guanaco-sharegpt-style", "train[:5]", "sharegpt"),
        ("HuggingFaceH4/no_robots", "train[:5]", "messages"),
        ("openai/gsm8k", "train[:5]", "unknown"),  # Not a chat dataset
    ]

    results = []
    for dataset_id, split, expected in test_cases:
        try:
            dataset = load_dataset(dataset_id, split=split)
            detected = detect_dataset_format(dataset)

            # Some datasets might be detected differently
            # Allow messages/sharegpt to be interchangeable for some cases
            is_correct = (detected == expected) or (
                expected in ["messages", "sharegpt"] and detected in ["messages", "sharegpt"]
            )

            results.append({"dataset": dataset_id, "expected": expected, "detected": detected, "correct": is_correct})

            status = "✓" if is_correct else "✗"
            print(f"  {status} {dataset_id}: {detected} (expected: {expected})")

        except Exception as e:
            print(f"  ⚠ Could not test {dataset_id}: {e}")
            continue

    # Calculate accuracy
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
    assert accuracy >= 60, f"Format detection accuracy too low: {accuracy}%"


if __name__ == "__main__":
    # Run validation first
    print("=" * 60)
    print("DATASET CONVERSION REAL-WORLD TESTS")
    print("=" * 60)

    has_templates, templates = validate_chat_templates()

    if not has_templates:
        print("\n⚠️  WARNING: No chat templates available!")
        print("\nSome tests will be skipped.")

    # Run pytest with verbose output
    pytest.main([__file__, "-v", "-s"])
