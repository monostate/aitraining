#!/usr/bin/env python3
"""Test manual column mapping for custom dataset formats."""

import json

from datasets import Dataset
from transformers import AutoTokenizer

from autotrain.preprocessor.llm import convert_with_column_mapping, detect_dataset_format, standardize_dataset


def test_custom_column_mapping():
    """Test that manual column mapping works for non-standard formats."""

    print("\n=== Testing Custom Column Mapping ===\n")

    # Example 1: Custom Q&A columns
    print("1. Custom Q&A format (query/response):")
    data = [
        {"query": "What is AI?", "response": "Artificial Intelligence is..."},
        {"query": "How does ML work?", "response": "Machine Learning works by..."},
    ]
    dataset = Dataset.from_list(data)

    # Auto-detection should work for query/response
    detected = detect_dataset_format(dataset)
    print(f"   Detected format: {detected}")
    assert detected == "qa", f"Expected 'qa' but got {detected}"

    # Convert using auto-detection
    converted = standardize_dataset(dataset)
    print(f"   Auto-converted first example:")
    print(f"   {json.dumps(converted[0]['messages'], indent=6)}")

    print("\n2. Completely custom columns (my_input/my_output):")
    # Example 2: Completely custom columns
    data = [
        {"my_input": "Translate: Hello", "my_output": "Bonjour"},
        {"my_input": "Translate: Goodbye", "my_output": "Au revoir"},
    ]
    dataset = Dataset.from_list(data)

    # Auto-detection won't work for these custom columns
    detected = detect_dataset_format(dataset)
    print(f"   Detected format: {detected}")
    assert detected == "unknown", f"Expected 'unknown' but got {detected}"

    # Use manual column mapping
    column_mapping = {"user_col": "my_input", "assistant_col": "my_output"}

    converted = convert_with_column_mapping(dataset, column_mapping)
    print(f"   Manually mapped first example:")
    print(f"   {json.dumps(converted[0]['messages'], indent=6)}")

    # Verify structure
    assert "messages" in converted.column_names
    assert converted[0]["messages"][0]["content"] == "Translate: Hello"
    assert converted[0]["messages"][1]["content"] == "Bonjour"

    print("\n3. Alpaca-like with custom columns:")
    # Example 3: Alpaca-style with custom column names
    data = [
        {
            "task": "Summarize the following text",
            "context": "The quick brown fox jumps over the lazy dog. This is a pangram.",
            "summary": "A pangram containing all letters of the alphabet.",
        }
    ]
    dataset = Dataset.from_list(data)

    column_mapping = {"instruction_col": "task", "input_col": "context", "output_col": "summary"}

    converted = convert_with_column_mapping(dataset, column_mapping)
    print(f"   Alpaca-style mapped:")
    print(f"   {json.dumps(converted[0]['messages'], indent=6)}")

    # Verify the instruction and input were combined
    expected_user_msg = (
        "Summarize the following text\n\nInput: The quick brown fox jumps over the lazy dog. This is a pangram."
    )
    assert converted[0]["messages"][0]["content"] == expected_user_msg

    print("\n4. With system message:")
    # Example 4: Including a system message
    data = [
        {
            "system_prompt": "You are a helpful assistant.",
            "user_question": "What is 2+2?",
            "bot_answer": "2+2 equals 4.",
        }
    ]
    dataset = Dataset.from_list(data)

    column_mapping = {"system_col": "system_prompt", "user_col": "user_question", "assistant_col": "bot_answer"}

    converted = convert_with_column_mapping(dataset, column_mapping)
    print(f"   With system message:")
    print(f"   {json.dumps(converted[0]['messages'], indent=6)}")

    # Verify all three messages
    assert len(converted[0]["messages"]) == 3
    assert converted[0]["messages"][0]["role"] == "system"
    assert converted[0]["messages"][1]["role"] == "user"
    assert converted[0]["messages"][2]["role"] == "assistant"

    print("\n✅ All column mapping tests passed!")


def test_standardize_with_column_mapping():
    """Test that standardize_dataset correctly uses column_mapping parameter."""

    print("\n=== Testing standardize_dataset with column_mapping ===\n")

    data = [
        {"prompt": "Write a poem", "completion": "Roses are red..."},
    ]
    dataset = Dataset.from_list(data)

    # Should detect as Q&A (prompt/completion is in our patterns)
    detected = detect_dataset_format(dataset)
    print(f"Dataset with prompt/completion detected as: {detected}")

    # Convert with manual mapping to override
    column_mapping = {"user_col": "prompt", "assistant_col": "completion"}

    converted = standardize_dataset(dataset, column_mapping=column_mapping)
    print(f"Converted with mapping: {json.dumps(converted[0]['messages'], indent=3)}")

    assert converted[0]["messages"][0]["content"] == "Write a poem"
    assert converted[0]["messages"][1]["content"] == "Roses are red..."

    print("\n✅ standardize_dataset with column_mapping works correctly!")


if __name__ == "__main__":
    test_custom_column_mapping()
    test_standardize_with_column_mapping()
