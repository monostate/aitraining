#!/usr/bin/env python3
"""Test that Cebolinha dataset format works with AutoTrain dataset conversion."""

import json

from datasets import Dataset
from transformers import AutoTokenizer

from autotrain.preprocessor.llm import apply_chat_template, detect_dataset_format, standardize_dataset


def test_cebolinha_qa_format():
    """Test that Q&A format from Cebolinha dataset works correctly."""

    # Create sample dataset in Q&A format (like Cebolinha)
    data = [
        {"question": "Por que o carteiro foi à feira?", "answer": "Polque tinha uma calta na manga."},
        {"question": "Como você está hoje?", "answer": "Estou muito bem, obligado!"},
        {"question": "O que é um computador?", "answer": "É uma máquina que plocessa infolmações."},
    ]

    dataset = Dataset.from_list(data)

    # Test format detection
    detected_format = detect_dataset_format(dataset)
    print(f"Detected format: {detected_format}")
    assert detected_format == "qa", f"Expected 'qa' format, got {detected_format}"

    # Test conversion to messages format
    converted = standardize_dataset(dataset, "messages", auto_detect=True)
    print(f"\nFirst example after conversion:")
    print(f"Columns: {converted.column_names}")
    print(json.dumps(converted[0]["messages"], indent=2, ensure_ascii=False))

    # Verify structure
    assert "messages" in converted.column_names
    assert len(converted[0]["messages"]) == 2
    assert converted[0]["messages"][0]["role"] == "user"
    assert converted[0]["messages"][0]["content"] == "Por que o carteiro foi à feira?"
    assert converted[0]["messages"][1]["role"] == "assistant"
    assert converted[0]["messages"][1]["content"] == "Polque tinha uma calta na manga."

    # Test with chat template application
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    # Apply chat template
    templated = apply_chat_template(converted, tokenizer, messages_column="messages")

    print(f"\nFirst example after template application:")
    print(templated[0]["text"][:200] + "...")

    # Verify text field was created
    assert "text" in templated.column_names
    assert templated[0]["text"].startswith("<bos>")  # Gemma template starts with <bos>

    print("\n✅ All tests passed! Cebolinha Q&A format works perfectly with AutoTrain.")


if __name__ == "__main__":
    test_cebolinha_qa_format()
