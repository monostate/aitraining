"""
Test inference with models that have resized tokenizers.
This test ensures that models trained with additional tokens can be loaded and used for inference.
"""

import os
import tempfile

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from autotrain.generation import create_completer


def test_inference_with_resized_tokenizer():
    """Test that inference works with models that have resized tokenizers."""

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create a small model and tokenizer
        model_name = "hf-internal-testing/tiny-random-gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get initial sizes
        initial_vocab_size = len(tokenizer)
        initial_embedding_size = model.get_input_embeddings().num_embeddings

        # 2. Add special tokens (simulating what happens during training)
        special_tokens = {"additional_special_tokens": ["<SPECIAL1>", "<SPECIAL2>"]}
        num_added = tokenizer.add_special_tokens(special_tokens)
        assert num_added > 0, "Should have added special tokens"

        # 3. Resize model embeddings (this is what happens during training)
        model.resize_token_embeddings(len(tokenizer))

        # Verify the resize worked
        new_vocab_size = len(tokenizer)
        new_embedding_size = model.get_input_embeddings().num_embeddings
        assert new_vocab_size > initial_vocab_size
        assert new_embedding_size == new_vocab_size

        # 4. Save the model and tokenizer (simulating end of training)
        model_path = os.path.join(tmpdir, "test_model")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        # 5. Load the model for inference using create_completer
        # This should handle the tokenizer resizing automatically
        completer = create_completer(model=model_path, tokenizer=model_path, completer_type="token")

        # 6. Test inference
        test_text = "Hello world <SPECIAL1> test"
        result = completer.complete(test_text, max_new_tokens=10)

        # Should complete without errors
        assert result is not None
        assert result.text is not None
        assert len(result.text) > 0

        # 7. Verify the model and tokenizer sizes match
        assert len(completer.tokenizer) == completer.model.get_input_embeddings().num_embeddings


def test_inference_without_resize_still_works():
    """Test that normal models without resizing still work."""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save a model without any resizing
        model_name = "hf-internal-testing/tiny-random-gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_path = os.path.join(tmpdir, "test_model")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        # Load for inference
        completer = create_completer(model=model_path, tokenizer=model_path, completer_type="token")

        # Test inference
        result = completer.complete("Hello world", max_new_tokens=10)

        assert result is not None
        assert result.text is not None
        assert len(result.text) > 0


def test_inference_handles_mismatch_gracefully():
    """Test that inference handles tokenizer/model size mismatch gracefully."""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a model
        model_name = "hf-internal-testing/tiny-random-gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add tokens to tokenizer but DON'T resize model
        # (simulating a potential bug or edge case)
        special_tokens = {"additional_special_tokens": ["<SPECIAL1>", "<SPECIAL2>"]}
        tokenizer.add_special_tokens(special_tokens)

        # Save with mismatch
        model_path = os.path.join(tmpdir, "test_model")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        # create_completer should handle this mismatch
        completer = create_completer(model=model_path, tokenizer=model_path, completer_type="token")

        # After loading, sizes should match (model resized to match tokenizer)
        assert len(completer.tokenizer) == completer.model.get_input_embeddings().num_embeddings

        # Test inference works
        result = completer.complete("Test text", max_new_tokens=5)
        assert result is not None


if __name__ == "__main__":
    # Run tests
    test_inference_with_resized_tokenizer()
    print("✓ Test with resized tokenizer passed")

    test_inference_without_resize_still_works()
    print("✓ Test without resize passed")

    test_inference_handles_mismatch_gracefully()
    print("✓ Test mismatch handling passed")

    print("\n✅ All tokenizer resize inference tests passed!")
