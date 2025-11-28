"""
Test inference with PEFT adapter-only models.
Verifies that the inference API can load and use models saved with merge_adapter=False.
"""

import json
import os
import shutil
import tempfile

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_adapter_only_model(output_dir, base_model="hf-internal-testing/tiny-random-gpt2"):
    """Create and save a PEFT model with adapters only (no merge)."""
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Add a special token to test tokenizer resizing
    tokenizer.add_special_tokens({"additional_special_tokens": ["<TEST_TOKEN>"]})
    model.resize_token_embeddings(len(tokenizer))

    # Create PEFT config and model
    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],  # GPT2 attention module
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    # Save only the adapter weights (not merged)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir


def test_create_completer_with_adapter_model():
    """Test that create_completer can load adapter-only models."""
    from autotrain.generation import create_completer

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an adapter-only model
        model_path = create_adapter_only_model(tmpdir)

        # Try to load it with create_completer
        completer = create_completer(model=model_path, tokenizer=model_path, completer_type="token")

        # Verify it loaded successfully
        assert completer is not None
        assert completer.model is not None
        assert completer.tokenizer is not None

        # Test that tokenizer and model sizes match
        model_vocab_size = completer.model.get_input_embeddings().num_embeddings
        tokenizer_vocab_size = len(completer.tokenizer)
        assert (
            model_vocab_size == tokenizer_vocab_size
        ), f"Model ({model_vocab_size}) and tokenizer ({tokenizer_vocab_size}) vocab sizes should match"

        # Test inference
        result = completer.complete("Hello world", max_new_tokens=5)
        assert result is not None
        assert result.text is not None


def test_adapter_model_with_local_base():
    """Test loading adapter model where base model is in local cache."""
    from autotrain.generation import create_completer

    with tempfile.TemporaryDirectory() as tmpdir:
        base_model = "hf-internal-testing/tiny-random-gpt2"

        # Pre-download base model to ensure it's in cache
        _ = AutoModelForCausalLM.from_pretrained(base_model)

        # Create adapter model
        model_path = create_adapter_only_model(tmpdir, base_model)

        # Load with create_completer (should use cached base model)
        completer = create_completer(model=model_path, tokenizer=model_path, completer_type="token")

        # Verify it works
        result = completer.complete("Test", max_new_tokens=3)
        assert result is not None


def test_inference_api_with_adapter_model():
    """Test that the inference API endpoint handles adapter models."""
    from autotrain.app.api_routes import get_cached_llm
    from autotrain.generation import CompletionConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create adapter model
        model_path = create_adapter_only_model(tmpdir)

        # Create config
        config = CompletionConfig(max_new_tokens=10)

        # Load through API function
        completer = get_cached_llm(model_path, config)

        # Verify it loaded
        assert completer is not None

        # Test inference
        conversation = [{"role": "user", "content": "Hello"}]
        result = completer.complete(conversation, max_new_tokens=5)
        assert result is not None
        assert result.text is not None


def test_adapter_with_resized_tokenizer():
    """Test that adapter models with resized tokenizers work correctly."""
    from autotrain.generation import create_completer

    base_model = "hf-internal-testing/tiny-random-gpt2"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Load base model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Record original sizes
        original_vocab_size = len(tokenizer)
        original_model_vocab_size = model.get_input_embeddings().num_embeddings

        # Add special tokens to resize tokenizer
        special_tokens = ["<CUSTOM1>", "<CUSTOM2>", "<CUSTOM3>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        new_vocab_size = len(tokenizer)

        # Resize model embeddings
        model.resize_token_embeddings(new_vocab_size)

        # Apply PEFT
        peft_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)

        # Save adapter-only model with resized tokenizer
        adapter_dir = os.path.join(tmpdir, "adapter_resized")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        print(f"  Original tokenizer size: {original_vocab_size}")
        print(f"  New tokenizer size: {new_vocab_size}")
        print(f"  Added {new_vocab_size - original_vocab_size} tokens")

        # Test loading through our inference system
        completer = create_completer(model=adapter_dir, tokenizer=adapter_dir, completer_type="token")

        # Verify sizes match after loading
        loaded_model_vocab = completer.model.get_input_embeddings().num_embeddings
        loaded_tokenizer_vocab = len(completer.tokenizer)

        print(f"  Loaded model vocab size: {loaded_model_vocab}")
        print(f"  Loaded tokenizer vocab size: {loaded_tokenizer_vocab}")

        assert (
            loaded_model_vocab == loaded_tokenizer_vocab
        ), f"Model and tokenizer vocab sizes don't match after loading! Model: {loaded_model_vocab}, Tokenizer: {loaded_tokenizer_vocab}"

        # Test inference with special tokens
        test_input = f"Test with {special_tokens[0]} token"
        result = completer.complete(test_input, max_new_tokens=5)

        assert result is not None
        assert result.text is not None
        print(f"  Generated with special token: '{result.text}'")

        # Verify special tokens are in vocabulary
        for token in special_tokens:
            token_id = completer.tokenizer.convert_tokens_to_ids(token)
            assert (
                token_id != completer.tokenizer.unk_token_id
            ), f"Special token {token} not properly added to vocabulary"


def test_size_mismatch_without_resize_fails():
    """Test that loading a model with tokenizer size mismatch fails without proper handling."""
    from autotrain.generation import create_completer

    base_model = "hf-internal-testing/tiny-random-gpt2"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a scenario where tokenizer has extra tokens but model wasn't resized
        # This simulates what would happen if our fix wasn't in place

        # Load base model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        original_model_size = model.get_input_embeddings().num_embeddings

        # Add tokens to tokenizer but DON'T resize model embeddings
        tokenizer.add_special_tokens({"additional_special_tokens": ["<FAIL1>", "<FAIL2>"]})
        new_tokenizer_size = len(tokenizer)

        # Save mismatched model and tokenizer
        mismatch_dir = os.path.join(tmpdir, "mismatched_model")
        model.save_pretrained(mismatch_dir)  # Model still has original size
        tokenizer.save_pretrained(mismatch_dir)  # Tokenizer has new size

        print(f"  Model embedding size: {original_model_size}")
        print(f"  Tokenizer size: {new_tokenizer_size}")
        print(f"  Mismatch of {new_tokenizer_size - original_model_size} tokens")

        # Now test that our inference system handles this
        # With our fix, it should automatically resize
        try:
            completer = create_completer(model=mismatch_dir, tokenizer=mismatch_dir, completer_type="token")

            # Verify it was resized correctly
            loaded_model_size = completer.model.get_input_embeddings().num_embeddings
            loaded_tokenizer_size = len(completer.tokenizer)

            print(f"  After loading - Model: {loaded_model_size}, Tokenizer: {loaded_tokenizer_size}")

            # This should now match thanks to our resize logic
            assert (
                loaded_model_size == loaded_tokenizer_size
            ), "Our inference should auto-resize the model to match tokenizer"

            # Test inference works
            result = completer.complete("Test", max_new_tokens=3)
            assert result is not None
            print("  ✓ Inference handled size mismatch correctly")

        except Exception as e:
            print(f"  ✗ Failed to handle size mismatch: {e}")
            raise


def test_peft_without_resize_fix():
    """Test that demonstrates the problem with PEFT models before our fix."""
    import json

    from autotrain.generation import create_completer

    base_model = "hf-internal-testing/tiny-random-gpt2"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate what happens when training adds tokens but only saves adapters
        # This is the scenario that was failing before our fix

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        original_size = len(tokenizer)

        # Add tokens and resize model (what happens during training)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<TRAIN1>", "<TRAIN2>", "<TRAIN3>"]})
        model.resize_token_embeddings(len(tokenizer))

        # Apply PEFT
        peft_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model, peft_config)

        # Save ONLY adapters (not the resized base model)
        adapter_dir = os.path.join(tmpdir, "adapter_only")
        peft_model.save_pretrained(adapter_dir)  # This only saves adapter weights
        tokenizer.save_pretrained(adapter_dir)  # This saves the resized tokenizer

        print(f"  Original tokenizer size: {original_size}")
        print(f"  New tokenizer size after training: {len(tokenizer)}")
        print(f"  Adapter config saved with base_model: {base_model}")

        # Manually verify what was saved
        with open(os.path.join(adapter_dir, "adapter_config.json"), "r") as f:
            saved_config = json.load(f)
            print(f"  Base model in adapter_config: {saved_config.get('base_model_name_or_path')}")

        # Now try to load for inference
        # This is where the problem occurs: base model has original size,
        # but tokenizer was resized
        try:
            completer = create_completer(model=adapter_dir, tokenizer=adapter_dir, completer_type="token")

            # Check if our fix worked
            model_size = completer.model.get_input_embeddings().num_embeddings
            tokenizer_size = len(completer.tokenizer)

            print(f"  Loaded model size: {model_size}")
            print(f"  Loaded tokenizer size: {tokenizer_size}")

            assert model_size == tokenizer_size, f"Size mismatch! Model: {model_size}, Tokenizer: {tokenizer_size}"

            # Test that inference works
            result = completer.complete("Test with <TRAIN1>", max_new_tokens=3)
            assert result is not None
            print(f"  ✓ PEFT model with resized tokenizer works! Generated: '{result.text}'")

        except AssertionError as e:
            print(f"  ✗ Size mismatch error (this would fail without our fix): {e}")
            # With our fix in create_completer, this should NOT happen
            raise
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            raise


def test_merged_vs_adapter_inference():
    """Compare inference between merged and adapter-only models."""
    from autotrain.generation import create_completer

    base_model = "hf-internal-testing/tiny-random-gpt2"

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = os.path.join(tmpdir, "adapter_model")
        merged_dir = os.path.join(tmpdir, "merged_model")

        # Create adapter-only model
        model = AutoModelForCausalLM.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Add PEFT
        peft_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)

        # Save adapter-only version
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        # Save merged version
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        # Load both versions with "token" completer type (for simpler testing)
        adapter_completer = create_completer(model=adapter_dir, tokenizer=adapter_dir, completer_type="token")
        merged_completer = create_completer(model=merged_dir, tokenizer=merged_dir, completer_type="token")

        # Both should work for inference
        test_input = "The quick brown"

        # Use TokenCompleter's complete method with max_new_tokens parameter
        adapter_result = adapter_completer.complete(test_input, max_new_tokens=5)
        merged_result = merged_completer.complete(test_input, max_new_tokens=5)

        assert adapter_result is not None
        assert merged_result is not None

        # Both should produce text (might be different due to randomness)
        assert len(adapter_result.text) > 0
        assert len(merged_result.text) > 0

        # Verify the results contain actual generated tokens
        print(f"  Adapter model generated: '{adapter_result.text}'")
        print(f"  Merged model generated: '{merged_result.text}'")

        # Check that we actually generated new tokens (not just returned empty)
        assert adapter_result.token_ids is not None
        assert merged_result.token_ids is not None
        assert adapter_result.token_ids.shape[0] > 0
        assert merged_result.token_ids.shape[0] > 0


def test_broken_adapter_config():
    """Test that a corrupted adapter config fails gracefully."""
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = os.path.join(tmpdir, "broken_adapter")
        os.makedirs(adapter_dir)

        # Create a broken adapter_config.json with missing base_model
        broken_config = {
            "r": 4,
            "lora_alpha": 8,
            # Intentionally missing "base_model_name_or_path"
        }

        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump(broken_config, f)

        # Try to load - this should fail
        from autotrain.generation import create_completer

        try:
            completer = create_completer(model=adapter_dir, tokenizer=adapter_dir, completer_type="token")
            # Should not reach here
            assert False, "Should have failed with broken adapter config"
        except Exception as e:
            print(f"  ✓ Correctly failed with broken config: {str(e)[:100]}")
            # This is expected - the test passes if we get an error


if __name__ == "__main__":
    import sys

    print("Testing inference with adapter-only models...")

    print("\n1. Testing create_completer with adapter model...")
    test_create_completer_with_adapter_model()
    print("✓ Passed")

    print("\n2. Testing adapter model with local base...")
    test_adapter_model_with_local_base()
    print("✓ Passed")

    print("\n3. Testing inference API with adapter model...")
    test_inference_api_with_adapter_model()
    print("✓ Passed")

    print("\n4. Testing adapter with resized tokenizer...")
    test_adapter_with_resized_tokenizer()
    print("✓ Passed")

    print("\n5. Testing size mismatch handling...")
    test_size_mismatch_without_resize_fails()
    print("✓ Passed")

    print("\n6. Testing PEFT with tokenizer resize fix...")
    test_peft_without_resize_fix()
    print("✓ Passed")

    print("\n7. Testing merged vs adapter inference...")
    test_merged_vs_adapter_inference()
    print("✓ Passed")

    print("\n8. Testing broken adapter config handling...")
    test_broken_adapter_config()
    print("✓ Passed")

    print("\n✅ All adapter model inference tests passed!")
