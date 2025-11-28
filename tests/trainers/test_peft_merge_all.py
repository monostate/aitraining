"""
Comprehensive tests for PEFT merge functionality across all trainers.
Tests that PEFT models with merge_adapter=True save full weights correctly.
"""

import json
import os
import shutil
import tempfile

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


# Test with very small models for speed
TINY_LLM_MODEL = "hf-internal-testing/tiny-random-gpt2"
TINY_SEQ2SEQ_MODEL = "hf-internal-testing/tiny-random-t5"
TINY_VLM_MODEL = "hf-internal-testing/tiny-random-LlavaForConditionalGeneration"


@pytest.fixture(scope="module")
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_test_data_clm(temp_dir, num_samples=10):
    """Create minimal test data for CLM training."""
    # Create a proper dataset directory structure
    data_dir = os.path.join(temp_dir, "clm_data")
    os.makedirs(data_dir, exist_ok=True)

    # Create train.jsonl file
    train_path = os.path.join(data_dir, "train.jsonl")
    data = []
    for i in range(num_samples):
        data.append({"text": f"This is test sample {i}. It contains some text for training."})

    with open(train_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return data_dir


def create_test_data_seq2seq(temp_dir, num_samples=10):
    """Create minimal test data for Seq2Seq training."""
    # Create a proper dataset directory structure
    data_dir = os.path.join(temp_dir, "seq2seq_data")
    os.makedirs(data_dir, exist_ok=True)

    # Create train.jsonl file
    train_path = os.path.join(data_dir, "train.jsonl")
    data = []
    for i in range(num_samples):
        data.append({"text": f"Translate this sentence {i}", "target": f"Translated sentence {i}"})

    with open(train_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return data_dir


def test_clm_sft_peft_merge(temp_dir):
    """Test CLM SFT training with PEFT and merge_adapter=True."""
    from autotrain.trainers.clm.params import LLMTrainingParams
    from autotrain.trainers.clm.train_clm_sft import train

    data_path = create_test_data_clm(temp_dir)
    output_dir = os.path.join(temp_dir, "clm_sft_merged")

    config = LLMTrainingParams(
        model=TINY_LLM_MODEL,
        data_path=data_path,
        train_split="train",
        trainer="sft",
        project_name=output_dir,
        text_column="text",
        # PEFT settings
        peft=True,
        merge_adapter=True,  # This should merge and save full model
        lora_r=4,  # Small rank for testing
        lora_alpha=8,
        # Training settings
        epochs=1,  # Must be integer
        batch_size=2,
        lr=1e-4,
        logging_steps=1,
        save_total_limit=1,
        log="none",
        max_samples=5,  # Use only 5 samples for speed
    )

    # Train the model
    train(config)

    # Verify the model was saved as full weights, not adapters
    assert os.path.exists(output_dir), "Output directory should exist"

    # Check for model files (not adapter files)
    assert os.path.exists(os.path.join(output_dir, "pytorch_model.bin")) or os.path.exists(
        os.path.join(output_dir, "model.safetensors")
    ), "Merged model weights should be saved"

    # Adapter files should NOT exist if merge was successful
    assert not os.path.exists(
        os.path.join(output_dir, "adapter_config.json")
    ), "adapter_config.json should not exist after merge"
    assert not os.path.exists(os.path.join(output_dir, "adapter_model.bin")) and not os.path.exists(
        os.path.join(output_dir, "adapter_model.safetensors")
    ), "adapter weights should not exist after merge"

    # Try loading the model without PEFT to verify it's a full model
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # Test inference
    inputs = tokenizer("Test input", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5)
    assert outputs is not None, "Model should generate output"


def test_clm_sft_peft_no_merge(temp_dir):
    """Test CLM SFT training with PEFT and merge_adapter=False (keep adapters only)."""
    from autotrain.trainers.clm.params import LLMTrainingParams
    from autotrain.trainers.clm.train_clm_sft import train

    data_path = create_test_data_clm(temp_dir)
    output_dir = os.path.join(temp_dir, "clm_sft_adapters")

    config = LLMTrainingParams(
        model=TINY_LLM_MODEL,
        data_path=data_path,
        train_split="train",
        trainer="sft",
        project_name=output_dir,
        text_column="text",
        # PEFT settings
        peft=True,
        merge_adapter=False,  # Keep only adapters
        lora_r=4,
        lora_alpha=8,
        # Training settings
        epochs=1,  # Must be integer
        batch_size=2,
        lr=1e-4,
        logging_steps=1,
        save_total_limit=1,
        log="none",
        max_samples=5,
    )

    # Train the model
    train(config)

    # Verify only adapter files were saved
    assert os.path.exists(output_dir), "Output directory should exist"
    assert os.path.exists(
        os.path.join(output_dir, "adapter_config.json")
    ), "adapter_config.json should exist when merge_adapter=False"
    assert os.path.exists(os.path.join(output_dir, "adapter_model.bin")) or os.path.exists(
        os.path.join(output_dir, "adapter_model.safetensors")
    ), "adapter weights should exist when merge_adapter=False"

    # Full model weights should NOT exist
    assert not os.path.exists(os.path.join(output_dir, "pytorch_model.bin")) and not os.path.exists(
        os.path.join(output_dir, "model.safetensors")
    ), "Full model weights should not exist when merge_adapter=False"

    # Loading should require PEFT
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(TINY_LLM_MODEL)
    model = PeftModel.from_pretrained(base_model, output_dir)
    assert model is not None, "Should be able to load PEFT model"


def test_seq2seq_peft_merge(temp_dir):
    """Test Seq2Seq training with PEFT and merge_adapter=True."""
    from autotrain.trainers.seq2seq.__main__ import train
    from autotrain.trainers.seq2seq.params import Seq2SeqParams

    data_path = create_test_data_seq2seq(temp_dir)
    output_dir = os.path.join(temp_dir, "seq2seq_merged")

    config = Seq2SeqParams(
        model=TINY_SEQ2SEQ_MODEL,
        data_path=data_path,
        train_split="train",
        project_name=output_dir,
        text_column="text",
        target_column="target",
        # PEFT settings
        peft=True,
        merge_adapter=True,
        lora_r=4,
        lora_alpha=8,
        # Training settings
        epochs=1,  # Seq2Seq needs at least 1 epoch
        batch_size=2,
        lr=1e-4,
        logging_steps=1,
        save_total_limit=1,
        log="none",
        max_samples=5,  # Limit samples for speed
        max_seq_length=32,  # Reduce sequence length for speed
        max_target_length=32,
    )

    # Train the model
    train(config)

    # Verify full model was saved
    assert os.path.exists(output_dir), "Output directory should exist"

    # Check that model files exist (merged model should be present)
    assert os.path.exists(os.path.join(output_dir, "pytorch_model.bin")) or os.path.exists(
        os.path.join(output_dir, "model.safetensors")
    ), "Merged model weights should be saved"

    # Adapter files should NOT exist after merge (now consistent with CLM/VLM)
    assert not os.path.exists(
        os.path.join(output_dir, "adapter_config.json")
    ), "adapter_config.json should not exist after merge"
    assert not os.path.exists(os.path.join(output_dir, "adapter_model.bin")) and not os.path.exists(
        os.path.join(output_dir, "adapter_model.safetensors")
    ), "adapter weights should not exist after merge"

    # Try loading as regular model
    model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # Test inference
    inputs = tokenizer("Translate this", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5)
    assert outputs is not None, "Model should generate output"


@pytest.mark.skip(reason="VLM requires image data - implement if needed")
def test_vlm_peft_merge(temp_dir):
    """Test VLM training with PEFT and merge_adapter=True."""
    # TODO: Implement VLM test with proper image data
    pass


def test_tokenizer_resizing_with_merge(temp_dir):
    """Test that models with resized tokenizers work correctly after merge."""
    from autotrain.trainers.clm.params import LLMTrainingParams
    from autotrain.trainers.clm.train_clm_sft import train

    data_path = create_test_data_clm(temp_dir)

    # Add special tokens to the data that will require tokenizer resizing
    train_file = os.path.join(data_path, "train.jsonl")
    with open(train_file, "a") as f:
        for i in range(5):
            f.write(json.dumps({"text": f"Special token test <SPECIAL_{i}> in text"}) + "\n")

    output_dir = os.path.join(temp_dir, "clm_resized_tokenizer")

    config = LLMTrainingParams(
        model=TINY_LLM_MODEL,
        data_path=data_path,
        train_split="train",
        trainer="sft",
        project_name=output_dir,
        text_column="text",
        # PEFT settings
        peft=True,
        merge_adapter=True,
        lora_r=4,
        lora_alpha=8,
        # Training settings
        epochs=1,  # Must be integer
        batch_size=2,
        lr=1e-4,
        logging_steps=1,
        save_total_limit=1,
        log="none",
        max_samples=5,
    )

    # Train the model
    train(config)

    # Load the merged model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # Check that model and tokenizer sizes match
    model_vocab_size = model.get_input_embeddings().num_embeddings
    tokenizer_vocab_size = len(tokenizer)

    assert (
        model_vocab_size == tokenizer_vocab_size
    ), f"Model vocab size ({model_vocab_size}) should match tokenizer vocab size ({tokenizer_vocab_size})"

    # Test inference with the resized model
    inputs = tokenizer("Test with special tokens", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5)
    assert outputs is not None, "Model should generate output with resized tokenizer"


if __name__ == "__main__":
    # Run tests manually
    import sys

    with tempfile.TemporaryDirectory() as temp_dir:
        print("Testing CLM SFT with PEFT merge...")
        test_clm_sft_peft_merge(temp_dir)
        print("✓ CLM SFT PEFT merge test passed")

        print("\nTesting CLM SFT with PEFT no merge...")
        test_clm_sft_peft_no_merge(temp_dir)
        print("✓ CLM SFT PEFT no merge test passed")

        print("\nTesting Seq2Seq with PEFT merge...")
        test_seq2seq_peft_merge(temp_dir)
        print("✓ Seq2Seq PEFT merge test passed")

        print("\nTesting tokenizer resizing with merge...")
        test_tokenizer_resizing_with_merge(temp_dir)
        print("✓ Tokenizer resizing with merge test passed")

        print("\n✅ All PEFT merge tests passed!")
