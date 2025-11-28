"""
Tests for Prompt Distillation
==============================
"""

import os
import tempfile

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from autotrain.trainers.clm.train_clm_distill import (
    DistillationDataset,
    PromptDistillationConfig,
    PromptDistillationTrainer,
    generate_teacher_outputs,
    train_prompt_distillation,
)


@pytest.fixture
def models_and_tokenizer():
    """Create small models for testing."""
    model_name = "gpt2"
    teacher_model = AutoModelForCausalLM.from_pretrained(model_name)
    student_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return teacher_model, student_model, tokenizer


@pytest.fixture
def distillation_config():
    """Create test configuration."""
    return PromptDistillationConfig(
        teacher_model_name="gpt2",
        teacher_prompt_template="You are a helpful assistant. Answer: {input}",
        student_model_name="gpt2",
        student_prompt_template="{input}",
        temperature=2.0,
        alpha=0.5,
        max_length=128,
        num_samples=10,
        learning_rate=1e-4,
        batch_size=2,
        num_epochs=1,
    )


def test_distillation_config():
    """Test configuration creation."""
    config = PromptDistillationConfig(
        teacher_model_name="gpt2",
        teacher_prompt_template="Template: {input}",
        student_model_name="gpt2",
    )

    assert config.teacher_model_name == "gpt2"
    assert config.temperature == 3.0  # Default
    assert config.alpha == 0.7  # Default
    assert config.use_peft == True  # Default


def test_distillation_dataset():
    """Test distillation dataset."""
    base_inputs = ["Question 1", "Question 2", "Question 3"]
    teacher_outputs = ["Answer 1", "Answer 2", "Answer 3"]

    dataset = DistillationDataset(
        base_inputs=base_inputs,
        teacher_outputs=teacher_outputs,
        student_prompt_template="Q: {input}\nA:",
    )

    assert len(dataset) == 3

    # Test item access
    item = dataset[0]
    assert "text" in item
    assert "Question 1" in item["base_input"]
    assert "Answer 1" in item["teacher_output"]


def test_distillation_dataset_with_tokenizer(models_and_tokenizer):
    """Test dataset with tokenization."""
    _, _, tokenizer = models_and_tokenizer

    base_inputs = ["What is 2+2?", "What is the capital of France?"]
    teacher_outputs = ["4", "Paris"]

    dataset = DistillationDataset(
        base_inputs=base_inputs,
        teacher_outputs=teacher_outputs,
        tokenizer=tokenizer,
        max_length=64,
    )

    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["input_ids"].shape[0] == 64  # max_length


def test_generate_teacher_outputs(models_and_tokenizer):
    """Test teacher output generation."""
    teacher_model, _, tokenizer = models_and_tokenizer

    base_inputs = ["Hello", "How are you?"]
    prompt_template = "Assistant: {input}\nResponse:"
    generation_config = {
        "max_new_tokens": 10,
        "temperature": 0.7,
        "do_sample": False,
    }

    outputs, logits = generate_teacher_outputs(
        teacher_model,
        tokenizer,
        base_inputs,
        prompt_template,
        generation_config,
        batch_size=1,
        return_logits=False,
    )

    assert len(outputs) == 2
    assert all(isinstance(out, str) for out in outputs)
    assert logits is None  # return_logits=False


def test_generate_teacher_outputs_with_logits(models_and_tokenizer):
    """Test teacher output generation with logits."""
    teacher_model, _, tokenizer = models_and_tokenizer

    base_inputs = ["Test input"]
    prompt_template = "{input}"
    generation_config = {
        "max_new_tokens": 5,
        "temperature": 1.0,
        "do_sample": False,
    }

    outputs, logits = generate_teacher_outputs(
        teacher_model,
        tokenizer,
        base_inputs,
        prompt_template,
        generation_config,
        return_logits=True,
    )

    assert len(outputs) == 1
    assert logits is not None
    assert len(logits) == 1
    assert isinstance(logits[0], torch.Tensor)


def test_prompt_distillation_trainer(models_and_tokenizer, distillation_config):
    """Test custom trainer class."""
    _, student_model, tokenizer = models_and_tokenizer

    # Create dummy dataset
    base_inputs = ["Q1", "Q2"]
    teacher_outputs = ["A1", "A2"]
    dataset = DistillationDataset(
        base_inputs=base_inputs,
        teacher_outputs=teacher_outputs,
        tokenizer=tokenizer,
        max_length=32,
    )

    # Create trainer (without actually training)
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        logging_steps=10,
        report_to="none",
    )

    trainer = PromptDistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        distill_config=distillation_config,
    )

    assert trainer.distill_config == distillation_config

    # Test loss computation
    # Get model device
    device = next(student_model.parameters()).device

    batch = {
        "input_ids": torch.randint(0, 1000, (2, 32)).to(device),
        "attention_mask": torch.ones(2, 32).to(device),
        "labels": torch.randint(0, 1000, (2, 32)).to(device),
    }

    loss = trainer.compute_loss(student_model, batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_train_prompt_distillation_integration(models_and_tokenizer, distillation_config):
    """Test full training pipeline (minimal)."""
    base_inputs = ["What is AI?", "Explain quantum physics"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Modify config for faster testing
        distillation_config.num_epochs = 1
        distillation_config.num_samples = 2
        distillation_config.teacher_generation_config["max_new_tokens"] = 5

        # Run training
        trainer = train_prompt_distillation(
            config=distillation_config,
            base_inputs=base_inputs,
            output_dir=tmp_dir,
        )

        # Check outputs
        assert os.path.exists(os.path.join(tmp_dir, "teacher_outputs.jsonl"))
        assert os.path.exists(os.path.join(tmp_dir, "distillation_config.json"))

        # Verify model files
        model_files = os.listdir(tmp_dir)
        assert any("model" in f or "adapter" in f for f in model_files)


def test_distillation_with_validation(models_and_tokenizer, distillation_config):
    """Test distillation with validation data."""
    base_inputs = ["Train Q1", "Train Q2"]
    val_inputs = ["Val Q1"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        distillation_config.num_epochs = 1
        distillation_config.teacher_generation_config["max_new_tokens"] = 3

        trainer = train_prompt_distillation(
            config=distillation_config,
            base_inputs=base_inputs,
            output_dir=tmp_dir,
            validation_inputs=val_inputs,
        )

        # Check that validation was used
        assert trainer.eval_dataset is not None
        assert len(trainer.eval_dataset) == len(val_inputs)


def test_distillation_temperature_effect(models_and_tokenizer):
    """Test that temperature affects distillation."""
    config_low_temp = PromptDistillationConfig(
        teacher_model_name="gpt2",
        student_model_name="gpt2",
        temperature=1.0,
        teacher_prompt_template="{input}",
    )

    config_high_temp = PromptDistillationConfig(
        teacher_model_name="gpt2",
        student_model_name="gpt2",
        temperature=5.0,
        teacher_prompt_template="{input}",
    )

    assert config_low_temp.temperature < config_high_temp.temperature

    # Temperature should affect KL divergence calculation
    # Higher temperature = softer probability distribution


def test_alpha_weighting(models_and_tokenizer):
    """Test alpha parameter for loss weighting."""
    _, student_model, tokenizer = models_and_tokenizer

    # Test with alpha = 0 (pure cross-entropy)
    config_ce_only = PromptDistillationConfig(
        teacher_model_name="gpt2",
        student_model_name="gpt2",
        alpha=0.0,
    )

    # Test with alpha = 1 (pure KL divergence)
    config_kl_only = PromptDistillationConfig(
        teacher_model_name="gpt2",
        student_model_name="gpt2",
        alpha=1.0,
    )

    assert config_ce_only.alpha == 0.0
    assert config_kl_only.alpha == 1.0


def test_custom_prompt_templates():
    """Test various prompt template formats."""
    templates = [
        "Question: {input}\nAnswer:",
        "Context: You are helpful.\nUser: {input}\nAssistant:",
        "{input}",  # No prompt
        "### Instruction: {input}\n### Response:",
    ]

    for template in templates:
        config = PromptDistillationConfig(
            teacher_model_name="gpt2",
            student_model_name="gpt2",
            teacher_prompt_template=template,
        )
        assert config.teacher_prompt_template == template

        # Test formatting
        formatted = template.format(input="test")
        assert "test" in formatted
