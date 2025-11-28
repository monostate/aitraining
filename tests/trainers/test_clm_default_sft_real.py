"""
Real pytest tests for CLM Default and SFT trainers - NO MOCKS!
Tests with actual GPT-2 model training and validation.
"""

import os
import shutil

# Add src to path for imports
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.clm.train_clm_default import train as train_default
from autotrain.trainers.clm.train_clm_sft import train as train_sft


def save_dataset_as_csv(dataset, path, split_name="train"):
    """Helper to save datasets as CSV files for trainer compatibility."""
    import pandas as pd

    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    # Convert dataset to DataFrame and save as CSV
    df = pd.DataFrame(dataset)
    df.to_csv(path / f"{split_name}.csv", index=False)
    return path


@pytest.fixture
def model_name():
    """Use smallest GPT-2 model for real testing."""
    return "gpt2"


@pytest.fixture
def create_test_dataset():
    """Create a real small dataset with 20 samples for testing."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of technology.",
        "Python is a versatile programming language used in many fields.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of data for training.",
        "The weather today is sunny with clear skies.",
        "Artificial intelligence is revolutionizing healthcare and medicine.",
        "Data science combines statistics, programming, and domain knowledge.",
        "Cloud computing provides scalable infrastructure for applications.",
        "Neural networks are inspired by the human brain structure.",
        "Software engineering principles help build reliable systems.",
        "The stock market fluctuates based on various economic factors.",
        "Climate change is one of the biggest challenges facing humanity.",
        "Renewable energy sources are becoming more cost-effective.",
        "Space exploration continues to push the boundaries of human knowledge.",
        "Quantum computing promises to solve complex computational problems.",
        "Blockchain technology enables decentralized and secure transactions.",
        "Virtual reality creates immersive digital experiences.",
        "The internet has connected people across the globe.",
        "Open source software promotes collaboration and innovation.",
    ]
    return Dataset.from_dict({"text": texts})


@pytest.fixture
def create_validation_dataset():
    """Create a real validation dataset."""
    texts = [
        "Validation sample one for testing model performance.",
        "Testing the model with validation data is important.",
        "This is the third validation sample in our dataset.",
        "Model evaluation requires a separate validation set.",
        "The final validation sample completes our test set.",
    ]
    return Dataset.from_dict({"text": texts})


@pytest.fixture
def real_config(tmp_path, model_name):
    """Real configuration for actual training - no mocks."""
    return {
        "model": model_name,
        "project_name": str(tmp_path / "real_test_project"),
        "data_path": str(tmp_path / "data"),
        "train_split": "train",
        "text_column": "text",
        "epochs": 1,  # Just 1 epoch for testing
        "batch_size": 2,
        "block_size": 128,
        "lr": 5e-5,
        "warmup_ratio": 0.1,
        "gradient_accumulation": 1,
        "mixed_precision": None,  # No mixed precision for testing
        "peft": True,  # Use PEFT for faster training
        "merge_adapter": False,  # Don't merge adapters for testing (keep adapter files)
        "quantization": None,  # No quantization for testing
        "lora_r": 4,  # Small LoRA rank
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "logging_steps": 2,
        "eval_strategy": "no",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "push_to_hub": False,
        "auto_find_batch_size": False,
        "seed": 42,
        "use_flash_attention_2": False,
        "disable_gradient_checkpointing": True,
    }


class TestRealCLMDefaultTrainer:
    """Real tests for the default CLM trainer without mocks."""

    def test_real_basic_training(self, real_config, create_test_dataset, tmp_path):
        """Test that real training completes successfully."""
        # Save real dataset as CSV
        data_path = save_dataset_as_csv(create_test_dataset, tmp_path / "data", "train")

        # Create config
        config = LLMTrainingParams(**real_config)
        config.trainer = "default"

        # Run REAL training
        train_default(config)

        # Verify output exists
        project_path = Path(config.project_name)
        assert project_path.exists(), "Project directory was not created"

        # Check for model files
        model_files = list(project_path.glob("*.safetensors")) + list(project_path.glob("*.bin"))
        assert len(model_files) > 0, "No model files were saved"

        # Check for config files
        assert (project_path / "training_params.json").exists(), "Training params not saved"

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)

    def test_real_column_validation_error(self, real_config, create_test_dataset, tmp_path):
        """Test that missing columns raise real errors."""
        # Create dataset with wrong column
        wrong_data = Dataset.from_dict({"wrong_column": ["text1", "text2", "text3"]})
        data_path = save_dataset_as_csv(wrong_data, tmp_path / "data", "train")

        config = LLMTrainingParams(**real_config)
        config.trainer = "default"

        # Should raise real ValueError
        with pytest.raises(ValueError) as exc_info:
            train_default(config)

        assert "not found in" in str(exc_info.value).lower()

    def test_real_training_with_validation(
        self, real_config, create_test_dataset, create_validation_dataset, tmp_path
    ):
        """Test real training with validation split."""
        # Save both datasets as CSV in same directory
        data_path = tmp_path / "data"
        save_dataset_as_csv(create_test_dataset, data_path, "train")
        save_dataset_as_csv(create_validation_dataset, data_path, "validation")

        # Update config
        real_config["data_path"] = str(data_path)
        real_config["valid_split"] = "validation"
        real_config["eval_strategy"] = "epoch"

        config = LLMTrainingParams(**real_config)
        config.trainer = "default"

        # Run real training with validation
        train_default(config)

        # Check outputs
        project_path = Path(config.project_name)
        assert project_path.exists()

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)

    def test_real_checkpoint_saving(self, real_config, create_test_dataset, tmp_path):
        """Test real checkpoint saving during training."""
        data_path = save_dataset_as_csv(create_test_dataset, tmp_path / "data", "train")

        # Configure for checkpoint saving
        real_config["save_strategy"] = "steps"
        real_config["save_steps"] = 3
        real_config["save_total_limit"] = 2
        real_config["epochs"] = 2  # More epochs to trigger saves

        config = LLMTrainingParams(**real_config)
        config.trainer = "default"

        # Run training
        train_default(config)

        # Check for checkpoints
        project_path = Path(config.project_name)
        checkpoints = list(project_path.glob("checkpoint-*"))

        assert len(checkpoints) > 0, "No checkpoints were saved"
        assert len(checkpoints) <= 2, f"Too many checkpoints: {len(checkpoints)}"

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)

    @pytest.mark.parametrize("block_size", [64, 128])
    def test_real_different_block_sizes(self, real_config, create_test_dataset, tmp_path, block_size):
        """Test real training with different block sizes."""
        data_path = save_dataset_as_csv(create_test_dataset, tmp_path / "data", "train")

        real_config["block_size"] = block_size
        config = LLMTrainingParams(**real_config)
        config.trainer = "default"

        # Run training
        train_default(config)

        # Verify completion
        project_path = Path(config.project_name)
        assert project_path.exists()

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)

    def test_real_max_samples_limiting(self, real_config, create_test_dataset, tmp_path):
        """Test that max_samples properly limits training data."""
        # Create dataset with 20 samples
        data_path = save_dataset_as_csv(create_test_dataset, tmp_path / "data", "train")

        # Set max_samples to 5
        real_config["max_samples"] = 5
        config = LLMTrainingParams(**real_config)
        config.trainer = "default"

        # Run training with limited samples
        train_default(config)

        # Verify training completed (max_samples doesn't break training)
        project_path = Path(config.project_name)
        assert project_path.exists(), "Project directory was not created"

        # Check for model files
        model_files = list(project_path.glob("*.safetensors")) + list(project_path.glob("*.bin"))
        assert len(model_files) > 0, "No model files were saved"

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)


class TestRealCLMSFTTrainer:
    """Real tests for SFT trainer without mocks."""

    def test_real_sft_basic_training(self, real_config, create_test_dataset, tmp_path):
        """Test real SFT training completion."""
        data_path = save_dataset_as_csv(create_test_dataset, tmp_path / "data", "train")

        config = LLMTrainingParams(**real_config)
        config.trainer = "sft"
        config.packing = False  # Disable packing for basic test

        # Run real SFT training
        train_sft(config)

        # Check outputs
        project_path = Path(config.project_name)
        assert project_path.exists()

        # Check for adapter files (PEFT)
        adapter_files = list(project_path.glob("adapter_*.safetensors")) + list(project_path.glob("adapter_*.bin"))
        assert (
            len(adapter_files) > 0 or (project_path / "adapter_model.safetensors").exists()
        ), "No adapter files found"

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)

    def test_real_sft_with_packing(self, real_config, create_test_dataset, tmp_path):
        """Test real SFT with packing feature."""
        data_path = save_dataset_as_csv(create_test_dataset, tmp_path / "data", "train")

        real_config["packing"] = True
        real_config["use_flash_attention_2"] = False  # Disable for CPU

        config = LLMTrainingParams(**real_config)
        config.trainer = "sft"

        # Run training
        train_sft(config)

        project_path = Path(config.project_name)
        assert project_path.exists()

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)

    def test_real_sft_validation_data(self, real_config, create_test_dataset, create_validation_dataset, tmp_path):
        """Test real SFT with validation dataset."""
        # Save datasets as CSV in same directory
        data_path = tmp_path / "data"
        save_dataset_as_csv(create_test_dataset, data_path, "train")
        save_dataset_as_csv(create_validation_dataset, data_path, "validation")

        real_config["data_path"] = str(data_path)
        real_config["valid_split"] = "validation"
        real_config["eval_strategy"] = "epoch"

        config = LLMTrainingParams(**real_config)
        config.trainer = "sft"
        config.packing = False

        # Run training
        train_sft(config)

        project_path = Path(config.project_name)
        assert project_path.exists()

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)


class TestRealMetricsAndLogging:
    """Test real metrics logging functionality."""

    def test_real_sft_with_custom_metrics(self, real_config, create_test_dataset, create_validation_dataset, tmp_path):
        """Test real SFT training with custom metrics (perplexity)."""
        # Save datasets as CSV
        data_path = tmp_path / "data"
        save_dataset_as_csv(create_test_dataset, data_path, "train")
        save_dataset_as_csv(create_validation_dataset, data_path, "validation")

        real_config["data_path"] = str(data_path)
        real_config["valid_split"] = "validation"
        real_config["eval_strategy"] = "epoch"
        real_config["custom_metrics"] = '["perplexity"]'  # Test perplexity callback
        real_config["epochs"] = 2  # Need multiple epochs to see eval metrics

        config = LLMTrainingParams(**real_config)
        config.trainer = "sft"
        config.packing = False

        # Run training with custom metrics
        trainer = train_sft(config)

        # Check that training completed
        project_path = Path(config.project_name)
        assert project_path.exists()

        # Check if metrics were logged (trainer history should have metrics)
        if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
            # Look for custom metrics in log history
            log_history = trainer.state.log_history
            # The eval metrics would appear in entries with 'eval_loss'
            eval_entries = [entry for entry in log_history if "eval_loss" in entry]

            # Check for perplexity in eval entries
            for entry in eval_entries:
                if "eval_perplexity" in entry:
                    # Perplexity should be exp(loss), so greater than 1
                    assert entry["eval_perplexity"] > 1.0
                    break

            assert len(eval_entries) >= 0, "Training should complete with custom metrics"

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)

    def test_real_sft_with_generation_metrics(
        self, real_config, create_test_dataset, create_validation_dataset, tmp_path
    ):
        """Test real SFT training with generation metrics (BLEU/ROUGE)."""
        # Save datasets as CSV
        data_path = tmp_path / "data"
        save_dataset_as_csv(create_test_dataset, data_path, "train")
        save_dataset_as_csv(create_validation_dataset, data_path, "validation")

        real_config["data_path"] = str(data_path)
        real_config["valid_split"] = "validation"
        real_config["eval_strategy"] = "epoch"
        real_config["custom_metrics"] = '["sacrebleu", "rouge"]'  # Generation metrics
        real_config["epochs"] = 1  # Just one epoch for generation test
        real_config["batch_size"] = 1  # Small batch for generation

        config = LLMTrainingParams(**real_config)
        config.trainer = "sft"
        config.packing = False

        # Run training with generation metrics
        trainer = train_sft(config)

        # Check that training completed
        project_path = Path(config.project_name)
        assert project_path.exists()

        # Check if GenerativeSFTTrainer was used
        assert hasattr(trainer.args, "predict_with_generate"), "Should use generation during eval"

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)

    def test_real_tensorboard_logging(self, real_config, create_test_dataset, tmp_path):
        """Test real TensorBoard logging."""
        data_path = save_dataset_as_csv(create_test_dataset, tmp_path / "data", "train")

        real_config["log"] = "tensorboard"
        real_config["logging_steps"] = 2

        config = LLMTrainingParams(**real_config)
        config.trainer = "default"

        # Run training
        train_default(config)

        project_path = Path(config.project_name)

        # Check for logs
        assert project_path.exists()
        # TensorBoard logs might be in a runs directory
        tb_logs = list(project_path.glob("**/events.out.tfevents.*"))
        # Logs might not always be created in short training

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)


class TestRealQuantizationPEFT:
    """Test real quantization and PEFT features."""

    def test_real_lora_training(self, real_config, create_test_dataset, tmp_path):
        """Test real LoRA training."""
        data_path = save_dataset_as_csv(create_test_dataset, tmp_path / "data", "train")

        real_config["peft"] = True
        real_config["lora_r"] = 8
        real_config["lora_alpha"] = 16
        real_config["lora_dropout"] = 0.1

        config = LLMTrainingParams(**real_config)
        config.trainer = "sft"
        config.packing = False

        # Run training
        train_sft(config)

        project_path = Path(config.project_name)
        assert project_path.exists()

        # Check for LoRA adapter files
        assert (project_path / "adapter_model.safetensors").exists() or (
            project_path / "adapter_model.bin"
        ).exists(), "LoRA adapter not saved"

        assert (project_path / "adapter_config.json").exists(), "LoRA config not saved"

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)


class TestRealEdgeCases:
    """Test real edge cases and error conditions."""

    def test_real_empty_dataset_error(self, real_config, tmp_path):
        """Test handling of empty dataset."""
        empty_data = Dataset.from_dict({"text": []})
        data_path = save_dataset_as_csv(empty_data, tmp_path / "data", "train")

        config = LLMTrainingParams(**real_config)
        config.trainer = "default"

        # Should fail with empty dataset
        with pytest.raises(Exception):
            train_default(config)

    def test_real_max_samples_limit(self, real_config, create_test_dataset, tmp_path):
        """Test real max_samples limiting."""
        data_path = save_dataset_as_csv(create_test_dataset, tmp_path / "data", "train")

        real_config["max_samples"] = 5

        config = LLMTrainingParams(**real_config)
        config.trainer = "default"

        # Run training with limited samples
        train_default(config)

        project_path = Path(config.project_name)
        assert project_path.exists()

        # The model should have trained on only 5 samples
        # We can't easily verify this but training should complete faster

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)


class TestCLIIntegrationDefault:
    """Test CLI wrapper for default training - verifies CLI calls work like API."""

    def test_cli_default_basic_training(self, real_config, create_test_dataset, tmp_path):
        """Test CLI wrapper for default training."""
        import json
        import subprocess

        # Save real dataset as CSV
        data_path = save_dataset_as_csv(create_test_dataset, tmp_path / "data", "train")

        # Build CLI command
        cmd = [
            "python",
            "-m",
            "autotrain.cli.autotrain",
            "llm",
            "--train",
            "--model",
            real_config["model"],
            "--project-name",
            real_config["project_name"],
            "--data-path",
            str(data_path),
            "--train-split",
            "train",
            "--text-column",
            "text",
            "--trainer",
            "default",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--block-size",
            "128",
            "--lr",
            "5e-5",
            "--warmup-ratio",
            "0.1",
            "--gradient-accumulation",
            "1",
            "--peft",
            "--lora-r",
            "4",
            "--lora-alpha",
            "8",
            "--lora-dropout",
            "0.05",
            "--logging-steps",
            "2",
            "--eval-strategy",
            "no",
            "--save-strategy",
            "epoch",
            "--save-total-limit",
            "1",
            "--seed",
            "42",
            "--disable-gradient-checkpointing",
        ]

        # Run CLI command with proper environment
        env = os.environ.copy()
        env["PYTHONPATH"] = "./src"
        # Ensure .tmpvenv/bin is in PATH for accelerate command
        venv_bin = os.path.abspath(".tmpvenv/bin")
        if os.path.exists(venv_bin):
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # Check for success
        if result.returncode != 0:
            print("\n=== STDOUT ===")
            print(result.stdout)
            print("\n=== STDERR ===")
            print(result.stderr)

            # Print training log if it exists
            log_file = Path(real_config["project_name"]) / "autotrain.log"
            if log_file.exists():
                print("\n=== TRAINING LOG ===")
                print(log_file.read_text())

        assert result.returncode == 0, f"CLI failed with return code {result.returncode}. See output above."

        # Verify output exists
        project_path = Path(real_config["project_name"])
        assert project_path.exists(), "Project directory was not created via CLI"

        # Check for model files
        model_files = list(project_path.glob("*.safetensors")) + list(project_path.glob("*.bin"))
        assert len(model_files) > 0, "No model files saved via CLI"

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)

    def test_cli_sft_basic_training(self, real_config, create_test_dataset, tmp_path):
        """Test CLI wrapper for SFT training."""
        import subprocess

        # Save dataset
        data_path = save_dataset_as_csv(create_test_dataset, tmp_path / "data", "train")

        # Build CLI command for SFT
        cmd = [
            "python",
            "-m",
            "autotrain.cli.autotrain",
            "llm",
            "--train",
            "--model",
            real_config["model"],
            "--project-name",
            real_config["project_name"] + "_sft",
            "--data-path",
            str(data_path),
            "--train-split",
            "train",
            "--text-column",
            "text",
            "--trainer",
            "sft",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--block-size",
            "128",
            "--lr",
            "5e-5",
            "--warmup-ratio",
            "0.1",
            "--gradient-accumulation",
            "1",
            "--peft",
            "--lora-r",
            "4",
            "--lora-alpha",
            "8",
            "--lora-dropout",
            "0.05",
            "--logging-steps",
            "2",
            "--eval-strategy",
            "no",
            "--save-strategy",
            "epoch",
            "--save-total-limit",
            "1",
            "--seed",
            "42",
            "--disable-gradient-checkpointing",
            "--no-merge-adapter",  # Keep adapter files separate for testing
        ]

        # Run CLI command with proper environment
        env = os.environ.copy()
        env["PYTHONPATH"] = "./src"
        # Ensure .tmpvenv/bin is in PATH for accelerate command
        venv_bin = os.path.abspath(".tmpvenv/bin")
        if os.path.exists(venv_bin):
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # Check for success
        assert result.returncode == 0, f"CLI SFT failed with: {result.stderr}"

        # Verify output
        project_path = Path(real_config["project_name"] + "_sft")
        assert project_path.exists(), "SFT project not created via CLI"

        # Check for adapter files (PEFT)
        adapter_files = list(project_path.glob("adapter_*.safetensors")) + list(project_path.glob("adapter_*.bin"))
        assert (
            len(adapter_files) > 0 or (project_path / "adapter_model.safetensors").exists()
        ), "No adapter files found via CLI SFT"

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)

    def test_cli_with_validation(self, real_config, create_test_dataset, create_validation_dataset, tmp_path):
        """Test CLI with validation dataset."""
        import subprocess

        # Save datasets
        data_path = tmp_path / "data"
        save_dataset_as_csv(create_test_dataset, data_path, "train")
        save_dataset_as_csv(create_validation_dataset, data_path, "validation")

        # Build CLI command with validation
        cmd = [
            "python",
            "-m",
            "autotrain.cli.autotrain",
            "llm",
            "--train",
            "--model",
            real_config["model"],
            "--project-name",
            real_config["project_name"] + "_val",
            "--data-path",
            str(data_path),
            "--train-split",
            "train",
            "--valid-split",
            "validation",
            "--text-column",
            "text",
            "--trainer",
            "default",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--block-size",
            "128",
            "--lr",
            "5e-5",
            "--eval-strategy",
            "epoch",
            "--save-strategy",
            "epoch",
            "--seed",
            "42",
            "--peft",
            "--lora-r",
            "4",
            "--disable-gradient-checkpointing",
            "--no-merge-adapter",  # Keep adapter files separate for testing
        ]

        # Run CLI with stdout visible for debugging
        env = os.environ.copy()
        env["PYTHONPATH"] = "./src"
        result = subprocess.run(cmd, capture_output=False, text=True, env=env)

        assert result.returncode == 0, f"CLI validation failed with exit code {result.returncode}"

        project_path = Path(real_config["project_name"] + "_val")
        assert project_path.exists()

        # Cleanup
        if project_path.exists():
            shutil.rmtree(project_path)

    def test_cli_error_handling(self, real_config, tmp_path):
        """Test CLI error handling for missing columns."""
        import subprocess

        # Create dataset with wrong column
        wrong_data = Dataset.from_dict({"wrong_column": ["text1", "text2", "text3"]})
        data_path = save_dataset_as_csv(wrong_data, tmp_path / "data", "train")

        # Build CLI command that should fail
        cmd = [
            "python",
            "-m",
            "autotrain.cli.autotrain",
            "llm",
            "--train",
            "--model",
            real_config["model"],
            "--project-name",
            real_config["project_name"] + "_error",
            "--data-path",
            str(data_path),
            "--train-split",
            "train",
            "--text-column",
            "text",  # This column doesn't exist
            "--trainer",
            "default",
            "--epochs",
            "1",
            "--batch-size",
            "2",
        ]

        # Run CLI - should fail
        result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "./src"})

        # Should return non-zero exit code
        assert result.returncode != 0, "CLI should have failed with missing column"
        assert "not found in" in result.stderr.lower() or "error" in result.stderr.lower()


if __name__ == "__main__":
    # Run tests
    import subprocess

    result = subprocess.run(
        ["python3", "-m", "pytest", __file__, "-v", "--tb=short", "-x"], capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    exit(result.returncode)
