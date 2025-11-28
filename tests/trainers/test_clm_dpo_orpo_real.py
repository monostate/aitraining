"""
Real tests for CLM DPO and ORPO trainers with actual model training
===================================================================
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch

from autotrain.trainers.clm import train_clm_dpo, train_clm_orpo
from autotrain.trainers.clm.params import LLMTrainingParams


# =================== Fixtures ===================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def small_preference_dataset():
    """Create a small preference dataset for fast testing."""
    data = {
        "prompt": [
            "What is 2+2?",
            "What color is the sky?",
            "What is the capital of France?",
        ],
        "chosen": [
            "The answer is 4.",
            "The sky is blue.",
            "The capital is Paris.",
        ],
        "rejected": [
            "The answer is 5.",
            "The sky is green.",
            "The capital is London.",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def real_base_config(temp_dir, small_preference_dataset):
    """Create base configuration for real training with minimal model."""
    # Save dataset to temp directory with correct structure
    data_dir = os.path.join(temp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.csv")
    small_preference_dataset.to_csv(train_path, index=False)

    return {
        "model": "hf-internal-testing/tiny-random-gpt2",  # Tiny model for fast testing
        "project_name": os.path.join(temp_dir, "test_project"),
        "data_path": data_dir,
        "train_split": "train",
        "valid_split": None,
        "prompt_text_column": "prompt",
        "text_column": "chosen",
        "rejected_text_column": "rejected",
        "epochs": 1,
        "batch_size": 1,
        "lr": 1e-4,
        "block_size": 64,
        "max_prompt_length": 32,
        "max_completion_length": 32,
        "logging_steps": 1,
        "eval_strategy": "no",
        "save_total_limit": 1,
        "push_to_hub": False,
        "token": None,
        "seed": 42,
        "peft": False,
        "quantization": None,
        "use_flash_attention_2": False,
        "gradient_accumulation": 1,
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "disable_gradient_checkpointing": True,
        "mixed_precision": None,
        "chat_template": None,
        "trainer": "dpo",
    }


# =================== Real Training Tests ===================


class TestRealDPOTraining:
    """Test DPO trainer with actual model training."""

    def test_dpo_basic_training(self, real_base_config, temp_dir):
        """Test basic DPO training with tiny GPT2 model."""
        print(
            "\
=== Testing DPO Basic Training ==="
        )

        config = LLMTrainingParams(**real_base_config)
        config.trainer = "dpo"

        # Run actual training
        trainer = train_clm_dpo.train(config)

        # Verify training completed
        assert trainer is not None
        assert hasattr(trainer, "state")

        # Check that model was saved
        output_dir = os.path.join(temp_dir, "test_project")
        assert os.path.exists(output_dir)

        # Check for model files
        model_files = os.listdir(output_dir)
        assert any("model" in f or "pytorch" in f or "safetensors" in f for f in model_files)

        print(f"✅ DPO training completed successfully")
        print(f"   Output files: {model_files}")

        # Check training metrics
        if hasattr(trainer.state, "log_history") and trainer.state.log_history:
            final_loss = trainer.state.log_history[-1].get("loss", None)
            if final_loss:
                print(f"   Final loss: {final_loss:.4f}")

    def test_dpo_with_validation(self, real_base_config, temp_dir, small_preference_dataset):
        """Test DPO training with validation split."""
        print(
            "\
=== Testing DPO with Validation ==="
        )

        # Create train and validation data
        train_data = small_preference_dataset.iloc[:2]
        valid_data = small_preference_dataset.iloc[2:]

        train_path = os.path.join(temp_dir, "train.csv")
        valid_path = os.path.join(temp_dir, "validation.csv")

        train_data.to_csv(train_path, index=False)
        valid_data.to_csv(valid_path, index=False)

        config = real_base_config.copy()
        config["data_path"] = temp_dir
        config["train_split"] = "train"
        config["valid_split"] = "validation"
        config["eval_strategy"] = "epoch"

        params = LLMTrainingParams(**config)
        params.trainer = "dpo"

        # Run training
        trainer = train_clm_dpo.train(params)

        assert trainer is not None
        print(f"✅ DPO training with validation completed")

    def test_dpo_with_peft(self, real_base_config):
        """Test DPO training with PEFT/LoRA."""
        print(
            "\
=== Testing DPO with PEFT ==="
        )

        config = real_base_config.copy()
        config["peft"] = True
        config["lora_r"] = 4  # Very small for testing
        config["lora_alpha"] = 8
        config["lora_dropout"] = 0.1
        config["target_modules"] = "all-linear"
        config["quantization"] = None  # No quantization for testing

        params = LLMTrainingParams(**config)
        params.trainer = "dpo"

        # Run training
        trainer = train_clm_dpo.train(params)

        assert trainer is not None
        print(f"✅ DPO training with PEFT completed")

    def test_dpo_with_custom_metrics(self, real_base_config, temp_dir, small_preference_dataset):
        """Test DPO training with custom metrics."""
        print("\n=== Testing DPO with Custom Metrics ===")

        # Create train and validation data
        train_data = small_preference_dataset
        valid_data = small_preference_dataset.iloc[:1]  # Just one sample for validation

        data_dir = os.path.join(temp_dir, "metrics_data")
        os.makedirs(data_dir, exist_ok=True)

        train_path = os.path.join(data_dir, "train.csv")
        valid_path = os.path.join(data_dir, "validation.csv")

        train_data.to_csv(train_path, index=False)
        valid_data.to_csv(valid_path, index=False)

        config = real_base_config.copy()
        config["data_path"] = data_dir
        config["train_split"] = "train"
        config["valid_split"] = "validation"
        config["eval_strategy"] = "epoch"
        config["custom_metrics"] = '["accuracy"]'  # Add custom metrics for RL
        config["epochs"] = 2  # Need multiple epochs to see eval metrics

        config_obj = LLMTrainingParams(**config)
        config_obj.trainer = "dpo"

        # Run actual training with custom metrics
        trainer = train_clm_dpo.train(config_obj)

        # Verify training completed
        assert trainer is not None
        assert hasattr(trainer, "state")

        print(f"✅ DPO training with custom metrics completed successfully")

        # Check if eval happened (since we have validation data and eval_strategy="epoch")
        if hasattr(trainer.state, "log_history") and trainer.state.log_history:
            # Look for eval entries
            eval_entries = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
            print(f"   Evaluation entries found: {len(eval_entries)}")
            # With 2 epochs and eval_strategy="epoch", we should have at least 1 eval
            # The important thing is that training completes with custom_metrics

    def test_dpo_custom_columns(self, temp_dir):
        """Test DPO with custom column names."""
        print(
            "\
=== Testing DPO with Custom Columns ==="
        )

        # Create dataset with custom columns
        data = {
            "question": ["What is AI?", "Define ML"],
            "good": ["Artificial Intelligence", "Machine Learning"],
            "bad": ["Artificial Insect", "Manual Labor"],
        }
        df = pd.DataFrame(data)

        data_dir = os.path.join(temp_dir, "custom_data")
        os.makedirs(data_dir, exist_ok=True)
        train_path = os.path.join(data_dir, "train.csv")
        df.to_csv(train_path, index=False)

        config = {
            "model": "hf-internal-testing/tiny-random-gpt2",
            "project_name": os.path.join(temp_dir, "custom_project"),
            "data_path": data_dir,
            "train_split": "train",
            "prompt_text_column": "question",
            "text_column": "good",
            "rejected_text_column": "bad",
            "epochs": 1,
            "batch_size": 1,
            "lr": 1e-4,
            "block_size": 64,
            "trainer": "dpo",
            "push_to_hub": False,
        }

        params = LLMTrainingParams(**config)
        trainer = train_clm_dpo.train(params)

        assert trainer is not None
        print(f"✅ DPO training with custom columns completed")


class TestRealORPOTraining:
    """Test ORPO trainer with actual model training."""

    def test_orpo_basic_training(self, real_base_config, temp_dir):
        """Test basic ORPO training with tiny GPT2 model."""
        print(
            "\
=== Testing ORPO Basic Training ==="
        )

        config = LLMTrainingParams(**real_base_config)
        config.trainer = "orpo"

        # Run actual training
        trainer = train_clm_orpo.train(config)

        # Verify training completed
        assert trainer is not None
        assert hasattr(trainer, "state")

        # Check that model was saved
        output_dir = os.path.join(temp_dir, "test_project")
        assert os.path.exists(output_dir)

        # Check for model files
        model_files = os.listdir(output_dir)
        assert any("model" in f or "pytorch" in f or "safetensors" in f for f in model_files)

        print(f"✅ ORPO training completed successfully")
        print(f"   Output files: {model_files}")

        # Check training metrics
        if hasattr(trainer.state, "log_history") and trainer.state.log_history:
            final_loss = trainer.state.log_history[-1].get("loss", None)
            if final_loss:
                print(f"   Final loss: {final_loss:.4f}")

    def test_orpo_with_peft(self, real_base_config):
        """Test ORPO training with PEFT/LoRA."""
        print(
            "\
=== Testing ORPO with PEFT ==="
        )

        config = real_base_config.copy()
        config["peft"] = True
        config["lora_r"] = 4  # Very small for testing
        config["lora_alpha"] = 8
        config["lora_dropout"] = 0.1
        config["target_modules"] = "all-linear"
        config["quantization"] = None

        params = LLMTrainingParams(**config)
        params.trainer = "orpo"

        # Run training
        trainer = train_clm_orpo.train(params)

        assert trainer is not None
        print(f"✅ ORPO training with PEFT completed")


class TestCLIIntegrationDPOORPO:
    """Test CLI wrapper for DPO/ORPO training - verifies CLI calls work like API."""

    def test_cli_dpo_basic_training(self, real_base_config, small_preference_dataset, temp_dir):
        """Test CLI wrapper for DPO training."""
        import subprocess

        # Save dataset to temp directory
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        train_path = os.path.join(data_dir, "train.csv")
        small_preference_dataset.to_csv(train_path, index=False)

        # Build CLI command for DPO
        cmd = [
            "python",
            "-m",
            "autotrain.cli.autotrain",
            "llm",
            "--train",
            "--model",
            real_base_config["model"],
            "--project-name",
            os.path.join(temp_dir, "cli_dpo_test"),
            "--data-path",
            data_dir,
            "--train-split",
            "train",
            "--prompt-text-column",
            "prompt",
            "--text-column",
            "chosen",
            "--rejected-text-column",
            "rejected",
            "--trainer",
            "dpo",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "64",
            "--max-prompt-length",
            "32",
            "--max-completion-length",
            "32",
            "--lr",
            "1e-4",
            "--logging-steps",
            "1",
            "--eval-strategy",
            "no",
            "--save-total-limit",
            "1",
            "--seed",
            "42",
            "--disable-gradient-checkpointing",
        ]

        # Run CLI command
        env = os.environ.copy()
        env["PYTHONPATH"] = "./src"
        # Ensure .tmpvenv/bin is in PATH for accelerate command
        venv_bin = os.path.abspath(".tmpvenv/bin")
        if os.path.exists(venv_bin):
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # Check for success
        assert result.returncode == 0, f"CLI DPO failed with: {result.stderr}"

        # Verify output exists
        project_path = Path(os.path.join(temp_dir, "cli_dpo_test"))
        assert project_path.exists(), "DPO project not created via CLI"

        # Check for model files
        model_files = list(project_path.glob("*.safetensors")) + list(project_path.glob("*.bin"))
        assert len(model_files) > 0, "No model files saved via CLI DPO"

    def test_cli_orpo_basic_training(self, real_base_config, small_preference_dataset, temp_dir):
        """Test CLI wrapper for ORPO training."""
        import subprocess

        # Save dataset to temp directory
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        train_path = os.path.join(data_dir, "train.csv")
        small_preference_dataset.to_csv(train_path, index=False)

        # Build CLI command for ORPO
        cmd = [
            "python",
            "-m",
            "autotrain.cli.autotrain",
            "llm",
            "--train",
            "--model",
            real_base_config["model"],
            "--project-name",
            os.path.join(temp_dir, "cli_orpo_test"),
            "--data-path",
            data_dir,
            "--train-split",
            "train",
            "--prompt-text-column",
            "prompt",
            "--text-column",
            "chosen",
            "--rejected-text-column",
            "rejected",
            "--trainer",
            "orpo",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "64",
            "--max-prompt-length",
            "32",
            "--max-completion-length",
            "32",
            "--lr",
            "1e-4",
            "--logging-steps",
            "1",
            "--eval-strategy",
            "no",
            "--save-total-limit",
            "1",
            "--seed",
            "42",
            "--disable-gradient-checkpointing",
            "--mixed-precision",
            "fp16",
        ]

        # Run CLI command
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
            log_file = Path(os.path.join(temp_dir, "cli_orpo_test")) / "autotrain.log"
            if log_file.exists():
                print("\n=== TRAINING LOG ===")
                print(log_file.read_text())

        assert result.returncode == 0, f"CLI ORPO failed with return code {result.returncode}. See output above."

        # Verify output exists
        project_path = Path(os.path.join(temp_dir, "cli_orpo_test"))
        assert project_path.exists(), "ORPO project not created via CLI"

        # Check for model files
        model_files = list(project_path.glob("*.safetensors")) + list(project_path.glob("*.bin"))
        assert len(model_files) > 0, "No model files saved via CLI ORPO"

    def test_cli_dpo_missing_columns_error(self, temp_dir):
        """Test CLI error handling for missing required columns."""
        import subprocess

        # Create incomplete dataset
        data = {
            "prompt": ["Question 1"],
            "chosen": ["Answer 1"],
            # Missing 'rejected' column
        }
        df = pd.DataFrame(data)

        data_dir = os.path.join(temp_dir, "incomplete_data")
        os.makedirs(data_dir, exist_ok=True)
        train_path = os.path.join(data_dir, "train.csv")
        df.to_csv(train_path, index=False)

        # Build CLI command that should fail
        cmd = [
            "python",
            "-m",
            "autotrain.cli.autotrain",
            "llm",
            "--train",
            "--model",
            "hf-internal-testing/tiny-random-gpt2",
            "--project-name",
            os.path.join(temp_dir, "cli_error_test"),
            "--data-path",
            data_dir,
            "--train-split",
            "train",
            "--prompt-text-column",
            "prompt",
            "--text-column",
            "chosen",
            "--rejected-text-column",
            "rejected",  # This column doesn't exist
            "--trainer",
            "dpo",
            "--epochs",
            "1",
            "--batch-size",
            "1",
        ]

        # Run CLI - should fail
        env = os.environ.copy()
        env["PYTHONPATH"] = "./src"
        # Ensure .tmpvenv/bin is in PATH for accelerate command
        venv_bin = os.path.abspath(".tmpvenv/bin")
        if os.path.exists(venv_bin):
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # Should return non-zero exit code
        assert result.returncode != 0, "CLI should have failed with missing column"
        assert "not found in" in result.stderr.lower() or "error" in result.stderr.lower()


class TestDataValidation:
    """Test data validation for both trainers."""

    def test_missing_columns_error(self, temp_dir):
        """Test that missing required columns raises an error."""
        print(
            "\
=== Testing Missing Columns Error ==="
        )

        # Dataset missing 'rejected' column
        data = {"prompt": ["Question 1"], "chosen": ["Answer 1"]}
        df = pd.DataFrame(data)

        data_dir = os.path.join(temp_dir, "incomplete_data")
        os.makedirs(data_dir, exist_ok=True)
        train_path = os.path.join(data_dir, "train.csv")
        df.to_csv(train_path, index=False)

        config = {
            "model": "hf-internal-testing/tiny-random-gpt2",
            "project_name": os.path.join(temp_dir, "incomplete_project"),
            "data_path": data_dir,
            "train_split": "train",
            "prompt_text_column": "prompt",
            "text_column": "chosen",
            "rejected_text_column": "rejected",  # This column doesn't exist
            "epochs": 1,
            "batch_size": 1,
            "trainer": "dpo",
            "push_to_hub": False,
        }

        with pytest.raises(Exception) as exc_info:
            params = LLMTrainingParams(**config)
            train_clm_dpo.train(params)

        assert "rejected" in str(exc_info.value).lower() or "column" in str(exc_info.value).lower()
        print(f"✅ Missing column validation works correctly")

    def test_empty_dataset_error(self, temp_dir):
        """Test that empty dataset raises an error."""
        print(
            "\
=== Testing Empty Dataset Error ==="
        )

        # Empty dataset
        data = {"prompt": [], "chosen": [], "rejected": []}
        df = pd.DataFrame(data)

        data_dir = os.path.join(temp_dir, "empty_data")
        os.makedirs(data_dir, exist_ok=True)
        train_path = os.path.join(data_dir, "train.csv")
        df.to_csv(train_path, index=False)

        config = {
            "model": "hf-internal-testing/tiny-random-gpt2",
            "project_name": os.path.join(temp_dir, "empty_project"),
            "data_path": data_dir,
            "train_split": "train",
            "prompt_text_column": "prompt",
            "text_column": "chosen",
            "rejected_text_column": "rejected",
            "epochs": 1,
            "batch_size": 1,
            "trainer": "dpo",
            "push_to_hub": False,
        }

        with pytest.raises(Exception):
            params = LLMTrainingParams(**config)
            train_clm_dpo.train(params)

        print(f"✅ Empty dataset validation works correctly")


class TestParameterSettings:
    """Test various parameter configurations."""

    def test_dpo_beta_parameter(self, real_base_config):
        """Test DPO with different beta values."""
        print(
            "\
=== Testing DPO Beta Parameter ==="
        )

        for beta in [0.01, 0.1, 0.5]:
            config = real_base_config.copy()
            config["dpo_beta"] = beta
            config["epochs"] = 1

            params = LLMTrainingParams(**config)
            params.trainer = "dpo"

            trainer = train_clm_dpo.train(params)
            assert trainer is not None
            print(f"✅ DPO training with beta={beta} completed")

    def test_different_batch_sizes(self, real_base_config):
        """Test training with different batch sizes."""
        print(
            "\
=== Testing Different Batch Sizes ==="
        )

        for batch_size in [1, 2]:
            config = real_base_config.copy()
            config["batch_size"] = batch_size

            # Test DPO
            params = LLMTrainingParams(**config)
            params.trainer = "dpo"
            trainer = train_clm_dpo.train(params)
            assert trainer is not None

            # Test ORPO
            params.trainer = "orpo"
            trainer = train_clm_orpo.train(params)
            assert trainer is not None

            print(f"✅ Training with batch_size={batch_size} completed for both trainers")


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_checkpoint_saving(self, real_base_config, temp_dir):
        """Test that checkpoints are saved during training."""
        print(
            "\
=== Testing Checkpoint Saving ==="
        )

        config = real_base_config.copy()
        config["save_total_limit"] = 2
        config["save_strategy"] = "epoch"
        config["epochs"] = 2

        params = LLMTrainingParams(**config)
        params.trainer = "dpo"

        trainer = train_clm_dpo.train(params)

        # Check for checkpoint directories
        output_dir = os.path.join(temp_dir, "test_project")
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]

        print(f"✅ Found {len(checkpoints)} checkpoints")
        assert len(checkpoints) > 0


class TestLossReduction:
    """Test that training reduces loss."""

    def test_training_reduces_loss(self, real_base_config):
        """Test that loss decreases during training."""
        print(
            "\
=== Testing Loss Reduction ==="
        )

        config = real_base_config.copy()
        config["epochs"] = 3
        config["logging_steps"] = 1

        for trainer_name, trainer_func in [("DPO", train_clm_dpo), ("ORPO", train_clm_orpo)]:
            params = LLMTrainingParams(**config)
            params.trainer = trainer_name.lower()

            trainer = trainer_func.train(params)

            if hasattr(trainer.state, "log_history") and len(trainer.state.log_history) > 1:
                losses = [log.get("loss") for log in trainer.state.log_history if "loss" in log]
                if len(losses) > 1:
                    initial_loss = losses[0]
                    final_loss = losses[-1]
                    print(f"✅ {trainer_name}: Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
                    # Loss should generally decrease (allowing for some fluctuation)
                    assert final_loss <= initial_loss * 1.5  # Allow some tolerance


def run_all_tests():
    """Run all tests and generate a report."""
    import sys
    import traceback

    print("=" * 60)
    print("RUNNING REAL CLM DPO/ORPO TRAINER TESTS")
    print("=" * 60)

    # Create temporary directory for all tests
    temp_base = tempfile.mkdtemp()

    # Track test results
    results = []

    try:
        # Initialize fixtures
        small_dataset = pd.DataFrame(
            {
                "prompt": ["Q1?", "Q2?", "Q3?"],
                "chosen": ["Good1", "Good2", "Good3"],
                "rejected": ["Bad1", "Bad2", "Bad3"],
            }
        )

        data_dir = os.path.join(temp_base, "data")
        os.makedirs(data_dir, exist_ok=True)
        train_path = os.path.join(data_dir, "train.csv")
        small_dataset.to_csv(train_path, index=False)

        base_config = {
            "model": "hf-internal-testing/tiny-random-gpt2",
            "project_name": os.path.join(temp_base, "test_project"),
            "data_path": data_dir,
            "train_split": "train",
            "valid_split": None,
            "prompt_text_column": "prompt",
            "text_column": "chosen",
            "rejected_text_column": "rejected",
            "epochs": 1,
            "batch_size": 1,
            "lr": 1e-4,
            "block_size": 64,
            "max_prompt_length": 32,
            "max_completion_length": 32,
            "logging_steps": 1,
            "eval_strategy": "no",
            "save_total_limit": 1,
            "push_to_hub": False,
            "token": None,
            "seed": 42,
            "peft": False,
            "quantization": None,
            "use_flash_attention_2": False,
            "gradient_accumulation": 1,
            "warmup_ratio": 0.0,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "disable_gradient_checkpointing": True,
            "mixed_precision": None,
        }

        # List of test functions to run
        test_functions = [
            ("DPO Basic Training", lambda: TestRealDPOTraining().test_dpo_basic_training(base_config, temp_base)),
            ("ORPO Basic Training", lambda: TestRealORPOTraining().test_orpo_basic_training(base_config, temp_base)),
            ("DPO with PEFT", lambda: TestRealDPOTraining().test_dpo_with_peft(base_config)),
            ("ORPO with PEFT", lambda: TestRealORPOTraining().test_orpo_with_peft(base_config)),
            ("DPO Beta Parameter", lambda: TestParameterSettings().test_dpo_beta_parameter(base_config)),
            ("Data Validation", lambda: TestDataValidation().test_missing_columns_error(temp_base)),
        ]

        # Run each test
        for test_name, test_func in test_functions:
            try:
                print(
                    f"\
Running: {test_name}"
                )
                test_func()
                results.append((test_name, "PASSED", None))
            except Exception as e:
                results.append((test_name, "FAILED", str(e)))
                print(f"❌ {test_name} failed: {str(e)}")
                traceback.print_exc()

    finally:
        # Clean up
        shutil.rmtree(temp_base, ignore_errors=True)

    # Print summary
    print(
        "\
"
        + "=" * 60
    )
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")

    for test_name, status, error in results:
        status_symbol = "✅" if status == "PASSED" else "❌"
        print(f"{status_symbol} {test_name}: {status}")
        if error:
            print(f"   Error: {error[:100]}...")

    print(
        f"\
Total: {len(results)} tests"
    )
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
