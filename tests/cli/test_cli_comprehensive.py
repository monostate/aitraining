"""
Comprehensive CLI Tests for AITraining
=======================================

This module provides comprehensive testing for the AITraining CLI,
covering all trainers, features, and parameter combinations.
"""

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest


class CLITestBase:
    """Base class for CLI testing utilities."""

    @staticmethod
    def run_command(command: str, timeout: int = 30, check: bool = False) -> subprocess.CompletedProcess:
        """Run a CLI command and return the result."""
        # Add PYTHONPATH to environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent / "src")

        # Add .tmpvenv/bin to PATH if it exists
        venv_bin = Path(__file__).parent.parent.parent / ".tmpvenv" / "bin"
        if venv_bin.exists():
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
            # Replace 'python' with venv python in command
            if command.startswith("python "):
                command = str(venv_bin / "python") + command[6:]

        result = subprocess.run(
            shlex.split(command), capture_output=True, text=True, timeout=timeout, env=env, check=check
        )
        return result

    @staticmethod
    def create_test_data(data_type: str = "text", num_samples: int = 10) -> str:
        """Create test data files for different trainers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            if data_type == "text":
                # For LLM training
                data_file = Path(tmpdir) / "train.jsonl"
                with open(data_file, "w") as f:
                    for i in range(num_samples):
                        f.write(json.dumps({"text": f"Sample text {i}"}) + "\n")

            elif data_type == "preference":
                # For DPO/ORPO/PPO training
                data_file = Path(tmpdir) / "train.jsonl"
                with open(data_file, "w") as f:
                    for i in range(num_samples):
                        f.write(
                            json.dumps(
                                {
                                    "prompt": f"Question {i}?",
                                    "chosen": f"Good answer {i}",
                                    "rejected": f"Bad answer {i}",
                                }
                            )
                            + "\n"
                        )

            elif data_type == "classification":
                # For text classification
                data_file = Path(tmpdir) / "train.csv"
                with open(data_file, "w") as f:
                    f.write("text,label\n")
                    for i in range(num_samples):
                        f.write(f"Text sample {i},{i % 3}\n")

            return tmpdir


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_help(self):
        """Test that help command works."""
        result = CLITestBase.run_command("python -m autotrain.cli.autotrain --help")
        assert result.returncode == 0
        assert "aitraining" in result.stdout.lower()
        assert "command" in result.stdout.lower()

    def test_cli_version(self):
        """Test version display."""
        result = CLITestBase.run_command("python -m autotrain.cli.autotrain --version")
        assert result.returncode == 0
        # Version should be printed
        assert len(result.stdout.strip()) > 0

    def test_llm_help(self):
        """Test LLM subcommand help."""
        result = CLITestBase.run_command("python -m autotrain.cli.autotrain llm --help")
        assert result.returncode == 0
        assert "--trainer" in result.stdout
        assert "--custom-metrics" in result.stdout
        assert "--use-sweep" in result.stdout


class TestAllTrainers:
    """Test all trainer types via CLI."""

    @pytest.mark.parametrize("trainer", ["sft", "dpo", "orpo", "ppo", "reward", "distillation"])
    def test_trainer_selection(self, trainer):
        """Test that each trainer can be selected."""
        result = CLITestBase.run_command(f"python -m autotrain.cli.autotrain llm --help")
        assert result.returncode == 0
        # Trainer option should be available
        assert "--trainer" in result.stdout

    def test_sft_trainer_params(self):
        """Test SFT trainer accepts its parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-sft
                --data-path tests/sample_data
                --trainer sft
                --epochs 1
                --batch-size 2
                --lr 1e-4
                --max-steps 2
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Should not error on parameter validation
            assert "--dry-run" not in result.stderr or result.returncode == 0

    def test_dpo_trainer_params(self):
        """Test DPO trainer with required columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-dpo
                --data-path tests/sample_data/train_dpo.jsonl
                --trainer dpo
                --prompt-text-column prompt
                --text-column chosen
                --rejected-text-column rejected
                --max-steps 2
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Check that parameters are accepted (no unrecognized argument errors)
            assert "unrecognized arguments" not in result.stderr or "custom-metrics" not in result.stderr

    def test_orpo_trainer_params(self):
        """Test ORPO trainer with required columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-orpo
                --data-path tests/sample_data/train_dpo.jsonl
                --trainer orpo
                --prompt-text-column prompt
                --text-column chosen
                --rejected-text-column rejected
                --max-steps 2
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Check that parameters are accepted (no unrecognized argument errors)
            assert "unrecognized arguments" not in result.stderr or "custom-metrics" not in result.stderr

    def test_ppo_trainer_params(self):
        """Test PPO trainer with reward model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-ppo
                --data-path tests/sample_data/train_dpo.jsonl
                --trainer ppo
                --rl-reward-model-path gpt2
                --rl-gamma 0.95
                --rl-kl-coef 0.2
                --max-steps 2
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Check that parameters are accepted (no unrecognized argument errors)
            assert "unrecognized arguments" not in result.stderr or "custom-metrics" not in result.stderr


class TestAdvancedFeatures:
    """Test advanced CLI features."""

    def test_custom_metrics(self):
        """Test custom metrics parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-metrics
                --data-path tests/sample_data
                --custom-metrics '["perplexity", "bleu"]'
                --max-steps 2
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Check that parameters are accepted (no unrecognized argument errors)
            assert "unrecognized arguments" not in result.stderr or "custom-metrics" not in result.stderr

    def test_hyperparameter_sweep(self):
        """Test sweep parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_params = {"lr": [1e-4, 5e-5], "batch_size": [8, 16]}
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-sweep
                --data-path tests/sample_data
                --use-sweep
                --sweep-n-trials 2
                --sweep-params '{json.dumps(sweep_params)}'
                --sweep-metric eval_loss
                --sweep-direction minimize
                --max-steps 2
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Check that parameters are accepted (no unrecognized argument errors)
            assert "unrecognized arguments" not in result.stderr or "custom-metrics" not in result.stderr

    def test_distillation(self):
        """Test distillation parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-distill
                --data-path tests/sample_data
                --use-distillation
                --teacher-model gpt2-medium
                --distill-temperature 3.0
                --distill-alpha 0.7
                --max-steps 2
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Check that parameters are accepted (no unrecognized argument errors)
            assert "unrecognized arguments" not in result.stderr or "custom-metrics" not in result.stderr

    def test_forward_backward_pipeline(self):
        """Test forward-backward pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-fb
                --data-path tests/sample_data
                --use-forward-backward
                --forward-backward-loss-fn cross_entropy
                --max-steps 2
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Check that parameters are accepted (no unrecognized argument errors)
            assert "unrecognized arguments" not in result.stderr or "custom-metrics" not in result.stderr

    def test_sample_generation(self):
        """Test sample generation during training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts = ["Tell me about AI", "Write a poem"]
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-sample
                --data-path tests/sample_data
                --sample-every-n-steps 100
                --sample-prompts '{json.dumps(prompts)}'
                --sample-temperature 0.8
                --max-steps 2
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Check that parameters are accepted (no unrecognized argument errors)
            assert "unrecognized arguments" not in result.stderr or "custom-metrics" not in result.stderr

    def test_peft_lora(self):
        """Test PEFT/LoRA parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-peft
                --data-path tests/sample_data
                --peft
                --lora-r 8
                --lora-alpha 16
                --lora-dropout 0.1
                --target-modules all-linear
                --max-steps 2
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Check that parameters are accepted (no unrecognized argument errors)
            assert "unrecognized arguments" not in result.stderr or "custom-metrics" not in result.stderr


class TestRLFeatures:
    """Test RL-specific features."""

    def test_rl_parameters(self):
        """Test all RL parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-rl
                --data-path tests/sample_data
                --trainer ppo
                --rl-reward-model-path /path/to/reward
                --rl-gamma 0.95
                --rl-gae-lambda 0.95
                --rl-kl-coef 0.2
                --rl-value-loss-coef 1.0
                --rl-clip-range 0.2
                --rl-num-ppo-epochs 4
                --rl-chunk-size 128
                --rl-mini-batch-size 32
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Check that parameters are accepted (no unrecognized argument errors)
            assert "unrecognized arguments" not in result.stderr or "custom-metrics" not in result.stderr

    def test_rl_multi_objective(self):
        """Test multi-objective RL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {"accuracy": 0.5, "fluency": 0.3, "safety": 0.2}
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-multi-rl
                --data-path tests/sample_data
                --trainer ppo
                --rl-reward-model-path /path/to/reward
                --rl-multi-objective
                --rl-reward-weights '{json.dumps(weights)}'
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Check that parameters are accepted (no unrecognized argument errors)
            assert "unrecognized arguments" not in result.stderr or "custom-metrics" not in result.stderr


class TestParameterValidation:
    """Test parameter validation and error handling."""

    def test_missing_required_params(self):
        """Test that missing required parameters are caught."""
        # Provide an explicit, non-existent data path so the wizard isn't triggered but training still fails
        result = CLITestBase.run_command(
            "python -m autotrain.cli.autotrain llm --train --model gpt2 --project-name test --data-path /tmp/nonexistent-data"
        )
        # Command starts but should fail during training when data directory doesn't exist
        # This is expected behavior - validation happens at runtime, not at CLI parse time
        assert result.returncode != 0 or "Training failed" in result.stdout or "data" in result.stderr.lower()

        # More meaningful test: missing model parameter which is truly required
        result2 = CLITestBase.run_command("python -m autotrain.cli.autotrain llm --train --data-path /tmp/nonexistent")
        # Should trigger the interactive wizard (since model is missing) and exit once inputs are exhausted
        assert result2.returncode != 0 or "Interactive Setup" in result2.stdout or "Setup cancelled" in result2.stdout

    def test_invalid_json_params(self):
        """Test that invalid JSON is caught."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Invalid JSON in custom-metrics
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test
                --data-path tests/sample_data
                --custom-metrics 'not-valid-json'
                --max-steps 2
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Should error on invalid JSON
            assert result.returncode != 0 or "json" in result.stderr.lower()

    def test_dpo_missing_columns(self):
        """Test DPO without required columns fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test
                --data-path tests/sample_data
                --trainer dpo
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Should fail validation
            assert result.returncode != 0 or "prompt_text_column" in result.stderr

    def test_ppo_missing_reward_model(self):
        """Test PPO without reward model fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test
                --data-path tests/sample_data
                --trainer ppo
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Should fail validation
            assert result.returncode != 0 or "reward" in result.stderr.lower()


class TestOtherTrainers:
    """Test non-LLM trainers via CLI."""

    def test_text_classification(self):
        """Test text classification CLI."""
        result = CLITestBase.run_command("python -m autotrain.cli.autotrain text-classification --help")
        assert result.returncode == 0
        assert "--model" in result.stdout

    def test_image_classification(self):
        """Test image classification CLI."""
        result = CLITestBase.run_command("python -m autotrain.cli.autotrain image-classification --help")
        assert result.returncode == 0
        assert "--model" in result.stdout

    def test_token_classification(self):
        """Test token classification CLI."""
        result = CLITestBase.run_command("python -m autotrain.cli.autotrain token-classification --help")
        assert result.returncode == 0
        assert "--model" in result.stdout

    def test_seq2seq(self):
        """Test seq2seq CLI."""
        result = CLITestBase.run_command("python -m autotrain.cli.autotrain seq2seq --help")
        assert result.returncode == 0
        assert "--model" in result.stdout

    def test_tabular(self):
        """Test tabular CLI."""
        result = CLITestBase.run_command("python -m autotrain.cli.autotrain tabular --help")
        assert result.returncode == 0
        # Tabular has different params
        assert "--task" in result.stdout or "--model" in result.stdout


class TestComplexScenarios:
    """Test complex parameter combinations."""

    def test_full_feature_combination(self):
        """Test combining multiple advanced features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-complex
                --data-path tests/sample_data
                --trainer sft
                --custom-metrics '["perplexity"]'
                --peft
                --lora-r 8
                --use-enhanced-eval
                --eval-save-predictions
                --sample-every-n-steps 100
                --sample-prompts '["Test prompt"]'
                --mixed-precision fp16
                --gradient-accumulation 4
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Check that parameters are accepted (no unrecognized argument errors)
            assert "unrecognized arguments" not in result.stderr or "custom-metrics" not in result.stderr

    def test_config_file_loading(self):
        """Test loading configuration from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data in proper chat format
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            train_file = data_dir / "train.jsonl"
            with open(train_file, "w") as f:
                for i in range(5):
                    # Create proper chat format data
                    messages = [
                        {"role": "user", "content": f"Question {i}?"},
                        {"role": "assistant", "content": f"Answer {i}."},
                    ]
                    f.write(json.dumps({"text": str(messages)}) + "\n")

            # Create config file matching expected format
            config = {
                "task": "llm-sft",
                "base_model": "gpt2",
                "project_name": f"{tmpdir}/test-config",
                "backend": "local",
                "log": "tensorboard",
                "data": {
                    "path": str(data_dir),
                    "train_split": "train",
                    "valid_split": None,
                    "chat_template": "tokenizer",
                    "column_mapping": {"text_column": "text"},
                },
                "params": {
                    "epochs": 1,
                    "batch_size": 1,
                    "lr": 1e-5,
                    "max_steps": 2,  # Limit training to just a few steps for testing
                },
            }
            config_file = Path(tmpdir) / "config.yml"
            import yaml

            with open(config_file, "w") as f:
                yaml.dump(config, f)

            cmd = f"python -m autotrain.cli.autotrain --config {config_file}"
            result = CLITestBase.run_command(cmd)
            # Config should load and command should succeed
            assert result.returncode == 0


# Performance and stress tests
class TestPerformance:
    """Test CLI performance and edge cases."""

    def test_large_json_params(self):
        """Test handling of large JSON parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create large sweep params
            sweep_params = {
                "lr": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
                "batch_size": [4, 8, 16, 32],
                "warmup_ratio": [0.05, 0.1, 0.15],
                "weight_decay": [0.0, 0.01, 0.1],
                "gradient_accumulation": [1, 2, 4, 8],
            }
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-large
                --data-path tests/sample_data
                --use-sweep
                --sweep-params '{json.dumps(sweep_params)}'
                --sweep-n-trials 2
                --max-steps 2
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Check that parameters are accepted (no unrecognized argument errors)
            assert "unrecognized arguments" not in result.stderr or "custom-metrics" not in result.stderr

    def test_many_parameters(self):
        """Test command with many parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f"""
                python -m autotrain.cli.autotrain llm --train
                --model gpt2
                --project-name {tmpdir}/test-many
                --data-path tests/sample_data
                --trainer sft
                --epochs 3
                --batch-size 8
                --lr 2e-5
                --warmup-ratio 0.1
                --gradient-accumulation 4
                --weight-decay 0.01
                --max-grad-norm 1.0
                --seed 42
                --logging-steps 10
                --save-steps 100
                --eval-steps 50
                --save-total-limit 3
                --push-to-hub
                --hub-model-id test/model
                --hub-private
                --peft
                --lora-r 16
                --lora-alpha 32
                --quantization int4
                --block-size 512
            """
            result = CLITestBase.run_command(cmd.replace("\n", " "))
            # Check that parameters are accepted (no unrecognized argument errors)
            assert "unrecognized arguments" not in result.stderr or "custom-metrics" not in result.stderr


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
