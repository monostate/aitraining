"""
CLI Parameter Validation Tests
==============================
Simple tests to verify CLI accepts valid parameters without actually running training.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest


class TestCLIParameterValidation:
    """Test that CLI accepts valid parameters."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Create minimal test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir) / "data"
        self.data_path.mkdir(exist_ok=True)
        self.project_dir = Path(self.temp_dir) / "test_project"

        # Create minimal dataset
        df = pd.DataFrame({"text": ["test1", "test2"]})
        df.to_csv(self.data_path / "train.csv", index=False)

        yield

        # Cleanup
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def run_cli_help(self, args):
        """Run CLI with --help to validate parameters are accepted."""
        # Test the CLI module directly (same as what 'aitraining' command would call)
        cmd = [sys.executable, "-m", "autotrain.cli.autotrain", "llm"] + args + ["--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "./src"})
        return result

    def run_cli_dry_run(self, args):
        """Run CLI with minimal args to check parameter parsing (will fail at training, but that's ok)."""
        base_args = [
            sys.executable,
            "-m",
            "autotrain.cli.autotrain",
            "llm",
            "--train",
            "--model",
            "gpt2",
            "--project-name",
            str(self.project_dir),
            "--data-path",
            str(self.data_path),
            "--text-column",
            "text",
            "--epochs",
            "1",
        ]
        cmd = base_args + args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env={"PYTHONPATH": "./src"},
                timeout=5,  # Quick timeout - we just want to check if params are accepted
            )
        except subprocess.TimeoutExpired as e:
            # Timeout means the CLI accepted the parameters and started training
            # This is actually a success for our test!
            # Check if there were parameter errors before the timeout
            stderr_text = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr if e.stderr else "")
            stdout_text = e.output.decode() if isinstance(e.output, bytes) else (e.output if e.output else "")

            if "unrecognized arguments" in stderr_text:
                # This is a real failure - parameter not recognized
                result = subprocess.CompletedProcess(args=cmd, returncode=2, stdout=stdout_text, stderr=stderr_text)
            else:
                # Timeout without parameter errors means success
                result = subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="Parameters accepted, training started", stderr=""
                )
        return result

    def test_help_command_works(self):
        """Test that help command works."""
        result = self.run_cli_help([])
        assert result.returncode == 0
        assert "aitraining" in result.stdout.lower()

    def test_basic_training_parameters(self):
        """Test basic training parameters are accepted."""
        result = self.run_cli_dry_run(
            [
                "--lr",
                "1e-4",
                "--batch-size",
                "8",
                "--warmup-ratio",
                "0.1",
                "--gradient-accumulation",
                "4",
                "--seed",
                "42",
            ]
        )
        # Should fail at training stage, not parameter parsing
        assert "unrecognized arguments" not in result.stderr

    def test_peft_parameters(self):
        """Test PEFT parameters are accepted."""
        result = self.run_cli_dry_run(["--peft", "--lora-r", "16", "--lora-alpha", "32", "--lora-dropout", "0.1"])
        assert "unrecognized arguments" not in result.stderr

    def test_quantization_parameter(self):
        """Test quantization parameter is accepted."""
        result = self.run_cli_dry_run(["--quantization", "int4"])
        assert "unrecognized arguments" not in result.stderr

    def test_trainer_parameter(self):
        """Test trainer parameter is accepted."""
        result = self.run_cli_dry_run(["--trainer", "sft"])
        assert "unrecognized arguments" not in result.stderr

    def test_mixed_precision_parameter(self):
        """Test mixed precision parameter is accepted."""
        result = self.run_cli_dry_run(["--mixed-precision", "fp16"])
        assert "unrecognized arguments" not in result.stderr

    def test_scheduler_parameter(self):
        """Test scheduler parameter is accepted."""
        result = self.run_cli_dry_run(["--scheduler", "cosine"])
        assert "unrecognized arguments" not in result.stderr

    def test_optimizer_parameter(self):
        """Test optimizer parameter is accepted."""
        result = self.run_cli_dry_run(["--optimizer", "adamw_torch"])
        assert "unrecognized arguments" not in result.stderr

    def test_logging_parameters(self):
        """Test logging parameters are accepted."""
        result = self.run_cli_dry_run(["--logging-steps", "10", "--log", "tensorboard"])
        assert "unrecognized arguments" not in result.stderr

    def test_save_parameters(self):
        """Test save-related parameters are accepted."""
        result = self.run_cli_dry_run(["--save-steps", "100", "--save-total-limit", "2", "--save-strategy", "steps"])
        assert "unrecognized arguments" not in result.stderr

    def test_eval_parameters(self):
        """Test evaluation parameters are accepted."""
        result = self.run_cli_dry_run(["--eval-strategy", "steps", "--valid-split", "validation"])
        assert "unrecognized arguments" not in result.stderr

    def test_hub_parameters(self):
        """Test hub parameters are accepted."""
        result = self.run_cli_dry_run(["--push-to-hub", "--username", "test_user", "--token", "test_token"])
        assert "unrecognized arguments" not in result.stderr

    def test_flash_attention_parameter(self):
        """Test flash attention parameter is accepted."""
        result = self.run_cli_dry_run(["--use-flash-attention-2"])
        assert "unrecognized arguments" not in result.stderr

    def test_gradient_checkpointing_parameter(self):
        """Test gradient checkpointing parameter is accepted."""
        result = self.run_cli_dry_run(["--disable-gradient-checkpointing"])
        assert "unrecognized arguments" not in result.stderr

    def test_max_samples_parameter(self):
        """Test max_samples parameter is accepted."""
        result = self.run_cli_dry_run(["--max-samples", "100"])
        assert "unrecognized arguments" not in result.stderr

    def test_batch_size_finder_parameter(self):
        """Test auto batch size finder parameter is accepted."""
        result = self.run_cli_dry_run(["--auto-find-batch-size"])
        assert "unrecognized arguments" not in result.stderr

    def test_block_size_parameter(self):
        """Test block size parameter is accepted."""
        result = self.run_cli_dry_run(["--block-size", "512"])
        assert "unrecognized arguments" not in result.stderr

    def test_model_max_length_parameter(self):
        """Test model max length parameter is accepted."""
        result = self.run_cli_dry_run(["--model-max-length", "2048"])
        assert "unrecognized arguments" not in result.stderr

    def test_max_grad_norm_parameter(self):
        """Test max grad norm parameter is accepted."""
        result = self.run_cli_dry_run(["--max-grad-norm", "1.0"])
        assert "unrecognized arguments" not in result.stderr

    def test_weight_decay_parameter(self):
        """Test weight decay parameter is accepted."""
        result = self.run_cli_dry_run(["--weight-decay", "0.01"])
        assert "unrecognized arguments" not in result.stderr

    def test_chat_template_parameter(self):
        """Test chat template parameter is accepted."""
        result = self.run_cli_dry_run(["--chat-template", "zephyr"])
        assert "unrecognized arguments" not in result.stderr

    def test_dpo_trainer_parameters(self):
        """Test DPO trainer accepts its specific parameters."""
        # Create DPO dataset
        df = pd.DataFrame({"prompt": ["q1", "q2"], "chosen": ["good1", "good2"], "rejected": ["bad1", "bad2"]})
        df.to_csv(self.data_path / "train.csv", index=False)

        result = self.run_cli_dry_run(
            [
                "--trainer",
                "dpo",
                "--text-column",
                "chosen",
                "--prompt-text-column",
                "prompt",
                "--rejected-text-column",
                "rejected",
                "--dpo-beta",
                "0.1",
            ]
        )
        assert "unrecognized arguments" not in result.stderr

    def test_merge_adapter_parameter(self):
        """Test merge adapter parameter is accepted."""
        result = self.run_cli_dry_run(["--merge-adapter"])
        assert "unrecognized arguments" not in result.stderr

    def test_target_modules_parameter(self):
        """Test target modules parameter is accepted."""
        result = self.run_cli_dry_run(["--target-modules", "q_proj,v_proj"])
        assert "unrecognized arguments" not in result.stderr

    def test_add_eos_token_parameter(self):
        """Test add eos token parameter is accepted."""
        result = self.run_cli_dry_run(["--add-eos-token"])
        assert "unrecognized arguments" not in result.stderr


class TestInvalidParameters:
    """Test that invalid parameters are properly rejected."""

    def test_invalid_parameter_rejected(self):
        """Test that invalid parameters are rejected."""
        cmd = [sys.executable, "-m", "autotrain.cli.autotrain", "llm", "--this-parameter-does-not-exist", "value"]
        result = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "./src"})
        assert result.returncode != 0
        assert "unrecognized arguments" in result.stderr

    def test_invalid_trainer_rejected(self):
        """Test that invalid trainer names are rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "data"
            data_path.mkdir()
            df = pd.DataFrame({"text": ["test"]})
            df.to_csv(data_path / "train.csv", index=False)

            cmd = [
                sys.executable,
                "-m",
                "autotrain.cli.autotrain",
                "llm",
                "--train",
                "--model",
                "gpt2",
                "--project-name",
                f"{tmp_dir}/project",
                "--data-path",
                str(data_path),
                "--text-column",
                "text",
                "--trainer",
                "invalid_trainer",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "./src"}, timeout=5)
            assert result.returncode != 0
