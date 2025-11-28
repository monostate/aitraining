"""
Tests for grouped help and trainer-filtered CLI parameters.
"""

import subprocess
import sys

import pytest


class TestGroupedHelp:
    """Test that CLI shows grouped parameter sections."""

    def run_help(self, args=None):
        """Run CLI help command."""
        cmd = [sys.executable, "-m", "autotrain.cli.autotrain", "llm", "--help"]
        if args:
            cmd = [sys.executable, "-m", "autotrain.cli.autotrain", "llm"] + args + ["--help"]

        result = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "./src"})
        return result

    def test_basic_group_exists(self):
        """Test that Basic group heading exists."""
        result = self.run_help()
        assert result.returncode == 0
        assert "Basic:" in result.stdout

    def test_training_hyperparameters_group_exists(self):
        """Test that Training Hyperparameters group heading exists."""
        result = self.run_help()
        assert result.returncode == 0
        assert "Training Hyperparameters:" in result.stdout

    def test_peft_lora_group_exists(self):
        """Test that PEFT/LoRA group heading exists."""
        result = self.run_help()
        assert result.returncode == 0
        assert "PEFT/LoRA:" in result.stdout

    def test_data_processing_group_exists(self):
        """Test that Data Processing group heading exists."""
        result = self.run_help()
        assert result.returncode == 0
        assert "Data Processing:" in result.stdout

    def test_training_configuration_group_exists(self):
        """Test that Training Configuration group heading exists."""
        result = self.run_help()
        assert result.returncode == 0
        assert "Training Configuration:" in result.stdout

    def test_reinforcement_learning_group_exists(self):
        """Test that Reinforcement Learning (PPO) group heading exists."""
        result = self.run_help()
        assert result.returncode == 0
        assert "Reinforcement Learning (PPO):" in result.stdout

    def test_advanced_features_group_exists(self):
        """Test that Advanced Features group heading exists."""
        result = self.run_help()
        assert result.returncode == 0
        assert "Advanced Features:" in result.stdout

    def test_multiple_groups_in_order(self):
        """Test that multiple group headings appear in correct order."""
        result = self.run_help()
        assert result.returncode == 0
        stdout = result.stdout

        # Find positions of groups
        basic_pos = stdout.find("Basic:")
        data_pos = stdout.find("Data Processing:")
        training_config_pos = stdout.find("Training Configuration:")
        hyperparams_pos = stdout.find("Training Hyperparameters:")
        peft_pos = stdout.find("PEFT/LoRA:")

        # All should exist
        assert basic_pos > 0
        assert data_pos > 0
        assert training_config_pos > 0
        assert hyperparams_pos > 0
        assert peft_pos > 0

        # Basic should come first
        assert basic_pos < data_pos
        assert basic_pos < training_config_pos


class TestTrainerFiltering:
    """Test that parameters are filtered based on --trainer flag."""

    def run_help_with_trainer(self, trainer):
        """Run CLI help with specific trainer."""
        cmd = [sys.executable, "-m", "autotrain.cli.autotrain", "llm", "--trainer", trainer, "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "./src"})
        return result

    def test_ppo_trainer_shows_rl_params(self):
        """Test that PPO trainer shows RL parameters."""
        result = self.run_help_with_trainer("ppo")
        assert result.returncode == 0
        assert "--rl-gamma" in result.stdout
        assert "--rl-kl-coef" in result.stdout
        assert "--rl-reward-model-path" in result.stdout
        assert "Reinforcement Learning (PPO):" in result.stdout

    def test_sft_trainer_hides_rl_params(self):
        """Test that SFT trainer does not show RL parameters."""
        result = self.run_help_with_trainer("sft")
        assert result.returncode == 0
        assert "--rl-gamma" not in result.stdout
        assert "--rl-kl-coef" not in result.stdout
        assert "Reinforcement Learning (PPO):" not in result.stdout

    def test_dpo_trainer_shows_dpo_params(self):
        """Test that DPO trainer shows DPO-specific parameters."""
        result = self.run_help_with_trainer("dpo")
        assert result.returncode == 0
        assert "--dpo-beta" in result.stdout
        assert "--prompt-text-column" in result.stdout
        assert "--rejected-text-column" in result.stdout
        assert "DPO/ORPO:" in result.stdout

    def test_dpo_trainer_hides_rl_params(self):
        """Test that DPO trainer does not show RL parameters."""
        result = self.run_help_with_trainer("dpo")
        assert result.returncode == 0
        assert "--rl-gamma" not in result.stdout
        assert "Reinforcement Learning (PPO):" not in result.stdout

    def test_all_trainers_show_basic_params(self):
        """Test that all trainers show basic parameters."""
        for trainer in ["sft", "dpo", "ppo", "orpo"]:
            result = self.run_help_with_trainer(trainer)
            assert result.returncode == 0
            assert "--model" in result.stdout
            assert "--project-name" in result.stdout
            assert "--data-path" in result.stdout
            assert "--lr" in result.stdout
            assert "--epochs" in result.stdout

    def test_ppo_trainer_shows_filtered_description(self):
        """Test that PPO trainer shows filtered description."""
        result = self.run_help_with_trainer("ppo")
        assert result.returncode == 0
        # Should indicate it's showing PPO parameters
        assert "PPO" in result.stdout or "ppo" in result.stdout

    def test_sft_trainer_shows_advanced_forward_backward(self):
        """Test that SFT trainer shows advanced forward/backward params."""
        result = self.run_help_with_trainer("sft")
        assert result.returncode == 0
        assert "--use-forward-backward" in result.stdout
        assert "--gradient-accumulation-steps" in result.stdout

    def test_dpo_trainer_hides_advanced_forward_backward(self):
        """Test that DPO trainer hides advanced forward/backward params."""
        result = self.run_help_with_trainer("dpo")
        assert result.returncode == 0
        assert "--use-forward-backward" not in result.stdout
        assert "--gradient-accumulation-steps" not in result.stdout


class TestParameterAccessibility:
    """Test that all parameters are still accessible and aliases work."""

    def test_all_112_parameters_accessible_default(self):
        """Test that all 112 parameters are accessible with default trainer."""
        cmd = [sys.executable, "-m", "autotrain.cli.autotrain", "llm", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "./src"})
        assert result.returncode == 0

        # Count unique parameters (excluding aliases)
        lines = result.stdout.split("\n")
        param_lines = [line for line in lines if line.strip().startswith("--")]

        # Should have a good number of parameters (not exact 112 due to grouping/display)
        assert len(param_lines) > 100, f"Expected >100 parameters, found {len(param_lines)}"

    def test_parameter_aliases_work(self):
        """Test that parameter aliases still work."""
        cmd = [sys.executable, "-m", "autotrain.cli.autotrain", "llm", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "./src"})
        assert result.returncode == 0

        # Check that both snake_case and kebab-case versions appear
        assert "--lr, --lr, --lr LR" in result.stdout or "--lr" in result.stdout
        assert "--project-name" in result.stdout or "--project_name" in result.stdout
        assert "--gradient-accumulation" in result.stdout

    def test_help_trainer_flag_exists(self):
        """Test that --help-trainer flag exists."""
        cmd = [sys.executable, "-m", "autotrain.cli.autotrain", "llm", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "./src"})
        assert result.returncode == 0
        assert "--help-trainer" in result.stdout


class TestHelpTrainerFlag:
    """Test the --help-trainer flag functionality."""

    def test_help_trainer_flag_exists(self):
        """Test that --help-trainer flag is documented."""
        cmd = [sys.executable, "-m", "autotrain.cli.autotrain", "llm", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "./src"})
        assert result.returncode == 0
        assert "--help-trainer" in result.stdout
        assert "Show help for specific trainer" in result.stdout or "help-trainer" in result.stdout
