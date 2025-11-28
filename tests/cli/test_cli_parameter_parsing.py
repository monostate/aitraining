"""
Test CLI parameter parsing without running actual training.
This ensures all 107+ parameters are correctly parsed from CLI to API.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pytest


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from autotrain.cli.run_llm import RunAutoTrainLLMCommand
from autotrain.cli.utils import get_field_info
from autotrain.trainers.clm.params import LLMTrainingParams


class TestCLIParameterParsing:
    """Test that CLI correctly parses all parameters into LLMTrainingParams."""

    def parse_args_to_params(self, args_list):
        """Parse CLI arguments and convert to LLMTrainingParams."""
        # Create parser like the real CLI does
        parser = argparse.ArgumentParser()
        commands_parser = parser.add_subparsers()
        RunAutoTrainLLMCommand.register_subcommand(commands_parser)

        # Parse the arguments
        args = parser.parse_args(args_list)

        # Apply the same merge_adapter handling as RunAutoTrainLLMCommand.__init__
        if getattr(args, "merge_adapter", None) is None:
            delattr(args, "merge_adapter")

        # Convert to params (this is what the CLI does internally)
        params = LLMTrainingParams(**vars(args))
        return params

    def test_basic_parameters(self):
        """Test basic training parameters are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            args_list = [
                "llm",
                "--train",
                "--model",
                "gpt2",
                "--project-name",
                f"{tmp_dir}/project",
                "--data-path",
                f"{tmp_dir}/data",
                "--text-column",
                "text",
                "--lr",
                "1e-4",
                "--epochs",
                "3",
                "--batch-size",
                "8",
                "--warmup-ratio",
                "0.1",
                "--gradient-accumulation",
                "4",
                "--seed",
                "42",
            ]

            params = self.parse_args_to_params(args_list)

            # Verify parameters
            assert params.model == "gpt2"
            assert params.lr == 1e-4
            assert params.epochs == 3
            assert params.batch_size == 8
            assert params.warmup_ratio == 0.1
            assert params.gradient_accumulation == 4
            assert params.seed == 42

    def test_peft_parameters(self):
        """Test PEFT/LoRA parameters are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            args_list = [
                "llm",
                "--train",
                "--model",
                "gpt2",
                "--project-name",
                f"{tmp_dir}/project",
                "--data-path",
                f"{tmp_dir}/data",
                "--text-column",
                "text",
                "--peft",
                "--lora-r",
                "32",
                "--lora-alpha",
                "64",
                "--lora-dropout",
                "0.05",
                "--quantization",
                "int8",
            ]

            params = self.parse_args_to_params(args_list)

            # Verify PEFT parameters
            assert params.peft is True
            assert params.lora_r == 32
            assert params.lora_alpha == 64
            assert params.lora_dropout == 0.05
            assert params.quantization == "int8"

    def test_dpo_specific_parameters(self):
        """Test DPO-specific parameters are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            args_list = [
                "llm",
                "--train",
                "--trainer",
                "dpo",
                "--model",
                "gpt2",
                "--project-name",
                f"{tmp_dir}/project",
                "--data-path",
                f"{tmp_dir}/data",
                "--text-column",
                "chosen",
                "--prompt-text-column",
                "prompt",
                "--rejected-text-column",
                "rejected",
                "--dpo-beta",
                "0.1",
                "--max-prompt-length",
                "512",
                "--max-completion-length",
                "1024",
            ]

            params = self.parse_args_to_params(args_list)

            # Verify DPO parameters
            assert params.trainer == "dpo"
            assert params.prompt_text_column == "prompt"
            assert params.rejected_text_column == "rejected"
            assert params.dpo_beta == 0.1
            assert params.max_prompt_length == 512
            assert params.max_completion_length == 1024

    def test_ppo_rl_parameters(self):
        """Test PPO RL parameters are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a dummy reward model file
            reward_model_path = Path(tmp_dir) / "reward_model"
            reward_model_path.mkdir()

            args_list = [
                "llm",
                "--train",
                "--trainer",
                "ppo",
                "--model",
                "gpt2",
                "--project-name",
                f"{tmp_dir}/project",
                "--data-path",
                f"{tmp_dir}/data",
                "--text-column",
                "text",
                "--rl-reward-model-path",
                str(reward_model_path),
                "--rl-gamma",
                "0.99",
                "--rl-gae-lambda",
                "0.95",
                "--rl-kl-coef",
                "0.2",
                "--rl-value-loss-coef",
                "0.5",
                "--rl-clip-range",
                "0.2",
                "--rl-value-clip-range",
                "0.2",
                "--rl-num-ppo-epochs",
                "4",
                "--rl-mini-batch-size",
                "2",
                "--rl-max-new-tokens",
                "256",
                "--rl-top-k",
                "50",
                "--rl-top-p",
                "0.95",
                "--rl-temperature",
                "0.8",
            ]

            params = self.parse_args_to_params(args_list)

            # Verify PPO parameters
            assert params.trainer == "ppo"
            assert params.rl_reward_model_path == str(reward_model_path)
            assert params.rl_gamma == 0.99
            assert params.rl_gae_lambda == 0.95
            assert params.rl_kl_coef == 0.2
            assert params.rl_value_loss_coef == 0.5
            assert params.rl_clip_range == 0.2
            assert params.rl_value_clip_range == 0.2
            assert params.rl_num_ppo_epochs == 4
            assert params.rl_mini_batch_size == 2
            assert params.rl_max_new_tokens == 256
            assert params.rl_top_k == 50
            assert params.rl_top_p == 0.95
            assert params.rl_temperature == 0.8

    def test_advanced_features(self):
        """Test advanced features like distillation and sweep."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            args_list = [
                "llm",
                "--train",
                "--model",
                "gpt2",
                "--project-name",
                f"{tmp_dir}/project",
                "--data-path",
                f"{tmp_dir}/data",
                "--text-column",
                "text",
                "--trainer",
                "sft",
                "--use-distillation",
                "--teacher-model",
                "gpt2-large",
                "--distill-temperature",
                "3.0",
                "--distill-alpha",
                "0.5",
                "--use-sweep",
                "--sweep-n-trials",
                "10",
                "--sweep-metric",
                "eval_loss",
                "--sweep-direction",
                "minimize",
            ]

            params = self.parse_args_to_params(args_list)

            # Verify advanced parameters
            assert params.trainer == "sft"
            assert params.use_distillation is True
            assert params.teacher_model == "gpt2-large"
            assert params.distill_temperature == 3.0
            assert params.distill_alpha == 0.5
            assert params.use_sweep is True
            assert params.sweep_n_trials == 10
            assert params.sweep_metric == "eval_loss"
            assert params.sweep_direction == "minimize"

    def test_all_parameters_have_cli_args(self):
        """Test that all LLMTrainingParams fields have corresponding CLI arguments."""
        # Get all fields from the params class
        params_fields = set(LLMTrainingParams.model_fields.keys())

        # Get all CLI arguments
        cli_args = get_field_info(LLMTrainingParams)
        cli_arg_names = {arg["arg"].replace("--", "").replace("-", "_") for arg in cli_args}

        # Some fields are internal and not exposed via CLI
        internal_fields = {"backend", "config", "func", "version", "train", "deploy", "inference", "wandb_visualizer"}

        # Check that all public fields have CLI args
        public_fields = params_fields - internal_fields
        missing_cli_args = public_fields - cli_arg_names

        assert len(missing_cli_args) == 0, f"Fields without CLI args: {missing_cli_args}"

    def test_parameter_count(self):
        """Test that we have the expected number of parameters."""
        cli_args = get_field_info(LLMTrainingParams)

        # We expect 128 parameters:
        # - 107 original
        # - 5 PPO parameters (rl_value_clip_range, rl_max_new_tokens, rl_top_k, rl_top_p, rl_temperature)
        # - 6 inference parameters (inference_prompts, inference_max_tokens, inference_temperature,
        #                           inference_top_p, inference_top_k, inference_output)
        # - 1 W&B visualizer toggle handled via paired --wandb-visualizer/--no-wandb-visualizer flags
        # - 9 dataset conversion parameters (use_sharegpt_mapping, sharegpt_mapping_enabled,
        #                                    suggested_chat_template, column_mapping, runtime_mapping,
        #                                    map_eos_token, auto_convert_dataset, conversation_extension,
        #                                    apply_chat_template)
        assert len(cli_args) == 128, f"Expected 128 parameters, got {len(cli_args)}"
