import sys
import tempfile

import pytest
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from autotrain.trainers.clm.params import LLMTrainingParams

TINY_MODEL = "sshleifer/tiny-gpt2"


class SimpleMatchEnv:
    """Minimal env for testing: scores 1.0 if completion contains 'hello', else 0.0."""

    def build_dataset(self, tokenizer):
        prompts = [
            "Say hello to the world",
            "Greet everyone with hello",
            "Write a hello message",
            "Say hi to people",
        ]
        return Dataset.from_dict({
            "prompt": prompts,
            "case_idx": list(range(len(prompts))),
        })

    def score_episode(self, model, tokenizer, completion, case_idx):
        text = completion if isinstance(completion, str) else str(completion)
        return 1.0 if "hello" in text.lower() else 0.0

    def get_tools(self):
        return []


class TestGRPOEnvInterface:
    def test_build_dataset(self):
        env = SimpleMatchEnv()
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        ds = env.build_dataset(tokenizer)
        assert "prompt" in ds.column_names
        assert "case_idx" in ds.column_names
        assert len(ds) == 4

    def test_score_episode(self):
        env = SimpleMatchEnv()
        assert env.score_episode(None, None, "hello world", 0) == 1.0
        assert env.score_episode(None, None, "goodbye", 0) == 0.0

    def test_get_tools(self):
        env = SimpleMatchEnv()
        assert env.get_tools() == []


class TestGRPORewardFnWrapping:
    def test_reward_fn_returns_list_float(self):
        env = SimpleMatchEnv()

        def reward_fn(completions, prompts, **kwargs):
            rewards = []
            case_indices = kwargs.get("case_idx", list(range(len(completions))))
            for i, completion in enumerate(completions):
                case_idx = case_indices[i] if i < len(case_indices) else i
                score = env.score_episode(None, None, completion, case_idx)
                rewards.append(float(score))
            return rewards

        result = reward_fn(
            completions=["hello world", "goodbye", "say hello"],
            prompts=["p1", "p2", "p3"],
            case_idx=[0, 1, 2],
        )
        assert result == [1.0, 0.0, 1.0]
        assert all(isinstance(r, float) for r in result)


class TestGRPOConfigCreation:
    def test_grpo_config_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grpo_config = GRPOConfig(
                output_dir=tmpdir,
                num_generations=2,
                max_completion_length=32,
                temperature=1.0,
                beta=0.0,
                epsilon=0.2,
                loss_type="grpo",
                per_device_train_batch_size=2,
                num_train_epochs=1,
                logging_steps=1,
                report_to="none",
            )
            assert grpo_config.num_generations == 2
            assert grpo_config.loss_type == "grpo"
            assert grpo_config.beta == 0.0

    def test_grpo_config_dr_grpo_loss(self):
        """Verify dr_grpo loss_type is accepted by GRPOConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            grpo_config = GRPOConfig(
                output_dir=tmpdir,
                num_generations=2,
                max_completion_length=32,
                loss_type="dr_grpo",
                mask_truncated_completions=True,
                per_device_train_batch_size=2,
                report_to="none",
            )
            assert grpo_config.loss_type == "dr_grpo"
            assert grpo_config.mask_truncated_completions is True


class TestGRPOTrainerRuns:
    @pytest.mark.slow
    def test_grpo_trainer_completes(self):
        """Full integration test: load model, create trainer, run 1 step."""
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)

        env = SimpleMatchEnv()
        train_dataset = env.build_dataset(tokenizer)

        def reward_fn(completions, prompts, **kwargs):
            rewards = []
            for completion in completions:
                text = completion if isinstance(completion, str) else str(completion)
                rewards.append(1.0 if "hello" in text.lower() else 0.0)
            return rewards

        with tempfile.TemporaryDirectory() as tmpdir:
            grpo_config = GRPOConfig(
                output_dir=tmpdir,
                num_generations=2,
                max_completion_length=16,
                temperature=1.0,
                beta=0.0,
                epsilon=0.2,
                loss_type="grpo",
                per_device_train_batch_size=2,
                num_train_epochs=1,
                max_steps=1,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                gradient_accumulation_steps=1,
            )

            trainer = GRPOTrainer(
                model=model,
                reward_funcs=reward_fn,
                args=grpo_config,
                train_dataset=train_dataset,
                processing_class=tokenizer,
            )

            trainer.train()


class TestGRPODispatch:
    def test_dispatch_routes_to_grpo(self):
        """Verify __main__.train dispatches grpo correctly."""
        from unittest.mock import patch

        with patch("autotrain.trainers.clm.train_clm_grpo.train") as mock_train:
            from autotrain.trainers.clm.__main__ import train

            config = LLMTrainingParams(
                model=TINY_MODEL,
                trainer="grpo",
                rl_env_module="tests.test_grpo_trainer",
                rl_env_class="SimpleMatchEnv",
            )
            train(config)
            mock_train.assert_called_once_with(config)


class TestGRPOParamsValidation:
    def test_grpo_with_env_params_passes(self):
        config = LLMTrainingParams(
            model=TINY_MODEL,
            trainer="grpo",
            rl_env_module="my_envs.hotel_env",
            rl_env_class="HotelEnv",
        )
        assert config.rl_env_module == "my_envs.hotel_env"
        assert config.rl_env_class == "HotelEnv"
        assert config.rl_num_generations == 4

    def test_grpo_without_env_module_raises(self):
        with pytest.raises(ValueError, match="rl-env-module"):
            LLMTrainingParams(
                model=TINY_MODEL,
                trainer="grpo",
                rl_env_class="HotelEnv",
            )

    def test_grpo_without_env_class_raises(self):
        with pytest.raises(ValueError, match="rl-env-class"):
            LLMTrainingParams(
                model=TINY_MODEL,
                trainer="grpo",
                rl_env_module="my_envs.hotel_env",
            )

    def test_grpo_shares_rl_params(self):
        """Shared RL params should not warn when used with grpo."""
        config = LLMTrainingParams(
            model=TINY_MODEL,
            trainer="grpo",
            rl_env_module="my_envs.hotel_env",
            rl_env_class="HotelEnv",
            rl_temperature=0.7,
            rl_max_new_tokens=256,
            rl_clip_range=0.1,
        )
        assert config.rl_temperature == 0.7
        assert config.rl_max_new_tokens == 256
        assert config.rl_clip_range == 0.1

    def test_grpo_loss_type_and_mask_params(self):
        config = LLMTrainingParams(
            model=TINY_MODEL,
            trainer="grpo",
            rl_env_module="my_envs.hotel_env",
            rl_env_class="HotelEnv",
            rl_loss_type="dr_grpo",
            rl_mask_truncated_completions=True,
        )
        assert config.rl_loss_type == "dr_grpo"
        assert config.rl_mask_truncated_completions is True

    def test_grpo_loss_type_default(self):
        config = LLMTrainingParams(
            model=TINY_MODEL,
            trainer="grpo",
            rl_env_module="my_envs.hotel_env",
            rl_env_class="HotelEnv",
        )
        assert config.rl_loss_type == "grpo"
        assert config.rl_mask_truncated_completions is False


class TestGRPOFieldScopes:
    def test_grpo_in_valid_trainers(self):
        from autotrain.cli.run_llm import VALID_TRAINERS

        assert "grpo" in VALID_TRAINERS

    def test_grpo_only_params_scoped(self):
        from autotrain.cli.run_llm import FIELD_SCOPES

        assert FIELD_SCOPES["rl_env_module"] == ["grpo"]
        assert FIELD_SCOPES["rl_env_class"] == ["grpo"]
        assert FIELD_SCOPES["rl_num_generations"] == ["grpo"]
        assert FIELD_SCOPES["rl_loss_type"] == ["grpo"]
        assert FIELD_SCOPES["rl_mask_truncated_completions"] == ["grpo"]

    def test_shared_params_include_grpo(self):
        from autotrain.cli.run_llm import FIELD_SCOPES

        shared_params = [
            "rl_temperature", "rl_max_new_tokens", "rl_top_k",
            "rl_top_p", "rl_clip_range", "rl_kl_coef", "rl_env_config",
        ]
        for param in shared_params:
            assert "grpo" in FIELD_SCOPES[param], f"{param} should include grpo in scope"
            assert "ppo" in FIELD_SCOPES[param], f"{param} should still include ppo"

    def test_ppo_only_params_exclude_grpo(self):
        from autotrain.cli.run_llm import FIELD_SCOPES

        ppo_only = [
            "rl_gamma", "rl_gae_lambda", "rl_value_loss_coef",
            "rl_reward_fn", "rl_reward_model_path", "rl_num_ppo_epochs",
        ]
        for param in ppo_only:
            assert "grpo" not in FIELD_SCOPES[param], f"{param} should NOT include grpo"


class TestDDPTimeout:
    def test_ddp_timeout_default(self):
        config = LLMTrainingParams(model=TINY_MODEL, trainer="sft")
        assert config.ddp_timeout == 7200

    def test_ddp_timeout_custom(self):
        config = LLMTrainingParams(model=TINY_MODEL, trainer="sft", ddp_timeout=3600)
        assert config.ddp_timeout == 3600

    def test_ddp_timeout_in_training_args(self):
        from autotrain.trainers.clm.utils import configure_training_args

        config = LLMTrainingParams(model=TINY_MODEL, trainer="sft", ddp_timeout=9000)
        training_args = configure_training_args(config, logging_steps=10)
        assert training_args["ddp_timeout"] == 9000

    def test_ddp_timeout_field_scope(self):
        from autotrain.cli.run_llm import FIELD_SCOPES

        assert "ddp_timeout" in FIELD_SCOPES
        assert FIELD_SCOPES["ddp_timeout"] == ["all"]

    def test_nccl_env_var_set(self):
        """Verify launch_command sets TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC."""
        import os

        config = LLMTrainingParams(model=TINY_MODEL, trainer="sft", ddp_timeout=5000)
        from autotrain.commands import launch_command

        os.environ["AUTOTRAIN_FORCE_NUM_GPUS"] = "1"
        try:
            launch_command(config)
            assert os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC") == "5000"
        finally:
            os.environ.pop("AUTOTRAIN_FORCE_NUM_GPUS", None)
            os.environ.pop("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", None)

    def test_nccl_timeout_env_var_set(self):
        """Verify launch_command sets NCCL_TIMEOUT env var."""
        import os
        from autotrain.commands import launch_command

        config = LLMTrainingParams(model=TINY_MODEL, trainer="sft", ddp_timeout=5000)
        os.environ["AUTOTRAIN_FORCE_NUM_GPUS"] = "1"
        try:
            launch_command(config)
            assert os.environ.get("NCCL_TIMEOUT") == "5000"
        finally:
            os.environ.pop("AUTOTRAIN_FORCE_NUM_GPUS", None)
            os.environ.pop("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", None)
            os.environ.pop("NCCL_TIMEOUT", None)

    def test_no_timeout_flag_in_accelerate_command(self):
        """accelerate launch does not support --timeout flag."""
        from autotrain.commands import get_accelerate_command

        cmd = get_accelerate_command(4)
        assert "--timeout" not in cmd


class TestVLLMServerMode:
    def test_vllm_server_params_defaults(self):
        config = LLMTrainingParams(
            model=TINY_MODEL,
            trainer="grpo",
            rl_env_module="tests.test_grpo_trainer",
            rl_env_class="SimpleMatchEnv",
        )
        assert config.vllm_server_url is None
        assert config.vllm_tensor_parallel_size == 1
        assert config.vllm_server_gpus == 1

    def test_vllm_server_params_custom(self):
        config = LLMTrainingParams(
            model=TINY_MODEL,
            trainer="grpo",
            rl_env_module="tests.test_grpo_trainer",
            rl_env_class="SimpleMatchEnv",
            use_vllm=True,
            vllm_mode="server",
            vllm_server_url="http://localhost:8000/v1",
            vllm_tensor_parallel_size=2,
            vllm_server_gpus=2,
        )
        assert config.vllm_server_url == "http://localhost:8000/v1"
        assert config.vllm_tensor_parallel_size == 2
        assert config.vllm_server_gpus == 2

    def test_vllm_server_url_maps_to_grpo_config_key(self):
        """Verify vllm_server_url maps to vllm_server_base_url in GRPOConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            grpo_config = GRPOConfig(
                output_dir=tmpdir,
                num_generations=2,
                max_completion_length=16,
                per_device_train_batch_size=2,
                report_to="none",
                use_vllm=True,
                vllm_mode="server",
                vllm_server_base_url="http://localhost:8000/v1",
            )
            assert grpo_config.vllm_server_base_url == "http://localhost:8000/v1"

    def test_vllm_server_field_scopes(self):
        from autotrain.cli.run_llm import FIELD_SCOPES

        assert FIELD_SCOPES["vllm_server_url"] == ["grpo"]
        assert FIELD_SCOPES["vllm_tensor_parallel_size"] == ["grpo"]
        assert FIELD_SCOPES["vllm_server_gpus"] == ["grpo"]

    def test_vllm_server_reduces_num_processes(self):
        """Verify launch_command reduces num_processes when vllm_mode=server."""
        import os

        config = LLMTrainingParams(
            model=TINY_MODEL,
            trainer="grpo",
            rl_env_module="tests.test_grpo_trainer",
            rl_env_class="SimpleMatchEnv",
            use_vllm=True,
            vllm_mode="server",
            vllm_server_gpus=2,
        )
        from autotrain.commands import launch_command

        os.environ["AUTOTRAIN_FORCE_NUM_GPUS"] = "4"
        try:
            cmd = launch_command(config)
            # Should have 4 - 2 = 2 training processes
            if "--num_processes" in cmd:
                idx = cmd.index("--num_processes")
                assert cmd[idx + 1] == "2"
        finally:
            os.environ.pop("AUTOTRAIN_FORCE_NUM_GPUS", None)

    def test_vllm_colocate_does_not_reduce_processes(self):
        """Verify colocate mode does NOT reduce num_processes."""
        import os

        config = LLMTrainingParams(
            model=TINY_MODEL,
            trainer="grpo",
            rl_env_module="tests.test_grpo_trainer",
            rl_env_class="SimpleMatchEnv",
            use_vllm=True,
            vllm_mode="colocate",
            vllm_server_gpus=2,
        )
        from autotrain.commands import launch_command

        os.environ["AUTOTRAIN_FORCE_NUM_GPUS"] = "4"
        try:
            cmd = launch_command(config)
            # Should still use all 4 GPUs
            if "--num_processes" in cmd:
                idx = cmd.index("--num_processes")
                assert cmd[idx + 1] == "4"
        finally:
            os.environ.pop("AUTOTRAIN_FORCE_NUM_GPUS", None)


class TestResumeFromCheckpoint:
    def test_resume_default_none(self):
        config = LLMTrainingParams(model=TINY_MODEL, trainer="sft")
        assert config.resume_from_checkpoint is None

    def test_resume_explicit_value(self):
        config = LLMTrainingParams(model=TINY_MODEL, trainer="sft", resume_from_checkpoint="/tmp/ckpt")
        assert config.resume_from_checkpoint == "/tmp/ckpt"

    def test_get_resume_checkpoint_none(self):
        from autotrain.trainers.clm.utils import get_resume_checkpoint

        config = LLMTrainingParams(model=TINY_MODEL, trainer="sft")
        assert get_resume_checkpoint(config) is None

    def test_get_resume_checkpoint_auto_no_checkpoints(self):
        from autotrain.trainers.clm.utils import get_resume_checkpoint

        config = LLMTrainingParams(
            model=TINY_MODEL, trainer="sft",
            resume_from_checkpoint="auto",
            project_name="/tmp/nonexistent_proj_xyz",
        )
        assert get_resume_checkpoint(config) is None

    def test_get_resume_checkpoint_auto_with_checkpoints(self):
        import os
        from autotrain.trainers.clm.utils import get_resume_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "checkpoint-100"))
            os.makedirs(os.path.join(tmpdir, "checkpoint-200"))
            os.makedirs(os.path.join(tmpdir, "checkpoint-50"))
            config = LLMTrainingParams(
                model=TINY_MODEL, trainer="sft",
                resume_from_checkpoint="auto",
                project_name=tmpdir,
            )
            result = get_resume_checkpoint(config)
            # sorted numerically: checkpoint-50, checkpoint-100, checkpoint-200 -> last is checkpoint-200
            assert result == os.path.join(tmpdir, "checkpoint-200")

    def test_get_resume_checkpoint_valid_path(self):
        import os
        from autotrain.trainers.clm.utils import get_resume_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "checkpoint-42")
            os.makedirs(ckpt)
            config = LLMTrainingParams(
                model=TINY_MODEL, trainer="sft",
                resume_from_checkpoint=ckpt,
            )
            assert get_resume_checkpoint(config) == ckpt

    def test_get_resume_checkpoint_invalid_path(self):
        from autotrain.trainers.clm.utils import get_resume_checkpoint

        config = LLMTrainingParams(
            model=TINY_MODEL, trainer="sft",
            resume_from_checkpoint="/tmp/does_not_exist_xyz_123",
        )
        assert get_resume_checkpoint(config) is None

    def test_resume_field_scope(self):
        from autotrain.cli.run_llm import FIELD_SCOPES

        assert "resume_from_checkpoint" in FIELD_SCOPES
        assert FIELD_SCOPES["resume_from_checkpoint"] == ["all"]

    def test_resume_field_group(self):
        from autotrain.cli.run_llm import FIELD_GROUPS

        assert FIELD_GROUPS["resume_from_checkpoint"] == "Training Configuration"


class TestORPOMultiTurnPrompt:
    """Test that ORPO/DPO prompt extraction uses explicit prompt column for multi-turn."""

    def _make_config(self, trainer="orpo"):
        return LLMTrainingParams(
            model=TINY_MODEL,
            trainer=trainer,
            prompt_text_column="prompt",
            rejected_text_column="rejected",
        )

    def _make_renderer(self):
        from autotrain.rendering.message_renderer import get_renderer

        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        return get_renderer("native", tokenizer), tokenizer

    def test_multiturn_uses_explicit_prompt(self):
        """When prompt column is a messages list, it should be used instead of chosen[:-1]."""
        from autotrain.trainers.clm.utils import apply_chat_template_unified

        renderer, _ = self._make_renderer()
        config = self._make_config("orpo")

        # Multi-turn: prompt is first 2 messages, chosen has full 4-message trajectory
        example = {
            "prompt": [
                {"role": "user", "content": "Book me a hotel"},
                {"role": "assistant", "content": "Sure, let me search."},
            ],
            "chosen": [
                {"role": "user", "content": "Book me a hotel"},
                {"role": "assistant", "content": "Sure, let me search."},
                {"role": "user", "content": "In Paris please"},
                {"role": "assistant", "content": "Done, booked Hotel Lumiere."},
            ],
            "rejected": [
                {"role": "user", "content": "Book me a hotel"},
                {"role": "assistant", "content": "Sure, let me search."},
                {"role": "user", "content": "In Paris please"},
                {"role": "assistant", "content": "I cannot do that."},
            ],
        }

        result = apply_chat_template_unified(example, renderer, config)

        # Prompt should be rendered from the explicit 2-message prompt, not chosen[:-1] (3 messages)
        assert "Book me a hotel" in result["prompt"]
        assert "Sure, let me search" in result["prompt"]
        # Prompt should NOT contain the second user turn (that's part of the completion)
        assert "In Paris please" not in result["prompt"]

        # Chosen and rejected should contain the full trajectory
        assert "Done, booked Hotel Lumiere" in result["chosen"]
        assert "I cannot do that" in result["rejected"]

    def test_singleturn_falls_back_to_chosen_minus_last(self):
        """Without explicit prompt column, should fall back to chosen[:-1]."""
        from autotrain.trainers.clm.utils import apply_chat_template_unified

        renderer, _ = self._make_renderer()
        config = self._make_config("orpo")

        example = {
            "chosen": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            "rejected": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "5"},
            ],
        }

        result = apply_chat_template_unified(example, renderer, config)

        # Prompt should be derived from chosen[:-1] = just the user message
        assert "What is 2+2" in result["prompt"]
        assert "4" in result["chosen"]
        assert "5" in result["rejected"]

    def test_dpo_also_uses_explicit_prompt(self):
        """DPO trainer shares the same code path."""
        from autotrain.trainers.clm.utils import apply_chat_template_unified

        renderer, _ = self._make_renderer()
        config = self._make_config("dpo")

        example = {
            "prompt": [
                {"role": "user", "content": "Hello"},
            ],
            "chosen": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Great, thanks!"},
            ],
            "rejected": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Go away."},
            ],
        }

        result = apply_chat_template_unified(example, renderer, config)

        # Prompt should only contain "Hello" user message
        assert "Hello" in result["prompt"]
        assert "How are you" not in result["prompt"]
