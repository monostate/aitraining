"""Test chat template auto-selection based on trainer type."""

import pytest

from autotrain.trainers.clm.params import LLMTrainingParams


class TestChatTemplateDefaults:
    """Test that chat_template defaults are set correctly based on trainer type."""

    def test_sft_trainer_defaults_to_tokenizer(self):
        """SFT trainer should default to 'tokenizer' template."""
        params = LLMTrainingParams(
            model="test-model", project_name="test-project", data_path="test-data", trainer="sft"
        )
        assert params.chat_template == "tokenizer"

    def test_dpo_trainer_defaults_to_tokenizer(self):
        """DPO trainer should default to 'tokenizer' template."""
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="dpo",
            model_ref="test-ref-model",  # DPO requires ref model
            prompt_text_column="prompt",  # Required for DPO
            rejected_text_column="rejected",  # Required for DPO
        )
        assert params.chat_template == "tokenizer"

    def test_orpo_trainer_defaults_to_tokenizer(self):
        """ORPO trainer should default to 'tokenizer' template."""
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="orpo",
            prompt_text_column="prompt",  # Required for ORPO
            rejected_text_column="rejected",  # Required for ORPO
        )
        assert params.chat_template == "tokenizer"

    def test_reward_trainer_defaults_to_tokenizer(self):
        """Reward trainer should default to 'tokenizer' template."""
        params = LLMTrainingParams(
            model="test-model", project_name="test-project", data_path="test-data", trainer="reward"
        )
        assert params.chat_template == "tokenizer"

    def test_default_trainer_defaults_to_none(self):
        """Default trainer (pretraining) should default to None (no template)."""
        params = LLMTrainingParams(
            model="test-model", project_name="test-project", data_path="test-data", trainer="default"
        )
        assert params.chat_template is None

    def test_ppo_trainer_defaults_to_none(self):
        """PPO trainer should default to None (uses reward model format)."""
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="ppo",
            rl_reward_model_path="OpenAssistant/reward-model-deberta-v3-large-v2",  # Use HF model ID
        )
        assert params.chat_template is None

    def test_explicit_chat_template_respected(self):
        """Explicitly set chat_template should be respected regardless of trainer."""
        # Test with SFT trainer but explicit 'chatml' template
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="sft",
            chat_template="chatml",
        )
        assert params.chat_template == "chatml"

        # Test with default trainer but explicit 'tokenizer' template
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="default",
            chat_template="tokenizer",
        )
        assert params.chat_template == "tokenizer"

    def test_none_string_converted_to_none(self):
        """String 'none' should be converted to None value."""
        params = LLMTrainingParams(
            model="test-model", project_name="test-project", data_path="test-data", trainer="sft", chat_template="none"
        )
        # 'none' string is converted to None for plain text training
        assert params.chat_template is None


class TestChatTemplateOptions:
    """Test that all documented chat templates are valid options."""

    @pytest.mark.parametrize("template", ["tokenizer", "chatml", "zephyr", "alpaca", "vicuna", "llama", "mistral"])
    def test_valid_chat_templates(self, template):
        """All documented templates should be accepted."""
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="sft",
            chat_template=template,
        )
        assert params.chat_template == template

    def test_explicit_none_respected(self):
        """Explicitly passing None should be respected for plain text training."""
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="sft",
            chat_template=None,  # Explicit None for plain text
        )
        # When None is explicitly passed, it should remain None
        assert params.chat_template is None

    def test_string_none_converted(self):
        """String 'none' should be converted to actual None."""
        params = LLMTrainingParams(
            model="test-model",
            project_name="test-project",
            data_path="test-data",
            trainer="sft",
            chat_template="none",  # String 'none'
        )
        # 'none' string should be converted to None
        assert params.chat_template is None
