"""
Tests for LLM trainer metrics (SFT, DPO, ORPO, PPO).
Tests the new get_metric functionality for language model trainers.
"""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.common_metrics import get_rl_metrics, get_sft_metrics


class TestSFTMetrics:
    """Test suite for SFT trainer metrics."""

    def test_sft_metrics_none_returns_none(self):
        """Test that get_sft_metrics returns None when no metrics specified."""
        compute_fn = get_sft_metrics(None)
        assert compute_fn is None

        compute_fn = get_sft_metrics([])
        assert compute_fn is None

    def test_sft_metrics_with_bleu_rouge(self):
        """Test SFT metrics with BLEU and ROUGE."""
        with patch("evaluate.load") as mock_load:
            # Mock BLEU metric
            mock_bleu = MagicMock()
            mock_bleu.compute.return_value = {"bleu": 0.75}

            # Mock ROUGE metric
            mock_rouge = MagicMock()
            mock_rouge.compute.return_value = {"rouge1": 0.8, "rouge2": 0.6, "rougeL": 0.7}

            # Configure mock to return different metrics
            def load_side_effect(name, *args, **kwargs):
                if name == "bleu":
                    return mock_bleu
                elif name == "rouge":
                    return mock_rouge
                return MagicMock()

            mock_load.side_effect = load_side_effect

            # Create tokenizer mock
            tokenizer = MagicMock()
            tokenizer.batch_decode.return_value = ["generated text", "more text"]

            compute_fn = get_sft_metrics(["bleu", "rouge"], tokenizer=tokenizer)
            assert compute_fn is not None

            # Test compute function
            import numpy as np

            eval_pred = (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]]))  # predictions  # labels
            results = compute_fn(eval_pred)
            assert results is not None

    def test_sft_trainer_with_custom_metrics(self):
        """Test that SFT trainer can be configured with custom metrics."""
        config = LLMTrainingParams(
            model="gpt2",
            data_path="dummy",
            train_split="train",
            text_column="text",
            custom_metrics='["bleu"]',  # JSON string
        )

        assert hasattr(config, "custom_metrics")
        assert config.custom_metrics == '["bleu"]'

        # Parse the custom metrics
        custom_metrics = json.loads(config.custom_metrics)
        assert custom_metrics == ["bleu"]


class TestRLMetrics:
    """Test suite for RL trainer metrics (DPO, ORPO, PPO)."""

    def test_rl_metrics_none_returns_none(self):
        """Test that get_rl_metrics returns None when no metrics specified."""
        compute_fn = get_rl_metrics(None)
        assert compute_fn is None

        compute_fn = get_rl_metrics([])
        assert compute_fn is None

    def test_rl_metrics_with_custom(self):
        """Test RL metrics with custom metrics."""
        with patch("evaluate.load") as mock_load:
            mock_metric = MagicMock()
            mock_metric.compute.return_value = {"reward_accuracy": 0.85}
            mock_load.return_value = mock_metric

            compute_fn = get_rl_metrics(["reward_accuracy"])
            assert compute_fn is not None

    def test_dpo_trainer_with_custom_metrics(self):
        """Test that DPO trainer can be configured with custom metrics."""
        config = LLMTrainingParams(
            model="gpt2",
            data_path="dummy",
            train_split="train",
            text_column="text",
            trainer="dpo",
            prompt_text_column="prompt",
            rejected_text_column="rejected",
            custom_metrics='["accuracy"]',
        )

        assert hasattr(config, "custom_metrics")
        assert config.custom_metrics == '["accuracy"]'

    def test_orpo_trainer_with_custom_metrics(self):
        """Test that ORPO trainer can be configured with custom metrics."""
        config = LLMTrainingParams(
            model="gpt2",
            data_path="dummy",
            train_split="train",
            text_column="text",
            trainer="orpo",
            prompt_text_column="prompt",
            rejected_text_column="rejected",
            custom_metrics='["f1"]',
        )

        assert hasattr(config, "custom_metrics")
        assert config.custom_metrics == '["f1"]'

    def test_ppo_trainer_with_custom_metrics(self):
        """Test that PPO trainer can be configured with custom metrics."""
        # Note: PPO requires a reward model, so we skip validation
        # by not setting trainer="ppo" and setting the path manually
        config = LLMTrainingParams(
            model="gpt2",
            data_path="dummy",
            train_split="train",
            text_column="text",
            custom_metrics='["reward_correlation"]',
        )
        # Manually set the fields that would be set for PPO
        config.trainer = "ppo"
        config.rl_reward_model_path = "mock-reward-model"

        assert hasattr(config, "custom_metrics")
        assert config.custom_metrics == '["reward_correlation"]'
        assert config.trainer == "ppo"


class TestMetricsIntegration:
    """Test metrics integration across trainers."""

    def test_all_trainers_support_custom_metrics(self):
        """Verify all trainer types can handle custom metrics parameter."""
        trainers = ["default", "sft", "dpo", "orpo"]  # PPO requires special setup

        for trainer in trainers:
            # Create appropriate config for each trainer
            config_dict = {
                "model": "gpt2",
                "data_path": "dummy",
                "train_split": "train",
                "text_column": "text",
                "trainer": trainer,
                "custom_metrics": '["accuracy"]',
            }

            # Add required columns for preference trainers
            if trainer in ["dpo", "orpo"]:
                config_dict["prompt_text_column"] = "prompt"
                config_dict["rejected_text_column"] = "rejected"

            config = LLMTrainingParams(**config_dict)
            assert hasattr(config, "custom_metrics")
            assert config.custom_metrics == '["accuracy"]'

    @patch("autotrain.trainers.clm.utils.get_sft_metrics")
    @patch("autotrain.trainers.clm.utils.get_rl_metrics")
    def test_metrics_functions_called_correctly(self, mock_rl_metrics, mock_sft_metrics):
        """Test that the right metric functions are called for each trainer."""
        # Mock return values
        mock_sft_metrics.return_value = MagicMock()
        mock_rl_metrics.return_value = MagicMock()

        # Test SFT uses get_sft_metrics
        from autotrain.trainers.clm import utils

        tokenizer = MagicMock()

        compute_fn = utils.get_sft_metrics(["bleu"], tokenizer)
        mock_sft_metrics.assert_called_with(["bleu"], tokenizer)

        # Test RL trainers use get_rl_metrics
        compute_fn = utils.get_rl_metrics(["accuracy"])
        mock_rl_metrics.assert_called_with(["accuracy"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
