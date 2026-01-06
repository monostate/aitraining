"""
Tests for CLM trainer utilities
"""

import pytest
from unittest.mock import MagicMock, patch


def _is_hf_available():
    """Check if HuggingFace Hub is accessible."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Quick check - just see if we can instantiate
        return True
    except Exception:
        return False


class TestGetModelMaxPositionEmbeddings:
    """Tests for auto-detecting max_position_embeddings from model config."""

    def test_regular_llm_detection(self):
        """Test detection for regular LLMs (max_position_embeddings in root config)."""
        from autotrain.trainers.clm.utils import get_model_max_position_embeddings

        # Mock the AutoConfig
        mock_config = MagicMock()
        mock_config.max_position_embeddings = 8192
        mock_config.text_config = None  # No text_config

        with patch("autotrain.trainers.clm.utils.AutoConfig") as MockAutoConfig:
            MockAutoConfig.from_pretrained.return_value = mock_config

            result = get_model_max_position_embeddings("mock-model")

            assert result == 8192
            MockAutoConfig.from_pretrained.assert_called_once()

    def test_vlm_detection_with_text_config(self):
        """Test detection for VLMs (max_position_embeddings in text_config)."""
        from autotrain.trainers.clm.utils import get_model_max_position_embeddings

        # Mock VLM config with text_config
        mock_text_config = MagicMock()
        mock_text_config.max_position_embeddings = 131072

        mock_config = MagicMock()
        mock_config.text_config = mock_text_config
        # Don't set max_position_embeddings on root

        with patch("autotrain.trainers.clm.utils.AutoConfig") as MockAutoConfig:
            MockAutoConfig.from_pretrained.return_value = mock_config

            result = get_model_max_position_embeddings("mock-vlm")

            assert result == 131072

    def test_gpt2_style_n_positions(self):
        """Test detection for models using n_positions (like GPT-2)."""
        from autotrain.trainers.clm.utils import get_model_max_position_embeddings

        mock_config = MagicMock()
        mock_config.text_config = None
        mock_config.max_position_embeddings = None
        mock_config.n_positions = 1024

        # Need to handle getattr properly
        del mock_config.max_position_embeddings

        with patch("autotrain.trainers.clm.utils.AutoConfig") as MockAutoConfig:
            MockAutoConfig.from_pretrained.return_value = mock_config

            result = get_model_max_position_embeddings("gpt2")

            assert result == 1024

    def test_fallback_when_not_found(self):
        """Test that None is returned when max_position_embeddings can't be found."""
        from autotrain.trainers.clm.utils import get_model_max_position_embeddings

        mock_config = MagicMock()
        mock_config.text_config = None
        # Remove the attributes so getattr returns None
        del mock_config.max_position_embeddings
        del mock_config.n_positions

        with patch("autotrain.trainers.clm.utils.AutoConfig") as MockAutoConfig:
            MockAutoConfig.from_pretrained.return_value = mock_config

            result = get_model_max_position_embeddings("unknown-model")

            assert result is None

    def test_exception_handling(self):
        """Test graceful handling of exceptions during config loading."""
        from autotrain.trainers.clm.utils import get_model_max_position_embeddings

        with patch("autotrain.trainers.clm.utils.AutoConfig") as MockAutoConfig:
            MockAutoConfig.from_pretrained.side_effect = Exception("Network error")

            result = get_model_max_position_embeddings("failing-model")

            assert result is None

    @pytest.mark.skipif(not _is_hf_available(), reason="HuggingFace Hub not available")
    def test_real_gemma_detection(self):
        """Test with real Gemma model (requires network)."""
        from autotrain.trainers.clm.utils import get_model_max_position_embeddings

        result = get_model_max_position_embeddings("google/gemma-2-2b")

        # Gemma 2 2B has 8192 context
        assert result == 8192

    @pytest.mark.skipif(not _is_hf_available(), reason="HuggingFace Hub not available")
    def test_real_vlm_detection(self):
        """Test with real VLM model (requires network)."""
        from autotrain.trainers.clm.utils import get_model_max_position_embeddings

        result = get_model_max_position_embeddings("google/gemma-3-4b-it")

        # Gemma 3 4B (VLM) has 128K context
        assert result == 131072

    @pytest.mark.skipif(not _is_hf_available(), reason="HuggingFace Hub not available")
    def test_real_gemma3n_detection(self):
        """Test with real Gemma 3n model (requires network)."""
        from autotrain.trainers.clm.utils import get_model_max_position_embeddings

        result = get_model_max_position_embeddings("google/gemma-3n-e4b-it")

        # Gemma 3n has 32K context
        assert result == 32768
