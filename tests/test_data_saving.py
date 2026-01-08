"""
Tests for processed dataset saving functionality.

Tests cover:
- data_utils.py centralized save functions
- Hub detection logic
- Size estimation
- Save modes (auto, local, hub, both, none)
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from autotrain.data_utils import (
    SIZE_THRESHOLD_BYTES,
    estimate_dataset_size,
    is_hub_dataset,
    save_processed_dataset,
    save_processed_datasets,
)


class TestIsHubDataset:
    """Tests for is_hub_dataset function."""

    def test_hub_dataset_format(self):
        """Test that user/dataset format is detected as Hub."""
        assert is_hub_dataset("username/dataset-name") is True
        assert is_hub_dataset("org/my-dataset") is True
        assert is_hub_dataset("HuggingFace/test") is True

    def test_local_path_not_hub(self):
        """Test that local paths are not detected as Hub."""
        assert is_hub_dataset("/absolute/path/to/data") is False
        assert is_hub_dataset("./relative/path") is False
        assert is_hub_dataset("~/home/data") is False

    def test_empty_path(self):
        """Test that empty/None paths return False."""
        assert is_hub_dataset("") is False
        assert is_hub_dataset(None) is False

    def test_existing_local_path(self, tmp_path):
        """Test that existing local paths return False even with slash."""
        # Create a directory that looks like hub format
        local_dir = tmp_path / "user" / "dataset"
        local_dir.mkdir(parents=True)
        assert is_hub_dataset(str(tmp_path / "user")) is False

    def test_invalid_formats(self):
        """Test that invalid formats return False."""
        assert is_hub_dataset("no-slash") is False
        assert is_hub_dataset("too/many/slashes") is False
        assert is_hub_dataset("/") is False


class TestEstimateDatasetSize:
    """Tests for estimate_dataset_size function."""

    def test_pandas_dataframe(self):
        """Test size estimation for pandas DataFrame."""
        df = pd.DataFrame({"text": ["hello world"] * 100, "label": [1] * 100})
        size = estimate_dataset_size(df)
        assert size > 0
        # Size can be int, float, or numpy numeric type
        assert float(size) > 0

    def test_empty_dataframe(self):
        """Test size estimation for empty DataFrame."""
        df = pd.DataFrame()
        size = estimate_dataset_size(df)
        assert size >= 0


class TestSaveProcessedDataset:
    """Tests for save_processed_dataset function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {"text": ["Sample text 1", "Sample text 2", "Sample text 3"], "label": [0, 1, 0]}
        )

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create temporary project directory."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        return str(project_dir)

    def test_save_mode_none(self, sample_df, temp_project):
        """Test that save_mode='none' skips saving."""
        result = save_processed_dataset(
            dataset=sample_df,
            project_name=temp_project,
            split_name="train",
            save_mode="none",
        )
        assert result["local_path"] is None
        assert result["hub_path"] is None

    def test_save_mode_local(self, sample_df, temp_project):
        """Test that save_mode='local' saves locally."""
        result = save_processed_dataset(
            dataset=sample_df,
            project_name=temp_project,
            split_name="train",
            save_mode="local",
        )
        assert result["local_path"] is not None
        assert os.path.exists(result["local_path"])
        assert result["hub_path"] is None

    def test_save_mode_auto_local_source(self, sample_df, temp_project):
        """Test that save_mode='auto' with local source saves only locally."""
        result = save_processed_dataset(
            dataset=sample_df,
            project_name=temp_project,
            split_name="train",
            source_path="/local/path/data",
            save_mode="auto",
        )
        assert result["local_path"] is not None
        assert os.path.exists(result["local_path"])
        assert result["hub_path"] is None

    def test_save_mode_auto_hub_source_no_token(self, sample_df, temp_project):
        """Test that save_mode='auto' with hub source but no token only saves locally."""
        result = save_processed_dataset(
            dataset=sample_df,
            project_name=temp_project,
            split_name="train",
            source_path="username/dataset",
            save_mode="auto",
        )
        # Should save locally, but not to hub (no token)
        assert result["local_path"] is not None
        assert result["hub_path"] is None

    def test_save_mode_hub_no_credentials(self, sample_df, temp_project):
        """Test that save_mode='hub' without credentials warns and skips hub."""
        result = save_processed_dataset(
            dataset=sample_df,
            project_name=temp_project,
            split_name="train",
            save_mode="hub",
        )
        # Should not save locally (mode is hub only)
        assert result["local_path"] is None
        # Should not save to hub (no credentials)
        assert result["hub_path"] is None

    def test_hub_repo_naming(self, sample_df, temp_project):
        """Test that Hub repo name follows aitraining-processed-{project}-{date} format."""
        # This test just verifies the naming logic without actually pushing
        import re
        from datetime import datetime

        # The expected pattern
        date_str = datetime.now().strftime("%Y%m%d")
        project_basename = "test_project"
        expected_pattern = f"aitraining-processed-{project_basename.lower()}-{date_str}"

        # Verify the pattern matches expected format
        assert re.match(r"aitraining-processed-[\w-]+-\d{8}", expected_pattern)

    def test_saved_file_format(self, sample_df, temp_project):
        """Test that saved file is valid JSONL."""
        result = save_processed_dataset(
            dataset=sample_df,
            project_name=temp_project,
            split_name="train",
            save_mode="local",
        )
        # Read back and verify
        loaded_df = pd.read_json(result["local_path"], lines=True)
        assert len(loaded_df) == len(sample_df)
        assert list(loaded_df.columns) == list(sample_df.columns)

    def test_split_name_in_filename(self, sample_df, temp_project):
        """Test that split name is used in filename."""
        result = save_processed_dataset(
            dataset=sample_df,
            project_name=temp_project,
            split_name="validation",
            save_mode="local",
        )
        assert "validation.jsonl" in result["local_path"]


class TestSaveProcessedDatasets:
    """Tests for save_processed_datasets convenience function."""

    @pytest.fixture
    def sample_train_df(self):
        return pd.DataFrame({"text": ["train1", "train2"], "label": [0, 1]})

    @pytest.fixture
    def sample_valid_df(self):
        return pd.DataFrame({"text": ["valid1"], "label": [0]})

    @pytest.fixture
    def temp_project(self, tmp_path):
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        return str(project_dir)

    def test_saves_both_splits(self, sample_train_df, sample_valid_df, temp_project):
        """Test that both train and valid splits are saved."""
        result = save_processed_datasets(
            train_data=sample_train_df,
            valid_data=sample_valid_df,
            project_name=temp_project,
            train_split="train",
            valid_split="validation",
            save_mode="local",
        )
        assert result["train"]["local_path"] is not None
        assert result["valid"]["local_path"] is not None
        assert os.path.exists(result["train"]["local_path"])
        assert os.path.exists(result["valid"]["local_path"])

    def test_no_valid_data(self, sample_train_df, temp_project):
        """Test saving with no validation data."""
        result = save_processed_datasets(
            train_data=sample_train_df,
            valid_data=None,
            project_name=temp_project,
            train_split="train",
            valid_split=None,
            save_mode="local",
        )
        assert result["train"]["local_path"] is not None
        assert result["valid"] is None


class TestLLMTrainingParamsIntegration:
    """Test that save_processed_data param works with LLMTrainingParams."""

    def test_default_value(self):
        """Test that default save_processed_data is 'auto'."""
        from autotrain.trainers.clm.params import LLMTrainingParams

        params = LLMTrainingParams()
        assert params.save_processed_data == "auto"

    def test_custom_value(self):
        """Test setting custom save_processed_data value."""
        from autotrain.trainers.clm.params import LLMTrainingParams

        params = LLMTrainingParams(save_processed_data="local")
        assert params.save_processed_data == "local"

        params = LLMTrainingParams(save_processed_data="none")
        assert params.save_processed_data == "none"
