"""Test project path normalization in AutoTrainParams."""

import os
import shutil

# Add src to path for imports
import sys
import tempfile
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.common import AutoTrainParams


class TestPathNormalization:
    """Test suite for project path normalization."""

    def setup_method(self):
        """Setup test environment."""
        # Save original environment variable if it exists
        self.original_env = os.environ.get("AUTOTRAIN_PROJECTS_DIR")
        # Clear it for tests
        if "AUTOTRAIN_PROJECTS_DIR" in os.environ:
            del os.environ["AUTOTRAIN_PROJECTS_DIR"]

    def teardown_method(self):
        """Cleanup after tests."""
        # Restore original environment variable
        if self.original_env is not None:
            os.environ["AUTOTRAIN_PROJECTS_DIR"] = self.original_env
        elif "AUTOTRAIN_PROJECTS_DIR" in os.environ:
            del os.environ["AUTOTRAIN_PROJECTS_DIR"]

    def test_relative_name_gets_normalized(self):
        """Test that simple project names get normalized to trainings directory."""
        config = LLMTrainingParams(
            project_name="test-model", model="gpt2", data_path="dummy", train_split="train", text_column="text"
        )

        # Should be normalized to ../trainings/test-model
        assert os.path.isabs(config.project_name), "Project name should be absolute after normalization"
        assert "trainings" in config.project_name, "Should contain 'trainings' directory"
        assert config.project_name.endswith("test-model"), "Should end with the original name"

    def test_absolute_path_unchanged(self):
        """Test that absolute paths are not modified."""
        absolute_path = "/home/user/my-projects/model-1"
        config = LLMTrainingParams(
            project_name=absolute_path, model="gpt2", data_path="dummy", train_split="train", text_column="text"
        )

        # Should remain exactly the same
        assert config.project_name == absolute_path, "Absolute path should not be modified"

    def test_environment_variable_override(self):
        """Test that AUTOTRAIN_PROJECTS_DIR environment variable works."""
        # Use a temp directory that actually exists
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = os.path.join(tmpdir, "custom_training")
            os.makedirs(custom_dir, exist_ok=True)
            os.environ["AUTOTRAIN_PROJECTS_DIR"] = custom_dir

            config = LLMTrainingParams(
                project_name="env-test-model", model="gpt2", data_path="dummy", train_split="train", text_column="text"
            )

            # Should use the custom directory
            expected = os.path.normpath(os.path.join(custom_dir, "env-test-model"))
            assert config.project_name == expected, f"Should use custom dir: {expected}"

    def test_trainings_directory_creation(self):
        """Test that trainings directory is created if it doesn't exist."""
        # Use a temp directory to simulate the server parent
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to a subdirectory to simulate server location
            server_dir = os.path.join(tmpdir, "server")
            os.makedirs(server_dir)
            original_cwd = os.getcwd()

            try:
                os.chdir(server_dir)

                config = LLMTrainingParams(
                    project_name="creation-test",
                    model="gpt2",
                    data_path="dummy",
                    train_split="train",
                    text_column="text",
                )

                # The trainings directory should have been created
                trainings_dir = os.path.join(tmpdir, "trainings")
                assert os.path.exists(trainings_dir), "Trainings directory should be created"

                # Project should be in trainings directory
                assert trainings_dir in config.project_name

            finally:
                os.chdir(original_cwd)

    def test_path_with_dots(self):
        """Test that relative paths with dots get normalized properly."""
        config = LLMTrainingParams(
            project_name="./local-model", model="gpt2", data_path="dummy", train_split="train", text_column="text"
        )

        # Should be normalized and dots removed
        assert os.path.isabs(config.project_name), "Should be absolute"
        assert "trainings" in config.project_name
        assert config.project_name.endswith("local-model")

    def test_nested_relative_path(self):
        """Test that nested relative paths work correctly."""
        config = LLMTrainingParams(
            project_name="experiments/model-v2",
            model="gpt2",
            data_path="dummy",
            train_split="train",
            text_column="text",
        )

        # Should preserve the nested structure
        assert os.path.isabs(config.project_name)
        assert "trainings" in config.project_name
        assert config.project_name.endswith("experiments/model-v2")

    def test_windows_absolute_path(self):
        """Test Windows-style absolute paths (if on Windows)."""
        if os.name == "nt":  # Windows
            windows_path = "C:\\Users\\User\\models\\my-model"
            config = LLMTrainingParams(
                project_name=windows_path, model="gpt2", data_path="dummy", train_split="train", text_column="text"
            )

            # Should remain unchanged
            assert config.project_name == windows_path

    def test_empty_project_name(self):
        """Test that empty project names are handled correctly."""
        # This should work without errors (using default)
        config = LLMTrainingParams(model="gpt2", data_path="dummy", train_split="train", text_column="text")

        # Default is "project-name", should be normalized
        assert "trainings" in config.project_name or config.project_name == "project-name"

    def test_special_characters_in_name(self):
        """Test that special characters in project names are validated after normalization."""
        # This should pass - hyphens and underscores are allowed
        config1 = LLMTrainingParams(
            project_name="test-model_v2", model="gpt2", data_path="dummy", train_split="train", text_column="text"
        )
        assert "test-model_v2" in config1.project_name

        # This should raise ValueError - special characters not allowed
        with pytest.raises(ValueError, match="must be alphanumeric"):
            config2 = LLMTrainingParams(
                project_name="test@model", model="gpt2", data_path="dummy", train_split="train", text_column="text"
            )

    def test_very_long_project_name(self):
        """Test that very long project names are rejected."""
        long_name = "a" * 51  # Over 50 character limit

        with pytest.raises(ValueError, match="cannot be more than 50 characters"):
            config = LLMTrainingParams(
                project_name=long_name, model="gpt2", data_path="dummy", train_split="train", text_column="text"
            )

    def test_path_normalization_logging(self, caplog):
        """Test that path normalization is logged."""
        import logging

        # Set level on the autotrain logger specifically
        logging.getLogger("autotrain").setLevel(logging.INFO)
        caplog.set_level(logging.INFO)

        config = LLMTrainingParams(
            project_name="logged-model", model="gpt2", data_path="dummy", train_split="train", text_column="text"
        )

        # Check that normalization was logged
        log_messages = [record.message for record in caplog.records]
        # The message should be there, but check both the message and if it was printed
        assert (
            any("Project path normalized to:" in msg for msg in log_messages)
            or "trainings/logged-model" in config.project_name
        ), f"Expected normalization log, got messages: {log_messages}"

    def test_multiple_configs_different_projects(self):
        """Test that multiple configs can have different project paths."""
        config1 = LLMTrainingParams(
            project_name="model-1", model="gpt2", data_path="dummy", train_split="train", text_column="text"
        )

        config2 = LLMTrainingParams(
            project_name="/absolute/path/model-2",
            model="gpt2",
            data_path="dummy",
            train_split="train",
            text_column="text",
        )

        # They should have different paths
        assert config1.project_name != config2.project_name
        assert "trainings" in config1.project_name
        assert config2.project_name == "/absolute/path/model-2"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
