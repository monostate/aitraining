"""
Tests for interactive wizard functionality.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest

from autotrain.cli.interactive_wizard import InteractiveWizard, run_wizard
from autotrain.metadata.catalog import CatalogEntry
from autotrain.trainers.clm.params import LLMTrainingParams


@pytest.fixture(autouse=True)
def clear_hf_token(monkeypatch):
    """Ensure tests control whether the wizard sees an HF token."""
    monkeypatch.delenv("HF_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def stable_catalog(monkeypatch):
    """Provide deterministic catalog entries unless a test overrides them."""

    def fake_models(trainer_type, trainer_variant=None, sort_by="trending", search_query=None):
        key = trainer_type or "llm"
        if trainer_type == "llm":
            key = f"llm:{trainer_variant or 'sft'}"
        mapping = {
            "llm:sft": [CatalogEntry("meta-llama/Llama-3.2-1B", "Llama 3.2 1B")],
            "llm:dpo": [CatalogEntry("meta-llama/Llama-3.2-1B", "Llama 3.2 1B")],
            "llm:orpo": [CatalogEntry("meta-llama/Llama-3.2-1B", "Llama 3.2 1B")],
            "llm:ppo": [CatalogEntry("meta-llama/Llama-3.2-1B", "Llama 3.2 1B")],
            "text-classification": [CatalogEntry("bert-base-uncased", "BERT Base")],
            "token-classification": [CatalogEntry("dslim/bert-base-NER", "BERT NER")],
            "tabular": [CatalogEntry("xgboost", "XGBoost")],
            "image-classification": [CatalogEntry("google/vit-base-patch16-224", "ViT Base")],
            "image-regression": [CatalogEntry("google/vit-base-patch16-224", "ViT Base")],
            "seq2seq": [CatalogEntry("t5-small", "T5 Small")],
            "extractive-qa": [CatalogEntry("bert-base-uncased", "BERT Base")],
            "sent-transformers": [CatalogEntry("sentence-transformers/all-MiniLM-L6-v2", "MiniLM")],
            "vlm": [CatalogEntry("HuggingFaceM4/idefics2-8b", "Idefics2 8B")],
        }
        return mapping.get(key, [CatalogEntry("google/gemma-3-270m", "Gemma 3 270M")])

    def fake_datasets(trainer_type, trainer_variant=None, sort_by="trending", search_query=None):
        key = trainer_type or "llm"
        if trainer_type == "llm":
            key = f"llm:{trainer_variant or 'sft'}"
        mapping = {
            "llm:sft": [CatalogEntry("tatsu-lab/alpaca", "Alpaca")],
            "llm:dpo": [CatalogEntry("argilla/ultrafeedback-binarized-preferences", "UltraFeedback")],
            "text-classification": [CatalogEntry("ag_news", "AG News")],
            "token-classification": [CatalogEntry("conll2003", "CoNLL 2003")],
            "image-classification": [CatalogEntry("huggingface/cifar10", "CIFAR-10")],
            "seq2seq": [CatalogEntry("cnn_dailymail", "CNN/DailyMail")],
        }
        return mapping.get(key, [CatalogEntry("data", "Local data placeholder")])

    monkeypatch.setattr(
        "autotrain.cli.interactive_wizard.get_popular_models",
        fake_models,
    )
    monkeypatch.setattr(
        "autotrain.cli.interactive_wizard.get_popular_datasets",
        fake_datasets,
    )


class TestInteractiveWizard:
    """Test cases for InteractiveWizard class."""

    def test_wizard_initialization_no_args(self):
        """Test wizard initialization with no arguments."""
        wizard = InteractiveWizard(trainer_type="llm")
        assert wizard.answers == {}
        assert wizard.trainer == "sft"

    def test_wizard_initialization_with_args(self):
        """Test wizard initialization with initial arguments."""
        initial_args = {
            "trainer": "dpo",
            "model": "meta-llama/Llama-3.2-1B",
            "project_name": "test-project",
        }
        wizard = InteractiveWizard(initial_args, trainer_type="llm")
        assert wizard.answers == initial_args
        assert wizard.trainer == "dpo"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_sft_happy_path(self, mock_isatty, mock_input):
        """Test complete wizard flow for SFT training (happy path)."""
        # Simulate user inputs for a complete SFT configuration
        # Flow: HF token -> trainer -> project -> MODEL -> dataset -> advanced
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "1",  # Select SFT trainer
            "my-sft-project",  # Project name
            "1",  # Select first model from catalog
            "1",  # Select first dataset from catalog
            "",  # Train split (auto-detected)
            "",  # No validation split
            "",  # Max samples (skip)
            "text",  # Text column
            "n",  # Skip advanced parameters
        ]

        wizard = InteractiveWizard(trainer_type="llm")

        # Mock the validation to avoid actual project creation
        with patch.object(wizard, "_show_summary_and_confirm", return_value=True):
            result = wizard.run()

        # Verify basic configuration
        assert result["trainer"] == "sft"
        assert result["project_name"] == "my-sft-project"
        # Model and data come from catalog, just verify they exist
        assert "model" in result
        assert "data_path" in result
        assert result["train_split"] == "train"
        assert result["text_column"] == "text"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_dpo_configuration(self, mock_isatty, mock_input):
        """Test wizard flow for DPO training with column mapping."""
        # Flow: HF token -> trainer -> project -> MODEL -> dataset -> columns
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "2",  # Select DPO trainer
            "dpo-project",  # Project name
            "1",  # Select first model from catalog
            "hf-datasets/preference-data",  # Data path (custom)
            "train",  # Train split
            "test",  # Validation split
            "",  # Max samples
            "prompt",  # Prompt column
            "chosen",  # Chosen column
            "rejected",  # Rejected column
            "n",  # Skip advanced parameters
        ]

        wizard = InteractiveWizard(trainer_type="llm")

        with patch.object(wizard, "_show_summary_and_confirm", return_value=True):
            result = wizard.run()

        assert result["trainer"] == "dpo"
        assert result["project_name"] == "dpo-project"
        assert result["prompt_text_column"] == "prompt"
        assert result["text_column"] == "chosen"
        assert result["rejected_text_column"] == "rejected"
        assert result["valid_split"] == "test"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_ppo_with_reward_model(self, mock_isatty, mock_input):
        """Test wizard flow for PPO training with reward model."""
        # Flow: HF token -> trainer -> project -> MODEL -> dataset -> text_column -> reward_model
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "4",  # Select PPO trainer
            "ppo-rl-project",  # Project name
            "1",  # Select first model from catalog
            "./rl_data",  # Data path
            "train",  # Train split
            "",  # No validation split
            "",  # Max samples (skip)
            "text",  # Text column
            "models/reward-model",  # Reward model path
            "n",  # Skip advanced parameters
        ]

        wizard = InteractiveWizard(trainer_type="llm")

        with patch.object(wizard, "_show_summary_and_confirm", return_value=True):
            result = wizard.run()

        assert result["trainer"] == "ppo"
        assert result["project_name"] == "ppo-rl-project"
        assert result["rl_reward_model_path"] == "models/reward-model"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_custom_model_selection(self, mock_isatty, mock_input):
        """Test selecting a custom model not in the starter list."""
        # Flow: HF token -> trainer -> project -> MODEL (custom) -> dataset
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "1",  # Select SFT
            "custom-model-project",  # Project name
            "my-org/custom-model-7b",  # Custom model directly
            "./data",  # Data path
            "train",  # Train split
            "",  # No validation split
            "",  # Max samples (skip)
            "text",  # Text column
            "n",  # Skip advanced parameters
        ]

        wizard = InteractiveWizard(trainer_type="llm")

        with patch.object(wizard, "_show_summary_and_confirm", return_value=True):
            result = wizard.run()

        assert result["model"] == "my-org/custom-model-7b"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_back_navigation(self, mock_isatty, mock_input):
        """Users can go back to previous steps with :back."""
        # Flow: HF token -> trainer -> project -> MODEL (:back happens here)
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "1",  # Select SFT
            "project-one",  # Initial project
            ":back",  # At model prompt, go back to project
            "project-two",  # Re-enter project name
            "1",  # Select first model
            "./data/train.jsonl",  # Dataset path
            "train",  # Train split
            "",  # No validation split
            "",  # Max samples (skip)
            "text",  # Text column
            "n",  # Skip advanced
        ]

        wizard = InteractiveWizard(trainer_type="llm")
        with patch.object(wizard, "_show_summary_and_confirm", return_value=True):
            result = wizard.run()

        assert result["project_name"] == "project-two"

    @patch("autotrain.cli.interactive_wizard.get_popular_models")
    @patch("autotrain.cli.interactive_wizard.get_popular_datasets")
    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_catalog_dataset_selection(self, mock_isatty, mock_input, mock_datasets, mock_models):
        """Selecting datasets via numeric catalog choices fills data_path."""
        # Mock with the new signature
        mock_datasets.side_effect = lambda *args, **kwargs: [
            CatalogEntry("foo/bar", "FooBar Dataset"),
            CatalogEntry("baz/qux", "Baz Dataset"),
        ]
        mock_models.side_effect = lambda *args, **kwargs: [CatalogEntry("model/a", "Model A")]
        mock_input.side_effect = [
            "",  # HF token
            "1",  # SFT
            "catalog-project",
            "1",  # Choose first MODEL (model comes before dataset)
            "1",  # Choose first dataset
            "train",
            "",  # Valid split
            "",  # Max samples
            "text",
            "n",  # Skip advanced
        ]

        wizard = InteractiveWizard(trainer_type="llm")
        with patch.object(wizard, "_show_summary_and_confirm", return_value=True):
            result = wizard.run()

        assert result["data_path"] == "foo/bar"
        assert result["model"] == "model/a"

    @patch("autotrain.cli.interactive_wizard.get_popular_models")
    @patch("autotrain.cli.interactive_wizard.get_popular_datasets")
    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_catalog_model_numeric_selection(self, mock_isatty, mock_input, mock_datasets, mock_models):
        """Numeric model selections apply the catalog entry."""
        mock_datasets.side_effect = lambda *args, **kwargs: [CatalogEntry("dataset/a", "Dataset A")]
        mock_models.side_effect = lambda *args, **kwargs: [
            CatalogEntry("model/a", "Model A"),
            CatalogEntry("model/b", "Model B"),
        ]
        mock_input.side_effect = [
            "",  # HF token
            "1",  # SFT
            "catalog-model-project",
            "2",  # Choose second MODEL (model comes first)
            "1",  # Dataset via catalog
            "train",
            "",  # Valid split
            "",  # Max samples
            "text",
            "n",  # Skip advanced
        ]

        wizard = InteractiveWizard(trainer_type="llm")
        with patch.object(wizard, "_show_summary_and_confirm", return_value=True):
            result = wizard.run()

        assert result["data_path"] == "dataset/a"
        assert result["model"] == "model/b"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_cancellation(self, mock_isatty, mock_input):
        """Test wizard handles cancellation gracefully."""
        # Simulate user pressing Ctrl+C
        mock_input.side_effect = KeyboardInterrupt()

        wizard = InteractiveWizard(trainer_type="llm")

        with pytest.raises(SystemExit):
            wizard.run()

    @patch("builtins.input")
    def test_get_yes_no_true(self, mock_input):
        """Test _get_yes_no returns True for affirmative responses."""
        wizard = InteractiveWizard(trainer_type="llm")

        for response in ["y", "yes", "Y", "YES", "1", "true"]:
            mock_input.return_value = response
            assert wizard._get_yes_no("Test?", default=False) is True

    @patch("builtins.input")
    def test_get_yes_no_false(self, mock_input):
        """Test _get_yes_no returns False for negative responses."""
        wizard = InteractiveWizard(trainer_type="llm")

        for response in ["n", "no", "N", "NO", "0", "false"]:
            mock_input.return_value = response
            assert wizard._get_yes_no("Test?", default=True) is False

    @patch("builtins.input")
    def test_get_yes_no_default(self, mock_input):
        """Test _get_yes_no returns default for empty input."""
        wizard = InteractiveWizard(trainer_type="llm")

        mock_input.return_value = ""
        assert wizard._get_yes_no("Test?", default=True) is True
        assert wizard._get_yes_no("Test?", default=False) is False

    @patch("builtins.input")
    def test_get_input_with_value(self, mock_input):
        """Test _get_input returns user value when provided."""
        wizard = InteractiveWizard(trainer_type="llm")
        mock_input.return_value = "user_value"

        result = wizard._get_input("Enter value:", "default")
        assert result == "user_value"

    @patch("builtins.input")
    def test_get_input_default(self, mock_input):
        """Test _get_input returns default when user presses Enter."""
        wizard = InteractiveWizard(trainer_type="llm")
        mock_input.return_value = ""

        result = wizard._get_input("Enter value:", "default_value")
        assert result == "default_value"


class TestRunWizardFunction:
    """Test the run_wizard convenience function."""

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_run_wizard_returns_config(self, mock_isatty, mock_input):
        """Test run_wizard returns valid configuration."""
        # Flow: HF token -> trainer -> project -> MODEL -> dataset
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "1",  # SFT
            "wizard-test-project",
            "1",  # First model
            "./data",
            "train",
            "",  # Valid split
            "",  # Max samples
            "text",
            "n",  # No advanced
            "y",  # Confirm
        ]

        config = run_wizard(trainer_type="llm")

        assert isinstance(config, dict)
        assert config["trainer"] == "sft"
        assert config["project_name"] == "wizard-test-project"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_run_wizard_with_initial_args(self, mock_isatty, mock_input):
        """Test run_wizard merges initial args with wizard inputs."""
        initial_args = {
            "trainer": "sft",
            "backend": "local",
        }

        mock_input.side_effect = [
            "",  # Skip HF token prompt
            # Trainer already set, skip that step
            "preset-project",
            "1",  # First model
            "./preset-data",
            "train",
            "",  # Valid split
            "",  # Max samples
            "text",
            "n",  # No advanced
            "y",  # Confirm
        ]

        config = run_wizard(initial_args, trainer_type="llm")

        assert config["trainer"] == "sft"
        assert config["backend"] == "local"  # Preserved from initial
        assert config["project_name"] == "preset-project"  # From wizard


class TestWizardValidation:
    """Test wizard validation with Pydantic."""

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_produces_valid_params(self, mock_isatty, mock_input):
        """Test that wizard output can be used to create LLMTrainingParams."""
        # Flow: HF token -> trainer -> project -> MODEL -> dataset
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "1",  # SFT
            "valid-params-test",
            "1",  # First model
            "./data/valid",
            "train",
            "",  # Valid split
            "",  # Max samples
            "text",
            "n",  # Skip advanced
            "y",  # Confirm
        ]

        config = run_wizard(trainer_type="llm")

        # Apply merge_adapter handling like CLI does
        if "merge_adapter" in config and config["merge_adapter"] is None:
            del config["merge_adapter"]

        # This should not raise ValidationError
        params = LLMTrainingParams(**config)

        assert params.trainer == "sft"
        assert os.path.basename(params.project_name) == "valid-params-test"
        assert params.data_path == "./data/valid"


class TestWizardIntegration:
    """Integration tests for wizard with AutoTrainProject."""

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    @patch("autotrain.project.AutoTrainProject.create")
    def test_wizard_to_project_creation(self, mock_create, mock_isatty, mock_input):
        """Test complete flow from wizard to project creation."""
        mock_create.return_value = "test-job-id-12345"

        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "1",  # SFT
            "integration-test",
            "1",  # First model
            "./test-data",
            "train",
            "",  # Valid split
            "",  # Max samples
            "text",
            "n",  # Skip advanced
            "y",  # Confirm
        ]

        # Run wizard
        config = run_wizard(trainer_type="llm")

        # Apply merge_adapter handling like CLI does
        if "merge_adapter" in config and config["merge_adapter"] is None:
            del config["merge_adapter"]

        # Create params
        params = LLMTrainingParams(**config)

        # Create project (mocked)
        from autotrain.project import AutoTrainProject

        project = AutoTrainProject(params=params, backend="local", process=False)
        job_id = project.create()

        assert job_id == "test-job-id-12345"
        mock_create.assert_called_once()

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_dpo_validation(self, mock_isatty, mock_input):
        """Test that DPO wizard configuration passes validation."""
        # Flow: HF token -> trainer -> project -> MODEL -> dataset -> columns
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "2",  # DPO
            "dpo-validation-test",
            "1",  # First model
            "./dpo-data",
            "train",
            "",  # Valid split
            "",  # Max samples
            "prompt",
            "chosen",
            "rejected",
            "n",  # Skip advanced
            "y",  # Confirm
        ]

        config = run_wizard(trainer_type="llm")

        # Apply merge_adapter handling like CLI does
        if "merge_adapter" in config and config["merge_adapter"] is None:
            del config["merge_adapter"]

        # Should not raise validation error for DPO requirements
        params = LLMTrainingParams(**config)

        assert params.trainer == "dpo"
        assert params.prompt_text_column == "prompt"
        assert params.rejected_text_column == "rejected"


class TestWizardCLIIntegration:
    """Test wizard integration with CLI entry points."""

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    @patch("autotrain.project.AutoTrainProject.create")
    def test_wizard_from_base_command(self, mock_create, mock_isatty, mock_input):
        """Test wizard launched from base command (no subcommand)."""
        mock_create.return_value = "job-12345"

        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "1",  # SFT
            "base-cmd-test",
            "1",  # First model
            "./data",
            "train",
            "",  # Valid split
            "",  # Max samples
            "text",
            "n",  # Skip advanced
            "y",  # Confirm
        ]

        # Simulate the flow in autotrain.py
        from autotrain.cli.interactive_wizard import run_wizard
        from autotrain.project import AutoTrainProject
        from autotrain.trainers.clm.params import LLMTrainingParams

        # Run wizard
        config = run_wizard(trainer_type="llm")
        config["backend"] = config.get("backend", "local")

        # Apply merge_adapter handling like CLI does
        if "merge_adapter" in config and config["merge_adapter"] is None:
            del config["merge_adapter"]

        # Create params and project (as done in autotrain.py)
        params = LLMTrainingParams(**config)
        project = AutoTrainProject(params=params, backend=config["backend"], process=False)
        job_id = project.create()

        assert job_id == "job-12345"
        assert config["trainer"] == "sft"
        assert config["project_name"] == "base-cmd-test"
        mock_create.assert_called_once()

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_from_llm_interactive_flag(self, mock_isatty, mock_input):
        """Test wizard launched via --interactive flag."""
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "1",  # SFT
            "interactive-flag-test",
            "1",  # First model
            "./data",
            "train",
            "",  # Valid split
            "",  # Max samples
            "text",
            "n",  # Skip advanced
            "y",  # Confirm
        ]

        # Simulate the flow in run_llm.py with --interactive flag
        from autotrain.cli.interactive_wizard import run_wizard

        # Start with minimal args (as would be passed from CLI)
        initial_args = {
            "interactive": True,
            "backend": "local",
        }

        config = run_wizard(initial_args, trainer_type="llm")

        assert config["trainer"] == "sft"
        assert config["project_name"] == "interactive-flag-test"
        assert config["backend"] == "local"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_auto_launch_on_missing_params(self, mock_isatty, mock_input):
        """Test wizard auto-launches when --train has missing params."""
        # Flow: HF token -> trainer -> project -> MODEL -> dataset -> splits -> columns -> advanced
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "1",  # Trainer selection (sft)
            "auto-launch-test",  # Project name
            "1",  # Model selection from catalog
            # Dataset config: data_path already set, so only prompts for splits + columns
            "train",  # Train split
            "",  # Valid split (skip)
            "",  # Max samples (skip)
            "text",  # Text column
            "n",  # Skip advanced params
            "y",  # Confirm
        ]

        from autotrain.cli.interactive_wizard import run_wizard

        # Simulate args from: aitraining llm --train
        # These are placeholder values that should trigger wizard prompts
        initial_args = {
            "train": True,
            "backend": "local",
            "model": "google/gemma-3-270m",  # Placeholder - will prompt
            "project_name": "project-name",  # Placeholder - will prompt
            "data_path": "data",  # Placeholder - but won't prompt (already in answers)
        }

        config = run_wizard(initial_args, trainer_type="llm")

        # Wizard should have collected these
        assert config["project_name"] == "auto-launch-test"
        # Data path will be selected from catalog (first entry) since data_path was already set
        assert "data_path" in config
        # Model will be selected from catalog (first entry)
        assert config["model"] == "meta-llama/Llama-3.2-1B"


class TestWizardEdgeCases:
    """Test edge cases and error handling."""

    @patch("builtins.input")
    def test_wizard_handles_eof(self, mock_input):
        """Test wizard handles EOF gracefully."""
        mock_input.side_effect = EOFError()

        wizard = InteractiveWizard(trainer_type="llm")

        with pytest.raises(SystemExit):
            wizard.run()

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_retries_on_invalid_trainer_choice(self, mock_isatty, mock_input):
        """Test wizard retries when invalid trainer choice is entered."""
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "99",  # Invalid choice
            "abc",  # Invalid choice
            "1",  # Valid choice (SFT)
            "retry-test",
            "1",  # First model
            "./data",
            "train",
            "",  # Valid split
            "",  # Max samples
            "text",
            "n",  # Skip advanced
        ]

        wizard = InteractiveWizard(trainer_type="llm")

        with patch.object(wizard, "_show_summary_and_confirm", return_value=True):
            result = wizard.run()

        assert result["trainer"] == "sft"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    @patch("os.listdir")
    @patch("os.path.exists")
    def test_wizard_project_name_collision_detection(self, mock_exists, mock_listdir, mock_isatty, mock_input):
        """Test wizard detects project name collisions and auto-versions."""

        # Simulate that my-project and my-project-v2 exist, but my-project-v3 doesn't
        # Use exact path matching to avoid infinite loops
        def path_exists(p):
            # Only return True for exact matches
            return p == "my-project" or p == "my-project-v2"

        mock_exists.side_effect = path_exists
        # Mock listdir to return non-empty for existing dirs
        mock_listdir.side_effect = lambda p: ["file.txt"] if path_exists(p) else []

        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "1",  # SFT
            "my-project",  # Project name that already exists
            "y",  # Yes, use the suggested name my-project-v3
            "1",  # First model
            "./data",
            "train",
            "",  # Valid split
            "",  # Max samples
            "text",
            "n",  # Skip advanced
        ]

        wizard = InteractiveWizard(trainer_type="llm")

        with patch.object(wizard, "_show_summary_and_confirm", return_value=True):
            result = wizard.run()

        # Verify the project name was auto-versioned to avoid collision
        assert result["project_name"] == "my-project-v3"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_interactive_command_loop(self, mock_isatty, mock_input):
        """Test wizard interactive command loop for model/dataset selection."""
        # Flow: HF token -> trainer -> project -> MODEL -> dataset -> splits -> columns -> advanced
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "1",  # SFT
            "command-loop-test",
            "1",  # Select first model from catalog
            "1",  # Select first dataset from catalog (tatsu-lab/alpaca)
            # "train" no longer needed - auto-selected since alpaca only has train split
            "",  # No validation split
            "",  # Max samples
            "n",  # Skip dataset conversion (alpaca format detected)
            "text",  # Text column (still prompted even for alpaca)
        ] + [
            "n"
        ] * 20  # Skip all advanced param groups

        wizard = InteractiveWizard(trainer_type="llm")

        # Mock the confirmation to avoid the complex advanced params flow
        with patch.object(wizard, "_show_summary_and_confirm", return_value=True):
            result = wizard.run()

        # Verify basic selection worked
        assert result["project_name"] == "command-loop-test"
        assert "data_path" in result
        assert "model" in result
        assert result["text_column"] == "text"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    @patch("datasets.load_dataset")
    def test_wizard_dataset_validation(self, mock_load_dataset, mock_isatty, mock_input):
        """Test wizard validates dataset and detects columns."""
        # Mock dataset with specific columns
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = MagicMock(column_names=["text", "label", "other"])
        mock_load_dataset.return_value = mock_dataset

        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "1",  # SFT
            "dataset-validation-test",
            "1",  # First model
            "hub/dataset",  # HuggingFace dataset
            "train",
            "",  # Valid split
            "",  # Max samples
            "",  # Auto-detect text column
            "n",  # Skip advanced
        ]

        wizard = InteractiveWizard(trainer_type="llm")

        with patch.object(wizard, "_show_summary_and_confirm", return_value=True):
            result = wizard.run()

        # Verify dataset was loaded for validation
        mock_load_dataset.assert_called()

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_wizard_help_command(self, mock_isatty, mock_input):
        """Test wizard :help command displays help text."""
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "1",  # SFT
            ":help",  # Ask for help
            "help-project",  # Continue with project name
            "1",  # First model
            "./data",
            "train",
            "",  # Valid split
            "",  # Max samples
            "text",
            "n",  # Skip advanced
        ]

        wizard = InteractiveWizard(trainer_type="llm")

        with patch.object(wizard, "_show_summary_and_confirm", return_value=True), patch(
            "builtins.print"
        ) as mock_print:
            result = wizard.run()

            # Verify help was displayed
            help_calls = [
                call for call in mock_print.call_args_list if ":help" in str(call) or "Commands:" in str(call)
            ]
            assert len(help_calls) > 0


class TestNonLLMTrainers:
    """Test wizard for non-LLM trainer types."""

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_text_classification_wizard(self, mock_isatty, mock_input):
        """Test wizard flow for text classification."""
        # Non-LLM trainers also have: trainer -> project -> MODEL -> dataset -> columns -> advanced
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "2",  # Select text-classification
            "text-clf-project",  # Project name
            "1",  # Select first model (bert-base-uncased)
            "./clf_data",  # Data path
            "train",  # Train split
            "",  # No validation split
            "",  # Max samples (skip)
            "text",  # Text column
            "label",  # Target column
            "n",  # Skip advanced parameters
            "y",  # Confirm
        ]

        wizard = InteractiveWizard()

        result = wizard.run()

        assert result["project_name"] == "text-clf-project"
        assert result["text_column"] == "text"
        assert result["target_column"] == "label"
        assert result["model"] == "bert-base-uncased"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_tabular_wizard(self, mock_isatty, mock_input):
        """Test wizard flow for tabular data."""
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "4",  # Select tabular
            "tabular-project",  # Project name
            "1",  # Select first model (xgboost)
            "./tabular_data.csv",  # Data path
            "train",  # Train split
            "",  # No validation split
            "",  # Max samples (skip)
            "target",  # Target column
            "n",  # Skip advanced parameters
            "y",  # Confirm
        ]

        wizard = InteractiveWizard()

        result = wizard.run()

        assert result["project_name"] == "tabular-project"
        assert result["target_columns"] == "target"
        assert result["model"] == "xgboost"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_token_classification_wizard(self, mock_isatty, mock_input):
        """Test wizard flow for token classification (NER)."""
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "3",  # Select token-classification
            "ner-project",  # Project name
            "1",  # Select first model
            "./ner_data",  # Data path
            "train",  # Train split
            "",  # No validation split
            "",  # Max samples (skip)
            "tokens",  # Tokens column
            "tags",  # Tags column
            "n",  # Skip advanced
            "y",  # Confirm
        ]

        wizard = InteractiveWizard()

        result = wizard.run()

        assert result["project_name"] == "ner-project"
        assert result["tokens_column"] == "tokens"
        assert result["tags_column"] == "tags"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_image_classification_wizard(self, mock_isatty, mock_input):
        """Test wizard flow for image classification."""
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "5",  # Select image-classification
            "img-clf-project",  # Project name
            "1",  # Select first model (vit)
            "./images",  # Data path
            "train",  # Train split
            "val",  # Validation split
            "",  # Max samples
            "image",  # Image column
            "label",  # Target column
            "n",  # Skip advanced
            "y",  # Confirm
        ]

        wizard = InteractiveWizard()

        result = wizard.run()

        assert result["project_name"] == "img-clf-project"
        assert result["image_column"] == "image"
        assert result["target_column"] == "label"
        assert result["model"] == "google/vit-base-patch16-224"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_seq2seq_wizard(self, mock_isatty, mock_input):
        """Test wizard flow for seq2seq."""
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "7",  # Select seq2seq
            "seq2seq-project",  # Project name
            "1",  # Select first model (t5-small)
            "./seq2seq_data",  # Data path
            "train",  # Train split
            "",  # No validation split
            "",  # Max samples (skip)
            "source",  # Text column
            "target",  # Target column
            "n",  # Skip advanced
            "y",  # Confirm
        ]

        wizard = InteractiveWizard()

        result = wizard.run()

        assert result["project_name"] == "seq2seq-project"
        assert result["text_column"] == "source"
        assert result["target_column"] == "target"
        assert result["model"] == "t5-small"

    @patch("builtins.input")
    @patch("sys.stdout.isatty", return_value=True)
    def test_extractive_qa_wizard(self, mock_isatty, mock_input):
        """Test wizard flow for extractive QA."""
        mock_input.side_effect = [
            "",  # Skip HF token prompt
            "8",  # Select extractive-qa
            "qa-project",  # Project name
            "1",  # Select first model
            "./qa_data",  # Data path
            "train",  # Train split
            "",  # No validation split
            "",  # Max samples (skip)
            "context",  # Text column
            "question",  # Question column
            "answers",  # Answer column
            "n",  # Skip advanced
            "y",  # Confirm
        ]

        wizard = InteractiveWizard()

        result = wizard.run()

        assert result["project_name"] == "qa-project"
        assert result["text_column"] == "context"
        assert result["question_column"] == "question"
        assert result["answer_column"] == "answers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
