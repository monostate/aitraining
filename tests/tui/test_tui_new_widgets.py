"""Tests for new TUI widgets (StatusBar, RunPreview, TokensModal, JsonViewer)."""

import asyncio
import json
import os
import sys


# Ensure src is importable when running tests directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from autotrain.cli.tui.app import AITrainingTUI
from autotrain.cli.tui.widgets.catalog_panel import CatalogPanel
from autotrain.cli.tui.widgets.json_viewer import JsonViewer
from autotrain.cli.tui.widgets.run_preview import RunPreview
from autotrain.cli.tui.widgets.status_bar import StatusBar
from autotrain.cli.tui.widgets.tokens_modal import TokensModal


class TestStatusBar:
    """Test StatusBar widget functionality."""

    def test_status_bar_initialization(self):
        """Test status bar initializes correctly."""
        status_bar = StatusBar()
        assert status_bar.trainer == "default"
        assert status_bar.model == "google/gemma-3-270m"  # Default model
        assert status_bar.dataset is None

    def test_status_bar_update(self):
        """Test status bar updates correctly."""
        status_bar = StatusBar()
        status_bar.update_status(trainer="sft", model="gpt2", dataset="imdb")
        assert status_bar.trainer == "sft"
        assert status_bar.model == "gpt2"
        assert status_bar.dataset == "imdb"

    def test_status_bar_partial_update(self):
        """Test status bar partial updates."""
        status_bar = StatusBar()
        status_bar.update_status(trainer="dpo")
        assert status_bar.trainer == "dpo"
        assert status_bar.model == "google/gemma-3-270m"  # Default model persists

        status_bar.update_status(model="llama2")
        assert status_bar.trainer == "dpo"
        assert status_bar.model == "llama2"


class TestRunPreview:
    """Test RunPreview widget functionality."""

    def test_run_preview_initialization(self):
        """Test run preview initializes correctly."""
        preview = RunPreview()
        assert preview.command == ""
        assert preview.show_wandb_hint is False
        assert preview.validation_errors == []

    def test_run_preview_update_command(self):
        """Test updating command in preview."""
        preview = RunPreview()
        test_command = "aitraining llm --model gpt2 --train"
        preview.update_command(test_command)
        assert preview.command == test_command

    def test_run_preview_update_hints(self):
        """Test updating hints in preview."""
        preview = RunPreview()
        preview.update_hints(show_wandb_hint=True, validation_errors=["Model is required"])
        assert preview.show_wandb_hint is True
        assert len(preview.validation_errors) == 1
        assert preview.validation_errors[0] == "Model is required"

    def test_run_preview_format_command(self):
        """Test command formatting."""
        preview = RunPreview()
        command = "aitraining llm --model gpt2 --data data --lr 1e-4 --train"
        formatted = preview._format_command(command)
        assert formatted is not None
        assert len(formatted) > 0


class TestTokensModal:
    """Test TokensModal widget functionality."""

    def test_tokens_modal_initialization(self):
        """Test tokens modal initializes correctly."""
        # Set test environment
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("WANDB_API_KEY", None)

        modal = TokensModal()
        assert modal.hf_token_value == ""
        assert modal.wandb_token_value == ""

    def test_tokens_modal_reads_env(self):
        """Test tokens modal reads from environment."""
        os.environ["HF_TOKEN"] = "test_hf_token"
        os.environ["WANDB_API_KEY"] = "test_wandb_key"

        modal = TokensModal()
        assert modal.hf_token_value == "test_hf_token"
        assert modal.wandb_token_value == "test_wandb_key"

        # Clean up
        del os.environ["HF_TOKEN"]
        del os.environ["WANDB_API_KEY"]


class TestJsonViewer:
    """Test JsonViewer widget functionality."""

    def test_json_viewer_initialization(self):
        """Test JSON viewer initializes correctly."""
        viewer = JsonViewer()
        assert viewer.json_data == {}

    def test_json_viewer_update(self):
        """Test updating JSON data."""
        viewer = JsonViewer()
        test_data = {"trainer": "sft", "model": "gpt2"}
        viewer.update_json(test_data)
        assert viewer.json_data == test_data


class TestCatalogPanel:
    """Test enhanced CatalogPanel functionality."""

    def test_catalog_panel_initialization(self):
        """Test catalog panel initializes correctly."""
        panel = CatalogPanel()
        assert panel._model_entries == []
        assert panel._dataset_entries == []

    def test_catalog_messages(self):
        """Test catalog panel message types."""
        # Test ApplyModel message
        msg = CatalogPanel.ApplyModel("gpt2")
        assert msg.model_id == "gpt2"

        # Test ApplyDataset message
        msg = CatalogPanel.ApplyDataset("imdb")
        assert msg.dataset_id == "imdb"

        # Test ShowToast message
        msg = CatalogPanel.ShowToast("Test message", "success")
        assert msg.message == "Test message"
        assert msg.type == "success"


class TestNewLayoutIntegration:
    """Test new layout integration in main app."""

    def test_app_has_status_bar(self):
        """Test that app includes status bar."""

        async def runner():
            app = AITrainingTUI(dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()
                status_bar = app.query_one("#status-bar", StatusBar)
                assert status_bar is not None

        asyncio.run(runner())

    def test_app_has_run_preview(self):
        """Test that app includes run preview."""

        async def runner():
            app = AITrainingTUI(dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()
                run_preview = app.query_one("#run-preview", RunPreview)
                assert run_preview is not None

        asyncio.run(runner())

    def test_app_has_main_tabs(self):
        """Test that app has main tabs container."""

        async def runner():
            app = AITrainingTUI(dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()
                # Check for main tabs container
                from textual.widgets import TabbedContent

                main_tabs = app.query_one("#main-tabs", TabbedContent)
                assert main_tabs is not None

        asyncio.run(runner())

    def test_app_has_basic_and_advanced_tabs(self):
        """Test that app has basic and advanced tabs."""

        async def runner():
            app = AITrainingTUI(dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()
                # Check for parameter forms
                from autotrain.cli.tui.widgets.parameter_form import ParameterForm

                basic_form = app.query_one("#param-form-basic", ParameterForm)
                advanced_form = app.query_one("#param-form-advanced", ParameterForm)
                assert basic_form is not None
                assert advanced_form is not None

        asyncio.run(runner())

    def test_catalog_panel_exists(self):
        """Test that catalog panel exists in layout."""

        async def runner():
            app = AITrainingTUI(dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()
                catalog = app.query_one("#catalog-panel", CatalogPanel)
                assert catalog is not None

        asyncio.run(runner())

    def test_tokens_modal_action(self):
        """Test that tokens modal action is callable."""

        async def runner():
            app = AITrainingTUI(dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()
                # Check that action exists
                assert hasattr(app, "action_show_tokens")

        asyncio.run(runner())

    def test_parameter_change_updates_status_bar(self):
        """Test that parameter changes update status bar."""

        async def runner():
            app = AITrainingTUI(dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()

                # Set a parameter
                app.state.set_parameter("model", "test-model")
                app._update_status_bar()

                status_bar = app.query_one("#status-bar", StatusBar)
                assert status_bar.model == "test-model"

        asyncio.run(runner())

    def test_trainer_change_updates_status_bar(self):
        """Test that trainer changes update status bar."""

        async def runner():
            app = AITrainingTUI(dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()

                # Change trainer
                app.state.set_trainer("ppo")
                app._update_status_bar()

                status_bar = app.query_one("#status-bar", StatusBar)
                assert status_bar.trainer == "ppo"

        asyncio.run(runner())

    def test_command_preview_updates_run_preview(self):
        """Test that command preview updates run preview widget."""

        async def runner():
            app = AITrainingTUI(dry_run=True)
            async with app.run_test() as pilot:
                await pilot.pause()

                # Set parameters and update command
                app.state.set_parameter("model", "gpt2")
                app.state.set_parameter("data_path", "data")
                await app._update_command_preview()

                run_preview = app.query_one("#run-preview", RunPreview)
                assert "gpt2" in run_preview.command

        asyncio.run(runner())
