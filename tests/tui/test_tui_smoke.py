"""Smoke tests for AITraining TUI."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from autotrain.cli.run_tui import RunAutoTrainTUICommand
from autotrain.cli.tui.runner import CommandRunner
from autotrain.cli.tui.state.app_state import AppState


class TestTUIImport:
    """Test that TUI modules can be imported."""

    def test_import_run_tui(self):
        """Test importing run_tui module."""
        from autotrain.cli import run_tui

        assert hasattr(run_tui, "RunAutoTrainTUICommand")

    def test_import_app(self):
        """Test importing TUI app module."""
        from autotrain.cli.tui import app

        assert hasattr(app, "AITrainingTUI")

    def test_import_widgets(self):
        """Test importing TUI widget modules."""
        from autotrain.cli.tui.widgets import (
            catalog_panel,
            context_panel,
            group_list,
            json_viewer,
            parameter_form,
            run_preview,
            status_bar,
            tokens_modal,
            trainer_selector,
        )

        assert hasattr(trainer_selector, "TrainerSelector")
        assert hasattr(group_list, "GroupList")
        assert hasattr(parameter_form, "ParameterForm")
        assert hasattr(context_panel, "ContextPanel")
        assert hasattr(catalog_panel, "CatalogPanel")
        assert hasattr(status_bar, "StatusBar")
        assert hasattr(run_preview, "RunPreview")
        assert hasattr(tokens_modal, "TokensModal")
        assert hasattr(json_viewer, "JsonViewer")

    def test_import_state(self):
        """Test importing state management module."""
        from autotrain.cli.tui.state import app_state

        assert hasattr(app_state, "AppState")

    def test_import_runner(self):
        """Test importing command runner module."""
        from autotrain.cli.tui import runner

        assert hasattr(runner, "CommandRunner")


class TestTUICommand:
    """Test TUI command functionality."""

    def test_tui_command_creation(self):
        """Test creating TUI command instance."""
        args = MagicMock()
        args.theme = "dark"
        args.dry_run = False
        args.config = None

        cmd = RunAutoTrainTUICommand(args)
        assert cmd.theme == "dark"
        assert cmd.dry_run is False
        assert cmd.config_file is None

    def test_tty_check_non_tty(self):
        """Test TTY check fails appropriately."""
        args = MagicMock()
        args.theme = "dark"
        args.dry_run = False
        args.config = None

        cmd = RunAutoTrainTUICommand(args)

        with patch("sys.stdin.isatty", return_value=False):
            assert cmd._check_tty() is False

    def test_tty_check_success(self):
        """Test TTY check passes in TTY environment."""
        args = MagicMock()
        args.theme = "dark"
        args.dry_run = False
        args.config = None

        cmd = RunAutoTrainTUICommand(args)

        with patch("sys.stdin.isatty", return_value=True):
            with patch("sys.stdout.isatty", return_value=True):
                with patch.dict(os.environ, {"TERM": "xterm"}):
                    assert cmd._check_tty() is True

    def test_run_exits_on_non_tty(self):
        """Test that run exits with code 2 on non-TTY."""
        args = MagicMock()
        args.theme = "dark"
        args.dry_run = False
        args.config = None

        cmd = RunAutoTrainTUICommand(args)

        with patch.object(cmd, "_check_tty", return_value=False):
            with pytest.raises(SystemExit) as exc_info:
                cmd.run()
            assert exc_info.value.code == 2


class TestAppState:
    """Test application state management."""

    def test_state_initialization(self):
        """Test state initialization."""
        state = AppState()
        assert state.current_trainer == "default"
        assert state.current_group == "Basic"
        assert state.parameters == {}
        assert state.modified_fields == set()

    def test_set_trainer(self):
        """Test setting trainer."""
        state = AppState()
        state.set_trainer("sft")
        assert state.current_trainer == "sft"

    def test_set_parameter(self):
        """Test setting parameters."""
        state = AppState()
        state.fields = [{"arg": "--model", "default": "gpt2", "type": str}]
        state.initialize_fields(state.fields)

        state.set_parameter("model", "llama2")
        assert state.get_parameter("model") == "llama2"
        assert "model" in state.modified_fields

    def test_reset_parameter(self):
        """Test resetting parameter to default."""
        state = AppState()
        state.fields = [{"arg": "--model", "default": "gpt2", "type": str}]
        state.initialize_fields(state.fields)

        state.set_parameter("model", "llama2")
        state.reset_parameter("model")
        assert state.get_parameter("model") == "gpt2"
        assert "model" not in state.modified_fields

    def test_export_config(self):
        """Test exporting configuration."""
        state = AppState()
        state.fields = [
            {"arg": "--model", "default": "gpt2", "type": str},
            {"arg": "--lr", "default": 0.001, "type": float},
        ]
        state.initialize_fields(state.fields)

        state.set_trainer("sft")
        state.set_parameter("model", "llama2")

        config = state.export_config()
        assert config["trainer"] == "sft"
        assert config["parameters"]["model"] == "llama2"
        assert "lr" not in config["parameters"]  # Not modified

    def test_import_config(self):
        """Test importing configuration."""
        state = AppState()
        state.fields = [
            {"arg": "--model", "default": "gpt2", "type": str},
            {"arg": "--lr", "default": 0.001, "type": float},
        ]
        state.initialize_fields(state.fields)

        config = {"trainer": "dpo", "parameters": {"model": "mistral", "lr": 0.0001}}

        state.import_config(config)
        assert state.current_trainer == "dpo"
        assert state.get_parameter("model") == "mistral"
        assert state.get_parameter("lr") == 0.0001

    def test_validate_parameters_required(self):
        """Test parameter validation for required fields."""
        state = AppState()
        state.fields = [
            {"arg": "--model", "default": None, "type": str},
            {"arg": "--project-name", "default": None, "type": str},
        ]
        state.initialize_fields(state.fields)

        errors = state.validate_parameters()
        assert "model" in errors
        assert "project_name" in errors

        state.set_parameter("model", "gpt2")
        state.set_parameter("project_name", "my-project")
        errors = state.validate_parameters()
        assert "model" not in errors
        assert "project_name" not in errors

    def test_validate_parameters_ppo(self):
        """Test PPO-specific validation."""
        state = AppState()
        state.fields = [
            {"arg": "--model", "default": None, "type": str},
            {"arg": "--project-name", "default": None, "type": str},
            {"arg": "--rl-reward-model-path", "default": None, "type": str, "scope": ["ppo"]},
        ]
        state.initialize_fields(state.fields)
        state.set_parameter("model", "gpt2")
        state.set_parameter("project_name", "test")

        state.set_trainer("ppo")
        errors = state.validate_parameters()
        assert "rl_reward_model_path" in errors

        state.set_parameter("rl_reward_model_path", "/path/to/model")
        errors = state.validate_parameters()
        assert "rl_reward_model_path" not in errors


class TestCommandRunner:
    """Test command runner functionality."""

    def test_runner_initialization(self):
        """Test command runner initialization."""
        runner = CommandRunner(dry_run=True)
        assert runner.dry_run is True
        assert runner.current_process is None

    def test_is_running(self):
        """Test checking if process is running."""
        runner = CommandRunner()
        assert runner.is_running() is False

        # Simulate a running process
        runner.current_process = MagicMock()
        runner.current_process.returncode = None
        assert runner.is_running() is True

        # Simulate a finished process
        runner.current_process.returncode = 0
        assert runner.is_running() is False


class TestConfigSaveLoad:
    """Test configuration save/load functionality."""

    def test_save_json_config(self):
        """Test saving configuration as JSON."""
        state = AppState()
        state.fields = [{"arg": "--model", "default": "gpt2", "type": str}]
        state.initialize_fields(state.fields)
        state.set_parameter("model", "llama2")

        config = state.export_config()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            temp_path = f.name

        try:
            # Read back and verify
            with open(temp_path) as f:
                loaded = json.load(f)

            assert loaded["trainer"] == "default"
            assert loaded["parameters"]["model"] == "llama2"
        finally:
            os.unlink(temp_path)

    def test_save_yaml_config(self):
        """Test saving configuration as YAML."""
        state = AppState()
        state.fields = [{"arg": "--model", "default": "gpt2", "type": str}]
        state.initialize_fields(state.fields)
        state.set_parameter("model", "llama2")

        config = state.export_config()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            # Read back and verify
            with open(temp_path) as f:
                loaded = yaml.safe_load(f)

            assert loaded["trainer"] == "default"
            assert loaded["parameters"]["model"] == "llama2"
        finally:
            os.unlink(temp_path)
