"""Interactive tests for AITraining TUI (Textual pilot-based)."""

import asyncio
import os
import sys


# Ensure src is importable when running tests directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from autotrain.cli.tui.app import AITrainingTUI
from autotrain.cli.tui.widgets.group_list import GroupList
from autotrain.cli.tui.widgets.parameter_form import ParameterForm
from autotrain.cli.tui.widgets.run_preview import RunPreview
from autotrain.cli.tui.widgets.status_bar import StatusBar
from autotrain.cli.tui.widgets.trainer_selector import TrainerSelector


def test_switch_to_ppo_shows_rl_group():
    async def runner():
        app = AITrainingTUI(dry_run=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Switch trainer via handler to trigger group auto-switch
            await app.handle_trainer_change(TrainerSelector.TrainerChanged("ppo"))

            group_list = app.query_one("#group-list", GroupList)
            assert "Reinforcement Learning (PPO)" in group_list.groups
            # Default group should be RL group after our change
            assert app.state.current_group == "Reinforcement Learning (PPO)"

    asyncio.run(runner())


def test_select_blank_does_not_crash():
    async def runner():
        app = AITrainingTUI(dry_run=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Force a field with None to render in a Select (chat_template)
            app.state.set_parameter("chat_template", None)
            await app._refresh_ui()
            # If we reach here, Select.BLANK was set and no exception occurred
            assert True

    asyncio.run(runner())


def test_css_parses_and_mounts():
    async def runner():
        app = AITrainingTUI(dry_run=True)
        async with app.run_test() as pilot:
            # If CSS fails to parse, mounting would raise; let the loop run once
            await pilot.pause()
            # If we got here, mount succeeded
            assert True

    asyncio.run(runner())


def test_status_bar_updates_on_param_change():
    """Test that status bar updates when parameters change."""

    async def runner():
        app = AITrainingTUI(dry_run=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Set model parameter
            app.state.set_parameter("model", "gpt2")
            app._update_status_bar()

            status_bar = app.query_one("#status-bar", StatusBar)
            assert status_bar.model == "gpt2"

    asyncio.run(runner())


def test_run_preview_shows_command():
    """Test that run preview widget displays command."""

    async def runner():
        app = AITrainingTUI(dry_run=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.state.set_parameter("model", "gpt2")
            app.state.set_parameter("data_path", "data")
            await app._update_command_preview()

            run_preview = app.query_one("#run-preview", RunPreview)
            assert run_preview.command is not None
            assert len(run_preview.command) > 0

    asyncio.run(runner())


def test_wandb_context_includes_command():
    async def runner():
        app = AITrainingTUI(dry_run=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.state.set_parameter("model", "gpt2")
            app.state.set_parameter("project_name", os.path.join(os.getcwd(), "tui_wandb_proj"))
            app.state.set_parameter("data_path", "data")
            app.state.set_parameter("log", "wandb")
            app.state.set_parameter("wandb_visualizer", True)
            context = app._get_wandb_visualizer_context()
            assert context is not None
            assert os.path.isabs(context["project_path"])
            assert "wandb beta leet" in context["command"]

    asyncio.run(runner())
