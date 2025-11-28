"""Main TUI Application for AITraining."""

import asyncio
import asyncio.subprocess
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.panel import Panel
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Header, Input, Label, ListItem, ListView, RichLog, Select, TabbedContent, TextArea

from autotrain import logger
from autotrain.cli.run_llm import FIELD_GROUPS, FIELD_SCOPES
from autotrain.cli.utils import get_field_info
from autotrain.trainers.clm.params import LLMTrainingParams

from .runner import CommandRunner
from .state.app_state import AppState
from .widgets.catalog_panel import CatalogPanel
from .widgets.custom_footer import CustomFooter
from .widgets.group_list import GroupList

# Import our custom widgets and components
from .widgets.parameter_form import ParameterForm
from .widgets.run_preview import RunPreview
from .widgets.tokens_modal import TokensModal
from .widgets.trainer_selector import TrainerSelector


class TUILogHandler(logging.Handler):
    """Custom logging handler that writes to a RichLog widget."""

    def __init__(self, log_widget: Optional[RichLog] = None):
        super().__init__()
        self.log_widget = log_widget

    def set_widget(self, widget: RichLog) -> None:
        """Set the RichLog widget to write to."""
        self.log_widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the RichLog widget."""
        if self.log_widget is None:
            return  # Silently drop logs if no widget is set

        try:
            msg = self.format(record)

            # Color code by level
            if record.levelno >= logging.ERROR:
                markup = f"[red]{msg}[/red]"
            elif record.levelno >= logging.WARNING:
                markup = f"[yellow]{msg}[/yellow]"
            elif record.levelno >= logging.INFO:
                markup = f"[cyan]{msg}[/cyan]"
            else:
                markup = f"[dim]{msg}[/dim]"

            self.log_widget.write(markup)
        except Exception:
            # Silently ignore any errors to avoid infinite recursion
            pass


class AITrainingTUI(App):
    """AITraining Terminal User Interface Application."""

    CSS_PATH = "theme.tcss"

    BINDINGS = [
        Binding("ctrl+s", "save_config", "Save Config", priority=True),
        Binding("ctrl+l", "load_config", "Load Config", priority=True),
        Binding("ctrl+r", "run_training", "Run", priority=True, show=True),
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("q", "quit", "Quit", show=False),
        Binding("f1", "toggle_help", "Help", priority=True),
        Binding("f5", "refresh", "Refresh", show=False),
        Binding("ctrl+p", "show_command", "Preview Command"),
        Binding("ctrl+d", "toggle_dry_run", "Toggle Dry Run"),
        Binding("ctrl+t", "toggle_theme", "Toggle Theme"),
        Binding("t", "show_tokens", "Tokens", show=True),
        Binding("/", "search", "Search Parameters"),
        Binding("escape", "clear_search", "Clear Search", show=False),
    ]

    TITLE = "AITraining TUI"

    def __init__(
        self,
        theme: str = "dark",
        dry_run: bool = False,
        config_file: Optional[str] = None,
    ):
        """Initialize the TUI application."""
        super().__init__()
        self.theme_name = theme
        self.dry_run = dry_run
        self.initial_config = config_file
        self.state = AppState()
        self.runner = CommandRunner(dry_run=dry_run)
        self._wandb_process: Optional[asyncio.subprocess.Process] = None
        self.wandb_visualizer_task: Optional[asyncio.Task] = None
        self.tui_log_handler: Optional[TUILogHandler] = None
        self._log_handler_ready = False

        # CRITICAL: Suppress all autotrain logging to console BEFORE any params instantiation
        # This prevents log spam during LLMTrainingParams creation
        autotrain_logger = logging.getLogger("autotrain")
        for handler in autotrain_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                autotrain_logger.removeHandler(handler)
        autotrain_logger.setLevel(logging.CRITICAL)  # Suppress until TUI log handler is ready

    def compose(self) -> ComposeResult:
        """Compose the TUI layout with modern tabbed interface."""
        # Header with branding
        yield Header(show_clock=True)

        # Main content container (horizontal layout)
        with Container(id="content-container"):
            # Left sidebar: Trainer selector and groups
            with Vertical(id="left-sidebar"):
                yield Label("=== AITraining ===", id="branding")
                yield TrainerSelector(id="trainer-selector")
                yield GroupList(id="group-list")

            # Center panel - Just parameters, simple and clean
            with VerticalScroll(id="center-panel"):
                yield Label("Parameters", id="param-header")
                yield ParameterForm(id="param-form")

            # Right panel - Command preview, model/dataset selects, and logs
            with Vertical(id="right-panel"):
                # Top section: Command preview
                with Vertical(id="command-section"):
                    yield RunPreview(id="run-preview")

                # Middle section: Model and dataset selectors
                with Vertical(id="selectors-section"):
                    yield Label("Popular Models", id="models-header")
                    yield Select(
                        [
                            ("google/gemma-2-2b", "google/gemma-2-2b"),
                            ("meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B"),
                            ("meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B"),
                            ("microsoft/Phi-3.5-mini-instruct", "microsoft/Phi-3.5-mini-instruct"),
                            ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
                            ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
                            ("openai/gpt-oss-28b", "openai/gpt-oss-28b"),
                            ("HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B"),
                        ],
                        prompt="Select a model",
                        id="model-select",
                    )

                    yield Label("Popular Datasets", id="datasets-header")
                    yield Select(
                        [
                            ("tatsu-lab/alpaca", "tatsu-lab/alpaca"),
                            ("timdettmers/openassistant-guanaco", "timdettmers/openassistant-guanaco"),
                            ("mlabonne/orpo-dpo-mix-40k", "mlabonne/orpo-dpo-mix-40k"),
                            (
                                "argilla/distilabel-capybara-dpo-7k-binarized",
                                "argilla/distilabel-capybara-dpo-7k-binarized",
                            ),
                            (
                                "argilla/ultrafeedback-binarized-preferences",
                                "argilla/ultrafeedback-binarized-preferences",
                            ),
                        ],
                        prompt="Select a dataset",
                        id="dataset-select",
                    )

                # Bottom section: Logs (takes remaining space)
                with Vertical(id="logs-section"):
                    yield Label("Logs", id="logs-header")
                    yield RichLog(id="log-viewer", highlight=True, markup=True)

        # Custom footer with status bar and keybinding tips combined
        yield CustomFooter(id="custom-footer")

    async def on_mount(self) -> None:
        """Initialize the app when mounted."""
        self.title = "AITraining TUI [EXPERIMENTAL] - Interactive Parameter Configuration"
        self.sub_title = (
            f"EXPERIMENTAL FEATURE | Theme: {self.theme_name} | Dry Run: {'ON' if self.dry_run else 'OFF'}"
        )

        # PRIORITY 1: Set up custom logging handler FIRST before any other initialization
        # This must happen before any LLMTrainingParams instantiation to prevent log spam
        try:
            # Get the RichLog widget from the right panel
            log_viewer = self.query_one("#log-viewer", RichLog)

            # 1. Configure LOGURU logger (used by autotrain internally)
            try:
                from loguru import logger as loguru_logger

                # Remove all existing loguru sinks (default is stderr/stdout)
                loguru_logger.remove()

                # Add sink to RichLog
                def rich_sink(message):
                    # Determine level color if possible, but message comes formatted if we use format string
                    # Just write the raw message for now, letting rich handle markup if present
                    # We strip to avoid double newlines
                    text = message.strip()
                    if text:
                        # Simple coloring based on level string presence if not already marked up
                        if "ERROR" in text or "CRITICAL" in text:
                            log_viewer.write(f"[red]{text}[/red]")
                        elif "WARNING" in text:
                            log_viewer.write(f"[yellow]{text}[/yellow]")
                        elif "DEBUG" in text:
                            log_viewer.write(f"[dim]{text}[/dim]")
                        else:
                            log_viewer.write(f"[cyan]{text}[/cyan]")

                # Add the sink with a format that doesn't duplicate what we do manually
                loguru_logger.add(rich_sink, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

                # Also verify autotrain.logger is using this
                # (autotrain.logger IS loguru logger)
            except ImportError:
                pass

            # 2. Configure STANDARD LOGGING (used by other libraries like transformers/textual)
            # Create our custom handler
            self.tui_log_handler = TUILogHandler(log_viewer)
            self.tui_log_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )

            # Get the autotrain logger and configure it (in case it's using std logging too)
            autotrain_logger = logging.getLogger("autotrain")

            # Remove any remaining console handlers
            for handler in autotrain_logger.handlers[:]:
                if isinstance(handler, (logging.StreamHandler, logging.FileHandler)):
                    autotrain_logger.removeHandler(handler)

            # Add our custom handler and restore normal log level
            autotrain_logger.addHandler(self.tui_log_handler)
            autotrain_logger.setLevel(logging.INFO)  # Restore from CRITICAL

            # Also redirect the module logger
            logger.addHandler(self.tui_log_handler)
            logger.setLevel(logging.INFO)

            # Mark handler as ready
            self._log_handler_ready = True

            # Log initial message to confirm it's working
            log_viewer.write("[dim]Logging system initialized. Logs will appear here.[/dim]\n")

        except Exception:
            # Fall back to suppressing logs if custom handler fails
            try:
                logging.getLogger("autotrain").setLevel(logging.WARNING)
                logger.setLevel(logging.WARNING)
            except Exception:
                pass

        # Theme selection is currently cosmetic; Textual themes must be registered first.
        # We avoid setting App.theme directly to prevent InvalidThemeError in environments
        # where custom themes aren't registered. Styling is handled via CSS.

        # Load initial state
        await self._initialize_state()

        # Load config if provided
        if self.initial_config:
            await self._load_config_file(self.initial_config)

        # Initialize UI with state - do this AFTER all widgets are mounted
        # The first refresh might happen before child widgets are fully ready,
        # so we schedule a proper refresh after the mount cycle completes
        self._reset_wandb_viewer()

        # Initial UI refresh after widgets are mounted
        self.call_after_refresh(self._refresh_ui_sync)

        # Show experimental warning
        self.notify(
            "⚠️ EXPERIMENTAL: This TUI is under active development. Use 'aitraining llm' for production.",
            severity="warning",
            timeout=8,
        )

        # Show quick tips so users discover keybindings immediately
        self.notify("Tips: F1 for Help • Ctrl+R Run • t Tokens • / Search", severity="information", timeout=4)

    def _refresh_ui_sync(self) -> None:
        """Synchronous wrapper for _refresh_ui to use with call_after_refresh."""
        import asyncio

        try:
            asyncio.create_task(self._refresh_ui())
        except Exception:
            # Fallback if we can't create task
            pass

    async def _initialize_state(self) -> None:
        """Initialize the application state."""
        # Get field info from LLMTrainingParams
        # Use enforce_scope=False in TUI to allow graceful degradation with warnings
        # instead of hard failures when metadata is incomplete
        field_info = get_field_info(LLMTrainingParams, FIELD_GROUPS, FIELD_SCOPES, enforce_scope=False)

        # Filter out the "trainer" field as it's controlled exclusively by TrainerSelector
        field_info = [f for f in field_info if f["arg"] != "--trainer"]

        # Initialize state with field info
        self.state.initialize_fields(field_info)

        # Set default trainer
        self.state.set_trainer("default")

        # Group fields by their groups
        groups = {}
        for field in field_info:
            group = field.get("group", "Other")
            if group not in groups:
                groups[group] = []
            groups[group].append(field)

        self.state.groups = groups
        self.state.current_group = "Basic"

    async def _refresh_ui(self) -> None:
        """Refresh the UI based on current state."""
        try:
            # Update trainer selector
            trainer_selector = self.query_one("#trainer-selector", TrainerSelector)
            trainer_selector.set_trainer(self.state.current_trainer)

            # Update group list
            group_list = self.query_one("#group-list", GroupList)
            visible_groups = self._get_visible_groups()
            group_list.set_groups(visible_groups)
            group_list.set_current_group(self.state.current_group)

            # Update parameter form with current group's fields
            try:
                param_form = self.query_one("#param-form", ParameterForm)
                visible_fields = self._get_visible_fields_for_group(self.state.current_group)
                logger.debug(f"Updating form with {len(visible_fields)} fields from group: {self.state.current_group}")
                await param_form.set_fields(visible_fields, self.state.parameters)
            except Exception as form_error:
                logger.warning(f"Could not update parameter form: {str(form_error)}")

            # Update command preview and run preview panel
            await self._update_command_preview()
            await self._update_run_preview()

            # Update status bar
            self._update_status_bar()

            # Update catalog panel
            self._update_catalog_panel()

        except Exception as e:
            # Use str() to avoid format string issues with error messages containing braces
            logger.error(f"Error refreshing UI: {str(e)}", exc_info=True)

    def _update_catalog_panel(self) -> None:
        """Refresh the catalog suggestions tab."""
        try:
            catalog_panel = self.query_one("#catalog-panel", CatalogPanel)
        except Exception:
            return
        catalog_panel.update_catalog("llm", self.state.current_trainer)

    async def _update_run_preview(self) -> None:
        """Update the run preview panel with hints."""
        try:
            run_preview = self.query_one("#run-preview", RunPreview)

            # Check if W&B visualizer should be shown
            show_wandb_hint = False
            params = self._compute_llm_params()
            if params and params.log == "wandb" and params.wandb_visualizer:
                show_wandb_hint = True

            # Check for validation errors
            validation_errors = []
            if not self.state.parameters.get("model"):
                validation_errors.append("Model is required")
            if not self.state.parameters.get("data_path"):
                validation_errors.append("Dataset is required")

            run_preview.update_hints(show_wandb_hint=show_wandb_hint, validation_errors=validation_errors)
        except Exception:
            pass  # Widget may not be mounted yet

    def _update_status_bar(self) -> None:
        """Update the status bar with current state."""
        try:
            footer = self.query_one("#custom-footer", CustomFooter)
            footer.update_status(
                trainer=self.state.current_trainer,
                model=self.state.parameters.get("model"),
                dataset=self.state.parameters.get("data_path"),
            )
        except Exception:
            pass  # Widget may not be mounted yet

    def _get_visible_groups(self) -> List[str]:
        """Get list of groups visible for current trainer."""
        visible_groups = set()

        for field in self.state.fields:
            # Check if field is visible for current trainer
            scopes = field.get("scope", ["all"])
            if "all" in scopes or self.state.current_trainer in scopes:
                group = field.get("group", "Other")
                visible_groups.add(group)

        # Order groups according to FIELD_GROUPS ordering
        group_order = [
            "Basic",
            "Data Processing",
            "Training Configuration",
            "Training Hyperparameters",
            "PEFT/LoRA",
            "DPO/ORPO",
            "Hub Integration",
            "Knowledge Distillation",
            "Hyperparameter Sweep",
            "Enhanced Evaluation",
            "Reinforcement Learning (PPO)",
            "Advanced Features",
            "Inference",
        ]

        ordered = [g for g in group_order if g in visible_groups]
        remaining = sorted(visible_groups - set(ordered))
        return ordered + remaining

    def _get_visible_fields_for_group(self, group: str) -> List[Dict]:
        """Get fields visible for current trainer and group."""
        visible = []

        for field in self.state.fields:
            # Check group
            if field.get("group", "Other") != group:
                continue

            # Check scope - filter by trainer
            scopes = field.get("scope", ["all"])
            # Show field if scope includes "all" OR current trainer is in the scope list
            if "all" in scopes or self.state.current_trainer in scopes:
                visible.append(field)
            # else: field is filtered out because trainer is not in scope

        return visible

    async def _update_command_preview(self) -> None:
        """Update the command preview based on current parameters."""
        # Build command
        command = self._build_command()
        command_str = " ".join(command)

        # Update run preview widget
        try:
            run_preview = self.query_one("#run-preview", RunPreview)
            run_preview.update_command(command_str)
        except Exception:
            pass  # Widget may not be mounted yet

    def _build_command(self) -> List[str]:
        """Build the CLI command from current parameters."""
        command = ["aitraining", "llm"]

        # Add trainer if not default
        if self.state.current_trainer != "default":
            command.append(f"--trainer={self.state.current_trainer}")

        # Add all non-None parameters
        for key, value in self.state.parameters.items():
            if value is not None and value != "":
                # Skip defaults to keep command clean
                field_info = self._get_field_by_name(key)
                if field_info:
                    default = field_info.get("default")
                    if value == default:
                        continue

                # Convert key to CLI format
                cli_key = key.replace("_", "-")

                # Handle boolean flags
                if isinstance(value, bool):
                    if value:
                        command.append(f"--{cli_key}")
                else:
                    command.append(f"--{cli_key}={value}")

        # Add --train flag
        command.append("--train")

        return command

    def _get_field_by_name(self, name: str) -> Optional[Dict]:
        """Get field info by field name."""
        for field in self.state.fields:
            field_name = field["arg"].lstrip("-").replace("-", "_")
            if field_name == name:
                return field
        return None

    @work
    async def action_save_config(self) -> None:
        """Save configuration to file."""
        # Show save dialog
        save_dialog = SaveConfigDialog()
        filename = await self.push_screen_wait(save_dialog)

        if filename:
            try:
                config = {
                    "trainer": self.state.current_trainer,
                    "parameters": {k: v for k, v in self.state.parameters.items() if v is not None and v != ""},
                }

                path = Path(filename)
                if path.suffix == ".yaml" or path.suffix == ".yml":
                    import yaml

                    with open(path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False)
                else:
                    with open(path, "w") as f:
                        json.dump(config, f, indent=2)

                log_viewer = self.query_one("#log-viewer", RichLog)
                log_viewer.write(f"[green]✓[/green] Configuration saved to {path}")

            except Exception as e:
                log_viewer = self.query_one("#log-viewer", RichLog)
                log_viewer.write(f"[red]✗[/red] Failed to save config: {e}")

    @work
    async def action_load_config(self) -> None:
        """Load configuration from file."""
        # Show load dialog
        load_dialog = LoadConfigDialog()
        filename = await self.push_screen_wait(load_dialog)

        if filename:
            await self._load_config_file(filename)

    async def _load_config_file(self, filename: str) -> None:
        """Load configuration from a file."""
        try:
            path = Path(filename)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")

            if path.suffix == ".yaml" or path.suffix == ".yml":
                import yaml

                with open(path) as f:
                    config = yaml.safe_load(f)
            else:
                with open(path) as f:
                    config = json.load(f)

            # Apply config
            if "trainer" in config:
                self.state.set_trainer(config["trainer"])

            if "parameters" in config:
                for key, value in config["parameters"].items():
                    self.state.set_parameter(key, value)

            # Refresh UI
            await self._refresh_ui()

            log_viewer = self.query_one("#log-viewer", RichLog)
            log_viewer.write(f"[green]✓[/green] Configuration loaded from {path}")

        except Exception as e:
            log_viewer = self.query_one("#log-viewer", RichLog)
            log_viewer.write(f"[red]✗[/red] Failed to load config: {e}")

    async def action_run_training(self) -> None:
        """Run training with current configuration."""
        await self._stop_wandb_visualizer()

        # Build command
        command = self._build_command()

        # Switch to logs tab
        tabs = self.query_one("#main-tabs", TabbedContent)
        tabs.active = "tab-logs"

        # Clear logs
        log_viewer = self.query_one("#log-viewer", RichLog)
        log_viewer.clear()

        # Show command
        log_viewer.write(Panel(" ".join(command), title="Command", border_style="blue"))

        if self.dry_run:
            log_viewer.write("[yellow]DRY RUN MODE[/yellow] - Command will not be executed")
            log_viewer.write("[dim]This is a preview of what would be executed[/dim]")
            return

        # Run command
        log_viewer.write("\n[cyan]Starting training...[/cyan]\n")

        wandb_context = self._get_wandb_visualizer_context()
        if wandb_context:
            if wandb_context.get("command"):
                log_viewer.write(f"[dim]W&B visualizer command:[/dim] {wandb_context['command']}")
            self.wandb_visualizer_task = asyncio.create_task(self._run_wandb_visualizer_process(wandb_context))

        try:
            # Run in background
            await self.runner.run_command(command, log_viewer)
        except Exception as e:
            log_viewer.write(f"\n[red]Error:[/red] {e}")
        finally:
            await self._stop_wandb_visualizer()

    async def action_toggle_help(self) -> None:
        """Toggle help overlay."""
        help_screen = HelpScreen()
        await self.push_screen(help_screen)

    async def action_toggle_dry_run(self) -> None:
        """Toggle dry run mode."""
        self.dry_run = not self.dry_run
        self.runner.dry_run = self.dry_run
        self.sub_title = f"Theme: {self.theme_name} | Dry Run: {'ON' if self.dry_run else 'OFF'}"

        log_viewer = self.query_one("#log-viewer", RichLog)
        log_viewer.write(f"Dry run mode: {'[green]ON[/green]' if self.dry_run else '[red]OFF[/red]'}")

    async def action_toggle_theme(self) -> None:
        """Toggle between light and dark themes."""
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.sub_title = f"Theme: {self.theme_name} | Dry Run: {'ON' if self.dry_run else 'OFF'}"

    async def action_show_command(self) -> None:
        """Show command in the run preview panel (right side)."""
        # Command is always visible in the RunPreview widget on the right
        # This action now just confirms the preview is visible
        self.notify("Command preview is shown in the right panel", severity="information", timeout=2)

    @work
    async def action_show_tokens(self) -> None:
        """Show tokens modal for managing API tokens."""
        tokens_modal = TokensModal()
        result = await self.push_screen_wait(tokens_modal)
        if result:
            # Tokens were updated, refresh status bar
            self._update_status_bar()
            log_viewer = self.query_one("#log-viewer", RichLog)
            log_viewer.write("[green]✓[/green] API tokens updated")

    @work
    async def action_search(self) -> None:
        """Open parameter search."""
        search_dialog = SearchDialog(self.state.fields)
        result = await self.push_screen_wait(search_dialog)

        if result:
            # Find the field's group and switch to it
            field = result
            group = field.get("group", "Other")

            # Switch to group
            self.state.current_group = group
            await self._refresh_ui()

            # Highlight the field (future enhancement)
            log_viewer = self.query_one("#log-viewer", RichLog)
            log_viewer.write(f"Jumped to: {field['arg']} in {group}")

    @on(TrainerSelector.TrainerChanged)
    async def handle_trainer_change(self, event: TrainerSelector.TrainerChanged) -> None:
        """Handle trainer selection change."""
        self.state.set_trainer(event.trainer)

        # Choose a sensible default group for this trainer
        preferred = self._default_group_for_trainer(event.trainer)
        visible_groups = self._get_visible_groups()
        if preferred in visible_groups:
            self.state.current_group = preferred
        elif self.state.current_group not in visible_groups and visible_groups:
            # Fallback to first visible group if current group is no longer available
            self.state.current_group = visible_groups[0]

        await self._refresh_ui()

    def _default_group_for_trainer(self, trainer: str) -> str:
        """Return the default group to focus for a trainer selection."""
        trainer = (trainer or "default").lower()
        if trainer == "ppo":
            return "Reinforcement Learning (PPO)"
        if trainer in ("dpo", "orpo"):
            return "DPO/ORPO"
        return "Basic"

    @on(GroupList.GroupSelected)
    async def handle_group_change(self, event: GroupList.GroupSelected) -> None:
        """Handle group selection change."""
        self.state.current_group = event.group
        await self._refresh_ui()

    @on(ParameterForm.ParameterChanged)
    async def handle_parameter_change(self, event: ParameterForm.ParameterChanged) -> None:
        """Handle parameter value change."""
        self.state.set_parameter(event.name, event.value)
        await self._update_command_preview()
        await self._update_run_preview()
        self._update_status_bar()

    @on(CatalogPanel.ApplyModel)
    async def handle_catalog_model(self, event: CatalogPanel.ApplyModel) -> None:
        """Apply a catalog model selection."""
        self.state.set_parameter("model", event.model_id)
        await self._refresh_ui()
        self._log_catalog_event(f"[green]✓[/green] Model set to {event.model_id}")

    @on(CatalogPanel.ApplyDataset)
    async def handle_catalog_dataset(self, event: CatalogPanel.ApplyDataset) -> None:
        """Apply a catalog dataset selection."""
        self.state.set_parameter("data_path", event.dataset_id)
        await self._refresh_ui()
        self._log_catalog_event(f"[green]✓[/green] Dataset set to {event.dataset_id}")

    @on(CatalogPanel.ShowToast)
    def handle_catalog_toast(self, event: CatalogPanel.ShowToast) -> None:
        """Handle toast notification from catalog panel."""
        severity = "information" if event.type == "success" else event.type
        self.notify(event.message, severity=severity, timeout=3)

    @on(Select.Changed, "#model-select")
    async def handle_model_select(self, event: Select.Changed) -> None:
        """Handle model selection from dropdown."""
        if event.value:
            self.state.set_parameter("model", event.value)
            await self._refresh_ui()
            self.notify(f"Model set to {event.value}", severity="information")

    @on(Select.Changed, "#dataset-select")
    async def handle_dataset_select(self, event: Select.Changed) -> None:
        """Handle dataset selection from dropdown."""
        if event.value:
            self.state.set_parameter("data_path", event.value)
            await self._refresh_ui()
            self.notify(f"Dataset set to {event.value}", severity="information")

    @on(TokensModal.TokensUpdated)
    def handle_tokens_updated(self, event: TokensModal.TokensUpdated) -> None:
        """Handle tokens update from tokens modal."""
        self._update_status_bar()
        log_viewer = self.query_one("#log-viewer", RichLog)
        log_viewer.write("[green]✓[/green] API tokens updated")

    def _log_catalog_event(self, message: str) -> None:
        """Write catalog actions to the log viewer."""
        try:
            log_viewer = self.query_one("#log-viewer", RichLog)
        except Exception:
            return
        log_viewer.write(message)

    def _reset_wandb_viewer(self) -> None:
        """Reset the W&B viewer log."""
        # In the new layout, we don't have a separate wandb-viewer
        # W&B output goes to the log viewer

    def _build_llm_params_snapshot(self) -> Dict[str, Any]:
        """Build parameter snapshot for LLM params instantiation."""
        snapshot: Dict[str, Any] = {}
        for key, value in self.state.parameters.items():
            if value not in (None, ""):
                snapshot[key] = value
        return snapshot

    def _compute_llm_params(self) -> Optional[LLMTrainingParams]:
        """Instantiate LLMTrainingParams to leverage defaults."""
        try:
            # Temporarily suppress logs during LLMTrainingParams instantiation
            # to prevent log spam in the TUI (especially "Project path normalized to:")
            import contextlib
            import io

            # If log handler is ready, logs go to the log viewer, otherwise suppress them
            if hasattr(self, "_log_handler_ready") and self._log_handler_ready:
                # Handler is ready, logs will go to the proper place
                return LLMTrainingParams(**self._build_llm_params_snapshot())
            else:
                # Handler not ready yet, suppress stdout/stderr to prevent UI corruption
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    # Also temporarily raise log level to suppress any remaining log output
                    # Note: autotrain logger is a LoggingAdapter, we need the underlying logger
                    underlying_logger = logging.getLogger("autotrain")
                    old_level = underlying_logger.level
                    underlying_logger.setLevel(logging.CRITICAL)
                    try:
                        result = LLMTrainingParams(**self._build_llm_params_snapshot())
                    finally:
                        underlying_logger.setLevel(old_level)
                    return result
        except Exception as exc:
            logger.debug(f"Failed to compute LLM params for visualizer: {exc}")
            return None

    def _get_wandb_visualizer_context(self) -> Optional[Dict[str, Any]]:
        """Determine if W&B visualizer should run and return context."""
        params = self._compute_llm_params()
        if not params:
            return None
        if params.log != "wandb" or not params.wandb_visualizer:
            return None
        project_path = str(Path(params.project_name).resolve())
        command = f'WANDB_DIR="{project_path}" wandb beta leet "{project_path}"'
        return {
            "project_path": project_path,
            "wandb_token": params.wandb_token,
            "command": command,
        }

    async def _run_wandb_visualizer_process(self, context: Dict[str, Any]) -> None:
        """Launch and stream W&B LEET output."""
        # In the new layout, all output goes to log-viewer
        viewer = self.query_one("#log-viewer", RichLog)
        viewer.write("[cyan]Launching W&B LEET visualizer...[/cyan]")

        env = os.environ.copy()
        project_path = context["project_path"]
        os.makedirs(project_path, exist_ok=True)
        # Pin all W&B paths to the project path to avoid repo root writes
        env["WANDB_DIR"] = project_path
        env["WANDB_CACHE_DIR"] = project_path
        env["WANDB_DATA_DIR"] = project_path
        token = context.get("wandb_token")
        if token:
            env["WANDB_API_KEY"] = token

        cmd = [sys.executable, "-m", "wandb", "beta", "leet", project_path]
        if context.get("command"):
            viewer.write(f"[dim]Reopen later with:[/dim] {context['command']}")

        try:
            self._wandb_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError:
            viewer.write("[red]`wandb` executable not found. Install wandb>=0.23.0 to use LEET.[/red]")
            self._wandb_process = None
            return
        except Exception as exc:
            viewer.write(f"[red]Failed to launch W&B visualizer:[/red] {exc}")
            self._wandb_process = None
            return

        async def pipe(stream, prefix: str = ""):
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    viewer.write(f"{prefix}{text}" if prefix else text)

        try:
            await asyncio.gather(
                pipe(self._wandb_process.stdout),
                pipe(self._wandb_process.stderr, "[yellow]ERR:[/yellow] "),
            )
            rc = await self._wandb_process.wait()
            if rc == 0:
                viewer.write("[green]W&B visualizer exited.[/green]")
            else:
                viewer.write(f"[red]W&B visualizer exited with code {rc}[/red]")
        except asyncio.CancelledError:
            if self._wandb_process and self._wandb_process.returncode is None:
                self._wandb_process.terminate()
                try:
                    await asyncio.wait_for(self._wandb_process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._wandb_process.kill()
                    await self._wandb_process.wait()
            viewer.write("[yellow]W&B visualizer stopped.[/yellow]")
            raise
        finally:
            self._wandb_process = None
            self.wandb_visualizer_task = None

    async def _stop_wandb_visualizer(self) -> None:
        """Stop the W&B visualizer task/process if running."""
        if self.wandb_visualizer_task:
            self.wandb_visualizer_task.cancel()
            try:
                await self.wandb_visualizer_task
            except asyncio.CancelledError:
                pass
            self.wandb_visualizer_task = None
        elif self._wandb_process and self._wandb_process.returncode is None:
            self._wandb_process.terminate()
            try:
                await asyncio.wait_for(self._wandb_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._wandb_process.kill()
                await self._wandb_process.wait()
            self._wandb_process = None
        else:
            self._wandb_process = None
        self._reset_wandb_viewer()


class SaveConfigDialog(ModalScreen):
    """Modal dialog for saving configuration."""

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Save Configuration", id="dialog-title")
            yield Input(placeholder="Enter filename (e.g., config.json or config.yaml)", id="filename-input")
            with Horizontal(id="dialog-buttons"):
                yield Button("Save", variant="primary", id="save-button")
                yield Button("Cancel", variant="default", id="cancel-button")

    @on(Button.Pressed, "#save-button")
    async def save(self) -> None:
        """Save and close dialog."""
        filename = self.query_one("#filename-input", Input).value
        if filename:
            self.dismiss(filename)

    @on(Button.Pressed, "#cancel-button")
    async def cancel(self) -> None:
        """Cancel and close dialog."""
        self.dismiss(None)


class LoadConfigDialog(ModalScreen):
    """Modal dialog for loading configuration."""

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Load Configuration", id="dialog-title")
            yield Input(placeholder="Enter filename (e.g., config.json or config.yaml)", id="filename-input")
            with Horizontal(id="dialog-buttons"):
                yield Button("Load", variant="primary", id="load-button")
                yield Button("Cancel", variant="default", id="cancel-button")

    @on(Button.Pressed, "#load-button")
    async def load(self) -> None:
        """Load and close dialog."""
        filename = self.query_one("#filename-input", Input).value
        if filename:
            self.dismiss(filename)

    @on(Button.Pressed, "#cancel-button")
    async def cancel(self) -> None:
        """Cancel and close dialog."""
        self.dismiss(None)


class SearchDialog(ModalScreen):
    """Modal dialog for searching parameters."""

    def __init__(self, fields: List[Dict]):
        """Initialize search dialog."""
        super().__init__()
        self.fields = fields
        self.filtered_fields = fields

    def compose(self) -> ComposeResult:
        with Container(id="search-dialog"):
            yield Label("Search Parameters", id="dialog-title")
            yield Input(placeholder="Type to search...", id="search-input")
            yield ListView(id="search-results")
            with Horizontal(id="dialog-buttons"):
                yield Button("Select", variant="primary", id="select-button")
                yield Button("Cancel", variant="default", id="cancel-button")

    async def on_mount(self) -> None:
        """Focus search input on mount."""
        self.query_one("#search-input", Input).focus()

    @on(Input.Changed, "#search-input")
    async def filter_results(self, event: Input.Changed) -> None:
        """Filter results based on search query."""
        query = event.value.lower()

        if query:
            self.filtered_fields = [
                field
                for field in self.fields
                if query in field["arg"].lower() or query in field.get("help", "").lower()
            ]
        else:
            self.filtered_fields = self.fields

        # Update list
        results = self.query_one("#search-results", ListView)
        results.clear()

        for field in self.filtered_fields[:20]:  # Limit to 20 results
            name = field["arg"].lstrip("-")
            help_text = field.get("help", "")[:50]
            results.append(ListItem(Label(f"{name}: {help_text}")))

    @on(Button.Pressed, "#select-button")
    async def select(self) -> None:
        """Select the highlighted field."""
        results = self.query_one("#search-results", ListView)
        if results.highlighted_child and self.filtered_fields:
            index = results.highlighted_child
            if index < len(self.filtered_fields):
                self.dismiss(self.filtered_fields[index])

    @on(Button.Pressed, "#cancel-button")
    async def cancel(self) -> None:
        """Cancel and close dialog."""
        self.dismiss(None)


class HelpScreen(ModalScreen):
    """Help screen with keybindings and instructions."""

    def compose(self) -> ComposeResult:
        help_text = """
# AITraining TUI Help

## Keyboard Shortcuts

### Navigation
- **Tab / Shift+Tab**: Navigate between panels
- **↑/↓**: Navigate lists and menus
- **Enter**: Select/Activate
- **Escape**: Clear search / Close dialogs

### Actions
- **Ctrl+R**: Run training
- **Ctrl+S**: Save configuration
- **Ctrl+L**: Load configuration
- **Ctrl+P**: Preview command
- **Ctrl+D**: Toggle dry run mode
- **Ctrl+T**: Toggle theme

### Other
- **/**: Search parameters
- **F1**: Show this help
- **q / Ctrl+C**: Quit

## Tips

1. Select a trainer to filter relevant parameters
2. Navigate groups to see categorized parameters
3. Modified values are highlighted
4. Use dry run mode to test without execution
5. Save configurations for reuse

Press any key to close this help screen.
        """

        with Container(id="help-screen"):
            yield TextArea(help_text, read_only=True, id="help-content")
            yield Button("Close", id="close-help")

    @on(Button.Pressed, "#close-help")
    async def close(self) -> None:
        """Close help screen."""
        self.dismiss()

    def on_key(self, event) -> None:
        """Close on any key press."""
        self.dismiss()
