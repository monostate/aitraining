"""Run preview panel showing live command preview with hints."""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widget import Widget
from textual.widgets import Static


class RunPreview(Widget):
    """Widget displaying live command preview with contextual hints.

    Shows:
    - Current training command that will be executed
    - W&B LEET hint when wandb logging is enabled with visualizer
    - Other contextual hints based on configuration
    """

    DEFAULT_CSS = """
    RunPreview {
        height: auto;
        min-height: 6;
        max-height: 15;
        padding: 0;
    }

    #preview-panel {
        height: 100%;
        padding: 0;
        background: $panel;
        border: round $primary;
    }

    .preview-header {
        text-style: bold;
        color: $primary;
        padding: 0 1;
        border-bottom: solid $primary;
        text-align: center;
        background: $primary 10%;
    }

    #preview-scroll {
        height: auto;
        padding: 0;
    }

    #preview-command {
        padding: 1;
        background: $surface;
        border: none;
        margin: 0;
    }

    .command-text {
        color: $text;
    }

    #preview-hints {
        padding: 0;
    }

    .hint-box {
        color: $warning;
        text-style: italic;
        padding: 1;
        background: $warning 10%;
        border: round $warning;
        margin: 0 0 1 0;
    }

    .hint-wandb {
        color: $success;
        text-style: italic;
        padding: 1;
        background: $success 10%;
        border: round $success;
        margin: 0 0 1 0;
    }

    .hint-error {
        color: $error;
        text-style: italic;
        padding: 1;
        background: $error 10%;
        border: round $error;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the run preview widget."""
        super().__init__(**kwargs)
        self.command = ""
        self.show_wandb_hint = False
        self.validation_errors = []

    def compose(self) -> ComposeResult:
        """Compose the run preview UI."""
        with Container(id="preview-panel"):
            yield Static("⚡ Run Preview", classes="preview-header")

            with Vertical(id="preview-scroll"):
                with Container(id="preview-command"):
                    yield Static("", id="command-text", classes="command-text")

                with Container(id="preview-hints"):
                    yield Static("", id="hints-text")

    def update_command(self, command: str) -> None:
        """Update the command preview.

        Args:
            command: The command string to display
        """
        self.command = command
        self._refresh_display()

    def update_hints(self, show_wandb_hint: bool = False, validation_errors: list[str] | None = None) -> None:
        """Update contextual hints.

        Args:
            show_wandb_hint: Whether to show W&B LEET hint
            validation_errors: List of validation error messages
        """
        self.show_wandb_hint = show_wandb_hint
        self.validation_errors = validation_errors or []
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh the preview display."""
        # Update command text
        try:
            command_widget = self.query_one("#command-text", Static)
        except Exception:
            # Not mounted yet in isolated widget tests
            return
        if self.command:
            # Format command with line breaks for readability
            formatted_command = self._format_command(self.command)
            command_widget.update(formatted_command)
        else:
            command_widget.update("[dim]No command generated yet[/dim]")

        # Update hints
        try:
            hints_widget = self.query_one("#hints-text", Static)
        except Exception:
            return
        hints_parts = []

        # W&B LEET hint
        if self.show_wandb_hint:
            hints_parts.append(
                "[bold green]💡 W&B Visualizer Active[/bold green]\n"
                "[green]The W&B LEET visualizer will launch automatically "
                "when training starts.[/green]"
            )

        # Validation errors
        if self.validation_errors:
            for error in self.validation_errors:
                hints_parts.append(f"[bold red]⚠ {error}[/bold red]")

        # Additional hints based on command
        if self.command:
            if "--quantization" in self.command:
                hints_parts.append("[yellow]💡 Quantization enabled - may reduce memory usage[/yellow]")
            if "--mixed_precision" in self.command:
                hints_parts.append("[yellow]💡 Mixed precision training enabled[/yellow]")

        if hints_parts:
            hints_text = "\n\n".join(hints_parts)
            hints_widget.update(hints_text)
        else:
            hints_widget.update("")

    def _format_command(self, command: str) -> str:
        """Format command for better readability.

        Args:
            command: Raw command string

        Returns:
            Formatted command with syntax highlighting
        """
        # Split on long commands for readability
        parts = command.split(" ")

        # Highlight the binary
        if parts:
            parts[0] = f"[bold cyan]{parts[0]}[/bold cyan]"

        # Highlight flags
        formatted_parts = []
        for part in parts:
            if part.startswith("--"):
                # Split flag and value
                if "=" in part:
                    flag, value = part.split("=", 1)
                    formatted_parts.append(f"[bold magenta]{flag}[/bold magenta]=[green]{value}[/green]")
                else:
                    formatted_parts.append(f"[bold magenta]{part}[/bold magenta]")
            else:
                formatted_parts.append(part)

        # Join with smart line breaks (every ~80 chars or at flags)
        result_lines = []
        current_line = []
        current_length = 0

        for part in formatted_parts:
            # Remove markup for length calculation
            clean_part = self._strip_markup(part)
            part_length = len(clean_part)

            if current_length + part_length + 1 > 70 and current_line:
                result_lines.append(" ".join(current_line))
                current_line = [part]
                current_length = part_length
            else:
                current_line.append(part)
                current_length += part_length + 1

        if current_line:
            result_lines.append(" ".join(current_line))

        return "\n  ".join(result_lines)

    def _strip_markup(self, text: str) -> str:
        """Strip Rich markup from text for length calculation.

        Args:
            text: Text with Rich markup

        Returns:
            Plain text without markup
        """
        import re

        return re.sub(r"\[.*?\]", "", text)

    def on_mount(self) -> None:
        """Handle mount event."""
        self._refresh_display()
