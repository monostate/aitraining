"""Custom footer widget with persistent keybinding tips and status bar."""

import os

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static


class CustomFooter(Container):
    """Custom footer displaying status bar and keybinding tips."""

    DEFAULT_CSS = """
    CustomFooter {
        layout: vertical;
        dock: bottom;
        height: 4;
        background: $panel;
        border-top: solid $primary;
    }

    #footer-status-row {
        layout: horizontal;
        height: 2;
        align: left middle;
        padding: 0 1;
        border-bottom: solid $secondary;
    }

    #footer-keybinds-row {
        layout: horizontal;
        height: 1;
        align: center middle;
        padding: 0 1;
    }

    .status-text {
        padding: 0 1;
        width: 100%;
    }

    .keybind-tip {
        padding: 0 2;
        color: $text;
    }

    .keybind-key {
        color: $success;
        text-style: bold;
    }

    .keybind-sep {
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the custom footer."""
        super().__init__(**kwargs)
        self.trainer = "default"
        self.model = "google/gemma-3-270m"  # Default model
        self.dataset = None

    def compose(self) -> ComposeResult:
        """Compose the footer UI with status bar on top and keybindings below."""
        # Status bar row
        with Horizontal(id="footer-status-row"):
            yield Static("Initializing status...", id="footer-status-text", classes="status-text")

        # Keybindings row
        with Horizontal(id="footer-keybinds-row"):
            yield Static(
                "[#10b981]F1[/#10b981] Help [dim]•[/dim] "
                "[#10b981]Ctrl+R[/#10b981] Run [dim]•[/dim] "
                "[#10b981]t[/#10b981] Tokens [dim]•[/dim] "
                "[#10b981]/[/#10b981] Search",
                classes="keybind-tip",
                markup=True,
            )

    def update_status(self, trainer: str | None = None, model: str | None = None, dataset: str | None = None) -> None:
        """Update status bar information.

        Args:
            trainer: Current trainer type
            model: Current model name
            dataset: Current dataset name
        """
        if trainer is not None:
            self.trainer = trainer
        if model is not None:
            self.model = model
        if dataset is not None:
            self.dataset = dataset

        self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh the status bar display."""
        status_parts = []

        # Trainer
        trainer_display = self.trainer.upper() if self.trainer else "DEFAULT"
        status_parts.append(f"[bold cyan]Trainer:[/bold cyan] [bold green]{trainer_display}[/bold green]")

        # Model
        if self.model:
            # Check if it's the default model
            if self.model == "google/gemma-3-270m":
                status_parts.append(
                    f"[bold cyan]Model:[/bold cyan] [bold green]DEFAULT (google/gemma-3-270m)[/bold green]"
                )
            else:
                model_short = self.model.split("/")[-1] if "/" in self.model else self.model
                if len(model_short) > 25:
                    model_short = model_short[:22] + "..."
                status_parts.append(f"[bold cyan]Model:[/bold cyan] [bold green]{model_short}[/bold green]")
        else:
            status_parts.append(f"[bold cyan]Model:[/bold cyan] [bold red]Not set[/bold red]")

        # Dataset
        if self.dataset:
            dataset_short = self.dataset.split("/")[-1] if "/" in self.dataset else self.dataset
            if len(dataset_short) > 25:
                dataset_short = dataset_short[:22] + "..."
            status_parts.append(f"[bold cyan]Dataset:[/bold cyan] [bold green]{dataset_short}[/bold green]")
        else:
            status_parts.append(f"[bold cyan]Dataset:[/bold cyan] [bold red]Not set[/bold red]")

        # HF Token status
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            status_parts.append(f"[bold cyan]HF:[/bold cyan] [bold green]✓[/bold green]")
        else:
            status_parts.append(f"[bold cyan]HF:[/bold cyan] [bold red]✗[/bold red]")

        # W&B Token status
        wandb_token = os.environ.get("WANDB_API_KEY", "")
        if wandb_token:
            status_parts.append(f"[bold cyan]W&B:[/bold cyan] [bold green]✓[/bold green]")
        else:
            status_parts.append(f"[bold cyan]W&B:[/bold cyan] [bold red]✗[/bold red]")

        # Join with separator
        status_text = " [dim]│[/dim] ".join(status_parts)

        # Update the display
        try:
            status_widget = self.query_one("#footer-status-text", Static)
            status_widget.update(status_text)
        except Exception:
            # Widget not yet mounted; skip safely
            return

    def on_mount(self) -> None:
        """Handle mount event."""
        self._refresh_display()
