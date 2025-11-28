"""Status bar widget showing current configuration state."""

import os

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static


class StatusBar(Widget):
    """Status bar displaying trainer, model, dataset, and token status.

    Shows:
    - Selected trainer type
    - Current model (if set)
    - Current dataset (if set)
    - HuggingFace token status (present/missing)
    - W&B token status (present/missing)
    """

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 3;
        background: $panel;
        border-top: solid $primary;
        padding: 0 1;
    }

    #status-content {
        layout: horizontal;
        height: 100%;
        align: left middle;
    }

    .status-item {
        padding: 0 1 0 0;
    }

    .status-label {
        color: $text-muted;
        text-style: italic;
    }

    .status-value {
        text-style: bold;
        color: $success;
    }

    .status-value-missing {
        text-style: bold;
        color: $error;
    }

    .status-separator {
        color: $text-disabled;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the status bar."""
        super().__init__(**kwargs)
        self.trainer = "default"
        self.model = "google/gemma-3-270m"  # Default model
        self.dataset = None

    def compose(self) -> ComposeResult:
        """Compose the status bar UI."""
        with Horizontal(id="status-content"):
            yield Static("", id="status-text", classes="status-item")

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
            status_widget = self.result if False else self.query_one("#status-text", Static)  # type: ignore[attr-defined]
            status_widget.update(status_text)
        except Exception:
            # Widget not yet mounted or child not composed in test context; skip safely.
            return

    def on_mount(self) -> None:
        """Handle mount event."""
        self._refresh_display()
