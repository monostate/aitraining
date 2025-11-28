"""Tokens modal for managing HuggingFace and W&B API tokens."""

import os

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static


class TokensModal(ModalScreen[bool]):
    """Modal screen for setting HuggingFace and Weights & Biases tokens.

    This modal allows users to update their API tokens which are used for:
    - HuggingFace: Model/dataset access and Hub uploads
    - W&B: Experiment tracking and visualization

    Tokens are stored in environment variables for the session.
    """

    DEFAULT_CSS = """
    TokensModal {
        align: center middle;
    }

    #tokens-modal-dialog {
        width: 75;
        height: 22;
        padding: 2;
        background: $panel;
        border: thick $primary;
    }
    """

    class TokensUpdated(Message):
        """Message emitted when tokens are updated."""

        def __init__(self, hf_token: str | None, wandb_token: str | None) -> None:
            self.hf_token = hf_token
            self.wandb_token = wandb_token
            super().__init__()

    def __init__(self) -> None:
        """Initialize the tokens modal."""
        super().__init__()
        self.hf_token_value = os.environ.get("HF_TOKEN", "")
        self.wandb_token_value = os.environ.get("WANDB_API_KEY", "")

    def compose(self) -> ComposeResult:
        """Compose the modal UI."""
        with Container(id="tokens-modal-dialog"):
            yield Static("🔑 API Tokens", classes="modal-title")

            with Vertical(classes="token-field"):
                yield Label("HuggingFace Token:", classes="token-label")
                yield Input(
                    value=self.hf_token_value,
                    placeholder="hf_...",
                    password=True,
                    id="hf-token-input",
                    classes="token-input",
                )
                yield Label("Used for model/dataset access and Hub uploads", classes="field-help")

            with Vertical(classes="token-field"):
                yield Label("Weights & Biases Token:", classes="token-label")
                yield Input(
                    value=self.wandb_token_value,
                    placeholder="Enter W&B API key",
                    password=True,
                    id="wandb-token-input",
                    classes="token-input",
                )
                yield Label("Used for experiment tracking and visualization", classes="field-help")

            with Container(classes="modal-buttons"):
                yield Button("Save", variant="success", id="save-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    @on(Button.Pressed, "#save-btn")
    def handle_save(self) -> None:
        """Handle save button press."""
        hf_input = self.query_one("#hf-token-input", Input)
        wandb_input = self.query_one("#wandb-token-input", Input)

        hf_token = hf_input.value.strip()
        wandb_token = wandb_input.value.strip()

        # Update environment variables
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        elif "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]

        if wandb_token:
            os.environ["WANDB_API_KEY"] = wandb_token
        elif "WANDB_API_KEY" in os.environ:
            del os.environ["WANDB_API_KEY"]

        # Post message to parent app
        self.app.post_message(
            self.TokensUpdated(
                hf_token=hf_token if hf_token else None, wandb_token=wandb_token if wandb_token else None
            )
        )

        self.dismiss(True)

    @on(Button.Pressed, "#cancel-btn")
    def handle_cancel(self) -> None:
        """Handle cancel button press."""
        self.dismiss(False)

    def on_key(self, event) -> None:
        """Handle key presses."""
        if event.key == "escape":
            self.dismiss(False)
