"""Trainer selector widget for AITraining TUI."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Label, Select, Static


class TrainerSelector(Static):
    """Widget for selecting the training mode/trainer."""

    # Available trainers
    TRAINERS = [
        ("default", "Default"),
        ("sft", "SFT (Supervised Fine-Tuning)"),
        ("dpo", "DPO (Direct Preference Optimization)"),
        ("orpo", "ORPO (Odds Ratio Preference Optimization)"),
        ("ppo", "PPO (Proximal Policy Optimization)"),
        ("reward", "Reward Model Training"),
    ]

    class TrainerChanged(Message):
        """Message emitted when trainer selection changes."""

        def __init__(self, trainer: str):
            self.trainer = trainer
            super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the trainer selector widget."""
        with Vertical(id="trainer-container"):
            yield Label("Trainer", classes="section-label")
            yield Select(
                options=[(label, value) for value, label in self.TRAINERS],
                value="default",
                id="trainer-select",
                allow_blank=False,
            )

    def set_trainer(self, trainer: str) -> None:
        """Set the current trainer."""
        select = self.query_one("#trainer-select", Select)
        if trainer in [t[0] for t in self.TRAINERS]:
            select.value = trainer

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle trainer selection change."""
        if event.value is not None:
            self.post_message(self.TrainerChanged(event.value))
