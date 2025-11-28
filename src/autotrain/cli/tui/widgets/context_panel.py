"""Context panel widget for displaying parameter information."""

from typing import Any, Dict, Optional

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class ContextPanel(Static):
    """Widget for displaying contextual help and information."""

    def __init__(self, **kwargs):
        """Initialize the context panel."""
        super().__init__(**kwargs)
        self.current_field: Optional[Dict] = None
        self.current_value: Any = None

    def compose(self) -> ComposeResult:
        """Compose the context panel."""
        with Vertical(id="context-container"):
            yield Static(
                Panel(
                    "Select a parameter to see details",
                    title="📋 Context",
                    border_style="dim",
                ),
                id="context-content",
            )

    def show_field_info(self, field: Dict, current_value: Any = None) -> None:
        """Display information about a field."""
        self.current_field = field
        self.current_value = current_value

        # Build info panel
        field_name = field["arg"].lstrip("-").replace("-", "_")
        field_type = field.get("type", str).__name__
        field_help = field.get("help", "No description available")
        field_default = field.get("default")
        field_scope = field.get("scope", ["all"])
        field_group = field.get("group", "Other")

        # Create a table for field info
        table = Table(show_header=False, box=None, padding=0)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value")

        table.add_row("Field", field_name)
        table.add_row("Type", field_type)
        table.add_row("Group", field_group)

        # Scope
        if "all" in field_scope:
            scope_str = "All trainers"
        else:
            scope_str = ", ".join(field_scope)
        table.add_row("Scope", scope_str)

        # Default value
        if field_default is not None:
            table.add_row("Default", str(field_default))

        # Current value
        if current_value is not None and current_value != field_default:
            table.add_row("Current", f"[yellow]{current_value}[/yellow]")

        # Help text - with special handling for chat_template
        if field_name == "chat_template":
            # Enhanced help for chat_template field
            help_text = (
                "[bold cyan]Chat Template Configuration[/bold cyan]\n\n"
                "[yellow]⚠️ Only change this if you know what you're doing![/yellow]\n\n"
                f"{field_help}\n\n"
                "[bold]Available Options:[/bold]\n\n"
                "• [green]tokenizer[/green] (recommended) - Auto-detects the model's built-in template\n"
                "  Example: Gemma uses <start_of_turn>, Llama uses <|start_header_id|>\n\n"
                "• [cyan]none[/cyan] - No template, plain text training (for pretraining/completion)\n"
                "  Example: 'The quick brown fox jumps over...'\n\n"
                "• [dim]chatml[/dim] - ChatML format\n"
                "  Example: <|im_start|>user\\nHello<|im_end|>\\n<|im_start|>assistant\\n...\n\n"
                "• [dim]zephyr[/dim] - Zephyr format\n"
                "  Example: <|user|>\\nHello<|assistant|>\\n...\n\n"
                "• [dim]alpaca[/dim] - Alpaca instruction format\n"
                "• [dim]vicuna[/dim] - Vicuna conversation format\n"
                "• [dim]llama[/dim] - Llama chat format\n"
                "• [dim]mistral[/dim] - Mistral instruction format\n\n"
                "[bold]Default Behavior:[/bold]\n"
                "• SFT/DPO/ORPO/Reward trainers → 'tokenizer' (auto-detect)\n"
                "• Default trainer (pretraining) → None (plain text)\n\n"
                "[bold red]Important:[/bold red] Your dataset format must match the template!\n"
                '• With template: Data needs JSON chat format [{"role":"user","content":"..."}]\n'
                "• Without template: Data can be plain text"
            )
            help_panel = Panel(
                help_text,
                title="Chat Template Help",
                border_style="cyan",
                padding=(0, 1),
            )
        else:
            help_panel = Panel(
                field_help,
                title="Description",
                border_style="dim",
                padding=(0, 1),
            )

        # Validation info
        validation_text = self._get_validation_info(field, current_value)
        if validation_text:
            validation_panel = Panel(
                validation_text,
                title="Validation",
                border_style="green" if "✓" in validation_text else "yellow",
                padding=(0, 1),
            )
        else:
            validation_panel = None

        # Combine all elements as a single renderable group (avoid mounting unattached widgets)
        elements = [table, help_panel]
        if validation_panel:
            elements.append(validation_panel)
        content_group = Group(*elements)

        # Update the display
        context_content = self.query_one("#context-content", Static)
        context_content.update(Panel(content_group, title=f"📋 {field_name}", border_style="blue"))

    def _get_validation_info(self, field: Dict, value: Any) -> str:
        """Get validation information for a field."""
        field_name = field["arg"].lstrip("-").replace("-", "_")
        field_type = field.get("type", str)

        validation_messages = []

        # Type validation
        if value is not None and value != "":
            if field_type == int:
                try:
                    int(value)
                    validation_messages.append("[OK] Valid integer")
                except (ValueError, TypeError):
                    validation_messages.append("[!] Must be an integer")
            elif field_type == float:
                try:
                    float(value)
                    validation_messages.append("[OK] Valid number")
                except (ValueError, TypeError):
                    validation_messages.append("[!] Must be a number")

        # JSON validation for JSON fields
        if field_name in [
            "sweep_params",
            "token_weights",
            "custom_loss_weights",
            "custom_metrics",
            "rl_reward_weights",
            "rl_env_config",
        ]:
            if value and isinstance(value, str):
                import json

                try:
                    json.loads(value)
                    validation_messages.append("[OK] Valid JSON")
                except json.JSONDecodeError as e:
                    validation_messages.append(f"[!] Invalid JSON: {str(e)}")

        # Special validations
        if field_name == "lr" and value:
            try:
                lr = float(value)
                if lr <= 0:
                    validation_messages.append("[!] Learning rate must be positive")
                elif lr > 1:
                    validation_messages.append("[!] Learning rate unusually high (>1)")
            except (ValueError, TypeError):
                pass

        if field_name == "batch_size" and value:
            try:
                bs = int(value)
                if bs <= 0:
                    validation_messages.append("[!] Batch size must be positive")
            except (ValueError, TypeError):
                pass

        if field_name == "epochs" and value:
            try:
                epochs = int(value)
                if epochs <= 0:
                    validation_messages.append("[!] Epochs must be positive")
            except (ValueError, TypeError):
                pass

        return "\n".join(validation_messages) if validation_messages else ""

    def clear(self) -> None:
        """Clear the context panel."""
        self.current_field = None
        self.current_value = None
        context_content = self.query_one("#context-content", Static)
        context_content.update(
            Panel(
                "Select a parameter to see details",
                title="📋 Context",
                border_style="dim",
            )
        )
