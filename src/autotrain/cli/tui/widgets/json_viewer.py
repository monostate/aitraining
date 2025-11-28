"""JSON viewer widget with copy functionality."""

import json


try:
    import pyperclip

    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widget import Widget
from textual.widgets import Button, Static, TextArea


class JsonViewer(Widget):
    """Widget for displaying and copying JSON configuration.

    Features:
    - Monospace formatted JSON display
    - Copy to clipboard button
    - Read-only view with syntax highlighting
    """

    DEFAULT_CSS = """
    JsonViewer {
        height: 100%;
        padding: 0;
    }

    #json-viewer-container {
        height: 100%;
        padding: 1;
        background: $panel;
        border: round $primary;
    }

    #json-header {
        height: auto;
        layout: horizontal;
        padding: 1;
        background: $primary 10%;
        border-bottom: solid $primary;
    }

    #json-title {
        text-style: bold;
        color: $primary;
        width: 1fr;
        content-align: left middle;
    }

    #json-copy-btn {
        width: auto;
    }

    #json-content {
        height: 1fr;
        padding: 1;
        background: $surface;
        border: round $primary;
        margin: 1 0 0 0;
    }

    #json-text-area {
        height: 100%;
        background: $surface;
    }

    .copy-feedback {
        color: $success;
        text-style: bold;
        padding: 0 1;
    }
    """

    class CopySuccess(Static):
        """Temporary success message widget."""

    def __init__(self) -> None:
        """Initialize the JSON viewer."""
        super().__init__()
        self.json_data = {}

    def compose(self) -> ComposeResult:
        """Compose the JSON viewer UI."""
        with Container(id="json-viewer-container"):
            with Horizontal(id="json-header"):
                yield Static("📄 Configuration JSON", id="json-title")
                yield Button("📋 Copy", id="json-copy-btn", variant="primary")

            with Container(id="json-content"):
                yield TextArea(
                    "", language="json", theme="monokai", read_only=True, show_line_numbers=True, id="json-text-area"
                )

    def update_json(self, data: dict) -> None:
        """Update the JSON display.

        Args:
            data: Dictionary to display as JSON
        """
        self.json_data = data
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh the JSON text area."""
        try:
            text_area = self.query_one("#json-text-area", TextArea)
        except Exception:
            # Not mounted yet in isolated widget tests
            return

        if self.json_data:
            # Format JSON with indentation
            formatted_json = json.dumps(self.json_data, indent=2, sort_keys=True)
            text_area.text = formatted_json
        else:
            text_area.text = "{\n  // No configuration data yet\n}"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle copy button press."""
        if event.button.id == "json-copy-btn":
            self._copy_to_clipboard()

    def _copy_to_clipboard(self) -> None:
        """Copy JSON to clipboard and show feedback."""
        if not self.json_data:
            self.notify("No data to copy", severity="warning", timeout=2)
            return

        if not HAS_PYPERCLIP:
            self.notify("Clipboard functionality requires pyperclip", severity="warning", timeout=3)
            return

        try:
            # Format JSON for clipboard
            formatted_json = json.dumps(self.json_data, indent=2, sort_keys=True)

            # Copy to clipboard
            pyperclip.copy(formatted_json)

            # Show success notification
            self.notify("✓ Copied to clipboard!", severity="information", timeout=2)

            # Briefly change button text
            button = self.query_one("#json-copy-btn", Button)
            original_label = button.label
            button.label = "✓ Copied!"

            # Reset button label after 1 second
            self.set_timer(1.0, lambda: setattr(button, "label", original_label))

        except Exception as e:
            self.notify(f"Failed to copy: {str(e)}", severity="error", timeout=3)

    def on_mount(self) -> None:
        """Handle mount event."""
        self._refresh_display()
