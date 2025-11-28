"""Group list widget for parameter organization."""

from typing import List

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Label, ListItem, ListView, Static


class GroupList(Static):
    """Widget for displaying and selecting parameter groups."""

    class GroupSelected(Message):
        """Message emitted when a group is selected."""

        def __init__(self, group: str):
            self.group = group
            super().__init__()

    def __init__(self, **kwargs):
        """Initialize the group list."""
        super().__init__(**kwargs)
        self.groups: List[str] = []
        self.current_group: str = "Basic"

    def compose(self) -> ComposeResult:
        """Compose the group list widget."""
        with Vertical(id="groups-container"):
            yield Label("📁 Parameter Groups", classes="section-label")
            yield ListView(id="groups-listview")

    def set_groups(self, groups: List[str]) -> None:
        """Set the available groups."""
        self.groups = groups
        self._update_list()

    def set_current_group(self, group: str) -> None:
        """Set the currently selected group."""
        self.current_group = group
        self._update_list()

    def _update_list(self) -> None:
        """Update the list view with groups."""
        listview = self.query_one("#groups-listview", ListView)
        listview.clear()

        for group in self.groups:
            # Add indicator for current group
            if group == self.current_group:
                label = Label(f"> {group}", classes="selected-group")
            else:
                label = Label(f"  {group}")

            item = ListItem(label)
            listview.append(item)

        # Highlight current group
        try:
            index = self.groups.index(self.current_group)
            listview.index = index
        except (ValueError, IndexError):
            pass

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle group selection."""
        if event.item:
            # Extract group name from the label
            label_widget = event.item.query_one(Label)
            renderable = getattr(label_widget, "renderable", "")
            # Support both Rich Text (with .plain) and plain str
            if hasattr(renderable, "plain"):
                text_value = renderable.plain
            elif isinstance(renderable, str):
                text_value = renderable
            else:
                text_value = str(renderable)

            group_name = text_value.strip()
            if group_name.startswith(">"):
                group_name = group_name[1:].strip()
            else:
                group_name = group_name.strip()

            self.post_message(self.GroupSelected(group_name))
