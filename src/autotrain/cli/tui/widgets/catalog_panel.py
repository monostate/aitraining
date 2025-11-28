"""Widgets for displaying curated model/dataset catalogs inside the TUI."""

from __future__ import annotations

import logging
from typing import List, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.message import Message
from textual.widgets import Label, ListItem, ListView, Static

from autotrain.metadata.catalog import CatalogEntry, get_popular_datasets, get_popular_models


logger = logging.getLogger(__name__)


class CatalogPanel(Static):
    """Enhanced catalog view with improved visual hierarchy and feedback.

    Features:
    - Bold labels with faint descriptions
    - Subtle toast/log notifications on selection
    - Better visual separation between models and datasets
    """

    DEFAULT_CSS = """
    CatalogPanel {
        height: 100%;
        padding: 0;
    }

    #catalog-container {
        height: 100%;
        padding: 0;
    }

    .catalog-section {
        height: 50%;
        padding: 1;
        border: round $primary;
        margin: 0 0 1 0;
        background: $panel;
    }

    .catalog-title {
        text-style: bold;
        color: $primary;
        padding: 1;
        background: $primary 10%;
        border-bottom: solid $primary;
        margin: 0 0 1 0;
    }

    #catalog-models-list {
        height: 1fr;
        background: $panel;
        border: none;
    }

    #catalog-datasets-list {
        height: 1fr;
        background: $panel;
        border: none;
    }

    .catalog-item-label {
        text-style: bold;
        color: $text;
    }

    .catalog-item-desc {
        color: $text-muted;
        text-style: italic;
    }

    .empty-state {
        text-align: center;
        color: $text-muted;
        padding: 2;
        text-style: italic;
    }
    """

    class ApplyModel(Message):
        """Posted when the user selects a model."""

        def __init__(self, model_id: str):
            self.model_id = model_id
            super().__init__()

    class ApplyDataset(Message):
        """Posted when the user selects a dataset."""

        def __init__(self, dataset_id: str):
            self.dataset_id = dataset_id
            super().__init__()

    class ShowToast(Message):
        """Posted to request a toast notification."""

        def __init__(self, message: str, type: str = "success"):
            self.message = message
            self.type = type
            super().__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_entries: List[CatalogEntry] = []
        self._dataset_entries: List[CatalogEntry] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="catalog-container"):
            # Models section
            with Container(classes="catalog-section"):
                yield Static("✨ Popular Models", classes="catalog-title")
                yield ListView(id="catalog-models-list")

            # Datasets section
            with Container(classes="catalog-section"):
                yield Static("📚 Popular Datasets", classes="catalog-title")
                yield ListView(id="catalog-datasets-list")

    def update_catalog(self, trainer_type: str = "llm", trainer_variant: Optional[str] = None) -> None:
        """Refresh catalog entries based on the selected trainer."""
        try:
            self._model_entries = get_popular_models(trainer_type, trainer_variant)[:12]
            self._dataset_entries = get_popular_datasets(trainer_type, trainer_variant)[:12]
            self._refresh_list("catalog-models-list", self._model_entries, empty_message="No curated models available")
            self._refresh_list(
                "catalog-datasets-list", self._dataset_entries, empty_message="No curated datasets available"
            )
            logger.debug(
                f"Catalog updated for {trainer_type}/{trainer_variant}: {len(self._model_entries)} models, {len(self._dataset_entries)} datasets"
            )
        except Exception as e:
            logger.error(f"Failed to update catalog: {e}")

    def _refresh_list(self, list_id: str, entries: List[CatalogEntry], empty_message: str) -> None:
        """Refresh a catalog list with enhanced formatting."""
        try:
            list_view = self.query_one(f"#{list_id}", ListView)
        except Exception:
            # Not mounted yet
            return

        list_view.clear()

        if not entries:
            list_view.append(ListItem(Label(empty_message, classes="empty-state")))
            return

        for entry in entries:
            # Create rich text with bold label and faint description
            label_text = Text()

            # Bold label
            label = entry.label if entry.label else entry.id
            label_text.append(label, style="bold")

            # Faint description
            if entry.description:
                label_text.append("\n  ")
                label_text.append(entry.description, style="dim italic")

            item = CatalogListItem(entry, label_text)
            list_view.append(item)

    def on_mount(self) -> None:
        """Initialize the catalog with default entries when mounted."""
        # Populate with default LLM catalog entries
        self.update_catalog("llm", "default")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Dispatch apply events with toast notification when an item is selected."""
        item = event.item
        if not isinstance(item, CatalogListItem):
            return

        # Determine which list and post appropriate message
        if event.list_view.id == "catalog-models-list":
            model_name = item.entry.label if item.entry.label else item.entry.id
            self.post_message(self.ApplyModel(item.entry.id))
            # Post toast notification
            self.post_message(self.ShowToast(f"✓ Applied model: {model_name}", "success"))
        elif event.list_view.id == "catalog-datasets-list":
            dataset_name = item.entry.label if item.entry.label else item.entry.id
            self.post_message(self.ApplyDataset(item.entry.id))
            # Post toast notification
            self.post_message(self.ShowToast(f"✓ Applied dataset: {dataset_name}", "success"))


class CatalogListItem(ListItem):
    """List item that stores the associated catalog entry with enhanced formatting."""

    def __init__(self, entry: CatalogEntry, label_text: Text | str):
        if isinstance(label_text, str):
            label_text = Text(label_text)
        super().__init__(Label(label_text))
        self.entry = entry
