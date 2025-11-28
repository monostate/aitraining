from unittest.mock import MagicMock, patch

import pytest

from autotrain.metadata.catalog import (
    CatalogEntry,
    _fetch_trending,
    _hf_list_top_datasets,
    _hf_list_top_models,
    format_params,
    get_popular_datasets,
    get_popular_models,
)


def test_get_popular_models_llm_fallback():
    models = get_popular_models("llm", "sft")
    assert len(models) > 0


def test_get_popular_datasets_llm_variants():
    sft_datasets = get_popular_datasets("llm", "sft")
    dpo_datasets = get_popular_datasets("llm", "dpo")
    assert len(sft_datasets) >= 1
    assert len(dpo_datasets) >= 1
    # ensure variant-specific lists differ
    assert sft_datasets[0].id != dpo_datasets[0].id


def test_format_params():
    """Test parameter count formatting."""
    assert format_params(None) == ""
    assert format_params(1000) == "(1K)"
    assert format_params(1500000) == "(2M)"
    assert format_params(2700000000) == "(2.7B)"
    assert format_params(1200000000000) == "(1.2T)"


@patch("requests.get")
def test_fetch_trending_success(mock_get):
    """Test successful trending fetch."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "recentlyTrending": [
            {"repoData": {"repoId": "model1", "numParameters": 1000000}},
            {"repoId": "model2"},
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    entries = _fetch_trending("model", limit=5)

    assert len(entries) == 2
    assert entries[0].id == "model1"
    assert entries[0].params == 1000000
    assert entries[1].id == "model2"
    assert entries[1].params is None


@patch("requests.get")
def test_fetch_trending_failure(mock_get):
    """Test trending fetch failure handling."""
    mock_get.side_effect = Exception("Network error")

    entries = _fetch_trending("model")
    assert entries == []


@patch("huggingface_hub.HfApi")
def test_hf_list_top_models_with_search(mock_hf_api_class):
    """Test model listing with search query."""
    mock_api = MagicMock()
    mock_hf_api_class.return_value = mock_api

    mock_model = MagicMock()
    mock_model.modelId = "searched/model"
    mock_api.list_models.return_value = [mock_model]

    entries = _hf_list_top_models(trainer_type="llm", trainer_variant="sft", search_query="gemma")

    assert len(entries) == 1
    assert entries[0].id == "searched/model"

    # Verify search parameter was passed
    call_args = mock_api.list_models.call_args
    assert "search" in call_args[1]
    assert call_args[1]["search"] == "gemma"


@patch("requests.get")
def test_hf_list_top_datasets_with_sort(mock_get):
    """Test dataset listing with different sort options."""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"id": "dataset1"},
        {"id": "dataset2"},
    ]
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    # Test with downloads sort
    entries = _hf_list_top_datasets(trainer_type="text-classification", trainer_variant=None, sort_by="downloads")

    assert len(entries) == 2
    assert entries[0].id == "dataset1"

    # Verify sort parameter
    call_args = mock_get.call_args
    assert call_args[1]["params"]["sort"] == "downloads"


def test_get_popular_models_with_sort_and_search():
    """Test get_popular_models with sort and search parameters."""
    # Clear the cache first
    get_popular_models.cache_clear()

    # Test with search (should attempt to fetch from hub)
    with patch("autotrain.metadata.catalog._hf_list_top_models") as mock_hf:
        mock_hf.return_value = [CatalogEntry("found/model", "Found Model")]

        models = get_popular_models(
            trainer_type="llm", trainer_variant="sft", sort_by="downloads", search_query="test"
        )

        # Verify it tried to fetch with search
        mock_hf.assert_called_once_with("llm", "sft", limit=20, sort_by="downloads", search_query="test")


def test_get_popular_datasets_with_trending():
    """Test get_popular_datasets with trending sort."""
    # Clear the cache first
    get_popular_datasets.cache_clear()

    with patch("autotrain.metadata.catalog._fetch_trending") as mock_trending:
        mock_trending.return_value = [
            CatalogEntry("trending/dataset1", "Trending 1"),
            CatalogEntry("trending/dataset2", "Trending 2"),
        ]

        datasets = get_popular_datasets(trainer_type="llm", trainer_variant="sft", sort_by="trending")

        # Should call trending API for datasets
        mock_trending.assert_called_once_with("dataset", 20)
