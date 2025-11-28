"""Shared catalog metadata for popular models and datasets."""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

from autotrain import logger
from autotrain.cli.trainer_metadata import TRAINER_METADATA


#
# Note: We deliberately avoid importing any FastAPI modules at import-time so
# this catalog remains usable from CLI/TUI contexts. We prefer huggingface_hub
# for live listings and fall back to small curated lists when offline.
#


@dataclass(frozen=True)
class CatalogEntry:
    """Simple container for model/dataset metadata."""

    id: str
    label: str
    description: str = ""
    params: Optional[int] = None


def format_params(num_params: Optional[int]) -> str:
    """Format parameter count into human-readable string."""
    if not num_params:
        return ""
    if num_params >= 1e12:
        return f"({num_params/1e12:.1f}T)"
    elif num_params >= 1e9:
        return f"({num_params/1e9:.1f}B)"
    elif num_params >= 1e6:
        return f"({num_params/1e6:.0f}M)"
    else:
        return f"({num_params/1e3:.0f}K)"


def _trainer_key(trainer_type: Optional[str], trainer_variant: Optional[str]) -> str:
    """Normalize trainer identifiers to a catalog key."""
    if not trainer_type:
        return "llm:sft"
    if trainer_type == "llm":
        variant = trainer_variant or "sft"
        return f"llm:{variant}"
    return trainer_type


def _model_catalog_fallback(trainer_key: str) -> List[CatalogEntry]:
    """Return static fallback models when dynamic fetch is unavailable."""
    if trainer_key.startswith("llm:dpo"):
        entries = [
            CatalogEntry("mistralai/Mistral-7B-Instruct-v0.2", "Mistral 7B Instruct"),
            CatalogEntry("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B Instruct"),
            CatalogEntry("HuggingFaceH4/zephyr-7b-beta", "Zephyr 7B Beta"),
        ]
    elif trainer_key.startswith("llm:ppo"):
        entries = [
            CatalogEntry("meta-llama/Llama-3.2-1B", "Llama 3.2 1B"),
            CatalogEntry("Qwen/Qwen2.5-0.5B", "Qwen2.5 0.5B"),
        ]
    elif trainer_key.startswith("llm"):
        entries = [
            CatalogEntry("google/gemma-3-270m", "Gemma 3 270M (default)"),
            CatalogEntry("meta-llama/Llama-3.2-1B", "Llama 3.2 1B"),
            CatalogEntry("meta-llama/Llama-3.2-3B", "Llama 3.2 3B"),
            CatalogEntry("Qwen/Qwen2.5-0.5B", "Qwen2.5 0.5B"),
            CatalogEntry("HuggingFaceTB/SmolLM2-360M", "SmolLM2 360M"),
        ]
    else:
        metadata = TRAINER_METADATA.get(trainer_key)
        starter = metadata.get("starter_models", []) if metadata else []
        entries = [CatalogEntry(model_id, model_id) for model_id in starter]
    return entries


def _dataset_catalog_fallback(trainer_key: str) -> List[CatalogEntry]:
    """Static curated datasets for each trainer."""
    catalog: Dict[str, List[CatalogEntry]] = {
        "llm:sft": [
            CatalogEntry("tatsu-lab/alpaca", "Stanford Alpaca", "52k instruction-following samples"),
            CatalogEntry("yahma/alpaca-cleaned", "Alpaca Cleaned", "Cleaner variant of Alpaca"),
            CatalogEntry("HuggingFaceH4/ultrachat_200k", "UltraChat 200k", "Long-form chat prompts"),
            CatalogEntry("Open-Orca/SlimOrca", "SlimOrca", "Preference distilled Orca data"),
            CatalogEntry("LDJnr/Puffin", "Puffin", "English instruction fine-tuning"),
        ],
        "llm:dpo": [
            CatalogEntry(
                "argilla/ultrafeedback-binarized-preferences", "UltraFeedback", "Preference pairs for DPO/ORPO"
            ),
            CatalogEntry("Anthropic/hh-rlhf", "HH RLHF", "Helpful/Harmless preference data"),
            CatalogEntry("OpenAssistant/oasst_top1_2023-08-25", "OpenAssistant Top-1", "Human preference picks"),
        ],
        "llm:orpo": [
            CatalogEntry("argilla/ultrafeedback-binarized-preferences", "UltraFeedback", "Preference pairs for ORPO"),
            CatalogEntry("mosaicml/instruct-v3", "Mosaic Instruct", "Paired responses for preference training"),
        ],
        "llm:ppo": [
            CatalogEntry("Open-Orca/SlimOrca", "SlimOrca", "Pairs suitable for reward modeling"),
            CatalogEntry("Intel/orca_dpo_pairs", "Orca DPO Pairs", "Short preference prompts"),
            CatalogEntry("openbmb/UltraInteract_pair", "UltraInteract", "Conversational preference data"),
        ],
        "text-classification": [
            CatalogEntry("ag_news", "AG News", "News topic classification"),
            CatalogEntry("yelp_review_full", "Yelp Review Full", "5-class sentiment"),
            CatalogEntry("dbpedia_14", "DBPedia 14", "Ontology classification"),
        ],
        "token-classification": [
            CatalogEntry("conll2003", "CoNLL 2003", "NER dataset"),
            CatalogEntry("wnut_17", "WNUT 2017", "Emerging entities"),
        ],
        "tabular": [
            CatalogEntry("autogluon/ames", "Ames Housing", "Regression benchmark"),
            CatalogEntry("autogluon/otto", "Otto Group", "Multi-class classification"),
        ],
        "image-classification": [
            CatalogEntry("huggingface/cifar10", "CIFAR-10", "10-class images"),
            CatalogEntry("huggingface/beans", "Beans", "Plant disease detection"),
        ],
        "image-regression": [
            CatalogEntry("Shenggan/BraTS-GLIOMA", "BraTS Glioma", "Medical regression labels"),
        ],
        "extractive-qa": [
            CatalogEntry("squad", "SQuAD v1.1", "Reading comprehension"),
            CatalogEntry("rajpurkar/squad_v2", "SQuAD v2", "Q&A with unanswerables"),
        ],
        "seq2seq": [
            CatalogEntry("cnn_dailymail", "CNN/DailyMail", "Summarization"),
            CatalogEntry("wmt16", "WMT16", "Machine translation corpus"),
        ],
        "sent-transformers": [
            CatalogEntry("sentence-transformers/all-nli", "AllNLI", "STS + NLI pairs"),
        ],
        "vlm": [
            CatalogEntry("laion/OIG", "OIG", "Open Instruction Generalist prompts"),
            CatalogEntry("llava-bench-instruct/llava-bench-instruct", "LLaVA-Bench", "VLM instructions"),
        ],
    }
    return catalog.get(trainer_key, [])


def _map_trainer_to_model_pipeline(trainer_type: Optional[str], trainer_variant: Optional[str]) -> Optional[str]:
    """Best-effort mapping from trainer to HF model pipeline_tag."""
    if not trainer_type:
        return "text-generation"
    if trainer_type == "llm":
        # SFT/DPO/ORPO/Reward/PPO all ultimately fine-tune text-generation models
        return "text-generation"
    return {
        "text-classification": "text-classification",
        "token-classification": "token-classification",
        "seq2seq": "summarization",  # heuristic; could be translation too
        "extractive-qa": "question-answering",
        "sent-transformers": "sentence-similarity",
        "vlm": "image-text-to-text",
        "image-classification": "image-classification",
        "image-regression": "image-classification",  # no direct regression pipeline in hub listing
        "tabular": None,
    }.get(trainer_type, None)


def _map_trainer_to_dataset_task(trainer_type: Optional[str], trainer_variant: Optional[str]) -> Optional[str]:
    """Best-effort mapping from trainer to HF datasets task_categories filter."""
    if not trainer_type:
        return None
    if trainer_type == "llm":
        # No clear 'text-generation' dataset category; keep curated fallback for LLM
        return None
    return {
        "text-classification": "text-classification",
        "token-classification": "token-classification",
        "seq2seq": "summarization",  # heuristic; "translation" also valid depending on use
        "extractive-qa": "question-answering",
        "sent-transformers": "sentence-similarity",
        "vlm": "visual-question-answering",
        "image-classification": "image-classification",
        "image-regression": "image-classification",
        "tabular": None,
    }.get(trainer_type, None)


def _fetch_trending(repo_type: str, limit: int = 20) -> List[CatalogEntry]:
    """Fetch trending models/datasets from Hugging Face."""
    try:
        import requests

        resp = requests.get(f"https://huggingface.co/api/trending?type={repo_type}&limit={limit}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        entries: List[CatalogEntry] = []
        # The API returns a list of objects, sometimes nested
        current_list = data.get("recentlyTrending") if isinstance(data, dict) else data

        for item in current_list:
            repo_id = item.get("repoData", {}).get("repoId") or item.get("repoId")
            if not repo_id:
                continue

            # Extract useful info
            params = item.get("repoData", {}).get("numParameters") or item.get("numParameters")

            entries.append(CatalogEntry(repo_id, repo_id, "", params))
            if len(entries) >= limit:
                break
        return entries
    except Exception as exc:
        logger.debug(f"Unable to fetch trending {repo_type}: {exc}")
        return []


def _hf_list_top_models(
    trainer_type: Optional[str],
    trainer_variant: Optional[str],
    limit: int = 20,
    sort_by: str = "downloads",
    search_query: Optional[str] = None,
) -> List[CatalogEntry]:
    """Fetch top models from Hugging Face hub via huggingface_hub."""
    try:
        from huggingface_hub import HfApi

        pipeline = _map_trainer_to_model_pipeline(trainer_type, trainer_variant)
        # If searching, we might want to relax pipeline filter if it restricts too much,
        # but keeping it ensures relevant models.

        api = HfApi()

        # Map sort_by to HfApi sort
        sort_key = sort_by
        direction = -1

        if sort_by == "trending":
            # HfApi doesn't support trending sort directly, use the trending API
            return _fetch_trending("model", limit)

        if sort_by == "recent":
            sort_key = "createdAt"
        elif sort_by == "modified":
            sort_key = "lastModified"

        args = {
            "sort": sort_key,
            "direction": direction,
            "limit": limit,
        }

        if pipeline:
            args["filter"] = pipeline

        if search_query:
            args["search"] = search_query

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.utils._deprecation")
            results = api.list_models(**args)
        entries: List[CatalogEntry] = []
        # results may be ModelInfo objects
        for r in results:
            model_id = getattr(r, "modelId", None) or getattr(r, "id", None) or str(r)
            if not model_id:
                continue

            # Try to get params if available (ModelInfo might have it in safetensors metadata)
            # but HfApi list_models might not return full details.
            # We'll accept it might be None.
            params = None
            # getattr(r, "safetensors", {}) ...

            entries.append(CatalogEntry(model_id, model_id, "", params))
        return entries
    except Exception as exc:
        logger.debug(f"Unable to fetch models from HF hub: {exc}")
        return []


def _hf_list_top_datasets(
    trainer_type: Optional[str],
    trainer_variant: Optional[str],
    limit: int = 20,
    sort_by: str = "downloads",
    search_query: Optional[str] = None,
) -> List[CatalogEntry]:
    """Fetch top datasets from Hugging Face hub via raw REST API (task filter)."""
    try:
        # Use requests to call the public REST endpoint for datasets
        import os

        import requests

        if sort_by == "trending" and not search_query:
            return _fetch_trending("dataset", limit)

        task = _map_trainer_to_dataset_task(trainer_type, trainer_variant)

        params = {
            "sort": sort_by if sort_by != "recent" else "createdAt",
            "limit": str(limit),
        }

        if sort_by == "trending":
            # Fallback if search is used with trending (API doesn't support search+trending easily)
            # We'll just search and sort by downloads
            params["sort"] = "downloads"

        if task:
            params["task_categories"] = task

        if search_query:
            params["search"] = search_query

        headers = {}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        resp = requests.get("https://huggingface.co/api/datasets", params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        entries: List[CatalogEntry] = []
        for item in data:
            ds_id = item.get("id")
            if not ds_id:
                continue

            # Dataset size/rows?
            # Usually not in list response
            entries.append(CatalogEntry(ds_id, ds_id))
        return entries
    except Exception as exc:
        logger.debug(f"Unable to fetch datasets from HF hub: {exc}")
        return []


@functools.lru_cache(maxsize=64)
def get_popular_models(
    trainer_type: Optional[str],
    trainer_variant: Optional[str] = None,
    sort_by: str = "trending",
    search_query: Optional[str] = None,
) -> List[CatalogEntry]:
    """Return curated model IDs for the given trainer, preferring hub metadata when available."""
    trainer_key = _trainer_key(trainer_type, trainer_variant)

    # 1) Try HF hub directly (works in CLI/TUI without FastAPI)
    entries = _hf_list_top_models(trainer_type, trainer_variant, limit=20, sort_by=sort_by, search_query=search_query)
    if entries:
        return entries

    # 2) Try UI-centric model map if available (FastAPI context) - Only if no search/sort
    if sort_by == "trending" and not search_query:
        try:
            from autotrain.app.models import fetch_models  # optional

            model_map = fetch_models()
            model_key = {
                "text-classification": "text-classification",
                "token-classification": "token-classification",
                "tabular": "tabular-classification",
                "image-classification": "image-classification",
                "image-regression": "image-regression",
                "seq2seq": "seq2seq",
                "extractive-qa": "extractive-qa",
                "sent-transformers": "sentence-transformers",
                "vlm": "vlm",
            }.get(trainer_type or "llm", "llm")
            hub_entries = model_map.get(model_key, [])
            if hub_entries:
                return [CatalogEntry(model_id, model_id) for model_id in hub_entries[:20]]
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Unable to fetch hub model catalog via app.models: {exc}")

    # 3) Fallback curated list
    if not search_query:
        return _model_catalog_fallback(trainer_key)

    return []


@functools.lru_cache(maxsize=64)
def get_popular_datasets(
    trainer_type: Optional[str],
    trainer_variant: Optional[str] = None,
    sort_by: str = "trending",
    search_query: Optional[str] = None,
) -> List[CatalogEntry]:
    """Return curated dataset IDs for the given trainer."""
    trainer_key = _trainer_key(trainer_type, trainer_variant)

    # 1) Try HF hub for supported task categories
    entries = _hf_list_top_datasets(
        trainer_type, trainer_variant, limit=20, sort_by=sort_by, search_query=search_query
    )
    if entries:
        return entries

    # 2) Fallback curated list (only if no search)
    if not search_query:
        entries = _dataset_catalog_fallback(trainer_key)
        if entries:
            return entries
        # 3) Final fallback to TRAINER_METADATA defaults
        if trainer_type and trainer_type in TRAINER_METADATA:
            metadata = TRAINER_METADATA[trainer_type]
            default = metadata.get("default_dataset")
            if default:
                return [CatalogEntry(default, default)]
    return []
