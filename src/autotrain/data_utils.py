"""
Centralized utilities for saving processed training data.

This module provides functions to save processed datasets locally and/or to the Hub,
with intelligent defaults based on dataset source and size.
"""

import os
from typing import Optional, Union

import pandas as pd
from datasets import Dataset

from autotrain import logger


# Size threshold in bytes (5GB)
SIZE_THRESHOLD_BYTES = 5 * 1024 * 1024 * 1024


def is_hub_dataset(data_path: str) -> bool:
    """
    Check if a dataset path refers to a HuggingFace Hub dataset.

    Args:
        data_path: The dataset path to check

    Returns:
        True if the path appears to be a Hub dataset (contains '/' and doesn't exist locally)
    """
    if not data_path:
        return False
    # Hub datasets have format "username/dataset-name" or "org/dataset-name"
    # Local paths either exist or start with . or /
    if os.path.exists(data_path):
        return False
    if data_path.startswith((".", "/", "~")):
        return False
    # Check if it looks like a Hub path (contains exactly one /)
    parts = data_path.split("/")
    return len(parts) == 2 and all(part.strip() for part in parts)


def estimate_dataset_size(dataset: Union[Dataset, pd.DataFrame]) -> int:
    """
    Estimate the size of a dataset in bytes.

    Args:
        dataset: The dataset to estimate size for

    Returns:
        Estimated size in bytes
    """
    try:
        if isinstance(dataset, pd.DataFrame):
            return dataset.memory_usage(deep=True).sum()
        elif hasattr(dataset, "data"):
            # HuggingFace Dataset
            return dataset.data.nbytes if hasattr(dataset.data, "nbytes") else 0
        else:
            # Fallback: convert to pandas and estimate
            df = dataset.to_pandas() if hasattr(dataset, "to_pandas") else pd.DataFrame(dataset)
            return df.memory_usage(deep=True).sum()
    except Exception as e:
        logger.warning(f"Could not estimate dataset size: {e}")
        return 0


def save_processed_dataset(
    dataset: Union[Dataset, pd.DataFrame],
    project_name: str,
    split_name: str,
    source_path: Optional[str] = None,
    username: Optional[str] = None,
    token: Optional[str] = None,
    save_mode: str = "auto",
    size_threshold_bytes: int = SIZE_THRESHOLD_BYTES,
) -> dict:
    """
    Save processed dataset locally and/or to the Hub.

    Default behavior (save_mode="auto"):
    - Always save locally unless dataset > size_threshold (5GB)
    - If dataset came from Hub, also push to Hub as private dataset
    - If dataset > size_threshold, only push to Hub (skip local to save disk space)

    Args:
        dataset: The processed dataset to save
        project_name: Name of the project (used for directory and Hub repo name)
        split_name: Name of the split (e.g., "train", "validation")
        source_path: Original dataset path (to detect if from Hub)
        username: HuggingFace username (required for Hub push)
        token: HuggingFace token (required for Hub push)
        save_mode: One of "auto", "local", "hub", "both", "none"
        size_threshold_bytes: Size threshold for hub-only saving (default 5GB)

    Returns:
        Dict with keys: "local_path" (str or None), "hub_path" (str or None)
    """
    result = {"local_path": None, "hub_path": None}

    if save_mode == "none":
        logger.info("save_mode='none', skipping dataset save")
        return result

    # Convert to DataFrame for consistent handling
    if isinstance(dataset, pd.DataFrame):
        df = dataset
    else:
        df = dataset.to_pandas() if hasattr(dataset, "to_pandas") else pd.DataFrame(dataset)

    # Estimate size
    dataset_size = estimate_dataset_size(df)
    is_large = dataset_size > size_threshold_bytes
    from_hub = is_hub_dataset(source_path) if source_path else False

    logger.info(f"Dataset size: {dataset_size / (1024*1024):.2f} MB, from_hub: {from_hub}, is_large: {is_large}")

    # Determine what to save based on mode
    save_local = False
    save_hub = False

    if save_mode == "auto":
        # Auto mode: local + hub if from hub, or hub-only if large
        if is_large:
            save_hub = True
            save_local = False
            logger.info("Dataset > 5GB, will only push to Hub to save disk space")
        else:
            save_local = True
            save_hub = from_hub
    elif save_mode == "local":
        save_local = True
    elif save_mode == "hub":
        save_hub = True
    elif save_mode == "both":
        save_local = True
        save_hub = True

    # Save locally
    if save_local:
        local_dir = os.path.join(project_name, "data_processed")
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, f"{split_name}.jsonl")
        df.to_json(local_path, orient="records", lines=True, force_ascii=False)
        result["local_path"] = local_path
        logger.info(f"Processed dataset saved locally: {local_path}")

    # Push to Hub
    if save_hub:
        if not username or not token:
            logger.warning("Cannot push to Hub: username or token not provided")
        else:
            try:
                from datetime import datetime

                from datasets import Dataset as HFDataset
                from huggingface_hub import HfApi

                # Convert back to HF Dataset for push_to_hub
                if isinstance(df, pd.DataFrame):
                    hf_dataset = HFDataset.from_pandas(df)
                else:
                    hf_dataset = df

                # Create repo name: {username}/aitraining-processed-{project_name}-{YYYYMMDD}
                # Clean project name for repo
                clean_project = os.path.basename(project_name).replace(" ", "-").lower()
                date_str = datetime.now().strftime("%Y%m%d")
                hub_repo = f"{username}/aitraining-processed-{clean_project}-{date_str}"

                hf_dataset.push_to_hub(
                    hub_repo,
                    split=split_name,
                    private=True,
                    token=token,
                )
                result["hub_path"] = hub_repo
                logger.info(f"Processed dataset pushed to Hub: {hub_repo} (private)")

                # Create dataset card with processing info
                try:
                    api = HfApi(token=token)
                    card_content = f"""---
license: other
task_categories:
- text-generation
tags:
- aitraining
- processed
---

# AITraining Processed Dataset

This dataset was automatically processed by [AITraining](https://github.com/monostate/aitraining).

## Source

- **Original dataset**: `{source_path or 'local'}`
- **Processed on**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Project**: `{project_name}`

## Processing

This dataset has been processed with chat template application and format standardization.
The original data structure has been preserved while applying necessary transformations for training.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{hub_repo}", split="{split_name}")
```

---
*Generated automatically by AITraining*
"""
                    api.upload_file(
                        path_or_fileobj=card_content.encode(),
                        path_in_repo="README.md",
                        repo_id=hub_repo,
                        repo_type="dataset",
                        token=token,
                    )
                    logger.info(f"Dataset card created for {hub_repo}")
                except Exception as card_error:
                    logger.warning(f"Could not create dataset card: {card_error}")

            except Exception as e:
                logger.error(f"Failed to push dataset to Hub: {e}")

    return result


def save_processed_datasets(
    train_data: Union[Dataset, pd.DataFrame],
    valid_data: Optional[Union[Dataset, pd.DataFrame]],
    project_name: str,
    train_split: str = "train",
    valid_split: Optional[str] = None,
    source_path: Optional[str] = None,
    username: Optional[str] = None,
    token: Optional[str] = None,
    save_mode: str = "auto",
) -> dict:
    """
    Save both training and validation processed datasets.

    Convenience function that calls save_processed_dataset for both splits.

    Args:
        train_data: Processed training dataset
        valid_data: Processed validation dataset (optional)
        project_name: Name of the project
        train_split: Name of training split (default "train")
        valid_split: Name of validation split (optional)
        source_path: Original dataset path
        username: HuggingFace username
        token: HuggingFace token
        save_mode: One of "auto", "local", "hub", "both", "none"

    Returns:
        Dict with keys: "train" and "valid", each containing local_path and hub_path
    """
    result = {"train": None, "valid": None}

    # Save training data
    result["train"] = save_processed_dataset(
        dataset=train_data,
        project_name=project_name,
        split_name=train_split,
        source_path=source_path,
        username=username,
        token=token,
        save_mode=save_mode,
    )

    # Save validation data if provided
    if valid_data is not None and valid_split:
        result["valid"] = save_processed_dataset(
            dataset=valid_data,
            project_name=project_name,
            split_name=valid_split,
            source_path=source_path,
            username=username,
            token=token,
            save_mode=save_mode,
        )

    return result
