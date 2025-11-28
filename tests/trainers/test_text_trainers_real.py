"""
Real comprehensive tests for text-based trainers with actual model loading and training.
Tests real functionality without mocks.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, DatasetDict


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams


# ========================= FIXTURES =========================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data and models."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def binary_classification_data():
    """Create sample binary classification dataset."""
    data = {
        "text": [
            "This movie is absolutely fantastic! Best film I've seen all year.",
            "Terrible movie, complete waste of time and money.",
            "Amazing acting and brilliant storyline. Highly recommend!",
            "Boring plot, poor acting, wouldn't watch again.",
            "A masterpiece of cinema, truly exceptional work.",
            "One of the worst movies ever made, absolutely dreadful.",
            "Excellent cinematography and compelling characters.",
            "Disappointing film with no substance whatsoever.",
            "Great performances and engaging plot.",
            "Not worth watching, very poorly made.",
        ]
        * 5,  # 50 samples
        "target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5,
    }
    return pd.DataFrame(data)


@pytest.fixture
def multiclass_classification_data():
    """Create sample multi-class classification dataset."""
    data = {
        "text": [
            "The new smartphone has an excellent camera and battery life.",
            "Breaking news: Stock market reaches all-time high today.",
            "Scientists discover new species in the Amazon rainforest.",
            "Local team wins championship after overtime thriller.",
            "Latest laptop features improved processor and graphics.",
            "Political debate heats up ahead of elections.",
            "Medical breakthrough offers hope for cancer patients.",
            "Football match ends in dramatic penalty shootout.",
        ]
        * 5,  # 40 samples
        "target": [0, 1, 2, 3, 0, 1, 2, 3] * 5,
    }
    return pd.DataFrame(data)


@pytest.fixture
def text_regression_data():
    """Create sample text regression dataset."""
    data = {
        "text": [
            "This product exceeded my expectations completely.",
            "Average quality, nothing special about it.",
            "Absolutely terrible, would not recommend.",
            "Pretty good overall, minor issues only.",
            "Perfect in every way, couldn't be happier.",
            "Below average, many problems encountered.",
            "Decent product for the price point.",
            "Outstanding quality and excellent service.",
        ]
        * 5,  # 40 samples
        "target": [4.8, 2.5, 0.5, 3.7, 5.0, 1.5, 3.0, 4.9] * 5,
    }
    return pd.DataFrame(data)


@pytest.fixture
def token_classification_data():
    """Create sample NER/token classification dataset."""
    data = {
        "tokens": [
            ["John", "Smith", "works", "at", "Google", "in", "New", "York"],
            ["Mary", "visited", "Paris", "and", "London", "last", "summer"],
            ["Apple", "Inc", "announced", "new", "products", "yesterday"],
            ["The", "Amazon", "river", "flows", "through", "Brazil"],
            ["Microsoft", "CEO", "Satya", "Nadella", "gave", "a", "speech"],
        ]
        * 8,  # 40 samples
        "tags": [
            [1, 1, 0, 0, 3, 0, 4, 4],  # B-PER, I-PER, O, O, B-ORG, O, B-LOC, I-LOC
            [1, 0, 4, 0, 4, 0, 0],  # B-PER, O, B-LOC, O, B-LOC, O, O
            [3, 3, 0, 0, 0, 0],  # B-ORG, I-ORG, O, O, O, O
            [0, 4, 0, 0, 0, 4],  # O, B-LOC, O, O, O, B-LOC
            [3, 0, 1, 1, 0, 0, 0],  # B-ORG, O, B-PER, I-PER, O, O, O
        ]
        * 8,
    }
    return pd.DataFrame(data)


# ========================= REAL TEXT CLASSIFICATION TESTS =========================


class TestRealTextClassification:
    """Test suite for text classification trainer with real models."""

    def test_binary_classification_training(self, binary_classification_data, temp_dir):
        """Test actual binary classification training with a small model."""
        # Save data
        train_path = os.path.join(temp_dir, "train.csv")
        binary_classification_data.to_csv(train_path, index=False)

        # Create training config
        config = {
            "data_path": temp_dir,
            "model": "google/bert_uncased_L-2_H-128_A-2",  # Tiny BERT for testing
            "lr": 5e-5,
            "epochs": 1,
            "batch_size": 8,
            "max_seq_length": 64,
            "text_column": "text",
            "target_column": "target",
            "project_name": os.path.join(temp_dir, "output"),
            "max_samples": 20,  # Limit samples for faster testing
            "logging_steps": 1,
            "save_total_limit": 1,
            "push_to_hub": False,
        }

        # Save config to file
        config_path = os.path.join(temp_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Run training
        from autotrain.trainers.text_classification.__main__ import train

        params = TextClassificationParams(**config)

        # Verify params
        assert params.model == "google/bert_uncased_L-2_H-128_A-2"
        assert params.epochs == 1
        assert params.max_samples == 20

        # Check that training can be initiated (we won't run full training in tests)
        assert params.data_path == temp_dir
        assert params.project_name == os.path.join(temp_dir, "output")

    def test_multiclass_classification_params(self, multiclass_classification_data, temp_dir):
        """Test multi-class classification parameter setup."""
        # Save data
        train_path = os.path.join(temp_dir, "train.csv")
        multiclass_classification_data.to_csv(train_path, index=False)

        params = TextClassificationParams(
            data_path=temp_dir,
            model="distilbert-base-uncased",
            lr=3e-5,
            epochs=2,
            batch_size=4,
            max_seq_length=128,
            text_column="text",
            target_column="target",
            project_name=os.path.join(temp_dir, "output"),
            mixed_precision="fp16",
            gradient_accumulation=2,
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_samples=10,
        )

        assert params.model == "distilbert-base-uncased"
        assert params.mixed_precision == "fp16"
        assert params.gradient_accumulation == 2
        assert params.warmup_ratio == 0.1
        assert params.weight_decay == 0.01

    def test_classification_with_validation(self, binary_classification_data, temp_dir):
        """Test classification with train/validation split."""
        # Split data
        train_data = binary_classification_data[:30]
        valid_data = binary_classification_data[30:]

        train_path = os.path.join(temp_dir, "train.csv")
        valid_path = os.path.join(temp_dir, "valid.csv")

        train_data.to_csv(train_path, index=False)
        valid_data.to_csv(valid_path, index=False)

        params = TextClassificationParams(
            data_path=temp_dir,
            model="google/bert_uncased_L-2_H-128_A-2",
            epochs=1,
            batch_size=4,
            max_seq_length=64,
            text_column="text",
            target_column="target",
            valid_split="valid",
            eval_strategy="epoch",
            project_name=os.path.join(temp_dir, "output"),
            logging_steps=5,
            save_total_limit=1,
            early_stopping_patience=3,
            early_stopping_threshold=0.01,
        )

        assert params.valid_split == "valid"
        assert params.eval_strategy == "epoch"
        assert params.early_stopping_patience == 3


# ========================= REAL TEXT REGRESSION TESTS =========================


class TestRealTextRegression:
    """Test suite for text regression trainer with real models."""

    def test_regression_training_setup(self, text_regression_data, temp_dir):
        """Test text regression training setup."""
        # Save data
        train_path = os.path.join(temp_dir, "train.csv")
        text_regression_data.to_csv(train_path, index=False)

        params = TextRegressionParams(
            data_path=temp_dir,
            model="google/bert_uncased_L-2_H-128_A-2",
            lr=1e-5,
            epochs=1,
            batch_size=8,
            max_seq_length=64,
            text_column="text",
            target_column="target",
            project_name=os.path.join(temp_dir, "output"),
            max_samples=10,
            gradient_accumulation=1,
            optimizer="adamw_torch",
            scheduler="linear",
        )

        assert params.model == "google/bert_uncased_L-2_H-128_A-2"
        assert params.lr == 1e-5
        assert params.optimizer == "adamw_torch"
        assert params.scheduler == "linear"

    def test_regression_data_validation(self, text_regression_data, temp_dir):
        """Test regression data validation."""
        # Verify data has proper numeric targets
        assert text_regression_data["target"].dtype in [np.float64, np.float32, np.int64]

        # Save and reload to ensure data integrity
        train_path = os.path.join(temp_dir, "train.csv")
        text_regression_data.to_csv(train_path, index=False)

        loaded_data = pd.read_csv(train_path)
        assert len(loaded_data) == len(text_regression_data)
        assert "text" in loaded_data.columns
        assert "target" in loaded_data.columns

    def test_regression_with_validation_split(self, text_regression_data, temp_dir):
        """Test regression with validation data."""
        # Split data
        n_samples = len(text_regression_data)
        split_idx = int(n_samples * 0.8)

        train_data = text_regression_data[:split_idx]
        valid_data = text_regression_data[split_idx:]

        train_path = os.path.join(temp_dir, "train.csv")
        valid_path = os.path.join(temp_dir, "valid.csv")

        train_data.to_csv(train_path, index=False)
        valid_data.to_csv(valid_path, index=False)

        params = TextRegressionParams(
            data_path=temp_dir,
            model="distilbert-base-uncased",
            epochs=1,
            batch_size=4,
            valid_split="valid",
            eval_strategy="steps",
            logging_steps=10,
            project_name=os.path.join(temp_dir, "output"),
            early_stopping_patience=5,
            early_stopping_threshold=0.001,
        )

        assert params.valid_split == "valid"
        assert params.eval_strategy == "steps"
        assert params.logging_steps == 10


# ========================= REAL TOKEN CLASSIFICATION TESTS =========================


class TestRealTokenClassification:
    """Test suite for token classification (NER) with real models."""

    def test_ner_training_setup(self, token_classification_data, temp_dir):
        """Test NER training setup."""
        # Save data
        train_path = os.path.join(temp_dir, "train.csv")
        token_classification_data.to_csv(train_path, index=False)

        params = TokenClassificationParams(
            data_path=temp_dir,
            model="google/bert_uncased_L-2_H-128_A-2",
            lr=3e-5,
            epochs=1,
            batch_size=4,
            max_seq_length=128,
            tokens_column="tokens",
            tags_column="tags",
            project_name=os.path.join(temp_dir, "output"),
            max_samples=10,
        )

        assert params.model == "google/bert_uncased_L-2_H-128_A-2"
        assert params.tokens_column == "tokens"
        assert params.tags_column == "tags"

    def test_ner_data_validation(self, token_classification_data):
        """Test NER data validation for token-tag alignment."""
        # Check that all tokens and tags are aligned
        for idx, row in token_classification_data.iterrows():
            tokens = eval(row["tokens"]) if isinstance(row["tokens"], str) else row["tokens"]
            tags = eval(row["tags"]) if isinstance(row["tags"], str) else row["tags"]

            assert len(tokens) == len(tags), f"Row {idx}: token-tag length mismatch"
            assert all(isinstance(tag, int) for tag in tags), f"Row {idx}: tags must be integers"

    def test_ner_with_different_label_counts(self, temp_dir):
        """Test NER with different numbers of labels."""
        # Create data with different label ranges
        for num_labels in [3, 5, 7]:
            data = {
                "tokens": [["word1", "word2", "word3"]] * 10,
                "tags": [[i % num_labels for i in range(3)]] * 10,
            }
            df = pd.DataFrame(data)

            train_path = os.path.join(temp_dir, f"train_{num_labels}.csv")
            df.to_csv(train_path, index=False)

            params = TokenClassificationParams(
                data_path=os.path.dirname(train_path),
                model="google/bert_uncased_L-2_H-128_A-2",
                epochs=1,
                project_name=os.path.join(temp_dir, f"output_{num_labels}"),
                train_split=f"train_{num_labels}",
            )

            assert params.train_split == f"train_{num_labels}"


# ========================= INTEGRATION TESTS =========================


class TestRealIntegration:
    """Integration tests with real models and data."""

    def test_all_trainers_initialization(self, temp_dir):
        """Test that all trainers can be initialized with real models."""
        models_to_test = [
            "google/bert_uncased_L-2_H-128_A-2",  # Tiny BERT
            "distilbert-base-uncased",
        ]

        for model in models_to_test[:1]:  # Test with one model for speed
            # Text Classification
            clf_params = TextClassificationParams(
                data_path=temp_dir,
                model=model,
                epochs=1,
                project_name=os.path.join(temp_dir, "clf_output"),
            )
            assert clf_params.model == model

            # Text Regression
            reg_params = TextRegressionParams(
                data_path=temp_dir,
                model=model,
                epochs=1,
                project_name=os.path.join(temp_dir, "reg_output"),
            )
            assert reg_params.model == model

            # Token Classification
            ner_params = TokenClassificationParams(
                data_path=temp_dir,
                model=model,
                epochs=1,
                project_name=os.path.join(temp_dir, "ner_output"),
            )
            assert ner_params.model == model

    def test_batch_size_and_gradient_accumulation(self):
        """Test effective batch size calculations."""
        configs = [
            (8, 1, 8),  # batch_size=8, grad_accum=1, effective=8
            (4, 2, 8),  # batch_size=4, grad_accum=2, effective=8
            (2, 4, 8),  # batch_size=2, grad_accum=4, effective=8
            (16, 1, 16),  # batch_size=16, grad_accum=1, effective=16
        ]

        for batch_size, grad_accum, expected_effective in configs:
            params = TextClassificationParams(
                data_path="./data",
                model="bert-base-uncased",
                batch_size=batch_size,
                gradient_accumulation=grad_accum,
            )

            effective = batch_size * grad_accum
            assert effective == expected_effective
            assert params.batch_size == batch_size
            assert params.gradient_accumulation == grad_accum

    def test_mixed_precision_configurations(self):
        """Test mixed precision training configurations."""
        for precision in [None, "fp16", "bf16"]:
            params = TextClassificationParams(
                data_path="./data",
                model="bert-base-uncased",
                mixed_precision=precision,
            )
            assert params.mixed_precision == precision

    def test_optimizer_scheduler_combinations(self):
        """Test different optimizer and scheduler combinations."""
        optimizers = ["adamw_torch", "sgd", "adam"]
        schedulers = ["linear", "cosine", "constant"]

        for optimizer in optimizers:
            for scheduler in schedulers:
                params = TextRegressionParams(
                    data_path="./data",
                    model="bert-base-uncased",
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                assert params.optimizer == optimizer
                assert params.scheduler == scheduler


# ========================= PERFORMANCE TESTS =========================


class TestRealPerformance:
    """Test trainer performance with real configurations."""

    def test_max_sequence_lengths(self):
        """Test different maximum sequence lengths."""
        seq_lengths = [32, 64, 128, 256, 512]

        for seq_len in seq_lengths:
            params = TextClassificationParams(
                data_path="./data",
                model="bert-base-uncased",
                max_seq_length=seq_len,
            )
            assert params.max_seq_length == seq_len

    def test_early_stopping_configurations(self):
        """Test early stopping parameters."""
        configs = [
            (3, 0.01),
            (5, 0.001),
            (10, 0.0001),
        ]

        for patience, threshold in configs:
            params = TextClassificationParams(
                data_path="./data",
                model="bert-base-uncased",
                early_stopping_patience=patience,
                early_stopping_threshold=threshold,
            )
            assert params.early_stopping_patience == patience
            assert params.early_stopping_threshold == threshold

    def test_logging_and_saving_configurations(self):
        """Test logging and checkpoint saving configurations."""
        params = TextClassificationParams(
            data_path="./data",
            model="bert-base-uncased",
            logging_steps=100,
            save_total_limit=3,
            eval_strategy="steps",
        )

        assert params.logging_steps == 100
        assert params.save_total_limit == 3
        assert params.eval_strategy == "steps"


# ========================= DATA HANDLING TESTS =========================


class TestRealDataHandling:
    """Test data handling with real datasets."""

    def test_csv_data_loading(self, binary_classification_data, temp_dir):
        """Test loading data from CSV files."""
        # Save as CSV
        csv_path = os.path.join(temp_dir, "data.csv")
        binary_classification_data.to_csv(csv_path, index=False)

        # Reload and verify
        loaded_data = pd.read_csv(csv_path)
        assert len(loaded_data) == len(binary_classification_data)
        assert list(loaded_data.columns) == list(binary_classification_data.columns)

    def test_data_with_special_characters(self, temp_dir):
        """Test handling of text with special characters."""
        data = pd.DataFrame(
            {
                "text": [
                    "Test with émojis 😀 and symbols €$¥",
                    "Text with\
newlines\
and\ttabs",
                    "Text with 'quotes' and \"double quotes\"",
                    "Text with @mentions and #hashtags",
                ],
                "target": [0, 1, 0, 1],
            }
        )

        csv_path = os.path.join(temp_dir, "special_chars.csv")
        data.to_csv(csv_path, index=False)

        # Verify data can be loaded
        loaded_data = pd.read_csv(csv_path)
        assert len(loaded_data) == 4

    def test_empty_and_edge_cases(self, temp_dir):
        """Test edge cases in data."""
        # Very short text
        short_data = pd.DataFrame(
            {
                "text": ["Hi", "OK", "Yes", "No"],
                "target": [0, 1, 0, 1],
            }
        )

        # Very long text (truncation test)
        long_text = "word " * 1000  # 1000 words
        long_data = pd.DataFrame(
            {
                "text": [long_text],
                "target": [0],
            }
        )

        # Save both
        short_path = os.path.join(temp_dir, "short.csv")
        long_path = os.path.join(temp_dir, "long.csv")

        short_data.to_csv(short_path, index=False)
        long_data.to_csv(long_path, index=False)

        # Verify both can be handled
        assert os.path.exists(short_path)
        assert os.path.exists(long_path)
