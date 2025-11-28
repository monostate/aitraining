"""
Comprehensive tests for Seq2Seq and Extractive Question Answering trainers.

This module tests both trainers with:
- Model loading and configuration
- Data validation and preprocessing
- Training with various configurations
- Metrics computation (BLEU/ROUGE for Seq2Seq, EM/F1 for QA)
- Checkpoint saving and loading
- Edge cases and error handling
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from autotrain.trainers.extractive_question_answering import utils as qa_utils
from autotrain.trainers.extractive_question_answering.__main__ import train as qa_train
from autotrain.trainers.extractive_question_answering.dataset import ExtractiveQuestionAnsweringDataset
from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
from autotrain.trainers.seq2seq import utils as seq2seq_utils
from autotrain.trainers.seq2seq.__main__ import train as seq2seq_train
from autotrain.trainers.seq2seq.dataset import Seq2SeqDataset
from autotrain.trainers.seq2seq.params import Seq2SeqParams


def save_dataset_as_csv(dataset, path, split_name="train"):
    """Helper to save datasets as CSV files for trainer compatibility."""
    import pandas as pd

    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    # Convert dataset to DataFrame
    df = pd.DataFrame(dataset)
    # Convert answers column to JSON string for proper CSV storage
    if "answers" in df.columns:
        df["answers"] = df["answers"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
    df.to_csv(path / f"{split_name}.csv", index=False)
    return path


# =============================================================================
# Seq2Seq Trainer Tests
# =============================================================================


class TestSeq2SeqTrainer:
    """Test suite for the Seq2Seq trainer."""

    @pytest.fixture
    def seq2seq_data(self):
        """Create sample data for translation/summarization tasks."""
        train_data = {
            "text": [
                "Hello world",
                "How are you?",
                "Machine learning is awesome",
                "Natural language processing",
                "Deep learning models",
            ],
            "target": [
                "Bonjour le monde",
                "Comment allez-vous?",
                "L'apprentissage automatique est génial",
                "Traitement du langage naturel",
                "Modèles d'apprentissage profond",
            ],
        }

        valid_data = {
            "text": [
                "Good morning",
                "Thank you very much",
            ],
            "target": [
                "Bonjour",
                "Merci beaucoup",
            ],
        }

        return Dataset.from_dict(train_data), Dataset.from_dict(valid_data)

    @pytest.fixture
    def seq2seq_config(self, tmp_path):
        """Create a basic Seq2Seq configuration."""
        return Seq2SeqParams(
            model="t5-small",
            data_path=str(tmp_path / "data"),
            project_name=str(tmp_path / "output"),
            text_column="text",
            target_column="target",
            max_seq_length=128,
            max_target_length=128,
            batch_size=2,
            epochs=1,
            lr=5e-5,
            seed=42,
            train_split="train",
            valid_split="validation",
            logging_steps=1,
            save_total_limit=1,
            push_to_hub=False,
            mixed_precision=None,
            max_samples=5,  # Limit samples for faster testing
        )

    def test_seq2seq_model_loading_t5(self, seq2seq_config):
        """Test loading T5 model for Seq2Seq tasks."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model = AutoModelForSeq2SeqLM.from_pretrained(seq2seq_config.model)
        tokenizer = AutoTokenizer.from_pretrained(seq2seq_config.model)

        assert model is not None
        assert tokenizer is not None
        assert hasattr(model, "generate")
        assert hasattr(model.config, "decoder_start_token_id")

    def test_seq2seq_dataset_preparation(self, seq2seq_data, seq2seq_config):
        """Test Seq2Seq dataset preparation with input/target columns."""
        train_data, _ = seq2seq_data
        tokenizer = AutoTokenizer.from_pretrained(seq2seq_config.model)

        dataset = Seq2SeqDataset(data=train_data, tokenizer=tokenizer, config=seq2seq_config)

        assert len(dataset) == len(train_data)

        # Check first sample
        sample = dataset[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

        # Verify shapes
        assert len(sample["input_ids"]) <= seq2seq_config.max_seq_length
        assert len(sample["labels"]) <= seq2seq_config.max_target_length

    def test_seq2seq_data_validation(self, seq2seq_config):
        """Test data validation for Seq2Seq tasks."""
        # Test with missing columns
        invalid_data = {
            "text": ["Hello world"],
            # Missing 'target' column
        }
        invalid_dataset = Dataset.from_dict(invalid_data)

        tokenizer = AutoTokenizer.from_pretrained(seq2seq_config.model)

        with pytest.raises(KeyError):
            dataset = Seq2SeqDataset(data=invalid_dataset, tokenizer=tokenizer, config=seq2seq_config)

    def test_seq2seq_training_basic(self, seq2seq_data, seq2seq_config, tmp_path):
        """Test basic Seq2Seq training pipeline."""
        train_data, valid_data = seq2seq_data

        # Save data
        data_path = tmp_path / "data"
        data_path.mkdir(exist_ok=True)
        save_dataset_as_csv(train_data, data_path, "train")
        save_dataset_as_csv(valid_data, data_path, "validation")

        # Update config
        seq2seq_config.data_path = str(data_path)
        seq2seq_config.project_name = str(tmp_path / "output")

        # Mock the training to avoid long execution
        with patch("autotrain.trainers.seq2seq.__main__.Seq2SeqTrainer") as mock_trainer:
            mock_instance = MagicMock()
            mock_trainer.return_value = mock_instance
            mock_instance.train.return_value = None
            mock_instance.save_model.return_value = None
            mock_instance.evaluate.return_value = {
                "eval_loss": 0.5,
                "eval_rouge1": 45.0,
                "eval_rouge2": 20.0,
                "eval_rougeL": 40.0,
            }

            # Run training
            seq2seq_train(seq2seq_config)

            # Verify trainer was called
            assert mock_trainer.called
            assert mock_instance.train.called
            assert mock_instance.save_model.called

    def test_seq2seq_metrics_computation(self):
        """Test BLEU/ROUGE metrics computation for Seq2Seq."""
        tokenizer = AutoTokenizer.from_pretrained("t5-small")

        # Create sample predictions and labels
        predictions = ["This is a test", "Another prediction"]
        references = ["This is a test", "Another reference"]

        # Tokenize
        pred_ids = [tokenizer.encode(p, truncation=True, max_length=128) for p in predictions]
        ref_ids = [tokenizer.encode(r, truncation=True, max_length=128) for r in references]

        # Pad to same length
        max_len = max(max(len(p) for p in pred_ids), max(len(r) for r in ref_ids))
        pred_ids = [p + [tokenizer.pad_token_id] * (max_len - len(p)) for p in pred_ids]

        # Properly pad reference IDs with -100 for ignored positions
        ref_ids_padded = []
        for ref in ref_ids:
            # Pad with -100 (ignore index for loss calculation)
            padded = ref + [-100] * (max_len - len(ref))
            ref_ids_padded.append(padded)

        # Convert to tensors
        import numpy as np

        pred_tensor = np.array(pred_ids)
        ref_tensor = np.array(ref_ids_padded)

        # Compute metrics
        metrics = seq2seq_utils._seq2seq_metrics((pred_tensor, ref_tensor), tokenizer)

        assert "rouge1" in metrics
        assert "rouge2" in metrics
        assert "rougeL" in metrics
        assert "gen_len" in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_seq2seq_checkpoint_saving(self, seq2seq_data, seq2seq_config, tmp_path):
        """Test checkpoint saving functionality for Seq2Seq."""
        train_data, valid_data = seq2seq_data

        output_dir = tmp_path / "checkpoints"
        output_dir.mkdir(exist_ok=True)

        seq2seq_config.project_name = str(output_dir)
        seq2seq_config.save_total_limit = 2
        seq2seq_config.eval_strategy = "steps"
        seq2seq_config.logging_steps = 2

        # Create a small model for testing
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        tokenizer = AutoTokenizer.from_pretrained("t5-small")

        # Save model
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Verify files exist
        assert (output_dir / "config.json").exists()
        assert (output_dir / "pytorch_model.bin").exists() or (output_dir / "model.safetensors").exists()

    def test_seq2seq_with_peft(self, seq2seq_config):
        """Test Seq2Seq training with PEFT/LoRA configuration."""
        seq2seq_config.peft = True
        seq2seq_config.lora_r = 8
        seq2seq_config.lora_alpha = 16
        seq2seq_config.lora_dropout = 0.1
        seq2seq_config.target_modules = "q_proj,v_proj"

        assert seq2seq_config.peft is True
        assert seq2seq_config.lora_r == 8
        assert seq2seq_config.lora_alpha == 16

        # Verify PEFT configuration is valid
        from peft import LoraConfig, TaskType

        lora_config = LoraConfig(
            r=seq2seq_config.lora_r,
            lora_alpha=seq2seq_config.lora_alpha,
            lora_dropout=seq2seq_config.lora_dropout,
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

        assert lora_config.r == 8
        assert lora_config.lora_alpha == 16


# =============================================================================
# Extractive QA Trainer Tests
# =============================================================================


class TestExtractiveQATrainer:
    """Test suite for the Extractive Question Answering trainer."""

    @pytest.fixture
    def qa_data(self):
        """Create sample SQuAD-style QA data."""
        train_data = {
            "id": ["1", "2", "3", "4", "5"],
            "context": [
                "The capital of France is Paris. It is known for the Eiffel Tower.",
                "Python is a programming language. It was created by Guido van Rossum.",
                "Machine learning is a subset of artificial intelligence.",
                "The Amazon rainforest is located in South America.",
                "Albert Einstein developed the theory of relativity.",
            ],
            "question": [
                "What is the capital of France?",
                "Who created Python?",
                "What is machine learning?",
                "Where is the Amazon rainforest?",
                "What did Einstein develop?",
            ],
            "answers": [
                {"text": ["Paris"], "answer_start": [25]},  # Fixed position
                {"text": ["Guido van Rossum"], "answer_start": [52]},  # Fixed position
                {"text": ["a subset of artificial intelligence"], "answer_start": [20]},  # Fixed position
                {"text": ["South America"], "answer_start": [36]},  # Fixed position
                {"text": ["the theory of relativity"], "answer_start": [26]},
            ],
        }

        valid_data = {
            "id": ["6", "7"],
            "context": [
                "The sun is a star at the center of our solar system.",
                "Water freezes at 0 degrees Celsius.",
            ],
            "question": [
                "What is the sun?",
                "At what temperature does water freeze?",
            ],
            "answers": [
                {"text": ["a star"], "answer_start": [11]},
                {"text": ["0 degrees Celsius"], "answer_start": [18]},
            ],
        }

        return Dataset.from_dict(train_data), Dataset.from_dict(valid_data)

    @pytest.fixture
    def qa_config(self, tmp_path):
        """Create a basic QA configuration."""
        return ExtractiveQuestionAnsweringParams(
            model="bert-base-uncased",
            data_path=str(tmp_path / "data"),
            project_name=str(tmp_path / "output"),
            text_column="context",
            question_column="question",
            answer_column="answers",
            max_seq_length=384,
            max_doc_stride=128,
            batch_size=2,
            epochs=1,
            lr=3e-5,
            seed=42,
            train_split="train",
            valid_split="validation",
            logging_steps=1,
            save_total_limit=1,
            push_to_hub=False,
            mixed_precision=None,
            max_samples=5,  # Limit samples for faster testing
        )

    def test_qa_model_loading_bert(self, qa_config):
        """Test loading BERT model for QA tasks."""
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        model = AutoModelForQuestionAnswering.from_pretrained(qa_config.model, ignore_mismatched_sizes=True)
        tokenizer = AutoTokenizer.from_pretrained(qa_config.model)

        assert model is not None
        assert tokenizer is not None
        assert hasattr(model, "qa_outputs")

    def test_qa_dataset_preparation(self, qa_data, qa_config):
        """Test QA dataset preparation with SQuAD-style data."""
        train_data, _ = qa_data
        tokenizer = AutoTokenizer.from_pretrained(qa_config.model)

        dataset = ExtractiveQuestionAnsweringDataset(data=train_data, tokenizer=tokenizer, config=qa_config)

        assert len(dataset) == len(train_data)

        # Check first sample
        sample = dataset[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "start_positions" in sample
        assert "end_positions" in sample

        # Verify answer positions are valid
        assert sample["start_positions"] >= 0
        assert sample["end_positions"] >= sample["start_positions"]

    def test_qa_data_validation(self, qa_config):
        """Test data validation for QA tasks."""
        # Test with missing context column
        invalid_data = {
            "question": ["What is the capital?"],
            "answers": [{"text": ["Paris"], "answer_start": [0]}],
            # Missing 'context' column
        }
        invalid_dataset = Dataset.from_dict(invalid_data)

        tokenizer = AutoTokenizer.from_pretrained(qa_config.model)

        with pytest.raises(KeyError):
            dataset = ExtractiveQuestionAnsweringDataset(data=invalid_dataset, tokenizer=tokenizer, config=qa_config)

    def test_qa_answer_extraction(self, qa_data, qa_config):
        """Test answer extraction from context."""
        train_data, _ = qa_data
        tokenizer = AutoTokenizer.from_pretrained(qa_config.model)

        # Process first example
        example = {
            qa_config.text_column: train_data[qa_config.text_column][0],
            qa_config.question_column: train_data[qa_config.question_column][0],
            qa_config.answer_column: train_data[qa_config.answer_column][0],
        }

        # Verify answer extraction
        context = example[qa_config.text_column]
        answer_info = example[qa_config.answer_column]
        answer_text = answer_info["text"][0]
        answer_start = answer_info["answer_start"][0]

        extracted = context[answer_start : answer_start + len(answer_text)]
        assert extracted == answer_text

    def test_qa_training_basic(self, qa_data, qa_config, tmp_path):
        """Test basic QA training pipeline."""
        train_data, valid_data = qa_data

        # Save data
        data_path = tmp_path / "data"
        data_path.mkdir(exist_ok=True)
        save_dataset_as_csv(train_data, data_path, "train")
        save_dataset_as_csv(valid_data, data_path, "validation")

        # Update config
        qa_config.data_path = str(data_path)
        qa_config.project_name = str(tmp_path / "output")

        # Mock the training to avoid long execution
        with patch("autotrain.trainers.extractive_question_answering.__main__.Trainer") as mock_trainer:
            mock_instance = MagicMock()
            mock_trainer.return_value = mock_instance
            mock_instance.train.return_value = None
            mock_instance.save_model.return_value = None
            mock_instance.evaluate.return_value = {
                "eval_loss": 0.5,
                "eval_exact_match": 60.0,
                "eval_f1": 75.0,
            }

            # Run training
            qa_train(qa_config)

            # Verify trainer was called
            assert mock_trainer.called
            assert mock_instance.train.called
            assert mock_instance.save_model.called

    def test_qa_metrics_computation(self, qa_data, qa_config):
        """Test exact match and F1 metrics computation for QA."""
        train_data, _ = qa_data

        # Create mock predictions and references
        predictions = [
            {"id": "1", "prediction_text": "Paris"},
            {"id": "2", "prediction_text": "Guido van Rossum"},
        ]

        references = [
            {"id": "1", "answers": {"text": ["Paris"], "answer_start": [26]}},
            {"id": "2", "answers": {"text": ["Guido van Rossum"], "answer_start": [50]}},
        ]

        # Test exact match - both should be 100% for exact matches
        try:
            from evaluate import load

            squad_metric = load("squad")
        except ImportError:
            from datasets import load_metric

            squad_metric = load_metric("squad")

        result = squad_metric.compute(predictions=predictions, references=references)

        assert "exact_match" in result
        assert "f1" in result
        assert result["exact_match"] == 100.0  # Perfect match
        assert result["f1"] == 100.0  # Perfect F1

    def test_qa_with_unanswerable(self, qa_config):
        """Test QA with unanswerable questions (SQuAD v2 style)."""
        # Create data with unanswerable question
        data = {
            "id": ["1", "2"],
            "context": [
                "The capital of France is Paris.",
                "Python is a programming language.",
            ],
            "question": [
                "What is the capital of France?",
                "What is the capital of Germany?",  # Unanswerable
            ],
            "answers": [
                {"text": ["Paris"], "answer_start": [26]},
                {"text": [], "answer_start": [-1]},  # No answer
            ],
        }

        dataset = Dataset.from_dict(data)
        tokenizer = AutoTokenizer.from_pretrained(qa_config.model)

        # Process validation features
        processed = qa_utils.prepare_qa_validation_features(examples=data, tokenizer=tokenizer, config=qa_config)

        assert "input_ids" in processed
        assert "attention_mask" in processed
        assert "offset_mapping" in processed

    def test_qa_checkpoint_saving(self, qa_data, qa_config, tmp_path):
        """Test checkpoint saving functionality for QA."""
        train_data, valid_data = qa_data

        output_dir = tmp_path / "checkpoints"
        output_dir.mkdir(exist_ok=True)

        qa_config.project_name = str(output_dir)
        qa_config.save_total_limit = 2

        # Create a small model for testing
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        model = AutoModelForQuestionAnswering.from_pretrained(qa_config.model, ignore_mismatched_sizes=True)
        tokenizer = AutoTokenizer.from_pretrained(qa_config.model)

        # Save model
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Verify files exist
        assert (output_dir / "config.json").exists()
        assert (output_dir / "pytorch_model.bin").exists() or (output_dir / "model.safetensors").exists()
        assert (output_dir / "tokenizer_config.json").exists()

    def test_qa_long_context_handling(self, qa_config):
        """Test handling of long contexts with sliding window."""
        # Create a very long context
        long_context = " ".join(["This is sentence number " + str(i) + "." for i in range(100)])

        data = {
            "id": ["1"],
            "context": [long_context],
            "question": ["What is sentence 50?"],
            "answers": [{"text": ["sentence number 50"], "answer_start": [long_context.find("sentence number 50")]}],
        }

        dataset = Dataset.from_dict(data)
        tokenizer = AutoTokenizer.from_pretrained(qa_config.model)

        # Process with sliding window
        processed = qa_utils.prepare_qa_validation_features(examples=data, tokenizer=tokenizer, config=qa_config)

        # Should create multiple features from one example due to stride
        assert len(processed["input_ids"]) >= 1
        assert all(len(ids) <= qa_config.max_seq_length for ids in processed["input_ids"])


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for both trainers."""

    def test_seq2seq_end_to_end(self, tmp_path):
        """Test complete Seq2Seq training pipeline end-to-end."""
        # Create simple dataset
        data = {
            "text": ["Hello", "World", "Test"],
            "target": ["Bonjour", "Monde", "Test"],
        }
        dataset = Dataset.from_dict(data)

        # Save dataset
        data_path = tmp_path / "data"
        save_dataset_as_csv(dataset, data_path, "train")

        # Create config
        config = Seq2SeqParams(
            model="t5-small",
            data_path=str(data_path),
            project_name=str(tmp_path / "output"),
            text_column="text",
            target_column="target",
            max_seq_length=32,
            max_target_length=32,
            batch_size=1,
            epochs=1,
            train_split="train",
            max_samples=3,
            logging_steps=1,
            push_to_hub=False,
        )

        # Mock trainer to speed up test
        with patch("autotrain.trainers.seq2seq.__main__.Seq2SeqTrainer") as mock_trainer:
            mock_instance = MagicMock()
            mock_trainer.return_value = mock_instance
            mock_instance.train.return_value = None
            mock_instance.save_model.return_value = None

            # Run training
            seq2seq_train(config)

            assert mock_trainer.called

    def test_qa_end_to_end(self, tmp_path):
        """Test complete QA training pipeline end-to-end."""
        # Create simple dataset
        data = {
            "id": ["1", "2"],
            "context": [
                "Paris is the capital of France.",
                "Berlin is the capital of Germany.",
            ],
            "question": [
                "What is the capital of France?",
                "What is the capital of Germany?",
            ],
            "answers": [
                {"text": ["Paris"], "answer_start": [0]},
                {"text": ["Berlin"], "answer_start": [0]},
            ],
        }
        dataset = Dataset.from_dict(data)

        # Save dataset
        data_path = tmp_path / "data"
        save_dataset_as_csv(dataset, data_path, "train")

        # Create config
        config = ExtractiveQuestionAnsweringParams(
            model="bert-base-uncased",
            data_path=str(data_path),
            project_name=str(tmp_path / "output"),
            text_column="context",
            question_column="question",
            answer_column="answers",
            max_seq_length=128,
            batch_size=1,
            epochs=1,
            train_split="train",
            max_samples=2,
            logging_steps=1,
            push_to_hub=False,
        )

        # Mock trainer to speed up test
        with patch("autotrain.trainers.extractive_question_answering.__main__.Trainer") as mock_trainer:
            mock_instance = MagicMock()
            mock_trainer.return_value = mock_instance
            mock_instance.train.return_value = None
            mock_instance.save_model.return_value = None

            # Run training
            qa_train(config)

            assert mock_trainer.called

    def test_model_card_generation(self, tmp_path):
        """Test model card generation for both trainers."""
        # Test Seq2Seq model card
        seq2seq_config = Seq2SeqParams(
            model="t5-small",
            data_path="test_dataset",
            project_name=str(tmp_path / "seq2seq"),
            valid_split="validation",
        )

        mock_trainer = MagicMock()
        mock_trainer.evaluate.return_value = {
            "eval_loss": 0.5,
            "eval_rouge1": 45.0,
            "eval_rouge2": 20.0,
            "eval_rougeL": 40.0,
        }

        model_card = seq2seq_utils.create_model_card(seq2seq_config, mock_trainer)

        assert "Seq2Seq" in model_card
        assert "rouge1" in model_card
        assert "t5-small" in model_card

        # Test QA model card
        qa_config = ExtractiveQuestionAnsweringParams(
            model="bert-base-uncased",
            data_path="test_dataset",
            project_name=str(tmp_path / "qa"),
            valid_split="validation",
        )

        mock_trainer.evaluate.return_value = {
            "eval_loss": 0.3,
            "eval_exact_match": 70.0,
            "eval_f1": 85.0,
        }

        model_card = qa_utils.create_model_card(qa_config, mock_trainer)

        assert "Extractive Question Answering" in model_card
        assert "bert-base-uncased" in model_card


# =============================================================================
# Performance and Edge Case Tests
# =============================================================================


class TestPerformanceAndEdgeCases:
    """Test performance considerations and edge cases."""

    def test_seq2seq_empty_input(self):
        """Test Seq2Seq with empty input."""
        data = {
            "text": [""],
            "target": ["Empty input test"],
        }
        dataset = Dataset.from_dict(data)

        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        config = Seq2SeqParams(model="t5-small")

        # Should handle empty input gracefully
        seq2seq_dataset = Seq2SeqDataset(data=dataset, tokenizer=tokenizer, config=config)

        sample = seq2seq_dataset[0]
        assert "input_ids" in sample

    def test_qa_no_answer_found(self):
        """Test QA when answer is not in context."""
        data = {
            "id": ["1"],
            "context": ["The sky is blue."],
            "question": ["What color is the grass?"],
            "answers": [{"text": [], "answer_start": [-1]}],  # No answer
        }
        dataset = Dataset.from_dict(data)

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        config = ExtractiveQuestionAnsweringParams(model="bert-base-uncased")

        # Should handle no answer case
        qa_dataset = ExtractiveQuestionAnsweringDataset(data=dataset, tokenizer=tokenizer, config=config)

        # For unanswerable questions, positions should be 0
        sample = qa_dataset[0]
        assert sample["start_positions"] == 0
        assert sample["end_positions"] == 0

    def test_seq2seq_very_long_sequence(self):
        """Test Seq2Seq with very long sequences."""
        # Create very long text
        long_text = " ".join(["word"] * 1000)
        long_target = " ".join(["translation"] * 1000)

        data = {
            "text": [long_text],
            "target": [long_target],
        }
        dataset = Dataset.from_dict(data)

        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        config = Seq2SeqParams(
            model="t5-small",
            max_seq_length=128,
            max_target_length=128,
        )

        seq2seq_dataset = Seq2SeqDataset(data=dataset, tokenizer=tokenizer, config=config)

        sample = seq2seq_dataset[0]

        # Should truncate to max length
        assert len(sample["input_ids"]) <= config.max_seq_length
        assert len(sample["labels"]) <= config.max_target_length

    def test_mixed_precision_configs(self):
        """Test mixed precision training configurations."""
        # Test fp16
        config_fp16 = Seq2SeqParams(mixed_precision="fp16")
        assert config_fp16.mixed_precision == "fp16"

        # Test bf16
        config_bf16 = Seq2SeqParams(mixed_precision="bf16")
        assert config_bf16.mixed_precision == "bf16"

        # Test None (default)
        config_none = Seq2SeqParams(mixed_precision=None)
        assert config_none.mixed_precision is None

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_different_batch_sizes(self, batch_size):
        """Test training with different batch sizes."""
        config = Seq2SeqParams(batch_size=batch_size)
        assert config.batch_size == batch_size

        config_qa = ExtractiveQuestionAnsweringParams(batch_size=batch_size)
        assert config_qa.batch_size == batch_size

    def test_gradient_accumulation(self):
        """Test gradient accumulation settings."""
        config = Seq2SeqParams(
            batch_size=1,
            gradient_accumulation=4,
        )

        # Effective batch size = batch_size * gradient_accumulation
        effective_batch = config.batch_size * config.gradient_accumulation
        assert effective_batch == 4

    def test_early_stopping_configuration(self):
        """Test early stopping parameters."""
        config = Seq2SeqParams(
            early_stopping_patience=3,
            early_stopping_threshold=0.001,
        )

        assert config.early_stopping_patience == 3
        assert config.early_stopping_threshold == 0.001

    def test_scheduler_optimizer_combinations(self):
        """Test different scheduler and optimizer combinations."""
        schedulers = ["linear", "cosine", "constant"]
        optimizers = ["adamw_torch", "sgd", "adafactor"]

        for scheduler in schedulers:
            for optimizer in optimizers:
                config = Seq2SeqParams(
                    scheduler=scheduler,
                    optimizer=optimizer,
                )
                assert config.scheduler == scheduler
                assert config.optimizer == optimizer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
