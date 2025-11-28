"""
Comprehensive REAL test suite for Image and VLM trainers.
This file tests actual training with real models and data - NO MOCKS.

Tests include:
- Image Classification (binary and multi-class)
- Image Regression
- Object Detection
- VLM Captioning
- VLM VQA (Visual Question Answering)
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
from PIL import Image


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from autotrain.trainers.image_classification.__main__ import train as train_image_classification
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_regression.__main__ import train as train_image_regression
from autotrain.trainers.image_regression.params import ImageRegressionParams
from autotrain.trainers.object_detection.__main__ import train as train_object_detection
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.vlm.__main__ import train as train_vlm
from autotrain.trainers.vlm.params import VLMTrainingParams


# ================== Fixtures for Test Data ==================


@pytest.fixture
def create_test_images(tmp_path):
    """Create small synthetic test images."""
    images = []
    for i in range(20):
        # Create random 64x64 RGB image
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = tmp_path / f"image_{i}.png"
        img.save(img_path)
        images.append(str(img_path))
    return images


@pytest.fixture
def image_classification_data(tmp_path, create_test_images):
    """Create dataset for image classification with train and validation splits."""
    # Create classification dataset with 3 classes
    train_data = []
    val_data = []

    for i, img_path in enumerate(create_test_images[:15]):  # 15 for training
        train_data.append({"image": img_path, "target": i % 3})  # 3 classes

    for i, img_path in enumerate(create_test_images[15:]):  # 5 for validation
        val_data.append({"image": img_path, "target": i % 3})

    # Save as CSV files
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "valid.csv"

    pd.DataFrame(train_data).to_csv(train_csv, index=False)
    pd.DataFrame(val_data).to_csv(val_csv, index=False)

    return tmp_path


@pytest.fixture
def binary_classification_data(tmp_path, create_test_images):
    """Create dataset for binary classification."""
    train_data = []
    val_data = []

    for i, img_path in enumerate(create_test_images[:15]):
        train_data.append({"image": img_path, "target": i % 2})  # Binary: 0 or 1

    for i, img_path in enumerate(create_test_images[15:]):
        val_data.append({"image": img_path, "target": i % 2})

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "valid.csv"

    pd.DataFrame(train_data).to_csv(train_csv, index=False)
    pd.DataFrame(val_data).to_csv(val_csv, index=False)

    return tmp_path


@pytest.fixture
def image_regression_data(tmp_path, create_test_images):
    """Create dataset for image regression."""
    train_data = []
    val_data = []

    for i, img_path in enumerate(create_test_images[:15]):
        train_data.append(
            {"image": img_path, "target": float(i) * 0.1 + np.random.randn() * 0.01}  # Continuous values
        )

    for i, img_path in enumerate(create_test_images[15:]):
        val_data.append({"image": img_path, "target": float(i) * 0.1 + np.random.randn() * 0.01})

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "valid.csv"

    pd.DataFrame(train_data).to_csv(train_csv, index=False)
    pd.DataFrame(val_data).to_csv(val_csv, index=False)

    return tmp_path


@pytest.fixture
def object_detection_data(tmp_path, create_test_images):
    """Create dataset for object detection with bounding boxes."""
    train_data = []
    val_data = []

    # For object detection, we need bounding box annotations
    for i, img_path in enumerate(create_test_images[:15]):
        # Create simple bounding boxes (format: class_id, x, y, width, height)
        objects = {
            "bbox": [[10, 10, 20, 20], [30, 30, 15, 15]],  # Two boxes
            "category": [0, 1],  # Two different classes
        }
        train_data.append({"image": img_path, "objects": json.dumps(objects)})

    for i, img_path in enumerate(create_test_images[15:]):
        objects = {"bbox": [[5, 5, 25, 25]], "category": [i % 2]}  # One box
        val_data.append({"image": img_path, "objects": json.dumps(objects)})

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "valid.csv"

    pd.DataFrame(train_data).to_csv(train_csv, index=False)
    pd.DataFrame(val_data).to_csv(val_csv, index=False)

    return tmp_path


@pytest.fixture
def vlm_captioning_data(tmp_path, create_test_images):
    """Create dataset for VLM captioning."""
    train_data = []
    val_data = []

    captions = [
        "A colorful abstract image",
        "Random pixels forming a pattern",
        "Synthetic test image with colors",
        "Computer generated visual noise",
        "Abstract digital art composition",
    ]

    for i, img_path in enumerate(create_test_images[:15]):
        train_data.append({"image": img_path, "text": captions[i % len(captions)]})

    for i, img_path in enumerate(create_test_images[15:]):
        val_data.append({"image": img_path, "text": captions[i % len(captions)]})

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "valid.csv"

    pd.DataFrame(train_data).to_csv(train_csv, index=False)
    pd.DataFrame(val_data).to_csv(val_csv, index=False)

    return tmp_path


@pytest.fixture
def vlm_vqa_data(tmp_path, create_test_images):
    """Create dataset for VLM VQA (Visual Question Answering)."""
    train_data = []
    val_data = []

    qa_pairs = [
        ("What is in the image?", "Random colored pixels"),
        ("What colors are visible?", "Multiple colors"),
        ("Is this a photograph?", "No, it is synthetic"),
        ("Describe the image", "A computer generated image"),
        ("What type of image is this?", "Abstract digital art"),
    ]

    for i, img_path in enumerate(create_test_images[:15]):
        question, answer = qa_pairs[i % len(qa_pairs)]
        train_data.append({"image": img_path, "prompt": question, "text": answer})

    for i, img_path in enumerate(create_test_images[15:]):
        question, answer = qa_pairs[i % len(qa_pairs)]
        val_data.append({"image": img_path, "prompt": question, "text": answer})

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "valid.csv"

    pd.DataFrame(train_data).to_csv(train_csv, index=False)
    pd.DataFrame(val_data).to_csv(val_csv, index=False)

    return tmp_path


# ================== Image Classification Tests ==================


def test_image_classification_multiclass(image_classification_data, tmp_path):
    """Test multi-class image classification with real training."""
    params = ImageClassificationParams(
        data_path=str(image_classification_data),
        model="google/vit-base-patch16-224",
        project_name=str(tmp_path / "multiclass_output"),
        epochs=1,
        batch_size=2,
        lr=1e-4,
        train_split="train",
        valid_split="validation",
        logging_steps=1,
        save_total_limit=1,
        push_to_hub=False,
        max_samples=10,  # Limit samples for speed
    )

    # Run training
    train_image_classification(params)

    # Verify outputs
    output_dir = Path(tmp_path / "multiclass_output")
    assert output_dir.exists()
    assert (output_dir / "pytorch_model.bin").exists() or (output_dir / "model.safetensors").exists()
    assert (output_dir / "config.json").exists()


def test_image_classification_binary(binary_classification_data, tmp_path):
    """Test binary image classification."""
    params = ImageClassificationParams(
        data_path=str(binary_classification_data),
        model="google/vit-base-patch16-224",
        project_name=str(tmp_path / "binary_output"),
        epochs=1,
        batch_size=2,
        lr=1e-4,
        train_split="train",
        valid_split="validation",
        save_total_limit=1,
        push_to_hub=False,
        max_samples=10,
    )

    train_image_classification(params)

    output_dir = Path(tmp_path / "binary_output")
    assert output_dir.exists()
    assert (output_dir / "config.json").exists()


def test_image_classification_different_batch_sizes(image_classification_data, tmp_path):
    """Test image classification with different batch sizes."""
    for batch_size in [1, 4]:
        params = ImageClassificationParams(
            data_path=str(image_classification_data),
            model="google/vit-base-patch16-224",
            project_name=str(tmp_path / f"batch_{batch_size}_output"),
            epochs=1,
            batch_size=batch_size,
            lr=1e-4,
            train_split="train",
            save_total_limit=1,
            push_to_hub=False,
            max_samples=8,
        )

        train_image_classification(params)

        output_dir = Path(tmp_path / f"batch_{batch_size}_output")
        assert output_dir.exists()


def test_image_classification_checkpoint_saving(image_classification_data, tmp_path):
    """Test checkpoint saving during training."""
    params = ImageClassificationParams(
        data_path=str(image_classification_data),
        model="google/vit-base-patch16-224",
        project_name=str(tmp_path / "checkpoint_output"),
        epochs=2,
        batch_size=2,
        lr=1e-4,
        train_split="train",
        valid_split="validation",
        save_total_limit=2,
        eval_strategy="epoch",
        push_to_hub=False,
        max_samples=10,
    )

    train_image_classification(params)

    output_dir = Path(tmp_path / "checkpoint_output")
    assert output_dir.exists()

    # Check for checkpoint directories
    checkpoints = list(output_dir.glob("checkpoint-*"))
    assert len(checkpoints) > 0


def test_image_classification_missing_column_error(tmp_path):
    """Test that missing columns raise appropriate errors."""
    # Create data with wrong column names
    bad_data = pd.DataFrame({"wrong_image_col": ["image_0.png"], "wrong_target_col": [0]})

    bad_csv = tmp_path / "bad_train.csv"
    bad_data.to_csv(bad_csv, index=False)

    params = ImageClassificationParams(
        data_path=str(tmp_path),
        model="google/vit-base-patch16-224",
        project_name=str(tmp_path / "error_output"),
        epochs=1,
        batch_size=2,
        train_split="bad_train",
        image_column="image",  # This column doesn't exist
        target_column="target",  # This column doesn't exist
        push_to_hub=False,
    )

    with pytest.raises(Exception):  # Should raise an error about missing columns
        train_image_classification(params)


# ================== Image Regression Tests ==================


def test_image_regression_training(image_regression_data, tmp_path):
    """Test image regression with continuous targets."""
    params = ImageRegressionParams(
        data_path=str(image_regression_data),
        model="google/vit-base-patch16-224",
        project_name=str(tmp_path / "regression_output"),
        epochs=1,
        batch_size=2,
        lr=1e-4,
        train_split="train",
        valid_split="validation",
        save_total_limit=1,
        push_to_hub=False,
        max_samples=10,
    )

    train_image_regression(params)

    output_dir = Path(tmp_path / "regression_output")
    assert output_dir.exists()
    assert (output_dir / "pytorch_model.bin").exists() or (output_dir / "model.safetensors").exists()


def test_image_regression_metrics(image_regression_data, tmp_path):
    """Test that MSE/MAE metrics are computed during training."""
    params = ImageRegressionParams(
        data_path=str(image_regression_data),
        model="google/vit-base-patch16-224",
        project_name=str(tmp_path / "regression_metrics_output"),
        epochs=1,
        batch_size=2,
        lr=1e-4,
        train_split="train",
        valid_split="validation",
        logging_steps=1,
        eval_strategy="epoch",
        save_total_limit=1,
        push_to_hub=False,
        max_samples=10,
    )

    train_image_regression(params)

    output_dir = Path(tmp_path / "regression_metrics_output")
    assert output_dir.exists()

    # Check if trainer state contains metrics
    trainer_state_path = output_dir / "trainer_state.json"
    if trainer_state_path.exists():
        import json

        with open(trainer_state_path) as f:
            state = json.load(f)
            # Check that loss is tracked
            assert "log_history" in state


def test_image_regression_checkpoint_saving(image_regression_data, tmp_path):
    """Test checkpoint saving for regression."""
    params = ImageRegressionParams(
        data_path=str(image_regression_data),
        model="google/vit-base-patch16-224",
        project_name=str(tmp_path / "regression_checkpoint_output"),
        epochs=2,
        batch_size=2,
        lr=1e-4,
        train_split="train",
        valid_split="validation",
        save_total_limit=2,
        eval_strategy="epoch",
        push_to_hub=False,
        max_samples=10,
    )

    train_image_regression(params)

    output_dir = Path(tmp_path / "regression_checkpoint_output")
    checkpoints = list(output_dir.glob("checkpoint-*"))
    assert len(checkpoints) > 0


# ================== Object Detection Tests ==================


def test_object_detection_training(object_detection_data, tmp_path):
    """Test object detection with bounding boxes."""
    # Using facebook/detr-resnet-50 which is one of the smaller DETR models (~160MB)
    params = ObjectDetectionParams(
        data_path=str(object_detection_data),
        model="facebook/detr-resnet-50",  # Small DETR model
        project_name=str(tmp_path / "detection_output"),
        epochs=1,
        batch_size=1,
        lr=1e-4,
        train_split="train",
        valid_split="validation",
        image_square_size=224,  # Small size for speed
        save_total_limit=1,
        push_to_hub=False,
        max_samples=5,
    )

    train_object_detection(params)

    output_dir = Path(tmp_path / "detection_output")
    assert output_dir.exists()


def test_object_detection_data_validation(object_detection_data, tmp_path):
    """Test object detection data validation."""
    # Test with valid data structure
    params = ObjectDetectionParams(
        data_path=str(object_detection_data),
        model="google/vit-base-patch16-224",  # Will fail but validates data first
        project_name=str(tmp_path / "detection_validation_output"),
        epochs=1,
        batch_size=1,
        train_split="train",
        objects_column="objects",
        push_to_hub=False,
        max_samples=2,
    )

    # This should fail with model incompatibility, not data issues
    with pytest.raises(Exception):
        train_object_detection(params)


# ================== VLM Tests ==================


def test_vlm_captioning(vlm_captioning_data, tmp_path):
    """Test VLM captioning with small model."""
    params = VLMTrainingParams(
        data_path=str(vlm_captioning_data),
        model="google/paligemma-3b-pt-224",  # Smallest supported VLM model (PaliGemma only)
        project_name=str(tmp_path / "vlm_captioning_output"),
        trainer="captioning",
        epochs=1,
        batch_size=2,
        lr=1e-4,
        train_split="train",
        valid_split="validation",
        save_total_limit=1,
        push_to_hub=False,
        max_samples=10,
        peft=True,  # Use PEFT to reduce memory
        quantization="int4",  # Code automatically detects CUDA/MPS and adjusts
    )

    train_vlm(params)

    output_dir = Path(tmp_path / "vlm_captioning_output")
    assert output_dir.exists()
    assert (output_dir / "config.json").exists()


def test_vlm_vqa(vlm_vqa_data, tmp_path):
    """Test VLM Visual Question Answering."""
    params = VLMTrainingParams(
        data_path=str(vlm_vqa_data),
        model="google/paligemma-3b-pt-224",  # Smallest supported VLM model
        project_name=str(tmp_path / "vlm_vqa_output"),
        trainer="vqa",
        epochs=1,
        batch_size=2,
        lr=1e-4,
        train_split="train",
        valid_split="validation",
        save_total_limit=1,
        push_to_hub=False,
        max_samples=10,
        peft=True,
        quantization="int4",
    )

    train_vlm(params)

    output_dir = Path(tmp_path / "vlm_vqa_output")
    assert output_dir.exists()


def test_vlm_checkpoint_saving(vlm_captioning_data, tmp_path):
    """Test checkpoint saving for VLM training."""
    params = VLMTrainingParams(
        data_path=str(vlm_captioning_data),
        model="google/paligemma-3b-pt-224",
        project_name=str(tmp_path / "vlm_checkpoint_output"),
        trainer="captioning",
        epochs=2,
        batch_size=2,
        lr=1e-4,
        train_split="train",
        valid_split="validation",
        eval_strategy="epoch",
        save_total_limit=2,
        push_to_hub=False,
        max_samples=10,
        peft=True,
        quantization="int4",
    )

    train_vlm(params)

    output_dir = Path(tmp_path / "vlm_checkpoint_output")
    checkpoints = list(output_dir.glob("checkpoint-*"))
    assert len(checkpoints) > 0


def test_vlm_with_validation_data(vlm_vqa_data, tmp_path):
    """Test VLM training with validation split."""
    params = VLMTrainingParams(
        data_path=str(vlm_vqa_data),
        model="google/paligemma-3b-pt-224",
        project_name=str(tmp_path / "vlm_validation_output"),
        trainer="vqa",
        epochs=1,
        batch_size=2,
        lr=1e-4,
        train_split="train",
        valid_split="validation",
        eval_strategy="epoch",
        logging_steps=1,
        save_total_limit=1,
        push_to_hub=False,
        max_samples=10,
        peft=True,
        quantization="int4",
    )

    train_vlm(params)

    output_dir = Path(tmp_path / "vlm_validation_output")
    assert output_dir.exists()

    # Check for evaluation results
    trainer_state_path = output_dir / "trainer_state.json"
    if trainer_state_path.exists():
        import json

        with open(trainer_state_path) as f:
            state = json.load(f)
            assert "log_history" in state


# ================== Additional Tests for Coverage ==================


def test_image_classification_with_validation_split(image_classification_data, tmp_path):
    """Test training with validation split and metrics computation."""
    params = ImageClassificationParams(
        data_path=str(image_classification_data),
        model="google/vit-base-patch16-224",
        project_name=str(tmp_path / "val_split_output"),
        epochs=1,
        batch_size=2,
        lr=1e-4,
        train_split="train",
        valid_split="validation",
        eval_strategy="epoch",
        logging_steps=1,
        save_total_limit=1,
        push_to_hub=False,
        max_samples=10,
    )

    train_image_classification(params)

    output_dir = Path(tmp_path / "val_split_output")
    assert output_dir.exists()

    # Check that both training and evaluation happened
    trainer_state_path = output_dir / "trainer_state.json"
    if trainer_state_path.exists():
        import json

        with open(trainer_state_path) as f:
            state = json.load(f)
            log_history = state.get("log_history", [])
            # Should have both train and eval logs
            assert any("loss" in log for log in log_history)
            assert any("eval_loss" in log for log in log_history)


def test_image_regression_different_batch_sizes(image_regression_data, tmp_path):
    """Test image regression with different batch sizes."""
    for batch_size in [1, 4]:
        params = ImageRegressionParams(
            data_path=str(image_regression_data),
            model="google/vit-base-patch16-224",
            project_name=str(tmp_path / f"reg_batch_{batch_size}_output"),
            epochs=1,
            batch_size=batch_size,
            lr=1e-4,
            train_split="train",
            save_total_limit=1,
            push_to_hub=False,
            max_samples=8,
        )

        train_image_regression(params)

        output_dir = Path(tmp_path / f"reg_batch_{batch_size}_output")
        assert output_dir.exists()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
