"""
Comprehensive test suite for Image and VLM trainers including:
- Image Classification
- Image Regression
- Object Detection
- VLM Captioning
- VLM VQA
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    TrainingArguments,
)

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
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def dummy_image():
    """Create a dummy RGB image for testing."""
    # Create a small 32x32 RGB image with random colors
    img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGB")


@pytest.fixture
def classification_dataset(dummy_image):
    """Create a dummy dataset for image classification."""
    # Generate 20 images with 3 classes
    images = []
    labels = []

    for i in range(20):
        # Create unique image for each sample
        img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="RGB")
        images.append(img)
        labels.append(i % 3)  # 3 classes

    # Create features with proper image feature
    features = Features({"image": ImageFeature(), "target": Value("int32")})

    dataset = Dataset.from_dict({"image": images, "target": labels}, features=features)

    return dataset


@pytest.fixture
def regression_dataset(dummy_image):
    """Create a dummy dataset for image regression."""
    images = []
    scores = []

    for i in range(20):
        img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="RGB")
        images.append(img)
        scores.append(float(i) / 10.0)  # Regression scores

    features = Features({"image": ImageFeature(), "target": Value("float32")})

    dataset = Dataset.from_dict({"image": images, "target": scores}, features=features)

    return dataset


@pytest.fixture
def object_detection_dataset(dummy_image):
    """Create a dummy dataset for object detection."""
    images = []
    annotations = []

    for i in range(20):
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="RGB")
        images.append(img)

        # Create bounding boxes and labels
        objects = {
            "bbox": [[10, 10, 30, 30], [35, 35, 55, 55]],  # [x, y, width, height]
            "category": [0, 1],  # Two object categories
        }
        annotations.append(objects)

    features = Features(
        {"image": ImageFeature(), "objects": {"bbox": [[Value("float32")]], "category": [Value("int32")]}}
    )

    dataset = Dataset.from_dict({"image": images, "objects": annotations}, features=features)

    return dataset


@pytest.fixture
def vlm_dataset(dummy_image):
    """Create a dummy dataset for VLM (Vision-Language Model) tasks."""
    images = []
    texts = []
    answers = []

    for i in range(20):
        img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="RGB")
        images.append(img)
        texts.append(f"What is in image {i}?")
        answers.append(f"This is test image {i}")

    features = Features({"image": ImageFeature(), "text": Value("string"), "answer": Value("string")})

    dataset = Dataset.from_dict({"image": images, "text": texts, "answer": answers}, features=features)

    return dataset


# ================== Image Classification Tests ==================


class TestImageClassification:
    """Test suite for image classification trainer."""

    def test_params_initialization(self):
        """Test ImageClassificationParams initialization with default values."""
        params = ImageClassificationParams(data_path="dummy/path", project_name="test_project")

        assert params.model == "google/vit-base-patch16-224"
        assert params.lr == 5e-5
        assert params.epochs == 3
        assert params.batch_size == 8
        assert params.image_column == "image"
        assert params.target_column == "target"

    def test_params_custom_values(self):
        """Test ImageClassificationParams with custom values."""
        params = ImageClassificationParams(
            data_path="dummy/path",
            project_name="test_project",
            model="microsoft/resnet-18",
            lr=1e-4,
            epochs=5,
            batch_size=16,
            max_samples=100,
        )

        assert params.model == "microsoft/resnet-18"
        assert params.lr == 1e-4
        assert params.epochs == 5
        assert params.batch_size == 16
        assert params.max_samples == 100

    @patch("autotrain.trainers.image_classification.__main__.Trainer")
    @patch("autotrain.trainers.image_classification.__main__.AutoModelForImageClassification")
    @patch("autotrain.trainers.image_classification.__main__.AutoImageProcessor")
    def test_train_with_mock_model(self, mock_processor, mock_model, mock_trainer, classification_dataset, temp_dir):
        """Test image classification training with mocked model."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        # Create config
        config = ImageClassificationParams(
            data_path="dummy/path",
            project_name=temp_dir,
            model="google/vit-base-patch16-224",
            epochs=1,
            batch_size=4,
            max_samples=10,
            train_split="train",
            push_to_hub=False,
            token=None,
        )

        # Mock dataset loading
        with patch("autotrain.trainers.image_classification.__main__.load_dataset") as mock_load:
            mock_load.return_value = classification_dataset

            # Run training
            train_image_classification(config)

            # Verify model was loaded
            mock_model.from_pretrained.assert_called()
            mock_processor.from_pretrained.assert_called()

            # Verify trainer was called
            mock_trainer_instance.train.assert_called_once()
            mock_trainer_instance.save_model.assert_called_once()

    def test_data_validation(self, classification_dataset):
        """Test data validation for image classification."""
        # Check dataset has required columns
        assert "image" in classification_dataset.column_names
        assert "target" in classification_dataset.column_names

        # Check image types
        for img in classification_dataset["image"]:
            assert isinstance(img, Image.Image)

        # Check label types and range
        labels = classification_dataset["target"]
        assert all(isinstance(label, int) for label in labels)
        assert min(labels) >= 0
        assert max(labels) == 2  # 3 classes (0, 1, 2)


# ================== Image Regression Tests ==================


class TestImageRegression:
    """Test suite for image regression trainer."""

    def test_params_initialization(self):
        """Test ImageRegressionParams initialization."""
        params = ImageRegressionParams(data_path="dummy/path", project_name="test_project")

        assert params.model == "google/vit-base-patch16-224"
        assert params.lr == 5e-5
        assert params.epochs == 3
        assert params.batch_size == 8

    @patch("autotrain.trainers.image_regression.__main__.Trainer")
    @patch("autotrain.trainers.image_regression.__main__.AutoModelForImageClassification")
    @patch("autotrain.trainers.image_regression.__main__.AutoImageProcessor")
    def test_train_regression(self, mock_processor, mock_model, mock_trainer, regression_dataset, temp_dir):
        """Test image regression training."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        config = ImageRegressionParams(
            data_path="dummy/path",
            project_name=temp_dir,
            model="google/vit-base-patch16-224",
            epochs=1,
            batch_size=4,
            max_samples=10,
            train_split="train",
            push_to_hub=False,
            token=None,
        )

        with patch("autotrain.trainers.image_regression.__main__.load_dataset") as mock_load:
            mock_load.return_value = regression_dataset

            train_image_regression(config)

            mock_model.from_pretrained.assert_called()
            mock_trainer_instance.train.assert_called_once()

    def test_regression_data_validation(self, regression_dataset):
        """Test data validation for regression dataset."""
        assert "image" in regression_dataset.column_names
        assert "target" in regression_dataset.column_names

        # Check target values are floats
        targets = regression_dataset["target"]
        assert all(isinstance(t, float) for t in targets)


# ================== Object Detection Tests ==================


class TestObjectDetection:
    """Test suite for object detection trainer."""

    def test_params_initialization(self):
        """Test ObjectDetectionParams initialization."""
        params = ObjectDetectionParams(data_path="dummy/path", project_name="test_project")

        # Default model for object detection
        assert params.model == "facebook/detr-resnet-50"
        assert params.lr == 5e-5
        assert params.epochs == 3
        assert params.batch_size == 8

    @patch("autotrain.trainers.object_detection.__main__.Trainer")
    @patch("autotrain.trainers.object_detection.__main__.AutoModelForObjectDetection")
    @patch("autotrain.trainers.object_detection.__main__.AutoImageProcessor")
    def test_train_object_detection(
        self, mock_processor, mock_model, mock_trainer, object_detection_dataset, temp_dir
    ):
        """Test object detection training."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        config = ObjectDetectionParams(
            data_path="dummy/path",
            project_name=temp_dir,
            model="facebook/detr-resnet-50",
            epochs=1,
            batch_size=2,
            max_samples=10,
            train_split="train",
            push_to_hub=False,
            token=None,
        )

        with patch("autotrain.trainers.object_detection.__main__.load_dataset") as mock_load:
            mock_load.return_value = object_detection_dataset

            train_object_detection(config)

            mock_model.from_pretrained.assert_called()
            mock_trainer_instance.train.assert_called_once()

    def test_object_detection_data_validation(self, object_detection_dataset):
        """Test data validation for object detection dataset."""
        assert "image" in object_detection_dataset.column_names
        assert "objects" in object_detection_dataset.column_names

        # Check structure of objects
        for obj in object_detection_dataset["objects"]:
            assert "bbox" in obj
            assert "category" in obj
            assert len(obj["bbox"]) == len(obj["category"])

            # Check bbox format
            for bbox in obj["bbox"]:
                assert len(bbox) == 4  # [x, y, width, height]


# ================== VLM Tests ==================


class TestVLM:
    """Test suite for Vision-Language Model trainers."""

    def test_vlm_params_initialization(self):
        """Test VLMTrainingParams initialization."""
        params = VLMTrainingParams(
            data_path="dummy/path",
            project_name="test_project",
            model="microsoft/Florence-2-base",
            trainer="captioning",
        )

        assert params.model == "microsoft/Florence-2-base"
        assert params.trainer == "captioning"
        assert params.lr == 5e-5
        assert params.epochs == 3
        assert params.batch_size == 2  # VLM typically uses smaller batch size

    def test_vlm_trainer_types(self):
        """Test VLM supports both captioning and VQA trainers."""
        # Test captioning trainer
        params_caption = VLMTrainingParams(
            data_path="dummy/path",
            project_name="test_project",
            model="microsoft/Florence-2-base",
            trainer="captioning",
        )
        assert params_caption.trainer == "captioning"

        # Test VQA trainer
        params_vqa = VLMTrainingParams(
            data_path="dummy/path", project_name="test_project", model="microsoft/Florence-2-base", trainer="vqa"
        )
        assert params_vqa.trainer == "vqa"

    @patch("autotrain.trainers.vlm.train_vlm_generic.train")
    @patch("autotrain.trainers.vlm.utils.check_model_support")
    def test_train_vlm_captioning(self, mock_check_support, mock_train_generic, vlm_dataset, temp_dir):
        """Test VLM captioning trainer."""
        mock_check_support.return_value = True

        config = VLMTrainingParams(
            data_path="dummy/path",
            project_name=temp_dir,
            model="microsoft/Florence-2-base",
            trainer="captioning",
            epochs=1,
            batch_size=2,
            max_samples=10,
            train_split="train",
            push_to_hub=False,
            token=None,
        )

        train_vlm(config)

        mock_check_support.assert_called_once_with(config)
        mock_train_generic.assert_called_once_with(config)

    @patch("autotrain.trainers.vlm.train_vlm_generic.train")
    @patch("autotrain.trainers.vlm.utils.check_model_support")
    def test_train_vlm_vqa(self, mock_check_support, mock_train_generic, vlm_dataset, temp_dir):
        """Test VLM VQA trainer."""
        mock_check_support.return_value = True

        config = VLMTrainingParams(
            data_path="dummy/path",
            project_name=temp_dir,
            model="microsoft/Florence-2-base",
            trainer="vqa",
            epochs=1,
            batch_size=2,
            max_samples=10,
            train_split="train",
            push_to_hub=False,
            token=None,
        )

        train_vlm(config)

        mock_check_support.assert_called_once_with(config)
        mock_train_generic.assert_called_once_with(config)

    def test_vlm_data_validation(self, vlm_dataset):
        """Test data validation for VLM dataset."""
        assert "image" in vlm_dataset.column_names
        assert "text" in vlm_dataset.column_names
        assert "answer" in vlm_dataset.column_names

        # Check data types
        for i in range(len(vlm_dataset)):
            assert isinstance(vlm_dataset[i]["image"], Image.Image)
            assert isinstance(vlm_dataset[i]["text"], str)
            assert isinstance(vlm_dataset[i]["answer"], str)

    def test_unsupported_model_error(self):
        """Test that unsupported models raise an error."""
        with patch("autotrain.trainers.vlm.utils.check_model_support") as mock_check:
            mock_check.return_value = False

            config = VLMTrainingParams(
                data_path="dummy/path", project_name="test_project", model="unsupported/model", trainer="captioning"
            )

            with pytest.raises(ValueError, match="not supported"):
                train_vlm(config)

    def test_unsupported_trainer_error(self):
        """Test that unsupported trainer types raise an error."""
        with patch("autotrain.trainers.vlm.utils.check_model_support") as mock_check:
            mock_check.return_value = True

            config = VLMTrainingParams(
                data_path="dummy/path",
                project_name="test_project",
                model="microsoft/Florence-2-base",
                trainer="unsupported_trainer",
            )

            with pytest.raises(ValueError, match="trainer.*not supported"):
                train_vlm(config)


# ================== Integration and Checkpoint Tests ==================


class TestCheckpointSaving:
    """Test checkpoint saving functionality for all trainers."""

    @patch("autotrain.trainers.image_classification.__main__.Trainer")
    @patch("autotrain.trainers.image_classification.__main__.AutoModelForImageClassification")
    @patch("autotrain.trainers.image_classification.__main__.AutoImageProcessor")
    def test_checkpoint_saving_image_classification(
        self, mock_processor, mock_model, mock_trainer, classification_dataset, temp_dir
    ):
        """Test that checkpoints are saved during image classification training."""
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        config = ImageClassificationParams(
            data_path="dummy/path",
            project_name=temp_dir,
            model="google/vit-base-patch16-224",
            epochs=2,
            batch_size=4,
            save_total_limit=2,
            eval_strategy="epoch",
            train_split="train",
            valid_split="validation",
            push_to_hub=False,
        )

        with patch("autotrain.trainers.image_classification.__main__.load_dataset") as mock_load:
            mock_load.return_value = classification_dataset

            train_image_classification(config)

            # Verify save_model was called
            mock_trainer_instance.save_model.assert_called_with(temp_dir)
            mock_processor_instance.save_pretrained.assert_called_with(temp_dir)


class TestDataValidation:
    """Test data validation for all trainer types."""

    def test_invalid_num_classes(self, temp_dir):
        """Test that training fails with less than 2 classes."""
        # Create dataset with only 1 class
        images = [Image.new("RGB", (32, 32)) for _ in range(10)]
        labels = [0] * 10  # All same class

        features = Features({"image": ImageFeature(), "target": Value("int32")})

        invalid_dataset = Dataset.from_dict({"image": images, "target": labels}, features=features)

        config = ImageClassificationParams(
            data_path="dummy/path",
            project_name=temp_dir,
            model="google/vit-base-patch16-224",
            epochs=1,
            batch_size=4,
            train_split="train",
            push_to_hub=False,
        )

        with patch("autotrain.trainers.image_classification.__main__.load_dataset") as mock_load:
            mock_load.return_value = invalid_dataset

            with pytest.raises(ValueError, match="Invalid number of classes"):
                train_image_classification(config)

    def test_mismatched_classes_train_valid(self, temp_dir):
        """Test that training fails when train and valid have different number of classes."""
        # Train dataset with 3 classes
        train_images = [Image.new("RGB", (32, 32)) for _ in range(15)]
        train_labels = [i % 3 for i in range(15)]

        train_features = Features({"image": ImageFeature(), "target": Value("int32")})

        train_dataset = Dataset.from_dict({"image": train_images, "target": train_labels}, features=train_features)

        # Valid dataset with 2 classes
        valid_images = [Image.new("RGB", (32, 32)) for _ in range(10)]
        valid_labels = [i % 2 for i in range(10)]

        valid_dataset = Dataset.from_dict({"image": valid_images, "target": valid_labels}, features=train_features)

        config = ImageClassificationParams(
            data_path="dummy/path",
            project_name=temp_dir,
            model="google/vit-base-patch16-224",
            epochs=1,
            batch_size=4,
            train_split="train",
            valid_split="validation",
            push_to_hub=False,
        )

        with patch("autotrain.trainers.image_classification.__main__.load_dataset") as mock_load:
            mock_load.side_effect = [train_dataset, valid_dataset]

            with pytest.raises(ValueError, match="Number of classes in train and valid are not the same"):
                train_image_classification(config)


# ================== Performance Tests ==================


class TestPerformanceOptimization:
    """Test performance optimizations like max_samples."""

    def test_max_samples_limits_dataset(self, classification_dataset, temp_dir):
        """Test that max_samples properly limits the dataset size."""
        config = ImageClassificationParams(
            data_path="dummy/path",
            project_name=temp_dir,
            model="google/vit-base-patch16-224",
            epochs=1,
            batch_size=4,
            max_samples=6,  # Limit to 6 samples
            train_split="train",
            push_to_hub=False,
        )

        with patch("autotrain.trainers.image_classification.__main__.load_dataset") as mock_load:
            mock_load.return_value = classification_dataset

            with patch("autotrain.trainers.image_classification.__main__.Trainer") as mock_trainer:
                mock_trainer_instance = MagicMock()
                mock_trainer.return_value = mock_trainer_instance

                with patch("autotrain.trainers.image_classification.__main__.AutoModelForImageClassification"):
                    with patch("autotrain.trainers.image_classification.__main__.AutoImageProcessor"):
                        train_image_classification(config)

                        # Check that train_dataset passed to Trainer has correct size
                        call_kwargs = mock_trainer.call_args.kwargs
                        train_dataset = call_kwargs.get("train_dataset")

                        # Due to balanced sampling, we should have 2 samples per class (6 total / 3 classes)
                        assert len(train_dataset) <= 6


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
