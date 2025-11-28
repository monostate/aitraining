# Image Classification Trainer

## Overview

The Image Classification trainer enables fine-tuning of vision transformer models for categorizing images into predefined classes. It supports both binary classification (two classes) and multi-class classification (multiple classes) tasks using state-of-the-art vision models.

This trainer handles the complete pipeline from data loading through model training and evaluation, with support for data augmentation, mixed precision training, and automatic model card generation. The implementation is based on the Hugging Face Transformers library and uses Albumentations for robust image augmentation.

## Use Cases

- **Product Categorization:** Classify products by type, category, or quality
- **Medical Image Analysis:** Identify diseases or abnormalities in medical scans
- **Wildlife Monitoring:** Classify animal species from camera trap images
- **Quality Control:** Detect defects or classify product quality in manufacturing
- **Content Moderation:** Identify inappropriate or sensitive visual content
- **Document Classification:** Categorize scanned documents by type
- **Agricultural Monitoring:** Identify crop diseases or classify plant types
- **Retail Analytics:** Classify customer behavior or store layouts
- **Fashion Classification:** Categorize clothing items by style, color, or type
- **Satellite Image Analysis:** Classify land use or detect changes in terrain

## Supported Models

Any model compatible with AutoModelForImageClassification from Hugging Face:

### Vision Transformers (ViT)
- `google/vit-base-patch16-224` (default)
- `google/vit-large-patch16-224`
- `microsoft/beit-base-patch16-224`
- `microsoft/swin-base-patch4-window7-224`

### ConvNet-based Models
- `microsoft/resnet-50`
- `facebook/convnext-base-224`
- `microsoft/efficientnet-b0`

### Hybrid Models
- `facebook/deit-base-patch16-224`
- `facebook/convnext-tiny-224`

### Custom Models
Any custom vision model from Hugging Face Hub that supports AutoModelForImageClassification

## Data Format and Structure

### Hugging Face Dataset Format

The trainer expects datasets with an image column and a target column containing class labels.

**Example Dataset Structure:**
```python
from datasets import Dataset, Features, ClassLabel, Image

features = Features({
    'image': Image(),
    'target': ClassLabel(names=['cat', 'dog', 'bird'])
})

# Dataset automatically handles PIL Images
dataset = Dataset.from_dict({
    'image': ['path/to/image1.jpg', 'path/to/image2.jpg'],
    'target': [0, 1]  # class indices
}, features=features)
```

### Directory Structure

For local data, organize images in class-named folders:
```
data/
  train/
    cat/
      image1.jpg
      image2.jpg
    dog/
      image3.jpg
      image4.jpg
    bird/
      image5.jpg
```

Or provide a preprocessed Hugging Face Dataset:
```
project-name/autotrain-data/
  train/
  validation/
```

### CSV Format (Custom Loading)

While the trainer expects Hugging Face datasets, you can prepare CSV files for conversion:
```csv
image_path,label
/path/to/cat1.jpg,cat
/path/to/dog1.jpg,dog
/path/to/bird1.jpg,bird
```

Convert to Dataset format before training:
```python
import pandas as pd
from datasets import Dataset, Features, ClassLabel, Image

df = pd.read_csv('data.csv')
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column('image_path', Image())
```

## Parameters

### Required Parameters

- `data_path` (str): Path to the dataset (Hugging Face dataset name or local directory)
- `model` (str): Pre-trained model name or path (default: "google/vit-base-patch16-224")
- `project_name` (str): Output directory name (default: "project-name")

### Data Parameters

- `image_column` (str): Column name for images (default: "image")
- `target_column` (str): Column name for target labels (default: "target")
- `train_split` (str): Training data split name (default: "train")
- `valid_split` (Optional[str]): Validation data split name (default: None)
- `max_samples` (Optional[int]): Maximum samples to use for testing/debugging (default: None)

### Training Parameters

- `lr` (float): Learning rate (default: 5e-5)
- `epochs` (int): Number of training epochs (default: 3)
- `batch_size` (int): Training batch size (default: 8)
- `warmup_ratio` (float): Warmup proportion for learning rate scheduler (default: 0.1)
- `gradient_accumulation` (int): Gradient accumulation steps (default: 1)
- `optimizer` (str): Optimizer type (default: "adamw_torch")
- `scheduler` (str): Learning rate scheduler (default: "linear")
- `weight_decay` (float): Weight decay for optimizer (default: 0.0)
- `max_grad_norm` (float): Maximum gradient norm for clipping (default: 1.0)
- `seed` (int): Random seed for reproducibility (default: 42)

### Advanced Parameters

- `mixed_precision` (Optional[str]): Mixed precision mode - "fp16", "bf16", or None (default: None)
- `auto_find_batch_size` (bool): Automatically find optimal batch size (default: False)
- `save_total_limit` (int): Maximum checkpoints to keep (default: 1)
- `logging_steps` (int): Steps between logging (-1 for auto) (default: -1)
- `eval_strategy` (str): Evaluation strategy during training (default: "epoch")
- `early_stopping_patience` (int): Epochs with no improvement before stopping (default: 5)
- `early_stopping_threshold` (float): Minimum improvement threshold (default: 0.01)

### Hub Parameters

- `push_to_hub` (bool): Push model to Hugging Face Hub (default: False)
- `username` (Optional[str]): Hugging Face username for pushing models
- `token` (Optional[str]): Hugging Face Hub authentication token
- `log` (str): Experiment tracking method - "none", "wandb", "tensorboard" (default: "none")

## Command Line Usage

### Basic Binary Classification

```bash
autotrain image-classification \
  --model google/vit-base-patch16-224 \
  --data-path hf-datasets/cats-vs-dogs \
  --project-name cat-dog-classifier \
  --image-column image \
  --target-column label \
  --train-split train \
  --valid-split validation \
  --epochs 5 \
  --batch-size 16 \
  --lr 2e-5 \
  --train
```

### Multi-class Classification with Mixed Precision

```bash
autotrain image-classification \
  --model microsoft/swin-base-patch4-window7-224 \
  --data-path ./animal_dataset \
  --project-name animal-classifier \
  --image-column image \
  --target-column species \
  --train-split train \
  --valid-split test \
  --epochs 10 \
  --batch-size 8 \
  --lr 3e-5 \
  --mixed-precision fp16 \
  --gradient-accumulation 4 \
  --early-stopping-patience 3 \
  --train
```

### Training with Hub Upload

```bash
autotrain image-classification \
  --model google/vit-base-patch16-224 \
  --data-path medical-images/xray-classification \
  --project-name medical-xray-classifier \
  --epochs 15 \
  --batch-size 32 \
  --lr 1e-5 \
  --scheduler cosine \
  --push-to-hub \
  --username your-username \
  --token $HF_TOKEN \
  --train
```

## Python API Usage

### Basic Training Script

```python
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_classification import train

# Configure training parameters
params = ImageClassificationParams(
    model="google/vit-base-patch16-224",
    data_path="food101",
    project_name="food-classifier",
    image_column="image",
    target_column="label",
    train_split="train",
    valid_split="validation",
    lr=2e-5,
    epochs=5,
    batch_size=16,
    warmup_ratio=0.1,
    gradient_accumulation=2,
    optimizer="adamw_torch",
    scheduler="linear",
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=42,
    mixed_precision="fp16",
    auto_find_batch_size=False,
    save_total_limit=2,
    early_stopping_patience=3,
    early_stopping_threshold=0.01,
    push_to_hub=False
)

# Start training
train(params)
```

### Training with Custom Dataset

```python
from datasets import load_dataset, DatasetDict
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_classification import train

# Load your custom dataset
dataset = load_dataset("imagefolder", data_dir="./my_images")

# Save it in the expected format
dataset_dict = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})
dataset_dict.save_to_disk("./processed_data")

# Configure and train
params = ImageClassificationParams(
    model="microsoft/resnet-50",
    data_path="./processed_data",
    project_name="custom-classifier",
    train_split="train",
    valid_split="validation",
    epochs=10,
    batch_size=32,
    lr=3e-5
)

train(params)
```

### Advanced Configuration with Logging

```python
import os
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_classification import train

# Set up WandB logging
os.environ["WANDB_PROJECT"] = "image-classification"

params = ImageClassificationParams(
    model="facebook/convnext-base-224",
    data_path="cifar100",
    project_name="cifar100-classifier",
    train_split="train",
    valid_split="test",
    lr=1e-4,
    epochs=20,
    batch_size=64,
    warmup_ratio=0.05,
    gradient_accumulation=1,
    optimizer="adamw_torch",
    scheduler="cosine",
    weight_decay=0.05,
    mixed_precision="bf16",
    logging_steps=50,
    eval_strategy="steps",
    save_total_limit=3,
    early_stopping_patience=5,
    log="wandb",  # Enable WandB logging
    push_to_hub=True,
    username="my-username",
    token=os.environ["HF_TOKEN"]
)

train(params)
```

## Data Preparation and Augmentation

### Built-in Augmentation Pipeline

The trainer automatically applies augmentations during training:

**Training Augmentations:**
- Random resized crop to model input size
- Random 90-degree rotation
- Horizontal flip (50% probability)
- Random brightness and contrast adjustment (20% probability)
- Normalization using ImageNet statistics

**Validation Augmentations:**
- Resize to model input size
- Normalization only (no random augmentations)

These are implemented using Albumentations in `utils.py`:

```python
# Training transforms
train_transforms = A.Compose([
    A.RandomResizedCrop(height=height, width=width),
    A.RandomRotate90(),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])

# Validation transforms
val_transforms = A.Compose([
    A.Resize(height=height, width=width),
    A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])
```

### Preparing Your Dataset

#### From Image Folders

```python
from datasets import load_dataset

# Load from directory structure
dataset = load_dataset(
    "imagefolder",
    data_dir="./my_data",
    split="train"
)

# The dataset will automatically infer classes from folder names
print(dataset.features)
```

#### From Custom Sources

```python
from datasets import Dataset, Features, ClassLabel, Image as HFImage
from PIL import Image
import os

# Collect image paths and labels
images = []
labels = []

for label_idx, class_name in enumerate(['cat', 'dog', 'bird']):
    class_dir = f'data/{class_name}'
    for img_name in os.listdir(class_dir):
        images.append(os.path.join(class_dir, img_name))
        labels.append(label_idx)

# Create dataset
dataset = Dataset.from_dict({
    'image': images,
    'target': labels
})

# Cast image column to Image type
dataset = dataset.cast_column('image', HFImage())

# Add class names
dataset = dataset.cast_column(
    'target',
    ClassLabel(names=['cat', 'dog', 'bird'])
)
```

### Handling Imbalanced Datasets

For imbalanced classes, the trainer includes balanced sampling when using `max_samples`:

```python
# The trainer automatically balances samples per class
params = ImageClassificationParams(
    model="google/vit-base-patch16-224",
    data_path="imbalanced_dataset",
    max_samples=1000,  # Will sample ~333 per class for 3 classes
    # ... other params
)
```

For custom class weighting, you may need to modify the dataset:

```python
from collections import Counter

# Check class distribution
labels = dataset['train']['target']
class_counts = Counter(labels)
print(f"Class distribution: {class_counts}")

# Oversample minority classes
from datasets import concatenate_datasets

minority_class_data = dataset.filter(lambda x: x['target'] == minority_class)
balanced_dataset = concatenate_datasets([
    dataset,
    minority_class_data,
    minority_class_data
])
```

## Best Practices

### 1. Start with a Pre-trained Model

Always use a pre-trained vision model as your starting point. These models have learned robust visual features from large datasets like ImageNet.

### 2. Choose the Right Model Size

- **Small datasets (<1000 images):** Use smaller models like `google/vit-base-patch16-224`
- **Medium datasets (1000-10000 images):** Try `microsoft/swin-base-patch4-window7-224`
- **Large datasets (>10000 images):** Can use larger models like `google/vit-large-patch16-224`

### 3. Hyperparameter Tuning

Start with these recommended ranges:
- Learning rate: 1e-5 to 5e-5
- Batch size: 8-32 (depending on GPU memory)
- Epochs: 3-10 (use early stopping)
- Warmup ratio: 0.05-0.1

### 4. Monitor Training Progress

Always use a validation split to monitor for overfitting:
```python
params = ImageClassificationParams(
    # ... other params
    valid_split="validation",
    eval_strategy="epoch",
    early_stopping_patience=3,
    save_total_limit=2
)
```

### 5. Use Mixed Precision Training

Enable mixed precision for faster training and lower memory usage:
```python
params = ImageClassificationParams(
    # ... other params
    mixed_precision="fp16"  # or "bf16" for newer GPUs
)
```

### 6. Data Quality Matters

- Ensure consistent image quality across classes
- Remove corrupt or mislabeled images
- Validate class definitions are clear and mutually exclusive
- Maintain similar image sizes and aspect ratios when possible

## Evaluation Metrics

### Binary Classification

The trainer computes the following metrics for binary classification:
- **Accuracy:** Overall correctness percentage
- **F1 Score:** Harmonic mean of precision and recall
- **Precision:** Ratio of correct positive predictions
- **Recall:** Ratio of true positives identified
- **AUC (Area Under ROC Curve):** Measure of classifier performance

### Multi-class Classification

For multi-class problems, the trainer computes:
- **Accuracy:** Overall correctness percentage
- **F1 Macro:** Unweighted mean F1 across classes
- **F1 Micro:** Globally computed F1 score
- **F1 Weighted:** Weighted by class support
- **Precision (Macro/Micro/Weighted)**
- **Recall (Macro/Micro/Weighted)**

### Accessing Metrics

Metrics are automatically logged during training and saved in the model card:

```python
# After training, metrics are in the model's README.md
with open(f"{params.project_name}/README.md", 'r') as f:
    model_card = f.read()
    print(model_card)
```

## Troubleshooting Common Issues

### Out of Memory Errors

**Symptoms:** CUDA out of memory, RuntimeError during training

**Solutions:**
1. Reduce batch size:
   ```python
   params.batch_size = 4
   ```

2. Enable gradient accumulation:
   ```python
   params.batch_size = 4
   params.gradient_accumulation = 4  # Effective batch size = 16
   ```

3. Use mixed precision:
   ```python
   params.mixed_precision = "fp16"
   ```

4. Try a smaller model:
   ```python
   params.model = "google/vit-base-patch16-224"  # instead of vit-large
   ```

5. Enable automatic batch size finding:
   ```python
   params.auto_find_batch_size = True
   ```

### Poor Model Performance

**Symptoms:** Low accuracy, high loss, not converging

**Solutions:**
1. Increase training epochs:
   ```python
   params.epochs = 10  # instead of 3
   ```

2. Adjust learning rate:
   ```python
   params.lr = 2e-5  # try different values in range [1e-5, 5e-5]
   ```

3. Check data quality:
   - Verify labels are correct
   - Ensure images are not corrupt
   - Check class balance

4. Use a different scheduler:
   ```python
   params.scheduler = "cosine"  # instead of "linear"
   ```

5. Add regularization:
   ```python
   params.weight_decay = 0.01
   ```

### Training is Too Slow

**Symptoms:** Very slow iteration speed, long training time

**Solutions:**
1. Enable mixed precision:
   ```python
   params.mixed_precision = "fp16"
   ```

2. Increase batch size (if memory allows):
   ```python
   params.batch_size = 32
   ```

3. Reduce logging frequency:
   ```python
   params.logging_steps = 100
   ```

4. Reduce validation frequency:
   ```python
   params.eval_strategy = "epoch"  # instead of "steps"
   ```

### Model Not Learning

**Symptoms:** Loss stays constant, accuracy doesn't improve

**Solutions:**
1. Verify data loading:
   ```python
   # Check that images and labels are correctly loaded
   from datasets import load_from_disk
   dataset = load_from_disk(params.data_path)
   print(dataset['train'][0])
   ```

2. Increase learning rate:
   ```python
   params.lr = 5e-5  # higher learning rate
   ```

3. Remove weight decay initially:
   ```python
   params.weight_decay = 0.0
   ```

4. Check class distribution:
   ```python
   # Ensure classes are balanced
   from collections import Counter
   labels = dataset['train']['target']
   print(Counter(labels))
   ```

### Validation Metrics Not Appearing

**Symptoms:** No validation metrics in logs or model card

**Solutions:**
1. Ensure validation split is specified:
   ```python
   params.valid_split = "validation"  # Must be set
   ```

2. Check validation data exists:
   ```python
   from datasets import load_from_disk
   dataset = load_from_disk(params.data_path)
   print(dataset.keys())  # Should include validation split
   ```

3. Verify eval strategy:
   ```python
   params.eval_strategy = "epoch"  # Not "no"
   ```

## Example Projects

### Product Quality Classification

```python
from datasets import load_dataset
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_classification import train

# 1. Prepare dataset
dataset = load_dataset("imagefolder", data_dir="./product_images")
dataset.save_to_disk("./product_data")

# 2. Configure training
params = ImageClassificationParams(
    model="microsoft/resnet-50",
    data_path="./product_data",
    project_name="product-quality-classifier",
    image_column="image",
    target_column="label",  # "good", "defective"
    train_split="train",
    valid_split="test",
    lr=3e-5,
    epochs=8,
    batch_size=32,
    mixed_precision="fp16",
    early_stopping_patience=3
)

# 3. Train model
train(params)

# 4. Use the trained model
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

processor = AutoImageProcessor.from_pretrained("./product-quality-classifier")
model = AutoModelForImageClassification.from_pretrained("./product-quality-classifier")

image = Image.open("test_product.jpg")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()
print(f"Quality: {model.config.id2label[predicted_class]}")
```

### Medical Image Classification

```python
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_classification import train

# Medical imaging often requires careful handling
params = ImageClassificationParams(
    model="google/vit-base-patch16-224",
    data_path="medical-imaging/chest-xray",
    project_name="pneumonia-classifier",
    train_split="train",
    valid_split="validation",
    lr=1e-5,  # Lower learning rate for medical images
    epochs=15,
    batch_size=16,
    warmup_ratio=0.1,
    optimizer="adamw_torch",
    scheduler="cosine",
    weight_decay=0.01,
    mixed_precision="fp16",
    early_stopping_patience=5,
    seed=42,
    log="wandb"  # Track experiments carefully
)

train(params)
```

## Integration with Hugging Face Hub

### Pushing Model to Hub

```python
params = ImageClassificationParams(
    model="google/vit-base-patch16-224",
    data_path="./my_data",
    project_name="my-classifier",
    push_to_hub=True,
    username="my-hf-username",
    token="hf_your_token_here",
    # ... other params
)

train(params)
# Model will be automatically uploaded to huggingface.co/my-hf-username/my-classifier
```

### Using Trained Model

```python
from transformers import pipeline

# Load from Hub
classifier = pipeline(
    "image-classification",
    model="my-hf-username/my-classifier"
)

# Classify new images
result = classifier("path/to/image.jpg")
print(result)
# [{'label': 'cat', 'score': 0.98}]
```

## See Also

- [Object Detection](./ObjectDetection.md) - For detecting and localizing objects in images
- [VLM (Vision-Language Models)](./VLM.md) - For VQA and image captioning tasks
- [Image Regression](../vision/ImageRegression.md) - For predicting continuous values from images
