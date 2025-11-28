# Image Regression Trainer

## Overview

The Image Regression trainer fine-tunes vision models to predict continuous numerical values from images. Unlike classification which predicts discrete categories, regression outputs continuous values like scores, measurements, or quantities.

## Use Cases

- **Age Estimation:** Predict person's age from facial images
- **Quality Assessment:** Score product quality (0-100)
- **Medical Imaging:** Measure tumor size, bone density
- **Real Estate:** Estimate property values from photos
- **Depth Estimation:** Predict distance/depth in scenes
- **Pose Estimation:** Predict joint angles and positions
- **Weather Prediction:** Estimate temperature, humidity from sky images
- **Manufacturing:** Measure defect sizes, tolerances

## Supported Models

Any AutoModelForImageClassification compatible model (configured for regression):
- **Vision Transformers:** ViT, DeiT, Swin, BEiT
- **ConvNets:** ResNet, EfficientNet, ConvNeXT, RegNet
- **Hybrid:** CvT, LeViT
- **Specialized:** DINOv2 (self-supervised features)

## Data Format

### Directory Structure
```
data/
├── train/
│   ├── image1.jpg  # filename or metadata contains target value
│   ├── image2.jpg
│   └── ...
└── validation/
    ├── image1.jpg
    └── ...
```

### CSV with Targets
```csv
image_path,target_value
images/sample1.jpg,25.5
images/sample2.jpg,67.3
images/sample3.jpg,12.8
```

### JSON Format
```json
{"image": "path/to/image1.jpg", "target": 25.5}
{"image": "path/to/image2.jpg", "target": 67.3}
```

### Hugging Face Dataset
```python
from datasets import Dataset

data = {
    "image": ["img1.jpg", "img2.jpg"],
    "label": [25.5, 67.3]  # continuous values
}
dataset = Dataset.from_dict(data)
```

## Parameters

### Required Parameters
- `model`: Pre-trained model name or path
- `data_path`: Path to training data
- `image_column`: Column containing image paths (default: "image")
- `target_column`: Column containing target values (default: "label")

### Training Parameters
- `lr`: Learning rate (default: 5e-5)
- `epochs`: Number of epochs (default: 10)
- `batch_size`: Batch size (default: 8)
- `warmup_ratio`: Warmup proportion (default: 0.1)
- `gradient_accumulation`: Gradient accumulation steps (default: 1)

### Image Parameters
- `image_size`: Input image size (default: 224)
- `augmentation`: Enable augmentation (default: True)
- `normalize`: Normalization parameters
- `preprocessing`: Custom preprocessing function

### Advanced Parameters
- `loss_function`: Loss function (mse, mae, huber)
- `metric`: Evaluation metric (rmse, mae, r2)
- `mixed_precision`: Enable mixed precision training
- `gradient_checkpointing`: Save memory during training

## Command Line Usage

### Basic Training
```bash
autotrain image-regression \
  --model google/vit-base-patch16-224 \
  --data-path ./age_estimation_data \
  --image-column image_path \
  --target-column age \
  --output-dir ./age_model \
  --train
```

### Advanced Configuration
```bash
autotrain image-regression \
  --model microsoft/swin-base-patch4-window7-224 \
  --data-path ./quality_scores.csv \
  --image-size 384 \
  --epochs 20 \
  --batch-size 16 \
  --lr 2e-5 \
  --loss-function huber \
  --metric rmse \
  --mixed-precision fp16 \
  --train
```

## Python API Usage

```python
from autotrain.trainers.image_regression.params import ImageRegressionParams
from autotrain.trainers.image_regression import train

# Configure parameters
params = ImageRegressionParams(
    model="google/vit-base-patch16-224",
    data_path="./regression_data",
    image_column="image",
    target_column="score",
    train_split="train",
    valid_split="validation",
    image_size=224,
    epochs=15,
    batch_size=32,
    lr=5e-5,
    loss_function="mse",
    augmentation=True,
    output_dir="./regression_model"
)

# Train model
train(params)
```

## Data Preparation

### Image Preprocessing
```python
from torchvision import transforms
from PIL import Image

# Standard preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Load and preprocess
image = Image.open("sample.jpg")
tensor = transform(image)
```

### Data Augmentation
```python
# Augmentation for regression (be careful with target consistency)
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

### Target Normalization
```python
import numpy as np

# Normalize targets for better training
targets = np.array([25, 67, 12, 89, 45])
mean = targets.mean()
std = targets.std()

normalized_targets = (targets - mean) / std

# Remember to denormalize predictions
predictions = model_output * std + mean
```

## Loss Functions

### Mean Squared Error (MSE)
- Default choice
- Penalizes large errors more
- Sensitive to outliers

### Mean Absolute Error (MAE)
- Robust to outliers
- Linear penalty
- Good for data with outliers

### Huber Loss
- Combination of MSE and MAE
- Robust to outliers
- Smooth gradient

## Evaluation Metrics

- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **R² Score:** Coefficient of determination
- **Pearson Correlation:** Linear relationship
- **Spearman Correlation:** Monotonic relationship

## Best Practices

1. **Target Scaling:** Normalize/standardize target values
2. **Data Quality:** Ensure consistent image quality
3. **Augmentation:** Use carefully - maintain target validity
4. **Model Selection:** ViT variants often work well
5. **Loss Function:** Choose based on data distribution
6. **Validation:** Use proper train/val/test splits

## Troubleshooting

### High Training Loss
- Check target value ranges
- Normalize targets
- Verify data loading
- Try different loss functions

### Overfitting
- Add more augmentation
- Use dropout
- Reduce model size
- Add regularization

### Poor Predictions
- Check if targets are normalized
- Verify image preprocessing
- Try different architectures
- Increase training data

## Example Projects

### Age Estimation System
```python
from autotrain.trainers.image_regression import train
from autotrain.trainers.image_regression.params import ImageRegressionParams
import pandas as pd

# Prepare data
df = pd.DataFrame({
    'image_path': ['faces/person1.jpg', 'faces/person2.jpg'],
    'age': [25, 45]
})
df.to_csv('age_data.csv', index=False)

# Configure training
params = ImageRegressionParams(
    model="microsoft/swin-tiny-patch4-window7-224",
    data_path="age_data.csv",
    image_column="image_path",
    target_column="age",
    epochs=20,
    batch_size=16,
    loss_function="huber",
    output_dir="./age_estimator"
)

# Train
model = train(params)

# Inference
from transformers import pipeline
estimator = pipeline("image-classification", model="./age_estimator")

# Predict age
result = estimator("new_face.jpg")
predicted_age = result[0]['score']  # Regression value
print(f"Estimated age: {predicted_age:.1f}")
```

### Quality Score Prediction
```python
# Train quality scorer for manufacturing
params = ImageRegressionParams(
    model="google/vit-base-patch16-224",
    data_path="./product_images",
    target_column="quality_score",  # 0-100 scale
    image_size=384,
    epochs=30,
    loss_function="mse",
    metric="rmse",
    output_dir="./quality_scorer"
)

model = train(params)
```

## Advanced Techniques

### Multi-Target Regression
```python
# Predict multiple values (e.g., x, y coordinates)
params = ImageRegressionParams(
    target_columns=["x_coord", "y_coord"],  # Multiple targets
    loss_function="mse",
    # ... other params
)
```

### Custom Loss Functions
```python
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, pred, target):
        return (self.weights * (pred - target) ** 2).mean()

# Use in training
params.custom_loss = WeightedMSELoss(weights=[1.0, 2.0])
```

## See Also

- [Image Classification](./ImageClassification.md) - For discrete categories
- [Object Detection](./ObjectDetection.md) - For localization tasks
- [VLM](./VLM.md) - For image-text tasks