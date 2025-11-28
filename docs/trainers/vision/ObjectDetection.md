# Object Detection Trainer

## Overview

The Object Detection trainer enables fine-tuning of transformer-based object detection models for identifying and localizing multiple objects within images. Unlike image classification which assigns a single label to an entire image, object detection identifies where objects are located by predicting bounding boxes and class labels for each detected object.

This trainer implements a complete pipeline for object detection training, including automatic data augmentation, support for various annotation formats, evaluation using standard COCO metrics (mAP, mAR), and seamless integration with Hugging Face Hub. The implementation leverages state-of-the-art models like DETR (DEtection TRansformer) and supports mixed precision training for efficiency.

## Use Cases

- **Autonomous Vehicles:** Detect pedestrians, vehicles, traffic signs, and obstacles
- **Retail Analytics:** Track products on shelves, monitor inventory, count items
- **Security and Surveillance:** Identify people, vehicles, or suspicious objects in video feeds
- **Manufacturing Quality Control:** Detect defects, missing components, or assembly errors
- **Medical Imaging:** Locate tumors, lesions, or anatomical structures in scans
- **Wildlife Monitoring:** Track and count animals in their natural habitats
- **Agricultural Monitoring:** Detect pests, diseases, or ripe fruits on plants
- **Sports Analytics:** Track players, balls, and equipment during games
- **Document Processing:** Locate and extract tables, figures, or text regions
- **Warehouse Automation:** Identify packages, pallets, and items for robotic handling
- **Traffic Monitoring:** Count vehicles, detect violations, monitor congestion
- **Drone Imagery Analysis:** Identify objects of interest in aerial photographs

## Supported Models

Any model compatible with AutoModelForObjectDetection from Hugging Face:

### DETR (DEtection TRansformer)
- `facebook/detr-resnet-50` (default)
- `facebook/detr-resnet-101`
- `facebook/detr-resnet-50-dc5`

### Conditional DETR
- `microsoft/conditional-detr-resnet-50`

### YOLOS (You Only Look at One Sequence)
- `hustvl/yolos-tiny`
- `hustvl/yolos-small`
- `hustvl/yolos-base`

### Table Transformer
- `microsoft/table-transformer-detection`

### Custom Models
Any custom object detection model from Hugging Face Hub that supports AutoModelForObjectDetection

## Data Format and Structure

### COCO Format (Recommended)

The trainer expects datasets with bounding boxes in COCO format: `[x, y, width, height]` where:
- `x`: Left coordinate of the bounding box
- `y`: Top coordinate of the bounding box
- `width`: Width of the bounding box
- `height`: Height of the bounding box

**Hugging Face Dataset Structure:**
```python
from datasets import Dataset, Features, Sequence, Value, ClassLabel, Image

features = Features({
    'image': Image(),
    'objects': {
        'bbox': Sequence(Sequence(Value('float32'))),  # [[x, y, w, h], ...]
        'category': Sequence(ClassLabel(names=['person', 'car', 'dog']))
    }
})

dataset = Dataset.from_dict({
    'image': ['path/to/image1.jpg'],
    'objects': {
        'bbox': [[[100, 50, 200, 150], [300, 100, 100, 80]]],  # Two objects
        'category': [[0, 1]]  # person, car
    }
}, features=features)
```

### JSON Format

For custom datasets, you can structure data as JSON:
```json
{
  "image": "path/to/image.jpg",
  "objects": {
    "bbox": [[50, 50, 100, 100], [200, 150, 150, 200]],
    "category": [0, 1]
  }
}
```

### Directory Structure for Custom Loading

```
data/
  images/
    image1.jpg
    image2.jpg
  annotations/
    image1.json
    image2.json
```

**Annotation JSON Example:**
```json
{
  "bbox": [[x1, y1, w1, h1], [x2, y2, w2, h2]],
  "category": [0, 1],
  "category_names": ["cat", "dog"]
}
```

### Converting from Other Formats

#### From Pascal VOC to COCO
```python
def voc_to_coco(xmin, ymin, xmax, ymax):
    """Convert Pascal VOC [xmin, ymin, xmax, ymax] to COCO [x, y, w, h]"""
    x = xmin
    y = ymin
    w = xmax - xmin
    h = ymax - ymin
    return [x, y, w, h]
```

#### From YOLO to COCO
```python
def yolo_to_coco(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO normalized [x_center, y_center, w, h] to COCO [x, y, w, h]"""
    x = (x_center - width / 2) * img_width
    y = (y_center - height / 2) * img_height
    w = width * img_width
    h = height * img_height
    return [x, y, w, h]
```

## Parameters

### Required Parameters

- `data_path` (str): Path to the dataset (Hugging Face dataset name or local directory)
- `model` (str): Pre-trained model name or path (default: "facebook/detr-resnet-50")
- `project_name` (str): Output directory name (default: "project-name")

### Data Parameters

- `image_column` (str): Column name for images (default: "image")
- `objects_column` (str): Column name for objects/annotations (default: "objects")
- `train_split` (str): Training data split name (default: "train")
- `valid_split` (Optional[str]): Validation data split name (default: None)
- `max_samples` (Optional[int]): Maximum samples to use for testing/debugging (default: None)
- `image_square_size` (Optional[int]): Resize longest edge then pad to square (default: 600)

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

### Basic Object Detection Training

```bash
autotrain object-detection \
  --model facebook/detr-resnet-50 \
  --data-path coco-dataset \
  --project-name coco-detector \
  --image-column image \
  --objects-column objects \
  --train-split train \
  --valid-split validation \
  --epochs 10 \
  --batch-size 4 \
  --lr 1e-5 \
  --image-square-size 800 \
  --train
```

### Training with Mixed Precision and Early Stopping

```bash
autotrain object-detection \
  --model facebook/detr-resnet-101 \
  --data-path ./custom_detection_data \
  --project-name vehicle-detector \
  --image-column image \
  --objects-column annotations \
  --train-split train \
  --valid-split test \
  --epochs 20 \
  --batch-size 2 \
  --lr 5e-6 \
  --gradient-accumulation 8 \
  --mixed-precision fp16 \
  --image-square-size 1024 \
  --early-stopping-patience 3 \
  --scheduler cosine \
  --weight-decay 0.0001 \
  --train
```

### Training with Hub Upload

```bash
autotrain object-detection \
  --model facebook/detr-resnet-50 \
  --data-path my-org/product-detection-dataset \
  --project-name product-detector-v1 \
  --epochs 15 \
  --batch-size 4 \
  --lr 1e-5 \
  --image-square-size 640 \
  --push-to-hub \
  --username your-username \
  --token $HF_TOKEN \
  --log wandb \
  --train
```

## Python API Usage

### Basic Training Script

```python
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.object_detection import train

# Configure training parameters
params = ObjectDetectionParams(
    model="facebook/detr-resnet-50",
    data_path="detection-datasets/coco-sample",
    project_name="object-detector",
    image_column="image",
    objects_column="objects",
    train_split="train",
    valid_split="validation",
    lr=1e-5,
    epochs=10,
    batch_size=4,
    warmup_ratio=0.1,
    gradient_accumulation=4,
    optimizer="adamw_torch",
    scheduler="linear",
    weight_decay=0.0001,
    max_grad_norm=1.0,
    seed=42,
    image_square_size=800,
    mixed_precision="fp16",
    auto_find_batch_size=False,
    save_total_limit=2,
    early_stopping_patience=5,
    early_stopping_threshold=0.01,
    push_to_hub=False
)

# Start training
train(params)
```

### Training with Custom Dataset

```python
from datasets import Dataset, Features, Sequence, Value, ClassLabel, Image
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.object_detection import train

# Define your dataset structure
features = Features({
    'image': Image(),
    'objects': {
        'bbox': Sequence(Sequence(Value('float32'))),
        'category': Sequence(ClassLabel(names=['car', 'person', 'bicycle']))
    }
})

# Create dataset from your annotations
train_data = Dataset.from_dict({
    'image': ['path/to/img1.jpg', 'path/to/img2.jpg'],
    'objects': {
        'bbox': [
            [[100, 100, 200, 200], [300, 150, 100, 150]],  # Image 1: 2 objects
            [[50, 50, 150, 200]]  # Image 2: 1 object
        ],
        'category': [
            [0, 1],  # car, person
            [2]  # bicycle
        ]
    }
}, features=features)

# Save dataset
from datasets import DatasetDict
dataset_dict = DatasetDict({
    'train': train_data,
    'validation': val_data
})
dataset_dict.save_to_disk('./my_detection_data')

# Configure and train
params = ObjectDetectionParams(
    model="facebook/detr-resnet-50",
    data_path="./my_detection_data",
    project_name="custom-detector",
    train_split="train",
    valid_split="validation",
    epochs=15,
    batch_size=2,
    lr=1e-5,
    image_square_size=640
)

train(params)
```

### Advanced Configuration with Monitoring

```python
import os
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.object_detection import train

# Set up experiment tracking
os.environ["WANDB_PROJECT"] = "object-detection"
os.environ["WANDB_ENTITY"] = "my-team"

params = ObjectDetectionParams(
    model="facebook/detr-resnet-101",
    data_path="surveillance-dataset",
    project_name="security-camera-detector",
    train_split="train",
    valid_split="validation",
    lr=5e-6,
    epochs=30,
    batch_size=2,
    warmup_ratio=0.05,
    gradient_accumulation=8,  # Effective batch size = 16
    optimizer="adamw_torch",
    scheduler="cosine",
    weight_decay=0.0001,
    mixed_precision="fp16",
    image_square_size=1024,
    logging_steps=25,
    eval_strategy="epoch",
    save_total_limit=3,
    early_stopping_patience=5,
    log="wandb",
    push_to_hub=True,
    username="my-username",
    token=os.environ["HF_TOKEN"]
)

train(params)
```

## Data Preparation and Augmentation

### Built-in Augmentation Pipeline

The trainer automatically applies robust augmentations optimized for object detection:

**Training Augmentations:**
- Random resized crop with bounding box safety (20% probability)
- Blur variants: Gaussian blur, motion blur, or defocus (10% probability)
- Perspective transformation (10% probability)
- Horizontal flip (50% probability)
- Random brightness and contrast adjustment (50% probability)
- Hue/Saturation/Value adjustment (10% probability)
- Resize longest edge and pad to square
- Bounding box clipping and minimum area filtering (25 pixels)

**Validation Augmentations:**
- Resize longest edge to target size
- Pad to square
- Bounding box clipping only (no random augmentations)

These are implemented using Albumentations with proper bounding box handling:

```python
train_transforms = A.Compose([
    A.Compose([
        A.SmallestMaxSize(max_size=max_size, p=1.0),
        A.RandomSizedBBoxSafeCrop(height=max_size, width=max_size, p=1.0),
    ], p=0.2),
    A.OneOf([
        A.Blur(blur_limit=7, p=0.5),
        A.MotionBlur(blur_limit=7, p=0.5),
        A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
    ], p=0.1),
    A.Perspective(p=0.1),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.1),
    A.LongestMaxSize(max_size=max_size),
    A.PadIfNeeded(max_size, max_size, border_mode=0, value=(128, 128, 128)),
], bbox_params=A.BboxParams(format='coco', label_fields=['category'], clip=True, min_area=25))
```

### Preparing Annotations

#### From COCO JSON

```python
import json
from datasets import Dataset
from PIL import Image

def load_coco_dataset(images_dir, annotations_file):
    """Load COCO format annotations into Hugging Face Dataset"""
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # Build category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_names = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: x['id'])]

    # Build image ID to annotations mapping
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # Create dataset
    data = []
    for img_info in coco_data['images']:
        img_id = img_info['id']
        anns = img_to_anns.get(img_id, [])

        bboxes = [ann['bbox'] for ann in anns]  # Already in COCO format [x, y, w, h]
        categories_list = [ann['category_id'] for ann in anns]

        data.append({
            'image': f"{images_dir}/{img_info['file_name']}",
            'objects': {
                'bbox': bboxes,
                'category': categories_list
            }
        })

    return Dataset.from_list(data)

# Usage
train_dataset = load_coco_dataset('./images/train', './annotations/train.json')
val_dataset = load_coco_dataset('./images/val', './annotations/val.json')
```

#### From Pascal VOC XML

```python
import xml.etree.ElementTree as ET
from datasets import Dataset
import os

def parse_voc_xml(xml_path):
    """Parse Pascal VOC XML annotation"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bboxes = []
    categories = []

    for obj in root.findall('object'):
        category = obj.find('name').text
        bbox_elem = obj.find('bndbox')

        xmin = float(bbox_elem.find('xmin').text)
        ymin = float(bbox_elem.find('ymin').text)
        xmax = float(bbox_elem.find('xmax').text)
        ymax = float(bbox_elem.find('ymax').text)

        # Convert VOC [xmin, ymin, xmax, ymax] to COCO [x, y, w, h]
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        bboxes.append([x, y, w, h])
        categories.append(category)

    return bboxes, categories

def load_voc_dataset(images_dir, annotations_dir, category_names):
    """Load Pascal VOC dataset"""
    data = []
    category_to_id = {name: idx for idx, name in enumerate(category_names)}

    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(annotations_dir, xml_file)
        img_name = xml_file.replace('.xml', '.jpg')
        img_path = os.path.join(images_dir, img_name)

        bboxes, categories = parse_voc_xml(xml_path)
        category_ids = [category_to_id[cat] for cat in categories]

        data.append({
            'image': img_path,
            'objects': {
                'bbox': bboxes,
                'category': category_ids
            }
        })

    return Dataset.from_list(data)

# Usage
category_names = ['person', 'car', 'dog', 'cat']
dataset = load_voc_dataset('./images', './annotations', category_names)
```

### Handling Small Objects

For datasets with many small objects, increase the `image_square_size`:

```python
params = ObjectDetectionParams(
    model="facebook/detr-resnet-50",
    data_path="small-objects-dataset",
    image_square_size=1024,  # Larger size for small objects
    # ... other params
)
```

### Balancing Object Classes

The trainer samples evenly when using `max_samples` for debugging:

```python
params = ObjectDetectionParams(
    model="facebook/detr-resnet-50",
    data_path="imbalanced-dataset",
    max_samples=500,  # Will sample evenly across dataset
    # ... other params
)
```

## Best Practices

### 1. Start with Pre-trained COCO Models

Always use models pre-trained on COCO or similar detection datasets:
```python
params.model = "facebook/detr-resnet-50"  # Pre-trained on COCO
```

### 2. Choose Appropriate Image Size

Balance between object detail and memory usage:
- **Small objects:** `image_square_size=1024` or higher
- **Medium objects:** `image_square_size=800` (default: 600)
- **Large objects:** `image_square_size=512` or `640`

### 3. Adjust Batch Size and Gradient Accumulation

Object detection is memory-intensive:
```python
params.batch_size = 2  # Small batch size
params.gradient_accumulation = 8  # Effective batch size = 16
```

### 4. Use Appropriate Learning Rates

Object detection typically requires lower learning rates:
- Start: 1e-5 to 5e-6
- DETR models: 1e-5
- Fine-tuning: 5e-6 to 1e-6

### 5. Monitor mAP Metrics

Always use validation split to track mean Average Precision:
```python
params = ObjectDetectionParams(
    # ... other params
    valid_split="validation",
    eval_strategy="epoch"
)
```

### 6. Train for More Epochs

Object detection usually needs more epochs than classification:
```python
params.epochs = 15  # or more for complex datasets
params.early_stopping_patience = 5
```

### 7. Ensure Quality Annotations

- Tight bounding boxes around objects
- Consistent annotation style
- No missing annotations
- Correct category labels
- Remove occluded or ambiguous objects if needed

## Evaluation Metrics

The trainer computes comprehensive COCO-style detection metrics:

### Primary Metrics

- **mAP (mean Average Precision):** Overall detection performance across all IoU thresholds (0.5 to 0.95)
- **mAP@0.5:** Average Precision at IoU threshold of 0.5 (more lenient)
- **mAP@0.75:** Average Precision at IoU threshold of 0.75 (stricter)

### Size-based Metrics

- **mAP_small:** Performance on small objects (area < 32^2 pixels)
- **mAP_medium:** Performance on medium objects (32^2 < area < 96^2)
- **mAP_large:** Performance on large objects (area > 96^2 pixels)

### Recall Metrics (mAR)

- **mAR@1:** Maximum recall with 1 detection per image
- **mAR@10:** Maximum recall with 10 detections per image
- **mAR@100:** Maximum recall with 100 detections per image
- **mAR_small, mAR_medium, mAR_large:** Recall by object size

### Per-Class Metrics

When validation split is provided, the trainer also computes per-class mAP and mAR:
```python
# Metrics are saved in the model card
{
    'map': 0.45,
    'map_50': 0.68,
    'map_75': 0.48,
    'map_person': 0.52,
    'map_car': 0.61,
    'mar_100_person': 0.58,
    'mar_100_car': 0.65
}
```

## Troubleshooting Common Issues

### Out of Memory Errors

**Symptoms:** CUDA out of memory during training

**Solutions:**
1. Reduce batch size:
   ```python
   params.batch_size = 1  # Minimum
   ```

2. Enable gradient accumulation:
   ```python
   params.batch_size = 1
   params.gradient_accumulation = 16  # Effective batch size = 16
   ```

3. Reduce image size:
   ```python
   params.image_square_size = 512  # Smaller images
   ```

4. Use mixed precision:
   ```python
   params.mixed_precision = "fp16"
   ```

5. Use a smaller model:
   ```python
   params.model = "hustvl/yolos-tiny"
   ```

### Poor Detection Performance

**Symptoms:** Low mAP, missed detections, poor localization

**Solutions:**
1. Train for more epochs:
   ```python
   params.epochs = 20  # More training
   ```

2. Lower learning rate:
   ```python
   params.lr = 5e-6  # Instead of 1e-5
   ```

3. Increase image size:
   ```python
   params.image_square_size = 1024  # For small objects
   ```

4. Check annotation quality:
   - Verify bounding boxes are accurate
   - Ensure consistent labeling
   - Check for missing annotations

5. Add weight decay:
   ```python
   params.weight_decay = 0.0001
   ```

6. Use cosine scheduler:
   ```python
   params.scheduler = "cosine"
   ```

### Model Not Converging

**Symptoms:** Loss stays high, mAP near zero

**Solutions:**
1. Verify data format:
   ```python
   # Check first sample
   from datasets import load_from_disk
   dataset = load_from_disk(params.data_path)
   sample = dataset['train'][0]
   print(sample)
   # Verify 'objects' contains 'bbox' and 'category'
   ```

2. Check category IDs are correct:
   ```python
   # Category IDs should start from 0
   all_categories = []
   for sample in dataset['train']:
       all_categories.extend(sample['objects']['category'])
   print(f"Category range: {min(all_categories)} to {max(all_categories)}")
   ```

3. Verify bounding box format:
   ```python
   # Should be [x, y, width, height] in COCO format
   sample_bbox = dataset['train'][0]['objects']['bbox'][0]
   print(f"Bbox format: {sample_bbox}")
   ```

4. Increase learning rate slightly:
   ```python
   params.lr = 2e-5  # Try slightly higher
   ```

### Training is Too Slow

**Symptoms:** Very slow iterations, long epoch times

**Solutions:**
1. Enable mixed precision:
   ```python
   params.mixed_precision = "fp16"
   ```

2. Increase batch size if memory allows:
   ```python
   params.batch_size = 4  # Instead of 2
   ```

3. Reduce logging frequency:
   ```python
   params.logging_steps = 100
   ```

4. Reduce validation frequency:
   ```python
   params.eval_strategy = "epoch"  # Not "steps"
   ```

5. Use smaller images:
   ```python
   params.image_square_size = 640  # Instead of 1024
   ```

### Bounding Box Format Errors

**Symptoms:** "Invalid bbox" errors, training crashes

**Solutions:**
1. Verify COCO format [x, y, w, h]:
   ```python
   # Correct: [x, y, width, height]
   bbox = [100, 50, 200, 150]

   # If you have VOC format [xmin, ymin, xmax, ymax], convert:
   def voc_to_coco(xmin, ymin, xmax, ymax):
       return [xmin, ymin, xmax - xmin, ymax - ymin]
   ```

2. Ensure coordinates are not normalized:
   ```python
   # Bounding boxes should be in absolute pixels, not normalized [0, 1]
   # If normalized, multiply by image dimensions
   ```

3. Check for negative or zero dimensions:
   ```python
   # Width and height must be positive
   for bbox in bboxes:
       assert bbox[2] > 0 and bbox[3] > 0, f"Invalid bbox: {bbox}"
   ```

### No Validation Metrics

**Symptoms:** Validation metrics not appearing

**Solutions:**
1. Ensure valid_split is set:
   ```python
   params.valid_split = "validation"
   ```

2. Install pycocotools or faster-coco-eval:
   ```bash
   pip install pycocotools
   # or for numpy 2.x compatibility:
   pip install faster-coco-eval
   ```

3. Verify validation data exists:
   ```python
   from datasets import load_from_disk
   dataset = load_from_disk(params.data_path)
   print(dataset.keys())  # Should include validation split
   ```

## Example Projects

### Vehicle Detection System

```python
from datasets import load_dataset
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.object_detection import train

# 1. Load pre-annotated dataset
dataset = load_dataset("keremberke/vehicle-detection")
dataset.save_to_disk("./vehicle_data")

# 2. Configure training
params = ObjectDetectionParams(
    model="facebook/detr-resnet-50",
    data_path="./vehicle_data",
    project_name="vehicle-detector",
    image_column="image",
    objects_column="objects",
    train_split="train",
    valid_split="validation",
    lr=1e-5,
    epochs=15,
    batch_size=2,
    gradient_accumulation=8,
    image_square_size=800,
    mixed_precision="fp16",
    early_stopping_patience=5,
    optimizer="adamw_torch",
    scheduler="cosine",
    weight_decay=0.0001
)

# 3. Train model
train(params)

# 4. Use the trained model
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image, ImageDraw
import torch

processor = AutoImageProcessor.from_pretrained("./vehicle-detector")
model = AutoModelForObjectDetection.from_pretrained("./vehicle-detector")

image = Image.open("test_image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=target_sizes
)[0]

# Visualize detections
draw = ImageDraw.Draw(image)
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    label_name = model.config.id2label[label.item()]
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1]), f"{label_name}: {round(score.item(), 3)}", fill="red")

image.save("result.jpg")
```

### Custom Product Detection

```python
from datasets import Dataset, Features, Sequence, Value, ClassLabel, Image
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.object_detection import train
import json

# 1. Load your annotations
def load_custom_annotations(images_dir, annotations_file):
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    dataset_dict = {
        'image': [],
        'objects': {
            'bbox': [],
            'category': []
        }
    }

    for item in data:
        dataset_dict['image'].append(f"{images_dir}/{item['filename']}")
        dataset_dict['objects']['bbox'].append(item['bboxes'])
        dataset_dict['objects']['category'].append(item['categories'])

    features = Features({
        'image': Image(),
        'objects': {
            'bbox': Sequence(Sequence(Value('float32'))),
            'category': Sequence(ClassLabel(names=['product_a', 'product_b', 'product_c']))
        }
    })

    return Dataset.from_dict(dataset_dict, features=features)

train_data = load_custom_annotations('./images/train', './annotations/train.json')
val_data = load_custom_annotations('./images/val', './annotations/val.json')

from datasets import DatasetDict
dataset_dict = DatasetDict({'train': train_data, 'validation': val_data})
dataset_dict.save_to_disk('./product_detection_data')

# 2. Configure and train
params = ObjectDetectionParams(
    model="facebook/detr-resnet-50",
    data_path="./product_detection_data",
    project_name="product-detector",
    train_split="train",
    valid_split="validation",
    lr=1e-5,
    epochs=20,
    batch_size=4,
    gradient_accumulation=4,
    image_square_size=640,
    mixed_precision="fp16",
    push_to_hub=True,
    username="your-username",
    token="your-hf-token"
)

train(params)
```

## Integration with Hugging Face Hub

### Pushing Model to Hub

```python
params = ObjectDetectionParams(
    model="facebook/detr-resnet-50",
    data_path="./my_detection_data",
    project_name="my-object-detector",
    push_to_hub=True,
    username="my-hf-username",
    token="hf_your_token_here",
    # ... other params
)

train(params)
# Model uploaded to: huggingface.co/my-hf-username/my-object-detector
```

### Using Trained Model for Inference

```python
from transformers import pipeline
from PIL import Image

# Load from Hub
detector = pipeline(
    "object-detection",
    model="my-hf-username/my-object-detector"
)

# Detect objects
image = Image.open("test_image.jpg")
results = detector(image, threshold=0.5)

for result in results:
    print(f"Detected {result['label']} with confidence {result['score']:.3f}")
    print(f"  Location: {result['box']}")
```

## See Also

- [Image Classification](./ImageClassification.md) - For classifying entire images
- [VLM (Vision-Language Models)](./VLM.md) - For VQA and image captioning tasks
- [Image Segmentation](../vision/ImageSegmentation.md) - For pixel-level object segmentation
