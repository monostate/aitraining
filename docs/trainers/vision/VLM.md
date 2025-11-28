# Vision-Language Model (VLM) Trainer

## Overview

The Vision-Language Model (VLM) trainer enables fine-tuning of multimodal models that combine computer vision and natural language processing. These models can understand and generate text based on visual inputs, making them suitable for tasks like Visual Question Answering (VQA), image captioning, and multimodal understanding.

This trainer focuses on fine-tuning state-of-the-art VLMs like PaliGemma for custom vision-language tasks. It supports PEFT (Parameter-Efficient Fine-Tuning) with LoRA, quantization for efficient training, and seamless integration with Hugging Face Hub. The implementation handles both VQA (where the model answers questions about images) and captioning (where the model generates descriptions of images).

## Use Cases

### Visual Question Answering (VQA)
- **Customer Support:** Answer questions about product images or diagrams
- **Medical Diagnosis:** Answer clinical questions based on medical imagery
- **Document Understanding:** Extract information from charts, graphs, and forms
- **Educational Tools:** Interactive learning with visual materials
- **Accessibility:** Describe visual content for visually impaired users
- **E-commerce:** Answer product-related questions from images
- **Scientific Research:** Query visual data from experiments or observations

### Image Captioning
- **Content Generation:** Automatically describe images for alt text or metadata
- **Social Media:** Generate captions for posts and stories
- **Asset Management:** Catalog and search image databases
- **News and Journalism:** Quick descriptions for photo archives
- **Surveillance:** Describe security footage events
- **Accessibility:** Generate descriptions for screen readers

### Multimodal Understanding
- **Scene Understanding:** Comprehend complex visual scenes and relationships
- **Visual Reasoning:** Answer questions requiring multi-step visual reasoning
- **Document AI:** Understand and extract information from documents with text and images
- **Retail:** Product identification and attribute extraction
- **Healthcare:** Analyze medical reports combining images and text

## Supported Models

Currently, the trainer primarily supports PaliGemma models:

### PaliGemma (Recommended)
- `google/paligemma-3b-pt-224` (default) - Pre-trained, 224x224 resolution
- `google/paligemma-3b-pt-448` - Pre-trained, 448x448 resolution
- `google/paligemma-3b-pt-896` - Pre-trained, 896x896 resolution
- `google/paligemma-3b-mix-224` - Mix of tasks, 224x224
- `google/paligemma-3b-mix-448` - Mix of tasks, 448x448

**Model Characteristics:**
- Based on SigLIP vision encoder and Gemma language model
- Supports multiple vision-language tasks
- Efficient with PEFT/LoRA fine-tuning
- Supports quantization (int4, int8)

### Future Support
The architecture is designed to support additional models:
- Florence-2 models (planned)
- LLaVA models (planned)
- Custom VLM models from Hugging Face Hub

## Data Format and Structure

### VQA (Visual Question Answering) Format

The trainer expects datasets with three columns:
- **image:** PIL Image or path to image file
- **prompt:** The question or instruction (will be prefixed with "answer ")
- **text:** The expected answer

**Dataset Structure:**
```python
from datasets import Dataset, Features, Image, Value

features = Features({
    'image': Image(),
    'prompt': Value('string'),  # Question
    'text': Value('string')     # Answer
})

dataset = Dataset.from_dict({
    'image': ['path/to/image1.jpg', 'path/to/image2.jpg'],
    'prompt': ['What color is the car?', 'How many people are visible?'],
    'text': ['red', '3']
}, features=features)
```

### Captioning Format

For image captioning, you can either:
1. Omit the `prompt` column (uses default "describe" prompt)
2. Provide a simple prompt like "caption" or "describe"

**Dataset Structure:**
```python
from datasets import Dataset, Features, Image, Value

features = Features({
    'image': Image(),
    'text': Value('string')  # Caption
})

dataset = Dataset.from_dict({
    'image': ['path/to/image1.jpg', 'path/to/image2.jpg'],
    'text': [
        'A red car parked on the street',
        'Three people walking in a park'
    ]
}, features=features)
```

### CSV Format Example

**VQA CSV:**
```csv
image,prompt,text
/path/to/img1.jpg,What is in this image?,A cat sitting on a couch
/path/to/img2.jpg,What color is the shirt?,blue
/path/to/img3.jpg,How many items are shown?,5
```

**Captioning CSV:**
```csv
image,text
/path/to/img1.jpg,A beautiful sunset over the ocean
/path/to/img2.jpg,A group of friends having dinner
```

### JSON Lines Format

```json
{"image": "/path/to/img1.jpg", "prompt": "What is this?", "text": "A dog"}
{"image": "/path/to/img2.jpg", "prompt": "Describe the scene", "text": "A beach at sunset"}
```

## Parameters

### Required Parameters

- `model` (str): Pre-trained model name or path (default: "google/paligemma-3b-pt-224")
- `data_path` (str): Path to the dataset (default: "data")
- `project_name` (str): Output directory name (default: "project-name")

### Data Parameters

- `image_column` (Optional[str]): Column name for images (default: "image")
- `text_column` (str): Column name for text/answers (default: "text")
- `prompt_text_column` (Optional[str]): Column name for prompts/questions (default: "prompt")
- `train_split` (str): Training data split name (default: "train")
- `valid_split` (Optional[str]): Validation data split name (default: None)
- `max_samples` (Optional[int]): Maximum samples to use for testing/debugging (default: None)

### Training Parameters

- `trainer` (str): Trainer type - "vqa" or "captioning" (default: "vqa")
- `lr` (float): Learning rate (default: 5e-5)
- `epochs` (int): Number of training epochs (default: 3)
- `batch_size` (int): Training batch size (default: 2)
- `warmup_ratio` (float): Warmup proportion for learning rate scheduler (default: 0.1)
- `gradient_accumulation` (int): Gradient accumulation steps (default: 4)
- `optimizer` (str): Optimizer type (default: "adamw_torch")
- `scheduler` (str): Learning rate scheduler (default: "linear")
- `weight_decay` (float): Weight decay for optimizer (default: 0.0)
- `max_grad_norm` (float): Maximum gradient norm for clipping (default: 1.0)
- `seed` (int): Random seed for reproducibility (default: 42)

### PEFT/LoRA Parameters

- `peft` (bool): Use PEFT (LoRA) for efficient fine-tuning (default: False)
- `quantization` (Optional[str]): Quantization type - "int4", "int8", or None (default: "int4")
- `lora_r` (int): LoRA rank (default: 16)
- `lora_alpha` (int): LoRA alpha parameter (default: 32)
- `lora_dropout` (float): LoRA dropout rate (default: 0.05)
- `target_modules` (Optional[str]): Target modules for LoRA (default: "all-linear")
- `merge_adapter` (bool): Merge adapter weights after training (default: False)

### Advanced Parameters

- `mixed_precision` (Optional[str]): Mixed precision mode - "fp16", "bf16", or None (default: None)
- `disable_gradient_checkpointing` (bool): Disable gradient checkpointing (default: False)
- `auto_find_batch_size` (bool): Automatically find optimal batch size (default: False)
- `save_total_limit` (int): Maximum checkpoints to keep (default: 1)
- `logging_steps` (int): Steps between logging (-1 for auto) (default: -1)
- `eval_strategy` (str): Evaluation strategy during training (default: "epoch")

### Hub Parameters

- `push_to_hub` (bool): Push model to Hugging Face Hub (default: False)
- `username` (Optional[str]): Hugging Face username for pushing models
- `token` (Optional[str]): Hugging Face Hub authentication token
- `log` (str): Experiment tracking method - "none", "wandb", "tensorboard" (default: "none")

## Command Line Usage

### Basic VQA Training

```bash
autotrain vlm \
  --model google/paligemma-3b-pt-224 \
  --data-path ./vqa_data \
  --project-name vqa-model \
  --trainer vqa \
  --image-column image \
  --prompt-text-column prompt \
  --text-column text \
  --train-split train \
  --valid-split validation \
  --epochs 5 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --lr 2e-5 \
  --peft \
  --quantization int4 \
  --train
```

### Image Captioning Training

```bash
autotrain vlm \
  --model google/paligemma-3b-pt-448 \
  --data-path ./caption_data \
  --project-name caption-model \
  --trainer captioning \
  --image-column image \
  --text-column text \
  --train-split train \
  --valid-split validation \
  --epochs 3 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --lr 1e-5 \
  --peft \
  --lora-r 32 \
  --lora-alpha 64 \
  --train
```

### Training with Hub Upload

```bash
autotrain vlm \
  --model google/paligemma-3b-pt-224 \
  --data-path medical-vqa-dataset \
  --project-name medical-vqa-assistant \
  --trainer vqa \
  --epochs 10 \
  --batch-size 2 \
  --gradient-accumulation 16 \
  --lr 5e-6 \
  --peft \
  --quantization int4 \
  --push-to-hub \
  --username your-username \
  --token $HF_TOKEN \
  --log wandb \
  --train
```

## Python API Usage

### Basic VQA Training Script

```python
from autotrain.trainers.vlm.params import VLMTrainingParams
from autotrain.trainers.vlm import train

# Configure VQA training parameters
params = VLMTrainingParams(
    model="google/paligemma-3b-pt-224",
    data_path="vqa-dataset",
    project_name="custom-vqa-model",
    trainer="vqa",
    image_column="image",
    prompt_text_column="prompt",
    text_column="text",
    train_split="train",
    valid_split="validation",
    lr=2e-5,
    epochs=5,
    batch_size=2,
    warmup_ratio=0.1,
    gradient_accumulation=8,
    optimizer="adamw_torch",
    scheduler="linear",
    weight_decay=0.0,
    max_grad_norm=1.0,
    seed=42,
    peft=True,
    quantization="int4",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    merge_adapter=False,
    mixed_precision="fp16",
    disable_gradient_checkpointing=False,
    auto_find_batch_size=False,
    save_total_limit=2,
    push_to_hub=False
)

# Start training
train(params)
```

### Image Captioning Training

```python
from autotrain.trainers.vlm.params import VLMTrainingParams
from autotrain.trainers.vlm import train

# Configure captioning parameters
params = VLMTrainingParams(
    model="google/paligemma-3b-pt-448",  # Higher resolution
    data_path="./caption_data",
    project_name="image-captioner",
    trainer="captioning",
    image_column="image",
    text_column="caption",
    prompt_text_column=None,  # Not needed for captioning
    train_split="train",
    valid_split="test",
    lr=1e-5,
    epochs=3,
    batch_size=4,
    gradient_accumulation=4,
    peft=True,
    quantization="int4",
    lora_r=32,
    lora_alpha=64,
    mixed_precision="bf16"
)

train(params)
```

### Training with Custom Dataset

```python
from datasets import Dataset, Features, Image as HFImage, Value
from autotrain.trainers.vlm.params import VLMTrainingParams
from autotrain.trainers.vlm import train
import pandas as pd

# 1. Prepare your dataset
df = pd.read_csv('vqa_data.csv')

# Create dataset
features = Features({
    'image': HFImage(),
    'prompt': Value('string'),
    'text': Value('string')
})

dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column('image', HFImage())

# Split and save
from datasets import DatasetDict
dataset = dataset.train_test_split(test_size=0.1, seed=42)
dataset_dict = DatasetDict({
    'train': dataset['train'],
    'validation': dataset['test']
})
dataset_dict.save_to_disk('./processed_vqa_data')

# 2. Configure and train
params = VLMTrainingParams(
    model="google/paligemma-3b-pt-224",
    data_path="./processed_vqa_data",
    project_name="my-vqa-model",
    trainer="vqa",
    train_split="train",
    valid_split="validation",
    epochs=8,
    batch_size=2,
    gradient_accumulation=8,
    lr=2e-5,
    peft=True,
    quantization="int4"
)

train(params)
```

### Advanced Configuration with Full Fine-tuning

```python
from autotrain.trainers.vlm.params import VLMTrainingParams
from autotrain.trainers.vlm import train
import os

# Full fine-tuning (no PEFT) - requires more GPU memory
params = VLMTrainingParams(
    model="google/paligemma-3b-pt-224",
    data_path="large-vqa-dataset",
    project_name="full-finetuned-vqa",
    trainer="vqa",
    train_split="train",
    valid_split="validation",
    lr=1e-5,
    epochs=3,
    batch_size=1,
    gradient_accumulation=32,
    optimizer="adamw_torch",
    scheduler="cosine",
    weight_decay=0.01,
    peft=False,  # Full fine-tuning
    quantization=None,  # No quantization
    mixed_precision="bf16",
    disable_gradient_checkpointing=False,
    logging_steps=50,
    eval_strategy="steps",
    save_total_limit=3,
    log="wandb",
    push_to_hub=True,
    username="my-username",
    token=os.environ["HF_TOKEN"]
)

train(params)
```

## Data Preparation and Augmentation

### Preparing VQA Dataset

```python
from datasets import Dataset, Features, Image, Value
import pandas as pd

# Load your VQA data
data = {
    'image': [],
    'prompt': [],
    'text': []
}

# Example: Load from CSV or other source
df = pd.read_csv('vqa_annotations.csv')
for _, row in df.iterrows():
    data['image'].append(row['image_path'])
    data['prompt'].append(row['question'])
    data['text'].append(row['answer'])

# Create Hugging Face dataset
features = Features({
    'image': Image(),
    'prompt': Value('string'),
    'text': Value('string')
})

dataset = Dataset.from_dict(data, features=features)

# Split into train/validation
dataset = dataset.train_test_split(test_size=0.1, seed=42)

from datasets import DatasetDict
dataset_dict = DatasetDict({
    'train': dataset['train'],
    'validation': dataset['test']
})

dataset_dict.save_to_disk('./vqa_dataset')
```

### Preparing Captioning Dataset

```python
from datasets import Dataset, Features, Image, Value
import os
from PIL import Image as PILImage

# Collect images and captions
images = []
captions = []

for img_file in os.listdir('./images'):
    if img_file.endswith(('.jpg', '.png')):
        img_path = os.path.join('./images', img_file)

        # Get corresponding caption
        caption_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        with open(os.path.join('./captions', caption_file), 'r') as f:
            caption = f.read().strip()

        images.append(img_path)
        captions.append(caption)

# Create dataset
features = Features({
    'image': Image(),
    'text': Value('string')
})

dataset = Dataset.from_dict({
    'image': images,
    'text': captions
}, features=features)

dataset.save_to_disk('./caption_dataset')
```

### Data Augmentation Considerations

VLM models are typically robust to image variations, but consider:

1. **Question Paraphrasing:** Create variations of questions for the same image
   ```python
   variations = [
       "What color is the car?",
       "What is the color of the car?",
       "Identify the car's color"
   ]
   ```

2. **Answer Standardization:** Normalize answers for consistency
   ```python
   def normalize_answer(text):
       return text.lower().strip()
   ```

3. **Image Quality:** Ensure images are clear and properly formatted
   ```python
   from PIL import Image

   def validate_image(img_path):
       try:
           img = Image.open(img_path)
           img.verify()
           return True
       except:
           return False
   ```

### Handling Multiple Correct Answers

For VQA, some questions may have multiple valid answers:

```python
# Store multiple answers
data = {
    'image': ['img1.jpg'],
    'prompt': ['What animals are shown?'],
    'text': ['dog and cat']  # Or separate answers with delimiter
}

# Or create multiple training examples
for answer in ['dog and cat', 'cat and dog', 'two animals']:
    data['image'].append('img1.jpg')
    data['prompt'].append('What animals are shown?')
    data['text'].append(answer)
```

## Best Practices

### 1. Start with PEFT/LoRA

Always use PEFT for VLM fine-tuning unless you have very large GPU memory:
```python
params.peft = True
params.quantization = "int4"
params.lora_r = 16
```

### 2. Choose Appropriate Batch Size and Accumulation

VLMs are very memory-intensive:
```python
params.batch_size = 2  # Small per-device batch
params.gradient_accumulation = 8  # Effective batch size = 16
```

### 3. Use Lower Learning Rates

VLMs are sensitive to learning rate:
- With PEFT: 1e-5 to 5e-5
- Full fine-tuning: 1e-6 to 1e-5

### 4. Select the Right Model Resolution

- **224px:** Fastest, good for simple VQA
- **448px:** Better detail, suitable for most tasks
- **896px:** Best quality, requires most memory

```python
# For detailed visual understanding
params.model = "google/paligemma-3b-pt-448"
```

### 5. Monitor Training Carefully

VLM training can be unstable:
```python
params.valid_split = "validation"
params.eval_strategy = "epoch"
params.save_total_limit = 3
```

### 6. Merge Adapters for Deployment

After training with PEFT, consider merging:
```python
params.merge_adapter = True  # Merge LoRA weights into base model
```

### 7. Use Gradient Checkpointing

Reduce memory usage:
```python
params.disable_gradient_checkpointing = False  # Enable checkpointing
```

### 8. Quality Over Quantity

For VLM tasks, high-quality annotations matter more than dataset size. Aim for:
- Clear, unambiguous questions
- Accurate, concise answers
- Diverse visual scenarios
- Consistent annotation style

## Evaluation and Metrics

### Training Metrics

During training, the trainer logs:
- **Loss:** Training and validation loss
- **Perplexity:** Model's confidence in predictions
- **Learning Rate:** Current learning rate value

### Manual Evaluation

After training, evaluate the model manually:

```python
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch

# Load trained model
processor = PaliGemmaProcessor.from_pretrained("./my-vqa-model")
model = PaliGemmaForConditionalGeneration.from_pretrained("./my-vqa-model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.eval().to(device)

# Test on new image
image = Image.open("test_image.jpg")
prompt = "What is in this image?"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Question: {prompt}")
print(f"Answer: {result}")
```

### Automated Evaluation Metrics

For VQA, you can compute:
- **Exact Match:** Percentage of exact answer matches
- **F1 Score:** Token-level overlap between predicted and ground truth
- **BLEU Score:** For open-ended captioning tasks

```python
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu

def evaluate_vqa(predictions, ground_truths):
    exact_matches = sum([pred == gt for pred, gt in zip(predictions, ground_truths)])
    exact_match_rate = exact_matches / len(predictions)

    # Token-level F1
    # (implementation depends on tokenization strategy)

    return {
        'exact_match': exact_match_rate
    }
```

## Troubleshooting Common Issues

### Out of Memory Errors

**Symptoms:** CUDA OOM during training or inference

**Solutions:**
1. Reduce batch size:
   ```python
   params.batch_size = 1
   params.gradient_accumulation = 16
   ```

2. Enable quantization:
   ```python
   params.quantization = "int4"
   ```

3. Use smaller model resolution:
   ```python
   params.model = "google/paligemma-3b-pt-224"  # Instead of 448 or 896
   ```

4. Enable gradient checkpointing:
   ```python
   params.disable_gradient_checkpointing = False
   ```

5. Use mixed precision:
   ```python
   params.mixed_precision = "fp16"  # or "bf16"
   ```

### Model Generates Gibberish

**Symptoms:** Outputs are random or nonsensical

**Solutions:**
1. Check data format:
   ```python
   # Verify dataset structure
   from datasets import load_from_disk
   dataset = load_from_disk(params.data_path)
   print(dataset['train'][0])
   ```

2. Lower learning rate:
   ```python
   params.lr = 1e-5  # Or even lower
   ```

3. Increase warmup:
   ```python
   params.warmup_ratio = 0.1  # Or higher
   ```

4. Train for more epochs:
   ```python
   params.epochs = 10
   ```

5. Check for data quality issues:
   - Verify answers are correct
   - Ensure images load properly
   - Check for inconsistent annotations

### Training Loss Not Decreasing

**Symptoms:** Loss stays high or constant

**Solutions:**
1. Increase learning rate slightly:
   ```python
   params.lr = 5e-5  # Try higher value
   ```

2. Check gradient accumulation:
   ```python
   params.gradient_accumulation = 8  # Ensure effective batch size is reasonable
   ```

3. Verify PEFT configuration:
   ```python
   params.peft = True
   params.lora_r = 32  # Try higher rank
   params.lora_alpha = 64
   ```

4. Check data:
   ```python
   # Ensure prompt and text columns are correct
   sample = dataset['train'][0]
   print(f"Prompt: {sample['prompt']}")
   print(f"Text: {sample['text']}")
   ```

### Model Not Using GPU

**Symptoms:** Training is very slow, GPU not utilized

**Solutions:**
1. Verify CUDA availability:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device: {torch.cuda.get_device_name(0)}")
   ```

2. Check quantization (requires CUDA):
   ```python
   # Quantization only works on CUDA
   if not torch.cuda.is_available():
       params.quantization = None
   ```

3. Install proper CUDA drivers and PyTorch version

### Inference is Slow

**Symptoms:** Generating answers takes too long

**Solutions:**
1. Merge adapter for faster inference:
   ```python
   params.merge_adapter = True  # During training
   ```

2. Use smaller model resolution:
   ```python
   # Use 224 instead of 448 or 896 for inference
   ```

3. Adjust generation parameters:
   ```python
   generated_ids = model.generate(
       **inputs,
       max_new_tokens=50,  # Reduce if possible
       do_sample=False,  # Greedy is faster than sampling
       num_beams=1  # No beam search
   )
   ```

4. Use int8 inference quantization:
   ```python
   from transformers import BitsAndBytesConfig

   quantization_config = BitsAndBytesConfig(load_in_8bit=True)
   model = PaliGemmaForConditionalGeneration.from_pretrained(
       "model_path",
       quantization_config=quantization_config
   )
   ```

## Example Projects

### Medical VQA System

```python
from datasets import Dataset, Features, Image, Value
from autotrain.trainers.vlm.params import VLMTrainingParams
from autotrain.trainers.vlm import train
import pandas as pd

# 1. Prepare medical VQA dataset
df = pd.read_csv('medical_vqa.csv')

features = Features({
    'image': Image(),
    'prompt': Value('string'),
    'text': Value('string')
})

dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column('image', Image())

# Split data
dataset = dataset.train_test_split(test_size=0.15, seed=42)

from datasets import DatasetDict
dataset_dict = DatasetDict({
    'train': dataset['train'],
    'validation': dataset['test']
})
dataset_dict.save_to_disk('./medical_vqa_data')

# 2. Configure training with careful parameters for medical domain
params = VLMTrainingParams(
    model="google/paligemma-3b-pt-448",  # Higher resolution for detail
    data_path="./medical_vqa_data",
    project_name="medical-vqa-assistant",
    trainer="vqa",
    train_split="train",
    valid_split="validation",
    lr=1e-5,  # Lower LR for medical domain
    epochs=10,
    batch_size=2,
    gradient_accumulation=8,
    warmup_ratio=0.1,
    peft=True,
    quantization="int4",
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    seed=42,
    mixed_precision="bf16"
)

# 3. Train model
train(params)

# 4. Use for inference
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch

processor = PaliGemmaProcessor.from_pretrained("./medical-vqa-assistant")
model = PaliGemmaForConditionalGeneration.from_pretrained("./medical-vqa-assistant")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.eval().to(device)

# Answer medical question
image = Image.open("xray.jpg")
question = "Is there any abnormality visible in this X-ray?"

inputs = processor(text=question, images=image, return_tensors="pt").to(device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=100)

answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(f"Q: {question}")
print(f"A: {answer}")
```

### E-commerce Product Q&A

```python
from autotrain.trainers.vlm.params import VLMTrainingParams
from autotrain.trainers.vlm import train

# Train on product images and Q&A pairs
params = VLMTrainingParams(
    model="google/paligemma-3b-pt-224",
    data_path="product-qa-dataset",
    project_name="product-qa-assistant",
    trainer="vqa",
    image_column="product_image",
    prompt_text_column="customer_question",
    text_column="answer",
    train_split="train",
    valid_split="validation",
    lr=2e-5,
    epochs=5,
    batch_size=4,
    gradient_accumulation=4,
    peft=True,
    quantization="int4",
    push_to_hub=True,
    username="my-store",
    token="hf_token"
)

train(params)

# Deploy as API endpoint or chatbot
```

### Automated Alt Text Generation

```python
from autotrain.trainers.vlm.params import VLMTrainingParams
from autotrain.trainers.vlm import train

# Train captioning model for accessibility
params = VLMTrainingParams(
    model="google/paligemma-3b-pt-448",
    data_path="./alt_text_dataset",
    project_name="alt-text-generator",
    trainer="captioning",
    image_column="image",
    text_column="alt_text",
    train_split="train",
    valid_split="validation",
    lr=1e-5,
    epochs=3,
    batch_size=4,
    gradient_accumulation=4,
    peft=True,
    quantization="int4",
    lora_r=16,
    lora_alpha=32
)

train(params)

# Use to generate alt text automatically
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

processor = PaliGemmaProcessor.from_pretrained("./alt-text-generator")
model = PaliGemmaForConditionalGeneration.from_pretrained("./alt-text-generator")

def generate_alt_text(image_path):
    image = Image.open(image_path)
    inputs = processor(text="describe", images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    alt_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return alt_text

# Batch process images
import os
for img_file in os.listdir('./images'):
    alt_text = generate_alt_text(f'./images/{img_file}')
    print(f"{img_file}: {alt_text}")
```

## Integration with Hugging Face Hub

### Pushing Model to Hub

```python
params = VLMTrainingParams(
    model="google/paligemma-3b-pt-224",
    data_path="./my_vqa_data",
    project_name="my-vqa-model",
    trainer="vqa",
    push_to_hub=True,
    username="my-hf-username",
    token="hf_your_token_here",
    # ... other params
)

train(params)
# Model uploaded to: huggingface.co/my-hf-username/my-vqa-model
```

### Using PEFT Adapter from Hub

```python
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
import torch

# Load base model
base_model_id = "google/paligemma-3b-pt-224"
adapter_model_id = "my-username/my-vqa-model"

processor = PaliGemmaProcessor.from_pretrained(base_model_id)
base_model = PaliGemmaForConditionalGeneration.from_pretrained(base_model_id)

# Load PEFT adapter
model = PeftModel.from_pretrained(base_model, adapter_model_id)

# Optional: Merge for faster inference
model = model.merge_and_unload()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.eval().to(device)

# Use the model
from PIL import Image
image = Image.open("test.jpg")
question = "What do you see?"

inputs = processor(text=question, images=image, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)
answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(answer)
```

## Performance Optimization

### Memory Optimization

1. **Use Quantization:**
   ```python
   params.quantization = "int4"  # Reduces memory by ~4x
   ```

2. **Enable Gradient Checkpointing:**
   ```python
   params.disable_gradient_checkpointing = False
   ```

3. **Reduce Batch Size:**
   ```python
   params.batch_size = 1
   params.gradient_accumulation = 16
   ```

### Speed Optimization

1. **Use Mixed Precision:**
   ```python
   params.mixed_precision = "bf16"  # Faster on modern GPUs
   ```

2. **Reduce Logging:**
   ```python
   params.logging_steps = 100
   ```

3. **Use Compiled Models (PyTorch 2.0+):**
   ```python
   import torch
   model = torch.compile(model)
   ```

### Inference Optimization

1. **Merge Adapters:**
   ```python
   # During training
   params.merge_adapter = True

   # Or after training
   from peft import PeftModel
   model = PeftModel.from_pretrained(base_model, adapter_path)
   model = model.merge_and_unload()
   model.save_pretrained("merged_model")
   ```

2. **Use Static KV Cache:**
   ```python
   model.generation_config.use_cache = True
   ```

3. **Batch Inference:**
   ```python
   # Process multiple images at once
   images = [Image.open(f"img{i}.jpg") for i in range(4)]
   prompts = ["What is this?"] * 4

   inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
   outputs = model.generate(**inputs, max_new_tokens=100)
   answers = processor.batch_decode(outputs, skip_special_tokens=True)
   ```

## See Also

- [Image Classification](./ImageClassification.md) - For classifying entire images
- [Object Detection](./ObjectDetection.md) - For detecting and localizing objects
- [Text Generation](../clm/README.md) - For pure language model training
- [Multimodal Understanding](https://huggingface.co/blog/paligemma) - PaliGemma blog post
