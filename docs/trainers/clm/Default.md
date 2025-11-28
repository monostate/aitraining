# AutoTrain Advanced - Default/Generic CLM Training Guide

## Table of Contents
- [Overview](#overview)
- [When to Use Default Training](#when-to-use-default-training)
- [Quick Start](#quick-start)
- [Parameters](#parameters)
- [Data Format](#data-format)
- [Examples](#examples)
- [Comparison with SFT](#comparison-with-sft)
- [Best Practices](#best-practices)

## Overview

The Default trainer provides generic Causal Language Model (CLM) training, implementing standard autoregressive language modeling. This is the most basic form of training where the model learns to predict the next token given previous tokens.

### Key Features

- ✅ **Pure Language Modeling** - Standard next-token prediction
- ✅ **Simple Data Format** - Just text, no special formatting
- ✅ **Minimal Preprocessing** - Direct text tokenization
- ✅ **PEFT/LoRA Support** - Efficient training for large models
- ✅ **Continuous Pre-training** - Extend model knowledge

### How Default Training Works

```
Input: "The quick brown"
Target: "quick brown fox"

Loss = CrossEntropy(predicted_next_token, actual_next_token)
```

The model learns to predict each token given all previous tokens, training on raw text without instruction formatting.

## When to Use Default Training

Use Default trainer when:
- **Continuous pre-training** - Adapting to new domains
- **Raw text data** - No instruction/response pairs
- **Language modeling** - Learning text patterns
- **Domain adaptation** - Specializing on specific text

Use SFT instead when:
- **Instruction following** - Q&A or task-specific
- **Structured outputs** - Need specific formats
- **Chat models** - Conversational abilities
- **Labeled pairs** - Have input-output examples

## Quick Start

### CLI Usage

```bash
# Basic default training
aitraining llm \
  --model gpt2 \
  --trainer default \
  --data-path ./corpus.txt \
  --text-column text \
  --project-name my-lm-model \
  --epochs 3 \
  --batch-size 8 \
  --lr 5e-5

# Domain adaptation with LoRA
aitraining llm \
  --model meta-llama/Llama-2-7b-hf \
  --trainer default \
  --data-path ./medical_texts.json \
  --text-column content \
  --project-name llama-medical \
  --peft \
  --lora-r 32 \
  --lora-alpha 64 \
  --quantization int4

# Continuous pre-training
aitraining llm \
  --model ./my-base-model \
  --trainer default \
  --data-path ./new_data.parquet \
  --text-column text \
  --project-name continued-pretraining \
  --block-size 2048 \
  --epochs 1
```

### Python API

```python
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.clm import train

# Configure default training
config = LLMTrainingParams(
    model="gpt2",
    trainer="default",
    data_path="./corpus.json",
    text_column="text",
    project_name="my-lm-model",

    # Training parameters
    epochs=3,
    batch_size=8,
    lr=5e-5,
    warmup_ratio=0.1,

    # Model config
    block_size=1024,
    add_eos_token=True,

    # Optional: PEFT
    peft=True,
    lora_r=16,
)

# Train
model = train(config)
```

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--trainer` | str | "default" | Must be "default" for generic CLM |
| `--model` | str | required | Base model name or path |
| `--data-path` | str | required | Path to training data |
| `--text-column` | str | "text" | Column containing text |
| `--project-name` | str | required | Output directory |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | 3 | Number of epochs |
| `--batch-size` | int | 8 | Batch size |
| `--lr` | float | 5e-5 | Learning rate |
| `--warmup-ratio` | float | 0.1 | Warmup ratio |
| `--gradient-accumulation` | int | 4 | Gradient accumulation |
| `--weight-decay` | float | 0.0 | Weight decay |

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--block-size` | int | -1 | Maximum sequence length |
| `--add-eos-token` | bool | True | Add EOS token to sequences |
| `--padding` | str | "right" | Padding side |
| `--max-seq-length` | int | 2048 | Max sequence length |

## Data Format

Default trainer accepts simple text data:

### Text File
```text
This is the first document about machine learning.
Here is another document about deep learning.
Natural language processing is fascinating.
```

### JSON Format
```json
[
  {"text": "First document content..."},
  {"text": "Second document content..."},
  {"text": "Third document content..."}
]
```

### CSV Format
```csv
text
"Machine learning is a subset of artificial intelligence..."
"Deep learning uses neural networks..."
"NLP processes human language..."
```

### Parquet Format
```python
import pandas as pd

df = pd.DataFrame({
    "text": [
        "Document 1 content...",
        "Document 2 content...",
        "Document 3 content..."
    ]
})
df.to_parquet("corpus.parquet")
```

## Examples

### Example 1: Domain Adaptation

```python
from autotrain.trainers.clm import train
from autotrain.trainers.clm.params import LLMTrainingParams

# Adapt GPT-2 to medical domain
config = LLMTrainingParams(
    model="gpt2",
    trainer="default",
    data_path="./medical_papers.json",
    text_column="abstract",
    project_name="gpt2-medical",

    # Smaller learning rate for adaptation
    lr=2e-5,
    epochs=2,
    batch_size=4,

    # Longer sequences for papers
    block_size=1024,

    # Save checkpoints
    save_strategy="epoch",
)

model = train(config)
```

### Example 2: Continuous Pre-training

```python
# Continue pre-training with new data
config = LLMTrainingParams(
    model="./my-existing-model",
    trainer="default",
    data_path="./wikipedia_2024.json",
    text_column="content",
    project_name="model-updated-2024",

    # Single epoch for updating
    epochs=1,
    batch_size=16,
    lr=1e-5,

    # Match original training
    block_size=2048,

    # Resume from checkpoint
    resume_from_checkpoint=True,
)

updated_model = train(config)
```

### Example 3: Code Model Training

```python
# Train on code corpus
code_config = LLMTrainingParams(
    model="codeparrot/codeparrot-small",
    trainer="default",
    data_path="./python_code.json",
    text_column="code",
    project_name="python-lm",

    # Code-specific settings
    block_size=2048,  # Longer for code
    add_eos_token=False,  # Code doesn't need EOS

    # Training
    epochs=5,
    batch_size=4,
    lr=5e-5,
)

code_model = train(code_config)
```

### Example 4: Efficient Large Model Training

```python
# Train Llama with LoRA
config = LLMTrainingParams(
    model="meta-llama/Llama-2-7b-hf",
    trainer="default",
    data_path="./large_corpus.parquet",
    text_column="text",
    project_name="llama-domain-adapted",

    # PEFT for efficiency
    peft=True,
    lora_r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules="q_proj,k_proj,v_proj,o_proj",

    # Optimization
    quantization="int4",
    mixed_precision="bf16",
    gradient_checkpointing=True,

    # Training
    epochs=1,
    batch_size=1,
    gradient_accumulation=32,
    lr=2e-4,
)

model = train(config)
```

## Comparison with SFT

| Aspect | Default | SFT |
|--------|---------|-----|
| **Data Format** | Raw text | Instruction-response pairs |
| **Use Case** | Language modeling | Task-specific training |
| **Preprocessing** | Minimal | Template formatting |
| **Loss** | Standard CLM | Masked on inputs |
| **Output** | Continuation | Structured responses |
| **Best For** | Domain adaptation | Fine-tuning for tasks |

### When to Use Each

**Default Training:**
```python
# Good for: Learning domain language
data = {"text": "The patient presented with acute symptoms..."}
```

**SFT Training:**
```python
# Good for: Task-specific outputs
data = {
    "text": "Question: What are the symptoms?\nAnswer: The symptoms include..."
}
```

## Best Practices

### 1. Data Preparation

**Quality over Quantity**
```python
# Clean your corpus
def clean_text(text):
    # Remove unwanted characters
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove very short texts
    if len(text.split()) < 10:
        return None
    return text

cleaned_data = [clean_text(doc) for doc in raw_data]
cleaned_data = [doc for doc in cleaned_data if doc]
```

**Deduplication**
```python
# Remove duplicates
from datasets import Dataset

dataset = Dataset.from_dict({"text": texts})
dataset = dataset.filter(lambda x: len(x["text"]) > 100)
dataset = dataset.deduplicate(column="text")
```

### 2. Training Strategy

**Curriculum Learning**
```python
# Start with shorter sequences, then longer
for block_size in [512, 1024, 2048]:
    config = LLMTrainingParams(
        trainer="default",
        block_size=block_size,
        project_name=f"model_{block_size}",
        # ...
    )
    model = train(config)
```

**Learning Rate Scheduling**
```python
config = LLMTrainingParams(
    trainer="default",
    lr=5e-5,
    scheduler="cosine",  # Better for longer training
    warmup_ratio=0.05,  # Shorter warmup for pre-training
    # ...
)
```

### 3. Memory Optimization

```python
# For large models
config = LLMTrainingParams(
    trainer="default",

    # Reduce memory usage
    gradient_checkpointing=True,
    mixed_precision="bf16",

    # Smaller batches with accumulation
    batch_size=1,
    gradient_accumulation=32,

    # Efficient attention
    use_flash_attention_2=True,

    # ...
)
```

### 4. Monitoring

```python
# Track perplexity
config = LLMTrainingParams(
    trainer="default",
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    log="wandb",  # or "tensorboard"

    # Save best model
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    # ...
)
```

## Troubleshooting

### Issue: High Loss / No Convergence

```python
# Solutions:
# 1. Check data quality
print(f"Average doc length: {np.mean([len(d.split()) for d in data])}")
print(f"Unique docs: {len(set(data))}")

# 2. Adjust learning rate
config.lr = 1e-5  # Lower for stability

# 3. Increase warmup
config.warmup_ratio = 0.1
```

### Issue: Overfitting

```python
# Add regularization
config.weight_decay = 0.01

# Reduce epochs
config.epochs = 1

# Add dropout (if using LoRA)
config.lora_dropout = 0.1
```

### Issue: OOM Errors

```python
# Reduce sequence length
config.block_size = 512

# Enable memory optimizations
config.gradient_checkpointing = True
config.quantization = "int8"

# Use PEFT
config.peft = True
config.lora_r = 8
```

## Advanced Topics

### Custom Data Collator

```python
from transformers import DataCollatorForLanguageModeling

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Custom processing
        batch = super().__call__(features)
        # Add custom logic
        return batch

# Use in training
data_collator = CustomDataCollator(
    tokenizer=tokenizer,
    mlm=False,  # False for CLM
    pad_to_multiple_of=8,  # For efficiency
)
```

### Multi-Dataset Training

```python
# Combine multiple corpora
datasets = []
for path in ["corpus1.json", "corpus2.json", "corpus3.json"]:
    df = pd.read_json(path)
    datasets.append(df)

combined = pd.concat(datasets, ignore_index=True)
combined = combined.sample(frac=1).reset_index(drop=True)  # Shuffle
combined.to_json("combined_corpus.json")

# Train on combined
config.data_path = "combined_corpus.json"
```

## Next Steps

- [SFT Training](./SFT.md) - For instruction-following models
- [DPO Training](./DPO.md) - For preference alignment
- [Distillation Training](./Distillation.md) - For knowledge transfer
- [PPO Training](./PPO.md) - For reinforcement learning