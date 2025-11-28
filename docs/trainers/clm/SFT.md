# AutoTrain Advanced - Supervised Fine-Tuning (SFT) Guide

## Table of Contents
- [Overview](#overview)
- [When to Use SFT](#when-to-use-sft)
- [Quick Start](#quick-start)
- [Parameters](#parameters)
- [Training Features](#training-features)
- [Examples](#examples)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)

## Overview

Supervised Fine-Tuning (SFT) is the standard approach for adapting pre-trained language models to specific tasks using labeled data. AutoTrain's SFT trainer leverages TRL (Transformer Reinforcement Learning) library to provide a production-ready implementation with advanced features.

### Key Features

- ✅ **Standard Fine-tuning** with cross-entropy loss
- ✅ **PEFT/LoRA Support** for parameter-efficient training
- ✅ **Quantization** (4-bit, 8-bit) for memory efficiency
- ✅ **Chat Templates** for conversation-style training
- ✅ **Flash Attention** support for faster training
- ✅ **Gradient Checkpointing** for memory optimization
- ✅ **Unsloth Integration** for optimized kernels
- ✅ **Knowledge Distillation** (when combined with `--use-distillation`)

## When to Use SFT

Use SFT when you have:
- **Labeled examples** showing desired outputs
- **Task-specific data** (Q&A pairs, instructions, conversations)
- **Need for controlled outputs** following specific formats
- **Limited compute** (can use PEFT/LoRA)

**Not recommended for:**
- Preference learning (use DPO/ORPO)
- Reward optimization (use PPO)
- Unlabeled data (use pre-training)

## Quick Start

### CLI Usage

```bash
# Basic SFT training
aitraining llm \
  --model gpt2 \
  --trainer sft \
  --data-path ./data \
  --text-column text \
  --project-name my-sft-model \
  --epochs 3 \
  --batch-size 8 \
  --lr 5e-5

# With PEFT/LoRA
aitraining llm \
  --model meta-llama/Llama-2-7b-hf \
  --trainer sft \
  --data-path ./data \
  --text-column text \
  --project-name llama-sft-lora \
  --peft \
  --lora-r 16 \
  --lora-alpha 32 \
  --quantization int4 \
  --epochs 3

# Chat-style training
aitraining llm \
  --model gpt2 \
  --trainer sft \
  --data-path ./chat_data \
  --text-column messages \
  --chat-template chatml \
  --project-name chat-model \
  --epochs 3
```

### Python API

```python
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.clm import train

# Configure training
config = LLMTrainingParams(
    model="gpt2",
    trainer="sft",
    data_path="./data",
    text_column="text",
    project_name="my-sft-model",

    # Training hyperparameters
    epochs=3,
    batch_size=8,
    lr=5e-5,
    warmup_ratio=0.1,
    gradient_accumulation=4,

    # Optional: PEFT
    peft=True,
    lora_r=16,
    lora_alpha=32,

    # Optional: Optimization
    mixed_precision="fp16",
    gradient_checkpointing=True,
    use_flash_attention_2=True,
)

# Train
model = train(config)
```

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | required | Model name or path |
| `--data-path` | str | required | Path to training data |
| `--text-column` | str | "text" | Column containing training text |
| `--project-name` | str | required | Output directory name |
| `--trainer` | str | "sft" | Must be "sft" for supervised fine-tuning |

### Training Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | 3 | Number of training epochs |
| `--batch-size` | int | 8 | Training batch size |
| `--lr` | float | 5e-5 | Learning rate |
| `--warmup-ratio` | float | 0.1 | Warmup ratio for scheduler |
| `--gradient-accumulation` | int | 4 | Gradient accumulation steps |
| `--optimizer` | str | "adamw_torch" | Optimizer type |
| `--scheduler` | str | "linear" | LR scheduler type |
| `--weight-decay` | float | 0.0 | Weight decay coefficient |
| `--max-grad-norm` | float | 1.0 | Max gradient norm for clipping |

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--max-seq-length` | int | 2048 | Maximum sequence length |
| `--padding` | str | "right" | Padding side (left/right) |
| `--chat-template` | str | None | Chat template (chatml, zephyr, etc.) |
| `--add-eos-token` | bool | True | Add EOS token to sequences |
| `--block-size` | int | -1 | Block size for training |

### PEFT/LoRA Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--peft` | bool | False | Enable PEFT/LoRA |
| `--lora-r` | int | 16 | LoRA rank |
| `--lora-alpha` | int | 32 | LoRA alpha |
| `--lora-dropout` | float | 0.05 | LoRA dropout |
| `--target-modules` | str | "all-linear" | Target modules for LoRA |
| `--merge-adapter` | bool | False | Merge adapter after training |

### Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--quantization` | str | None | Quantization (int4, int8) |
| `--mixed-precision` | str | None | Mixed precision (fp16, bf16) |
| `--use-flash-attention-2` | bool | False | Use Flash Attention 2 |
| `--disable-gradient-checkpointing` | bool | False | Disable gradient checkpointing |
| `--unsloth` | bool | False | Use Unsloth optimization |
| `--packing` | bool | False | Pack sequences for efficiency |

## Training Features

### 1. Knowledge Distillation

Combine SFT with distillation for improved performance:

```bash
aitraining llm \
  --model gpt2 \
  --trainer sft \
  --use-distillation \
  --teacher-model gpt2-large \
  --distill-temperature 3.0 \
  --distill-alpha 0.7 \
  --data-path ./data \
  --project-name distilled-model
```

### 2. Chat Template Training

Train conversational models with structured formats:

```bash
aitraining llm \
  --model gpt2 \
  --trainer sft \
  --data-path ./conversations.json \
  --text-column messages \
  --chat-template chatml \
  --project-name chatbot
```

Supported templates:
- `chatml`: ChatML format
- `zephyr`: Zephyr format
- `llama`: Llama chat format
- `alpaca`: Alpaca instruction format
- `vicuna`: Vicuna format
- `mistral`: Mistral format

### 3. Custom Metrics

Enable generation metrics during training:

```python
config = LLMTrainingParams(
    trainer="sft",
    use_enhanced_eval=True,
    eval_metrics="perplexity,bleu,rouge",
    eval_save_predictions=True,
    # ... other params
)
```

## Examples

### Example 1: Fine-tune GPT-2 on Custom Dataset

```python
import pandas as pd
from autotrain.trainers.clm import train
from autotrain.trainers.clm.params import LLMTrainingParams

# Prepare data
data = pd.DataFrame({
    "text": [
        "Question: What is Python? Answer: Python is a programming language.",
        "Question: What is ML? Answer: Machine Learning is a subset of AI.",
        # ... more examples
    ]
})
data.to_csv("./train.csv", index=False)

# Configure and train
config = LLMTrainingParams(
    model="gpt2",
    trainer="sft",
    data_path="./train.csv",
    text_column="text",
    project_name="gpt2-qa",
    epochs=5,
    batch_size=4,
    lr=2e-5,
    max_seq_length=512,
)

model = train(config)
```

### Example 2: LoRA Fine-tuning for Large Models

```python
config = LLMTrainingParams(
    model="meta-llama/Llama-2-7b-hf",
    trainer="sft",
    data_path="./instructions.json",
    text_column="text",
    project_name="llama2-lora",

    # PEFT configuration
    peft=True,
    lora_r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules="q_proj,v_proj,k_proj,o_proj",

    # Optimization
    quantization="int4",
    mixed_precision="fp16",
    gradient_checkpointing=True,

    # Training
    epochs=3,
    batch_size=1,
    gradient_accumulation=16,
    lr=1e-4,
)

model = train(config)
```

### Example 3: Chat Model Training

```python
# Prepare chat data
chat_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language."}
        ]
    },
    # ... more conversations
]

# Save as JSON
import json
with open("chat_data.json", "w") as f:
    json.dump(chat_data, f)

# Train
config = LLMTrainingParams(
    model="gpt2",
    trainer="sft",
    data_path="chat_data.json",
    text_column="messages",
    chat_template="chatml",
    project_name="chatbot",
    epochs=3,
    batch_size=4,
)

model = train(config)
```

## Advanced Usage

### Custom Data Preprocessing

```python
from autotrain.trainers.clm.train_clm_sft import CustomSFTTrainer
from transformers import AutoTokenizer

class MySFTTrainer(CustomSFTTrainer):
    def preprocess_function(self, examples):
        # Custom preprocessing logic
        texts = []
        for text in examples[self.text_column]:
            # Apply custom formatting
            formatted = f"### Instruction: {text}\n### Response: "
            texts.append(formatted)
        return {"text": texts}

# Use custom trainer
trainer = MySFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
```

### Integration with Weights & Biases

```python
config = LLMTrainingParams(
    trainer="sft",
    log="wandb",  # Enable W&B logging
    project_name="my-project",
    # ... other params
)
```

### Multi-GPU Training

```bash
# Use all available GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 aitraining llm \
  --trainer sft \
  --distributed-backend "ddp" \
  # ... other params

# Or with accelerate
accelerate launch --multi_gpu --num_processes 4 \
  aitraining llm --trainer sft ...
```

## Best Practices

### 1. Data Quality
- **Clean your data**: Remove duplicates, fix formatting issues
- **Balance your dataset**: Ensure diverse examples
- **Validate outputs**: Check that target outputs are correct
- **Use validation split**: Monitor overfitting

### 2. Hyperparameter Tuning
- **Start small**: Begin with small batch sizes and models
- **Learning rate**: Start with 5e-5 for small models, 1e-5 for large
- **Warmup**: Use 5-10% warmup for stability
- **Gradient accumulation**: Increase for larger effective batch size

### 3. Memory Optimization
- **Use PEFT**: Reduces memory by 90%+ for large models
- **Enable gradient checkpointing**: Trade compute for memory
- **Use quantization**: int4/int8 for inference, training
- **Reduce sequence length**: Only use what you need

### 4. Training Stability
- **Monitor loss**: Should decrease smoothly
- **Check gradient norms**: Spikes indicate instability
- **Use early stopping**: Prevent overfitting
- **Save checkpoints**: Regular saves prevent data loss

### 5. Evaluation
- **Use proper metrics**: Perplexity, BLEU, ROUGE for generation
- **Test on held-out data**: Never evaluate on training data
- **Human evaluation**: Final quality check
- **A/B testing**: Compare with baseline

## Troubleshooting

### Common Issues

**OOM (Out of Memory)**
```bash
# Reduce batch size
--batch-size 1 --gradient-accumulation 32

# Enable memory optimizations
--gradient-checkpointing --quantization int4

# Use PEFT
--peft --lora-r 8
```

**Loss Not Decreasing**
```bash
# Reduce learning rate
--lr 1e-5

# Increase warmup
--warmup-ratio 0.2

# Check data quality
```

**Slow Training**
```bash
# Enable Flash Attention
--use-flash-attention-2

# Use mixed precision
--mixed-precision fp16

# Enable packing
--packing
```

## Next Steps

- [DPO Training](./DPO.md) - For preference-based optimization
- [PPO Training](./PPO.md) - For reinforcement learning
- [ORPO Training](./ORPO.md) - For odds ratio preference optimization
- [Reward Model Training](./Reward.md) - For RLHF reward models