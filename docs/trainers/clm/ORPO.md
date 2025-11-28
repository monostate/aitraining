# AutoTrain Advanced - Odds Ratio Preference Optimization (ORPO) Guide

## Table of Contents
- [Overview](#overview)
- [ORPO vs DPO](#orpo-vs-dpo)
- [When to Use ORPO](#when-to-use-orpo)
- [Quick Start](#quick-start)
- [Parameters](#parameters)
- [Data Format](#data-format)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Overview

Odds Ratio Preference Optimization (ORPO) is an advanced preference learning method that combines supervised fine-tuning with preference alignment in a single training phase. Unlike DPO which requires a reference model, ORPO directly optimizes for preferences while maintaining language modeling capabilities.

### Key Features

- ✅ **Single-Stage Training** - Combined SFT and preference optimization
- ✅ **No Reference Model** - More memory efficient than DPO
- ✅ **Odds Ratio Objective** - More principled preference modeling
- ✅ **Better Generalization** - Maintains strong language modeling
- ✅ **PEFT/LoRA Support** - Efficient large model training

### How ORPO Works

ORPO optimizes using an odds ratio between chosen and rejected responses:

```
Loss = λ * L_SFT + (1-λ) * L_OR

Where:
L_SFT = -log P(chosen|prompt)  # Standard language modeling loss
L_OR = -log σ(log odds_chosen - log odds_rejected)  # Odds ratio loss
```

The key insight: ORPO penalizes the model for assigning high probability to rejected responses while rewarding chosen responses, without needing a separate reference model.

## ORPO vs DPO

### Comparison

| Aspect | ORPO | DPO |
|--------|------|-----|
| **Reference Model** | Not needed ✅ | Required |
| **Memory Usage** | Lower | Higher |
| **Training Stages** | Single | Two (SFT → DPO) |
| **Language Modeling** | Preserved | May degrade |
| **Stability** | Very stable | Stable |
| **Convergence** | Faster | Moderate |

### Key Advantages of ORPO

1. **Efficiency**: No reference model means ~50% memory savings
2. **Simplicity**: Single training stage instead of SFT→DPO pipeline
3. **Quality**: Better maintains general capabilities
4. **Speed**: Faster convergence due to combined objective

## When to Use ORPO

Use ORPO when:
- **Starting from scratch** - Training from base model
- **Memory constrained** - Limited GPU memory
- **Want simplicity** - Single-stage training
- **Need general capabilities** - Maintain broad skills

Use DPO instead when:
- **Have good SFT model** - Already fine-tuned
- **Need fine control** - Separate SFT and preference stages
- **Have specific reference** - Want to stay close to reference

## Quick Start

### CLI Usage

```bash
# Basic ORPO training
aitraining llm \
  --model gpt2 \
  --trainer orpo \
  --data-path ./preferences.json \
  --prompt-text-column prompt \
  --text-column chosen \
  --rejected-text-column rejected \
  --project-name my-orpo-model \
  --orpo-alpha 1.0 \
  --epochs 3 \
  --batch-size 4

# ORPO with LoRA for large models
aitraining llm \
  --model meta-llama/Llama-2-7b-hf \
  --trainer orpo \
  --data-path ./preferences.json \
  --prompt-text-column prompt \
  --text-column chosen \
  --rejected-text-column rejected \
  --project-name llama-orpo \
  --peft \
  --lora-r 32 \
  --lora-alpha 64 \
  --quantization int4 \
  --orpo-alpha 1.0

# ORPO with custom weighting
aitraining llm \
  --model gpt2 \
  --trainer orpo \
  --data-path ./data \
  --prompt-text-column instruction \
  --text-column response_good \
  --rejected-text-column response_bad \
  --project-name weighted-orpo \
  --orpo-alpha 0.5  # Balance SFT and OR losses
```

### Python API

```python
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.clm import train

# Configure ORPO training
config = LLMTrainingParams(
    model="gpt2",
    trainer="orpo",
    data_path="./preferences.json",

    # ORPO columns
    prompt_text_column="prompt",
    text_column="chosen",
    rejected_text_column="rejected",

    # ORPO specific
    orpo_alpha=1.0,  # Weight for odds ratio loss

    # Training
    project_name="my-orpo-model",
    epochs=3,
    batch_size=4,
    lr=5e-5,  # Can use higher LR than DPO

    # Optional: PEFT
    peft=True,
    lora_r=16,
    lora_alpha=32,
)

# Train
model = train(config)
```

## Parameters

### ORPO-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--trainer` | str | "orpo" | Must be "orpo" for ORPO training |
| `--prompt-text-column` | str | required | Column containing prompts |
| `--text-column` | str | required | Column with chosen responses |
| `--rejected-text-column` | str | required | Column with rejected responses |
| `--orpo-alpha` | float | 1.0 | Weight for OR loss (0=pure SFT, 1=pure OR) |
| `--max-prompt-length` | int | 128 | Maximum prompt length |
| `--max-completion-length` | int | None | Maximum response length |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | 3 | Number of training epochs |
| `--batch-size` | int | 4 | Training batch size |
| `--lr` | float | 5e-5 | Learning rate (can be higher than DPO) |
| `--warmup-ratio` | float | 0.1 | Warmup ratio |
| `--gradient-accumulation` | int | 4 | Gradient accumulation steps |
| `--weight-decay` | float | 0.01 | Weight decay |

## Data Format

ORPO uses the same data format as DPO - preference pairs:

### JSON Format
```json
[
  {
    "prompt": "Explain quantum computing",
    "chosen": "Quantum computing leverages quantum mechanical phenomena...",
    "rejected": "Quantum computers are just fast computers..."
  }
]
```

### Creating Quality Preference Data

```python
import json

# High-quality preferences for ORPO
preferences = []

# Example: Code generation preferences
preferences.append({
    "prompt": "Write a Python function to reverse a string",
    "chosen": """def reverse_string(s: str) -> str:
    '''Reverse a string using slicing.'''
    return s[::-1]""",
    "rejected": "def rev(x): return ''.join(reversed(x))"
})

# Example: Instruction following
preferences.append({
    "prompt": "List 3 benefits of exercise",
    "chosen": """Three key benefits of exercise are:
1. Improved cardiovascular health
2. Better mental well-being and mood
3. Increased strength and endurance""",
    "rejected": "Exercise is good for you."
})

with open("orpo_data.json", "w") as f:
    json.dump(preferences, f, indent=2)
```

## Examples

### Example 1: Basic ORPO Training

```python
from autotrain.trainers.clm import train
from autotrain.trainers.clm.params import LLMTrainingParams

# Simple ORPO setup
config = LLMTrainingParams(
    model="gpt2",
    trainer="orpo",
    data_path="./preferences.json",
    prompt_text_column="prompt",
    text_column="chosen",
    rejected_text_column="rejected",
    orpo_alpha=1.0,
    project_name="gpt2-orpo",
    epochs=3,
    batch_size=4,
    lr=5e-5,
)

model = train(config)

# Test the model
from transformers import pipeline
generator = pipeline("text-generation", model=model)
result = generator("Explain machine learning", max_length=100)
```

### Example 2: ORPO with Ablation Study

```python
# Compare different alpha values
alphas = [0.0, 0.5, 1.0, 2.0]  # 0=SFT only, higher=more preference weight

results = {}
for alpha in alphas:
    config = LLMTrainingParams(
        model="gpt2",
        trainer="orpo",
        data_path="./preferences.json",
        prompt_text_column="prompt",
        text_column="chosen",
        rejected_text_column="rejected",
        orpo_alpha=alpha,
        project_name=f"orpo_alpha_{alpha}",
        epochs=2,
        batch_size=4,
        lr=5e-5,
        eval_strategy="epoch",
    )

    model = train(config)
    results[alpha] = model

# Evaluate each model
for alpha, model in results.items():
    print(f"Alpha {alpha}: {evaluate(model)}")
```

### Example 3: Large Model ORPO with LoRA

```python
config = LLMTrainingParams(
    model="meta-llama/Llama-2-7b-hf",
    trainer="orpo",
    data_path="./high_quality_preferences.json",

    # Data columns
    prompt_text_column="instruction",
    text_column="preferred_response",
    rejected_text_column="rejected_response",

    # ORPO configuration
    orpo_alpha=1.0,
    max_prompt_length=256,
    max_completion_length=512,

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
    project_name="llama2-orpo-aligned",
    epochs=2,
    batch_size=1,
    gradient_accumulation=16,
    lr=2e-4,
    warmup_ratio=0.1,
    weight_decay=0.01,

    # Evaluation
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
)

model = train(config)
```

### Example 4: Multi-Task ORPO

```python
# Combine different types of preferences
import pandas as pd

# Create diverse preference data
qa_preferences = [...]  # Q&A preferences
code_preferences = [...]  # Code generation preferences
creative_preferences = [...]  # Creative writing preferences

# Combine all preferences
all_preferences = qa_preferences + code_preferences + creative_preferences

# Shuffle for better training
import random
random.shuffle(all_preferences)

# Save and train
import json
with open("multi_task_preferences.json", "w") as f:
    json.dump(all_preferences, f)

config = LLMTrainingParams(
    model="gpt2-medium",
    trainer="orpo",
    data_path="multi_task_preferences.json",
    prompt_text_column="prompt",
    text_column="chosen",
    rejected_text_column="rejected",
    orpo_alpha=1.0,
    project_name="multi-task-orpo",
    epochs=5,
    batch_size=8,
    lr=3e-5,
)

model = train(config)
```

## Best Practices

### 1. Data Quality

**Preference Clarity**
```python
# Good: Clear quality difference
{
    "prompt": "Write a haiku about coding",
    "chosen": "Bugs hide in the code\nPatience reveals the solution\nClean logic emerges",
    "rejected": "coding is hard work\ncomputers do things\nlots of typing"
}

# Bad: Similar quality
{
    "prompt": "What is 2+2?",
    "chosen": "4",
    "rejected": "Four"  # Both are correct
}
```

**Diversity is Key**
- Include various task types
- Balance difficulty levels
- Cover edge cases
- Avoid repetitive patterns

### 2. Hyperparameter Tuning

**Alpha Parameter (λ)**
- `α = 0.0`: Pure SFT (no preference learning)
- `α = 0.5`: Balanced SFT and preferences
- `α = 1.0`: Standard ORPO (recommended start)
- `α > 1.0`: Strong preference focus

```python
# Finding optimal alpha
def find_best_alpha():
    alphas = [0.5, 0.75, 1.0, 1.25, 1.5]
    best_score = 0
    best_alpha = 1.0

    for alpha in alphas:
        config = LLMTrainingParams(
            trainer="orpo",
            orpo_alpha=alpha,
            # ... other params
        )
        model = train(config)
        score = evaluate_preferences(model)

        if score > best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha
```

**Learning Rate**
- Can use higher LR than DPO (5e-5 to 1e-4)
- Use warmup for stability
- Consider cosine scheduler

### 3. Training Strategy

**Progressive Training**
```python
# Stage 1: Focus on SFT
config_stage1 = LLMTrainingParams(
    trainer="orpo",
    orpo_alpha=0.3,  # More SFT
    epochs=2,
    # ...
)

# Stage 2: Increase preference weight
config_stage2 = LLMTrainingParams(
    trainer="orpo",
    orpo_alpha=1.0,  # Balanced
    epochs=2,
    model="./stage1_model",  # Continue from stage 1
    # ...
)
```

**Curriculum Learning**
```python
# Start with easy preferences, then harder ones
easy_preferences = [...]  # Clear preferences
hard_preferences = [...]  # Subtle preferences

# Train progressively
for i, preferences in enumerate([easy_preferences, hard_preferences]):
    config = LLMTrainingParams(
        trainer="orpo",
        data_path=f"preferences_{i}.json",
        model="./prev_model" if i > 0 else "gpt2",
        # ...
    )
```

### 4. Monitoring

**Key Metrics**
- `loss/orpo`: Combined loss (should decrease)
- `loss/sft`: Language modeling loss
- `loss/odds_ratio`: Preference loss
- `rewards/margins`: Preference margins (should increase)

**Early Stopping**
```python
config = LLMTrainingParams(
    trainer="orpo",
    early_stopping_patience=3,
    early_stopping_threshold=0.01,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    # ...
)
```

### 5. Evaluation

**Automated Metrics**
```python
from autotrain.evaluation import evaluate_model

def evaluate_orpo_model(model, test_preferences):
    metrics = {}

    # Preference accuracy
    correct = 0
    for item in test_preferences:
        score_chosen = model.score(item["prompt"], item["chosen"])
        score_rejected = model.score(item["prompt"], item["rejected"])
        if score_chosen > score_rejected:
            correct += 1

    metrics["preference_accuracy"] = correct / len(test_preferences)

    # Language modeling perplexity
    metrics["perplexity"] = calculate_perplexity(model, test_set)

    return metrics
```

**Human Evaluation**
- A/B testing against baseline
- Quality ratings
- Task-specific metrics

## Troubleshooting

### Issue: Poor Preference Learning

```python
# Solutions:
# 1. Increase alpha
config.orpo_alpha = 1.5

# 2. Improve data quality
# Filter for clearer preferences

# 3. Increase model capacity
config.lora_r = 32  # If using LoRA
```

### Issue: Degraded Language Modeling

```python
# Reduce alpha to maintain LM capabilities
config.orpo_alpha = 0.5

# Or use progressive training
# Start with lower alpha, gradually increase
```

### Issue: Slow Convergence

```python
# 1. Increase learning rate
config.lr = 1e-4

# 2. Reduce batch size for more updates
config.batch_size = 2
config.gradient_accumulation = 8

# 3. Use better initialization
# Start from SFT model instead of base
```

### Issue: OOM Errors

```python
# Standard memory optimizations
config.gradient_checkpointing = True
config.peft = True
config.lora_r = 8
config.quantization = "int4"
config.max_seq_length = 512  # Reduce if needed
```

## Advanced Topics

### Custom ORPO Loss

```python
# Modify ORPO loss weighting dynamically
class DynamicORPOTrainer(ORPOTrainer):
    def compute_loss(self, model, inputs):
        # Start with more SFT, gradually increase OR weight
        progress = self.state.global_step / self.state.max_steps
        dynamic_alpha = min(1.0, progress * 2)  # 0 to 1 over half training

        # Compute losses with dynamic weight
        loss = dynamic_alpha * or_loss + (1 - dynamic_alpha) * sft_loss
        return loss
```

### Ensemble ORPO

```python
# Train multiple ORPO models with different alphas
models = []
for alpha in [0.5, 1.0, 1.5]:
    config = LLMTrainingParams(
        trainer="orpo",
        orpo_alpha=alpha,
        project_name=f"ensemble_{alpha}",
        # ...
    )
    models.append(train(config))

# Ensemble inference
def ensemble_generate(prompt):
    outputs = [model.generate(prompt) for model in models]
    # Vote or average
    return best_output(outputs)
```

## Comparison Summary

### ORPO vs Other Methods

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **SFT** | Supervised learning | Simple, stable | No preference learning |
| **DPO** | Preference with reference | Stable, proven | Needs reference model |
| **ORPO** | Preference from scratch | No reference, efficient | Less studied |
| **PPO** | Complex rewards | Maximum flexibility | Complex, unstable |

## Next Steps

- [DPO Training](./DPO.md) - When you have a good reference model
- [PPO Training](./PPO.md) - For complex reward functions
- [Reward Model Training](./Reward.md) - Build custom reward models
- [SFT Training](./SFT.md) - Basic supervised fine-tuning