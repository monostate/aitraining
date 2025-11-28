# AutoTrain Advanced - Direct Preference Optimization (DPO) Guide

## Table of Contents
- [Overview](#overview)
- [When to Use DPO](#when-to-use-dpo)
- [Quick Start](#quick-start)
- [Parameters](#parameters)
- [Data Format](#data-format)
- [Examples](#examples)
- [DPO vs RLHF](#dpo-vs-rlhf)
- [Best Practices](#best-practices)

## Overview

Direct Preference Optimization (DPO) is a simpler and more stable alternative to RLHF (Reinforcement Learning from Human Feedback) that directly optimizes for human preferences without requiring a separate reward model. AutoTrain implements DPO using TRL's production-ready trainer.

### Key Features

- ✅ **No Reward Model Needed** - Direct optimization from preferences
- ✅ **Stable Training** - Avoids RL instabilities
- ✅ **Memory Efficient** - Lower memory than PPO
- ✅ **PEFT/LoRA Support** - For large model training
- ✅ **Reference Model Caching** - Efficient KL computation
- ✅ **Multiple Loss Functions** - Sigmoid, hinge, IPO variants

### How DPO Works

DPO optimizes the policy to maximize the log likelihood of preferred responses while minimizing it for rejected responses, with a KL divergence constraint from a reference model:

```
Loss = -log(σ(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
```

Where:
- `y_w` = preferred (chosen) response
- `y_l` = rejected response
- `π` = current policy
- `π_ref` = reference policy
- `β` = temperature parameter controlling KL penalty

## When to Use DPO

Use DPO when you have:
- **Preference data** (chosen vs rejected responses)
- **Need alignment** with human preferences
- **Want stability** compared to PPO/RLHF
- **Limited compute** (more efficient than PPO)

**Advantages over PPO:**
- No reward model training required
- More stable and predictable
- Lower memory usage
- Faster convergence

**Limitations:**
- Requires paired preference data
- Less flexible than reward models
- Can overfit to preference dataset

## Quick Start

### CLI Usage

```bash
# Basic DPO training
aitraining llm \
  --model gpt2 \
  --trainer dpo \
  --data-path ./preferences.json \
  --prompt-text-column prompt \
  --text-column chosen \
  --rejected-text-column rejected \
  --project-name my-dpo-model \
  --dpo-beta 0.1 \
  --epochs 3 \
  --batch-size 4

# With LoRA for large models
aitraining llm \
  --model meta-llama/Llama-2-7b-hf \
  --trainer dpo \
  --data-path ./preferences.json \
  --prompt-text-column prompt \
  --text-column chosen \
  --rejected-text-column rejected \
  --project-name llama-dpo-lora \
  --peft \
  --lora-r 16 \
  --lora-alpha 32 \
  --quantization int4 \
  --dpo-beta 0.1

# With reference model
aitraining llm \
  --model gpt2 \
  --model-ref gpt2 \
  --trainer dpo \
  --data-path ./preferences.json \
  --prompt-text-column prompt \
  --text-column chosen \
  --rejected-text-column rejected \
  --project-name dpo-with-ref \
  --dpo-beta 0.2
```

### Python API

```python
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.clm import train

# Configure DPO training
config = LLMTrainingParams(
    model="gpt2",
    trainer="dpo",
    data_path="./preferences.json",

    # DPO-specific columns
    prompt_text_column="prompt",
    text_column="chosen",  # Preferred response
    rejected_text_column="rejected",  # Rejected response

    # DPO parameters
    dpo_beta=0.1,  # KL penalty coefficient
    model_ref="gpt2",  # Reference model (optional)

    # Training parameters
    project_name="my-dpo-model",
    epochs=3,
    batch_size=4,
    lr=5e-7,  # Lower LR for DPO

    # Optional: PEFT
    peft=True,
    lora_r=16,
)

# Train
model = train(config)
```

## Parameters

### DPO-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--trainer` | str | "dpo" | Must be "dpo" for DPO training |
| `--prompt-text-column` | str | required | Column with prompts |
| `--text-column` | str | required | Column with chosen responses |
| `--rejected-text-column` | str | required | Column with rejected responses |
| `--dpo-beta` | float | 0.1 | KL penalty coefficient (0.01-0.5 typical) |
| `--model-ref` | str | None | Reference model (uses model if None) |
| `--max-prompt-length` | int | 128 | Maximum prompt length |
| `--max-completion-length` | int | None | Maximum response length |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | 3 | Training epochs |
| `--batch-size` | int | 4 | Batch size (keep small) |
| `--lr` | float | 5e-7 | Learning rate (use lower than SFT) |
| `--warmup-ratio` | float | 0.1 | Warmup ratio |
| `--gradient-accumulation` | int | 4 | Gradient accumulation |

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--max-seq-length` | int | 512 | Maximum sequence length |
| `--padding` | str | "right" | Padding side |
| `--truncation` | bool | True | Enable truncation |

## Data Format

DPO requires preference data with three columns:

### JSON Format
```json
[
  {
    "prompt": "Write a poem about AI",
    "chosen": "Circuits hum with digital dreams...",
    "rejected": "AI is computer stuff..."
  },
  {
    "prompt": "Explain quantum computing",
    "chosen": "Quantum computing leverages quantum mechanics...",
    "rejected": "Quantum computers are fast..."
  }
]
```

### CSV Format
```csv
prompt,chosen,rejected
"Write a poem about AI","Circuits hum with...","AI is computer..."
"Explain quantum computing","Quantum computing...","Quantum computers..."
```

### Parquet Format
```python
import pandas as pd

df = pd.DataFrame({
    "prompt": ["Write a poem", "Explain ML"],
    "chosen": ["Beautiful verses...", "ML is..."],
    "rejected": ["Bad poem", "Wrong explanation"]
})
df.to_parquet("preferences.parquet")
```

## Examples

### Example 1: Basic DPO Training

```python
import json
from autotrain.trainers.clm import train
from autotrain.trainers.clm.params import LLMTrainingParams

# Prepare preference data
preferences = [
    {
        "prompt": "How do I stay healthy?",
        "chosen": "To stay healthy: 1) Eat balanced diet 2) Exercise regularly 3) Get enough sleep 4) Manage stress",
        "rejected": "Just don't get sick lol"
    },
    {
        "prompt": "Explain machine learning",
        "chosen": "Machine learning is a subset of AI where computers learn patterns from data...",
        "rejected": "Computers learning stuff"
    }
]

with open("preferences.json", "w") as f:
    json.dump(preferences, f)

# Train with DPO
config = LLMTrainingParams(
    model="gpt2",
    trainer="dpo",
    data_path="preferences.json",
    prompt_text_column="prompt",
    text_column="chosen",
    rejected_text_column="rejected",
    dpo_beta=0.1,
    project_name="dpo-aligned",
    epochs=3,
    batch_size=2,
    lr=5e-7,
)

model = train(config)
```

### Example 2: DPO with LoRA on Large Model

```python
config = LLMTrainingParams(
    model="meta-llama/Llama-2-7b-hf",
    trainer="dpo",
    data_path="./preferences_large.json",

    # Columns
    prompt_text_column="instruction",
    text_column="preferred",
    rejected_text_column="rejected",

    # DPO config
    dpo_beta=0.2,
    max_prompt_length=256,
    max_completion_length=512,

    # PEFT for efficiency
    peft=True,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules="q_proj,v_proj",

    # Optimization
    quantization="int4",
    mixed_precision="fp16",
    gradient_checkpointing=True,

    # Training
    project_name="llama-dpo-aligned",
    epochs=2,
    batch_size=1,
    gradient_accumulation=8,
    lr=1e-6,
)

model = train(config)
```

### Example 3: Creating Preference Data from Ratings

```python
import pandas as pd

# Convert ratings to preferences
responses = pd.DataFrame({
    "prompt": ["Write code", "Write code", "Explain AI", "Explain AI"],
    "response": ["def hello():", "hello world", "AI is...", "AI means..."],
    "rating": [5, 2, 4, 3]
})

# Create preference pairs
preferences = []
prompts = responses['prompt'].unique()

for prompt in prompts:
    prompt_responses = responses[responses['prompt'] == prompt]
    sorted_responses = prompt_responses.sort_values('rating', ascending=False)

    if len(sorted_responses) >= 2:
        preferences.append({
            "prompt": prompt,
            "chosen": sorted_responses.iloc[0]['response'],
            "rejected": sorted_responses.iloc[-1]['response']
        })

# Save for DPO training
import json
with open("preferences.json", "w") as f:
    json.dump(preferences, f)
```

## DPO vs RLHF

### Comparison Table

| Aspect | DPO | RLHF (PPO) |
|--------|-----|------------|
| **Complexity** | Simple | Complex |
| **Stability** | Very stable | Can be unstable |
| **Memory Usage** | Lower | Higher |
| **Training Time** | Faster | Slower |
| **Reward Model** | Not needed | Required |
| **Flexibility** | Limited | High |
| **Data Requirements** | Preference pairs | Can use any rewards |
| **Hyperparameter Sensitivity** | Low | High |

### When to Choose DPO

Choose DPO when:
- You have preference data available
- You want stable, predictable training
- You have limited computational resources
- You don't need complex reward shaping

### When to Choose RLHF/PPO

Choose PPO when:
- You need custom reward functions
- You want maximum flexibility
- You have computational resources
- You need online learning

## Best Practices

### 1. Data Quality

**Preference Quality**
- Ensure clear preference differences
- Avoid contradictory preferences
- Balance prompt diversity
- Include edge cases

**Data Collection**
```python
# Good: Clear preference
{
    "prompt": "Summarize this article",
    "chosen": "Here's a concise summary: [well-structured summary]",
    "rejected": "[rambling, unclear response]"
}

# Bad: Unclear preference
{
    "prompt": "Tell me about dogs",
    "chosen": "Dogs are animals",
    "rejected": "Dogs are pets"  # Both are correct
}
```

### 2. Hyperparameter Tuning

**Beta Parameter**
- Lower β (0.01-0.05): More deviation from reference
- Medium β (0.1-0.2): Balanced (recommended)
- Higher β (0.3-0.5): Stay close to reference

```python
# Experiment with beta
for beta in [0.01, 0.1, 0.2]:
    config = LLMTrainingParams(
        trainer="dpo",
        dpo_beta=beta,
        project_name=f"dpo_beta_{beta}",
        # ... other params
    )
```

**Learning Rate**
- Use 10x lower than SFT (e.g., 5e-7 instead of 5e-6)
- Increase warmup ratio to 0.2
- Use cosine scheduler for stability

### 3. Reference Model

**Options:**
- Same as base model (default)
- Pre-trained checkpoint
- SFT checkpoint (recommended)

```python
# Train SFT first, then DPO
# Step 1: SFT
sft_config = LLMTrainingParams(
    trainer="sft",
    model="gpt2",
    project_name="gpt2-sft",
    # ...
)
sft_model = train(sft_config)

# Step 2: DPO with SFT as reference
dpo_config = LLMTrainingParams(
    trainer="dpo",
    model="./gpt2-sft",  # Start from SFT
    model_ref="./gpt2-sft",  # Use SFT as reference
    project_name="gpt2-dpo",
    # ...
)
```

### 4. Monitoring

**Key Metrics**
- `rewards/chosen`: Should increase
- `rewards/rejected`: Should decrease
- `rewards/margins`: Should be positive and increase
- `loss/dpo`: Should decrease

**Early Stopping**
```python
config = LLMTrainingParams(
    trainer="dpo",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    metric_for_best_model="rewards/margins",
    greater_is_better=True,
    # ...
)
```

### 5. Common Pitfalls

**Overfitting**
- Use validation split
- Monitor margin growth
- Stop if margins become too large

**Mode Collapse**
- Happens when β is too low
- Model generates same response
- Increase β or add diversity penalty

**Preference Inconsistency**
- Check data for contradictions
- Use consistent annotators
- Consider multi-annotator agreement

## Troubleshooting

### Issue: Training Loss Not Decreasing

```python
# Solutions:
# 1. Reduce learning rate
config.lr = 1e-7

# 2. Increase beta
config.dpo_beta = 0.2

# 3. Check data quality
# Ensure clear preferences
```

### Issue: Model Deviates Too Much

```python
# Increase KL penalty
config.dpo_beta = 0.3

# Use stronger reference model
config.model_ref = "gpt2-medium"
```

### Issue: OOM Errors

```python
# Reduce sequence lengths
config.max_prompt_length = 128
config.max_completion_length = 256

# Use PEFT
config.peft = True
config.lora_r = 8

# Enable gradient checkpointing
config.gradient_checkpointing = True
```

## Advanced Topics

### Custom DPO Loss Variants

```python
# IPO (Identity Preference Optimization)
config = LLMTrainingParams(
    trainer="dpo",
    loss_type="ipo",  # If supported
    # ...
)

# cDPO (conservative DPO)
config = LLMTrainingParams(
    trainer="dpo",
    label_smoothing=0.1,  # If supported
    # ...
)
```

### Multi-Turn Conversations

```python
# Format multi-turn as single sequence
conversation = {
    "prompt": "User: Hello\nAssistant: Hi!\nUser: How are you?",
    "chosen": "I'm doing well, thank you for asking!",
    "rejected": "meh"
}
```

## Next Steps

- [ORPO Training](./ORPO.md) - Odds Ratio Preference Optimization
- [PPO Training](./PPO.md) - For maximum flexibility
- [Reward Model Training](./Reward.md) - Build custom reward models
- [SFT Training](./SFT.md) - Supervised fine-tuning basics