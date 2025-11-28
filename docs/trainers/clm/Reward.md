# AutoTrain Advanced - Reward Model Training Guide

## Table of Contents
- [Overview](#overview)
- [When to Use Reward Models](#when-to-use-reward-models)
- [Quick Start](#quick-start)
- [Parameters](#parameters)
- [Data Format](#data-format)
- [Examples](#examples)
- [Using Reward Models](#using-reward-models)
- [Best Practices](#best-practices)

## Overview

Reward models are the foundation of RLHF (Reinforcement Learning from Human Feedback). They learn to predict human preferences and provide reward signals for training language models with PPO. AutoTrain implements reward model training using TRL's production-ready RewardTrainer.

### Key Features

- ✅ **Preference Learning** - Learn from human feedback
- ✅ **Multiple Architectures** - Classification or regression heads
- ✅ **PEFT/LoRA Support** - Efficient training for large models
- ✅ **Built for PPO** - Direct integration with PPO training
- ✅ **Custom Metrics** - Accuracy, ranking metrics
- ✅ **Multi-GPU Support** - Distributed training

### How Reward Models Work

A reward model takes a prompt and response as input and outputs a scalar reward:

```
Input: (prompt, response) → Model → Reward score

Training: Learn to assign higher scores to preferred responses
Usage: Guide PPO training by providing rewards
```

The model learns from paired comparisons where one response is preferred over another.

## When to Use Reward Models

Use reward models when:
- **Building RLHF pipeline** - Core component for PPO
- **Have preference data** - Human annotations available
- **Need custom scoring** - Task-specific quality metrics
- **Want interpretability** - Explicit reward signals

Not needed when:
- **Using DPO/ORPO** - Direct preference optimization
- **Have explicit rewards** - Known scoring functions
- **Limited data** - Insufficient preferences

## Quick Start

### CLI Usage

```bash
# Basic reward model training
aitraining llm \
  --model gpt2 \
  --trainer reward \
  --data-path ./preferences.json \
  --prompt-text-column prompt \
  --text-column chosen \
  --rejected-text-column rejected \
  --project-name my-reward-model \
  --epochs 3 \
  --batch-size 8

# Reward model with LoRA
aitraining llm \
  --model meta-llama/Llama-2-7b-hf \
  --trainer reward \
  --data-path ./preferences.json \
  --prompt-text-column prompt \
  --text-column chosen \
  --rejected-text-column rejected \
  --project-name llama-reward-lora \
  --peft \
  --lora-r 16 \
  --lora-alpha 32 \
  --quantization int4

# Reward model for specific task
aitraining llm \
  --model bert-base-uncased \
  --trainer reward \
  --data-path ./code_preferences.json \
  --prompt-text-column instruction \
  --text-column good_code \
  --rejected-text-column bad_code \
  --project-name code-reward-model \
  --max-seq-length 512
```

### Python API

```python
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.clm import train

# Configure reward model training
config = LLMTrainingParams(
    model="gpt2",
    trainer="reward",
    data_path="./preferences.json",

    # Data columns
    prompt_text_column="prompt",
    text_column="chosen",  # Preferred response
    rejected_text_column="rejected",  # Rejected response

    # Training parameters
    project_name="my-reward-model",
    epochs=3,
    batch_size=8,
    lr=1e-5,
    warmup_ratio=0.1,

    # Optional: PEFT
    peft=True,
    lora_r=16,
    lora_alpha=32,
)

# Train
reward_model = train(config)
```

## Parameters

### Reward Model Specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--trainer` | str | "reward" | Must be "reward" for reward model |
| `--prompt-text-column` | str | required | Column with prompts |
| `--text-column` | str | required | Column with chosen responses |
| `--rejected-text-column` | str | required | Column with rejected responses |
| `--max-seq-length` | int | 512 | Maximum sequence length |
| `--num-labels` | int | 1 | Number of reward outputs (usually 1) |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | 3 | Training epochs |
| `--batch-size` | int | 8 | Batch size |
| `--lr` | float | 1e-5 | Learning rate |
| `--warmup-ratio` | float | 0.1 | Warmup ratio |
| `--weight-decay` | float | 0.01 | Weight decay |
| `--gradient-accumulation` | int | 1 | Gradient accumulation |

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | required | Base model for reward head |
| `--peft` | bool | False | Enable LoRA |
| `--lora-r` | int | 16 | LoRA rank |
| `--lora-alpha` | int | 32 | LoRA alpha |
| `--quantization` | str | None | int4/int8 quantization |

## Data Format

Reward models require preference pairs - chosen vs rejected responses:

### JSON Format
```json
[
  {
    "prompt": "Write a Python function to sort a list",
    "chosen": "def sort_list(lst):\n    return sorted(lst)",
    "rejected": "lst.sort()"
  },
  {
    "prompt": "Explain machine learning",
    "chosen": "Machine learning is a branch of AI that enables computers to learn from data...",
    "rejected": "ML is when computers learn stuff"
  }
]
```

### CSV Format
```csv
prompt,chosen,rejected
"Write a function","def func(): return 'good'","func = bad"
"Explain AI","Artificial Intelligence is...","AI is robots"
```

### Creating Preference Data

```python
# From ratings
def create_preferences_from_ratings(responses_with_scores):
    preferences = []
    prompts = responses_with_scores.groupby('prompt')

    for prompt, group in prompts:
        sorted_responses = group.sort_values('score', ascending=False)
        if len(sorted_responses) >= 2:
            preferences.append({
                "prompt": prompt,
                "chosen": sorted_responses.iloc[0]['response'],
                "rejected": sorted_responses.iloc[-1]['response']
            })

    return preferences

# From A/B tests
def create_preferences_from_ab(ab_test_results):
    preferences = []
    for test in ab_test_results:
        if test['winner'] == 'A':
            chosen, rejected = test['response_a'], test['response_b']
        else:
            chosen, rejected = test['response_b'], test['response_a']

        preferences.append({
            "prompt": test['prompt'],
            "chosen": chosen,
            "rejected": rejected
        })

    return preferences
```

## Examples

### Example 1: Basic Reward Model

```python
from autotrain.trainers.clm import train
from autotrain.trainers.clm.params import LLMTrainingParams
import json

# Create preference data
preferences = [
    {
        "prompt": "How do I stay healthy?",
        "chosen": "To stay healthy: eat balanced diet, exercise regularly, sleep well, manage stress",
        "rejected": "idk just don't get sick"
    },
    {
        "prompt": "Explain photosynthesis",
        "chosen": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen...",
        "rejected": "plants make food from sun"
    }
]

with open("preferences.json", "w") as f:
    json.dump(preferences, f)

# Train reward model
config = LLMTrainingParams(
    model="gpt2",
    trainer="reward",
    data_path="preferences.json",
    prompt_text_column="prompt",
    text_column="chosen",
    rejected_text_column="rejected",
    project_name="basic-reward-model",
    epochs=5,
    batch_size=4,
    lr=1e-5,
)

reward_model = train(config)
```

### Example 2: Code Quality Reward Model

```python
# Specialized reward model for code quality
code_preferences = [
    {
        "prompt": "Write a function to calculate factorial",
        "chosen": """def factorial(n: int) -> int:
    '''Calculate factorial of n recursively.'''
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
        "rejected": "def fact(n): return 1 if n==0 else n*fact(n-1)"
    }
]

config = LLMTrainingParams(
    model="codeparrot/codeparrot-small",
    trainer="reward",
    data_path="code_preferences.json",
    prompt_text_column="prompt",
    text_column="chosen",
    rejected_text_column="rejected",
    project_name="code-quality-reward",
    max_seq_length=1024,
    epochs=10,
    batch_size=8,
)

code_reward_model = train(config)
```

### Example 3: Large Model with LoRA

```python
config = LLMTrainingParams(
    model="meta-llama/Llama-2-7b-hf",
    trainer="reward",
    data_path="./large_preferences.json",

    # Data columns
    prompt_text_column="instruction",
    text_column="preferred",
    rejected_text_column="rejected",

    # PEFT configuration
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
    project_name="llama-reward-lora",
    epochs=2,
    batch_size=2,
    gradient_accumulation=8,
    lr=5e-6,
    warmup_ratio=0.1,
)

reward_model = train(config)
```

### Example 4: Multi-Aspect Reward Model

```python
# Train multiple reward models for different aspects
aspects = ["helpfulness", "harmlessness", "honesty"]
reward_models = {}

for aspect in aspects:
    config = LLMTrainingParams(
        model="gpt2",
        trainer="reward",
        data_path=f"./{aspect}_preferences.json",
        prompt_text_column="prompt",
        text_column="chosen",
        rejected_text_column="rejected",
        project_name=f"reward-{aspect}",
        epochs=5,
        batch_size=8,
    )

    reward_models[aspect] = train(config)

# Combined scoring
def get_combined_reward(prompt, response):
    scores = {}
    for aspect, model in reward_models.items():
        scores[aspect] = model.predict(prompt, response)

    # Weighted combination
    weights = {"helpfulness": 0.5, "harmlessness": 0.3, "honesty": 0.2}
    total = sum(scores[a] * weights[a] for a in aspects)
    return total
```

## Using Reward Models

### With PPO Training

```python
# Step 1: Train reward model
reward_config = LLMTrainingParams(
    model="gpt2",
    trainer="reward",
    data_path="preferences.json",
    # ... reward model params
    project_name="my-reward-model",
)
reward_model = train(reward_config)

# Step 2: Use in PPO training
ppo_config = LLMTrainingParams(
    model="gpt2",
    trainer="ppo",
    rl_reward_model_path="./my-reward-model",  # Use trained reward model
    data_path="./prompts.json",
    project_name="ppo-with-reward",
    # ... PPO params
)
ppo_model = train(ppo_config)
```

### Standalone Scoring

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load trained reward model
model = AutoModelForSequenceClassification.from_pretrained("./my-reward-model")
tokenizer = AutoTokenizer.from_pretrained("./my-reward-model")

def score_response(prompt, response):
    # Combine prompt and response
    text = f"{prompt} {tokenizer.sep_token} {response}"

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )

    # Get reward score
    with torch.no_grad():
        outputs = model(**inputs)
        reward = outputs.logits.squeeze().item()

    return reward

# Example usage
prompt = "Write a haiku about coding"
response1 = "Bugs hide in the code\nPatience reveals solutions\nClean logic emerges"
response2 = "coding is hard\ncomputers are confusing\nI don't like it"

score1 = score_response(prompt, response1)
score2 = score_response(prompt, response2)

print(f"Response 1 score: {score1:.2f}")
print(f"Response 2 score: {score2:.2f}")
```

### Ranking Responses

```python
def rank_responses(prompt, responses, reward_model, tokenizer):
    scores = []

    for response in responses:
        text = f"{prompt} {tokenizer.sep_token} {response}"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            score = reward_model(**inputs).logits.squeeze().item()
            scores.append((response, score))

    # Sort by score
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked

# Example
responses = [
    "Great explanation here...",
    "Okay answer...",
    "Poor response..."
]
ranked = rank_responses("Explain AI", responses, model, tokenizer)
```

## Best Practices

### 1. Data Quality

**Preference Quality**
- Clear quality differences between chosen/rejected
- Consistent annotation guidelines
- Multiple annotators when possible
- Regular quality checks

```python
# Good preference pair
{
    "prompt": "Explain recursion",
    "chosen": "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem...",
    "rejected": "recursion is recursion"
}

# Bad preference pair (too similar)
{
    "prompt": "What is 2+2?",
    "chosen": "4",
    "rejected": "four"
}
```

### 2. Data Balance

```python
# Balance different types of preferences
def balance_preferences(preferences):
    # Group by prompt type/difficulty
    easy = [p for p in preferences if is_easy(p["prompt"])]
    medium = [p for p in preferences if is_medium(p["prompt"])]
    hard = [p for p in preferences if is_hard(p["prompt"])]

    # Sample equally
    min_size = min(len(easy), len(medium), len(hard))
    balanced = (
        random.sample(easy, min_size) +
        random.sample(medium, min_size) +
        random.sample(hard, min_size)
    )

    return balanced
```

### 3. Training Strategy

**Progressive Training**
```python
# Start with clear preferences, then subtle
stages = [
    ("clear_preferences.json", 3),  # 3 epochs
    ("subtle_preferences.json", 2),  # 2 epochs
]

model = None
for data_path, epochs in stages:
    config = LLMTrainingParams(
        model=model or "gpt2",
        trainer="reward",
        data_path=data_path,
        epochs=epochs,
        # ...
    )
    model = train(config)
```

**Validation Strategy**
```python
# Hold out test set for evaluation
from sklearn.model_selection import train_test_split

train_prefs, test_prefs = train_test_split(
    preferences,
    test_size=0.2,
    random_state=42
)

# Train on train set
config.data_path = "train_preferences.json"

# Evaluate on test set
def evaluate_reward_model(model, test_data):
    correct = 0
    for item in test_data:
        score_chosen = score_response(item["prompt"], item["chosen"])
        score_rejected = score_response(item["prompt"], item["rejected"])
        if score_chosen > score_rejected:
            correct += 1

    accuracy = correct / len(test_data)
    return accuracy
```

### 4. Preventing Overfitting

```python
config = LLMTrainingParams(
    trainer="reward",
    # Regularization
    weight_decay=0.01,
    dropout=0.1,  # If supported

    # Early stopping
    eval_strategy="steps",
    eval_steps=100,
    early_stopping_patience=3,
    early_stopping_threshold=0.01,
    load_best_model_at_end=True,

    # Small learning rate
    lr=5e-6,
    warmup_ratio=0.1,

    # ...
)
```

### 5. Calibration

```python
# Calibrate reward scores
def calibrate_rewards(model, calibration_set):
    scores = []
    labels = []  # 1 for chosen, 0 for rejected

    for item in calibration_set:
        # Score chosen
        score_c = score_response(item["prompt"], item["chosen"])
        scores.append(score_c)
        labels.append(1)

        # Score rejected
        score_r = score_response(item["prompt"], item["rejected"])
        scores.append(score_r)
        labels.append(0)

    # Fit calibration (e.g., Platt scaling)
    from sklearn.linear_model import LogisticRegression
    calibrator = LogisticRegression()
    calibrator.fit(np.array(scores).reshape(-1, 1), labels)

    return calibrator
```

## Troubleshooting

### Issue: Reward Model Always Outputs Same Score

```python
# Solutions:
# 1. Check data balance
print(f"Unique prompts: {len(set(p['prompt'] for p in preferences))}")

# 2. Reduce learning rate
config.lr = 1e-6

# 3. Add regularization
config.weight_decay = 0.1
```

### Issue: Poor Preference Accuracy

```python
# 1. Improve data quality
# Filter for clear preferences only

# 2. Increase model size
config.model = "gpt2-medium"  # Larger model

# 3. Train longer
config.epochs = 10
```

### Issue: Overfitting

```python
# 1. Add dropout
config.lora_dropout = 0.2  # If using LoRA

# 2. Reduce model capacity
config.lora_r = 8  # Smaller rank

# 3. More data augmentation
# Paraphrase prompts, swap preference pairs
```

## Advanced Topics

### Ensemble Reward Models

```python
# Train ensemble for better robustness
models = []
for seed in [42, 123, 456]:
    config = LLMTrainingParams(
        trainer="reward",
        seed=seed,
        project_name=f"reward_seed_{seed}",
        # ...
    )
    models.append(train(config))

# Ensemble prediction
def ensemble_score(prompt, response):
    scores = [score_response(prompt, response, m) for m in models]
    return np.mean(scores)  # Or median for robustness
```

### Active Learning

```python
# Select most informative preferences
def select_uncertain_pairs(model, candidate_pairs):
    uncertainties = []

    for pair in candidate_pairs:
        score_c = score_response(pair["prompt"], pair["chosen"])
        score_r = score_response(pair["prompt"], pair["rejected"])
        uncertainty = abs(score_c - score_r)  # Small diff = uncertain
        uncertainties.append((uncertainty, pair))

    # Select most uncertain
    uncertainties.sort(key=lambda x: x[0])
    selected = [pair for _, pair in uncertainties[:100]]

    return selected
```

## Integration with PPO

See [PPO Training](./PPO.md) for complete RLHF pipeline:

```python
# Complete RLHF Pipeline
# 1. Train Reward Model (this guide)
# 2. Use with PPO for RLHF
# 3. Evaluate final model
```

## Next Steps

- [PPO Training](./PPO.md) - Use reward model for RLHF
- [DPO Training](./DPO.md) - Alternative without reward model
- [ORPO Training](./ORPO.md) - Single-stage preference learning
- [SFT Training](./SFT.md) - Basic supervised fine-tuning