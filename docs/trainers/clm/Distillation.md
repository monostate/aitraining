# AutoTrain Advanced - Prompt Distillation Training Guide

## Table of Contents
- [Overview](#overview)
- [When to Use Distillation](#when-to-use-distillation)
- [Quick Start](#quick-start)
- [Parameters](#parameters)
- [How It Works](#how-it-works)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Advanced Usage](#advanced-usage)

## Overview

Prompt Distillation is an advanced training technique that teaches student models to internalize complex prompting strategies from teacher models. Instead of using long, detailed prompts at inference time, the model learns to produce teacher-quality outputs with minimal or no prompting.

### Key Features

- ✅ **Prompt Internalization** - Student learns prompt behaviors
- ✅ **Inference Cost Reduction** - Shorter prompts, same quality
- ✅ **Knowledge Transfer** - From large to small models
- ✅ **Temperature Control** - Soft target distribution matching
- ✅ **Flexible Templates** - Custom teacher/student prompts
- ✅ **Combined Loss** - KL divergence + cross-entropy

### The Distillation Advantage

Traditional approach:
```
Input: "You are an expert. Think step-by-step. Question: What is 2+2?"
Model: [Needs full prompt for good output]
```

After distillation:
```
Input: "What is 2+2?"
Model: [Produces expert-quality output without prompt]
```

## When to Use Distillation

Use Distillation when:
- **Expensive prompts** - Complex prompts slow inference
- **API cost reduction** - Minimize token usage
- **Model compression** - Transfer from large to small
- **Prompt engineering** - Bake in optimal prompts
- **Deployment efficiency** - Faster inference needed

Not recommended for:
- **No teacher available** - Need existing good model
- **Dynamic prompts** - Prompts change frequently
- **Small data** - Requires substantial examples

## Quick Start

### CLI Usage

```bash
# Basic prompt distillation
aitraining llm \
  --model gpt2 \
  --trainer distillation \
  --teacher-model gpt2-large \
  --teacher-prompt-template "You are an expert. Think step-by-step.\n\nQuestion: {input}\n\nAnswer:" \
  --student-prompt-template "{input}" \
  --data-path ./queries.json \
  --text-column text \
  --project-name distilled-model \
  --distill-temperature 3.0 \
  --distill-alpha 0.7

# Distillation with SFT (recommended)
aitraining llm \
  --model gpt2 \
  --trainer sft \
  --use-distillation \
  --teacher-model gpt-3.5-turbo \
  --distill-temperature 3.5 \
  --distill-alpha 0.75 \
  --data-path ./data \
  --project-name efficient-model

# Large model distillation with LoRA
aitraining llm \
  --model meta-llama/Llama-2-7b-hf \
  --trainer distillation \
  --teacher-model meta-llama/Llama-2-70b-hf \
  --teacher-prompt-template "### System: You are a helpful, respectful assistant.\n### Human: {input}\n### Assistant:" \
  --student-prompt-template "Q: {input}\nA:" \
  --peft \
  --lora-r 32 \
  --quantization int4 \
  --data-path ./instructions.json
```

### Python API

```python
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.clm import train

# Configure distillation
config = LLMTrainingParams(
    model="gpt2",
    trainer="distillation",

    # Teacher configuration
    teacher_model="gpt2-large",
    teacher_prompt_template="""You are an expert assistant.
    Think through the problem step-by-step.

    Question: {input}

    Detailed Answer:""",

    # Student configuration
    student_prompt_template="{input}",

    # Distillation parameters
    distill_temperature=3.0,  # Soften distributions
    distill_alpha=0.7,  # Balance KL and CE loss
    distill_max_teacher_length=512,

    # Training
    data_path="./queries.json",
    text_column="text",
    project_name="distilled-gpt2",
    epochs=3,
    batch_size=4,
    lr=5e-5,
)

model = train(config)
```

### Integrated with SFT (Recommended)

```python
# Better approach: SFT with distillation
config = LLMTrainingParams(
    model="gpt2",
    trainer="sft",
    use_distillation=True,  # Enable distillation in SFT

    teacher_model="gpt2-large",
    teacher_prompt_template="Expert: {input}",
    student_prompt_template="{input}",
    distill_temperature=3.0,
    distill_alpha=0.7,

    # Rest of SFT parameters
    data_path="./data",
    text_column="text",
    project_name="sft-distilled",
    epochs=3,
)
```

## Parameters

### Distillation-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--trainer` | str | "distillation" | Use "distillation" or "sft" with --use-distillation |
| `--use-distillation` | bool | False | Enable distillation in SFT trainer |
| `--teacher-model` | str | required | Teacher model name or path |
| `--teacher-prompt-template` | str | "{input}" | Template for teacher prompts |
| `--student-prompt-template` | str | "{input}" | Template for student prompts |
| `--distill-temperature` | float | 3.0 | Temperature for softening (2.0-4.0 typical) |
| `--distill-alpha` | float | 0.7 | KL loss weight (0=CE only, 1=KL only) |
| `--distill-max-teacher-length` | int | 512 | Max length for teacher outputs |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | 3 | Training epochs |
| `--batch-size` | int | 4 | Batch size (keep small for quality) |
| `--lr` | float | 5e-5 | Learning rate |
| `--warmup-ratio` | float | 0.1 | Warmup ratio |
| `--gradient-accumulation` | int | 4 | Gradient accumulation |

## How It Works

### The Distillation Process

1. **Teacher Generation**: Teacher model generates outputs with complex prompts
2. **Student Training**: Student learns to match teacher's distribution
3. **Dual Loss**: Combines KL divergence and cross-entropy

```python
# Simplified distillation loss
teacher_logits = teacher_model(teacher_prompt + input)
student_logits = student_model(student_prompt + input)

# KL divergence loss
kl_loss = KL_div(
    softmax(student_logits / temperature),
    softmax(teacher_logits / temperature)
) * temperature²

# Cross-entropy loss
ce_loss = CrossEntropy(student_logits, targets)

# Combined loss
total_loss = alpha * kl_loss + (1 - alpha) * ce_loss
```

### Temperature Effects

- **Low (1.0-2.0)**: Sharp distributions, exact matching
- **Medium (2.0-3.0)**: Balanced softness (recommended)
- **High (3.0-5.0)**: Very soft, learns general patterns

### Alpha Balancing

- **α = 0.0**: Pure cross-entropy (no distillation)
- **α = 0.5**: Balanced KL and CE
- **α = 0.7**: More weight on KL (recommended)
- **α = 1.0**: Pure KL divergence

## Examples

### Example 1: Basic Prompt Internalization

```python
from autotrain.trainers.clm import train
from autotrain.trainers.clm.params import LLMTrainingParams

# Teacher uses complex prompt, student learns to work without it
config = LLMTrainingParams(
    model="gpt2",
    trainer="distillation",

    teacher_model="gpt2-xl",
    teacher_prompt_template="""You are a helpful AI assistant with expertise in many domains.
    Always provide detailed, accurate, and well-structured answers.
    Break down complex topics into understandable parts.

    User Question: {input}

    Assistant Response:""",

    student_prompt_template="Q: {input}\nA:",  # Minimal prompt

    distill_temperature=3.0,
    distill_alpha=0.75,

    data_path="./questions.json",
    text_column="question",
    project_name="internalized-prompts",
    epochs=5,
    batch_size=4,
)

model = train(config)

# Test: Student now produces expert-quality answers with minimal prompting
```

### Example 2: Large to Small Model Distillation

```python
# Distill Llama-70B to Llama-7B
config = LLMTrainingParams(
    model="meta-llama/Llama-2-7b-hf",
    trainer="distillation",

    teacher_model="meta-llama/Llama-2-70b-hf",
    teacher_prompt_template="{input}",  # Same prompt
    student_prompt_template="{input}",  # Focus on knowledge transfer

    # Higher temperature for large model distillation
    distill_temperature=4.0,
    distill_alpha=0.8,

    # PEFT for efficiency
    peft=True,
    lora_r=64,
    lora_alpha=128,
    quantization="int4",

    data_path="./diverse_instructions.json",
    text_column="instruction",
    project_name="llama-7b-distilled",
    epochs=2,
    batch_size=1,
    gradient_accumulation=16,
)

small_model = train(config)
```

### Example 3: Chain-of-Thought Distillation

```python
# Internalize reasoning patterns
config = LLMTrainingParams(
    model="gpt2",
    trainer="distillation",

    teacher_model="gpt-3.5-turbo",  # Or any CoT model
    teacher_prompt_template="""Let's solve this step-by-step.
    First, I'll break down the problem.
    Then, I'll work through each part.
    Finally, I'll provide the answer.

    Problem: {input}

    Step-by-step solution:""",

    student_prompt_template="Problem: {input}\nSolution:",

    distill_temperature=2.5,
    distill_alpha=0.7,

    data_path="./math_problems.json",
    text_column="problem",
    project_name="cot-distilled",
    epochs=5,
)

reasoning_model = train(config)
```

### Example 4: Multi-Teacher Distillation

```python
# Advanced: Distill from multiple teachers
teachers = [
    ("gpt2-xl", "You are a creative writer. {input}"),
    ("bert-large", "Analyze this text: {input}"),
    ("t5-large", "Summarize: {input}"),
]

# Train on outputs from all teachers
combined_data = []
for teacher_model, teacher_prompt in teachers:
    # Generate teacher outputs
    teacher_outputs = generate_with_teacher(
        teacher_model,
        teacher_prompt,
        queries
    )
    combined_data.extend(teacher_outputs)

# Distill combined knowledge
config = LLMTrainingParams(
    model="gpt2",
    trainer="sft",  # Use SFT with prepared data
    use_distillation=True,
    data_path="./combined_teacher_outputs.json",
    project_name="multi-teacher-distilled",
)
```

## Best Practices

### 1. Teacher Selection

**Choose Appropriate Teachers**
```python
# Good: Larger model of same family
teacher = "gpt2-xl"
student = "gpt2"

# Good: Specialized model
teacher = "code-davinci-002"  # Code expert
student = "gpt2"  # General model

# Bad: Very different architectures
teacher = "t5-base"  # Encoder-decoder
student = "gpt2"  # Decoder-only
```

### 2. Prompt Engineering

**Effective Prompt Templates**
```python
# Detailed teacher prompt
teacher_template = """### Role: Expert {role}
### Context: {context}
### Instructions: Provide detailed, thoughtful response
### Input: {input}
### Response:"""

# Minimal student prompt
student_template = "{input}:"

# This maximizes compression ratio
```

### 3. Temperature Tuning

```python
# Find optimal temperature
temperatures = [2.0, 2.5, 3.0, 3.5, 4.0]
results = {}

for temp in temperatures:
    config = LLMTrainingParams(
        distill_temperature=temp,
        project_name=f"distill_temp_{temp}",
        # ... other params
    )
    model = train(config)
    results[temp] = evaluate(model)

optimal_temp = max(results, key=results.get)
```

### 4. Data Diversity

```python
# Ensure diverse training data
def prepare_distillation_data(queries):
    diverse_data = []

    # Include different types
    categories = categorize_queries(queries)
    for category in categories:
        # Sample evenly
        samples = random.sample(
            category,
            min(1000, len(category))
        )
        diverse_data.extend(samples)

    # Shuffle
    random.shuffle(diverse_data)
    return diverse_data
```

### 5. Progressive Distillation

```python
# Gradually reduce prompt complexity
stages = [
    (1, "Very detailed teacher prompt..."),
    (2, "Moderate teacher prompt..."),
    (3, "Simple teacher prompt..."),
]

model = base_model
for stage, teacher_prompt in stages:
    config = LLMTrainingParams(
        model=model,
        teacher_prompt_template=teacher_prompt,
        project_name=f"stage_{stage}",
        epochs=2,
        # ...
    )
    model = train(config)
```

## Advanced Usage

### Custom Distillation Loss

```python
from autotrain.trainers.clm.train_clm_distill import PromptDistillationTrainer

class CustomDistillationTrainer(PromptDistillationTrainer):
    def compute_loss(self, model, inputs):
        # Get teacher and student outputs
        teacher_outputs = self.teacher_model(**teacher_inputs)
        student_outputs = model(**student_inputs)

        # Custom loss computation
        kl_loss = self.compute_kl_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            temperature=self.config.distill_temperature
        )

        # Add custom regularization
        reg_loss = self.compute_regularization(student_outputs)

        # Combine losses
        total_loss = (
            self.config.distill_alpha * kl_loss +
            (1 - self.config.distill_alpha) * ce_loss +
            0.1 * reg_loss
        )

        return total_loss
```

### Online Distillation

```python
# Generate teacher outputs on-the-fly
class OnlineDistillationTrainer(PromptDistillationTrainer):
    def generate_teacher_outputs(self, batch):
        # Real-time teacher generation
        with torch.no_grad():
            teacher_inputs = self.prepare_teacher_inputs(batch)
            teacher_outputs = self.teacher_model.generate(
                **teacher_inputs,
                max_new_tokens=self.config.distill_max_teacher_length,
                temperature=0.8,
                do_sample=True,
            )
        return teacher_outputs
```

### Distillation with Reinforcement

```python
# Combine with reward signals
def distill_with_rewards(config, reward_model):
    # Standard distillation
    distilled_model = train(config)

    # Fine-tune with rewards
    from autotrain.trainers.rl import PPOTrainer

    ppo_config = PPOConfig(
        model=distilled_model,
        reward_model=reward_model,
        # ...
    )

    final_model = PPOTrainer(ppo_config).train()
    return final_model
```

## Troubleshooting

### Issue: Student Not Learning

```python
# Solutions:
# 1. Increase temperature (softer targets)
config.distill_temperature = 4.0

# 2. Adjust alpha (more CE loss)
config.distill_alpha = 0.5

# 3. Simplify teacher prompt gradually
# Start with student_prompt similar to teacher_prompt
```

### Issue: Quality Degradation

```python
# 1. Use higher quality teacher
config.teacher_model = "larger-better-model"

# 2. Increase training data
# More examples for better generalization

# 3. Lower temperature (more precise matching)
config.distill_temperature = 2.0
```

### Issue: OOM with Teacher

```python
# 1. Use gradient checkpointing
config.gradient_checkpointing = True

# 2. Generate teacher outputs offline
# Pre-compute and save teacher outputs

# 3. Use smaller teacher
config.teacher_model = "smaller-teacher"

# 4. Reduce batch size
config.batch_size = 1
config.gradient_accumulation = 16
```

## Comparison with Other Methods

| Method | Purpose | Teacher Needed | Inference Cost | Training Complexity |
|--------|---------|----------------|----------------|-------------------|
| **Distillation** | Prompt internalization | Yes | Low | Moderate |
| **SFT** | Task learning | No | Medium | Low |
| **DPO** | Preference learning | No | Medium | Moderate |
| **Fine-tuning** | Adaptation | No | High | Low |

## Evaluation Metrics

```python
def evaluate_distillation(student_model, teacher_model, test_data):
    metrics = {}

    # Output quality comparison
    student_outputs = generate(student_model, test_data, simple_prompt)
    teacher_outputs = generate(teacher_model, test_data, complex_prompt)

    # Similarity metrics
    metrics['output_similarity'] = compute_similarity(
        student_outputs,
        teacher_outputs
    )

    # Inference speed gain
    student_time = measure_inference_time(student_model, simple_prompt)
    teacher_time = measure_inference_time(teacher_model, complex_prompt)
    metrics['speedup'] = teacher_time / student_time

    # Token reduction
    metrics['prompt_compression'] = len(complex_prompt) / len(simple_prompt)

    return metrics
```

## Next Steps

- [SFT Training](./SFT.md) - Standard fine-tuning
- [DPO Training](./DPO.md) - Preference optimization
- [Default Training](./Default.md) - Basic language modeling
- [PPO Training](./PPO.md) - Reinforcement learning