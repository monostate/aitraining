# AITraining Parameter Compatibility Guide

## Overview

AITraining supports multiple command types with different parameters. This guide explains which parameters apply to which commands and trainers.

## Command Types

AITraining supports the following command types:

### Language Model Commands (`aitraining llm`)
- Supports multiple trainers: generic, sft, dpo, orpo, ppo, reward
- ~111 total parameters available
- Detailed compatibility matrix below

### Classification Commands
- `aitraining text-classification` - ~30 parameters
- `aitraining image-classification` - ~29 parameters
- `aitraining token-classification` - ~25 parameters

### Sequence-to-Sequence Commands
- `aitraining seq2seq` - ~28 parameters
- `aitraining extractive-qa` - ~26 parameters

### Embedding/Similarity Commands
- `aitraining sentence-transformers` - ~24 parameters

### Vision Commands
- `aitraining image-regression` - ~25 parameters
- `aitraining object-detection` - ~31 parameters
- `aitraining vlm` - ~35 parameters

### Regression Commands
- `aitraining text-regression` - ~26 parameters
- `aitraining tabular` - ~22 parameters

---

## LLM Command Trainers

- **generic** (default) - Basic causal language modeling
- **sft** - Supervised fine-tuning with chat templates
- **dpo** - Direct Preference Optimization
- **orpo** - Odds Ratio Preference Optimization
- **reward** - Train reward model for RLHF (produces scorer, not generator)
- **ppo** - Proximal Policy Optimization (requires reward model)

## Cross-Cutting Features

These are NOT trainers, but flags that modify training behavior:

- **use_distillation** - Apply knowledge distillation (works with: generic, sft, dpo, orpo)
- **use_sweep** - Hyperparameter search (works with: all trainers)
- **use_enhanced_eval** - Advanced evaluation metrics (works with: all trainers)

## Parameter Compatibility Matrix

### Core Parameters (All Trainers)

| Parameter | generic | sft | dpo | orpo | reward | ppo |
|-----------|---------|-----|-----|------|--------|-----|
| model | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| data_path | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| lr | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| epochs | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| batch_size | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| gradient_accumulation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| optimizer | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| scheduler | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| warmup_ratio | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| weight_decay | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| seed | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| mixed_precision | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| block_size | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| peft | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| lora_r | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| lora_alpha | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| lora_dropout | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Trainer-Specific Parameters

#### DPO & ORPO Only
| Parameter | generic | sft | dpo | orpo | reward | ppo |
|-----------|---------|-----|-----|------|--------|-----|
| model_ref | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| dpo_beta | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| max_prompt_length | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| max_completion_length | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| prompt_text_column | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| rejected_text_column | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ |

#### PPO Only (RL Parameters)
| Parameter | generic | sft | dpo | orpo | reward | ppo |
|-----------|---------|-----|-----|------|--------|-----|
| rl_reward_model_path | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| rl_gamma | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| rl_gae_lambda | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| rl_kl_coef | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| rl_value_loss_coef | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| rl_clip_range | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| rl_num_ppo_epochs | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| rl_chunk_size | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| rl_mini_batch_size | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |

#### Distillation (Cross-Cutting)
When `use_distillation=True`:

| Parameter | generic | sft | dpo | orpo | reward | ppo |
|-----------|---------|-----|-----|------|--------|-----|
| teacher_model | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| teacher_prompt_template | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| student_prompt_template | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| distill_temperature | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| distill_alpha | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |

**Note:** Distillation not supported with reward/PPO trainers.

#### Sweep (Cross-Cutting)
When `use_sweep=True`:

| Parameter | All Trainers |
|-----------|--------------|
| sweep_backend | ✓ |
| sweep_n_trials | ✓ |
| sweep_metric | ✓ |
| sweep_direction | ✓ |
| sweep_params | ✓ |

#### Enhanced Evaluation (Cross-Cutting)
When `use_enhanced_eval=True`:

| Parameter | All Trainers |
|-----------|--------------|
| eval_metrics | ✓ |
| eval_dataset_path | ✓ |
| eval_batch_size | ✓ |
| eval_save_predictions | ✓ |
| eval_benchmark | ✓ |

## Common Confusion Points

### 1. Reward Model is Different
**⚠️  IMPORTANT:** The reward model trainer produces a `AutoModelForSequenceClassification` (scorer), not a `AutoModelForCausalLM` (generator).

- **Cannot** generate text
- **Returns** a single score per input
- **Used with** PPO trainer for RLHF

### 2. Distillation is Not a Trainer
`trainer=distillation` is being removed. Use `use_distillation=True` with a base trainer instead.

**Old (exclusive):**
```bash
--trainer distillation
```

**New (compositional):**
```bash
--trainer sft --use-distillation=True --teacher-model gpt-4
```

### 3. PPO Requires Reward Model

PPO will fail if you don't provide a reward model:

```bash
# Step 1: Train reward model
aitraining llm --trainer reward --data-path preference_data

# Step 2: Use with PPO
aitraining llm --trainer ppo --rl-reward-model-path /path/to/reward/model
```

### 4. Model Reference vs Reward Model

- **model_ref** → Used by DPO/ORPO as reference policy (generator model)
- **rl_reward_model_path** → Used by PPO as reward scorer (scorer model)

These are different concepts!

## Parameter Validation

AutoTrain will warn you if you set parameters that don't apply to your trainer:

```bash
# Example: Setting RL params for SFT
aitraining llm --trainer sft --rl-gamma 0.95
# Warning: rl_gamma only applies to PPO, will be ignored
```

## Common Parameters Across All Commands

These parameters are available for ALL aitraining commands:

### Core Training Parameters
- `--train`: Execute training mode
- `--model MODEL`: Base model to fine-tune
- `--project-name PROJECT_NAME`: Output project name
- `--data-path DATA_PATH`: Path to training data
- `--lr LR`: Learning rate
- `--epochs EPOCHS`: Number of training epochs
- `--batch-size BATCH_SIZE`: Training batch size
- `--seed SEED`: Random seed for reproducibility
- `--backend BACKEND`: Training backend (local, spaces, endpoint)

### Data Configuration
- `--train-split TRAIN_SPLIT`: Name of training split (default: "train")
- `--valid-split VALID_SPLIT`: Name of validation split
- `--max-samples MAX_SAMPLES`: Maximum samples to use

### Optimization
- `--optimizer OPTIMIZER`: Optimizer type (adamw_torch, sgd, etc.)
- `--scheduler SCHEDULER`: LR scheduler (linear, cosine, constant)
- `--warmup-ratio WARMUP_RATIO`: Warmup ratio
- `--warmup-steps WARMUP_STEPS`: Warmup steps
- `--gradient-accumulation GRADIENT_ACCUMULATION`: Gradient accumulation steps
- `--weight-decay WEIGHT_DECAY`: Weight decay coefficient
- `--max-grad-norm MAX_GRAD_NORM`: Maximum gradient norm for clipping

### Performance
- `--mixed-precision MIXED_PRECISION`: Mixed precision training (fp16, bf16)
- `--auto-find-batch-size`: Automatically find optimal batch size
- `--disable-gradient-checkpointing`: Disable gradient checkpointing

### Logging & Saving
- `--log LOG`: Logging backend (tensorboard, wandb, none)
- `--logging-steps LOGGING_STEPS`: Steps between logging
- `--save-strategy SAVE_STRATEGY`: Save strategy (epoch, steps, no)
- `--save-steps SAVE_STEPS`: Steps between saves
- `--save-total-limit SAVE_TOTAL_LIMIT`: Max checkpoints to keep

### HuggingFace Hub
- `--push-to-hub`: Push model to HuggingFace Hub
- `--username USERNAME`: HuggingFace username
- `--token TOKEN`: HuggingFace API token
- `--repo-id REPO_ID`: Repository ID

## Command-Specific Parameters

Each command type has additional specific parameters:

### Text Classification Specific
- `--text-column TEXT_COLUMN`: Column containing text data
- `--target-column TARGET_COLUMN`: Column containing labels

### Image Classification Specific
- `--image-column IMAGE_COLUMN`: Column containing image paths
- `--target-column TARGET_COLUMN`: Column containing labels
- `--image-size IMAGE_SIZE`: Target image size
- `--augment`: Enable data augmentation

### Seq2Seq Specific
- `--text-column TEXT_COLUMN`: Source text column
- `--target-column TARGET_COLUMN`: Target text column
- `--max-seq-length MAX_SEQ_LENGTH`: Maximum sequence length
- `--max-target-length MAX_TARGET_LENGTH`: Maximum target length

### Tabular Specific
- `--task TASK`: Task type (classification, regression)
- `--categorical-columns`: Categorical feature columns
- `--numerical-columns`: Numerical feature columns
- `--num-trials NUM_TRIALS`: Hyperparameter optimization trials

## Questions?

- GitHub Issues: https://github.com/huggingface/autotrain-advanced/issues
- Documentation: https://huggingface.co/docs/autotrain