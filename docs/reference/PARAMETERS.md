# AutoTrain Trainer-Specific Parameter Guide

This guide provides detailed information about each trainer and its compatible parameters.

## Table of Contents
1. [Generic Trainer](#generic-trainer)
2. [SFT Trainer](#sft-trainer)
3. [DPO Trainer](#dpo-trainer)
4. [ORPO Trainer](#orpo-trainer)
5. [Reward Trainer](#reward-trainer)
6. [PPO Trainer](#ppo-trainer)

---

## Generic Trainer

### Purpose
Basic causal language modeling for text generation. Use this for standard language model training without specific alignment objectives.

### Required Parameters
```bash
--model <model_name>           # Base model to fine-tune
--data-path <path>             # Path to training data
--lr <learning_rate>           # Learning rate
--epochs <num_epochs>          # Number of training epochs
--batch-size <size>            # Training batch size
```

### Optional Parameters
```bash
# Training Configuration
--gradient-accumulation <steps>  # Gradient accumulation steps (default: 1)
--optimizer <name>                # Optimizer: adamw_torch, adamw_hf, sgd, adafactor
--scheduler <name>                # LR scheduler: linear, cosine, constant
--warmup-ratio <ratio>           # Warmup ratio (default: 0.1)
--weight-decay <value>           # Weight decay (default: 0.01)
--seed <value>                   # Random seed (default: 42)
--mixed-precision <type>         # fp16, bf16, or fp32
--block-size <size>              # Max sequence length (default: 1024)

# PEFT/LoRA Configuration
--peft                           # Enable LoRA
--lora-r <rank>                  # LoRA rank (default: 16)
--lora-alpha <alpha>             # LoRA alpha (default: 32)
--lora-dropout <dropout>         # LoRA dropout (default: 0.05)

# Cross-Cutting Features
--use-distillation               # Enable knowledge distillation
--use-sweep                      # Enable hyperparameter search
--use-enhanced-eval              # Enable advanced evaluation
```

### Incompatible Parameters
```bash
# These will be ignored with warnings:
--model-ref                      # Only for DPO/ORPO
--dpo-beta                       # Only for DPO
--rl-*                          # Only for PPO
--max-prompt-length             # Only for DPO/ORPO
--rejected-text-column          # Only for preference-based trainers
```

### Example Usage
```bash
# Basic training
autotrain llm \
  --trainer generic \
  --model meta-llama/Llama-2-7b-hf \
  --data-path ./data/train.jsonl \
  --lr 2e-5 \
  --epochs 3 \
  --batch-size 4

# With LoRA and distillation
autotrain llm \
  --trainer generic \
  --model meta-llama/Llama-2-7b-hf \
  --data-path ./data/train.jsonl \
  --lr 2e-4 \
  --epochs 3 \
  --batch-size 8 \
  --peft \
  --lora-r 32 \
  --use-distillation \
  --teacher-model meta-llama/Llama-2-13b-hf
```

---

## SFT Trainer

### Purpose
Supervised fine-tuning with chat templates for instruction-following and conversational AI.

### Required Parameters
```bash
--model <model_name>           # Base model to fine-tune
--data-path <path>             # Path to instruction/chat data
--lr <learning_rate>           # Learning rate
--epochs <num_epochs>          # Number of training epochs
--batch-size <size>            # Training batch size
```

### Optional Parameters
```bash
# All parameters from Generic trainer, plus:
--chat-template <template>      # Chat template format
--add-eos-token                # Add EOS token to responses
--max-seq-length <length>       # Max sequence length for chat

# Cross-Cutting Features
--use-distillation               # Enable knowledge distillation
--use-sweep                      # Enable hyperparameter search
--use-enhanced-eval              # Enable advanced evaluation
```

### Incompatible Parameters
```bash
# These will be ignored with warnings:
--model-ref                      # Only for DPO/ORPO
--dpo-beta                       # Only for DPO
--rl-*                          # Only for PPO
--max-prompt-length             # Only for DPO/ORPO
--rejected-text-column          # Only for preference-based trainers
```

### Example Usage
```bash
# Basic SFT training
autotrain llm \
  --trainer sft \
  --model mistralai/Mistral-7B-v0.1 \
  --data-path ./data/instructions.jsonl \
  --lr 2e-5 \
  --epochs 3 \
  --batch-size 4 \
  --chat-template chatml

# With LoRA and enhanced evaluation
autotrain llm \
  --trainer sft \
  --model mistralai/Mistral-7B-v0.1 \
  --data-path ./data/instructions.jsonl \
  --lr 2e-4 \
  --epochs 3 \
  --batch-size 8 \
  --peft \
  --lora-r 64 \
  --use-enhanced-eval \
  --eval-metrics '["bleu", "rouge", "bertscore"]' \
  --eval-dataset-path ./data/test.jsonl
```

---

## DPO Trainer

### Purpose
Direct Preference Optimization for aligning models with human preferences without reward modeling.

### Required Parameters
```bash
--model <model_name>             # Base model to fine-tune
--data-path <path>               # Path to preference data
--lr <learning_rate>             # Learning rate
--epochs <num_epochs>            # Number of training epochs
--batch-size <size>              # Training batch size
--prompt-text-column <column>    # Column name for prompts
--rejected-text-column <column>  # Column name for rejected responses
```

### Optional Parameters
```bash
# All core parameters from Generic trainer, plus:
--model-ref <model_name>         # Reference model (default: base model)
--dpo-beta <value>               # DPO beta parameter (default: 0.1)
--max-prompt-length <length>     # Max prompt length (default: 512)
--max-completion-length <length> # Max completion length (default: 1024)

# Cross-Cutting Features
--use-distillation               # Enable knowledge distillation
--use-sweep                      # Enable hyperparameter search
--use-enhanced-eval              # Enable advanced evaluation
```

### Incompatible Parameters
```bash
# These will be ignored with warnings:
--rl-*                          # Only for PPO
```

### Example Usage
```bash
# Basic DPO training
autotrain llm \
  --trainer dpo \
  --model meta-llama/Llama-2-7b-hf \
  --data-path ./data/preferences.jsonl \
  --lr 5e-7 \
  --epochs 2 \
  --batch-size 2 \
  --prompt-text-column prompt \
  --rejected-text-column rejected \
  --dpo-beta 0.1

# With reference model and LoRA
autotrain llm \
  --trainer dpo \
  --model meta-llama/Llama-2-7b-hf \
  --model-ref meta-llama/Llama-2-7b-hf \
  --data-path ./data/preferences.jsonl \
  --lr 5e-7 \
  --epochs 2 \
  --batch-size 4 \
  --prompt-text-column prompt \
  --rejected-text-column rejected \
  --dpo-beta 0.2 \
  --peft \
  --lora-r 32
```

---

## ORPO Trainer

### Purpose
Odds Ratio Preference Optimization - an alternative to DPO with different optimization dynamics.

### Required Parameters
```bash
--model <model_name>             # Base model to fine-tune
--data-path <path>               # Path to preference data
--lr <learning_rate>             # Learning rate
--epochs <num_epochs>            # Number of training epochs
--batch-size <size>              # Training batch size
--prompt-text-column <column>    # Column name for prompts
--rejected-text-column <column>  # Column name for rejected responses
```

### Optional Parameters
```bash
# All core parameters from Generic trainer, plus:
--model-ref <model_name>         # Reference model (default: base model)
--max-prompt-length <length>     # Max prompt length (default: 512)
--max-completion-length <length> # Max completion length (default: 1024)
--orpo-alpha <value>             # ORPO alpha parameter (default: 1.0)

# Cross-Cutting Features
--use-distillation               # Enable knowledge distillation
--use-sweep                      # Enable hyperparameter search
--use-enhanced-eval              # Enable advanced evaluation
```

### Incompatible Parameters
```bash
# These will be ignored with warnings:
--dpo-beta                       # Only for DPO
--rl-*                          # Only for PPO
```

### Example Usage
```bash
# Basic ORPO training
autotrain llm \
  --trainer orpo \
  --model mistralai/Mistral-7B-v0.1 \
  --data-path ./data/preferences.jsonl \
  --lr 5e-7 \
  --epochs 2 \
  --batch-size 2 \
  --prompt-text-column prompt \
  --rejected-text-column rejected

# With sweep for hyperparameter search
autotrain llm \
  --trainer orpo \
  --model mistralai/Mistral-7B-v0.1 \
  --data-path ./data/preferences.jsonl \
  --lr 5e-7 \
  --epochs 2 \
  --batch-size 4 \
  --prompt-text-column prompt \
  --rejected-text-column rejected \
  --use-sweep \
  --sweep-n-trials 10 \
  --sweep-metric eval_loss \
  --sweep-direction minimize
```

---

## Reward Trainer

### Purpose
Train a reward model for RLHF. **Note:** This produces a classifier model (`AutoModelForSequenceClassification`), not a text generator.

### Required Parameters
```bash
--model <model_name>             # Base model to train as reward model
--data-path <path>               # Path to preference data
--lr <learning_rate>             # Learning rate
--epochs <num_epochs>            # Number of training epochs
--batch-size <size>              # Training batch size
--rejected-text-column <column>  # Column name for rejected responses
```

### Optional Parameters
```bash
# All core parameters from Generic trainer, plus:
--num-labels <num>               # Number of output labels (default: 1)
--label-smoothing <value>        # Label smoothing (default: 0.0)

# Cross-Cutting Features (Limited)
--use-sweep                      # Enable hyperparameter search
--use-enhanced-eval              # Enable advanced evaluation
```

### Incompatible Parameters
```bash
# These will be ignored with warnings:
--model-ref                      # Not applicable for reward models
--dpo-beta                       # Only for DPO
--rl-*                          # Only for PPO
--use-distillation              # Not supported for reward models
--teacher-model                 # Not supported for reward models
```

### Example Usage
```bash
# Basic reward model training
autotrain llm \
  --trainer reward \
  --model bert-base-uncased \
  --data-path ./data/preferences.jsonl \
  --lr 2e-5 \
  --epochs 3 \
  --batch-size 16 \
  --rejected-text-column rejected

# With LoRA for efficient training
autotrain llm \
  --trainer reward \
  --model roberta-large \
  --data-path ./data/preferences.jsonl \
  --lr 2e-5 \
  --epochs 3 \
  --batch-size 32 \
  --rejected-text-column rejected \
  --peft \
  --lora-r 8
```

### Output Model Type
⚠️ **Important:** The output is a scorer model, not a generator:
- **Model Type:** `AutoModelForSequenceClassification`
- **Output:** Single score per input
- **Cannot:** Generate text
- **Use Case:** Score generations for PPO training

---

## PPO Trainer

### Purpose
Proximal Policy Optimization for RLHF using a trained reward model.

### Required Parameters
```bash
--model <model_name>             # Base model to fine-tune
--data-path <path>               # Path to prompt data
--lr <learning_rate>             # Learning rate
--epochs <num_epochs>            # Number of training epochs
--batch-size <size>              # Training batch size
--rl-reward-model-path <path>   # Path to trained reward model
```

### Optional Parameters
```bash
# All core parameters from Generic trainer, plus:

# RL-specific parameters
--rl-gamma <value>               # Discount factor (default: 0.99)
--rl-gae-lambda <value>          # GAE lambda (default: 0.95)
--rl-kl-coef <value>             # KL penalty coefficient (default: 0.2)
--rl-value-loss-coef <value>    # Value loss coefficient (default: 0.5)
--rl-clip-range <value>          # PPO clip range (default: 0.2)
--rl-num-ppo-epochs <num>        # PPO epochs per batch (default: 4)
--rl-chunk-size <size>           # Chunk size for PPO (default: 128)
--rl-mini-batch-size <size>      # Mini-batch size for PPO (default: 32)

# Cross-Cutting Features (Limited)
--use-sweep                      # Enable hyperparameter search
--use-enhanced-eval              # Enable advanced evaluation
```

### Incompatible Parameters
```bash
# These will be ignored with warnings:
--model-ref                      # Not used in PPO
--dpo-beta                       # Only for DPO
--max-prompt-length             # Only for DPO/ORPO
--rejected-text-column          # Not used in PPO
--use-distillation              # Not supported with PPO
--teacher-model                 # Not supported with PPO
```

### Example Usage
```bash
# Step 1: Train reward model (see Reward Trainer section)
autotrain llm \
  --trainer reward \
  --model bert-base-uncased \
  --data-path ./data/preferences.jsonl \
  --lr 2e-5 \
  --epochs 3 \
  --batch-size 16 \
  --output-dir ./reward_model

# Step 2: PPO training
autotrain llm \
  --trainer ppo \
  --model meta-llama/Llama-2-7b-hf \
  --data-path ./data/prompts.jsonl \
  --lr 1e-5 \
  --epochs 1 \
  --batch-size 4 \
  --rl-reward-model-path ./reward_model \
  --rl-gamma 0.99 \
  --rl-kl-coef 0.2

# With LoRA for efficient training
autotrain llm \
  --trainer ppo \
  --model meta-llama/Llama-2-7b-hf \
  --data-path ./data/prompts.jsonl \
  --lr 1e-5 \
  --epochs 1 \
  --batch-size 8 \
  --rl-reward-model-path ./reward_model \
  --rl-gamma 0.99 \
  --rl-kl-coef 0.1 \
  --rl-num-ppo-epochs 4 \
  --peft \
  --lora-r 32
```

### Workflow
1. **Train Reward Model:** Use the reward trainer to create a scoring model
2. **Prepare Prompts:** Create a dataset of prompts for PPO training
3. **Run PPO:** Use the trained reward model to guide policy optimization

---

## Cross-Cutting Features

### Knowledge Distillation
Available for: generic, sft, dpo, orpo

```bash
--use-distillation \
--teacher-model <model_name> \
--teacher-prompt-template <template> \
--student-prompt-template <template> \
--distill-temperature <temp> \
--distill-alpha <alpha>
```

### Hyperparameter Sweep
Available for: all trainers

```bash
--use-sweep \
--sweep-backend <wandb|optuna> \
--sweep-n-trials <num> \
--sweep-metric <metric> \
--sweep-direction <maximize|minimize> \
--sweep-params '<json_config>'
```

### Enhanced Evaluation
Available for: all trainers

```bash
--use-enhanced-eval \
--eval-metrics '["perplexity", "bleu", "rouge"]' \
--eval-dataset-path <path> \
--eval-batch-size <size> \
--eval-save-predictions \
--eval-benchmark <benchmark_name>
```

## Migration Notes

### From Old Advanced Mode
If you were using `--advanced` flag:
- Remove `--advanced` flag
- All advanced parameters are now available by default
- Trainer-specific validation will warn about incompatible parameters

### From Distillation Trainer
If you were using `--trainer distillation`:
```bash
# Old way (removed)
--trainer distillation --teacher-model gpt-4

# New way (compositional)
--trainer sft --use-distillation --teacher-model gpt-4
```

## Parameter Validation

AutoTrain automatically validates parameters and provides warnings:

```bash
# Example: Using DPO parameter with SFT trainer
autotrain llm --trainer sft --dpo-beta 0.1
# Warning: dpo_beta only applies to DPO trainer, will be ignored

# Example: Using distillation with PPO
autotrain llm --trainer ppo --use-distillation
# Error: Distillation is not supported with PPO trainer
```

## Best Practices

### 1. Start Simple
Begin with core parameters, then add advanced features:
```bash
# Start here
--trainer sft --model <model> --data-path <data> --lr 2e-5 --epochs 3

# Then add features
--peft --lora-r 32
--use-enhanced-eval
--use-distillation
```

### 2. Use Appropriate Batch Sizes
- **Small models (<7B):** batch_size 8-16
- **Medium models (7B-13B):** batch_size 4-8
- **Large models (>13B):** batch_size 1-4
- Use gradient_accumulation to simulate larger batches

### 3. Choose the Right Trainer
- **generic:** General text generation, completion tasks
- **sft:** Instruction following, chatbots
- **dpo/orpo:** Alignment with preferences
- **reward + ppo:** Complex RLHF workflows

### 4. Monitor Memory Usage
```bash
# Use mixed precision for memory efficiency
--mixed-precision bf16

# Use LoRA for large models
--peft --lora-r 32

# Reduce sequence length if needed
--block-size 512
```

## Troubleshooting

### Common Issues

1. **"Parameter X not recognized"**
   - Check if parameter applies to your chosen trainer
   - Review compatibility matrix in PARAMETER_GUIDE.md

2. **"Reward model not found"**
   - Ensure reward model training completed successfully
   - Verify path to reward model checkpoint

3. **"Out of memory"**
   - Reduce batch_size
   - Enable mixed_precision
   - Use PEFT/LoRA
   - Reduce block_size

4. **"Validation loss increasing"**
   - Lower learning rate
   - Increase warmup_ratio
   - Check for data quality issues

## Support

- **Documentation:** https://huggingface.co/docs/autotrain
- **GitHub Issues:** https://github.com/huggingface/autotrain-advanced/issues
- **Community Forum:** https://discuss.huggingface.co/c/autotrain