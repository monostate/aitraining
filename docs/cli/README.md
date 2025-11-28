# AITraining CLI Documentation

## Overview
AITraining (formerly AutoTrain) provides a comprehensive command-line interface for training various types of models including language models, text classifiers, image models, and more.

## Command Structure
```bash
aitraining <command> [OPTIONS]
```

## Available Commands

AITraining supports multiple model types and training tasks:

- `aitraining tui` - Interactive Terminal User Interface for parameter configuration
- `aitraining llm` - Language model training (SFT, DPO, ORPO, PPO, reward)
- `aitraining text-classification` - Text classification models
- `aitraining text-regression` - Text regression models
- `aitraining token-classification` - NER/Token classification
- `aitraining seq2seq` - Sequence-to-sequence models
- `aitraining sentence-transformers` - Sentence embedding models
- `aitraining image-classification` - Image classification
- `aitraining image-regression` - Image regression
- `aitraining object-detection` - Object detection models
- `aitraining extractive-qa` - Question answering
- `aitraining vlm` - Vision-language models
- `aitraining tabular` - Tabular data models

---

## Interactive TUI

AITraining now includes an interactive Terminal User Interface (TUI) for easier parameter configuration and training management.

### Quick Start
```bash
aitraining tui
```

The TUI provides:
- 🎯 Interactive parameter editing with validation
- 📁 Organized parameter groups for easy navigation
- 🔍 Search and filter parameters
- 💾 Save/load configuration presets
- 📊 Live training logs
- 🎨 Dark and light themes

For detailed TUI documentation, see [TUI.md](TUI.md).

---

## Interactive CLI Wizard

Running `aitraining` with no subcommand (or passing `--interactive` on any trainer command) launches the multi-trainer wizard:

- **HF token onboarding** – The very first prompt asks for a Hugging Face token so we can list private models/datasets. Press Enter to stay in public-only mode; the token is exported to `HF_TOKEN` for the remainder of the session and can still be edited later under *Hub Integration*.
- **Curated suggestions everywhere** – Dataset selection (Step 3) and model selection (Step 4) now show the same “popular” lists that power the browser UI. Enter a number to auto-fill a suggestion or type any custom path/model ID.
- **Command shortcuts** – Every prompt understands `:back`, `:help`, and `:exit`. You can revisit earlier steps, surface contextual help inline, or cancel gracefully without leaving orphaned state.
- **Restart-friendly steps** – When you go back, previously supplied values are shown as defaults so you can tweak just the fields you care about instead of re-answering everything.
- **W&B-first logging** – The wizard (and generated commands) default `--log wandb`, auto-launching the LEET sidecar when your terminal supports it. Pass `--log none` or `--no-wandb-visualizer` if you want to suppress it.

---


## Language Model Training (`aitraining llm`)

### Complete Parameter List (112 parameters)

#### Core Parameters
- `--data-path STR`: Path to the dataset
- `--model STR`: Model name to be used for training
- `--project-name STR`: Name of the project and output directory

#### Data Configuration
- `--add-eos-token BOOL`: Whether to add an EOS token at the end of sequences
- `--max-samples INT`: Maximum number of samples to use from dataset (for testing/debugging)
- `--prompt-text-column STR`: Column name for the prompt text
- `--rejected-text-column STR`: Column name for the rejected text data
- `--text-column STR`: Column name for the text data
- `--train-split STR`: Configuration for the training data split
- `--valid-split STR`: Configuration for the validation data split

#### Training Configuration
- `--auto-find-batch-size BOOL`: Whether to automatically find the optimal batch size
- `--batch-size INT`: Batch size for training
- `--disable-gradient-checkpointing BOOL`: Whether to disable gradient checkpointing
- `--distributed-backend STR`: Backend to use for distributed training
- `--epochs INT`: Number of training epochs
- `--gradient-accumulation INT`: Number of steps to accumulate gradients before updating
- `--gradient-accumulation-steps INT`
- `--log STR`: Logging method for experiment tracking (default: `wandb`, auto-launches LEET when possible)
- `--logging-steps INT`: Number of steps between logging events
- `--lr FLOAT`: Learning rate for training
- `--max-grad-norm FLOAT`: Maximum norm for gradient clipping
- `--mixed-precision STR`: Type of mixed precision to use (e.g., 'fp16', 'bf16', or None)
- `--optimizer STR`: Optimizer to use for training
- `--scheduler STR`: Learning rate scheduler to use
- `--seed INT`: Random seed for reproducibility
- `--wandb-visualizer BOOL`: Enable W&B visualizer (LEET). Default enabled when log='wandb'.
- `--no-wandb-visualizer`: Disable W&B visualizer (LEET).
- `--warmup-ratio FLOAT`: Proportion of training to perform learning rate warmup
- `--weight-decay FLOAT`: Weight decay to apply to the optimizer

#### Model Configuration
- `--attn-implementation STR`: Attention implementation to use (e.g., 'eager', 'sdpa', 'flash_attention_2')
- `--block-size INT`: Size of the blocks for training, can be a single integer or a list of integers
- `--chat-format STR`
- `--chat-template STR`: Template for chat-based models, options include: None, zephyr, chatml, tokenizer, or any Unsloth template name
- `--auto-convert-dataset BOOL`: Automatically detect and convert dataset format to messages format (alpaca → messages, sharegpt → messages)
- `--use-sharegpt-mapping BOOL`: Use Unsloth's ShareGPT mapping instead of converting to messages (keeps from/value format)
- `--conversation-extension INT`: Merge N single-turn examples into multi-turn conversations (1 = no merging, 2+ = merge)
- `--apply-chat-template BOOL`: Apply chat template after dataset conversion (renders messages to text)
- `--disable-gradient-checkpointing BOOL`: Whether to disable gradient checkpointing
- `--distributed-backend STR`: Backend to use for distributed training
- `--max-completion-length INT`: Maximum length of the completion
- `--max-prompt-length INT`: Maximum length of the prompt
- `--model-max-length INT`: Maximum length of the model input
- `--padding STR`: Side on which to pad sequences (left or right)
- `--trainer STR`: Type of trainer to use
- `--unsloth BOOL`: Whether to use the unsloth library
- `--use-flash-attention-2 BOOL`: Whether to use flash attention version 2

#### PEFT/LoRA Configuration
- `--lora-alpha INT`: Alpha parameter for LoRA
- `--lora-dropout FLOAT`: Dropout rate for LoRA
- `--lora-r INT`: Rank of the LoRA matrices
- `--merge-adapter`: Merge PEFT adapters into base model (easier deployment, larger file size). Default when neither flag is passed.
- `--no-merge-adapter`: Save only PEFT adapters without merging (smaller file size, requires base model at inference time)
- `--packing BOOL`: Pack multiple short sequences into single sequences for efficiency (requires flash_attention_2)
- `--peft BOOL`: Whether to use Parameter-Efficient Fine-Tuning (PEFT)
- `--quantization STR`: Quantization method to use (e.g., 'int4', 'int8', or None)
- `--target-modules STR`: Target modules for quantization or fine-tuning

#### DPO/ORPO Parameters
- `--dpo-beta FLOAT`: Beta parameter for DPO trainer
- `--model-ref STR`: Reference model for DPO trainer

#### PPO/RL Training
- `--rl-chunk-size INT`
- `--rl-clip-range FLOAT`
- `--rl-env-config STR`
- `--rl-env-type STR`
- `--rl-gae-lambda FLOAT`
- `--rl-gamma FLOAT`
- `--rl-kl-coef FLOAT`
- `--rl-mini-batch-size INT`
- `--rl-multi-objective BOOL`
- `--rl-num-ppo-epochs INT`
- `--rl-optimize-device-cache BOOL`
- `--rl-reward-fn STR`
- `--rl-reward-model-path STR`
- `--rl-reward-weights STR`
- `--rl-value-loss-coef FLOAT`

#### Knowledge Distillation
- `--distill-alpha FLOAT`
- `--distill-max-teacher-length INT`
- `--distill-temperature FLOAT`
- `--student-prompt-template STR`
- `--teacher-model STR`
- `--teacher-prompt-template STR`
- `--use-distillation BOOL`

#### Hyperparameter Sweep
- `--sweep-backend STR`
- `--sweep-direction STR`
- `--sweep-metric STR`
- `--sweep-n-trials INT`
- `--sweep-params STR`
- `--use-sweep BOOL`

#### Evaluation Configuration
- `--eval-batch-size INT`
- `--eval-benchmark STR`
- `--eval-dataset-path STR`
- `--eval-metrics STR`
- `--eval-save-predictions BOOL`
- `--eval-strategy STR`: Strategy for evaluation (e.g., 'epoch')
- `--use-enhanced-eval BOOL`

#### Logging & Checkpointing
- `--load-state-from STR`
- `--log STR`: Logging method for experiment tracking (default: `wandb`, auto-launches LEET when possible)
- `--logging-steps INT`: Number of steps between logging events
- `--manual-checkpoint-control BOOL`
- `--save-state-every-n-steps INT`
- `--save-steps INT`: Number of steps between checkpoint saves (when save_strategy='steps')
- `--save-strategy STR`: Strategy for saving checkpoints ('epoch', 'steps', or 'no')
- `--save-total-limit INT`: Maximum number of checkpoints to keep

> **W&B Live View**: When `--log wandb` is active, AutoTrain writes offline metrics to `WANDB_DIR=<project_dir>` and logs the exact replay command `WANDB_DIR="<project_dir>" wandb beta leet "<project_dir>"`. Use `--wandb-visualizer/--no-wandb-visualizer` to control the sidecar process and `--wandb-token` to sync offline runs back to the cloud.

#### HuggingFace Hub & W&B Integration
- `--push-to-hub BOOL`: Whether to push the model to the Hugging Face Hub
- `--token STR`: Hugging Face token for authentication
- `--username STR`: Hugging Face username for authentication
- `--wandb-token STR`: W&B API Token for syncing runs

> The interactive wizard now prompts for a Hugging Face token before trainer selection so private models/datasets can be listed. Press Enter to skip (public assets only) or supply a token to export it to `HF_TOKEN` for the session.

#### Advanced/Research Features
- `--custom-loss STR`
- `--custom-loss-weights STR`
- `--custom-metrics STR`
- `--forward-backward-custom-fn STR`
- `--forward-backward-loss-fn STR`
- `--grad-clip-value FLOAT`
- `--manual-optimizer-control BOOL`
- `--optimizer-step-frequency INT`
- `--sample-every-n-steps INT`
- `--sample-prompts STR`
- `--sample-temperature FLOAT`
- `--sample-top-k INT`
- `--sample-top-p FLOAT`
- `--token-weights STR`
- `--use-forward-backward BOOL`

### Help and Parameter Discovery

- Help output is grouped by section (e.g., "Basic", "Data Processing", "Training Hyperparameters", "PEFT/LoRA", "DPO/ORPO", "Reinforcement Learning (PPO)").
- Filter parameters shown in help by trainer using either the runtime trainer flag or a help-only preview flag:
  - Runtime filtering (also sets the trainer at run time):
    ```bash
    aitraining llm --trainer sft --help
    aitraining llm --trainer dpo --help
    aitraining llm --trainer ppo --help
    ```
  - Help-only preview (filters help without changing the runtime trainer):
    ```bash
    aitraining llm --preview-trainer sft --help
    aitraining llm --preview-trainer dpo --help
    ```
- Deprecated alias: `--help-trainer` is kept for compatibility but may be removed in a future release. Prefer `--preview-trainer`.

Examples:
```bash
# Show grouped help for all parameters
aitraining llm --help

# Show only SFT-related parameters
aitraining llm --trainer sft --help

# Preview DPO parameters in help without changing the runtime trainer
aitraining llm --preview-trainer dpo --help
```

### Dataset Conversion Feature

AITraining now includes automatic dataset format detection and conversion, compatible with Unsloth's approach. This feature can:
- Automatically detect dataset formats (Alpaca, ShareGPT, messages, DPO, plain text)
- Convert datasets to standard messages format for chat models
- Apply chat templates with proper tokenizer configuration
- Merge single-turn conversations into multi-turn dialogues

#### Supported Dataset Formats

1. **Alpaca Format**: `instruction`, `input`, `output` columns
2. **ShareGPT Format**: `conversations` with `from`/`value` pairs
3. **Messages Format**: `messages` with `role`/`content` pairs (standard format)
4. **DPO Format**: `prompt`, `chosen`, `rejected` columns
5. **Plain Text**: Simple `text` column

#### Dataset Conversion Examples

```bash
# Auto-detect and convert Alpaca dataset to messages format
aitraining llm \
  --model google/gemma-3-270m-it \
  --data-path tatsu-lab/alpaca \
  --auto-convert-dataset \
  --chat-template gemma \
  --trainer sft

# Convert ShareGPT format using mapping (preserves from/value)
aitraining llm \
  --model meta-llama/Llama-3-8B-Instruct \
  --data-path philschmid/guanaco-sharegpt-style \
  --use-sharegpt-mapping \
  --chat-template llama3 \
  --trainer sft

# Merge single-turn Alpaca into multi-turn conversations
aitraining llm \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --data-path yahma/alpaca-cleaned \
  --auto-convert-dataset \
  --conversation-extension 3 \
  --chat-template chatml \
  --trainer sft

# Full pipeline: detect, convert, and apply template
aitraining llm \
  --model google/gemma-3-270m-it \
  --data-path tatsu-lab/alpaca \
  --auto-convert-dataset \
  --apply-chat-template \
  --chat-template gemma \
  --trainer sft
```

#### Available Chat Templates

AITraining includes 32+ pre-configured chat templates extracted from Unsloth:
- **Llama family**: `llama`, `llama3`, `llama-3.1`
- **Gemma family**: `gemma`, `gemma2`, `gemma3`
- **Common formats**: `alpaca`, `chatml`, `zephyr`, `mistral`, `vicuna`
- **Qwen family**: `qwen3`, `qwen2.5`, `qwen3-instruct`
- **Others**: `phi-3`, `phi-4`, `yi-chat`, `starling`

Use `tokenizer` to use the model's default template, or specify a template name for override.

### Basic Usage Example
```bash
aitraining llm \
  --model gpt2 \
  --project-name my_llm_project \
  --data-path ./data \
  --text-column text \
  --epochs 3 \
  --batch-size 8 \
  --lr 5e-5
```

---

## Text Classification Training (`aitraining text-classification`)

### Complete Parameter List (30 parameters)

#### Core Parameters
- `--data-path STR`: Path to the dataset.
- `--model STR`: Name of the model to use
- `--project-name STR`: Name of the project

#### Data Configuration
- `--max-samples INT`: Maximum number of samples to use from dataset (for testing/debugging)
- `--target-column STR`: Name of the target column in the dataset
- `--text-column STR`: Name of the text column in the dataset
- `--train-split STR`: Name of the training split
- `--valid-split STR`: Name of the validation split

#### Training Hyperparameters
- `--auto-find-batch-size BOOL`: Whether to automatically find the batch size
- `--batch-size INT`: Training batch size
- `--early-stopping-patience INT`: Number of epochs with no improvement after which training will be stopped
- `--early-stopping-threshold FLOAT`: Threshold for measuring the new optimum to continue training
- `--epochs INT`: Number of training epochs
- `--gradient-accumulation INT`: Number of gradient accumulation steps
- `--lr FLOAT`: Learning rate
- `--max-grad-norm FLOAT`: Maximum gradient norm
- `--mixed-precision STR`: Mixed precision setting (fp16, bf16, or None)
- `--optimizer STR`: Optimizer to use
- `--scheduler STR`: Scheduler to use
- `--seed INT`: Random seed
- `--warmup-ratio FLOAT`: Warmup proportion
- `--weight-decay FLOAT`: Weight decay

#### Model Configuration
- `--max-seq-length INT`: Maximum sequence length

#### Evaluation Configuration
- `--eval-strategy STR`: Evaluation strategy

#### Logging & Checkpointing
- `--log STR`: Logging method for experiment tracking
- `--logging-steps INT`: Number of steps between logging
- `--save-total-limit INT`: Total number of checkpoints to save

#### HuggingFace Hub Integration
- `--push-to-hub BOOL`: Whether to push the model to the hub
- `--token STR`: Hub token for authentication
- `--username STR`: Hugging Face username

### Basic Usage Example
```bash
aitraining text-classification \
  --model bert-base-uncased \
  --project-name sentiment_classifier \
  --data-path ./data \
  --text-column text \
  --target-column label \
  --epochs 3 \
  --batch-size 16 \
  --lr 2e-5
```

---

## Text Regression Training (`aitraining text-regression`)

### Complete Parameter List (30 parameters)

#### Core Parameters
- `--data-path STR`: Path to the dataset.
- `--model STR`: Name of the pre-trained model to use
- `--project-name STR`: Name of the project for output directory

#### Data Configuration
- `--max-samples INT`: Maximum number of samples to use from dataset (for testing/debugging)
- `--target-column STR`: Name of the column containing target data
- `--text-column STR`: Name of the column containing text data
- `--train-split STR`: Name of the training data split
- `--valid-split STR`: Name of the validation data split

#### Training Hyperparameters
- `--auto-find-batch-size BOOL`: Whether to automatically find the batch size
- `--batch-size INT`: Batch size for training
- `--early-stopping-patience INT`: Number of epochs with no improvement after which training will be stopped
- `--early-stopping-threshold FLOAT`: Threshold for measuring the new optimum, to qualify as an improvement
- `--epochs INT`: Number of training epochs
- `--gradient-accumulation INT`: Number of steps to accumulate gradients before updating
- `--lr FLOAT`: Learning rate for the optimizer
- `--max-grad-norm FLOAT`: Maximum norm for the gradients
- `--mixed-precision STR`: Mixed precision training mode (fp16, bf16, or None)
- `--optimizer STR`: Optimizer to use
- `--scheduler STR`: Learning rate scheduler to use
- `--seed INT`: Random seed for reproducibility
- `--warmup-ratio FLOAT`: Proportion of training to perform learning rate warmup
- `--weight-decay FLOAT`: Weight decay to apply

#### Model Configuration
- `--max-seq-length INT`: Maximum sequence length for the inputs

#### Evaluation Configuration
- `--eval-strategy STR`: Evaluation strategy to use

#### Logging & Checkpointing
- `--log STR`: Logging method for experiment tracking
- `--logging-steps INT`: Number of steps between logging
- `--save-total-limit INT`: Maximum number of checkpoints to save

#### HuggingFace Hub Integration
- `--push-to-hub BOOL`: Whether to push the model to Hugging Face Hub
- `--token STR`: Token for accessing Hugging Face Hub
- `--username STR`: Hugging Face username

### Basic Usage Example
```bash
aitraining text-regression \
  --model bert-base-uncased \
  --project-name sentiment_scorer \
  --data-path ./reviews_data \
  --text-column review \
  --target-column rating \
  --epochs 5 \
  --batch-size 16 \
  --lr 2e-5
```

---

## Token Classification (NER) Training (`aitraining token-classification`)

### Complete Parameter List (30 parameters)

#### Core Parameters
- `--data-path STR`: Path to the dataset.
- `--model STR`: Name of the model to use
- `--project-name STR`: Name of the project

#### Data Configuration
- `--max-samples INT`: Maximum number of samples to use from dataset (for testing/debugging)
- `--tags-column STR`: Name of the tags column
- `--tokens-column STR`: Name of the tokens column
- `--train-split STR`: Name of the training split
- `--valid-split STR`: Name of the validation split

#### Training Hyperparameters
- `--auto-find-batch-size BOOL`: Whether to automatically find the batch size
- `--batch-size INT`: Training batch size
- `--early-stopping-patience INT`: Patience for early stopping
- `--early-stopping-threshold FLOAT`: Threshold for early stopping
- `--epochs INT`: Number of training epochs
- `--gradient-accumulation INT`: Gradient accumulation steps
- `--lr FLOAT`: Learning rate
- `--max-grad-norm FLOAT`: Maximum gradient norm
- `--mixed-precision STR`: Mixed precision setting (fp16, bf16, or None)
- `--optimizer STR`: Optimizer to use
- `--scheduler STR`: Scheduler to use
- `--seed INT`: Random seed
- `--warmup-ratio FLOAT`: Warmup proportion
- `--weight-decay FLOAT`: Weight decay

#### Model Configuration
- `--max-seq-length INT`: Maximum sequence length

#### Evaluation Configuration
- `--eval-strategy STR`: Evaluation strategy

#### Logging & Checkpointing
- `--log STR`: Logging method for experiment tracking
- `--logging-steps INT`: Number of steps between logging
- `--save-total-limit INT`: Total number of checkpoints to save

#### HuggingFace Hub Integration
- `--push-to-hub BOOL`: Whether to push the model to the Hugging Face hub
- `--token STR`: Hub token for authentication
- `--username STR`: Hugging Face username

### Basic Usage Example
```bash
aitraining token-classification \
  --model bert-base-cased \
  --project-name ner_model \
  --data-path ./ner_data \
  --tokens-column tokens \
  --tags-column tags \
  --epochs 5 \
  --batch-size 16 \
  --lr 5e-5
```

---

## Sequence-to-Sequence Training (`aitraining seq2seq`)

### Complete Parameter List (39 parameters)

#### Core Parameters
- `--data-path STR`: Path to the dataset.
- `--model STR`: Name of the model to be used
- `--project-name STR`: Name of the project or output directory

#### Data Configuration
- `--max-samples INT`: Maximum number of samples to use from dataset (for testing/debugging)
- `--target-column STR`: Name of the target text column in the dataset
- `--text-column STR`: Name of the text column in the dataset
- `--train-split STR`: Name of the training data split
- `--valid-split STR`: Name of the validation data split.

#### Training Hyperparameters
- `--auto-find-batch-size BOOL`: Whether to automatically find the batch size
- `--batch-size INT`: Training batch size
- `--early-stopping-patience INT`: Patience for early stopping
- `--early-stopping-threshold FLOAT`: Threshold for early stopping
- `--epochs INT`: Number of training epochs
- `--gradient-accumulation INT`: Number of gradient accumulation steps
- `--lr FLOAT`: Learning rate for training
- `--max-grad-norm FLOAT`: Maximum gradient norm for clipping
- `--mixed-precision STR`: Mixed precision training mode (fp16, bf16, or None).
- `--optimizer STR`: Optimizer to be used
- `--scheduler STR`: Learning rate scheduler to be used
- `--seed INT`: Random seed for reproducibility
- `--warmup-ratio FLOAT`: Proportion of warmup steps
- `--weight-decay FLOAT`: Weight decay for the optimizer

#### Model Configuration
- `--max-seq-length INT`: Maximum sequence length for input text
- `--max-target-length INT`: Maximum sequence length for target text

#### PEFT/LoRA Configuration
- `--lora-alpha INT`: LoRA-Alpha parameter for PEFT
- `--lora-dropout FLOAT`: LoRA-Dropout parameter for PEFT
- `--lora-r INT`: LoRA-R parameter for PEFT
- `--peft BOOL`: Whether to use Parameter-Efficient Fine-Tuning (PEFT)
- `--quantization STR`: Quantization mode (int4, int8, or None)
- `--target-modules STR`: Target modules for PEFT

#### Evaluation Configuration
- `--eval-strategy STR`: Evaluation strategy

#### Logging & Checkpointing
- `--log STR`: Logging method for experiment tracking
- `--logging-steps INT`: Number of steps between logging
- `--save-total-limit INT`: Maximum number of checkpoints to save

#### HuggingFace Hub Integration
- `--push-to-hub BOOL`: Whether to push the model to the Hugging Face Hub
- `--token STR`: Hub Token for authentication.
- `--username STR`: Hugging Face Username.

### Basic Usage Example
```bash
aitraining seq2seq \
  --model t5-small \
  --project-name translator \
  --data-path ./data \
  --text-column source \
  --target-column target \
  --epochs 5 \
  --batch-size 8 \
  --lr 3e-4
```

---

## Sentence Transformers Training (`aitraining sentence-transformers`)

### Complete Parameter List (33 parameters)

#### Core Parameters
- `--data-path STR`: Path to the dataset.
- `--model STR`: Name of the pre-trained model to use
- `--project-name STR`: Name of the project for output directory

#### Data Configuration
- `--max-samples INT`: Maximum number of samples to use from dataset (for testing/debugging)
- `--sentence1-column STR`: Name of the column containing the first sentence
- `--sentence2-column STR`: Name of the column containing the second sentence
- `--sentence3-column STR`: Name of the column containing the third sentence (if applicable)
- `--target-column STR`: Name of the column containing the target variable
- `--train-split STR`: Name of the training data split
- `--valid-split STR`: Name of the validation data split

#### Training Hyperparameters
- `--auto-find-batch-size BOOL`: Whether to automatically find the optimal batch size
- `--batch-size INT`: Batch size for training
- `--early-stopping-patience INT`: Number of epochs with no improvement after which training will be stopped
- `--early-stopping-threshold FLOAT`: Threshold for measuring the new optimum, to qualify as an improvement
- `--epochs INT`: Number of training epochs
- `--gradient-accumulation INT`: Number of steps to accumulate gradients before updating
- `--lr FLOAT`: Learning rate for training
- `--max-grad-norm FLOAT`: Maximum gradient norm for clipping
- `--mixed-precision STR`: Mixed precision training mode (fp16, bf16, or None)
- `--optimizer STR`: Optimizer to use
- `--scheduler STR`: Learning rate scheduler to use
- `--seed INT`: Random seed for reproducibility
- `--warmup-ratio FLOAT`: Proportion of training to perform learning rate warmup
- `--weight-decay FLOAT`: Weight decay to apply

#### Model Configuration
- `--max-seq-length INT`: Maximum sequence length for the input
- `--trainer STR`: Name of the trainer to use

#### Evaluation Configuration
- `--eval-strategy STR`: Evaluation strategy to use

#### Logging & Checkpointing
- `--log STR`: Logging method for experiment tracking
- `--logging-steps INT`: Number of steps between logging
- `--save-total-limit INT`: Maximum number of checkpoints to save

#### HuggingFace Hub Integration
- `--push-to-hub BOOL`: Whether to push the model to Hugging Face Hub
- `--token STR`: Token for accessing Hugging Face Hub
- `--username STR`: Hugging Face username

### Basic Usage Example
```bash
aitraining sentence-transformers \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --project-name similarity_model \
  --data-path ./similarity_data \
  --sentence1-column text1 \
  --sentence2-column text2 \
  --target-column score \
  --epochs 3 \
  --batch-size 32
```

---

## Image Classification Training (`aitraining image-classification`)

### Complete Parameter List (29 parameters)

#### Core Parameters
- `--data-path STR`: Path to the dataset.
- `--model STR`: Pre-trained model name or path
- `--project-name STR`: Name of the project for output directory

#### Data Configuration
- `--image-column STR`: Column name for images in the dataset
- `--max-samples INT`: Maximum number of samples to use from dataset (for testing/debugging)
- `--target-column STR`: Column name for target labels in the dataset
- `--train-split STR`: Name of the training data split
- `--valid-split STR`: Name of the validation data split.

#### Training Hyperparameters
- `--auto-find-batch-size BOOL`: Automatically find optimal batch size
- `--batch-size INT`: Batch size for training
- `--early-stopping-patience INT`: Number of epochs with no improvement for early stopping
- `--early-stopping-threshold FLOAT`: Threshold for early stopping
- `--epochs INT`: Number of epochs for training
- `--gradient-accumulation INT`: Number of gradient accumulation steps
- `--lr FLOAT`: Learning rate for the optimizer
- `--max-grad-norm FLOAT`: Maximum gradient norm for clipping
- `--mixed-precision STR`: Mixed precision training mode (fp16, bf16, or None).
- `--optimizer STR`: Optimizer type
- `--scheduler STR`: Learning rate scheduler type
- `--seed INT`: Random seed for reproducibility
- `--warmup-ratio FLOAT`: Warmup ratio for learning rate scheduler
- `--weight-decay FLOAT`: Weight decay for the optimizer

#### Evaluation Configuration
- `--eval-strategy STR`: Evaluation strategy during training

#### Logging & Checkpointing
- `--log STR`: Logging method for experiment tracking
- `--logging-steps INT`: Number of steps between logging
- `--save-total-limit INT`: Maximum number of checkpoints to keep

#### HuggingFace Hub Integration
- `--push-to-hub BOOL`: Whether to push the model to Hugging Face Hub
- `--token STR`: Hugging Face Hub token for authentication.
- `--username STR`: Hugging Face account username.

### Basic Usage Example
```bash
aitraining image-classification \
  --model google/vit-base-patch16-224 \
  --project-name image_classifier \
  --data-path ./images \
  --image-column image \
  --target-column label \
  --epochs 10 \
  --batch-size 16 \
  --lr 2e-5
```

---

## Image Regression Training (`aitraining image-regression`)

### Complete Parameter List (29 parameters)

#### Core Parameters
- `--data-path STR`: Path to the dataset.
- `--model STR`: Name of the model to use
- `--project-name STR`: Output directory name

#### Data Configuration
- `--image-column STR`: Image column name
- `--max-samples INT`: Maximum number of samples to use from dataset (for testing/debugging)
- `--target-column STR`: Target column name
- `--train-split STR`: Train split name
- `--valid-split STR`: Validation split name.

#### Training Hyperparameters
- `--auto-find-batch-size BOOL`: Whether to auto find batch size
- `--batch-size INT`: Training batch size
- `--early-stopping-patience INT`: Early stopping patience
- `--early-stopping-threshold FLOAT`: Early stopping threshold
- `--epochs INT`: Number of training epochs
- `--gradient-accumulation INT`: Gradient accumulation steps
- `--lr FLOAT`: Learning rate
- `--max-grad-norm FLOAT`: Max gradient norm
- `--mixed-precision STR`: Mixed precision type (fp16, bf16, or None).
- `--optimizer STR`: Optimizer to use
- `--scheduler STR`: Scheduler to use
- `--seed INT`: Random seed
- `--warmup-ratio FLOAT`: Warmup proportion
- `--weight-decay FLOAT`: Weight decay

#### Evaluation Configuration
- `--eval-strategy STR`: Evaluation strategy

#### Logging & Checkpointing
- `--log STR`: Logging using experiment tracking
- `--logging-steps INT`: Logging steps
- `--save-total-limit INT`: Save total limit

#### HuggingFace Hub Integration
- `--push-to-hub BOOL`: Whether to push to hub
- `--token STR`: Hub Token.
- `--username STR`: Hugging Face Username.

### Basic Usage Example
```bash
aitraining image-regression \
  --model google/vit-base-patch16-224 \
  --project-name age_predictor \
  --data-path ./face_images \
  --image-column image \
  --target-column age \
  --epochs 10 \
  --batch-size 16 \
  --lr 1e-4
```

---

## Object Detection Training (`aitraining object-detection`)

### Complete Parameter List (30 parameters)

#### Core Parameters
- `--data-path STR`: Path to the dataset.
- `--model STR`: Name of the model to be used
- `--project-name STR`: Name of the project for output directory

#### Data Configuration
- `--image-column STR`: Name of the image column in the dataset
- `--max-samples INT`: Maximum number of samples to use from dataset (for testing/debugging)
- `--objects-column STR`: Name of the target column in the dataset
- `--train-split STR`: Name of the training data split
- `--valid-split STR`: Name of the validation data split.

#### Training Hyperparameters
- `--auto-find-batch-size BOOL`: Whether to automatically find batch size
- `--batch-size INT`: Training batch size
- `--early-stopping-patience INT`: Number of epochs with no improvement after which training will be stopped
- `--early-stopping-threshold FLOAT`: Minimum change to qualify as an improvement
- `--epochs INT`: Number of training epochs
- `--gradient-accumulation INT`: Gradient accumulation steps
- `--lr FLOAT`: Learning rate
- `--max-grad-norm FLOAT`: Max gradient norm
- `--mixed-precision STR`: Mixed precision type (fp16, bf16, or None).
- `--optimizer STR`: Optimizer to be used
- `--scheduler STR`: Scheduler to be used
- `--seed INT`: Random seed
- `--warmup-ratio FLOAT`: Warmup proportion
- `--weight-decay FLOAT`: Weight decay

#### Model Configuration
- `--image-square-size INT`: Longest size to which the image will be resized, then padded to square

#### Evaluation Configuration
- `--eval-strategy STR`: Evaluation strategy

#### Logging & Checkpointing
- `--log STR`: Logging method for experiment tracking
- `--logging-steps INT`: Number of steps between logging
- `--save-total-limit INT`: Total number of checkpoints to save

#### HuggingFace Hub Integration
- `--push-to-hub BOOL`: Whether to push the model to the Hugging Face Hub
- `--token STR`: Hub Token for authentication.
- `--username STR`: Hugging Face Username.

### Basic Usage Example
```bash
aitraining object-detection \
  --model facebook/detr-resnet-50 \
  --project-name object_detector \
  --data-path ./coco_data \
  --image-column image \
  --objects-column objects \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4
```

---

## Extractive QA Training (`aitraining extractive-qa`)

### Complete Parameter List (32 parameters)

#### Core Parameters
- `--data-path STR`
- `--model STR`
- `--project-name STR`

#### Data Configuration
- `--answer-column STR`
- `--max-samples INT`
- `--question-column STR`
- `--text-column STR`
- `--train-split STR`
- `--valid-split STR`

#### Training Hyperparameters
- `--auto-find-batch-size BOOL`
- `--batch-size INT`
- `--early-stopping-patience INT`
- `--early-stopping-threshold FLOAT`
- `--epochs INT`
- `--gradient-accumulation INT`
- `--lr FLOAT`
- `--max-grad-norm FLOAT`
- `--mixed-precision STR`
- `--optimizer STR`
- `--scheduler STR`
- `--seed INT`
- `--warmup-ratio FLOAT`
- `--weight-decay FLOAT`

#### Model Configuration
- `--max-doc-stride INT`
- `--max-seq-length INT`

#### Evaluation Configuration
- `--eval-strategy STR`

#### Logging & Checkpointing
- `--log STR`
- `--logging-steps INT`
- `--save-total-limit INT`

#### HuggingFace Hub Integration
- `--push-to-hub BOOL`
- `--token STR`
- `--username STR`

### Basic Usage Example
```bash
aitraining extractive-qa \
  --model bert-base-uncased \
  --project-name qa_model \
  --data-path ./squad_data \
  --text-column context \
  --question-column question \
  --answer-column answers \
  --epochs 3 \
  --batch-size 16 \
  --lr 3e-5
```

---

## Vision-Language Model Training (`aitraining vlm`)

### Complete Parameter List (37 parameters)

#### Core Parameters
- `--data-path STR`: Data path
- `--model STR`: Model name
- `--project-name STR`: Output directory

#### Data Configuration
- `--image-column STR`: Image column
- `--max-samples INT`: Maximum number of samples to use from dataset (for testing/debugging)
- `--prompt-text-column STR`: Prompt (prefix) column
- `--text-column STR`: Text (answer) column
- `--train-split STR`: Train data config
- `--valid-split STR`: Validation data config

#### Training Hyperparameters
- `--auto-find-batch-size BOOL`: Auto find batch size
- `--batch-size INT`: Training batch size
- `--epochs INT`: Number of training epochs
- `--gradient-accumulation INT`: Gradient accumulation steps
- `--lr FLOAT`: Learning rate
- `--max-grad-norm FLOAT`: Max gradient norm
- `--mixed-precision STR`: Mixed precision (fp16, bf16, or None)
- `--optimizer STR`: Optimizer
- `--scheduler STR`: Scheduler
- `--seed INT`: Seed
- `--warmup-ratio FLOAT`: Warmup proportion
- `--weight-decay FLOAT`: Weight decay

#### Model Configuration
- `--disable-gradient-checkpointing BOOL`: Gradient checkpointing
- `--trainer STR`: Trainer type (captioning, vqa, segmentation, detection)

#### PEFT/LoRA Configuration
- `--lora-alpha INT`: Lora alpha
- `--lora-dropout FLOAT`: Lora dropout
- `--lora-r INT`: Lora r
- `--merge-adapter BOOL`: Merge adapter
- `--peft BOOL`: Use PEFT
- `--quantization STR`: Quantization (int4, int8, or None)
- `--target-modules STR`: Target modules

#### Evaluation Configuration
- `--eval-strategy STR`: Evaluation strategy

#### Logging & Checkpointing
- `--log STR`: Logging using experiment tracking
- `--logging-steps INT`: Logging steps
- `--save-total-limit INT`: Save total limit

#### HuggingFace Hub Integration
- `--push-to-hub BOOL`: Push to hub
- `--token STR`: Huggingface token
- `--username STR`: Hugging Face Username

### Basic Usage Example
```bash
aitraining vlm \
  --model google/paligemma-3b-pt-224 \
  --project-name image_captioning \
  --data-path ./vlm_data \
  --image-column image \
  --text-column caption \
  --epochs 5 \
  --batch-size 8 \
  --lr 5e-5
```

---

## Tabular Data Training (`aitraining tabular`)

### Complete Parameter List (20 parameters)

#### Core Parameters
- `--data-path STR`: Path to the dataset.
- `--model STR`: Name of the model to use
- `--project-name STR`: Name of the output directory

#### Data Configuration
- `--categorical-columns STR`: List of categorical columns.
- `--id-column STR`: Name of the ID column
- `--max-samples INT`: Maximum number of samples to use from dataset (for testing/debugging)
- `--numerical-columns STR`: List of numerical columns.
- `--target-columns STR`: Target column(s) in the dataset
- `--task STR`: Type of task (e.g., "classification")
- `--train-split STR`: Name of the training data split
- `--valid-split STR`: Name of the validation data split.

#### Training Hyperparameters
- `--num-trials INT`: Number of trials for hyperparameter optimization
- `--seed INT`: Random seed for reproducibility
- `--time-limit INT`: Time limit for training in seconds

#### HuggingFace Hub Integration
- `--push-to-hub BOOL`: Whether to push the model to the hub
- `--token STR`: Hub Token for authentication.
- `--username STR`: Hugging Face Username.

#### Additional Parameters
- `--categorical-imputer STR`: Imputer strategy for categorical columns.
- `--numeric-scaler STR`: Scaler strategy for numerical columns.
- `--numerical-imputer STR`: Imputer strategy for numerical columns.

### Basic Usage Example
```bash
aitraining tabular \
  --model xgboost \
  --project-name sales_predictor \
  --data-path ./sales_data.csv \
  --target-columns target \
  --task regression \
  --num-trials 100
```

---

## Notes

1. **Parameter Aliases**: Most parameters support both hyphenated (--param-name) and underscored (--param_name) formats
2. **JSON Parameters**: Some parameters accept JSON strings for complex configurations
3. **Backend Options**: The CLI supports local, spaces, and endpoint backends (configured separately)
4. **Data Formats**: Most commands support CSV, JSON, and Parquet data formats
5. **Mixed Precision**: Use `--mixed-precision fp16` or `--mixed-precision bf16` for faster training
6. **Auto Batch Size**: Use `--auto-find-batch-size` to automatically determine optimal batch size
7. **Hub Integration**: Use `--push-to-hub` with `--username` and `--token` to upload models to HuggingFace Hub

## Parameter Validation

The CLI performs automatic validation for:
- Required parameters based on command type
- Mutually exclusive parameters
- Trainer-specific requirements
- Data format compatibility
- Model architecture compatibility
