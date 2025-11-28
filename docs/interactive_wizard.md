# Interactive CLI Wizard

The AITraining interactive wizard provides a modern, user-friendly way to configure and launch training jobs for **all** supported trainer types—LLMs, text/vision/tabular models, seq2seq, extractive QA, sentence transformers, VLM, and more—from the command line. It guides you through all necessary steps with helpful prompts, validation, and confirmation.

## Features

- **Step-by-step configuration**: Clear progression through trainer selection, dataset setup, model choice, and advanced parameters
- **Trainer-specific guidance**: Automatically adapts questions based on selected trainer (LLM SFT/DPO/ORPO/PPO, text classification, token classification, tabular, image classification/regression, seq2seq, extractive QA, sentence transformers, VLM)
- **Smart defaults**: Press Enter to accept sensible defaults shown in brackets
- **Validation**: Real-time Pydantic validation with helpful error messages
- **Configuration summary**: Review all settings before starting training
- **Graceful cancellation**: Press Ctrl+C at any time to exit
- **Live logs by default**: `log=wandb` + LEET visualizer are now enabled automatically for CLI/TUI workflows

## How to Launch

There are three ways to launch the interactive wizard:

### 1. Base Command (Recommended for New Users)

Simply run `aitraining` without any arguments:

```bash
aitraining
```

This automatically launches the wizard with a friendly interface.

### 2. Explicit Interactive Flag

From the LLM subcommand:

```bash
aitraining llm --interactive
```

### 3. Auto-Launch on Missing Parameters

The wizard launches automatically when you specify `--train` but omit required parameters:

```bash
# This will launch the wizard since model, data, and project are missing
aitraining llm --train

# This will also launch the wizard to fill in missing details
aitraining llm --train --model gpt2
```

## Hugging Face Token Prompt

Before Step 0, the wizard now checks for a Hugging Face token so it can list private models/datasets and authenticate any Hub calls:

- If `HF_TOKEN` is already exported (or you passed `--token`), the wizard reuses it and surfaces a ✅ confirmation.
- Otherwise it prompts for a token (press Enter to skip and continue with public assets only).
- The captured token is exported to `HF_TOKEN` for the rest of the session and is still editable later under **Advanced → Hub Integration**.

Skip the prompt only if you are sure every asset you need is public.

## Wizard Flow

### Step 0: Choose Trainer Type

When launched from the base `aitraining` command (no subcommand), the wizard first prompts you to choose which trainer family you want to configure:

```
📋 Step 0: Choose Trainer Type
============================================================

Available trainer types:
  1 . Large Language Models (LLM) - text generation, chat, instruction following
  2 . Text Classification - categorize text into labels
  3 . Token Classification - NER, POS tagging
  4 . Tabular Data - classification or regression on structured data
  5 . Image Classification - categorize images
  6 . Image Regression - predict values from images
  7 . Sequence-to-Sequence - translation, summarization
  8 . Extractive QA - answer questions from context
  9 . Sentence Transformers - semantic similarity embeddings
  10. Vision-Language Models - multimodal tasks

Select trainer type [1-10, default: 1]:
```

Once trainer type is selected, the wizard switches into that trainer’s flow. For LLMs, you’ll see an additional sub-step to pick the LLM trainer variant.

### Step 1 (LLM only): Choose Training Type

For LLM workflows, the wizard then prompts you to choose the specific LLM trainer variant:

```
📋 Step 1: Choose Training Type
============================================================

Available trainers:
  1. sft             - Supervised Fine-Tuning (most common)
  2. dpo             - Direct Preference Optimization
  3. orpo            - Odds Ratio Preference Optimization
  4. ppo             - Proximal Policy Optimization (RL)
  5. default         - Generic training

Select trainer [1-5, default: 1]:
```

**What each trainer does:**
- **SFT** (Supervised Fine-Tuning): Train on text completion tasks, instruction following, chat
- **DPO** (Direct Preference Optimization): Train with preference pairs (chosen vs rejected responses)
- **ORPO** (Odds Ratio Preference Optimization): Alternative preference learning method
- **PPO** (Proximal Policy Optimization): Reinforcement learning with reward model

### Step 2: Basic Configuration

Provide essential project settings:

```
📋 Step 2: Basic Configuration
============================================================

Project name [my-llm-project]:
```

### Step 3: Dataset Configuration

Specify your training data:

```
📋 Step 3: Dataset Configuration
============================================================

Dataset location:
  • Enter local path (e.g., ./data)
  • Enter HuggingFace dataset ID (e.g., timdettmers/openassistant-guanaco)

Dataset path: ./data/train.jsonl
Training split [train]: train
Validation split (optional, press Enter to skip):
Maximum samples (optional, for testing/debugging):
```

The wizard now includes an optional `max_samples` parameter that allows you to limit the number of training samples used. This is useful for:
- Quick testing with a small subset of data
- Rapid prototyping without full dataset downloads
- CI/CD pipeline testing
- Development environment validation

Simply press Enter to use the full dataset, or enter a number (e.g., `100`) to limit training to that many samples.

**Column Mapping** (varies by trainer):

- **LLM – SFT**: Requires `text_column` containing the text to train on
- **LLM – DPO/ORPO**: Requires three columns:
  - `prompt_text_column`: The instruction/question
  - `text_column` (chosen): The preferred response
  - `rejected_text_column`: The non-preferred response
- **LLM – PPO**: Requires `text_column` and `rl_reward_model_path`
- **Text Classification**: `text_column`, `target_column`
- **Token Classification**: `tokens_column`, `tags_column`
- **Tabular**: `target_columns`, plus optional `categorical_columns` / `numerical_columns`
- **Image Classification / Regression**: `image_column`, `target_column`
- **Seq2Seq**: `text_column` (source), `target_column` (target), optional `max_target_length`
- **Extractive QA**: `text_column` (context), `question_column`, `answer_column`
- **Sentence Transformers**: `sentence1_column`, optional `sentence2_column` / `sentence3_column`, `target_column`
- **VLM**: `text_column`, `image_column`, optional `prompt_text_column`

### Step 4: Model Selection

Choose from popular models or enter a custom one:

```
📋 Step 4: Model Selection
============================================================

Popular models for sft:
  1. meta-llama/Llama-3.2-1B
  2. meta-llama/Llama-3.2-3B
  3. Qwen/Qwen2.5-0.5B
  4. Qwen/Qwen2.5-1.5B
  5. google/gemma-2-2b
  6. Enter custom model

Select model [1-6, default: 1]:
```

For custom models:
```
Enter model name or HF model ID: my-org/custom-model-7b
```

### Step 5: Advanced Configuration (Optional)

Configure advanced parameters organized by category:

```
📋 Step 5: Advanced Configuration (Optional)
============================================================

Would you like to configure advanced parameters?
  • Training hyperparameters (learning rate, batch size, etc.)
  • PEFT/LoRA settings
  • Model quantization
  • And more...

Configure advanced parameters? [y/N]: n
```

If you choose yes, you'll be prompted for parameter groups:

```
────────────────────────────────────────────────────────────
⚙️  Training Hyperparameters
────────────────────────────────────────────────────────────

Configure Training Hyperparameters parameters? [y/N]: y

lr
  (Learning rate for training)
lr [3e-05]:

epochs
  (Number of training epochs)
epochs [1]:

batch_size
  (Batch size for training)
batch_size [2]:
```

**Available parameter groups:**
- Basic
- Data Processing
- Training Configuration
- Training Hyperparameters
- PEFT/LoRA
- DPO/ORPO (trainer-specific)
- Hub Integration
- Knowledge Distillation
- Hyperparameter Sweep
- Enhanced Evaluation
- Reinforcement Learning (PPO) (trainer-specific)
- Advanced Features

### Step 6: Review and Confirm

Review your complete configuration:

```
============================================================
📋 Configuration Summary
============================================================

Basic Configuration:
  • project_name: my-sft-project
  • trainer: sft

Dataset:
  • data_path: ./data/train.jsonl
  • train_split: train
  • text_column: text

Model & Training:
  • model: meta-llama/Llama-3.2-1B
  • epochs: 1
  • batch_size: 2

────────────────────────────────────────────────────────────
✓ Configuration is valid!

============================================================

🚀 Start training with this configuration? [Y/n]:
```

After confirmation, training begins immediately!

## Example Workflows

### Quick SFT Training

```bash
$ aitraining

Select trainer: 1 (SFT)
Project name: my-chat-model
Dataset path: ./data/conversations.jsonl
Training split: train
Validation split: <Enter>
Text column: text
Select model: 1 (Llama-3.2-1B)
Configure advanced? n
Start training? y

🚀 Starting training...
✓ Training job started! Job ID: 12345
```

### DPO Training with Preferences

```bash
$ aitraining llm --interactive

Select trainer: 2 (DPO)
Project name: preference-tuned-model
Dataset path: username/preference-dataset
Training split: train
Validation split: test
Prompt column: prompt
Chosen column: chosen
Rejected column: rejected
Select model: 1 (Llama-3.2-1B)
Configure advanced? y
  Configure PEFT/LoRA? y
    peft: y
    lora_r: 16
    lora_alpha: 32
Start training? y
```

### PPO RL Training

```bash
$ aitraining

Select trainer: 4 (PPO)
Project name: rl-optimized-model
Dataset path: ./rl_prompts.jsonl
Training split: train
Text column: prompt
Reward model path: models/reward-model-7b
Select model: 1 (Llama-3.2-1B)
Configure advanced? n
Start training? y
```

## Tips and Best Practices

### Choosing Defaults

Most default values are sensible for initial experiments:
- `batch_size=2`: Small batch size for memory efficiency
- `epochs=1`: Quick test run
- `lr=3e-5`: Standard learning rate
- `quantization=int4`: Memory-efficient quantization

For production training, you'll likely want to configure advanced parameters.

### Dataset Paths

- **Local files**: Use relative (`./data`) or absolute paths (`/home/user/data`)
- **HuggingFace datasets**: Use format `username/dataset-name` or `org/dataset-name`
- Supported formats: JSONL, CSV, Parquet

### Column Mapping

Ensure your dataset has the required columns:

**SFT:**
```json
{"text": "This is the training text..."}
```

**DPO/ORPO:**
```json
{
  "prompt": "What is AI?",
  "chosen": "AI stands for Artificial Intelligence...",
  "rejected": "AI is a computer."
}
```

**PPO:**
```json
{"text": "Write a story about..."}
```

### Advanced Configuration

When to configure advanced parameters:
- **PEFT/LoRA**: For memory-efficient fine-tuning of large models
- **Training Hyperparameters**: To tune learning rate, batch size, epochs
- **Quantization**: To reduce memory usage (4-bit/8-bit)
- **Hub Integration**: To push model to HuggingFace Hub
- **Enhanced Evaluation**: For detailed metrics and benchmarks

### Validation

The wizard validates your configuration using Pydantic models. Common validation errors:

- **Missing reward model (PPO)**: Must provide `rl_reward_model_path`
- **Missing columns (DPO/ORPO)**: Must specify all three columns
- **Invalid paths**: Dataset path must exist or be valid HF dataset ID

The wizard will show these errors and let you fix them before training.

## Cancellation

Press **Ctrl+C** at any time to cancel the wizard:

```
^C

❌ Setup cancelled.
```

No training will start if you cancel.

## Integration with Existing Workflows

The wizard generates the same configuration as the CLI flags:

```bash
# Wizard equivalent to:
aitraining llm --train \
  --trainer sft \
  --project-name my-project \
  --data-path ./data \
  --model meta-llama/Llama-3.2-1B \
  --text-column text
```

You can also pre-fill values via CLI and let the wizard handle the rest:

```bash
# Set trainer and model, wizard fills in the rest
aitraining llm --interactive --trainer dpo --model gpt2
```

## Testing

The wizard includes comprehensive test coverage:

```bash
pytest tests/cli/test_interactive_wizard.py -v
```

Tests verify:
- Basic wizard flow for all trainer types
- Column mapping for each trainer
- Custom model selection
- Validation and error handling
- Cancellation behavior
- Integration with `LLMTrainingParams` and `AutoTrainProject`

## Technical Details

### Architecture

The wizard is implemented in `src/autotrain/cli/interactive_wizard.py` as the `InteractiveWizard` class:

- **Reuses existing metadata**: `FIELD_GROUPS`, `FIELD_SCOPES`, `get_field_info()`
- **Validates with Pydantic**: Uses `LLMTrainingParams` for validation
- **Integrates seamlessly**: Returns config dict compatible with `AutoTrainProject`

### Entry Points

1. **Base command** (`src/autotrain/cli/autotrain.py`):
   - Detects no subcommand → launches wizard
   - Creates `LLMTrainingParams` and `AutoTrainProject` automatically

2. **LLM subcommand** (`src/autotrain/cli/run_llm.py`):
   - `--interactive` flag → launches wizard explicitly
   - `--train` with missing params → auto-launches wizard
   - Merges wizard results into `self.args`

### Customization

To add new trainers or parameters:

1. Update `FIELD_GROUPS` and `FIELD_SCOPES` in `run_llm.py`
2. Add trainer to `STARTER_MODELS` in `interactive_wizard.py`
3. Update `_prompt_column_mapping()` for trainer-specific columns
4. Add tests in `test_interactive_wizard.py`

## Troubleshooting

**Wizard doesn't launch:**
- Check you're running in a TTY: `sys.stdout.isatty()` must be `True`
- Piped commands show help instead: `aitraining | cat` won't launch wizard

**Validation errors:**
- Read error messages carefully - they indicate which fields are missing/invalid
- Check your dataset has required columns
- Ensure reward model path exists (for PPO)

**Training doesn't start:**
- Make sure you confirmed with 'y' at the final prompt
- Check logs for error messages
- Verify dataset path is accessible

## See Also

- [CLI Parameter Guide](../README.md#cli-parameters)
- [LLM Training Guide](../docs/llm_training.md)
- [Dataset Format Guide](../docs/dataset_formats.md)
