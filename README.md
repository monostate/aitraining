<p align="center">
  <img src="https://raw.githubusercontent.com/monostate/aitraining/main/docs/images/terminal-wizard.png" alt="AITraining Interactive Wizard" width="700">
</p>

<p align="center">
  <a href="https://pypi.org/project/aitraining/"><img src="https://img.shields.io/pypi/v/aitraining.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/aitraining/"><img src="https://img.shields.io/pypi/pyversions/aitraining.svg" alt="Python versions"></a>
  <a href="https://github.com/monostate/aitraining/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <b>Train state-of-the-art ML models with minimal code</b>
</p>

<p align="center">
  English | <a href="README_PTBR.md">Portugues</a>
</p>

---

AITraining is an advanced machine learning training platform built on top of [AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced). It provides a streamlined interface for fine-tuning LLMs, vision models, and more.

## Highlights

### Automatic Dataset Conversion

Feed any dataset format and AITraining automatically detects and converts it. Supports 6 input formats with automatic detection:

| Format | Detection | Example Columns |
|--------|-----------|-----------------|
| **Alpaca** | instruction/input/output | `{"instruction": "...", "output": "..."}` |
| **ShareGPT** | from/value pairs | `{"conversations": [{"from": "human", ...}]}` |
| **Messages** | role/content | `{"messages": [{"role": "user", ...}]}` |
| **Q&A** | question/answer variants | `{"question": "...", "answer": "..."}` |
| **DPO** | prompt/chosen/rejected | For preference training |
| **Plain Text** | Single text column | Raw text for pretraining |

```bash
aitraining llm --train --auto-convert-dataset --chat-template gemma3 \
  --data-path tatsu-lab/alpaca --model google/gemma-3-270m-it
```

### 32 Chat Templates

Comprehensive template library with token-level weight control:

- **Llama family**: llama, llama-3, llama-3.1
- **Gemma family**: gemma, gemma-2, gemma-3, gemma-3n
- **Others**: mistral, qwen-2.5, phi-3, phi-4, chatml, alpaca, vicuna, zephyr

```python
from autotrain.rendering import get_renderer, ChatFormat, RenderConfig

config = RenderConfig(format=ChatFormat.CHATML, only_assistant=True)
renderer = get_renderer('chatml', tokenizer, config)
encoded = renderer.build_supervised_example(conversation)
# Returns: {'input_ids', 'labels', 'token_weights', 'attention_mask'}
```

### Custom RL Environments

Build custom reward functions for PPO training with three environment types:

```bash
# Text generation with custom reward
aitraining llm --train --trainer ppo \
  --rl-env-type text_generation \
  --rl-env-config '{"stop_sequences": ["</s>"]}' \
  --rl-reward-model-path ./reward_model

# Multi-objective rewards (correctness + formatting)
aitraining llm --train --trainer ppo \
  --rl-env-type multi_objective \
  --rl-env-config '{"reward_components": {"correctness": {"type": "keyword"}, "formatting": {"type": "length"}}}' \
  --rl-reward-weights '{"correctness": 1.0, "formatting": 0.1}'
```

### Hyperparameter Sweeps

Automated optimization with Optuna, random search, or grid search:

```python
from autotrain.utils import HyperparameterSweep, SweepConfig, ParameterRange

config = SweepConfig(
    backend="optuna",
    optimization_metric="eval_loss",
    optimization_mode="minimize",
    num_trials=20,
)

sweep = HyperparameterSweep(
    objective_function=train_model,
    config=config,
    parameters=[
        ParameterRange("learning_rate", "log_uniform", low=1e-5, high=1e-3),
        ParameterRange("batch_size", "categorical", choices=[4, 8, 16]),
    ]
)
result = sweep.run()
# Returns best_params, best_value, trial history
```

### Enhanced Evaluation Metrics

8 metrics beyond loss, with callbacks for periodic evaluation:

| Metric | Type | Use Case |
|--------|------|----------|
| **Perplexity** | Auto-computed | Language model quality |
| **BLEU** | Generation | Translation, summarization |
| **ROUGE** (1/2/L) | Generation | Summarization |
| **BERTScore** | Generation | Semantic similarity |
| **METEOR** | Generation | Translation |
| **F1/Accuracy** | Classification | Standard metrics |
| **Exact Match** | QA | Question answering |

```python
from autotrain.evaluation import Evaluator, EvaluationConfig, MetricType

config = EvaluationConfig(
    metrics=[MetricType.PERPLEXITY, MetricType.BLEU, MetricType.ROUGE, MetricType.BERTSCORE],
    save_predictions=True,
)
evaluator = Evaluator(model, tokenizer, config)
result = evaluator.evaluate(dataset)
```

### Auto LoRA Merge

After PEFT training, automatically merge adapters and save deployment-ready models:

```bash
# Default: merges adapters into full model
aitraining llm --train --peft --model meta-llama/Llama-3.2-1B

# Keep adapters separate (smaller files)
aitraining llm --train --peft --no-merge-adapter --model meta-llama/Llama-3.2-1B
```

---

## Screenshots

<p align="center">
  <img src="https://raw.githubusercontent.com/monostate/aitraining/main/docs/images/chat-screenshot.png" alt="Chat interface for testing trained models" width="700">
  <br>
  <em>Built-in chat interface for testing trained models with conversation history</em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/monostate/aitraining/main/docs/images/tui-wandb.png" alt="Terminal UI with W&B LEET integration" width="700">
  <br>
  <em>Terminal UI with real-time W&B LEET metrics visualization</em>
</p>

---

## Installation

```bash
pip install aitraining
```

Requirements: Python >= 3.10, PyTorch

## Quick Start

### Interactive Wizard

```bash
aitraining
```

The wizard guides you through:
1. Trainer type selection (LLM, vision, NLP, tabular)
2. Model selection with curated catalogs from HuggingFace
3. Dataset configuration with auto-format detection
4. Advanced parameters (PEFT, quantization, sweeps)

### Config File

```bash
aitraining --config config.yaml
```

### Python API

```python
from autotrain.trainers.clm import train
from autotrain.trainers.clm.params import LLMTrainingParams

config = LLMTrainingParams(
    model="meta-llama/Llama-3.2-1B",
    data_path="your-dataset",
    trainer="sft",
    epochs=3,
    batch_size=4,
    lr=2e-5,
    peft=True,
    auto_convert_dataset=True,
    chat_template="llama3",
)

train(config)
```

---

## Comparison

### AITraining vs AutoTrain vs Tinker

| Feature | AutoTrain | AITraining | Tinker |
|---------|-----------|------------|--------|
| **Trainers** |
| SFT/DPO/ORPO | Yes | Yes | Yes |
| PPO (RLHF) | Basic | Enhanced (TRL) | Advanced |
| Reward Modeling | Yes | Yes | No |
| Knowledge Distillation | No | Yes (KL + CE loss) | Yes (text-only) |
| **Data** |
| Auto Format Detection | No | Yes (6 formats) | No |
| Chat Template Library | Basic | 32 templates | 5 templates |
| Runtime Column Mapping | No | Yes | No |
| Conversation Extension | No | Yes | No |
| **Training** |
| Hyperparameter Sweeps | No | Yes (Optuna) | Manual |
| Custom RL Environments | No | Yes (3 types) | Yes |
| Multi-objective Rewards | No | Yes | Yes |
| Forward-Backward Pipeline | No | Yes | Yes |
| Async Off-Policy RL | No | No | Yes |
| Stream Minibatch | No | No | Yes |
| **Evaluation** |
| Metrics Beyond Loss | No | 8 metrics | Manual |
| Periodic Eval Callbacks | No | Yes | Yes |
| Custom Metric Registration | No | Yes | No |
| **Interface** |
| Interactive CLI Wizard | No | Yes | No |
| TUI (Experimental) | No | Yes | No |
| W&B LEET Visualizer | No | Yes | Yes |
| **Hardware** |
| Apple Silicon (MPS) | Limited | Full | No |
| Quantization (int4/int8) | Yes | Yes | Unknown |
| Multi-GPU | Yes | Yes | Yes |
| **Task Coverage** |
| Vision Tasks | Yes | Yes | No |
| NLP Tasks | Yes | Yes | No |
| Tabular Tasks | Yes | Yes | No |
| Tool Use Environments | No | No | Yes |
| Multiplayer RL | No | No | Yes |

---

## Supported Tasks

| Task | Trainers | Status |
|------|----------|--------|
| LLM Fine-tuning | SFT, DPO, ORPO, PPO, Reward, Distillation | Stable |
| Text Classification | Single/Multi-label | Stable |
| Token Classification | NER, POS tagging | Stable |
| Sequence-to-Sequence | Translation, Summarization | Stable |
| Image Classification | Single/Multi-label | Stable |
| Object Detection | YOLO, DETR | Stable |
| VLM Training | Vision-Language Models | Beta |
| Tabular | XGBoost, sklearn | Stable |
| Sentence Transformers | Semantic similarity | Stable |
| Extractive QA | SQuAD format | Stable |

---

## Configuration Example

```yaml
task: llm-sft
base_model: meta-llama/Llama-3.2-1B
project_name: my-finetune

data:
  path: your-dataset
  train_split: train
  auto_convert_dataset: true
  chat_template: llama3

params:
  epochs: 3
  batch_size: 4
  lr: 2e-5
  peft: true
  lora_r: 16
  lora_alpha: 32
  quantization: int4
  mixed_precision: bf16

# Optional: hyperparameter sweep
sweep:
  enabled: true
  backend: optuna
  n_trials: 10
  metric: eval_loss
```

---

## Documentation

- [Interactive Wizard Guide](docs/interactive_wizard.md)
- [Dataset Formats & Conversion](docs/dataset_formats.md)
- [Trainer Reference](docs/trainers/README.md)
- [Python API](docs/api/PYTHON_API.md)
- [RL API Reference](docs/reference/RL_API_REFERENCE.md)

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

Based on [AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced) by Hugging Face.

---

<p align="center">
  <a href="https://monostate.ai">Monostate AI</a>
</p>
