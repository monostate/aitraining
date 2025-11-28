# AITraining Documentation

Welcome to the AITraining (AutoTrain Advanced) documentation. This comprehensive guide covers CLI usage, Python API, and all training features.

## 📚 Documentation Structure

### 🖥️ [CLI Documentation](cli/)
- **[CLI Reference](cli/README.md)** - Complete list of all 111 CLI parameters with examples
- **[Parameter Compatibility](cli/PARAMETER_COMPATIBILITY.md)** - Which parameters work with which trainers

### 🐍 [API Documentation](api/)
- **[Python API](api/PYTHON_API.md)** - Comprehensive Python API reference with 1800+ lines
- **[API Integration](api/INTEGRATION.md)** - How CLI and API work together
- **[Inference API](api/INFERENCE_API_USAGE.md)** - Using completers for inference

### 🚂 [Trainer Guides](trainers/)
- **[PPO Training](trainers/PPO.md)** - Reinforcement learning with PPO
- Additional trainer guides coming soon (SFT, DPO, ORPO, Reward)

### 📖 [Reference](reference/)
- **[Parameter Details](reference/PARAMETERS.md)** - Detailed parameter explanations
- **[RL API Reference](reference/RL_API_REFERENCE.md)** - RL-specific API documentation
- **[Test Results](reference/TEST_RESULTS.md)** - Test coverage and results

### 🖼️ UI Documentation
- **[Web UI Form Refactor](ui_form_refactor.md)** - Form renderer with grouped parameters, collapsible panels, and PPO validation

### 📦 [Archive](archive/)
Development and historical documentation

---

## 🚀 Quick Start

### Installation
```bash
pip install autotrain-advanced
```

### Basic Usage

#### CLI Command Structure
```bash
# Note: The CLI command is now 'aitraining' (renamed from 'autotrain')
aitraining llm --train \
  --model gpt2 \
  --data-path ./data \
  --project-name my_model \
  --trainer sft \
  --epochs 3
```

#### Python API
```python
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.clm import train

params = LLMTrainingParams(
    model="gpt2",
    data_path="./data",
    project_name="my_model",
    trainer="sft",
    epochs=3
)

trainer = train(params)
```

---

## 📊 Available Commands

AITraining supports multiple model types and tasks:

### Language Models (`aitraining llm`)
| Trainer | Description | Use Case |
|---------|-------------|----------|
| `default` | Standard causal language modeling | General fine-tuning |
| `sft` | Supervised Fine-Tuning | Instruction following |
| `dpo` | Direct Preference Optimization | Alignment without RL |
| `orpo` | Odds Ratio Preference Optimization | Alternative to DPO |
| `ppo` | Proximal Policy Optimization | RLHF with reward model |
| `reward` | Reward model training | For use with PPO |

### Other Training Tasks
| Command | Description |
|---------|-------------|
| `aitraining text-classification` | Text classification models |
| `aitraining text-regression` | Text regression models |
| `aitraining token-classification` | NER/Token classification |
| `aitraining seq2seq` | Sequence-to-sequence models |
| `aitraining sentence-transformers` | Sentence embedding models |
| `aitraining image-classification` | Image classification |
| `aitraining image-regression` | Image regression |
| `aitraining object-detection` | Object detection models |
| `aitraining extractive-qa` | Question answering |
| `aitraining vlm` | Vision-language models |
| `aitraining tabular` | Tabular data models |

### Utilities
| Command | Description |
|---------|-------------|
| `aitraining app` | Launch web UI |
| `aitraining api` | Launch REST API server |
| `aitraining tools` | Various utility tools |
| `aitraining setup` | Setup and configuration |
| `aitraining spacerunner` | Deploy to HuggingFace Spaces |

---

## 🔑 Key Features

### Core Capabilities
- ✅ **111 Configurable Parameters** - Fine control over training
- ✅ **PEFT/LoRA Support** - Efficient fine-tuning
- ✅ **Flash Attention 2** - Faster training
- ✅ **Mixed Precision** - FP16/BF16 support
- ✅ **Quantization** - INT4/INT8 training

### Advanced Features
- ✅ **Knowledge Distillation** - Transfer knowledge from larger models
- ✅ **Hyperparameter Sweeps** - Automatic optimization
- ✅ **Enhanced Evaluation** - Multiple metrics (BLEU, ROUGE, etc.)
- ✅ **Multi-Objective RL** - Balance multiple objectives
- ✅ **Custom Loss Functions** - Flexible training objectives

---

## 📝 Important Notes

### CLI Command Name Change
The CLI command has been renamed from `autotrain` to `aitraining`. However:
- Python package name remains `autotrain`
- Import paths remain `from autotrain...`
- Only the CLI entry point changed

### Correct Usage
```bash
# CLI command (NEW)
aitraining llm --train ...

# Python imports (UNCHANGED)
from autotrain.trainers.clm import train
```

### Deprecated Features
- `--trainer distillation` is deprecated. Use `--trainer sft --use-distillation` instead
- `advanced_mode` and `manual_sampling` flags have been removed

---

## 📊 Parameter Count by Category

| Category | Count | Description |
|----------|-------|-------------|
| Core Training | 20 | Basic training parameters |
| Data Configuration | 10 | Data loading and processing |
| Optimization | 15 | Learning rate, optimizer settings |
| Model Configuration | 12 | Architecture and size settings |
| PEFT/LoRA | 8 | Parameter-efficient fine-tuning |
| Evaluation | 10 | Metrics and validation |
| Reinforcement Learning | 16 | PPO and reward modeling |
| Advanced Features | 20 | Distillation, sweeps, etc. |
| Total | **111** | All configurable parameters |

---

## 🔗 Quick Links

- [Complete CLI Reference](cli/README.md) - All parameters with examples
- [Python API Guide](api/PYTHON_API.md) - Programmatic usage
- [Parameter Compatibility](cli/PARAMETER_COMPATIBILITY.md) - What works with what
- [Test Coverage](reference/TEST_RESULTS.md) - Testing documentation

---

## 💡 Getting Help

1. Check the relevant documentation section above
2. Review the [Parameter Compatibility Guide](cli/PARAMETER_COMPATIBILITY.md)
3. Look at test examples in the codebase
4. Open an issue on GitHub for bugs or questions

---

## 🔧 Building Documentation (for contributors)

To build HuggingFace documentation site:

```bash
# Install doc builder
pip install git+https://github.com/huggingface/doc-builder@main

# Build docs
doc-builder build autotrain docs/source/ --build_dir ~/tmp/test-build

# Preview docs
doc-builder preview autotrain docs/source/
```

---

*Last Updated: October 2024*
*Documentation Version: 2.0.0*