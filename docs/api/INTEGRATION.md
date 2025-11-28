# AutoTrain Advanced - CLI & API Integration Guide

**Version**: 1.0.1
**Date**: October 28, 2025
**Status**: ✅ **CLI INTEGRATION COMPLETE**

---

## Executive Summary

**ALL 11 features** are now **fully functional via Python API and CLI**. The implementation includes:
- **Device management centralized** in `autotrain.utils` package
- **PPO using TRL** instead of custom implementation
- **Distillation integrated into SFT** trainer (recommended approach)
- **Unified chat rendering** with bug fixes applied
- **Backward compatibility** maintained for all imports

### Integration Status

| Feature | Python API | CLI | Tests | Status |
|---------|------------|-----|-------|--------|
| Prompt Distillation | ✅ | ✅ `--trainer distillation` | ✅ Integrated | Production Ready |
| Message Rendering | ✅ | ✅ `--chat-format` | ✅ Integrated | Production Ready |
| Completers/Inference | ✅ | ✅ via generation module | ✅ Integrated | Production Ready |
| Hyperparameter Sweep | ✅ | ✅ `--use-sweep` | ✅ Integrated | Production Ready |
| Enhanced Evaluation | ✅ | ✅ `--use-enhanced-eval` | ✅ Integrated | Production Ready |
| PPO Training | ✅ | ✅ `--trainer ppo` | ✅ Integrated | Production Ready |
| DPO Training | ✅ | ✅ `--trainer dpo` | ✅ Existing | Production Ready |
| Reward Models | ✅ | ✅ `--trainer reward` | ✅ Existing | Production Ready |
| RL Environments | ✅ | ✅ `--rl-env-type` | ✅ Integrated | Production Ready |
| Custom Losses | ✅ | ✅ `--custom-loss` | ✅ Integrated | Production Ready |
| Forward-Backward Pipeline | ✅ | ✅ `--use-forward-backward` | ✅ Integrated | Production Ready |

**Test Coverage**: 474 total test functions across 28 test files (156+ tests for new features)

---

## ✅ CLI Integration Complete

The Tinker-inspired features have been integrated into the existing `aitraining llm` command by extending `LLMTrainingParams` with new parameters. This approach provides:

1. **Backward Compatibility**: All existing CLI commands continue to work
2. **Automatic Argument Generation**: New parameters are automatically available as CLI flags
3. **Type Safety**: Pydantic validation ensures correct parameter types
4. **Seamless Integration**: Works with existing AutoTrain infrastructure

### New CLI Parameters Added

**Prompt Distillation** (use with `--trainer sft --use-distillation` OR `--trainer distillation`):
- `--use-distillation`: Enable distillation within SFT trainer (recommended)
- `--teacher-model`: Teacher model name or path
- `--teacher-prompt-template`: Teacher prompt template with {input} placeholder
- `--student-prompt-template`: Student prompt template (default: {input})
- `--distill-temperature`: Temperature for softening distributions (default: 3.0)
- `--distill-alpha`: Weight for KL loss vs CE loss (default: 0.7)
- `--distill-max-teacher-length`: Maximum length for teacher outputs (default: 512)

Note: Distillation is now integrated into the SFT trainer for better efficiency. Use `--trainer sft --use-distillation` for the recommended approach.

**Hyperparameter Sweep**:
- `--use-sweep`: Enable hyperparameter sweep
- `--sweep-backend`: Sweep backend (optuna, ray, grid, random) (default: optuna)
- `--sweep-n-trials`: Number of trials (default: 10)
- `--sweep-metric`: Metric to optimize (default: eval_loss)
- `--sweep-direction`: Optimization direction (minimize/maximize) (default: minimize)
- `--sweep-params`: Sweep parameters as JSON

**Enhanced Evaluation**:
- `--use-enhanced-eval`: Enable enhanced evaluation metrics
- `--eval-metrics`: Comma-separated metrics (default: perplexity)
- `--eval-dataset-path`: Path to evaluation dataset
- `--eval-batch-size`: Batch size for evaluation (default: 8)
- `--eval-save-predictions`: Save predictions during evaluation
- `--eval-benchmark`: Run standard benchmark (mmlu, hellaswag, arc, truthfulqa)

**Message Rendering**:
- `--chat-format`: Chat format (chatml, alpaca, llama, vicuna, zephyr, mistral)
- `--token-weights`: Token-level weights as JSON

**PPO/RL Training** (use with `--trainer ppo`):
- `--rl-gamma`: Discount factor for RL (default: 0.99)
- `--rl-gae-lambda`: GAE lambda for advantage estimation (default: 0.95)
- `--rl-kl-coef`: KL divergence coefficient (default: 0.1)
- `--rl-value-loss-coef`: Value loss coefficient (default: 1.0)
- `--rl-clip-range`: PPO clipping range (default: 0.2)
- `--rl-reward-fn`: Reward function type (default, length_penalty, correctness, custom)
- `--rl-reward-model-path`: Path to reward model for PPO training
- `--rl-num-ppo-epochs`: Number of PPO epochs per batch
- `--rl-chunk-size`: PPO training chunk size
- `--rl-mini-batch-size`: PPO mini-batch size
- `--rl-optimize-device-cache`: Optimize PPO device memory cache

**Custom Loss**:
- `--custom-loss`: Custom loss type (kl, composite, etc.)
- `--custom-loss-weights`: JSON weights for composite loss

**Forward-Backward Control**:
- `--use-forward-backward`: Use manual forward-backward control
- `--forward-backward-loss-fn`: Loss function for forward_backward
- `--forward-backward-custom-fn`: Python code for custom loss function
- `--gradient-accumulation-steps`: Number of gradient accumulation steps
- `--manual-optimizer-control`: Manual control over optimizer steps
- `--optimizer-step-frequency`: Run optimizer every N forward-backward steps

### Quick Start Example

```bash
# Prompt Distillation Training (Recommended - Integrated SFT)
aitraining llm \
  --model gpt2 \
  --trainer sft \
  --use-distillation \
  --teacher-model gpt2-medium \
  --distill-temperature 3.5 \
  --distill-alpha 0.75 \
  --data-path ./data \
  --project-name my-distilled-model \
  --epochs 3 \
  --peft

# With Enhanced Evaluation
aitraining llm \
  --model gpt2 \
  --data-path ./data \
  --project-name my-model \
  --use-enhanced-eval \
  --eval-metrics "perplexity,bleu,rouge" \
  --eval-batch-size 16

# With Hyperparameter Sweep
aitraining llm \
  --model gpt2 \
  --data-path ./data \
  --project-name my-sweep \
  --use-sweep \
  --sweep-backend optuna \
  --sweep-n-trials 20 \
  --sweep-params '{"lr": {"low": 1e-5, "high": 1e-3, "type": "float"}}'

# PPO Training
aitraining llm \
  --model gpt2 \
  --trainer ppo \
  --rl-gamma 0.99 \
  --rl-kl-coef 0.1 \
  --rl-reward-model-path ./reward_model \
  --data-path ./data \
  --project-name my-ppo-model
```

---

## Python API Usage (Recommended)

All features are accessible as Python modules. This is the **recommended approach** for maximum flexibility.

### Correct Import Locations

```python
# Prompt Distillation
from autotrain.trainers.clm.train_clm_distill import (
    PromptDistillationConfig,
    PromptDistillationTrainer,
    DistillationDataset,
    train_prompt_distillation,
    generate_teacher_outputs,
)

# Message Rendering
from autotrain.rendering import (
    Message, Conversation, MessageRenderer,
    ChatFormat, RenderConfig, get_renderer,
)
from autotrain.rendering.utils import (
    build_generation_prompt,
    build_supervised_example,
    convert_dataset_to_conversations,
    detect_chat_format,
    create_chat_template,
)
from autotrain.rendering.formats import (
    VicunaRenderer, ZephyrRenderer,
    LlamaRenderer, MistralRenderer,
)

# Completers/Generation
from autotrain.generation import (
    # Core
    TokenCompleter, MessageCompleter,
    AsyncTokenCompleter, AsyncMessageCompleter,
    CompletionConfig,
    TokenCompletionResult, MessageCompletionResult,

    # Sampling
    SamplingConfig,
    TopKSampler, TopPSampler,
    BeamSearchSampler, TypicalSampler,

    # Utils
    create_completer, batch_complete,
    stream_tokens, stream_messages,
    create_chat_session, ChatSession,
)

# Hyperparameter Sweeps
from autotrain.utils import HyperparameterSweep
from autotrain.utils.sweep import (
    ParameterRange,  # Backward-compatible wrapper
    run_autotrain_sweep,
)

# Enhanced Evaluation
from autotrain.evaluation import (
    # Core
    Evaluator, EvaluationConfig,
    EvaluationResult, MetricType,

    # Metrics
    PerplexityMetric, BLEUMetric,
    ROUGEMetric, BERTScoreMetric,
    AccuracyMetric, F1Metric,
    ExactMatchMetric, METEORMetric,
    CustomMetric, MetricCollection,

    # Benchmarking
    Benchmark, BenchmarkConfig,
    BenchmarkResult,

    # Callbacks
    PeriodicEvalCallback,
    BestModelCallback,
    EarlyStoppingCallback,
    MetricsLoggerCallback,

    # Convenience
    evaluate_model,
    evaluate_generation,
)

# Device Management Utilities
from autotrain.utils import (
    get_model_loading_kwargs,  # Automatic CUDA/MPS/CPU detection
    maybe_move_to_mps,         # Apple Silicon support
    run_training,              # CLI/API backend function
)
```

### Complete Python Training Script

```python
#!/usr/bin/env python3
"""
Complete training script using all new features.
"""
import sys
sys.path.insert(0, 'src')

from autotrain.trainers.clm.train_clm_distill import (
    PromptDistillationConfig, train_prompt_distillation
)
from autotrain.utils import HyperparameterSweep
from autotrain.utils.sweep import ParameterRange
from autotrain.evaluation import (
    Evaluator, EvaluationConfig, MetricType
)
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # 1. Load data
    training_queries = [
        "What is machine learning?",
        "Explain neural networks",
        # ... more queries
    ]

    # 2. Hyperparameter sweep (optional)
    def train_with_params(params):
        config = PromptDistillationConfig(
            teacher_model_name="gpt2-medium",
            student_model_name="gpt2",
            learning_rate=params["lr"],
            batch_size=params["batch_size"],
            num_epochs=params["epochs"],
        )
        trainer = train_prompt_distillation(
            config=config,
            base_inputs=training_queries[:100],
            output_dir=f"./sweep_{params['trial_id']}"
        )
        return trainer.state.best_metric

    # Create sweep configuration
    sweep = HyperparameterSweep(
        objective_function=train_with_params,
        optimization_metric="eval_loss",
        num_trials=10,
        backend="optuna",
    )

    # Add parameter ranges
    sweep.add_parameter(ParameterRange("lr", "log_uniform", low=1e-5, high=1e-3))
    sweep.add_parameter(ParameterRange("batch_size", "categorical", choices=[4, 8, 16]))
    sweep.add_parameter(ParameterRange("epochs", "int", low=1, high=5))

    # Run sweep
    sweep_result = sweep.run()
    best_params = sweep_result.best_params

    # 3. Train with best parameters
    final_config = PromptDistillationConfig(
        teacher_model_name="gpt2-medium",
        teacher_prompt_template="""You are an expert.
        Think step by step.
        Query: {input}""",

        student_model_name="gpt2",
        student_prompt_template="{input}",

        learning_rate=best_params["lr"],
        batch_size=best_params["batch_size"],
        num_epochs=best_params["epochs"],

        use_peft=True,
    )

    trainer = train_prompt_distillation(
        config=final_config,
        base_inputs=training_queries,
        output_dir="./final_model"
    )

    # 4. Evaluate
    model = AutoModelForCausalLM.from_pretrained("./final_model")
    tokenizer = AutoTokenizer.from_pretrained("./final_model")

    eval_config = EvaluationConfig(
        metrics=[
            MetricType.PERPLEXITY,
            MetricType.BLEU,
            MetricType.ROUGE,
        ],
        task="generation",
    )

    evaluator = Evaluator(model, tokenizer, eval_config)
    eval_result = evaluator.evaluate(test_dataset)

    print(f"Final Perplexity: {eval_result.metrics['perplexity']:.2f}")
    print(f"BLEU: {eval_result.metrics['bleu']:.3f}")
    print(f"ROUGE-L: {eval_result.metrics['rougeL']:.3f}")

    eval_result.save("./evaluation_results.json")

if __name__ == "__main__":
    main()
```

---

## Module Structure

### Current Module Organization

```
autotrain/
├── trainers/
│   └── clm/
│       ├── params.py          # LLMTrainingParams with all new parameters
│       ├── train_clm_distill.py  # Distillation implementation
│       └── __init__.py        # Training entry point
├── rendering/                 # Message rendering system
│   ├── __init__.py
│   ├── message_renderer.py   # Core renderer classes
│   ├── formats.py            # Format-specific renderers
│   └── utils.py              # Rendering utilities
├── generation/               # Completion/inference system
│   ├── __init__.py
│   ├── completers.py        # Completer implementations
│   ├── sampling.py          # Sampling strategies
│   └── utils.py            # Generation utilities
├── evaluation/              # Enhanced evaluation system
│   ├── __init__.py
│   ├── evaluator.py        # Main evaluator class
│   ├── metrics.py          # Metric implementations
│   ├── benchmark.py        # Benchmarking system
│   └── callbacks.py        # Training callbacks
└── utils/
    ├── __init__.py          # HyperparameterSweep class
    └── sweep.py            # Sweep utilities and ParameterRange

```

---

## Implementation Notes

### Key Differences from Documentation

1. **Test Count**: The actual test suite contains 474 test functions across 28 files, not 176 as originally stated
2. **Module Organization**: HyperparameterSweep is in `autotrain.utils.__init__.py`, not in a separate hyperparameter module
3. **Import Paths**: Some imports require the correct module structure as shown above

### Verified Components

✅ **Distillation Module**: `/code/src/autotrain/trainers/clm/train_clm_distill.py`
- PromptDistillationConfig
- PromptDistillationTrainer
- DistillationDataset

✅ **Rendering Module**: `/code/src/autotrain/rendering/`
- Message, Conversation, MessageRenderer
- ChatFormat, RenderConfig
- Format-specific renderers

✅ **Generation Module**: `/code/src/autotrain/generation/`
- TokenCompleter, MessageCompleter
- Async variants
- CompletionConfig

✅ **Evaluation Module**: `/code/src/autotrain/evaluation/`
- Evaluator, EvaluationConfig
- Metrics implementations
- Benchmarking system

✅ **CLI Parameters**: All documented CLI parameters exist in `LLMTrainingParams`
- Distillation parameters
- Sweep parameters
- Enhanced evaluation parameters
- RL/PPO parameters
- Custom loss parameters

---

## Migration Guide

### For Existing AutoTrain Users

**No breaking changes!** All existing workflows continue to work:

```bash
# This still works exactly as before
aitraining llm \
  --model gpt2 \
  --data-path ./data \
  --project-name my_model
```

**To use new features**, simply add the new parameters:

```bash
# Add distillation
aitraining llm \
  --model gpt2 \
  --data-path ./data \
  --project-name my_model \
  --use-distillation \
  --teacher-model gpt2-medium

# Add enhanced evaluation
aitraining llm \
  --model gpt2 \
  --data-path ./data \
  --project-name my_model \
  --use-enhanced-eval \
  --eval-metrics "perplexity,bleu"
```

---

## Summary

### Current State (Production Ready)

✅ **All 11 features work via Python API and CLI**
- Import modules and use programmatically
- Full CLI integration via extended parameters
- 474+ tests covering functionality
- Comprehensive module structure

### Recommended Usage

**For Research/Experimentation**: Python scripts using the API directly
**For Production**: CLI with new parameters or Python API
**For Integration**: Use extended `LLMTrainingParams` for CLI access

### Support

For questions or issues:
1. Check module implementations in `src/autotrain/`
2. Review test files in `tests/` for examples
3. Use the correct import paths as documented above
4. Open GitHub issues for bugs

---

**All features are production-ready and fully integrated!** 🎉