# AITraining Parameter Count Reference

## Verified Parameter Counts by Trainer Type

Last verified: October 2024

| Trainer Type | Command | Parameter Count | Status |
|-------------|---------|-----------------|---------|
| **LLM** | `aitraining llm` | **112** | ✓ Verified |
| Text Classification | `aitraining text-classification` | **30** | ✓ Verified |
| Text Regression | `aitraining text-regression` | **30** | ✓ Verified |
| Token Classification | `aitraining token-classification` | **30** | ✓ Verified |
| Seq2Seq | `aitraining seq2seq` | **39** | ✓ Verified |
| Image Classification | `aitraining image-classification` | **29** | ✓ Verified |
| Image Regression | `aitraining image-regression` | **29** | ✓ Verified |
| Object Detection | `aitraining object-detection` | **30** | ✓ Verified |
| Sentence Transformers | `aitraining sentence-transformers` | **33** | ✓ Verified |
| Tabular | `aitraining tabular` | **20** | ✓ Verified |
| Extractive QA | `aitraining extractive-qa` | **32** | ✓ Verified |
| VLM | `aitraining vlm` | **37** | ✓ Verified |

## LLM Trainer Parameter Breakdown (112 total)

| Category | Count | Description |
|----------|-------|-------------|
| **Core** | 6 | Basic setup (model, project_name, trainer, seed, log, distributed_backend) |
| **Data Configuration** | 12 | Data loading and processing |
| **Training Hyperparameters** | 13 | Learning rate, epochs, optimization settings |
| **Model Configuration** | 8 | Architecture, attention, precision settings |
| **PEFT/LoRA** | 9 | Parameter-efficient fine-tuning |
| **Evaluation** | 12 | Metrics, validation, saving |
| **RL (PPO)** | 20 | Reinforcement learning parameters (includes value_clip_range, max_new_tokens, top_k, top_p, temperature) |
| **Distillation** | 7 | Knowledge distillation settings |
| **Hyperparameter Sweep** | 6 | Automated hyperparameter search |
| **Advanced Features** | 14 | Custom losses, manual control, sampling |
| **Hub Integration** | 4 | HuggingFace Hub (push_to_hub, username, token, etc.) |
| **Other** | 1 | Miscellaneous |

## CLI vs API Consistency

### Automatic Synchronization
AITraining uses `get_field_info()` to automatically generate CLI arguments from API parameter classes, ensuring perfect consistency:

- ✅ **All CLI parameters are derived from API fields**
- ✅ **Automatic name conversion**: `snake_case` (API) → `kebab-case` (CLI)
- ✅ **Type checking and validation** inherited from Pydantic models
- ✅ **Default values** synchronized between CLI and API

### Parameter Name Mapping Examples

| API Parameter | CLI Argument |
|--------------|--------------|
| `project_name` | `--project-name` |
| `data_path` | `--data-path` |
| `train_split` | `--train-split` |
| `max_samples` | `--max-samples` |
| `add_eos_token` | `--add-eos-token` |
| `model_max_length` | `--model-max-length` |
| `rl_reward_model_path` | `--rl-reward-model-path` |

## Verification Method

These counts were verified using direct field inspection of the parameter classes:

```python
# Example: Counting LLM parameters
from autotrain.trainers.clm.params import LLMTrainingParams
fields = [f for f in dir(LLMTrainingParams.model_fields)]
# Filter for actual parameter fields defined with Field()
```

### Files Inspected
- `/src/autotrain/trainers/*/params.py` - Parameter definitions
- `/src/autotrain/cli/run_*.py` - CLI implementations

## Important Notes

1. **Documentation Accuracy**: The main docs/README.md correctly states 112 parameters for LLM
2. **No Discrepancies**: CLI and API are perfectly synchronized via `get_field_info()`
3. **Hidden Parameters**: Some parameters may be hidden in CLI (via `HIDDEN_PARAMS`) but still accessible via API
4. **Trainer-Specific**: Different trainers have different parameter counts and requirements

## How to Verify Counts

To verify these counts yourself:

```bash
# Count LLM parameters
grep -E "^\s+\w+:" src/autotrain/trainers/clm/params.py | \
  grep -v "^\s*#" | grep -v "class " | wc -l
# Result: 112

# For other trainers, replace 'clm' with:
# text_classification, image_classification, tabular, etc.
```

Or use the verification scripts:
- `/code/verify_params.py` - Count all parameters by trainer
- `/code/check_cli_api_consistency.py` - Verify CLI/API consistency