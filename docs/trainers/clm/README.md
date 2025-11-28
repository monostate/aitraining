# AutoTrain Advanced - Trainer Documentation

Complete documentation for all training methods available in AutoTrain Advanced.

## Available Trainers

### 📚 [Default/Generic CLM Training](./Default.md)
Pure language modeling with next-token prediction. For domain adaptation and continuous pre-training.

**Use when:** You have raw text data for pre-training
**Key features:** Simple, minimal preprocessing, domain adaptation
**Example:** Training on domain-specific corpus

---

### 🎯 [Supervised Fine-Tuning (SFT)](./SFT.md)
Standard supervised learning from labeled examples. The foundation for most fine-tuning tasks.

**Use when:** You have labeled input-output pairs
**Key features:** Simple, stable, well-understood
**Example:** Training on Q&A pairs, instructions, or conversations

---

### 🧬 [Prompt Distillation](./Distillation.md)
Teach models to internalize complex prompting strategies for efficient inference.

**Use when:** You want to reduce inference costs
**Key features:** Prompt internalization, knowledge transfer
**Example:** Compress GPT-4 prompts into GPT-2 behavior

---

### 🏆 [Direct Preference Optimization (DPO)](./DPO.md)
Align models with human preferences without training a separate reward model.

**Use when:** You have preference data (chosen vs rejected)
**Key features:** More stable than PPO, no reward model needed
**Example:** Improving response quality based on human feedback

---

### ⚖️ [Odds Ratio Preference Optimization (ORPO)](./ORPO.md)
Single-stage training combining SFT and preference learning.

**Use when:** Starting from scratch with preference data
**Key features:** No reference model needed, memory efficient
**Example:** Training aligned model in one step

---

### 🎮 [Proximal Policy Optimization (PPO)](./PPO.md)
Full reinforcement learning with custom reward functions.

**Use when:** You need maximum flexibility with rewards
**Key features:** Most flexible, supports any reward function
**Example:** Complex optimization with custom metrics

---

### 🏅 [Reward Model Training](./Reward.md)
Train models to score response quality for RLHF pipelines.

**Use when:** Building PPO/RLHF systems
**Key features:** Learn human preferences, foundation for PPO
**Example:** Creating scoring functions from human feedback

---

## Quick Comparison

| Trainer | Complexity | Stability | Flexibility | Memory Usage | Use Case |
|---------|------------|-----------|-------------|--------------|----------|
| **Default** | ⭐ Simplest | ⭐⭐⭐ Very Stable | ⭐ Basic | ⭐ Low | Language modeling |
| **SFT** | ⭐ Simple | ⭐⭐⭐ Very Stable | ⭐ Basic | ⭐ Low | Supervised learning |
| **Distillation** | ⭐⭐ Moderate | ⭐⭐⭐ Very Stable | ⭐⭐ Moderate | ⭐⭐ Medium | Prompt compression |
| **DPO** | ⭐⭐ Moderate | ⭐⭐⭐ Very Stable | ⭐⭐ Moderate | ⭐⭐ Medium | Preference alignment |
| **ORPO** | ⭐⭐ Moderate | ⭐⭐⭐ Very Stable | ⭐⭐ Moderate | ⭐ Low | Single-stage alignment |
| **PPO** | ⭐⭐⭐ Complex | ⭐⭐ Moderate | ⭐⭐⭐ High | ⭐⭐⭐ High | Complex rewards |
| **Reward** | ⭐⭐ Moderate | ⭐⭐⭐ Very Stable | ⭐⭐ Moderate | ⭐ Low | Scoring function |

## Training Pipeline Examples

### Basic Fine-tuning
```
Base Model → SFT → Deployed Model
```

### Preference Alignment (Simple)
```
Base Model → SFT → DPO → Deployed Model
```

### Single-Stage Alignment
```
Base Model → ORPO → Deployed Model
```

### Full RLHF Pipeline
```
Base Model → SFT → Reward Model Training
                ↓
            PPO with Reward Model → Deployed Model
```

## CLI Quick Reference

```bash
# SFT
aitraining llm --trainer sft --model gpt2 --data-path ./data

# DPO
aitraining llm --trainer dpo --model gpt2 --data-path ./preferences.json

# ORPO
aitraining llm --trainer orpo --model gpt2 --data-path ./preferences.json

# PPO
aitraining llm --trainer ppo --model gpt2 --rl-reward-model-path ./reward_model

# Reward Model
aitraining llm --trainer reward --model gpt2 --data-path ./preferences.json
```

> Tip: Preview trainer-specific parameters in grouped help without changing the runtime trainer:
```bash
aitraining llm --preview-trainer sft --help
aitraining llm --preview-trainer dpo --help
```

## Choosing the Right Trainer

### Decision Tree

```
Do you have labeled examples?
├── No → Collect data first
└── Yes → Do you have preference pairs?
    ├── No → Use SFT
    └── Yes → Do you need custom rewards?
        ├── No → Do you have a reference model?
        │   ├── No → Use ORPO
        │   └── Yes → Use DPO
        └── Yes → Train Reward Model → Use PPO
```

### By Use Case

**Chat/Instruction Following:** SFT → DPO
**Code Generation:** SFT with quality data
**Creative Writing:** ORPO for balanced creativity
**Safety Alignment:** DPO or PPO with safety rewards
**Task-Specific:** SFT for most cases

## Common Workflows

### 1. Basic Fine-tuning
```python
# Simple SFT for task-specific model
config = LLMTrainingParams(
    trainer="sft",
    model="gpt2",
    data_path="./data.json",
    project_name="task-model"
)
```

### 2. Preference Alignment
```python
# Step 1: SFT
sft_config = LLMTrainingParams(trainer="sft", ...)
sft_model = train(sft_config)

# Step 2: DPO
dpo_config = LLMTrainingParams(
    trainer="dpo",
    model="./sft_model",
    model_ref="./sft_model",
    ...
)
aligned_model = train(dpo_config)
```

### 3. Full RLHF
```python
# Step 1: Train reward model
reward_config = LLMTrainingParams(trainer="reward", ...)
reward_model = train(reward_config)

# Step 2: PPO training
ppo_config = LLMTrainingParams(
    trainer="ppo",
    rl_reward_model_path="./reward_model",
    ...
)
rlhf_model = train(ppo_config)
```

## Best Practices

1. **Start Simple:** Begin with SFT, add complexity as needed
2. **Data Quality:** Better data > more complex training
3. **Iterative:** Test each stage before adding more
4. **Monitor:** Track metrics at every step
5. **Validate:** Always use held-out test sets

## Resources

- [CLI Parameters Reference](../cli/README.md)
- [Python API Reference](../api/README.md)
- [Integration Guide](../api/INTEGRATION.md)
- [Examples Repository](https://github.com/huggingface/autotrain-advanced/examples)

## Other Trainer Categories

For documentation on other trainer types, see:
- [**NLP Trainers**](../nlp/) - Text classification, NER, QA, embeddings
- [**Vision Trainers**](../vision/) - Image classification, object detection, VLM
- [**Tabular Trainer**](../tabular/Tabular.md) - XGBoost, LightGBM, CatBoost
- [**Complete Overview**](../OVERVIEW.md) - All 14 trainer categories
- [**Master Index**](../README.md) - Quick navigation to all trainers

## Getting Help

- 📖 Check individual trainer guides for detailed information
- 🐛 Report issues on [GitHub](https://github.com/huggingface/autotrain-advanced/issues)
- 💬 Join discussions on [HuggingFace Forums](https://discuss.huggingface.co)
- 📧 Contact support for enterprise features