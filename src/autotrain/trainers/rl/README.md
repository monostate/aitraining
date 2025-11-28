# AutoTrain Advanced RL Module

## Overview

This module implements advanced reinforcement learning techniques for fine-tuning language models, providing a **local, PyTorch-native alternative** to cloud-based solutions like Tinker.

## Features

### 🎯 Core Capabilities

- **PPO (Proximal Policy Optimization)** - Stable policy gradient method
- **DPO (Direct Preference Optimization)** - Preference learning without RL
- **RLHF** - Reinforcement Learning from Human Feedback
- **Multi-Objective Training** - Optimize for multiple goals simultaneously
- **Custom Reward Functions** - Define your own reward signals
- **Async Training** - Efficient forward-backward pipeline

### 🏗️ Architecture

```
rl/
├── __init__.py                 # Public API exports
├── ppo.py                      # PPO trainer implementation
├── dpo.py                      # Direct Preference Optimization
├── forward_backward.py         # Async training pipeline
├── environments.py             # RL environments for text generation
└── reward_model.py             # Reward modeling for RLHF
```

## Installation

```bash
pip install torch transformers peft
```

## Quick Examples

### PPO Training

```python
from autotrain.trainers.rl import PPOTrainer, PPOConfig
from autotrain.trainers.rl.environments import TextGenerationEnv

# Setup
config = PPOConfig(learning_rate=1e-5)
env = TextGenerationEnv(tokenizer, prompts, reward_fn=custom_reward)

# Train
trainer = PPOTrainer(config, model, tokenizer)
trainer.train(env)
```

### DPO Training

```python
from autotrain.trainers.rl import DPOTrainer, DPOConfig

config = DPOConfig(beta=0.1)
trainer = DPOTrainer(config, model, tokenizer)
trainer.train_on_preferences(prompts, chosen, rejected)
```

### Multi-Objective Rewards

```python
from autotrain.trainers.rl.environments import MultiObjectiveRewardEnv

env = MultiObjectiveRewardEnv(
    tokenizer=tokenizer,
    prompts=prompts,
    reward_components={
        "accuracy": accuracy_reward,
        "fluency": fluency_reward,
        "safety": safety_reward,
    },
    reward_weights={"accuracy": 1.0, "fluency": 0.3, "safety": 0.5}
)
```

## Module Documentation

### `ppo.py` - PPO Trainer

Implements Proximal Policy Optimization for language models.

**Key Classes:**
- `PPOConfig` - Configuration for PPO training
- `PPOTrainer` - Main trainer class
- `PPOBuffer` - Experience buffer for training

**Features:**
- Clipped surrogate objective
- Value function training
- Advantage estimation (GAE)
- KL divergence constraints

### `dpo.py` - Direct Preference Optimization

Implements DPO for preference-based fine-tuning without explicit reward modeling.

**Key Classes:**
- `DPOConfig` - Configuration for DPO
- `DPOTrainer` - Main trainer class

**Features:**
- Direct optimization from preferences
- No reward model needed
- Implicit KL regularization
- Efficient batch processing

### `forward_backward.py` - Async Training Pipeline

Provides efficient forward-backward pass management inspired by Tinker.

**Key Classes:**
- `ForwardBackwardPipeline` - Async gradient computation
- `AsyncTrainingClient` - Non-blocking training operations
- `AsyncTrainingFuture` - Future-based API

**Features:**
- Queue-based gradient accumulation
- Parallel forward passes
- Custom loss function support
- Memory-efficient training

### `environments.py` - RL Environments

Text generation environments for reinforcement learning.

**Key Classes:**
- `TextGenerationEnv` - Basic text generation environment
- `MultiObjectiveRewardEnv` - Multi-objective optimization
- `PreferenceComparisonEnv` - Preference learning environment

**Factory Functions:**
- `create_math_problem_env()` - Math problem solving
- `create_code_generation_env()` - Code generation tasks

### `reward_model.py` - Reward Modeling

Reward models for RLHF training.

**Key Classes:**
- `RewardModel` - Basic reward model
- `PairwiseRewardModel` - Pairwise comparison model
- `MultiObjectiveRewardModel` - Multi-objective rewards
- `RewardModelTrainer` - Training utilities

**Features:**
- Multiple pooling strategies
- LoRA support for efficient fine-tuning
- Preference learning losses
- Bradley-Terry models

## Advanced Usage

### Custom Environments

```python
from autotrain.trainers.rl.environments import RLEnvironment

class MyCustomEnv(RLEnvironment):
    def reset(self):
        # Initialize episode
        return initial_observation

    def step(self, action):
        # Process action
        return StepResult(reward, done, next_obs)
```

### Custom Loss Functions

```python
from autotrain.trainers.losses.custom_loss import CustomLoss

class MyCustomLoss(CustomLoss):
    def compute_loss(self, predictions, targets, **kwargs):
        # Compute your loss
        return loss_tensor
```

### Combining Multiple Trainers

```python
# Step 1: Pre-train with DPO
dpo_trainer = DPOTrainer(dpo_config, model, tokenizer)
model = dpo_trainer.train_on_preferences(data)

# Step 2: Fine-tune with PPO
ppo_trainer = PPOTrainer(ppo_config, model, tokenizer)
model = ppo_trainer.train(environment)
```

## Performance Tips

1. **Memory Management**
   - Use gradient accumulation for large batches
   - Enable mixed precision training with `torch.amp`
   - Clear cache periodically with `torch.cuda.empty_cache()`

2. **Training Stability**
   - Start with small learning rates (1e-6 to 1e-5)
   - Use gradient clipping (max_norm=0.5)
   - Monitor KL divergence closely

3. **Efficient Sampling**
   - Use batch generation for environments
   - Cache tokenizer outputs
   - Implement early stopping in environments

## Testing

Run the test suite:

```bash
pytest tests/rl/ -v

# Individual test files
pytest tests/rl/test_environments.py -v
pytest tests/rl/test_reward_model.py -v
pytest tests/rl/test_losses.py -v
```

## Comparison with Other Libraries

| Feature | AutoTrain RL | Tinker | TRL | RLHF-Lab |
|---------|--------------|--------|-----|----------|
| Local Execution | ✅ | ❌ | ✅ | ✅ |
| Cloud API | ❌ | ✅ | ❌ | ❌ |
| PPO | ✅ | ✅ | ✅ | ✅ |
| DPO | ✅ | ✅ | ✅ | ❌ |
| Multi-Objective | ✅ | ✅ | ❌ | ❌ |
| Async Training | ✅ | ✅ | ❌ | ❌ |
| Custom Losses | ✅ | ✅ | Limited | Limited |
| HuggingFace Integration | ✅ | ❌ | ✅ | ✅ |

## Common Issues & Solutions

### Issue: Out of Memory

```python
# Solution 1: Reduce batch size
config.batch_size = 4

# Solution 2: Use gradient accumulation
config.gradient_accumulation_steps = 4

# Solution 3: Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

### Issue: Training Instability

```python
# Solution 1: Reduce learning rate
config.learning_rate = 1e-6

# Solution 2: Increase PPO clip parameter
config.clip_param = 0.3

# Solution 3: Add gradient clipping
config.max_grad_norm = 0.5
```

### Issue: Poor Performance

```python
# Solution 1: Improve reward signal
def better_reward(prompt, generated, full_text):
    score = base_reward(generated)
    score += shaping_reward(generated)  # Reward shaping
    return score

# Solution 2: Adjust exploration
config.entropy_coef = 0.02  # Increase exploration

# Solution 3: Use curriculum learning
env.set_difficulty("easy")
# ... train ...
env.set_difficulty("medium")
# ... train ...
env.set_difficulty("hard")
```

## Contributing

We welcome contributions! Priority areas:

1. **New Algorithms**
   - A2C/A3C
   - SAC (Soft Actor-Critic)
   - IMPALA

2. **Environment Types**
   - Multi-agent environments
   - Structured generation tasks
   - Tool-use environments

3. **Optimizations**
   - Distributed training
   - Improved memory efficiency
   - Faster sampling

## References

Key papers and inspirations:

1. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
2. **DPO**: Rafailov et al., "Direct Preference Optimization" (2023)
3. **RLHF**: Ouyang et al., "Training language models to follow instructions" (2022)
4. **Tinker**: Thinking Machines' approach to RL training

## License

Apache 2.0 - See LICENSE file for details

## Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Discord**: HuggingFace Discord #autotrain

---

*Built with ❤️ for the open-source community*