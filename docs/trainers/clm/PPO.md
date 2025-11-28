# AutoTrain Advanced - Reinforcement Learning Training Guide

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Components](#components)
- [Training Examples](#training-examples)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)

## Overview

AutoTrain Advanced RL module provides comprehensive reinforcement learning capabilities for language models, inspired by state-of-the-art approaches like those in Tinker. Unlike cloud-based solutions, this provides **two implementation paths**:

1. **CLI Integration**: Uses TRL (Transformers Reinforcement Learning) library for seamless integration with the AutoTrain CLI
2. **Direct API**: PyTorch-native implementations in `autotrain.trainers.rl` for programmatic use with complete control

### Key Features

- ✅ **PPO (Proximal Policy Optimization)** for stable policy learning
- ✅ **DPO (Direct Preference Optimization)** for preference-based training
- ✅ **Multi-Objective Rewards** for complex optimization goals
- ✅ **Custom Loss Functions** with flexible composition
- ✅ **Async Training Pipeline** for efficient computation
- ✅ **Reward Modeling** for RLHF (Reinforcement Learning from Human Feedback)

### Inspired by Tinker

While Tinker requires cloud infrastructure and API calls, AutoTrain Advanced provides similar capabilities with:
- Native PyTorch implementations
- Local execution without API dependencies
- Full control over model weights and training
- Integration with HuggingFace ecosystem

## Architecture

### Two Implementation Paths

1. **CLI Integration** (`autotrain/trainers/clm/train_clm_ppo.py`):
   - Uses TRL's PPOTrainer for production-ready RLHF
   - Integrated with AutoTrain CLI via `aitraining llm --trainer ppo`
   - Handles model loading, data processing, and training automatically

2. **Direct API** (`autotrain/trainers/rl/`):
   - PyTorch-native implementations for research and custom workflows
   - Full control over training loop and components
   - Modular design for easy customization

```
autotrain-advanced/src/autotrain/
├── trainers/
│   ├── clm/
│   │   └── train_clm_ppo.py    # CLI integration using TRL
│   ├── rl/                     # Direct API implementations
│   │   ├── __init__.py         # Module exports
│   │   ├── forward_backward.py # Async training pipeline
│   │   ├── ppo.py             # PPO trainer implementation
│   │   ├── dpo.py             # Direct Preference Optimization
│   │   ├── environments.py    # RL environments for text generation
│   │   └── reward_model.py    # Reward modeling for RLHF
│   └── losses/
│       ├── custom_loss.py     # Base framework for custom losses
│       ├── variance_loss.py   # Variance regularization
│       ├── kl_loss.py         # KL divergence constraints
│       ├── importance_sampling.py # Off-policy learning
│       └── ppo_loss.py        # PPO clipped objective
```

## Quick Start

### Installation

```bash
# Install AutoTrain Advanced with RL dependencies
pip install autotrain-advanced
pip install torch transformers peft trl
```

### CLI Usage (Production)

For production use, the CLI provides a streamlined interface using TRL:

```bash
# First, train a reward model
aitraining llm --train \
  --trainer reward \
  --model gpt2 \
  --data-path preference_data \
  --project-name reward_model

# Then use PPO with the reward model
aitraining llm --train \
  --trainer ppo \
  --model gpt2 \
  --rl-reward-model-path ./reward_model \
  --data-path prompts.csv \
  --project-name ppo_model
```

### Custom Environments via CLI

You can now wire custom RL environments directly into the CLI PPO trainer:

#### Text Generation Environment

```bash
aitraining llm --train \
  --trainer ppo \
  --model gpt2 \
  --rl-reward-model-path ./reward_model \
  --rl-env-type text_generation \
  --rl-env-config '{"stop_sequences": ["\n", "END"]}' \
  --data-path prompts.csv \
  --project-name ppo_custom_env
```

#### Multi-Objective Environment

```bash
aitraining llm --train \
  --trainer ppo \
  --model gpt2 \
  --rl-reward-model-path ./reward_model \
  --rl-env-type multi_objective \
  --rl-multi-objective true \
  --rl-env-config '{
    "reward_components": {
      "correctness": {"type": "keyword", "keywords": ["correct", "yes"]},
      "formatting": {"type": "keyword", "keywords": ["Answer:"]}
    },
    "reward_weights": {"correctness": 1.0, "formatting": 0.1}
  }' \
  --data-path math_problems.csv \
  --project-name ppo_multi_objective
```

**Alternative:** You can also specify reward weights separately:

```bash
aitraining llm --train \
  --trainer ppo \
  --model gpt2 \
  --rl-reward-model-path ./reward_model \
  --rl-env-type multi_objective \
  --rl-multi-objective true \
  --rl-reward-weights '{"correctness": 1.0, "formatting": 0.1}' \
  --rl-env-config '{
    "reward_components": {
      "correctness": {"type": "keyword", "keywords": ["correct"]},
      "formatting": {"type": "keyword", "keywords": ["Answer:"]}
    }
  }' \
  --data-path math_problems.csv \
  --project-name ppo_multi_objective
```

#### Preference Comparison Environment

```bash
aitraining llm --train \
  --trainer ppo \
  --model gpt2 \
  --rl-reward-model-path ./reward_model \
  --rl-env-type preference_comparison \
  --rl-env-config '{}' \
  --data-path preferences.csv \
  --project-name ppo_preferences
```

#### Environment Configuration Options

**Text Generation (`--rl-env-type text_generation`)**:
- `stop_sequences`: List of strings that end generation (e.g., `["\n", "END"]`)
- `reward_fn`: Custom reward function (not yet supported via CLI)

**Multi-Objective (`--rl-env-type multi_objective`)**:
- `reward_components` (required): Dict of reward component configs
  - Each component can have:
    - `type`: "length", "keyword", or default
    - `keywords`: List of keywords to match (for "keyword" type)
- `reward_weights`: Dict mapping component names to weights (can also use `--rl-reward-weights`)

**Preference Comparison (`--rl-env-type preference_comparison`)**:
- `preference_model`: Path to preference model (optional)
- `human_feedback_fn`: Custom feedback function (not yet supported via CLI)

**Defaults if not specified:**
- If `--rl-env-type` is not provided, the trainer uses standard TRL PPO behavior
- If `--rl-env-config` is not provided with an env type, defaults are used
- If `--rl-reward-model-path` is not provided, you must specify `--model-ref` for the reference model

### Direct API Usage (Research)

```python
from autotrain.trainers.rl import PPOTrainer, PPOConfig
from autotrain.trainers.rl.environments import TextGenerationEnv
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Configure PPO training
config = PPOConfig(
    model_name="gpt2",  # Model is loaded from config
    learning_rate=1e-5,
    batch_size=16,
    num_epochs=3,
    clip_param=0.2,
)

# Create environment with custom reward
def reward_fn(prompt, generated, full_text):
    # Custom reward logic
    if "correct" in generated.lower():
        return 1.0
    return 0.0

env = TextGenerationEnv(
    tokenizer=tokenizer,
    prompts=["What is 2+2?", "Explain gravity"],
    reward_fn=reward_fn,
)

# Train with PPO (Direct API)
trainer = PPOTrainer(config, tokenizer=tokenizer, reward_fn=reward_fn)
trainer.train(env)
```

## Components

### 1. Environments

#### TextGenerationEnv
Basic environment for text generation tasks with customizable rewards.

```python
from autotrain.trainers.rl.environments import TextGenerationEnv

env = TextGenerationEnv(
    tokenizer=tokenizer,
    prompts=training_prompts,
    max_length=512,
    reward_fn=your_reward_function,
    stop_sequences=["\\n", "END"],
    temperature=0.7,
)
```

#### MultiObjectiveRewardEnv
Environment supporting multiple reward objectives, similar to Tinker's approach.

```python
from autotrain.trainers.rl.environments import MultiObjectiveRewardEnv

def correctness_reward(prompt, generated, full_text):
    return 1.0 if check_answer(generated) else 0.0

def formatting_reward(prompt, generated, full_text):
    return 0.1 if "Answer:" in generated else -0.1

env = MultiObjectiveRewardEnv(
    tokenizer=tokenizer,
    prompts=math_problems,
    reward_components={
        "correctness": correctness_reward,
        "formatting": formatting_reward,
    },
    reward_weights={"correctness": 1.0, "formatting": 0.1},
)
```

### 2. Reward Models

#### Basic Reward Model
For RLHF training with human or AI feedback.

```python
from autotrain.trainers.rl.reward_model import RewardModel, RewardModelConfig

config = RewardModelConfig(
    model_name="bert-base-uncased",
    num_labels=1,
    pooling_strategy="last",
    learning_rate=1e-4,
)

reward_model = RewardModel(config)

# Train on preference data
trainer = RewardModelTrainer(reward_model, tokenizer, config)
trainer.train_on_preferences(chosen_texts, rejected_texts)
```

#### Multi-Objective Reward Model
For complex scenarios with multiple objectives.

```python
from autotrain.trainers.rl.reward_model import MultiObjectiveRewardModel

model = MultiObjectiveRewardModel(
    config,
    num_objectives=3,
    objective_weights=[0.5, 0.3, 0.2],
)
```

### 3. Custom Loss Functions

#### Composite Loss
Combine multiple loss functions with different weights.

```python
from autotrain.trainers.losses.custom_loss import CompositeLoss
from autotrain.trainers.losses.kl_loss import KLDivergenceLoss
from autotrain.trainers.losses.variance_loss import VarianceLoss

kl_loss = KLDivergenceLoss(target_kl=0.01)
var_loss = VarianceLoss(target_variance=1.0)

composite = CompositeLoss(
    losses=[kl_loss, var_loss],
    weights=[1.0, 0.1],
)
```

#### PPO Loss
Implements the clipped surrogate objective from PPO.

```python
from autotrain.trainers.losses.ppo_loss import PPOLoss

ppo_loss = PPOLoss(
    clip_param=0.2,
    value_clip=0.5,
    entropy_coef=0.01,
)
```

### 4. Forward-Backward Pipeline

Async training pipeline for efficient gradient computation.

```python
from autotrain.trainers.rl.forward_backward import ForwardBackwardPipeline

pipeline = ForwardBackwardPipeline(
    model=model,
    gradient_accumulation_steps=4,
    max_workers=2,
)

# Queue forward-backward passes
future1 = pipeline.forward_backward_async(batch1, loss_fn)
future2 = pipeline.forward_backward_async(batch2, loss_fn)

# Get results when ready
result1 = future1.result()
result2 = future2.result()
```

## Training Examples

### Example 1: Math Problem Solving with PPO

```python
from autotrain.trainers.rl import PPOTrainer, PPOConfig
from autotrain.trainers.rl.environments import create_math_problem_env
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Create specialized math environment
env = create_math_problem_env(tokenizer)

# Configure PPO for math problems
config = PPOConfig(
    model_name="gpt2",  # Model loaded from config
    learning_rate=1e-5,
    batch_size=8,
    num_epochs=5,
    clip_param=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
)

# Train
trainer = PPOTrainer(config, tokenizer=tokenizer)
results = trainer.train(
    env,
    num_episodes=1000,
    eval_every=100,
)
```

### Example 2: Preference Learning with DPO

```python
from autotrain.trainers.rl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = DPOConfig(
    model_name="gpt2",  # Model loaded from config
    beta=0.1,  # KL penalty coefficient
    learning_rate=5e-7,
    batch_size=4,
)

trainer = DPOTrainer(config, tokenizer=tokenizer)

# Train on preference pairs
trainer.train_on_preferences(
    prompts=prompts,
    chosen_responses=chosen,
    rejected_responses=rejected,
)
```

### Example 3: RLHF with Reward Model

```python
from autotrain.trainers.rl.reward_model import RewardModel, RewardModelConfig, RewardModelTrainer
from autotrain.trainers.rl import PPOTrainer, PPOConfig
from autotrain.trainers.rl.environments import TextGenerationEnv
import torch

# Step 1: Train reward model
reward_config = RewardModelConfig(
    model_name="gpt2",
    learning_rate=1e-4,
)
reward_model = RewardModel(reward_config)
reward_trainer = RewardModelTrainer(reward_model, tokenizer, reward_config)
reward_trainer.train_on_preferences(chosen_texts, rejected_texts)

# Step 2: Use reward model in PPO training
def reward_from_model(prompt, generated, full_text):
    with torch.no_grad():
        rewards = reward_model.predict_rewards(
            [full_text], tokenizer
        )
    return rewards[0]

env = TextGenerationEnv(
    tokenizer=tokenizer,
    prompts=prompts,
    reward_fn=reward_from_model,
)

# Step 3: Configure and train with PPO
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1e-5,
    batch_size=8,
)

ppo_trainer = PPOTrainer(ppo_config, tokenizer=tokenizer, reward_fn=reward_from_model)
ppo_trainer.train(env)
```

## API Reference

### PPOConfig

```python
class PPOConfig:
    # Required
    model_name: str                    # Model to train

    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 1

    # PPO specific
    ppo_epochs: int = 4               # Number of PPO epochs per batch
    gamma: float = 0.99               # Discount factor
    lam: float = 0.95                 # GAE lambda
    clip_ratio: float = 0.2           # PPO clip ratio
    value_clip: float = 0.2           # Value function clip ratio
    max_grad_norm: float = 1.0

    # KL penalty
    kl_penalty_coef: float = 0.01
    kl_target: float = 0.01

    # Regularization
    entropy_coef: float = 0.01
    value_coef: float = 0.5
```

### DPOConfig

```python
class DPOConfig:
    # Required
    model_name: str                    # Model to train

    # Training parameters
    learning_rate: float = 1e-6
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_epochs: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # DPO specific
    beta: float = 0.1                  # Temperature parameter for DPO loss
    label_smoothing: float = 0.0       # Label smoothing for robustness
    reference_free: bool = False       # Whether to use reference-free DPO

    # Generation parameters
    max_length: int = 512
    max_prompt_length: int = 256
```

### RewardModelConfig

```python
class RewardModelConfig:
    model_name: str
    num_labels: int = 1
    pooling_strategy: str = "last"  # "mean", "last", "cls"
    dropout_prob: float = 0.1
    temperature: float = 1.0
    use_lora: bool = False
    lora_rank: int = 8
    learning_rate: float = 1e-4
```

## Best Practices

### 1. Start with Small Learning Rates
RL training can be unstable. Start with small learning rates (1e-5 to 1e-6) and increase gradually.

### 2. Use Gradient Clipping
Always use gradient clipping to prevent training instability:
```python
config.max_grad_norm = 0.5
```

### 3. Monitor KL Divergence
Keep track of KL divergence between the policy and reference model:
```python
config.target_kl = 0.01  # Stop training if KL exceeds this
```

### 4. Implement Checkpointing
Save models regularly during training:
```python
trainer.save_checkpoint_every(100)  # Every 100 episodes
```

### 5. Use Multiple Seeds
Run experiments with different random seeds for robust results:
```python
for seed in [42, 123, 456]:
    torch.manual_seed(seed)
    trainer.train(env)
```

### 6. Balance Exploration vs Exploitation
Use temperature and entropy bonuses to control exploration:
```python
config.entropy_coef = 0.01  # Encourage exploration
env.temperature = 0.8  # Control randomness
```

## Comparison with Tinker

| Feature | Tinker | AutoTrain Advanced RL |
|---------|--------|----------------------|
| Execution | Cloud API | Local |
| Cost | Usage-based | Free (your hardware) |
| Control | Limited | Full |
| Customization | API constraints | Unlimited |
| Model Access | Via API | Direct weights access |
| Integration | Tinker ecosystem | HuggingFace ecosystem |
| Latency | Network dependent | Local speed |
| Privacy | Cloud processing | Fully private |

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Training Instability**
   - Reduce learning rate
   - Increase clip_param for PPO
   - Add gradient clipping

3. **Poor Reward Signal**
   - Design more informative rewards
   - Use reward shaping
   - Implement curriculum learning

## Advanced Topics

### Custom Environments

Create your own environment by inheriting from `RLEnvironment`:

```python
from autotrain.trainers.rl.environments import RLEnvironment

class CustomEnv(RLEnvironment):
    def reset(self):
        # Return initial observation
        pass

    def step(self, action):
        # Return StepResult(reward, done, next_obs)
        pass
```

### Custom Loss Functions

Implement custom losses by extending `CustomLoss`:

```python
from autotrain.trainers.losses.custom_loss import CustomLoss

class MyLoss(CustomLoss):
    def compute_loss(self, predictions, targets, mask=None, **kwargs):
        # Your loss computation
        return loss_tensor
```

## Contributing

We welcome contributions! Areas of interest:
- Additional RL algorithms (A2C, SAC, etc.)
- More environment types
- Advanced reward modeling techniques
- Performance optimizations

## License

Apache 2.0

## Acknowledgments

This implementation was inspired by concepts from:
- Tinker by Thinking Machines
- TRL by HuggingFace
- OpenAI's PPO implementation
- Anthropic's Constitutional AI research

## Support

For issues and questions:
- GitHub Issues: [autotrain-advanced/issues](https://github.com/huggingface/autotrain-advanced/issues)
- Discord: HuggingFace Discord #autotrain channel