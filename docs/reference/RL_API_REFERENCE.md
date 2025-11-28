# AutoTrain Advanced RL - API Reference

> **Important**: This document describes the experimental RL components in `autotrain.trainers.rl`.
> For production PPO training via CLI/API, use `--trainer ppo` which leverages TRL's PPOTrainer.
> The experimental implementations remain available for custom use cases.

## Table of Contents
- [Environments](#environments)
  - [TextGenerationEnv](#textgenerationenv)
  - [MultiObjectiveRewardEnv](#multiobjectiverewardenv)
  - [PreferenceComparisonEnv](#preferencecomparisonenv)
- [Reward Models](#reward-models)
  - [RewardModel](#rewardmodel)
  - [PairwiseRewardModel](#pairwiserewardmodel)
  - [MultiObjectiveRewardModel](#multiobjectiverewardmodel)
  - [RewardModelTrainer](#rewardmodeltrainer)
- [Loss Functions](#loss-functions)
  - [CustomLoss](#customloss)
  - [CompositeLoss](#compositeloss)
  - [PPOLoss](#ppoloss)
  - [KLDivergenceLoss](#kldivergenceclass)
  - [ImportanceSamplingLoss](#importancesamplingloss)
  - [VarianceLoss](#varianceloss)
- [Training Pipeline](#training-pipeline)
  - [ForwardBackwardPipeline](#forwardbackwardpipeline)
- [PPO Implementation](#ppo-implementation)
  - [PPOConfig](#ppoconfig)
  - [PPOTrainer](#ppotrainer)

---

## Module Organization

The RL components are organized as follows:

```
autotrain/trainers/
├── rl/
│   ├── __init__.py
│   ├── environments.py      # RL environments
│   ├── reward_model.py      # Reward models
│   ├── ppo.py               # Custom PPO implementation
│   ├── dpo.py               # DPO implementation
│   └── forward_backward.py  # Training pipeline
├── losses/
│   ├── __init__.py
│   ├── custom_loss.py       # Base and composite losses
│   ├── ppo_loss.py         # PPO loss
│   ├── kl_loss.py          # KL divergence loss
│   ├── variance_loss.py    # Variance loss
│   └── importance_sampling.py  # Importance sampling loss
└── clm/
    ├── train_clm_ppo.py     # TRL PPO integration (production)
    └── train_clm_dpo.py     # TRL DPO integration (production)
```

## Production vs Experimental

### Production (Using TRL)
```python
# Used by CLI with --trainer ppo
from trl import PPOTrainer, PPOConfig
from autotrain.trainers.clm.train_clm_ppo import train
```

### Experimental (Custom Implementation)
```python
# Custom implementations for research
from autotrain.trainers.rl import PPOTrainer, PPOConfig
from autotrain.trainers.rl.environments import TextGenerationEnv
```

---

## Environments

### Base Classes

```python
@dataclass
class Observation:
    """Observation from the environment."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StepResult:
    """Result from environment step."""
    reward: float
    done: bool
    next_observation: Observation
    info: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class Trajectory:
    """Complete trajectory from an episode."""
    observations: List[Observation]
    actions: List[torch.Tensor]
    rewards: List[float]
    log_probs: Optional[List[torch.Tensor]] = None
    values: Optional[List[torch.Tensor]] = None
```

### TextGenerationEnv

Text generation environment for reinforcement learning.

```python
class TextGenerationEnv(RLEnvironment):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        max_length: int = 512,
        reward_fn: Optional[Callable] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: float = 1.0
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenizer` | `PreTrainedTokenizer` | required | Tokenizer for encoding/decoding text |
| `prompts` | `List[str]` | required | List of prompts to generate from |
| `max_length` | `int` | 512 | Maximum sequence length |
| `reward_fn` | `Callable` | None | Function to compute rewards: `(prompt, generated, full_text) -> float` |
| `stop_sequences` | `List[str]` | None | Sequences that end generation |
| `temperature` | `float` | 1.0 | Temperature for sampling |

#### Methods

##### `reset() -> Observation`
Reset the environment with a new prompt.

##### `step(action: torch.Tensor) -> StepResult`
Take an action in the environment.

##### `render() -> str`
Get string representation of current state.

---

### MultiObjectiveRewardEnv

Environment with multiple reward objectives.

```python
class MultiObjectiveRewardEnv(TextGenerationEnv):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        reward_components: Dict[str, Callable],
        reward_weights: Optional[Dict[str, float]] = None,
        **kwargs
    )
```

#### Methods

##### `compute_multi_objective_reward(prompt, generated, full_text) -> Tuple[float, Dict[str, float]]`
Compute total reward and individual component scores.

---

### PreferenceComparisonEnv

Environment for preference learning and comparison.

```python
class PreferenceComparisonEnv(RLEnvironment):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        preference_model: Optional[nn.Module] = None,
        human_feedback_fn: Optional[Callable] = None,
        max_length: int = 512
    )
```

---

## Reward Models

### RewardModel

Base reward model for RLHF training.

```python
@dataclass
class RewardModelConfig:
    model_name: str
    num_labels: int = 1
    pooling_strategy: str = "last"  # "mean", "last", or "cls"
    dropout_prob: float = 0.1
    temperature: float = 1.0
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1

class RewardModel(nn.Module):
    def __init__(self, config: RewardModelConfig, base_model=None)
```

#### Methods

##### `forward(input_ids, attention_mask, return_dict=True) -> Union[Tensor, Dict]`
Forward pass through the reward model.

##### `compute_preference_loss(chosen_ids, chosen_mask, rejected_ids, rejected_mask, margin=0.0) -> Tensor`
Compute preference learning loss.

##### `predict_rewards(texts, tokenizer, max_length=512, batch_size=8) -> List[float]`
Predict rewards for a list of texts.

---

### PairwiseRewardModel

Reward model for direct pairwise comparisons.

```python
class PairwiseRewardModel(RewardModel):
    def forward_pair(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor
    ) -> torch.Tensor
```

##### `compute_bradley_terry_loss(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels) -> Tensor`
Compute Bradley-Terry model loss.

---

### MultiObjectiveRewardModel

Reward model with multiple objectives.

```python
class MultiObjectiveRewardModel(RewardModel):
    def __init__(
        self,
        config: RewardModelConfig,
        num_objectives: int = 3,
        objective_weights: Optional[List[float]] = None
    )
```

##### `forward(input_ids, attention_mask, return_all_objectives=False, return_dict=True)`
Returns individual objectives or combined weighted score.

##### `combine_objectives(multi_rewards) -> Tensor`
Combine multiple objectives into single reward.

---

### RewardModelTrainer

Trainer for reward models.

```python
class RewardModelTrainer:
    def __init__(
        self,
        model: RewardModel,
        tokenizer: PreTrainedTokenizer,
        config: RewardModelConfig,
        device: Optional[torch.device] = None
    )
```

##### `train_on_preferences(chosen_texts, rejected_texts, num_epochs=3, batch_size=8)`
Train the reward model on preference data.

##### `save_model(path: str)`
Save trained model to disk.

##### `load_model(path: str)`
Load model from disk.

---

## Loss Functions

**Note**: Loss functions are in `autotrain.trainers.losses` module.

### CustomLoss

Base class for custom loss functions.

```python
from autotrain.trainers.losses import CustomLoss, CustomLossConfig

@dataclass
class CustomLossConfig:
    name: str
    weight: float = 1.0
    reduction: str = "mean"  # "mean", "sum", or "none"
    normalize: bool = False
    clip_value: Optional[float] = None
    temperature: float = 1.0
    epsilon: float = 1e-8

class CustomLoss(nn.Module, ABC):
    def __init__(self, config: Optional[CustomLossConfig] = None)
```

#### Methods

##### `compute_loss(predictions, targets, mask=None, **kwargs) -> Tensor`
Abstract method to implement loss computation.

##### `forward(predictions, targets, mask=None, return_dict=False, **kwargs)`
Apply loss with reduction and weighting.

---

### CompositeLoss

Combine multiple loss functions.

```python
from autotrain.trainers.losses import CompositeLoss

class CompositeLoss(CustomLoss):
    def __init__(
        self,
        losses: List[CustomLoss],
        weights: Optional[List[float]] = None,
        config: Optional[CustomLossConfig] = None
    )
```

---

### PPOLoss

Proximal Policy Optimization loss.

```python
from autotrain.trainers.losses import PPOLoss

class PPOLoss(CustomLoss):
    def __init__(
        self,
        config: Optional[CustomLossConfig] = None,
        clip_param: float = 0.2,
        value_clip: Optional[float] = None,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5
    )
```

##### `compute_loss(log_probs, old_log_probs, advantages, values=None, old_values=None, returns=None, mask=None)`
Compute PPO loss with optional value function.

---

### KLDivergenceLoss

KL divergence loss for policy regularization.

```python
from autotrain.trainers.losses import KLDivergenceLoss

class KLDivergenceLoss(CustomLoss):
    def __init__(
        self,
        config: Optional[CustomLossConfig] = None,
        target_kl: float = 0.01,
        kl_coef: float = 0.1
    )
```

---

### ImportanceSamplingLoss

Importance sampling loss for off-policy training.

```python
from autotrain.trainers.losses import ImportanceSamplingLoss

class ImportanceSamplingLoss(CustomLoss):
    def __init__(
        self,
        config: Optional[CustomLossConfig] = None,
        clip_is_ratio: float = 5.0,
        normalize_advantages: bool = True
    )
```

---

### VarianceLoss

Variance regularization loss.

```python
from autotrain.trainers.losses import VarianceLoss

class VarianceLoss(CustomLoss):
    def __init__(
        self,
        config: Optional[CustomLossConfig] = None,
        target_variance: float = 1.0,
        beta: float = 0.1
    )
```

---

## Training Pipeline

### ForwardBackwardPipeline

Async pipeline for efficient training.

```python
from autotrain.trainers.rl.forward_backward import ForwardBackwardPipeline

class ForwardBackwardPipeline:
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        max_workers: int = 2,
        gradient_accumulation_steps: int = 1
    )
```

#### Data Classes

```python
@dataclass
class ForwardBackwardOutput:
    loss: float
    logits: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    gradients: Optional[Dict[str, torch.Tensor]] = None

@dataclass
class OptimStepOutput:
    step: int
    learning_rate: float
    grad_norm: float
    metrics: Dict[str, float] = field(default_factory=dict)

class AsyncTrainingFuture:
    def result(self, timeout: Optional[float] = None) -> Any
    def done(self) -> bool
    def cancel(self) -> bool
```

#### Methods

##### `forward_backward_async(batch, loss_fn) -> AsyncTrainingFuture`
Queue async forward-backward pass.

##### `optim_step_async(optimizer, scheduler=None) -> AsyncTrainingFuture`
Queue async optimizer step.

---

## PPO Implementation

### Custom PPO (Experimental)

Located in `autotrain.trainers.rl.ppo`.

```python
@dataclass
class PPOConfig:
    learning_rate: float = 1e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_epochs: int = 4
    clip_param: float = 0.2
    value_clip: Optional[float] = None
    kl_coef: float = 0.1
    target_kl: float = 0.01
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    normalize_advantages: bool = True
    use_adaptive_kl: bool = True
    device: str = "cuda"

class PPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    )
```

---

## Complete Example: RLHF Training

### Using Production TRL PPOTrainer (Recommended)

```python
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure PPO
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1
)

# Train with TRL
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
    # ... additional args
)

# Via CLI
# aitraining llm --trainer ppo --model gpt2 --rl-reward-model-path ./reward_model
```

### Using Experimental Custom Implementation

```python
from autotrain.trainers.rl import (
    PPOTrainer, PPOConfig,
    TextGenerationEnv,
    RewardModel, RewardModelConfig, RewardModelTrainer
)
from autotrain.trainers.losses import PPOLoss, KLDivergenceLoss, CompositeLoss

# Step 1: Train Reward Model
reward_config = RewardModelConfig(
    model_name="bert-base-uncased",
    learning_rate=1e-4,
    use_lora=True
)
reward_model = RewardModel(reward_config)
reward_trainer = RewardModelTrainer(reward_model, tokenizer, reward_config)

reward_trainer.train_on_preferences(
    chosen_texts=good_responses,
    rejected_texts=bad_responses,
    num_epochs=3
)

# Step 2: Create Environment
def reward_fn(prompt, generated, full_text):
    rewards = reward_model.predict_rewards([full_text], tokenizer)
    return rewards[0]

env = TextGenerationEnv(
    tokenizer=tokenizer,
    prompts=training_prompts,
    reward_fn=reward_fn,
    max_length=512
)

# Step 3: Setup Custom PPO
ppo_config = PPOConfig(
    learning_rate=1e-5,
    clip_param=0.2,
    target_kl=0.01
)

# Custom losses
ppo_loss = PPOLoss(clip_param=0.2, value_loss_coef=0.5)
kl_loss = KLDivergenceLoss(target_kl=0.01, kl_coef=0.1)
composite_loss = CompositeLoss([ppo_loss, kl_loss], weights=[1.0, 0.1])

# Train
ppo_trainer = PPOTrainer(ppo_config, model, tokenizer)
trained_model = ppo_trainer.train(env, loss_fn=composite_loss)

# Save
trained_model.save_pretrained("./rlhf_model")
reward_trainer.save_model("./reward_model.pt")
```

---

## Deprecation Notices

### Current Status

- **Production**: Use TRL's implementations via `autotrain.trainers.clm.train_clm_ppo`
- **Experimental**: Custom implementations in `autotrain.trainers.rl` remain available
- **Loss Functions**: Available in `autotrain.trainers.losses` module
- **No Deprecation**: All components are maintained, choose based on your needs

### Import Changes

```python
# Old (incorrect in docs):
from autotrain.trainers.rl.losses import PPOLoss  # ❌

# Correct:
from autotrain.trainers.losses import PPOLoss     # ✅
```

---

## Summary

The RL API provides both production-ready (TRL-based) and experimental (custom) implementations:

1. **For Production**: Use `--trainer ppo` CLI flag or TRL's PPOTrainer
2. **For Research**: Use custom implementations in `autotrain.trainers.rl`
3. **Loss Functions**: Import from `autotrain.trainers.losses`
4. **Environments**: Available in `autotrain.trainers.rl.environments`
5. **Reward Models**: Available in `autotrain.trainers.rl.reward_model`

All components are actively maintained and can be used based on your specific requirements.