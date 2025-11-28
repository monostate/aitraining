import json
import os
import signal
import sys
from typing import List

import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from huggingface_hub import repo_exists
from nvitop import Device

from autotrain import __version__, logger
from autotrain.app.db import AutoTrainDB
from autotrain.app.models import fetch_models
from autotrain.app.params import AppParams, get_task_params
from autotrain.app.utils import get_running_jobs, get_user_and_orgs, kill_process_by_pid, token_verification
from autotrain.app.wandb_visualizer import WandbVisualizerManager
from autotrain.dataset import (
    AutoTrainDataset,
    AutoTrainImageClassificationDataset,
    AutoTrainImageRegressionDataset,
    AutoTrainObjectDetectionDataset,
    AutoTrainVLMDataset,
)
from autotrain.help import get_app_help
from autotrain.project import AutoTrainProject


logger.info("Starting AutoTrain...")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
IS_RUNNING_IN_SPACE = "SPACE_ID" in os.environ
ENABLE_NGC = int(os.environ.get("ENABLE_NGC", 0))
ENABLE_NVCF = int(os.environ.get("ENABLE_NVCF", 0))
AUTOTRAIN_LOCAL = int(os.environ.get("AUTOTRAIN_LOCAL", 1))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = AutoTrainDB("autotrain.db")
WANDB_VISUALIZER_MANAGER = WandbVisualizerManager()
MODEL_CHOICE = fetch_models()

ui_router = APIRouter()
templates_path = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_path)

ASCII_BANNER = r"""
 █████╗ ██╗████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗     
██╔══██╗██║╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝     
███████║██║   ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗    
██╔══██║██║   ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║    
██║  ██║██║   ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝    
╚═╝  ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝     
        From zero to hero Machine Learning Training Platform
"""

UI_PARAMS = {
    # Training Configuration
    "mixed_precision": {
        "type": "dropdown",
        "label": "Mixed precision",
        "options": ["fp16", "bf16", "none"],
        "group": "Training Configuration",
        "help": "Use mixed precision training for faster computation and reduced memory usage",
    },
    "distributed_backend": {
        "type": "dropdown",
        "label": "Distributed backend",
        "options": ["ddp", "deepspeed"],
        "group": "Training Configuration",
        "help": "Choose distributed training backend (DDP or DeepSpeed)",
    },
    "use_flash_attention_2": {
        "type": "dropdown",
        "label": "Use flash attention",
        "options": [True, False],
        "group": "Training Configuration",
        "help": "Use Flash Attention 2 for faster attention computation",
    },
    "disable_gradient_checkpointing": {
        "type": "dropdown",
        "label": "Disable GC",
        "options": [True, False],
        "group": "Training Configuration",
        "help": "Disable gradient checkpointing (uses more memory but faster)",
    },
    "log": {
        "type": "dropdown",
        "label": "Logging",
        "options": ["none", "tensorboard", "wandb"],
        "group": "Training Configuration",
        "help": "Enable TensorBoard logging for training metrics",
    },
    "wandb_visualizer": {
        "type": "dropdown",
        "label": "W&B Visualizer (LEET)",
        "options": [True, False],
        "group": "Training Configuration",
        "help": "Stream the W&B LEET terminal dashboard locally (requires log='wandb')",
    },
    "logging_steps": {
        "type": "number",
        "label": "Logging steps",
        "group": "Training Configuration",
        "help": "Log training metrics every N steps",
    },
    "eval_strategy": {
        "type": "dropdown",
        "label": "Evaluation strategy",
        "options": ["epoch", "steps"],
        "group": "Training Configuration",
        "help": "When to run evaluation (per epoch or per N steps)",
    },
    "save_total_limit": {
        "type": "number",
        "label": "Save total limit",
        "group": "Training Configuration",
        "help": "Maximum number of checkpoints to keep",
    },
    "auto_find_batch_size": {
        "type": "dropdown",
        "label": "Auto find batch size",
        "options": [True, False],
        "group": "Training Configuration",
        "help": "Automatically find the largest batch size that fits in memory",
    },
    "packing": {
        "type": "dropdown",
        "label": "Enable Packing (SFT)",
        "options": [True, False, "auto"],
        "group": "Training Configuration",
        "help": "Pack multiple sequences into one batch for efficiency (SFT only)",
    },
    # Training Hyperparameters
    "epochs": {
        "type": "number",
        "label": "Epochs",
        "group": "Training Hyperparameters",
        "help": "Number of training epochs",
    },
    "batch_size": {
        "type": "number",
        "label": "Batch size",
        "group": "Training Hyperparameters",
        "help": "Training batch size per device",
    },
    "lr": {
        "type": "number",
        "label": "Learning rate",
        "group": "Training Hyperparameters",
        "help": "Learning rate for optimizer",
    },
    "optimizer": {
        "type": "dropdown",
        "label": "Optimizer",
        "options": ["adamw_torch", "adamw", "adam", "sgd"],
        "group": "Training Hyperparameters",
        "help": "Optimizer algorithm to use",
    },
    "scheduler": {
        "type": "dropdown",
        "label": "Scheduler",
        "options": ["linear", "cosine", "cosine_warmup", "constant"],
        "group": "Training Hyperparameters",
        "help": "Learning rate scheduler",
    },
    "warmup_ratio": {
        "type": "number",
        "label": "Warmup proportion",
        "group": "Training Hyperparameters",
        "help": "Proportion of training steps for warmup",
    },
    "gradient_accumulation": {
        "type": "number",
        "label": "Gradient accumulation",
        "group": "Training Hyperparameters",
        "help": "Accumulate gradients over N steps before updating",
    },
    "weight_decay": {
        "type": "number",
        "label": "Weight decay",
        "group": "Training Hyperparameters",
        "help": "L2 regularization coefficient",
    },
    "max_grad_norm": {
        "type": "number",
        "label": "Max grad norm",
        "group": "Training Hyperparameters",
        "help": "Maximum gradient norm for clipping",
    },
    "seed": {
        "type": "number",
        "label": "Seed",
        "group": "Training Hyperparameters",
        "help": "Random seed for reproducibility",
    },
    # Data Processing
    "block_size": {
        "type": "number",
        "label": "Block size",
        "group": "Data Processing",
        "help": "Maximum sequence length for training",
    },
    "model_max_length": {
        "type": "number",
        "label": "Model max length",
        "group": "Data Processing",
        "help": "Maximum sequence length for the model",
    },
    "add_eos_token": {
        "type": "dropdown",
        "label": "Add EOS token",
        "options": [True, False],
        "group": "Data Processing",
        "help": "Add end-of-sequence token to inputs",
    },
    "padding": {
        "type": "dropdown",
        "label": "Padding side",
        "options": ["right", "left", "none"],
        "group": "Data Processing",
        "help": "Side to pad sequences on",
    },
    "chat_template": {
        "type": "dropdown",
        "label": "Chat template",
        "options": ["none", "zephyr", "chatml", "tokenizer", "alpaca", "llama", "vicuna", "mistral"],
        "group": "Data Processing",
        "help": "Chat template format for conversation data",
    },
    "chat_format": {
        "type": "dropdown",
        "label": "Chat Format",
        "options": ["none", "chatml", "alpaca", "llama", "vicuna", "zephyr", "mistral"],
        "group": "Data Processing",
        "help": "Message rendering format for chat data",
    },
    "token_weights": {
        "type": "string",
        "label": "Token Weights (JSON)",
        "group": "Data Processing",
        "help": "JSON dict of token weights for weighted loss",
    },
    # PEFT/LoRA
    "quantization": {
        "type": "dropdown",
        "label": "Quantization",
        "options": ["int4", "int8", "none"],
        "group": "PEFT/LoRA",
        "help": "Quantize model weights for reduced memory",
    },
    "peft": {
        "type": "dropdown",
        "label": "PEFT/LoRA",
        "options": [True, False],
        "group": "PEFT/LoRA",
        "help": "Enable Parameter-Efficient Fine-Tuning with LoRA",
    },
    "lora_r": {
        "type": "number",
        "label": "Lora r",
        "group": "PEFT/LoRA",
        "help": "LoRA rank (lower = fewer parameters)",
    },
    "lora_alpha": {
        "type": "number",
        "label": "Lora alpha",
        "group": "PEFT/LoRA",
        "help": "LoRA scaling factor",
    },
    "lora_dropout": {
        "type": "number",
        "label": "Lora dropout",
        "group": "PEFT/LoRA",
        "help": "Dropout rate for LoRA layers",
    },
    "target_modules": {
        "type": "string",
        "label": "Target modules",
        "group": "PEFT/LoRA",
        "help": "Comma-separated list of modules to apply LoRA to (e.g., 'all-linear')",
    },
    "merge_adapter": {
        "type": "dropdown",
        "label": "Merge adapter",
        "options": [True, False],
        "group": "PEFT/LoRA",
        "help": "Merge LoRA weights into base model after training",
    },
    # DPO/ORPO
    "model_ref": {
        "type": "string",
        "label": "Reference model",
        "group": "DPO/ORPO",
        "help": "Reference model for DPO/ORPO (uses same model if not specified)",
    },
    "dpo_beta": {
        "type": "number",
        "label": "DPO beta",
        "group": "DPO/ORPO",
        "help": "KL penalty coefficient for DPO",
    },
    "max_prompt_length": {
        "type": "number",
        "label": "Prompt length",
        "group": "DPO/ORPO",
        "help": "Maximum prompt length for preference data",
    },
    "max_completion_length": {
        "type": "number",
        "label": "Completion length",
        "group": "DPO/ORPO",
        "help": "Maximum completion length",
    },
    # Hub Integration
    "unsloth": {
        "type": "dropdown",
        "label": "Unsloth",
        "options": [True, False],
        "group": "Hub Integration",
        "help": "Use Unsloth for faster training (limited model support)",
    },
    "wandb_token": {
        "type": "string",
        "label": "W&B API Token",
        "group": "Hub Integration",
        "help": "Optional token for syncing offline W&B runs to the cloud",
    },
    # Knowledge Distillation
    "use_distillation": {
        "type": "dropdown",
        "label": "Enable Distillation",
        "options": [True, False],
        "group": "Knowledge Distillation",
        "help": "Enable knowledge distillation from a teacher model",
    },
    "teacher_model": {
        "type": "string",
        "label": "Teacher Model",
        "group": "Knowledge Distillation",
        "help": "Hugging Face model ID for teacher model",
    },
    "teacher_prompt_template": {
        "type": "string",
        "label": "Teacher Prompt Template",
        "group": "Knowledge Distillation",
        "help": "Template for teacher model prompts",
    },
    "student_prompt_template": {
        "type": "string",
        "label": "Student Prompt Template",
        "group": "Knowledge Distillation",
        "help": "Template for student model prompts",
    },
    "distill_temperature": {
        "type": "number",
        "label": "Distillation Temperature",
        "group": "Knowledge Distillation",
        "help": "Temperature for softening probability distributions",
    },
    "distill_alpha": {
        "type": "number",
        "label": "Distillation Alpha (KL weight)",
        "group": "Knowledge Distillation",
        "help": "Weight for KL divergence loss (1-alpha for hard label loss)",
    },
    "distill_max_teacher_length": {
        "type": "number",
        "label": "Max Teacher Output Length",
        "group": "Knowledge Distillation",
        "help": "Maximum generation length for teacher model",
    },
    # Hyperparameter Sweep
    "use_sweep": {
        "type": "dropdown",
        "label": "Enable Hyperparameter Sweep",
        "options": [True, False],
        "group": "Hyperparameter Sweep",
        "help": "Enable automated hyperparameter optimization",
    },
    "sweep_backend": {
        "type": "dropdown",
        "label": "Sweep Backend",
        "options": ["optuna", "ray", "grid", "random"],
        "group": "Hyperparameter Sweep",
        "help": "Backend for hyperparameter search",
    },
    "sweep_n_trials": {
        "type": "number",
        "label": "Number of Sweep Trials",
        "group": "Hyperparameter Sweep",
        "help": "Number of trials to run for hyperparameter search",
    },
    "sweep_metric": {
        "type": "string",
        "label": "Metric to Optimize",
        "group": "Hyperparameter Sweep",
        "help": "Metric to optimize (e.g., 'eval/loss', 'eval/accuracy')",
    },
    "sweep_direction": {
        "type": "dropdown",
        "label": "Optimization Direction",
        "options": ["minimize", "maximize"],
        "group": "Hyperparameter Sweep",
        "help": "Whether to minimize or maximize the metric",
    },
    "sweep_params": {
        "type": "string",
        "label": "Sweep Parameters (JSON)",
        "group": "Hyperparameter Sweep",
        "help": 'JSON dict of parameters to sweep (e.g., {"lr": [1e-5, 1e-4]})',
    },
    # Enhanced Evaluation
    "use_enhanced_eval": {
        "type": "dropdown",
        "label": "Enable Enhanced Evaluation",
        "options": [True, False],
        "group": "Enhanced Evaluation",
        "help": "Enable enhanced evaluation with custom metrics",
    },
    "eval_metrics": {
        "type": "string",
        "label": "Evaluation Metrics (comma-separated)",
        "group": "Enhanced Evaluation",
        "help": "Comma-separated list of metrics (e.g., 'accuracy,f1,bleu')",
    },
    "eval_dataset_path": {
        "type": "string",
        "label": "Evaluation Dataset Path",
        "group": "Enhanced Evaluation",
        "help": "Path to custom evaluation dataset",
    },
    "eval_batch_size": {
        "type": "number",
        "label": "Evaluation Batch Size",
        "group": "Enhanced Evaluation",
        "help": "Batch size for evaluation",
    },
    "eval_save_predictions": {
        "type": "dropdown",
        "label": "Save Evaluation Predictions",
        "options": [True, False],
        "group": "Enhanced Evaluation",
        "help": "Save model predictions during evaluation",
    },
    "eval_benchmark": {
        "type": "dropdown",
        "label": "Standard Benchmark",
        "options": ["none", "mmlu", "hellaswag", "arc", "truthfulqa"],
        "group": "Enhanced Evaluation",
        "help": "Run standard benchmark evaluation",
    },
    # Reinforcement Learning (PPO)
    "rl_reward_model_path": {
        "type": "string",
        "label": "Reward Model Path (for PPO)",
        "group": "Reinforcement Learning (PPO)",
        "help": "REQUIRED for PPO: Path or HF model ID for reward model",
        "required_for_ppo": True,
        "is_ppo_requirement": True,  # This is the field that must be filled first
    },
    "rl_gamma": {
        "type": "number",
        "label": "RL Discount Factor (gamma)",
        "group": "Reinforcement Learning (PPO)",
        "help": "Discount factor for future rewards",
        "required_for_ppo": True,
    },
    "rl_gae_lambda": {
        "type": "number",
        "label": "GAE Lambda",
        "group": "Reinforcement Learning (PPO)",
        "help": "Lambda parameter for Generalized Advantage Estimation",
        "required_for_ppo": True,
    },
    "rl_kl_coef": {
        "type": "number",
        "label": "KL Divergence Coefficient",
        "group": "Reinforcement Learning (PPO)",
        "help": "Coefficient for KL divergence penalty",
        "required_for_ppo": True,
    },
    "rl_value_loss_coef": {
        "type": "number",
        "label": "Value Loss Coefficient",
        "group": "Reinforcement Learning (PPO)",
        "help": "Weight for value function loss",
        "required_for_ppo": True,
    },
    "rl_clip_range": {
        "type": "number",
        "label": "PPO Clipping Range",
        "group": "Reinforcement Learning (PPO)",
        "help": "Clipping range for PPO policy updates",
        "required_for_ppo": True,
    },
    "rl_reward_fn": {
        "type": "dropdown",
        "label": "Reward Function",
        "options": ["default", "length_penalty", "correctness", "custom"],
        "group": "Reinforcement Learning (PPO)",
        "help": "Reward function type to use",
        "required_for_ppo": True,
    },
    "rl_multi_objective": {
        "type": "dropdown",
        "label": "Multi-Objective Rewards",
        "options": [True, False],
        "group": "Reinforcement Learning (PPO)",
        "help": "Use multiple reward objectives",
        "required_for_ppo": True,
    },
    "rl_reward_weights": {
        "type": "string",
        "label": "Reward Weights (JSON)",
        "group": "Reinforcement Learning (PPO)",
        "help": "JSON dict of reward weights for multi-objective learning",
        "required_for_ppo": True,
    },
    "rl_env_type": {
        "type": "dropdown",
        "label": "RL Environment Type",
        "options": ["text_generation", "multi_objective", "preference_comparison"],
        "group": "Reinforcement Learning (PPO)",
        "help": "Type of RL environment",
        "required_for_ppo": True,
    },
    "rl_env_config": {
        "type": "string",
        "label": "RL Environment Config (JSON)",
        "group": "Reinforcement Learning (PPO)",
        "help": "JSON configuration for RL environment",
        "required_for_ppo": True,
    },
    "rl_num_ppo_epochs": {
        "type": "number",
        "label": "Number of PPO Epochs",
        "group": "Reinforcement Learning (PPO)",
        "help": "Number of epochs per PPO update",
        "required_for_ppo": True,
    },
    "rl_chunk_size": {
        "type": "number",
        "label": "PPO Chunk Size",
        "group": "Reinforcement Learning (PPO)",
        "help": "Size of chunks for PPO updates",
        "required_for_ppo": True,
    },
    "rl_mini_batch_size": {
        "type": "number",
        "label": "PPO Mini Batch Size",
        "group": "Reinforcement Learning (PPO)",
        "help": "Mini-batch size for PPO updates",
        "required_for_ppo": True,
    },
    "rl_optimize_device_cache": {
        "type": "dropdown",
        "label": "Optimize PPO Device Cache",
        "options": [True, False],
        "group": "Reinforcement Learning (PPO)",
        "help": "Optimize device memory caching for PPO",
        "required_for_ppo": True,
    },
    # Advanced Features
    "custom_loss": {
        "type": "dropdown",
        "label": "Custom Loss Function",
        "options": ["none", "kl", "composite", "ppo", "variance_reduced"],
        "group": "Advanced Features",
        "help": "Use custom loss function",
    },
    "custom_loss_weights": {
        "type": "string",
        "label": "Custom Loss Weights (JSON)",
        "group": "Advanced Features",
        "help": "JSON dict of loss component weights",
    },
    "use_forward_backward": {
        "type": "dropdown",
        "label": "[Advanced] Manual Forward-Backward Control",
        "options": [False, True],
        "group": "Advanced Features",
        "help": "Enable manual control of forward-backward passes",
    },
    "forward_backward_loss_fn": {
        "type": "dropdown",
        "label": "[Advanced] Loss Function",
        "options": ["none", "cross_entropy", "importance_sampling", "ppo", "custom"],
        "group": "Advanced Features",
        "help": "Loss function for forward-backward control",
    },
    "forward_backward_custom_fn": {
        "type": "textarea",
        "label": "[Advanced] Custom Loss Function Code",
        "help": "Python function: def custom_loss(model, inputs, outputs, **kwargs) -> torch.Tensor",
        "group": "Advanced Features",
    },
    "gradient_accumulation_steps": {
        "type": "number",
        "label": "[Advanced] Gradient Accumulation Steps",
        "group": "Advanced Features",
        "help": "Alternative gradient accumulation parameter",
    },
    "manual_optimizer_control": {
        "type": "dropdown",
        "label": "[Advanced] Manual Optimizer Control",
        "options": [False, True],
        "group": "Advanced Features",
        "help": "Enable manual optimizer control",
    },
    "optimizer_step_frequency": {
        "type": "number",
        "label": "[Advanced] Optimizer Step Frequency",
        "group": "Advanced Features",
        "help": "Frequency of optimizer steps",
    },
    "grad_clip_value": {
        "type": "number",
        "label": "[Advanced] Gradient Clipping Value",
        "group": "Advanced Features",
        "help": "Alternative gradient clipping parameter",
    },
    "manual_sampling": {
        "type": "dropdown",
        "label": "[Advanced] Manual Sampling Control",
        "options": [False, True],
        "group": "Advanced Features",
        "help": "Enable manual sampling control during training",
    },
    "sample_every_n_steps": {
        "type": "number",
        "label": "[Advanced] Sample Every N Steps",
        "group": "Advanced Features",
        "help": "Generate samples every N training steps",
    },
    "sample_prompts": {
        "type": "textarea",
        "label": "[Advanced] Sample Prompts (JSON)",
        "help": "JSON array of prompts to sample during training",
        "group": "Advanced Features",
    },
    "sample_temperature": {
        "type": "number",
        "label": "[Advanced] Sample Temperature",
        "group": "Advanced Features",
        "help": "Temperature for sampling",
    },
    "sample_top_k": {
        "type": "number",
        "label": "[Advanced] Sample Top-K",
        "group": "Advanced Features",
        "help": "Top-K for sampling",
    },
    "sample_top_p": {
        "type": "number",
        "label": "[Advanced] Sample Top-P",
        "group": "Advanced Features",
        "help": "Top-P (nucleus) for sampling",
    },
    "manual_checkpoint_control": {
        "type": "dropdown",
        "label": "[Advanced] Manual Checkpoint Control",
        "options": [False, True],
        "group": "Advanced Features",
        "help": "Enable manual checkpoint control",
    },
    "save_state_every_n_steps": {
        "type": "number",
        "label": "[Advanced] Save State Every N Steps",
        "group": "Advanced Features",
        "help": "Save training state every N steps",
    },
    "load_state_from": {
        "type": "string",
        "label": "[Advanced] Load State From Path",
        "group": "Advanced Features",
        "help": "Path to load training state from",
    },
    # Tabular-specific parameters
    "max_seq_length": {
        "type": "number",
        "label": "Max sequence length",
        "help": "Maximum sequence length for text data",
    },
    "early_stopping_patience": {
        "type": "number",
        "label": "Early stopping patience",
        "help": "Number of epochs without improvement before stopping",
    },
    "early_stopping_threshold": {
        "type": "number",
        "label": "Early stopping threshold",
        "help": "Minimum change to qualify as improvement",
    },
    "max_target_length": {
        "type": "number",
        "label": "Max target length",
        "help": "Maximum target sequence length",
    },
    "categorical_columns": {
        "type": "string",
        "label": "Categorical columns",
        "help": "Comma-separated list of categorical column names",
    },
    "numerical_columns": {
        "type": "string",
        "label": "Numerical columns",
        "help": "Comma-separated list of numerical column names",
    },
    "num_trials": {
        "type": "number",
        "label": "Number of trials",
        "help": "Number of AutoML trials",
    },
    "time_limit": {
        "type": "number",
        "label": "Time limit",
        "help": "Time limit in seconds for AutoML",
    },
    "categorical_imputer": {
        "type": "dropdown",
        "label": "Categorical imputer",
        "options": ["most_frequent", "none"],
        "help": "Strategy for imputing missing categorical values",
    },
    "numerical_imputer": {
        "type": "dropdown",
        "label": "Numerical imputer",
        "options": ["mean", "median", "none"],
        "help": "Strategy for imputing missing numerical values",
    },
    "numeric_scaler": {
        "type": "dropdown",
        "label": "Numeric scaler",
        "options": ["standard", "minmax", "maxabs", "robust", "none"],
        "help": "Strategy for scaling numerical features",
    },
    # Image-specific parameters
    "vae_model": {
        "type": "string",
        "label": "VAE model",
        "help": "VAE model for image generation",
    },
    "prompt": {
        "type": "string",
        "label": "Prompt",
        "help": "Text prompt for image generation",
    },
    "resolution": {
        "type": "number",
        "label": "Resolution",
        "help": "Image resolution",
    },
    "num_steps": {
        "type": "number",
        "label": "Number of steps",
        "help": "Number of diffusion steps",
    },
    "checkpointing_steps": {
        "type": "number",
        "label": "Checkpointing steps",
        "help": "Save checkpoint every N steps",
    },
    "use_8bit_adam": {
        "type": "dropdown",
        "label": "Use 8-bit Adam",
        "options": [True, False],
        "help": "Use 8-bit Adam optimizer for reduced memory",
    },
    "xformers": {
        "type": "dropdown",
        "label": "xFormers",
        "options": [True, False],
        "help": "Use xFormers for efficient attention",
    },
    "image_square_size": {
        "type": "number",
        "label": "Image square size",
        "help": "Square size for image resizing",
    },
    "max_doc_stride": {
        "type": "number",
        "label": "Max doc stride",
        "help": "Maximum stride for document processing",
    },
}


def graceful_exit(signum, frame):
    """
    Handles the SIGTERM signal to perform cleanup and exit the program gracefully.

    Args:
        signum (int): The signal number.
        frame (FrameType): The current stack frame (or None).

    Logs:
        Logs the receipt of the SIGTERM signal and the initiation of cleanup.

    Exits:
        Exits the program with status code 0.
    """
    logger.info("SIGTERM received. Performing cleanup...")
    sys.exit(0)


signal.signal(signal.SIGTERM, graceful_exit)


logger.info("AutoTrain started successfully")


def user_authentication(request: Request):
    """
    Authenticates the user based on the following priority:
    1. HF_TOKEN environment variable
    2. OAuth information in session
    3. Token in bearer header (not implemented in the given code)

    Args:
        request (Request): The incoming HTTP request object.

    Returns:
        str: The authenticated token if verification is successful.

    Raises:
        HTTPException: If the token is invalid or expired and the application is not running in a space.

    If the application is running in a space and authentication fails, it returns a login template response.
    """
    # If running locally and no token provided, allow access (return None or dummy token)
    # This is for the 'autotrain chat' use case where we want to run locally without HF_TOKEN
    if not IS_RUNNING_IN_SPACE and os.environ.get("HF_TOKEN") is None:
        return "local_mode"

    # priority: hf_token env var > oauth_info in session > token in bearer header
    token_from_env = os.environ.get("HF_TOKEN")
    if token_from_env is not None:
        if os.environ.get("AUTOTRAIN_SKIP_TOKEN_VERIFICATION", "0") == "1":
            return token_from_env
        try:
            _ = token_verification(token=token_from_env)
            return token_from_env
        except Exception as e:
            logger.error(f"Failed to verify token: {e}")
            if IS_RUNNING_IN_SPACE:
                return templates.TemplateResponse("login.html", {"request": request})
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token: HF_TOKEN",
                )

    if IS_RUNNING_IN_SPACE and "oauth_info" in request.session:
        try:
            _ = token_verification(token=request.session["oauth_info"]["access_token"])
            return request.session["oauth_info"]["access_token"]
        except Exception as e:
            request.session.pop("oauth_info", None)
            logger.error(f"Failed to verify token: {e}")
            return templates.TemplateResponse("login.html", {"request": request})

    if IS_RUNNING_IN_SPACE:
        return templates.TemplateResponse("login.html", {"request": request})

    # Local mode fallback if environment variable not set but function called
    return "local_mode"


@ui_router.get("/", response_class=HTMLResponse)
async def load_index(request: Request):
    """
    Redirects root to inference page as training UI is deprecated/removed.
    """
    return RedirectResponse("/inference")


@ui_router.get("/inference", response_class=HTMLResponse)
async def load_inference(request: Request, token: str = Depends(user_authentication)):
    """
    This function is used to load the inference page
    :return: HTMLResponse
    """
    # Handle local mode - no need to fetch user/orgs
    if token == "local_mode":
        _users = ["local"]
    else:
        try:
            _users = get_user_and_orgs(user_token=token)
        except Exception as e:
            logger.error(f"Failed to get user and orgs: {e}")
            if "oauth_info" in request.session:
                request.session.pop("oauth_info", None)
            return templates.TemplateResponse("login.html", {"request": request})
    context = {
        "request": request,
        "valid_users": _users,
        "version": __version__,
        "token": token,
        "banner": ASCII_BANNER,
    }
    return templates.TemplateResponse("inference.html", context)


@ui_router.get("/logout", response_class=HTMLResponse)
async def oauth_logout(request: Request, authenticated: bool = Depends(user_authentication)):
    """
    This function is used to logout the oauth user
    :return: HTMLResponse
    """
    request.session.pop("oauth_info", None)
    return RedirectResponse("/")


@ui_router.get("/params/{task}/{param_type}", response_class=JSONResponse)
async def fetch_params(task: str, param_type: str, authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the parameters for a given task
    :param task: str
    :param param_type: str (basic, full)
    :return: JSONResponse
    """
    logger.info(f"Task: {task}")
    task_params = get_task_params(task, param_type)
    if len(task_params) == 0:
        return {"error": "Task not found"}
    ui_params = {}
    for param in task_params:
        if param in UI_PARAMS:
            ui_params[param] = UI_PARAMS[param]
            ui_params[param]["default"] = task_params[param]
        else:
            logger.info(f"Param {param} not found in UI_PARAMS")

    ui_params = dict(sorted(ui_params.items(), key=lambda x: (x[1]["type"], x[1]["label"])))
    return ui_params


@ui_router.get("/model_choices/{task}", response_class=JSONResponse)
async def fetch_model_choices(
    task: str,
    custom_models: str = Query(None),
    authenticated: bool = Depends(user_authentication),
):
    """
    This function is used to fetch the model choices for a given task
    :param task: str
    :param custom_models: str (optional, comma separated list of custom models, query parameter)
    :return: JSONResponse
    """
    resp = []

    if custom_models is not None:
        custom_models = custom_models.split(",")
        for custom_model in custom_models:
            custom_model = custom_model.strip()
            resp.append({"id": custom_model, "name": custom_model})

    if os.environ.get("AUTOTRAIN_CUSTOM_MODELS", None) is not None:
        custom_models = os.environ.get("AUTOTRAIN_CUSTOM_MODELS")
        custom_models = custom_models.split(",")
        for custom_model in custom_models:
            custom_model = custom_model.strip()
            resp.append({"id": custom_model, "name": custom_model})

    if task == "text-classification":
        hub_models = MODEL_CHOICE["text-classification"]
    elif task.startswith("llm"):
        hub_models = MODEL_CHOICE["llm"]
    elif task.startswith("st:"):
        hub_models = MODEL_CHOICE["sentence-transformers"]
    elif task == "image-classification":
        hub_models = MODEL_CHOICE["image-classification"]
    elif task == "seq2seq":
        hub_models = MODEL_CHOICE["seq2seq"]
    elif task == "tabular:classification":
        hub_models = MODEL_CHOICE["tabular-classification"]
    elif task == "tabular:regression":
        hub_models = MODEL_CHOICE["tabular-regression"]
    elif task == "token-classification":
        hub_models = MODEL_CHOICE["token-classification"]
    elif task == "text-regression":
        hub_models = MODEL_CHOICE["text-regression"]
    elif task == "image-object-detection":
        hub_models = MODEL_CHOICE["image-object-detection"]
    elif task == "image-regression":
        hub_models = MODEL_CHOICE["image-regression"]
    elif task.startswith("vlm:"):
        hub_models = MODEL_CHOICE["vlm"]
    elif task == "extractive-qa":
        hub_models = MODEL_CHOICE["extractive-qa"]
    else:
        raise NotImplementedError

    for hub_model in hub_models:
        resp.append({"id": hub_model, "name": hub_model})
    return resp


@ui_router.post("/create_project", response_class=JSONResponse)
async def handle_form(
    project_name: str = Form(...),
    task: str = Form(...),
    base_model: str = Form(...),
    hardware: str = Form(...),
    params: str = Form(...),
    autotrain_user: str = Form(...),
    column_mapping: str = Form('{"default": "value"}'),
    data_files_training: List[UploadFile] = File(None),
    data_files_valid: List[UploadFile] = File(None),
    hub_dataset: str = Form(""),
    train_split: str = Form(""),
    valid_split: str = Form(""),
    token: str = Depends(user_authentication),
):
    """
    Handle form submission for creating and managing AutoTrain projects.

    Args:
        project_name (str): The name of the project.
        task (str): The task type (e.g., "image-classification", "text-classification").
        base_model (str): The base model to use for training.
        hardware (str): The hardware configuration (e.g., "local-ui").
        params (str): JSON string of additional parameters.
        autotrain_user (str): The username of the AutoTrain user.
        column_mapping (str): JSON string mapping columns to their roles.
        data_files_training (List[UploadFile]): List of training data files.
        data_files_valid (List[UploadFile]): List of validation data files.
        hub_dataset (str): The Hugging Face Hub dataset identifier.
        train_split (str): The training split identifier.
        valid_split (str): The validation split identifier.
        token (str): The authentication token.

    Returns:
        dict: A dictionary containing the success status and monitor URL.

    Raises:
        HTTPException: If there are conflicts or validation errors in the form submission.
    """
    train_split = train_split.strip()
    if len(train_split) == 0:
        train_split = None

    valid_split = valid_split.strip()
    if len(valid_split) == 0:
        valid_split = None

    logger.info(f"hardware: {hardware}")
    if hardware == "local-ui":
        running_jobs = get_running_jobs(DB)
        if running_jobs:
            raise HTTPException(
                status_code=409, detail="Another job is already running. Please wait for it to finish."
            )

    if repo_exists(f"{autotrain_user}/{project_name}", token=token):
        raise HTTPException(
            status_code=409,
            detail=f"Project {project_name} already exists. Please choose a different name.",
        )

    params = json.loads(params)
    # convert "null" to None
    for key in params:
        if params[key] == "null":
            params[key] = None
    column_mapping = json.loads(column_mapping)

    training_files = [f.file for f in data_files_training if f.filename != ""] if data_files_training else []
    validation_files = [f.file for f in data_files_valid if f.filename != ""] if data_files_valid else []

    if len(training_files) > 0 and len(hub_dataset) > 0:
        raise HTTPException(
            status_code=400, detail="Please either upload a dataset or choose a dataset from the Hugging Face Hub."
        )

    if len(training_files) == 0 and len(hub_dataset) == 0:
        raise HTTPException(
            status_code=400, detail="Please upload a dataset or choose a dataset from the Hugging Face Hub."
        )

    if len(hub_dataset) > 0:
        if not train_split:
            raise HTTPException(status_code=400, detail="Please enter a training split.")

    if len(hub_dataset) == 0:
        file_extension = os.path.splitext(data_files_training[0].filename)[1]
        file_extension = file_extension[1:] if file_extension.startswith(".") else file_extension
        if task == "image-classification":
            dset = AutoTrainImageClassificationDataset(
                train_data=training_files[0],
                token=token,
                project_name=project_name,
                username=autotrain_user,
                valid_data=validation_files[0] if validation_files else None,
                percent_valid=None,  # TODO: add to UI
                local=hardware.lower() == "local-ui",
            )
        elif task == "image-regression":
            dset = AutoTrainImageRegressionDataset(
                train_data=training_files[0],
                token=token,
                project_name=project_name,
                username=autotrain_user,
                valid_data=validation_files[0] if validation_files else None,
                percent_valid=None,  # TODO: add to UI
                local=hardware.lower() == "local-ui",
            )
        elif task == "image-object-detection":
            dset = AutoTrainObjectDetectionDataset(
                train_data=training_files[0],
                token=token,
                project_name=project_name,
                username=autotrain_user,
                valid_data=validation_files[0] if validation_files else None,
                percent_valid=None,  # TODO: add to UI
                local=hardware.lower() == "local-ui",
            )
        elif task.startswith("vlm:"):
            dset = AutoTrainVLMDataset(
                train_data=training_files[0],
                token=token,
                project_name=project_name,
                username=autotrain_user,
                column_mapping=column_mapping,
                valid_data=validation_files[0] if validation_files else None,
                percent_valid=None,  # TODO: add to UI
                local=hardware.lower() == "local-ui",
            )
        else:
            if task.startswith("llm"):
                dset_task = "lm_training"
            elif task.startswith("st:"):
                dset_task = "sentence_transformers"
            elif task == "text-classification":
                dset_task = "text_multi_class_classification"
            elif task == "text-regression":
                dset_task = "text_single_column_regression"
            elif task == "seq2seq":
                dset_task = "seq2seq"
            elif task.startswith("tabular"):
                if "," in column_mapping["label"]:
                    column_mapping["label"] = column_mapping["label"].split(",")
                else:
                    column_mapping["label"] = [column_mapping["label"]]
                column_mapping["label"] = [col.strip() for col in column_mapping["label"]]
                subtask = task.split(":")[-1].lower()
                if len(column_mapping["label"]) > 1 and subtask == "classification":
                    dset_task = "tabular_multi_label_classification"
                elif len(column_mapping["label"]) == 1 and subtask == "classification":
                    dset_task = "tabular_multi_class_classification"
                elif len(column_mapping["label"]) > 1 and subtask == "regression":
                    dset_task = "tabular_multi_column_regression"
                elif len(column_mapping["label"]) == 1 and subtask == "regression":
                    dset_task = "tabular_single_column_regression"
                else:
                    raise NotImplementedError
            elif task == "token-classification":
                dset_task = "text_token_classification"
            elif task == "extractive-qa":
                dset_task = "text_extractive_question_answering"
            else:
                raise NotImplementedError
            logger.info(f"Task: {dset_task}")
            logger.info(f"Column mapping: {column_mapping}")
            dset_args = dict(
                train_data=training_files,
                task=dset_task,
                token=token,
                project_name=project_name,
                username=autotrain_user,
                column_mapping=column_mapping,
                valid_data=validation_files,
                percent_valid=None,  # TODO: add to UI
                local=hardware.lower() == "local-ui",
                ext=file_extension,
            )
            if task in ("text-classification", "token-classification", "st:pair_class"):
                dset_args["convert_to_class_label"] = True
            dset = AutoTrainDataset(**dset_args)
        data_path = dset.prepare()
    else:
        data_path = hub_dataset
    app_params = AppParams(
        job_params_json=json.dumps(params),
        token=token,
        project_name=project_name,
        username=autotrain_user,
        task=task,
        data_path=data_path,
        base_model=base_model,
        column_mapping=column_mapping,
        using_hub_dataset=len(hub_dataset) > 0,
        train_split=None if len(hub_dataset) == 0 else train_split,
        valid_split=None if len(hub_dataset) == 0 else valid_split,
    )
    params = app_params.munge()
    project = AutoTrainProject(params=params, backend=hardware)
    job_id = project.create()

    monitor_url = ""
    if hardware == "local-ui":
        DB.add_job(job_id)
        monitor_url = "Monitor your job locally / in logs"
    elif hardware.startswith("ep-"):
        monitor_url = f"https://ui.endpoints.huggingface.co/{autotrain_user}/endpoints/{job_id}"
    elif hardware.startswith("spaces-"):
        monitor_url = f"https://hf.co/spaces/{job_id}"
    else:
        monitor_url = f"Success! Monitor your job in logs. Job ID: {job_id}"

    wandb_command = None
    wandb_visualizer_active = False
    wandb_visualizer_error = None

    is_local_wandb = hardware == "local-ui" and getattr(params, "log", "none") == "wandb"
    if is_local_wandb:
        run_dir = params.project_name
        wandb_command = f'WANDB_DIR="{run_dir}" wandb beta leet "{run_dir}"'
        if getattr(params, "wandb_visualizer", False):
            wandb_visualizer_active = WANDB_VISUALIZER_MANAGER.start(run_dir, getattr(params, "wandb_token", None))
            if not wandb_visualizer_active:
                wandb_visualizer_error = WANDB_VISUALIZER_MANAGER.status().get("error")
        else:
            WANDB_VISUALIZER_MANAGER.stop()
    else:
        WANDB_VISUALIZER_MANAGER.stop()

    response = {
        "success": "true",
        "monitor_url": monitor_url,
        "wandb_command": wandb_command,
        "wandb_visualizer_active": wandb_visualizer_active,
    }
    if wandb_visualizer_error:
        response["wandb_visualizer_error"] = wandb_visualizer_error
    return response


@ui_router.get("/help/{element_id}", response_class=JSONResponse)
async def fetch_help(element_id: str, authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the help text for a given element
    :param element_id: str
    :return: JSONResponse
    """
    msg = get_app_help(element_id)
    return {"message": msg}


@ui_router.get("/accelerators", response_class=JSONResponse)
async def available_accelerators(authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the number of available accelerators
    :return: JSONResponse
    """
    if AUTOTRAIN_LOCAL == 0:
        return {"accelerators": "Not available in cloud mode."}
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if cuda_available:
        num_gpus = torch.cuda.device_count()
    elif mps_available:
        num_gpus = 1
    else:
        num_gpus = 0
    return {"accelerators": num_gpus}


@ui_router.get("/is_model_training", response_class=JSONResponse)
async def is_model_training(authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the number of running jobs
    :return: JSONResponse
    """
    if AUTOTRAIN_LOCAL == 0:
        return {"model_training": "Not available in cloud mode."}
    running_jobs = get_running_jobs(DB)
    if running_jobs:
        return {"model_training": True, "pids": running_jobs}
    return {"model_training": False, "pids": []}


@ui_router.get("/logs", response_class=JSONResponse)
async def fetch_logs(authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the logs
    :return: JSONResponse
    """
    if not AUTOTRAIN_LOCAL:
        return {"logs": "Logs are only available in local mode."}
    log_file = "autotrain.log"
    with open(log_file, "r", encoding="utf-8") as f:
        logs = f.read()
    if len(str(logs).strip()) == 0:
        logs = "No logs available."

    logs = logs.split("\n")
    logs = logs[::-1]
    # remove lines containing /is_model_training & /accelerators
    logs = [log for log in logs if "/ui/" not in log and "/static/" not in log and "nvidia-ml-py" not in log]

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        devices = Device.all()
        device_logs = []
        for device in devices:
            device_logs.append(
                f"Device {device.index}: {device.name()} - {device.memory_used_human()}/{device.memory_total_human()}"
            )
        device_logs.append("-----------------")
        logs = device_logs + logs
    return {"logs": logs}


@ui_router.get("/stop_training", response_class=JSONResponse)
async def stop_training(authenticated: bool = Depends(user_authentication)):
    """
    This function is used to stop the training
    :return: JSONResponse
    """
    running_jobs = get_running_jobs(DB)
    WANDB_VISUALIZER_MANAGER.stop()
    if running_jobs:
        for _pid in running_jobs:
            try:
                kill_process_by_pid(_pid)
            except Exception:
                logger.info(f"Process {_pid} is already completed. Skipping...")
        return {"success": True}
    return {"success": False}


@ui_router.get("/wandb_visualizer/status", response_class=JSONResponse)
async def wandb_visualizer_status(authenticated: bool = Depends(user_authentication)):
    """Return current W&B LEET viewer status for the web UI."""
    if AUTOTRAIN_LOCAL == 0:
        return {"active": False, "message": "W&B visualizer is only available in local mode."}
    return WANDB_VISUALIZER_MANAGER.status()
