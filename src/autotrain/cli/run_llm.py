import sys
from argparse import ArgumentParser
from typing import Optional

from autotrain import logger
from autotrain.cli.utils import flag_was_provided, get_field_info
from autotrain.project import AutoTrainProject
from autotrain.trainers.clm.params import LLMTrainingParams

from . import BaseAutoTrainCommand


# Parameter grouping metadata
VALID_TRAINERS = {
    "default",
    "sft",
    "dpo",
    "orpo",
    "ppo",
    "reward",
    "distillation",
}
TRAINER_ALIASES = {
    "generic": "default",
}


def normalize_trainer_name(name: Optional[str]) -> str:
    """Normalize trainer aliases and default values."""
    if not name:
        normalized = "default"
    else:
        normalized = name.strip().lower()
        if not normalized:
            normalized = "default"
    return TRAINER_ALIASES.get(normalized, normalized)


FIELD_GROUPS = {
    # Core/Basic
    "model": "Basic",
    "project_name": "Basic",
    "data_path": "Basic",
    "train_split": "Basic",
    "valid_split": "Basic",
    "max_samples": "Basic",
    # Data Processing
    "add_eos_token": "Data Processing",
    "model_max_length": "Data Processing",
    "padding": "Data Processing",
    "text_column": "Data Processing",
    "prompt_text_column": "Data Processing",
    "rejected_text_column": "Data Processing",
    "block_size": "Data Processing",
    "chat_template": "Data Processing",
    "chat_format": "Data Processing",
    "token_weights": "Data Processing",
    "auto_convert_dataset": "Data Processing",
    "conversation_extension": "Data Processing",
    "apply_chat_template": "Data Processing",
    # Training Configuration
    "trainer": "Training Configuration",
    "use_flash_attention_2": "Training Configuration",
    "attn_implementation": "Training Configuration",
    "packing": "Training Configuration",
    "log": "Training Configuration",
    "disable_gradient_checkpointing": "Training Configuration",
    "logging_steps": "Training Configuration",
    "eval_strategy": "Training Configuration",
    "save_strategy": "Training Configuration",
    "save_steps": "Training Configuration",
    "save_total_limit": "Training Configuration",
    "auto_find_batch_size": "Training Configuration",
    "mixed_precision": "Training Configuration",
    "distributed_backend": "Training Configuration",
    "wandb_visualizer": "Training Configuration",
    # Training Hyperparameters
    "lr": "Training Hyperparameters",
    "epochs": "Training Hyperparameters",
    "batch_size": "Training Hyperparameters",
    "warmup_ratio": "Training Hyperparameters",
    "gradient_accumulation": "Training Hyperparameters",
    "optimizer": "Training Hyperparameters",
    "scheduler": "Training Hyperparameters",
    "weight_decay": "Training Hyperparameters",
    "max_grad_norm": "Training Hyperparameters",
    "seed": "Training Hyperparameters",
    # PEFT/LoRA
    "quantization": "PEFT/LoRA",
    "target_modules": "PEFT/LoRA",
    "merge_adapter": "PEFT/LoRA",
    "peft": "PEFT/LoRA",
    "lora_r": "PEFT/LoRA",
    "lora_alpha": "PEFT/LoRA",
    "lora_dropout": "PEFT/LoRA",
    # DPO/ORPO
    "model_ref": "DPO/ORPO",
    "dpo_beta": "DPO/ORPO",
    "max_prompt_length": "DPO/ORPO",
    "max_completion_length": "DPO/ORPO",
    # Hub Integration
    "push_to_hub": "Hub Integration",
    "username": "Hub Integration",
    "token": "Hub Integration",
    "wandb_token": "Hub Integration",
    "unsloth": "Hub Integration",
    # Knowledge Distillation
    "use_distillation": "Knowledge Distillation",
    "teacher_model": "Knowledge Distillation",
    "teacher_prompt_template": "Knowledge Distillation",
    "student_prompt_template": "Knowledge Distillation",
    "distill_temperature": "Knowledge Distillation",
    "distill_alpha": "Knowledge Distillation",
    "distill_max_teacher_length": "Knowledge Distillation",
    # Hyperparameter Sweep
    "use_sweep": "Hyperparameter Sweep",
    "sweep_backend": "Hyperparameter Sweep",
    "sweep_n_trials": "Hyperparameter Sweep",
    "sweep_metric": "Hyperparameter Sweep",
    "sweep_direction": "Hyperparameter Sweep",
    "sweep_params": "Hyperparameter Sweep",
    # Enhanced Evaluation
    "use_enhanced_eval": "Enhanced Evaluation",
    "eval_metrics": "Enhanced Evaluation",
    "eval_dataset_path": "Enhanced Evaluation",
    "eval_batch_size": "Enhanced Evaluation",
    "eval_save_predictions": "Enhanced Evaluation",
    "eval_benchmark": "Enhanced Evaluation",
    # Reinforcement Learning (PPO)
    "rl_gamma": "Reinforcement Learning (PPO)",
    "rl_gae_lambda": "Reinforcement Learning (PPO)",
    "rl_kl_coef": "Reinforcement Learning (PPO)",
    "rl_value_loss_coef": "Reinforcement Learning (PPO)",
    "rl_clip_range": "Reinforcement Learning (PPO)",
    "rl_reward_fn": "Reinforcement Learning (PPO)",
    "rl_multi_objective": "Reinforcement Learning (PPO)",
    "rl_reward_weights": "Reinforcement Learning (PPO)",
    "rl_env_type": "Reinforcement Learning (PPO)",
    "rl_env_config": "Reinforcement Learning (PPO)",
    "rl_reward_model_path": "Reinforcement Learning (PPO)",
    "rl_num_ppo_epochs": "Reinforcement Learning (PPO)",
    "rl_chunk_size": "Reinforcement Learning (PPO)",
    "rl_mini_batch_size": "Reinforcement Learning (PPO)",
    "rl_optimize_device_cache": "Reinforcement Learning (PPO)",
    "rl_value_clip_range": "Reinforcement Learning (PPO)",
    "rl_max_new_tokens": "Reinforcement Learning (PPO)",
    "rl_top_k": "Reinforcement Learning (PPO)",
    "rl_top_p": "Reinforcement Learning (PPO)",
    "rl_temperature": "Reinforcement Learning (PPO)",
    # Advanced/Research Features
    "custom_loss": "Advanced Features",
    "custom_loss_weights": "Advanced Features",
    "custom_metrics": "Advanced Features",
    "use_forward_backward": "Advanced Features",
    "forward_backward_loss_fn": "Advanced Features",
    "forward_backward_custom_fn": "Advanced Features",
    "gradient_accumulation_steps": "Advanced Features",
    "manual_optimizer_control": "Advanced Features",
    "optimizer_step_frequency": "Advanced Features",
    "grad_clip_value": "Advanced Features",
    "learning_rate": "Advanced Features",
    "num_epochs": "Advanced Features",
    "warmup_steps": "Advanced Features",
    "manual_checkpoint_control": "Advanced Features",
    "save_state_every_n_steps": "Advanced Features",
    "load_state_from": "Advanced Features",
    "sample_every_n_steps": "Advanced Features",
    "sample_prompts": "Advanced Features",
    "sample_temperature": "Advanced Features",
    "sample_top_k": "Advanced Features",
    "sample_top_p": "Advanced Features",
    # Inference
    "inference_prompts": "Inference",
    "inference_max_tokens": "Inference",
    "inference_temperature": "Inference",
    "inference_top_p": "Inference",
    "inference_top_k": "Inference",
    "inference_output": "Inference",
}

# Parameter scope metadata (which trainers can use each parameter)
FIELD_SCOPES = {
    # Core params - available to all
    "model": ["all"],
    "project_name": ["all"],
    "data_path": ["all"],
    "train_split": ["all"],
    "valid_split": ["all"],
    "max_samples": ["all"],
    "add_eos_token": ["all"],
    "model_max_length": ["all"],
    "padding": ["all"],
    "text_column": ["all"],
    "block_size": ["all"],
    # Training config - all trainers
    "trainer": ["all"],
    "use_flash_attention_2": ["all"],
    "attn_implementation": ["all"],
    "packing": ["all"],
    "log": ["all"],
    "disable_gradient_checkpointing": ["all"],
    "logging_steps": ["all"],
    "eval_strategy": ["all"],
    "save_strategy": ["all"],
    "save_steps": ["all"],
    "save_total_limit": ["all"],
    "auto_find_batch_size": ["all"],
    "mixed_precision": ["all"],
    "distributed_backend": ["all"],
    "wandb_visualizer": ["all"],
    # Hyperparameters - all trainers
    "lr": ["all"],
    "epochs": ["all"],
    "batch_size": ["all"],
    "warmup_ratio": ["all"],
    "gradient_accumulation": ["all"],
    "optimizer": ["all"],
    "scheduler": ["all"],
    "weight_decay": ["all"],
    "max_grad_norm": ["all"],
    "seed": ["all"],
    "chat_template": ["all"],
    "chat_format": ["all"],
    "auto_convert_dataset": ["all"],
    "use_sharegpt_mapping": ["all"],
    "sharegpt_mapping_enabled": ["all"],
    "suggested_chat_template": ["all"],
    "conversation_extension": ["all"],
    "apply_chat_template": ["all"],
    "column_mapping": ["all"],
    "runtime_mapping": ["all"],
    "map_eos_token": ["all"],
    # PEFT - all trainers
    "quantization": ["all"],
    "target_modules": ["all"],
    "merge_adapter": ["all"],
    "peft": ["all"],
    "lora_r": ["all"],
    "lora_alpha": ["all"],
    "lora_dropout": ["all"],
    # Hub - all trainers
    "push_to_hub": ["all"],
    "username": ["all"],
    "token": ["all"],
    "wandb_token": ["all"],
    "unsloth": ["all"],
    # DPO/ORPO specific
    "model_ref": ["dpo", "orpo", "ppo"],
    "dpo_beta": ["dpo", "orpo"],
    "max_prompt_length": ["dpo", "orpo"],
    "max_completion_length": ["dpo", "orpo", "sft", "default"],
    "prompt_text_column": ["dpo", "orpo"],
    "rejected_text_column": ["dpo", "orpo"],
    # Distillation specific
    "use_distillation": ["default", "sft"],
    "teacher_model": ["default", "sft"],
    "teacher_prompt_template": ["default", "sft"],
    "student_prompt_template": ["default", "sft"],
    "distill_temperature": ["default", "sft"],
    "distill_alpha": ["default", "sft"],
    "distill_max_teacher_length": ["default", "sft"],
    # Sweep - all trainers
    "use_sweep": ["all"],
    "sweep_backend": ["all"],
    "sweep_n_trials": ["all"],
    "sweep_metric": ["all"],
    "sweep_direction": ["all"],
    "sweep_params": ["all"],
    "wandb_sweep": ["all"],
    "wandb_sweep_project": ["all"],
    "wandb_sweep_entity": ["all"],
    "wandb_sweep_id": ["all"],
    "wandb_run_id": ["all"],
    # Enhanced eval - all trainers
    "use_enhanced_eval": ["all"],
    "eval_metrics": ["all"],
    "eval_dataset_path": ["all"],
    "eval_batch_size": ["all"],
    "eval_save_predictions": ["all"],
    "eval_benchmark": ["all"],
    # RL/PPO specific
    "rl_gamma": ["ppo"],
    "rl_gae_lambda": ["ppo"],
    "rl_kl_coef": ["ppo"],
    "rl_value_loss_coef": ["ppo"],
    "rl_clip_range": ["ppo"],
    "rl_reward_fn": ["ppo"],
    "rl_multi_objective": ["ppo"],
    "rl_reward_weights": ["ppo"],
    "rl_env_type": ["ppo"],
    "rl_env_config": ["ppo"],
    "rl_reward_model_path": ["ppo"],
    "rl_num_ppo_epochs": ["ppo"],
    "rl_chunk_size": ["ppo"],
    "rl_mini_batch_size": ["ppo"],
    "rl_optimize_device_cache": ["ppo"],
    "rl_value_clip_range": ["ppo"],
    "rl_max_new_tokens": ["ppo"],
    "rl_top_k": ["ppo"],
    "rl_top_p": ["ppo"],
    "rl_temperature": ["ppo"],
    # Advanced features
    "custom_loss": ["all"],
    "custom_loss_weights": ["all"],
    "custom_metrics": ["all"],
    "use_forward_backward": ["sft"],
    "forward_backward_loss_fn": ["sft"],
    "forward_backward_custom_fn": ["sft"],
    "gradient_accumulation_steps": ["sft"],
    "manual_optimizer_control": ["sft"],
    "optimizer_step_frequency": ["sft"],
    "grad_clip_value": ["sft"],
    "token_weights": ["all"],
    "manual_checkpoint_control": ["all"],
    "save_state_every_n_steps": ["all"],
    "load_state_from": ["all"],
    "sample_every_n_steps": ["all"],
    "sample_prompts": ["all"],
    "sample_temperature": ["all"],
    "sample_top_k": ["all"],
    "sample_top_p": ["all"],
    # Advanced params that appear in params but are for internal use
    "learning_rate": ["all"],
    "num_epochs": ["all"],
    "warmup_steps": ["all"],
    # Inference
    "inference_prompts": ["all"],
    "inference_max_tokens": ["all"],
    "inference_temperature": ["all"],
    "inference_top_p": ["all"],
    "inference_top_k": ["all"],
    "inference_output": ["all"],
}


def run_llm_command_factory(args):
    return RunAutoTrainLLMCommand(args)


class RunAutoTrainLLMCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        # Extract only the args after 'llm' subcommand for parsing
        args_after_llm = []
        if "llm" in sys.argv:
            llm_index = sys.argv.index("llm")
            args_after_llm = sys.argv[llm_index + 1 :]
        else:
            args_after_llm = sys.argv[1:]

        # Create a temporary parser to extract --trainer early
        # Use allow_abbrev=False to prevent --train from matching --trainer
        temp_parser = ArgumentParser(add_help=False, allow_abbrev=False)
        temp_parser.add_argument("--trainer", type=str, default="default")
        # Preview-only flags (help-time filtering); keep alias for backwards compatibility
        temp_parser.add_argument("--help-trainer", type=str, default=None, nargs="?")
        temp_parser.add_argument("--preview-trainer", type=str, default=None, nargs="?")

        # Parse only the subcommand args to get trainer value
        temp_args, _ = temp_parser.parse_known_args(args_after_llm)
        # Normalize runtime trainer
        selected_trainer = normalize_trainer_name(temp_args.trainer)
        # Help mode detection
        help_mode = ("--help" in args_after_llm) or ("-h" in args_after_llm)
        # Resolve preview trainer for help-only filtering
        _preview_raw = (temp_args.preview_trainer or temp_args.help_trainer or "").strip()
        _preview = normalize_trainer_name(_preview_raw) if _preview_raw else None
        if help_mode and _preview:
            selected_trainer = _preview

        # Get field info with group and scope metadata
        # Enable strict enforcement to catch missing metadata during development
        arg_list = get_field_info(LLMTrainingParams, FIELD_GROUPS, FIELD_SCOPES, enforce_scope=True)

        # Add command-specific args at the beginning
        command_args = [
            {
                "arg": "--train",
                "help": "Command to train the model",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--deploy",
                "help": "Command to deploy the model (limited availability)",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--inference",
                "help": "Command to run inference (limited availability)",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--interactive",
                "help": "Launch interactive wizard to configure training",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--backend",
                "help": "Backend",
                "required": False,
                "type": str,
                "default": "local",
            },
            {
                "arg": "--help-trainer",
                "help": "[Deprecated] Preview help for specific trainer (use --preview-trainer)",
                "required": False,
                "type": str,
                "nargs": "?",
            },
            {
                "arg": "--preview-trainer",
                "help": "Preview help for specific trainer (help-only; does not set --trainer)",
                "required": False,
                "type": str,
                "nargs": "?",
            },
            {
                "arg": "--wandb-visualizer",
                "help": "Enable W&B visualizer (LEET). Default enabled when log='wandb'.",
                "required": False,
                "action": "store_true",
                "dest": "wandb_visualizer",
                "group": "Training Configuration",
                "scope": ["all"],
            },
            {
                "arg": "--no-wandb-visualizer",
                "help": "Disable W&B visualizer (LEET).",
                "required": False,
                "action": "store_false",
                "dest": "wandb_visualizer",
                "group": "Training Configuration",
                "scope": ["all"],
            },
        ]
        arg_list = command_args + arg_list

        # Filter args based on selected trainer scope
        if selected_trainer and selected_trainer != "default":
            filtered_args = []
            for arg in arg_list:
                scope = arg.get("scope", ["all"])
                # Include if scope is "all" or if selected trainer is in scope list
                if "all" in scope or selected_trainer in scope:
                    filtered_args.append(arg)
            arg_list = filtered_args

        # Handle block_size special case
        arg_list = [arg for arg in arg_list if arg["arg"] != "--block-size"]
        arg_list.append(
            {
                "arg": "--block_size",
                "help": "Block size",
                "required": False,
                "type": str,
                "default": "1024",
                "alias": ["--block-size"],
                "group": "Data Processing",
            }
        )

        # Create the main parser
        run_llm_parser = parser.add_parser("llm", description="✨ Run AITraining LLM")

        # Show a more specific description in help mode
        if help_mode and (getattr(temp_args, "preview_trainer", None) or getattr(temp_args, "help_trainer", None)):
            _disp = (temp_args.preview_trainer or temp_args.help_trainer).strip().upper()
            run_llm_parser.description = f"✨ Run AITraining LLM (showing {_disp} trainer parameters)"
        elif selected_trainer and selected_trainer != "default" and ("--help" in sys.argv):
            run_llm_parser.description = (
                f"✨ Run AITraining LLM (showing {selected_trainer.upper()} trainer parameters)"
            )

        # Group arguments by their group metadata
        groups = {}
        ungrouped_args = []

        for arg in arg_list:
            group_name = arg.get("group")
            if group_name:
                if group_name not in groups:
                    groups[group_name] = run_llm_parser.add_argument_group(group_name)
                target_parser = groups[group_name]
            else:
                target_parser = run_llm_parser
                ungrouped_args.append(arg)

            # Add the argument
            names = [arg["arg"]] + arg.get("alias", [])
            if "action" in arg:
                # Special handling for merge_adapter: add both --merge-adapter and --no-merge-adapter
                if arg["arg"] == "--merge-adapter":
                    # Add positive flag (sets to True)
                    target_parser.add_argument(
                        "--merge-adapter",
                        dest="merge_adapter",
                        help="Merge PEFT adapters into base model (easier deployment, larger size)",
                        action="store_true",
                        default=None,  # None means "not specified, use Pydantic default"
                    )
                    # Add negative flag (sets to False)
                    target_parser.add_argument(
                        "--no-merge-adapter",
                        dest="merge_adapter",
                        help="Save only PEFT adapters (smaller size, requires base model)",
                        action="store_false",
                        default=None,
                    )
                else:
                    target_parser.add_argument(
                        *names,
                        dest=arg["arg"].replace("--", "").replace("-", "_"),
                        help=arg["help"],
                        required=arg.get("required", False),
                        action=arg.get("action"),
                        default=arg.get("default"),
                    )
            else:
                kwargs = {
                    "dest": arg["arg"].replace("--", "").replace("-", "_"),
                    "help": arg["help"],
                    "required": arg.get("required", False),
                    "type": arg.get("type"),
                    "default": arg.get("default"),
                    "choices": arg.get("choices"),
                }
                if "nargs" in arg:
                    kwargs["nargs"] = arg.get("nargs")
                target_parser.add_argument(*names, **kwargs)

        run_llm_parser.set_defaults(func=run_llm_command_factory)

    def __init__(self, args):
        self.args = args

        raw_trainer_value = getattr(self.args, "trainer", None)
        normalized_trainer = normalize_trainer_name(raw_trainer_value)
        if normalized_trainer not in VALID_TRAINERS:
            valid_options = sorted(VALID_TRAINERS.union(TRAINER_ALIASES.keys()))
            message = f"Invalid trainer '{raw_trainer_value}'. " f"Valid options: {', '.join(valid_options)}"
            logger.error(message)
            raise SystemExit(message)
        self.args.trainer = normalized_trainer

        store_true_arg_names = [
            "train",
            "deploy",
            "inference",
            "interactive",
            "add_eos_token",
            "peft",
            "auto_find_batch_size",
            "push_to_hub",
            # merge_adapter NOT in this list - handled specially with --merge-adapter/--no-merge-adapter flags
            "use_flash_attention_2",
            "disable_gradient_checkpointing",
            # Tinker features
            "use_distillation",
            "use_sweep",
            "use_enhanced_eval",
            "eval_save_predictions",
            # RL features
            "rl_multi_objective",
            "rl_optimize_device_cache",
            # Advanced
            "use_forward_backward",
            "manual_optimizer_control",
            "packing",
            "unsloth",
        ]
        for arg_name in store_true_arg_names:
            if getattr(self.args, arg_name, None) is None:
                setattr(self.args, arg_name, False)

        # Special handling for merge_adapter: if None, don't set it (let Pydantic use default)
        # If True or False, keep that value
        if getattr(self.args, "merge_adapter", None) is None:
            # Don't pass it to Pydantic at all - remove from args
            delattr(self.args, "merge_adapter")

        block_size_split = self.args.block_size.strip().split(",")
        if len(block_size_split) == 1:
            self.args.block_size = int(block_size_split[0])
        elif len(block_size_split) > 1:
            self.args.block_size = [int(x.strip()) for x in block_size_split]
        else:
            raise ValueError("Invalid block size")

        # Check if we should launch interactive wizard
        should_launch_wizard = False

        if self.args.interactive:
            # Explicitly requested interactive mode
            should_launch_wizard = True
        elif self.args.train:
            # Check if required params are missing when --train is specified
            missing_params = []
            project_flag = flag_was_provided(["--project-name", "--project_name"])
            data_flag = flag_was_provided(["--data-path", "--data_path"])
            model_flag = flag_was_provided(["--model"])

            if self.args.project_name is None or (not project_flag and self.args.project_name == "project-name"):
                missing_params.append("project_name")
            if self.args.data_path is None or (not data_flag and self.args.data_path == "data"):
                missing_params.append("data_path")
            if self.args.model is None or (not model_flag and self.args.model == "google/gemma-3-270m"):
                missing_params.append("model")

            if missing_params:
                logger.info(
                    f"Missing or default values detected for: {', '.join(missing_params)}. "
                    "Launching interactive wizard..."
                )
                should_launch_wizard = True

        if should_launch_wizard:
            from autotrain.cli.interactive_wizard import run_wizard

            # Collect current args to pass to wizard
            initial_args = {k: v for k, v in vars(self.args).items() if v is not None}

            # Run wizard
            try:
                wizard_config = run_wizard(initial_args, trainer_type="llm")

                # Merge wizard results back into self.args
                for key, value in wizard_config.items():
                    setattr(self.args, key, value)

                # Ensure train flag is set
                self.args.train = True

            except KeyboardInterrupt:
                logger.info("Interactive wizard cancelled by user.")
                sys.exit(0)
            except Exception as e:
                logger.error(f"Error in interactive wizard: {e}")
                raise

        if self.args.train:
            if self.args.project_name is None:
                raise ValueError("Project name must be specified")
            if self.args.data_path is None:
                raise ValueError("Data path must be specified")
            if self.args.model is None:
                raise ValueError("Model must be specified")
            if self.args.push_to_hub:
                # must have project_name, username and token OR project_name, token
                if self.args.username is None:
                    raise ValueError("Username must be specified for push to hub")
                if self.args.token is None:
                    raise ValueError("Token must be specified for push to hub")

            if self.args.backend.startswith("spaces") or self.args.backend.startswith("ep-"):
                if not self.args.push_to_hub:
                    raise ValueError("Push to hub must be specified for spaces backend")
                if self.args.username is None:
                    raise ValueError("Username must be specified for spaces backend")
                if self.args.token is None:
                    raise ValueError("Token must be specified for spaces backend")

        if self.args.deploy:
            raise NotImplementedError("Deploy is not implemented yet")

        if self.args.inference:
            if not self.args.model:
                raise ValueError("Model must be specified for inference")
            if not self.args.inference_prompts:
                raise ValueError("Prompts must be specified for inference (--inference-prompts)")

    def run(self):
        logger.info("Running LLM")

        # Handle inference mode
        if self.args.inference:
            return self._run_inference()

        if self.args.train:
            params = LLMTrainingParams(**vars(self.args))
            project = AutoTrainProject(params=params, backend=self.args.backend, process=True)
            job_id = project.create()
            logger.info(f"Job ID: {job_id}")

    def _run_inference(self):
        """Run inference using completers."""
        import json
        import os

        from autotrain.generation import CompletionConfig, create_completer

        logger.info("Running inference mode...")

        # Parse prompts
        if os.path.exists(self.args.inference_prompts):
            # Load from file
            with open(self.args.inference_prompts, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            # Comma-separated prompts
            prompts = [p.strip() for p in self.args.inference_prompts.split(",")]

        logger.info(f"Loaded {len(prompts)} prompts for inference")

        # Create completion config
        config = CompletionConfig(
            max_tokens=self.args.inference_max_tokens,
            temperature=self.args.inference_temperature,
            top_p=self.args.inference_top_p,
            top_k=self.args.inference_top_k,
        )

        # Create completer
        completer = create_completer(self.args.model, completer_type="message", config=config)

        # Run inference
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            response = completer.chat(prompt)
            results.append({"prompt": prompt, "response": response})

        # Save results
        output_path = self.args.inference_output or f"{self.args.project_name}/inference_results.json"
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Inference complete! Results saved to {output_path}")
        return results
