import sys
from typing import Any, Dict, Optional, Sequence, Type

from autotrain.backends.base import AVAILABLE_HARDWARE


def common_args():
    args = [
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
            "arg": "--username",
            "help": "Hugging Face Hub Username",
            "required": False,
            "type": str,
        },
        {
            "arg": "--backend",
            "help": "Backend to use: default or spaces. Spaces backend requires push_to_hub & username. Advanced users only.",
            "required": False,
            "type": str,
            "default": "local",
            "choices": AVAILABLE_HARDWARE.keys(),
        },
        {
            "arg": "--token",
            "help": "Your Hugging Face API token. Token must have write access to the model hub.",
            "required": False,
            "type": str,
        },
        {
            "arg": "--push-to-hub",
            "help": "Push to hub after training will push the trained model to the Hugging Face model hub.",
            "required": False,
            "action": "store_true",
        },
        {
            "arg": "--model",
            "help": "Base model to use for training",
            "required": True,
            "type": str,
        },
        {
            "arg": "--project-name",
            "help": "Output directory / repo id for trained model (must be unique on hub)",
            "required": True,
            "type": str,
        },
        {
            "arg": "--data-path",
            "help": "Train dataset to use. When using cli, this should be a directory path containing training and validation data in appropriate formats",
            "required": False,
            "type": str,
        },
        {
            "arg": "--train-split",
            "help": "Train dataset split to use",
            "required": False,
            "type": str,
            "default": "train",
        },
        {
            "arg": "--valid-split",
            "help": "Validation dataset split to use",
            "required": False,
            "type": str,
            "default": None,
        },
        {
            "arg": "--batch-size",
            "help": "Training batch size to use",
            "required": False,
            "type": int,
            "default": 2,
            "alias": ["--train-batch-size"],
        },
        {
            "arg": "--seed",
            "help": "Random seed for reproducibility",
            "required": False,
            "default": 42,
            "type": int,
        },
        {
            "arg": "--epochs",
            "help": "Number of training epochs",
            "required": False,
            "default": 1,
            "type": int,
        },
        {
            "arg": "--gradient-accumulation",
            "help": "Gradient accumulation steps",
            "required": False,
            "default": 1,
            "type": int,
            "alias": ["--gradient-accumulation"],
        },
        {
            "arg": "--disable-gradient-checkpointing",
            "help": "Disable gradient checkpointing",
            "required": False,
            "action": "store_true",
            "alias": ["--disable-gradient-checkpointing", "--disable-gc"],
        },
        {
            "arg": "--lr",
            "help": "Learning rate",
            "required": False,
            "default": 5e-4,
            "type": float,
        },
        {
            "arg": "--log",
            "help": "Use experiment tracking",
            "required": False,
            "type": str,
            "default": "none",
            "choices": ["none", "wandb", "tensorboard"],
        },
        {
            "arg": "--wandb-visualizer",
            "help": "Enable W&B visualizer (LEET). Default enabled when log='wandb'.",
            "required": False,
            "action": "store_true",
            "dest": "wandb_visualizer",
        },
        {
            "arg": "--no-wandb-visualizer",
            "help": "Disable W&B visualizer (LEET).",
            "required": False,
            "action": "store_false",
            "dest": "wandb_visualizer",
        },
        {
            "arg": "--wandb-token",
            "help": "W&B API Token for syncing runs",
            "required": False,
            "type": str,
        },
    ]
    return args


def python_type_from_schema_field(field_data: dict) -> Type:
    """Converts JSON schema field types to Python types."""
    type_map = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
    }
    field_type = field_data.get("type")
    if field_type:
        return type_map.get(field_type, str)
    elif "anyOf" in field_data:
        for type_option in field_data["anyOf"]:
            if type_option["type"] != "null":
                return type_map.get(type_option["type"], str)
    return str


def get_default_value(field_data: dict) -> Any:
    return field_data["default"]


def get_field_info(params_class, group_map=None, scope_map=None, enforce_scope=False):
    """
    Extract field info from params class with optional group and scope metadata.

    Args:
        params_class: Pydantic model class to extract fields from
        group_map: Optional dict mapping field names to group names
        scope_map: Optional dict mapping field names to list of trainer scopes
        enforce_scope: If True, require all fields to have scope metadata.
                      Default is False for backward compatibility. (default: False)

    Returns:
        List of dicts with field info including optional 'group' and 'scope' keys

    Raises:
        ValueError: If enforce_scope=True and any field is missing scope metadata
    """
    from autotrain import logger

    schema = params_class.model_json_schema()
    properties = schema.get("properties", {})
    field_info = []
    missing_scope_fields = []

    for field_name, field_data in properties.items():
        if field_name == "wandb_visualizer":
            continue
        temp_info = {
            "arg": f"--{field_name.replace('_', '-')}",
            "alias": [f"--{field_name}", f"--{field_name.replace('_', '-')}"],
            "type": python_type_from_schema_field(field_data),
            "help": field_data.get("title", ""),
            "default": get_default_value(field_data),
        }
        if temp_info["type"] == bool:
            temp_info["action"] = "store_true"

        # Add group metadata if provided
        if group_map:
            if field_name in group_map:
                temp_info["group"] = group_map[field_name]
            else:
                # Log warning only when enforcement is enabled
                if enforce_scope:
                    logger.warning(f"Field '{field_name}' is missing from group_map. Using 'Other' as default.")
                temp_info["group"] = "Other"
        # If no group_map provided, don't add group metadata

        # Add scope metadata if provided
        if scope_map:
            if field_name in scope_map:
                temp_info["scope"] = scope_map[field_name]
            else:
                # Track fields missing scope metadata when scope_map is provided
                missing_scope_fields.append(field_name)
                # Default to ["all"] to maintain backward compatibility
                temp_info["scope"] = ["all"]
                # Only log warnings when enforcement is enabled to avoid noise
                if enforce_scope:
                    logger.warning(
                        f"Field '{field_name}' is missing from scope_map. Defaulting to scope=['all']. "
                        "Please update FIELD_SCOPES to include this field."
                    )
        # If no scope_map provided, don't add scope metadata

        field_info.append(temp_info)

    # Enforce scope metadata if required
    if enforce_scope and missing_scope_fields:
        error_msg = (
            f"Scope metadata is required for all fields but missing for: {', '.join(missing_scope_fields)}. "
            "Please update FIELD_SCOPES in run_llm.py to include these fields, or pass enforce_scope=False."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    return field_info


def flag_was_provided(flag_names: Sequence[str]) -> bool:
    """
    Determine if any of the given CLI flags were explicitly supplied by the user.

    Args:
        flag_names: Iterable of flag strings (e.g., ["--model", "--model-name"])

    Returns:
        True if any of the flags appear in sys.argv (supports --flag=value syntax), False otherwise.
    """
    if not flag_names:
        return False

    cli_flags = set()
    for token in sys.argv[1:]:
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0]
        cli_flags.add(flag)

    return any(flag in cli_flags for flag in flag_names)


def should_launch_wizard(args, trainer_type: str) -> bool:
    """
    Determine if wizard should be launched based on args.

    Args:
        args: Parsed command line arguments
        trainer_type: Trainer type (llm, text-classification, etc.)

    Returns:
        True if wizard should be launched, False otherwise
    """
    # Explicit --interactive flag
    if hasattr(args, "interactive") and args.interactive:
        return True

    # For --train, check if required basics are missing or are placeholders
    if hasattr(args, "train") and args.train:
        # Check for missing required fields
        if not hasattr(args, "project_name") or not args.project_name:
            return True
        if not hasattr(args, "data_path") or not args.data_path:
            return True
        if not hasattr(args, "model") or not args.model:
            return True

        # Check for placeholder values
        placeholder_project = {"project-name"}
        placeholder_data = {"data"}
        placeholder_model = set()

        project_flag = flag_was_provided(["--project-name", "--project_name"])
        data_flag = flag_was_provided(["--data-path", "--data_path"])
        model_flag = flag_was_provided(["--model"])

        if args.project_name is None or args.project_name == "":
            return True
        if not project_flag and args.project_name in placeholder_project:
            return True
        if args.data_path is None or args.data_path == "":
            return True
        if not data_flag and args.data_path in placeholder_data:
            return True
        if args.model is None or args.model == "":
            return True
        if not model_flag and args.model in placeholder_model:
            return True

    return False


def launch_wizard_if_needed(args, trainer_type: str) -> Optional[Dict[str, Any]]:
    """
    Launch interactive wizard if conditions are met.

    Args:
        args: Parsed command line arguments
        trainer_type: Trainer type (llm, text-classification, etc.)

    Returns:
        Dict of wizard answers if wizard was launched, None otherwise
    """
    if should_launch_wizard(args, trainer_type):
        from autotrain import logger
        from autotrain.cli.interactive_wizard import run_wizard

        logger.info("Launching interactive wizard...")

        # Convert args to dict for initial values
        initial_args = vars(args) if hasattr(args, "__dict__") else {}

        # Run wizard with trainer type
        wizard_config = run_wizard(initial_args=initial_args, trainer_type=trainer_type)

        return wizard_config

    return None
