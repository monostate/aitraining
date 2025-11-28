"""Application state management for AITraining TUI."""

from typing import Any, Dict, List


class AppState:
    """Manages the application state for the TUI."""

    def __init__(self):
        """Initialize the application state."""
        self.fields: List[Dict] = []
        self.parameters: Dict[str, Any] = {}
        self.current_trainer: str = "default"
        self.current_group: str = "Basic"
        self.groups: Dict[str, List[Dict]] = {}
        self.search_query: str = ""
        self.modified_fields: set = set()

    def initialize_fields(self, field_info: List[Dict]) -> None:
        """Initialize fields from parameter schema."""
        self.fields = field_info
        # Initialize parameters with defaults
        for field in field_info:
            field_name = field["arg"].lstrip("-").replace("-", "_")
            default = field.get("default")
            self.parameters[field_name] = default

    def set_trainer(self, trainer: str) -> None:
        """Set the current trainer."""
        self.current_trainer = trainer

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a parameter value."""
        # Get default value
        default = None
        for field in self.fields:
            field_name = field["arg"].lstrip("-").replace("-", "_")
            if field_name == name:
                default = field.get("default")
                break

        # Update parameter
        self.parameters[name] = value

        # Track modifications
        if value != default:
            self.modified_fields.add(name)
        elif name in self.modified_fields:
            self.modified_fields.remove(name)

    def get_parameter(self, name: str) -> Any:
        """Get a parameter value."""
        return self.parameters.get(name)

    def get_modified_parameters(self) -> Dict[str, Any]:
        """Get only modified parameters."""
        modified = {}
        for name in self.modified_fields:
            value = self.parameters.get(name)
            if value is not None:
                modified[name] = value
        return modified

    def reset_parameter(self, name: str) -> None:
        """Reset a parameter to its default value."""
        for field in self.fields:
            field_name = field["arg"].lstrip("-").replace("-", "_")
            if field_name == name:
                default = field.get("default")
                self.parameters[name] = default
                if name in self.modified_fields:
                    self.modified_fields.remove(name)
                break

    def reset_all_parameters(self) -> None:
        """Reset all parameters to their default values."""
        for field in self.fields:
            field_name = field["arg"].lstrip("-").replace("-", "_")
            default = field.get("default")
            self.parameters[field_name] = default
        self.modified_fields.clear()

    def is_field_visible(self, field: Dict) -> bool:
        """Check if a field is visible for the current trainer."""
        scopes = field.get("scope", ["all"])
        return "all" in scopes or self.current_trainer in scopes

    def get_visible_fields(self) -> List[Dict]:
        """Get all fields visible for the current trainer."""
        return [f for f in self.fields if self.is_field_visible(f)]

    def get_fields_for_group(self, group: str) -> List[Dict]:
        """Get fields for a specific group."""
        return [f for f in self.fields if f.get("group", "Other") == group and self.is_field_visible(f)]

    def validate_parameters(self) -> Dict[str, List[str]]:
        """Validate all parameters and return errors."""
        errors = {}

        # Check required fields
        for field in self.fields:
            if not self.is_field_visible(field):
                continue

            field_name = field["arg"].lstrip("-").replace("-", "_")
            value = self.parameters.get(field_name)

            # Check required fields (model and project_name are always required)
            if field_name in ["model", "project_name"] and not value:
                if field_name not in errors:
                    errors[field_name] = []
                errors[field_name].append("This field is required")

            # Type validation
            if value is not None and value != "":
                field_type = field.get("type", str)
                if field_type == int:
                    try:
                        int(value)
                    except (ValueError, TypeError):
                        if field_name not in errors:
                            errors[field_name] = []
                        errors[field_name].append("Must be an integer")
                elif field_type == float:
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        if field_name not in errors:
                            errors[field_name] = []
                        errors[field_name].append("Must be a number")

        # Trainer-specific validations
        if self.current_trainer == "ppo":
            if not self.parameters.get("rl_reward_model_path"):
                errors["rl_reward_model_path"] = ["PPO requires a reward model path"]

        if self.current_trainer in ["dpo", "orpo"]:
            if not self.parameters.get("model_ref"):
                errors["model_ref"] = [f"{self.current_trainer.upper()} requires a reference model"]

        return errors

    def export_config(self) -> Dict[str, Any]:
        """Export the current configuration."""
        config = {"trainer": self.current_trainer, "parameters": {}}

        # Only export non-default values
        for field in self.fields:
            field_name = field["arg"].lstrip("-").replace("-", "_")
            value = self.parameters.get(field_name)
            default = field.get("default")

            if value is not None and value != "" and value != default:
                config["parameters"][field_name] = value

        return config

    def import_config(self, config: Dict[str, Any]) -> None:
        """Import a configuration."""
        # Set trainer
        if "trainer" in config:
            self.set_trainer(config["trainer"])

        # Set parameters
        if "parameters" in config:
            # Reset first
            self.reset_all_parameters()

            # Apply imported values
            for name, value in config["parameters"].items():
                if name in self.parameters:
                    self.set_parameter(name, value)
