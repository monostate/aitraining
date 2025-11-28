"""
Common classes and functions for all trainers.
"""

import json
import os
import shutil
import time
import traceback
from typing import Optional

import requests
from accelerate import PartialState
from huggingface_hub import HfApi
from pydantic import BaseModel, Field, model_validator
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from autotrain import is_colab, logger


ALLOW_REMOTE_CODE = os.environ.get("ALLOW_REMOTE_CODE", "true").lower() == "true"


def get_file_sizes(directory):
    """
    Calculate the sizes of all files in a given directory and its subdirectories.

    Args:
        directory (str): The path to the directory to scan for files.

    Returns:
        dict: A dictionary where the keys are the file paths and the values are the file sizes in gigabytes (GB).
    """
    file_sizes = {}
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            file_size_gb = file_size / (1024**3)  # Convert bytes to GB
            file_sizes[file_path] = file_size_gb
    return file_sizes


def remove_global_step(directory):
    """
    Removes directories that start with 'global_step' within the specified directory.

    This function traverses the given directory and its subdirectories in a bottom-up manner.
    If it finds any directory whose name starts with 'global_step', it deletes that directory
    and all its contents.

    Args:
        directory (str): The path to the directory to be traversed and cleaned.

    Returns:
        None
    """
    for root, dirs, _ in os.walk(directory, topdown=False):
        for name in dirs:
            if name.startswith("global_step"):
                folder_path = os.path.join(root, name)
                print(f"Removing folder: {folder_path}")
                shutil.rmtree(folder_path)


def remove_autotrain_data(config):
    """
    Removes the AutoTrain data directory and global step for a given project.

    Args:
        config (object): Configuration object that contains the project name.

    Raises:
        OSError: If the removal of the directory fails.
    """
    os.system(f"rm -rf {config.project_name}/autotrain-data")
    remove_global_step(config.project_name)


def save_training_params(config):
    """
    Saves the training parameters to a JSON file, excluding the "token" key if it exists.

    Args:
        config (object): Configuration object that contains the project name.

    The function checks if a file named 'training_params.json' exists in the directory
    specified by `config.project_name`. If the file exists, it loads the JSON content,
    removes the "token" key if present, and then writes the updated content back to the file.
    """
    if os.path.exists(f"{config.project_name}/training_params.json"):
        training_params = json.load(open(f"{config.project_name}/training_params.json"))
        if "token" in training_params:
            training_params.pop("token")
            json.dump(
                training_params,
                open(f"{config.project_name}/training_params.json", "w"),
                indent=4,
            )


def pause_endpoint(params):
    """
    Pauses a Hugging Face endpoint using the provided parameters.

    Args:
        params (dict or object): Parameters containing the token required for authorization.
            If a dictionary is provided, it should have a key "token" with the authorization token.
            If an object is provided, it should have an attribute `token` with the authorization token.

    Returns:
        dict: The JSON response from the API call to pause the endpoint.

    Raises:
        KeyError: If the "token" key is missing in the params dictionary.
        requests.exceptions.RequestException: If there is an issue with the API request.

    Environment Variables:
        ENDPOINT_ID: Should be set to the endpoint identifier in the format "username/project_name".
    """
    if isinstance(params, dict):
        token = params["token"]
    else:
        token = params.token
    endpoint_id = os.environ["ENDPOINT_ID"]
    username = endpoint_id.split("/")[0]
    project_name = endpoint_id.split("/")[1]
    api_url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{username}/{project_name}/pause"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(api_url, headers=headers, timeout=120)
    return r.json()


def pause_space(params, is_failure=False):
    """
    Pauses the Hugging Face space and optionally shuts down the endpoint.

    This function checks for the presence of "SPACE_ID" and "ENDPOINT_ID" in the environment variables.
    If "SPACE_ID" is found, it pauses the space and creates a discussion on the Hugging Face platform
    to notify the user about the status of the training run (success or failure).
    If "ENDPOINT_ID" is found, it pauses the endpoint.

    Args:
        params (object): An object containing the necessary parameters, including the token, username, and project name.
        is_failure (bool, optional): A flag indicating whether the training run failed. Defaults to False.

    Raises:
        Exception: If there is an error while creating the discussion on the Hugging Face platform.

    Logs:
        Info: Logs the status of pausing the space and endpoint.
        Warning: Logs any issues encountered while creating the discussion.
        Error: Logs if the model failed to train and the discussion was not created.
    """
    if "SPACE_ID" in os.environ:
        # shut down the space
        logger.info("Pausing space...")
        api = HfApi(token=params.token)

        if is_failure:
            msg = "Your training run has failed! Please check the logs for more details"
            title = "Your training has failed ❌"
        else:
            msg = "Your training run was successful! [Check out your trained model here]"
            msg += f"(https://huggingface.co/{params.username}/{params.project_name})"
            title = "Your training has finished successfully ✅"

        if not params.token.startswith("hf_oauth_"):
            try:
                api.create_discussion(
                    repo_id=os.environ["SPACE_ID"],
                    title=title,
                    description=msg,
                    repo_type="space",
                )
            except Exception as e:
                logger.warning(f"Failed to create discussion: {e}")
                if is_failure:
                    logger.error("Model failed to train and discussion was not created.")
                else:
                    logger.warning("Model trained successfully but discussion was not created.")

        api.pause_space(repo_id=os.environ["SPACE_ID"])
    if "ENDPOINT_ID" in os.environ:
        # shut down the endpoint
        logger.info("Pausing endpoint...")
        pause_endpoint(params)


def monitor(func):
    """
    A decorator that wraps a function to monitor its execution and handle exceptions.

    This decorator performs the following actions:
    1. Retrieves the 'config' parameter from the function's keyword arguments or positional arguments.
    2. Executes the wrapped function.
    3. If an exception occurs during the execution of the wrapped function, logs the error message and stack trace.
    4. Optionally pauses the execution if the environment variable 'PAUSE_ON_FAILURE' is set to 1.
    5. RE-RAISES the exception so validation errors bubble up to the user (critical for UX).

    Args:
        func (callable): The function to be wrapped by the decorator.

    Returns:
        callable: The wrapped function with monitoring capabilities.
    """

    def wrapper(*args, **kwargs):
        config = kwargs.get("config", None)
        if config is None and len(args) > 0:
            config = args[0]

        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"""{func.__name__} has failed due to an exception: {traceback.format_exc()}"""
            logger.error(error_message)
            logger.error(str(e))
            if int(os.environ.get("PAUSE_ON_FAILURE", 1)) == 1:
                pause_space(config, is_failure=True)
            # CRITICAL: Re-raise the exception so users see validation errors
            # This is essential for proper error handling and debugging
            raise

    return wrapper


class AutoTrainParams(BaseModel):
    """
    AutoTrainParams is a base class for all AutoTrain parameters.
    Attributes:
        Config (class): Configuration class for Pydantic model.
        protected_namespaces (tuple): Protected namespaces for the model.
    Methods:
        save(output_dir):
            Save parameters to a JSON file in the specified output directory.
        __str__():
            Return a string representation of the parameters, masking the token if present.
        __init__(**data):
            Initialize the parameters, check for unused/extra parameters, and warn the user if necessary.
            Raises ValueError if project_name is not alphanumeric (with hyphens allowed) or exceeds 50 characters.
    """

    # W&B Visualizer
    wandb_visualizer: Optional[bool] = Field(None, title="Enable W&B visualizer (LEET)")
    wandb_token: Optional[str] = Field(None, title="W&B API Token")

    class Config:
        protected_namespaces = ()

    @model_validator(mode="after")
    def validate_wandb_visualizer(self):
        """Set default for wandb_visualizer based on log param."""
        if self.wandb_visualizer is None:
            if hasattr(self, "log") and self.log == "wandb":
                self.wandb_visualizer = True
            else:
                self.wandb_visualizer = False
        return self

    def save(self, output_dir):
        """
        Save parameters to a json file.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "training_params.json")
        # save formatted json
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4))

    def __str__(self):
        """
        String representation of the parameters.
        """
        data = self.model_dump()
        data["token"] = "*****" if data.get("token") else None
        return str(data)

    def __init__(self, **data):
        """
        Initialize the parameters, check for unused/extra parameters and warn the user.

        Also normalizes project_name path:
        - If absolute path is provided, it's used as-is
        - If relative/name only, it's placed in a 'trainings' directory adjacent to server
        - Can be overridden with AUTOTRAIN_PROJECTS_DIR environment variable
        """
        super().__init__(**data)

        import os

        # Path normalization for project_name
        if hasattr(self, "project_name") and self.project_name and len(self.project_name) > 0:
            # If not an absolute path, normalize it
            if not os.path.isabs(self.project_name):
                # Get base directory from environment or use ../trainings relative to current directory
                # This keeps training outputs separate from the server directory
                base_dir = os.environ.get("AUTOTRAIN_PROJECTS_DIR")
                if base_dir is None:
                    # Create trainings directory at same level as server directory
                    server_parent = os.path.dirname(os.getcwd())
                    base_dir = os.path.join(server_parent, "trainings")

                # Create base directory if it doesn't exist
                os.makedirs(base_dir, exist_ok=True)

                # Normalize the path
                self.project_name = os.path.normpath(os.path.join(base_dir, self.project_name))

                # Log the path normalization for transparency
                if os.environ.get("AUTOTRAIN_TUI_MODE") != "1":
                    logger.info(f"Project path normalized to: {self.project_name}")

        if hasattr(self, "project_name") and self.project_name and len(self.project_name) > 0:
            # make sure project_name is always alphanumeric but can have hyphens or underscores. if not, raise ValueError
            # Only validate the basename, not the full path (for paths like /tmp/test_project)
            project_basename = os.path.basename(self.project_name)
            if not project_basename.replace("-", "").replace("_", "").isalnum():
                raise ValueError("project_name must be alphanumeric but can contain hyphens or underscores")

            # project name basename cannot be more than 50 characters
            if len(project_basename) > 50:
                raise ValueError("project_name basename cannot be more than 50 characters")

        # Parameters not supplied by the user
        defaults = set(self.model_fields.keys())
        supplied = set(data.keys())
        not_supplied = defaults - supplied
        if not_supplied and not is_colab:
            logger.warning(f"Parameters not supplied by user and set to default: {', '.join(not_supplied)}")

        # Parameters that were supplied but not used
        # This is a naive implementation. It might catch some internal Pydantic params.
        unused = supplied - set(self.model_fields)
        if unused:
            logger.warning(f"Parameters supplied but not used: {', '.join(unused)}")


class UploadLogs(TrainerCallback):
    """
    A callback to upload training logs to the Hugging Face Hub.

    Args:
        config (object): Configuration object containing necessary parameters.

    Attributes:
        config (object): Configuration object containing necessary parameters.
        api (HfApi or None): Instance of HfApi for interacting with the Hugging Face Hub.
        last_upload_time (float): Timestamp of the last upload.

    Methods:
        on_step_end(args, state, control, **kwargs):
            Called at the end of each training step. Uploads logs to the Hugging Face Hub if conditions are met.
    """

    def __init__(self, config):
        self.config = config
        self.api = None
        self.last_upload_time = 0

        if self.config.push_to_hub:
            if PartialState().process_index == 0:
                self.api = HfApi(token=config.token)
                self.api.create_repo(
                    repo_id=f"{self.config.username}/{self.config.project_name}", repo_type="model", private=True
                )

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.config.push_to_hub is False:
            return control

        if not os.path.exists(os.path.join(self.config.project_name, "runs")):
            return control

        if (state.global_step + 1) % self.config.logging_steps == 0 and self.config.log == "tensorboard":
            if PartialState().process_index == 0:
                current_time = time.time()
                if current_time - self.last_upload_time >= 600:
                    try:
                        self.api.upload_folder(
                            folder_path=os.path.join(self.config.project_name, "runs"),
                            repo_id=f"{self.config.username}/{self.config.project_name}",
                            path_in_repo="runs",
                        )
                    except Exception as e:
                        logger.warning(f"Failed to upload logs: {e}")
                        logger.warning("Continuing training...")

                    self.last_upload_time = current_time
        return control


class LossLoggingCallback(TrainerCallback):
    """
    LossLoggingCallback is a custom callback for logging loss during training.

    This callback inherits from `TrainerCallback` and overrides the `on_log` method
    to remove the "total_flos" key from the logs and log the remaining information
    if the current process is the local process zero.

    Methods:
        on_log(args, state, control, logs=None, **kwargs):
            Called when the logs are updated. Removes the "total_flos" key from the logs
            and logs the remaining information if the current process is the local process zero.

    Args:
        args: The training arguments.
        state: The current state of the Trainer.
        control: The control object for the Trainer.
        logs (dict, optional): The logs dictionary containing the training metrics.
        **kwargs: Additional keyword arguments.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(logs)


class TrainStartCallback(TrainerCallback):
    """
    TrainStartCallback is a custom callback for the Trainer class that logs a message when training begins.

    Methods:
        on_train_begin(args, state, control, **kwargs):
            Logs a message indicating that training is starting.

            Args:
                args: The training arguments.
                state: The current state of the Trainer.
                control: The control object for the Trainer.
                **kwargs: Additional keyword arguments.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Starting to train...")
