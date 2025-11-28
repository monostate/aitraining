import json
import os

from autotrain import logger


def detect_model_type(model_path: str) -> str:
    """Auto-detect model type from config or architecture

    Args:
        model_path: Either a local path or a HuggingFace Hub model ID
    """
    # Check if it's a local path or HF Hub model ID
    is_local = os.path.exists(model_path)

    if is_local:
        # Local model - check training_params.json first
        params_path = os.path.join(model_path, "training_params.json")
        if os.path.exists(params_path):
            try:
                with open(params_path) as f:
                    params = json.load(f)
                    if "task" in params:
                        task = params["task"]
                        # Map task names to simplified types if needed
                        if task.startswith("llm"):
                            return "llm"
                        if task.startswith("st:"):
                            return "sentence-transformers"
                        if task.startswith("vlm:"):
                            return "vlm"
                        return task
                    # Fallback to trainer if task is not present (older versions?)
                    if "trainer" in params:
                        trainer = params["trainer"]
                        # Map trainer types to model types for inference
                        if trainer in ["sft", "dpo", "orpo", "ppo", "reward", "distillation"]:
                            return "llm"
                        return trainer
            except Exception as e:
                logger.warning(f"Failed to read training_params.json in {model_path}: {e}")

        # Fallback: check model config.json for local models
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    return _detect_type_from_config(config)
            except Exception as e:
                logger.warning(f"Failed to read config.json in {model_path}: {e}")
    else:
        # HuggingFace Hub model - download config and check
        try:
            from huggingface_hub import hf_hub_download

            # Try to download config.json from HF Hub
            try:
                config_path = hf_hub_download(repo_id=model_path, filename="config.json")
                with open(config_path) as f:
                    config = json.load(f)
                    model_type = _detect_type_from_config(config)
                    if model_type != "unknown":
                        return model_type
            except Exception as e:
                logger.warning(f"Could not download config.json from HF Hub for {model_path}: {e}")

            # Special case: sentence-transformers models
            if "sentence-transformers" in model_path.lower():
                return "sentence-transformers"

        except ImportError:
            logger.warning("huggingface_hub not installed, cannot detect HF Hub model type")

    return "unknown"


def _detect_type_from_config(config: dict) -> str:
    """Detect model type from config.json dictionary"""
    # Check architectures field
    if "architectures" in config:
        arch = config["architectures"][0]
        if "CausalLM" in arch:
            return "llm"
        if "SequenceClassification" in arch:
            return "text-classification"
        if "TokenClassification" in arch:
            return "token-classification"
        if "QuestionAnswering" in arch:
            return "extractive-question-answering"
        if "ImageClassification" in arch:
            return "image-classification"
        if "ConditionalGeneration" in arch or "EncoderDecoder" in arch:
            return "seq2seq"
        if "VisionEncoderDecoder" in arch:
            return "vlm"
        if "ForImageRegression" in arch:
            return "image-regression"

    # Check model_type field as fallback
    if "model_type" in config:
        model_type = config["model_type"]
        if model_type in ["t5", "bart", "mbart", "pegasus"]:
            return "seq2seq"

    return "unknown"


def get_model_metadata(model_path: str) -> dict:
    """Get additional metadata about the model"""
    metadata = {"created_at": None, "size": None, "files": []}

    try:
        # Get creation time and size
        stat = os.stat(model_path)
        metadata["created_at"] = stat.st_ctime

        # Get list of files
        files = os.listdir(model_path)
        metadata["files"] = files

        # Calculate total size
        total_size = 0
        for f in files:
            fp = os.path.join(model_path, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
        metadata["size"] = total_size

    except Exception as e:
        logger.warning(f"Failed to get metadata for {model_path}: {e}")

    return metadata
