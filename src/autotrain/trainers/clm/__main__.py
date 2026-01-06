import argparse
import json

from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.common import monitor


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


@monitor
def train(config):
    """
    Main training dispatcher for CLM trainers.

    Note: Distillation is now controlled by the use_distillation flag in params.py
    If use_distillation=True, the distillation logic wraps the base trainer.
    This is handled internally by checking config.use_distillation.
    """
    if isinstance(config, dict):
        config = LLMTrainingParams(**config)

    # Set W&B run ID env var if provided (enables resuming existing run)
    if getattr(config, "wandb_run_id", None):
        import os

        from autotrain import logger

        os.environ["WANDB_RUN_ID"] = config.wandb_run_id
        os.environ["WANDB_RESUME"] = "allow"
        logger.info(f"Resuming W&B run: {config.wandb_run_id}")

    if config.trainer == "default":
        from autotrain.trainers.clm.train_clm_default import train as train_default

        train_default(config)

    elif config.trainer == "sft":
        from autotrain.trainers.clm.train_clm_sft import train as train_sft

        train_sft(config)

    elif config.trainer == "reward":
        from autotrain.trainers.clm.train_clm_reward import train as train_reward

        train_reward(config)

    elif config.trainer == "dpo":
        from autotrain.trainers.clm.train_clm_dpo import train as train_dpo

        train_dpo(config)

    elif config.trainer == "orpo":
        from autotrain.trainers.clm.train_clm_orpo import train as train_orpo

        train_orpo(config)

    elif config.trainer == "ppo":
        from autotrain.trainers.clm.train_clm_ppo import train as train_ppo

        train_ppo(config)

    elif config.trainer == "distillation":
        from autotrain import logger
        from autotrain.trainers.clm import train_clm_distill

        logger.info("Starting Prompt Distillation training...")
        logger.info("Note: This is different from inline distillation (--use-distillation)")
        logger.info("Prompt distillation internalizes complex prompts into the model.")
        train_clm_distill.train(config)

    else:
        raise ValueError(f"trainer `{config.trainer}` not supported")


if __name__ == "__main__":
    # Check if MPS should be disabled before importing torch-dependent modules
    import os

    if os.environ.get("AUTOTRAIN_DISABLE_MPS") == "1":
        # Monkey-patch torch to disable MPS detection
        import torch

        original_is_available = torch.backends.mps.is_available
        torch.backends.mps.is_available = lambda: False
        # Also prevent MPS from being used accidentally
        if hasattr(torch.backends.mps, "is_built"):
            torch.backends.mps.is_built = lambda: False

    _args = parse_args()
    training_config = json.load(open(_args.training_config))
    _config = LLMTrainingParams(**training_config)
    train(_config)
