import argparse
import sys

from autotrain import __version__, logger
from autotrain.cli.run_api import RunAutoTrainAPICommand
from autotrain.cli.run_app import RunAutoTrainAppCommand
from autotrain.cli.run_chat import RunAutoTrainChatCommand
from autotrain.cli.run_extractive_qa import RunAutoTrainExtractiveQACommand
from autotrain.cli.run_image_classification import RunAutoTrainImageClassificationCommand
from autotrain.cli.run_image_regression import RunAutoTrainImageRegressionCommand
from autotrain.cli.run_llm import RunAutoTrainLLMCommand
from autotrain.cli.run_object_detection import RunAutoTrainObjectDetectionCommand
from autotrain.cli.run_sent_tranformers import RunAutoTrainSentenceTransformersCommand
from autotrain.cli.run_seq2seq import RunAutoTrainSeq2SeqCommand
from autotrain.cli.run_setup import RunSetupCommand
from autotrain.cli.run_spacerunner import RunAutoTrainSpaceRunnerCommand
from autotrain.cli.run_tabular import RunAutoTrainTabularCommand
from autotrain.cli.run_text_classification import RunAutoTrainTextClassificationCommand
from autotrain.cli.run_text_regression import RunAutoTrainTextRegressionCommand
from autotrain.cli.run_token_classification import RunAutoTrainTokenClassificationCommand
from autotrain.cli.run_tools import RunAutoTrainToolsCommand
from autotrain.cli.run_tui import RunAutoTrainTUICommand
from autotrain.cli.run_vlm import RunAutoTrainVLMCommand
from autotrain.parser import AutoTrainConfigParser


ASCII_BANNER = r"""
 __  __  ___  _  _  ___  ___ _____ ___ _____ ___
|  \/  |/ _ \| \| |/ _ \/ __|_   _/_  |_   _| __|
| |\/| | (_) | .` | (_) \__ \ | | / _ \ | | | _|
|_|  |_|\___/|_|\_|\___/|___/ |_|/_/ \_\|_| |___|

           Monostate AI Models Development
"""

WELCOME_MESSAGE = """
Welcome to AITraining! Get started with:

  • aitraining llm --help          Show all LLM training options
  • aitraining portal              [EXPERIMENTAL] Launch interactive configuration UI
  • aitraining llm --train         Start training with your config

Quick Examples:

  # Train with grouped parameters
  aitraining llm --trainer sft --help

  # Train with specific trainer
  aitraining llm --train --model google/gemma-3-270m --data-path ./data --trainer sft

  # Fine-tune with LoRA
  aitraining llm --train --model meta-llama/Llama-2-7b --peft --lora-r 16

For detailed documentation, visit: https://github.com/huggingface/autotrain-advanced
"""


def main():
    parser = argparse.ArgumentParser(
        "AITraining advanced CLI",
        usage="aitraining <command> [<args>]",
        epilog="For more information about a command, run: `aitraining <command> --help`",
    )
    parser.add_argument("--version", "-v", help="Display AITraining version", action="store_true")
    parser.add_argument("--config", help="Optional configuration file", type=str)
    commands_parser = parser.add_subparsers(help="commands")

    # Register commands
    RunAutoTrainAppCommand.register_subcommand(commands_parser)
    RunAutoTrainChatCommand.register_subcommand(commands_parser)
    RunAutoTrainTUICommand.register_subcommand(commands_parser)
    RunAutoTrainLLMCommand.register_subcommand(commands_parser)
    RunSetupCommand.register_subcommand(commands_parser)
    RunAutoTrainAPICommand.register_subcommand(commands_parser)
    RunAutoTrainTextClassificationCommand.register_subcommand(commands_parser)
    RunAutoTrainImageClassificationCommand.register_subcommand(commands_parser)
    RunAutoTrainTabularCommand.register_subcommand(commands_parser)
    RunAutoTrainSpaceRunnerCommand.register_subcommand(commands_parser)
    RunAutoTrainSeq2SeqCommand.register_subcommand(commands_parser)
    RunAutoTrainTokenClassificationCommand.register_subcommand(commands_parser)
    RunAutoTrainToolsCommand.register_subcommand(commands_parser)
    RunAutoTrainTextRegressionCommand.register_subcommand(commands_parser)
    RunAutoTrainObjectDetectionCommand.register_subcommand(commands_parser)
    RunAutoTrainSentenceTransformersCommand.register_subcommand(commands_parser)
    RunAutoTrainImageRegressionCommand.register_subcommand(commands_parser)
    RunAutoTrainExtractiveQACommand.register_subcommand(commands_parser)
    RunAutoTrainVLMCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if args.version:
        print(__version__)
        exit(0)

    if args.config:
        logger.info(f"Using AITraining configuration: {args.config}")
        cp = AutoTrainConfigParser(args.config)
        cp.run()
        exit(0)

    if not hasattr(args, "func"):
        # Show ASCII banner and welcome message when no command is provided
        if sys.stdout.isatty():  # Only show in terminal, not when piped
            try:
                from rich.console import Console

                console = Console()
                console.print(ASCII_BANNER, style="orange1")
            except ImportError:
                print(ASCII_BANNER)

            # Launch interactive wizard (will prompt for trainer type)
            from autotrain.cli.interactive_wizard import run_wizard
            from autotrain.cli.trainer_metadata import TRAINER_METADATA
            from autotrain.project import AutoTrainProject
            from autotrain.trainers.clm.params import LLMTrainingParams

            try:
                # Run wizard to collect configuration (no trainer_type specified, will prompt)
                config = run_wizard()

                # Set backend and train flag
                config["backend"] = config.get("backend", "local")

                # Determine which params class to use based on trainer type in config
                # The wizard adds _trainer_type to the returned config
                trainer_type = config.pop("_trainer_type", None)

                if trainer_type == "llm" or (trainer_type is None and "trainer" in config):
                    # LLM trainer
                    params_class = LLMTrainingParams
                elif trainer_type:
                    # Use metadata for non-LLM trainers
                    metadata = TRAINER_METADATA.get(trainer_type)
                    if metadata:
                        params_class = metadata["params_class"]
                    else:
                        # Fallback to LLM if metadata not found
                        logger.warning(f"Unknown trainer type '{trainer_type}', defaulting to LLM")
                        params_class = LLMTrainingParams
                else:
                    # Default to LLM if we can't determine
                    params_class = LLMTrainingParams

                # Create training params
                logger.info(f"Config keys: {list(config.keys())}")
                logger.info(f"auto_convert_dataset in config: {config.get('auto_convert_dataset', 'NOT FOUND')}")
                logger.info(f"trainer in config: {config.get('trainer', 'NOT FOUND')}")
                params = params_class(**config)
                logger.info(
                    f"params.auto_convert_dataset after instantiation: {getattr(params, 'auto_convert_dataset', 'NOT FOUND')}"
                )

                # Create and launch project
                print("\n🚀 Starting training...")
                project = AutoTrainProject(params=params, backend=config["backend"], process=True)
                job_id = project.create()
                logger.info(f"Job ID: {job_id}")
                print(f"\n✓ Training job started! Job ID: {job_id}")

            except KeyboardInterrupt:
                print("\n\n❌ Setup cancelled.")
                exit(1)
            except Exception as e:
                logger.error(f"Error during setup: {e}")
                print(f"\n❌ Error: {e}")
                exit(1)
        else:
            # When piped, just show help
            parser.print_help()
        exit(0)

    command = args.func(args)
    command.run()


if __name__ == "__main__":
    main()
