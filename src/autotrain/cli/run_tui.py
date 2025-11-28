"""
AITraining TUI - Full-screen terminal user interface for AITraining CLI.
"""

import logging
import os
import sys
from argparse import ArgumentParser


# CRITICAL: Suppress autotrain logs BEFORE importing anything that might trigger them
# This prevents "Project path normalized to:" and other logs from corrupting the TUI
os.environ["AUTOTRAIN_TUI_MODE"] = "1"
logging.getLogger("autotrain").setLevel(logging.CRITICAL)

try:
    from loguru import logger as loguru_logger

    loguru_logger.remove()
except ImportError:
    pass

from autotrain import logger
from autotrain.cli import BaseAutoTrainCommand


class RunAutoTrainTUICommand(BaseAutoTrainCommand):
    """Command to run AITraining's Terminal User Interface (TUI)."""

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """Register the TUI subcommand with the argument parser."""
        run_tui_parser = parser.add_parser(
            "portal",
            description="[EXPERIMENTAL] Launch AITraining's interactive Terminal User Interface (Portal) - Under active development, not recommended for production use",
            help="[EXPERIMENTAL] Launch an interactive Portal for configuring and running AITraining (under active development)",
        )
        run_tui_parser.add_argument(
            "--theme",
            type=str,
            choices=["dark", "light"],
            default="dark",
            help="Color theme for the TUI (default: dark)",
        )
        run_tui_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Enable dry-run mode for testing (commands won't actually execute)",
        )
        run_tui_parser.add_argument(
            "--config",
            type=str,
            help="Load configuration from a JSON or YAML file on startup",
        )
        run_tui_parser.set_defaults(func=run_tui_command_factory)

    def __init__(self, args):
        self.args = args
        self.theme = args.theme
        self.dry_run = args.dry_run
        self.config_file = args.config if hasattr(args, "config") else None

    def run(self):
        """Run the TUI application."""
        # Check if we're in a TTY environment
        if not self._check_tty():
            logger.error("Error: AITraining TUI requires an interactive terminal (TTY).")
            print("\n❌ AITraining TUI requires an interactive terminal (TTY).")
            print("\nThe TUI cannot run in non-interactive environments such as:")
            print("  • CI/CD pipelines")
            print("  • Jupyter notebooks")
            print("  • Non-terminal environments")
            print("\n💡 Alternatives:")
            print("  • Use the standard CLI: aitraining llm --help")
            print("  • Set up your environment with a proper terminal")
            print("\n📖 For more information, see: docs/cli/TUI.md")
            sys.exit(2)

        # Lazy import to avoid loading heavy dependencies unless needed
        try:
            from autotrain.cli.tui.app import AITrainingTUI
        except ImportError as e:
            logger.error(f"Failed to import TUI dependencies: {e}")
            print("\n❌ Failed to load TUI dependencies.")
            print("\nPlease install the required packages:")
            print("  pip install textual rich")
            print("\nOr reinstall autotrain with:")
            print("  pip install -e .")
            sys.exit(1)

        # Show experimental warning and wait for confirmation
        print("\n⚠️  EXPERIMENTAL FEATURE WARNING")
        print("=" * 50)
        print("The AITraining Portal TUI is currently experimental")
        print("and under active development. It may have bugs or")
        print("incomplete features. For production use, please")
        print("use the standard CLI: aitraining llm --help")
        print("=" * 50)

        try:
            response = input("\nPress ENTER to continue or Ctrl+C to exit... ")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)

        # Launch the TUI application
        logger.info(f"Launching AITraining TUI with theme: {self.theme}")
        app = AITrainingTUI(
            theme=self.theme,
            dry_run=self.dry_run,
            config_file=self.config_file,
        )

        try:
            app.run()
        except KeyboardInterrupt:
            logger.info("TUI closed by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"TUI crashed: {e}")
            print(f"\n❌ TUI crashed unexpectedly: {e}")
            print("\nPlease report this issue at:")
            print("  https://github.com/huggingface/autotrain-advanced/issues")
            sys.exit(1)

    def _check_tty(self) -> bool:
        """Check if we're running in a TTY environment."""
        # Check stdin, stdout, and stderr for TTY
        if not sys.stdin.isatty():
            return False
        if not sys.stdout.isatty():
            return False
        # stderr can be redirected, so it's optional

        # Additional check for terminal environment
        term = os.environ.get("TERM", "")
        if term == "dumb" or not term:
            return False

        return True


def run_tui_command_factory(args):
    """Factory function to create a RunAutoTrainTUICommand instance."""
    return RunAutoTrainTUICommand(args)
