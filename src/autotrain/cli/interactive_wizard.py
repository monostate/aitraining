"""
Interactive CLI wizard for AITraining.

This module provides a modern, user-friendly wizard interface for configuring
and launching training jobs for all trainer types through the command line.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from autotrain import logger
from autotrain.cli.run_llm import FIELD_GROUPS as LLM_FIELD_GROUPS
from autotrain.cli.run_llm import FIELD_SCOPES as LLM_FIELD_SCOPES
from autotrain.cli.trainer_metadata import TRAINER_METADATA, get_trainer_display_name
from autotrain.cli.utils import get_field_info
from autotrain.metadata.catalog import CatalogEntry, format_params, get_popular_datasets, get_popular_models
from autotrain.trainers.clm.params import LLMTrainingParams


ASCII_BANNER = r"""
 █████╗ ██╗████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗     
██╔══██╗██║╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝     
███████║██║   ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗    
██╔══██║██║   ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║    
██║  ██║██║   ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝    
╚═╝  ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝     
        From zero to hero Machine Learning Training Platform
"""

# Placeholder values that indicate the user has not provided real input yet
PLACEHOLDER_PROJECT_NAME = {"project-name", ""}
PLACEHOLDER_DATA_PATH = {"data", ""}
PLACEHOLDER_MODEL = {"google/gemma-3-270m", ""}

# Group ordering for wizard flow
GROUP_ORDER = [
    "Basic",
    "Data Processing",
    "Training Configuration",
    "Training Hyperparameters",
    "PEFT/LoRA",
    "DPO/ORPO",
    "Hub Integration",
    "Knowledge Distillation",
    "Hyperparameter Sweep",
    "Enhanced Evaluation",
    "Reinforcement Learning (PPO)",
    "Advanced Features",
]


class WizardBackRequest(Exception):
    """Raised when the user wants to revisit the previous step."""


class WizardExitRequest(Exception):
    """Raised when the user cancels the wizard."""


class InteractiveWizard:
    """Interactive wizard for AITraining CLI."""

    def __init__(self, initial_args: Optional[Dict[str, Any]] = None, trainer_type: Optional[str] = None):
        """
        Initialize the wizard.

        Args:
            initial_args: Initial arguments from CLI (if any)
            trainer_type: Trainer type (llm, text-classification, etc.). If None, will prompt user.
        """
        self.answers: Dict[str, Any] = initial_args or {}
        self.trainer_type = trainer_type

        # For LLM, trainer is the sub-type (sft, dpo, etc.)
        # For other trainers, trainer_type is the main type
        if trainer_type == "llm":
            # If trainer is not in answers or is None, default to "sft"
            self.trainer = self.answers.get("trainer") or "sft"
        else:
            self.trainer = None
        self._current_step: Optional[str] = None
        self._is_revisit: bool = False

    def run(self) -> Dict[str, Any]:
        """
        Run the interactive wizard and return collected configuration.

        Returns:
            Dictionary of configuration parameters
        """
        self._show_banner()
        self._show_welcome()
        self._print_controls()
        self._ensure_hf_token()

        steps = self._build_steps()
        self._execute_steps(steps)

        if self.trainer_type:
            self.answers["_trainer_type"] = self.trainer_type

        return self.answers

    def _build_steps(self) -> List[Tuple[str, callable]]:
        """Construct the ordered list of steps based on trainer selection."""
        steps: List[Tuple[str, callable]] = []
        if self.trainer_type is None:
            steps.append(("trainer-type", self._prompt_trainer_type))
        if self.trainer_type == "llm" and "trainer" not in self.answers:
            steps.append(("llm-trainer", self._prompt_trainer))

        steps.extend(
            [
                ("basic", self._prompt_basic_config),
                ("model", self._prompt_model_selection),  # Model BEFORE dataset for chat template selection
                ("dataset", self._prompt_dataset_config),
                ("advanced", self._prompt_advanced_params),
                ("summary", self._final_confirmation),
            ]
        )
        return steps

    def _execute_steps(self, steps: List[Tuple[str, callable]]) -> None:
        """Iterate through the configured steps, honoring :back and :exit commands."""
        index = 0
        visit_counts = [0] * len(steps)

        while index < len(steps):
            step_name, step_callable = steps[index]
            self._current_step = step_name
            self._is_revisit = visit_counts[index] > 0
            try:
                step_callable()
                visit_counts[index] += 1

                # Special case: if we just selected trainer type as "llm",
                # we need to add the trainer selection step
                if step_name == "trainer-type" and self.trainer_type == "llm" and "trainer" not in self.answers:
                    # Insert llm-trainer step right after current position
                    steps.insert(index + 1, ("llm-trainer", self._prompt_trainer))
                    visit_counts.insert(index + 1, 0)

                index += 1
            except WizardBackRequest:
                if index == 0:
                    print("↩️  Already at the first step. Cannot go back further.")
                else:
                    index -= 1
            except WizardExitRequest:
                print("\n❌ Configuration cancelled.")
                sys.exit(0)

    def _final_confirmation(self) -> None:
        """Show summary and require confirmation."""
        if not self._show_summary_and_confirm():
            raise WizardExitRequest()

    def _ensure_hf_token(self):
        """Ensure we have a Hugging Face token early in the flow."""
        # Check multiple sources for HF token
        existing = self.answers.get("token") or os.environ.get("HF_TOKEN")

        # If not found, try to get from huggingface_hub (checks system-wide CLI auth)
        if not existing:
            try:
                from huggingface_hub import get_token

                existing = get_token()
            except Exception:
                pass

        if existing:
            self.answers["token"] = existing
            os.environ["HF_TOKEN"] = existing
            print("✓ Hugging Face token detected from previous input/environment.")
            return

        print("\nBefore we begin, enter a Hugging Face token so we can list private models/datasets.")
        print("Press Enter to continue without one (public resources only).")
        try:
            help_text = """This is like a password for Hugging Face (optional).

Why you might want one:
• Use private models (like Llama 3 or other gated models)
• Access your own private datasets
• Save your trained model online automatically

How to get one (takes 1 minute):
1. Go to huggingface.co and sign up (free!)
2. Click your profile → Settings → Access Tokens
3. Click "New token" → give it a name → Create
4. Copy the token (looks like: hf_abcdefghijk...)

Don't have one? No worries!
Just press Enter to skip - you can still train models with public data."""

            token = self._get_input("Hugging Face token (hf_...)", "", help_text=help_text).strip()
        except WizardExitRequest:
            print("\n❌ Configuration cancelled.")
            sys.exit(0)
        if token:
            self.answers["token"] = token
            os.environ["HF_TOKEN"] = token
            print("✓ Hugging Face token saved for this session.")
        else:
            print("⚠️  Continuing without a Hugging Face token. Only public models/datasets will be available.")

    def _show_banner(self):
        """Display ASCII banner."""
        if sys.stdout.isatty():
            try:
                from rich.console import Console

                console = Console()
                console.print(ASCII_BANNER, style="orange1")
            except ImportError:
                print(ASCII_BANNER)

    def _show_welcome(self):
        """Display welcome message."""
        print("\n🚀 Welcome to the AITraining Interactive Setup!\n")
        if self.trainer_type == "llm":
            print("I'll help you configure your LLM training job step by step.")
        elif self.trainer_type:
            display_name = get_trainer_display_name(self.trainer_type)
            print(f"I'll help you configure your {display_name} training job step by step.")
        else:
            print("I'll help you configure your training job step by step.")
        print("Press Enter to accept defaults shown in [brackets].\n")

    def _print_controls(self):
        """Explain global commands available during the wizard."""
        print("Commands available at any prompt:")
        print("  :back  → return to the previous step")
        print("  :help  → show extra information for the current prompt")
        print("  :exit  → cancel setup\n")

    def _prompt_trainer_type(self):
        """Prompt user to select trainer type."""
        print("\n" + "=" * 60)
        print("📋 Step 0: Choose Trainer Type")
        print("=" * 60)

        # Build trainer options
        trainers = {
            "1": ("llm", "Large Language Models (LLM) - text generation, chat, instruction following"),
            "2": ("text-classification", "Text Classification - categorize text into labels"),
            "3": ("token-classification", "Token Classification - NER, POS tagging"),
            "4": ("tabular", "Tabular Data - classification or regression on structured data"),
            "5": ("image-classification", "Image Classification - categorize images"),
            "6": ("image-regression", "Image Regression - predict values from images"),
            "7": ("seq2seq", "Sequence-to-Sequence - translation, summarization"),
            "8": ("extractive-qa", "Extractive QA - answer questions from context"),
            "9": ("sent-transformers", "Sentence Transformers - semantic similarity embeddings"),
            "10": ("vlm", "Vision-Language Models - multimodal tasks"),
        }

        print("\nAvailable trainer types:")
        for key, (name, desc) in trainers.items():
            print(f"  {key:2}. {desc}")

        help_text = """What kind of AI are you trying to build?

1. LLM - Make a chatbot or text generator
   • What it does: Generates human-like text, answers questions, follows instructions
   • Real example: Like ChatGPT, Claude, or coding assistants
   • You need: Text conversations or documents to learn from

2. Text Classification - Sort text into categories
   • What it does: Automatically labels text (is this email spam? is this review positive?)
   • Real example: Gmail's spam filter, Amazon review ratings
   • You need: Text examples with their correct labels

3. Token Classification - Label each word in a sentence
   • What it does: Identifies names, places, dates in text
   • Real example: Highlighting people's names in documents, finding addresses
   • You need: Text where each word is labeled

4. Tabular - Work with spreadsheet data
   • What it does: Makes predictions from rows and columns of data
   • Real example: Predict house prices, customer churn, loan approval
   • You need: A CSV or Excel file with historical data

5. Image Classification - Sort images into categories
   • What it does: Identifies what's in a picture
   • Real example: Is this a cat or dog? Is this product defective?
   • You need: Labeled images (folder structure or dataset)

6. Image Regression - Predict numbers from images
   • What it does: Estimates a value by looking at an image
   • Real example: Guess someone's age from photo, estimate damage cost
   • You need: Images with numerical values

7. Seq2Seq - Transform text to different text
   • What it does: Rewrites text in a different way
   • Real example: Google Translate, summarizing articles, paraphrasing
   • You need: Pairs of input/output text

8. Extractive QA - Answer questions from documents
   • What it does: Finds answers in a given text
   • Real example: "What's the return policy?" → finds answer in manual
   • You need: Questions, contexts, and where answers are located

9. Sentence Transformers - Find similar text
   • What it does: Understands if two sentences mean the same thing
   • Real example: "Find similar questions" in FAQ systems
   • You need: Pairs of similar/different sentences

10. VLM - Understand images AND text together
    • What it does: Describes images, answers questions about pictures
    • Real example: "What's in this image?", reading screenshots
    • You need: Images with descriptions or Q&A pairs

Pick the one that matches what you're trying to build!"""

        while True:
            choice = self._get_input("\nSelect trainer type [1-10, default: 1]", "1", help_text=help_text).strip()
            if choice in trainers:
                self.trainer_type = trainers[choice][0]
                display_name = get_trainer_display_name(self.trainer_type)
                print(f"✓ Selected: {display_name}")

                # Initialize trainer for LLM type if not already set
                if self.trainer_type == "llm" and not self.trainer:
                    self.trainer = self.answers.get("trainer") or "sft"
                    logger.debug(f"Initialized trainer to: {self.trainer}")

                break
            print("❌ Invalid choice. Please enter 1-10.")

    def _prompt_trainer(self):
        """Prompt user to select trainer type."""
        print("\n" + "=" * 60)
        print("📋 Step 1: Choose Training Type")
        print("=" * 60)

        trainers = {
            "1": ("sft", "Supervised Fine-Tuning (most common)"),
            "2": ("dpo", "Direct Preference Optimization"),
            "3": ("orpo", "Odds Ratio Preference Optimization"),
            "4": ("ppo", "Proximal Policy Optimization (RL)"),
            "5": ("default", "Generic training"),
        }

        print("\nAvailable trainers:")
        for key, (name, desc) in trainers.items():
            print(f"  {key}. {name:15} - {desc}")

        help_text = """How do you want to train your language model?

1. SFT - Teach it to follow instructions (most people want this!)
   • What it does: Makes the model follow commands and answer questions
   • Example: "Write me a poem" → model writes a poem
   • You need: Examples of prompts and good responses

2. DPO - Teach it what responses are better
   • What it does: Learns from good vs bad examples
   • Example: Show it a helpful answer vs unhelpful answer
   • You need: Same prompt with "good" and "bad" responses

3. ORPO - Like DPO but needs less data
   • What it does: Same as DPO but works with fewer examples
   • Example: Teaching preferences with limited data
   • You need: Some good/bad response pairs (fewer than DPO)

4. PPO - Train with rewards (advanced!)
   • What it does: Gets better using a scoring system
   • Example: Like training a dog with treats for good behavior
   • You need: A way to score responses (reward model)

5. Generic - Just continue training on text
   • What it does: Keeps learning from raw text
   • Example: Make model better at specific topics/domains
   • You need: Lots of text in your domain

🎯 Not sure? Pick option 1 (SFT) - it's what most people need!"""

        while True:
            choice = self._get_input("\nSelect trainer [1-5, default: 1]", "1", help_text=help_text).strip()
            # If empty, use default
            if not choice:
                choice = "1"
            if choice in trainers:
                self.trainer = trainers[choice][0]
                self.answers["trainer"] = self.trainer
                print(f"✓ Selected: {self.trainer}")
                logger.debug(f"Set trainer to: {self.trainer}")
                break
            print("❌ Invalid choice. Please enter 1-5.")

    def _prompt_basic_config(self):
        """Prompt for basic configuration."""
        print("\n" + "=" * 60)
        print("📋 Step 2: Basic Configuration")
        print("=" * 60)

        if self._should_prompt_field("project_name", PLACEHOLDER_PROJECT_NAME):
            default_name = self.answers.get("project_name", "my-llm-project")
            default_name = default_name or "my-llm-project"

            while True:
                help_text = """Your project's name (like naming a folder on your computer).

What happens with this name:
• Creates a folder to save your trained model
• Stores training progress and checkpoints
• Used to identify your training run later

Tips:
• Keep it short and descriptive (e.g., 'chatbot-v1', 'sentiment-analyzer')
• No spaces - use dashes or underscores
• If the folder exists, we'll suggest a new version automatically

Example: If training a customer support bot, use 'support-bot-v1'"""

                project_name = self._get_input(
                    f"\nProject name [{default_name}]",
                    default_name,
                    help_text=help_text,
                ).strip()

                if os.path.exists(project_name) and os.listdir(project_name):
                    print(f"⚠️  Directory '{project_name}' already exists and is not empty.")

                    # Suggest next available version
                    base_name = project_name
                    counter = 2
                    while os.path.exists(f"{base_name}-v{counter}"):
                        counter += 1
                    suggestion = f"{base_name}-v{counter}"

                    print(f"   Suggestion: Use '{suggestion}' instead?")
                    use_suggestion = self._get_yes_no(f"Use '{suggestion}'? [Y/n]", default=True)
                    if use_suggestion:
                        project_name = suggestion
                        break

                    use_anyway = self._get_yes_no(
                        "Use original name anyway? (May overwrite files) [y/N]", default=False
                    )
                    if use_anyway:
                        break
                else:
                    break

            self.answers["project_name"] = project_name
            print(f"✓ Project: {project_name}")
        else:
            print(f"✓ Project: {self.answers['project_name']}")

    def _prompt_dataset_config(self):
        """Prompt for dataset configuration."""
        print("\n" + "=" * 60)
        print("📋 Step 4: Dataset Configuration")
        print("=" * 60)

        default_path = self.answers.get("data_path", "data")
        if default_path in PLACEHOLDER_DATA_PATH:
            default_path = ""

        if "data_path" not in self.answers:
            # Interactive selection
            print("\nDataset options:")
            print("  • Local folder with CSV/JSON/Parquet files (e.g., ./data/my_dataset)")
            print("  • HuggingFace dataset ID (e.g., tatsu-lab/alpaca)")
            print("  • Choose from popular datasets below")

            selected_dataset = self._run_catalog_interaction_loop("datasets", get_popular_datasets)
            if selected_dataset:
                self.answers["data_path"] = selected_dataset
                print(f"✓ Dataset: {selected_dataset}")

                # Attempt validation
                self._validate_dataset(selected_dataset)

        # Train/valid splits
        if self._should_prompt_field("train_split"):
            # Use detected splits if available
            detected_splits = getattr(self, "_detected_splits", [])

            if detected_splits and len(detected_splits) == 1:
                # Auto-fill if only one split available
                train_split = detected_splits[0]
                print(f"\n✓ Using split: {train_split} (only split available)")
                self.answers["train_split"] = train_split
            elif detected_splits and "train" in detected_splits:
                # Auto-select "train" if it exists
                train_split = "train"
                print(f"\n✓ Using split: train (auto-selected from: {', '.join(detected_splits)})")
                self.answers["train_split"] = train_split
            else:
                # Prompt user
                default_train = (
                    self.answers.get("train_split", detected_splits[0] if detected_splits else "train") or "train"
                )
                print("\nTraining split name:")
                if detected_splits:
                    print(f"  Available splits: {', '.join(detected_splits)}")
                    print("  Enter the name of the split to use for training.")
                else:
                    print("  This is the NAME of the split in your dataset (like 'train', 'training').")
                    print("  NOT a percentage. HuggingFace datasets come with predefined splits.")

                help_text = """Dataset splits are named subsets of your data.

Common split names:
  • 'train' - Training data (most common)
  • 'test' - Test/evaluation data
  • 'validation' or 'valid' - Validation data
  • Custom names like 'train_v2', 'sample_100k', etc.

Examples:
  • HuggingFace datasets: Usually 'train', 'test', 'validation'
  • Local CSV/JSON files: Default is 'train'
  • Some datasets have unique names like 'train_sft' or 'train[:1000]' for subsets

Note: This is NOT asking for a percentage split (like 80/20).
It's asking for the exact name of the split in your dataset."""

                train_split = self._get_input(
                    f"Training split name [{default_train}]",
                    default_train,
                    help_text=help_text,
                ).strip()
                self.answers["train_split"] = train_split

        if self._should_prompt_field("valid_split"):
            detected_splits = getattr(self, "_detected_splits", [])
            validation_splits = [s for s in detected_splits if s in ["validation", "valid", "val", "dev", "test"]]

            if validation_splits:
                # Auto-suggest if validation split detected
                default_valid = validation_splits[0]
                print(f"\nValidation split detected: {default_valid}")
            else:
                default_valid = self.answers.get("valid_split", "")
                print("\nValidation split name:")

            print("  Optional. Leave blank if you don't want to use validation.")

            help_text = """Validation split for monitoring training progress.

Common validation split names:
  • 'validation' or 'valid' - Standard validation split
  • 'test' - Test data (sometimes used for validation during training)
  • 'dev' - Development/validation set
  • 'eval' - Evaluation set
  • Custom names like 'valid_clean', 'test_unseen', etc.

Benefits of using validation:
  • Monitor for overfitting during training
  • Automatic early stopping if performance degrades
  • Better model selection based on validation metrics

Leave empty if:
  • Your dataset doesn't have a validation split
  • You want to use all data for training
  • You plan to do manual evaluation later"""

            valid_split = self._get_input(
                f"Validation split name (optional) [{default_valid if default_valid else 'none'}]",
                default_valid,
                help_text=help_text,
            ).strip()
            if valid_split and valid_split.lower() not in ["none", "no"]:
                self.answers["valid_split"] = valid_split
            else:
                # Explicitly set to None if user doesn't want validation
                self.answers["valid_split"] = None

        # Max samples for testing/debugging
        if self._should_prompt_field("max_samples"):
            default_max_samples = self.answers.get("max_samples", "")
            max_samples_input = self._get_input(
                "Maximum samples (optional, for testing/debugging)",
                default_max_samples,
                help_text="Limit dataset to N samples for faster training/testing. Leave blank to use full dataset.",
            ).strip()
            if max_samples_input:
                try:
                    max_samples = int(max_samples_input)
                    if max_samples > 0:
                        self.answers["max_samples"] = max_samples
                except ValueError:
                    logger.warning(f"Invalid max_samples value '{max_samples_input}', ignoring")

        # Offer dataset conversion for LLM training
        logger.debug(f"Checking dataset conversion: trainer_type={self.trainer_type}, trainer={self.trainer}")
        if self.trainer_type == "llm" and self.trainer in ["sft", "dpo", "orpo", "reward"]:
            logger.debug("Calling _prompt_dataset_conversion()")
            self._prompt_dataset_conversion()
        else:
            logger.debug(f"Skipping dataset conversion: trainer_type={self.trainer_type}, trainer={self.trainer}")

        # Column mapping for trainer-specific needs
        # Skip column mapping if auto-conversion is enabled (it handles mapping automatically)
        if not self.answers.get("auto_convert_dataset", False):
            self._prompt_column_mapping()
        else:
            logger.debug("Skipping column mapping prompt since auto_convert_dataset is enabled")
            # When auto-convert is enabled, text_column will be set by the conversion process
            # We don't need to set it here

    def _validate_dataset(self, path: str):
        """Attempt to validate the dataset and print columns."""
        print("🔍 Validating dataset...")
        try:
            import pandas as pd

            df = None
            splits_detected = []

            # 1. Try local folder
            if os.path.exists(path):
                # Look for common files
                for ext in [".csv", ".json", ".jsonl", ".parquet"]:
                    files = [f for f in os.listdir(path) if f.endswith(ext)]
                    if files:
                        file_path = os.path.join(path, files[0])
                        if ext == ".csv":
                            df = pd.read_csv(file_path, nrows=5)
                        elif ext in [".json", ".jsonl"]:
                            df = pd.read_json(file_path, lines=True, nrows=5)
                        elif ext == ".parquet":
                            df = pd.read_parquet(file_path)
                            df = df.head(5)
                        # For local files, we typically have just "train" split
                        splits_detected = ["train"]
                        break

            # 2. Try Hugging Face Hub (if not found locally or path is not a dir)
            if df is None:
                from datasets import get_dataset_split_names, load_dataset

                try:
                    # First try to get available splits
                    splits_detected = get_dataset_split_names(path)
                except:
                    splits_detected = ["train"]  # Default fallback

                # Use streaming to avoid downloading huge datasets
                split_to_use = self.answers.get("train_split", splits_detected[0] if splits_detected else "train")
                ds = load_dataset(path, split=split_to_use, streaming=True, trust_remote_code=True)
                # Take 5 items
                samples = list(ds.take(5))
                if samples:
                    df = pd.DataFrame(samples)

            if df is not None:
                # Store detected splits for later use
                self._detected_splits = splits_detected

                # Display results
                if splits_detected:
                    print(f"✓ Dataset loaded. Splits found: {', '.join(splits_detected)}")
                print(f"✓ Dataset loaded. Columns found: {', '.join(df.columns)}")
                self._detected_columns = list(df.columns)

                # Basic checks for LLM
                if self.trainer_type == "llm" and "text" not in df.columns:
                    print("⚠️  No 'text' column found. You will need to specify the column mapping.")
            else:
                print("⚠️  Could not load dataset preview (might be empty or custom format).")

        except Exception as e:
            # Don't fail the wizard, just warn
            logger.debug(f"Dataset validation failed: {e}")
            print(f"⚠️  Could not validate dataset: {str(e)}")

    def _should_prompt_field(self, field_name: str, placeholders: Optional[set] = None) -> bool:
        """Determine if we should ask for a field again."""
        if self._is_revisit:
            return True
        if field_name not in self.answers:
            return True
        if placeholders and self.answers.get(field_name) in placeholders:
            return True
        return False

    def _print_catalog(self, label: str, entries: List[CatalogEntry]) -> None:
        """Pretty-print catalog entries."""
        if not entries:
            return
        print(f"\nPopular {label}:")
        for idx, entry in enumerate(entries, 1):
            text = f"  {idx}. {entry.label}"
            if entry.description:
                text += f" — {entry.description}"
            if entry.id != entry.label:
                text += f" ({entry.id})"
            print(text)
        print("  (Type the number to autofill, or provide your own value.)")

    def _resolve_catalog_choice(self, choice: str, entries: List[CatalogEntry]) -> Optional[str]:
        """Return the catalog selection if numeric choice is valid."""
        if not entries or not choice.isdigit():
            return None
        idx = int(choice)
        if 1 <= idx <= len(entries):
            return entries[idx - 1].id
        return None

    def _prompt_dataset_conversion(self):
        """Prompt user about dataset format conversion for LLM training."""
        print("\n🔄 Dataset Format Analysis:")

        # Try to analyze the dataset format
        try:
            import pandas as pd
            from datasets import load_dataset

            from ..preprocessor.llm import detect_dataset_format, get_available_chat_templates

            # Load a sample of the dataset to analyze format
            data_path = self.answers.get("data_path", "")
            train_split = self.answers.get("train_split", "train")

            if not data_path:
                logger.warning("No data_path found in answers, skipping dataset conversion")
                print("⚠️  No dataset path found, skipping format analysis")
                return

            # Load a small sample
            sample_dataset = None
            try:
                if os.path.exists(data_path):
                    # Local dataset
                    logger.debug(f"Loading local dataset from: {data_path}")
                    for ext in [".csv", ".json", ".jsonl", ".parquet"]:
                        files = [f for f in os.listdir(data_path) if f.endswith(ext)]
                        if files:
                            file_path = os.path.join(data_path, files[0])
                            if ext == ".csv":
                                df = pd.read_csv(file_path, nrows=100)
                            elif ext in [".json", ".jsonl"]:
                                df = pd.read_json(file_path, lines=True, nrows=100)
                            elif ext == ".parquet":
                                df = pd.read_parquet(file_path)
                                df = df.head(100)
                            sample_dataset = df
                            break
                else:
                    # HuggingFace dataset
                    logger.debug(f"Loading HuggingFace dataset: {data_path}, split: {train_split}")
                    print(f"  Loading dataset sample from HuggingFace: {data_path}")
                    ds = load_dataset(data_path, split=f"{train_split}[:100]", trust_remote_code=True)
                    sample_dataset = ds
                    logger.debug(
                        f"Loaded dataset with columns: {ds.column_names if hasattr(ds, 'column_names') else 'unknown'}"
                    )
            except Exception as e:
                logger.warning(f"Could not load dataset for format analysis: {e}")
                print(f"⚠️  Could not load dataset sample: {e}")
                print("  Skipping automatic format detection")
                return

            if sample_dataset is None:
                logger.warning("Sample dataset is None after loading attempt")
                print("⚠️  Could not load dataset sample")
                return

            # Detect the format
            logger.debug(f"Detecting dataset format with trainer type: {self.trainer}")
            detected_format = detect_dataset_format(sample_dataset, trainer_type=self.trainer)
            print(f"✓ Detected dataset format: {detected_format}")
            logger.debug(f"Detected format: {detected_format}")

            # Get dataset columns for potential manual mapping
            if hasattr(sample_dataset, "column_names"):
                dataset_columns = sample_dataset.column_names
            elif hasattr(sample_dataset, "columns"):
                dataset_columns = list(sample_dataset.columns)
            else:
                dataset_columns = []

            # Determine if conversion is needed
            needs_conversion = False
            if self.trainer in ["sft", "reward"]:
                if detected_format in ["sharegpt", "alpaca", "qa"]:
                    needs_conversion = True
                    print(f"  • Your dataset is in {detected_format} format")
                    print("  • This can be converted to the standard messages format for better compatibility")
                elif detected_format == "unknown" and dataset_columns:
                    needs_conversion = True
                    print(f"  • Could not auto-detect format. Dataset columns: {dataset_columns}")
                    print("  • You can manually specify which columns contain the user/assistant messages")
            elif self.trainer in ["dpo", "orpo"]:
                if detected_format == "dpo":
                    print("  • Your dataset is already in DPO format (prompt/chosen/rejected)")
                    needs_conversion = False
                elif detected_format in ["sharegpt", "alpaca", "messages"]:
                    print(f"  • Your dataset is in {detected_format} format")
                    print("  • Note: For DPO/ORPO, you'll need chosen/rejected pairs")
                    needs_conversion = True

            if detected_format == "messages":
                print("  • Your dataset is already in the standard messages format")
                needs_conversion = False
            elif detected_format == "plain_text":
                print("  • Your dataset contains plain text - no conversion needed")
                needs_conversion = False
            elif detected_format == "unknown":
                print("  • Dataset format is unknown - manual column mapping may be needed")
                needs_conversion = False

            # Ask user if they want to convert
            if needs_conversion:
                help_text = """Dataset format conversion helps ensure your model trains correctly.

Benefits of conversion:
  • Standardizes your data to messages format (role/content pairs)
  • Applies the correct chat template for your model
  • Prevents training on incorrectly formatted data
  • Follows Unsloth's recommended best practices

Formats we can convert:
  • ShareGPT (from/value) → Messages (role/content)
  • Alpaca (instruction/output) → Messages format
  • Can apply chat templates: llama3, gemma3, qwen2.5, chatml, etc.

Note for ShareGPT: We can either:
  1. Convert to messages format (recommended for consistency)
  2. Use Unsloth's mapping feature (faster, preserves original format)

This follows Unsloth's approach: normalize data first, then apply model-specific templates."""

                # Special handling for ShareGPT - offer mapping option
                if detected_format == "sharegpt":
                    print("\n  ShareGPT format detected. You have two options:")
                    print("  1. Convert to messages format (recommended)")
                    print("  2. Keep ShareGPT format and use Unsloth's mapping (faster)")

                    mapping_choice = self._get_input(
                        "\nChoose approach [1/2, default: 1]",
                        "1",
                        help_text="Option 1 converts your data to standard format. Option 2 keeps original format but requires Unsloth.",
                    ).strip()

                    if mapping_choice == "2":
                        self.answers["use_sharegpt_mapping"] = True
                        self.answers["auto_convert_dataset"] = False
                        print("✓ Will use Unsloth's ShareGPT mapping feature")

                        # Ask about custom runtime mapping
                        custom_mapping = (
                            self._get_input(
                                "\nUse custom column mapping? (y/N)",
                                "n",
                                help_text="By default, we map 'from'→'role' and 'value'→'content'. Choose 'y' if your columns have different names.",
                            )
                            .strip()
                            .lower()
                        )

                        if custom_mapping in ["y", "yes"]:
                            print("\n📋 Runtime Mapping Configuration:")
                            print("Map the keys inside your conversation messages to standard names.")

                            # Try to get a sample message to show its keys
                            sample_message_keys = []
                            try:
                                if hasattr(sample_dataset, "__getitem__"):
                                    first_item = sample_dataset[0]
                                    # Look for conversations/messages column
                                    for col in ["conversations", "messages", "chats", "dialog"]:
                                        if col in first_item and isinstance(first_item[col], list) and first_item[col]:
                                            sample_message_keys = list(first_item[col][0].keys())
                                            print(f"Sample message keys found: {', '.join(sample_message_keys)}")
                                            break
                            except:
                                pass

                            if not sample_message_keys:
                                print("(Could not detect message keys, common ones are: from, value, role, content)")

                            runtime_mapping = {}

                            role_col = self._get_input(
                                "\nKey containing role/sender (e.g., 'from', 'role', 'sender'): ",
                                "from",
                                help_text="The key that identifies who is speaking (e.g., 'from', 'role', 'sender')",
                            ).strip()
                            if role_col:
                                runtime_mapping["role"] = role_col

                            content_col = self._get_input(
                                "\nKey containing message content (e.g., 'value', 'content', 'text'): ",
                                "value",
                                help_text="The key that contains the actual message text",
                            ).strip()
                            if content_col:
                                runtime_mapping["content"] = content_col

                            # User/assistant role value mapping (single string per Unsloth API)
                            user_value = self._get_input(
                                "\nValue that means 'user' in the role field (e.g., 'human', 'user'): ",
                                "human",
                                help_text="What value in the role field represents the user?",
                            ).strip()
                            if user_value:
                                runtime_mapping["user"] = user_value

                            assistant_value = self._get_input(
                                "\nValue that means 'assistant' in the role field (e.g., 'gpt', 'assistant'): ",
                                "gpt",
                                help_text="What value in the role field represents the assistant?",
                            ).strip()
                            if assistant_value:
                                runtime_mapping["assistant"] = assistant_value

                            if runtime_mapping:
                                self.answers["runtime_mapping"] = runtime_mapping
                                print(f"✓ Runtime mapping configured: {runtime_mapping}")

                        # Ask about map_eos_token with smart default for chatml templates
                        chat_template = self.answers.get("chat_template", "")
                        is_chatml = any(t in str(chat_template).lower() for t in ["chatml", "gemma", "qwen"])

                        default_map_eos = "y" if is_chatml else "n"
                        default_text = "Y" if is_chatml else "y"

                        help_text = "This helps models that don't know when to stop generating. Maps tokens like <|im_end|> to the model's EOS token."
                        if is_chatml:
                            help_text += "\n\nRecommended for ChatML-style templates (chatml, gemma, qwen) as they often contain <|im_end|> tokens."

                        map_eos = (
                            self._get_input(
                                f"\nMap template end tokens to EOS? ({default_text}/{'n' if default_map_eos == 'y' else 'N'})",
                                default_map_eos,
                                help_text=help_text,
                            )
                            .strip()
                            .lower()
                        )

                        if map_eos in ["y", "yes"]:
                            self.answers["map_eos_token"] = True
                            print("✓ Will map template end tokens to EOS")

                        convert_choice = "n"  # Skip conversion
                    else:
                        convert_choice = "y"  # Proceed with conversion
                else:
                    convert_choice = (
                        self._get_input(
                            f"\nDo you want to analyze and convert your dataset to the model's chat format? (y/N)",
                            "n",
                            help_text=help_text,
                        )
                        .strip()
                        .lower()
                    )

                if convert_choice in ["y", "yes"]:
                    self.answers["auto_convert_dataset"] = True

                    # If format is unknown, prompt for column mapping
                    if detected_format == "unknown" and dataset_columns:
                        print("\n📋 Manual Column Mapping:")
                        print("Please identify which columns contain the conversation data.")
                        print(f"Available columns: {', '.join(dataset_columns)}")

                        column_mapping = {}

                        # Check if there's a system/instruction column
                        system_col = self._get_input(
                            "\nSystem/instruction column (press Enter to skip): ",
                            "",
                            help_text="Optional: Column containing system prompts or task instructions",
                        ).strip()
                        if system_col and system_col in dataset_columns:
                            column_mapping["system_col"] = system_col

                        # Check for Alpaca-style or Q&A style
                        is_alpaca_style = (
                            self._get_input(
                                "\nDoes your dataset have separate instruction and input columns? (y/N)",
                                "n",
                                help_text="Answer 'y' if you have columns like 'instruction' and 'input', 'n' for simple Q&A",
                            )
                            .strip()
                            .lower()
                        )

                        if is_alpaca_style in ["y", "yes"]:
                            # Alpaca-style mapping
                            instruction_col = self._get_input(
                                f"\nInstruction/task column (from: {', '.join(dataset_columns)}): ",
                                "",
                                help_text="Column containing the main instruction or task",
                            ).strip()
                            if instruction_col and instruction_col in dataset_columns:
                                column_mapping["instruction_col"] = instruction_col

                            input_col = self._get_input(
                                "\nInput/context column (press Enter to skip): ",
                                "",
                                help_text="Optional: Column containing additional input or context",
                            ).strip()
                            if input_col and input_col in dataset_columns:
                                column_mapping["input_col"] = input_col

                            output_col = self._get_input(
                                f"\nOutput/response column (from: {', '.join(dataset_columns)}): ",
                                "",
                                help_text="Column containing the model's expected output",
                            ).strip()
                            if output_col and output_col in dataset_columns:
                                column_mapping["output_col"] = output_col
                        else:
                            # Simple Q&A mapping
                            user_col = self._get_input(
                                f"\nUser/question column (from: {', '.join(dataset_columns)}): ",
                                "",
                                help_text="Column containing user messages or questions",
                            ).strip()
                            if user_col and user_col in dataset_columns:
                                column_mapping["user_col"] = user_col

                            assistant_col = self._get_input(
                                f"\nAssistant/answer column (from: {', '.join(dataset_columns)}): ",
                                "",
                                help_text="Column containing assistant responses or answers",
                            ).strip()
                            if assistant_col and assistant_col in dataset_columns:
                                column_mapping["assistant_col"] = assistant_col

                        if column_mapping:
                            self.answers["column_mapping"] = column_mapping
                            print(f"✓ Column mapping configured: {column_mapping}")
                        else:
                            print("⚠️ No column mapping specified. Dataset conversion may fail.")

                    # Try to find a matching template using our suggestion function
                    model_name = self.answers.get("model", "")
                    suggested_template = None

                    # Get template suggestion
                    try:
                        from ..preprocessor.chat_templates_standalone import get_template_for_model
                        from ..preprocessor.llm import get_available_chat_templates

                        available_templates = get_available_chat_templates()
                        if available_templates and model_name:
                            suggested_template = get_template_for_model(model_name)
                    except ImportError:
                        # Fallback to simple heuristics if Unsloth not available
                        if "llama" in model_name:
                            suggested_template = "llama3"  # Safe default
                        elif "gemma" in model_name:
                            suggested_template = "gemma"
                        elif "mistral" in model_name:
                            suggested_template = "mistral"

                    # Store suggestion separately, don't override
                    if suggested_template:
                        print(
                            f"  • Based on your model ({self.answers.get('model')}), Unsloth template suggestion: {suggested_template}"
                        )
                        self.answers["suggested_chat_template"] = suggested_template
                        print(
                            "  • Note: Using 'tokenizer' default unless you specify otherwise in chat_template parameter"
                        )
                    else:
                        print("  • Will use the model's built-in tokenizer template by default")

                    # Only set chat_template if it's not already set by user
                    if self.answers.get("chat_template") in [None, "", "tokenizer"]:
                        # Keep the safe default - let the tokenizer decide
                        self.answers["chat_template"] = "tokenizer"

                    # Ask about multi-turn conversation extension (for Alpaca datasets)
                    if detected_format == "alpaca":
                        help_text = """Conversation extension merges single-turn Alpaca rows into multi-turn conversations.

This is Unsloth's solution for making single-turn datasets work better:
  • Takes N random single-turn examples and combines them
  • Creates more natural, multi-turn conversations
  • Helps the model learn conversational flow

Setting it to 2-3 is usually optimal. Higher values may slow training."""

                        extend_choice = self._get_input(
                            "\nExtend single-turn data to multi-turn conversations? (1=no, 2-5=yes)",
                            "1",
                            help_text=help_text,
                        ).strip()

                        try:
                            extend_value = int(extend_choice)
                            if extend_value > 1:
                                self.answers["conversation_extension"] = extend_value
                                print(f"✓ Will merge {extend_value} single-turn examples into conversations")
                        except ValueError:
                            pass

                    # Show available chat templates if Unsloth is installed
                    templates = get_available_chat_templates()
                    if templates:
                        print("\n📋 Available chat templates from Unsloth:")

                        # Group templates by family for better display
                        template_groups = {}
                        for template_name in templates:
                            # Determine family based on template name
                            if any(x in template_name.lower() for x in ["llama-3", "llama3", "llama-2", "llama2"]):
                                family = "Llama"
                            elif any(x in template_name.lower() for x in ["gemma", "gemma2", "gemma3"]):
                                family = "Gemma"
                            elif any(x in template_name.lower() for x in ["phi-", "phi3", "phi4"]):
                                family = "Phi"
                            elif any(x in template_name.lower() for x in ["qwen"]):
                                family = "Qwen"
                            elif template_name in ["mistral", "zephyr", "chatml", "alpaca", "vicuna"]:
                                family = "Common Formats"
                            else:
                                family = "Other"

                            if family not in template_groups:
                                template_groups[family] = []
                            template_groups[family].append(template_name)

                        # Display grouped templates
                        for family in ["Llama", "Gemma", "Phi", "Qwen", "Common Formats", "Other"]:
                            if family in template_groups and template_groups[family]:
                                group = sorted(set(template_groups[family]))  # Remove duplicates and sort
                                if len(group) <= 3:
                                    print(f"  • {family}: {', '.join(group)}")
                                else:
                                    # Show first few and count for large groups
                                    shown = ", ".join(group[:3])
                                    print(f"  • {family}: {shown} + {len(group)-3} more")

                        print("\n  Note: The model's tokenizer template will be used by default")
                        print("  You can override this in the chat_template parameter if needed")
                else:
                    self.answers["auto_convert_dataset"] = False
                    print("✓ Skipping dataset conversion")
            else:
                # No conversion needed but we might still want to apply chat templates
                if detected_format == "messages" and self.trainer in ["sft", "reward"]:
                    help_text = """Your dataset is already in messages format but you may still want to:
  • Apply the model's specific chat template
  • Ensure proper formatting for your chosen model

This is optional but recommended for best results."""

                    template_choice = (
                        self._get_input("\nApply chat template to messages? (y/N)", "n", help_text=help_text)
                        .strip()
                        .lower()
                    )

                    if template_choice in ["y", "yes"]:
                        self.answers["apply_chat_template"] = True
                        print("✓ Will apply model's chat template to messages")

        except ImportError as e:
            logger.warning(f"Dataset conversion module not available: {e}")
            print(f"⚠️  Dataset format analysis skipped: {e}")
        except Exception as e:
            logger.warning(f"Error during dataset format analysis: {e}")
            print(f"⚠️  Dataset format analysis error: {e}")
            # Don't fail the wizard, this is optional

    def _prompt_column_mapping(self):
        """Prompt for dataset column mapping based on trainer type."""
        print("\n📝 Column Mapping:")

        # Handle LLM-specific column mapping
        if self.trainer_type == "llm":
            if self.trainer in ["sft", "default"]:
                # SFT needs text column
                if self._should_prompt_field("text_column"):
                    default_text = self.answers.get("text_column", "text") or "text"
                    help_text = """The column in your dataset containing the text to train on.

For instruction tuning (SFT):
• Should contain complete conversations or instruction-response pairs
• Example: "User: Write a poem. Assistant: Here's a beautiful poem..."
• Common names: 'text', 'conversation', 'dialogue', 'messages'

Format tips:
• Can be plain text or formatted conversations
• Should include both questions/instructions and responses
• The model will learn to generate similar text"""
                    text_col = self._get_input("Text column name", default_text, help_text=help_text).strip()
                    self.answers["text_column"] = text_col

            elif self.trainer in ["dpo", "orpo"]:
                # DPO/ORPO need prompt, chosen (text), and rejected columns
                print("\nDPO/ORPO requires three columns:")
                print("  • Prompt column: the instruction/question")
                print("  • Chosen column: the preferred response")
                print("  • Rejected column: the non-preferred response")

                # REQUIRED: prompt_text_column
                if self._should_prompt_field("prompt_text_column"):
                    prompt_default = self.answers.get("prompt_text_column", "prompt") or "prompt"
                    help_text = """The column containing the user's prompt or instruction.

This is what the user asks or requests:
• Example: "Write a story about a robot"
• Example: "Explain quantum physics simply"
• Common names: 'prompt', 'instruction', 'query', 'input'

The model learns to respond to these prompts."""
                    prompt_col = None
                    while not prompt_col:
                        prompt_col = self._get_input(
                            "\nPrompt column name [REQUIRED]", prompt_default, help_text=help_text
                        ).strip()
                        if not prompt_col:
                            print("❌ This field is required for DPO/ORPO training.")
                    self.answers["prompt_text_column"] = prompt_col

                # REQUIRED: text_column (chosen response)
                if self._should_prompt_field("text_column"):
                    chosen_default = self.answers.get("text_column", "chosen") or "chosen"
                    help_text = """The column containing the GOOD/preferred response.

This is the response you want the model to learn:
• The helpful, accurate, or preferred answer
• Example: A well-written, informative response
• Common names: 'chosen', 'accepted', 'positive', 'good'

The model learns to generate responses like these."""
                    chosen_col = None
                    while not chosen_col:
                        chosen_col = self._get_input(
                            "Chosen response column [REQUIRED]", chosen_default, help_text=help_text
                        ).strip()
                        if not chosen_col:
                            print("❌ This field is required for DPO/ORPO training.")
                    self.answers["text_column"] = chosen_col

                # REQUIRED: rejected_text_column
                if self._should_prompt_field("rejected_text_column"):
                    rejected_default = self.answers.get("rejected_text_column", "rejected") or "rejected"
                    help_text = """The column containing the BAD/non-preferred response.

This is the response you DON'T want:
• The unhelpful, incorrect, or non-preferred answer
• Example: A rude, incorrect, or low-quality response
• Common names: 'rejected', 'negative', 'bad', 'wrong'

The model learns to avoid generating responses like these."""
                    rejected_col = None
                    while not rejected_col:
                        rejected_col = self._get_input(
                            "Rejected response column [REQUIRED]", rejected_default, help_text=help_text
                        ).strip()
                        if not rejected_col:
                            print("❌ This field is required for DPO/ORPO training.")
                    self.answers["rejected_text_column"] = rejected_col

            elif self.trainer == "ppo":
                # PPO needs text/prompt column and reward model
                if self._should_prompt_field("text_column"):
                    text_default = self.answers.get("text_column", "text") or "text"
                    help_text = """The column containing prompts for PPO training.

For reinforcement learning with PPO:
• Contains the prompts/questions to generate responses for
• Example: "Write a helpful response about..."
• The PPO trainer will generate responses and score them
• Common names: 'text', 'prompt', 'query', 'instruction'"""
                    text_col = self._get_input("Text/prompt column name", text_default, help_text=help_text).strip()
                    self.answers["text_column"] = text_col

                # REQUIRED: reward model path for PPO
                print("\n⚠️  PPO requires a trained reward model!")
                if self._should_prompt_field("rl_reward_model_path"):
                    reward_model = None
                    help_text = """Path to a reward model for PPO training.

You must first train a reward model using:
  aitraining llm --trainer reward --data-path preference_data

Then use it with PPO:
  • Local path: ./models/my_reward_model
  • HuggingFace Hub: username/model-name

The reward model scores generated responses to guide training."""
                    while not reward_model:
                        reward_model = self._get_input(
                            "Reward model path [REQUIRED] (local path or HF model ID)",
                            self.answers.get("rl_reward_model_path", ""),
                            help_text=help_text,
                        ).strip()
                        if not reward_model:
                            print("❌ PPO training requires a reward model path.")
                    self.answers["rl_reward_model_path"] = reward_model
            return

        # Handle non-LLM trainer column mapping
        metadata = TRAINER_METADATA.get(self.trainer_type)
        if not metadata:
            return

        required_columns = metadata.get("required_columns", [])

        # Prompt for each required column
        for col_name in required_columns:
            if not self._should_prompt_field(col_name):
                continue

            default = self.answers.get(col_name)
            if not default:
                default = col_name.replace("_column", "").replace("_", " ").strip()
            display_name = col_name.replace("_", " ").title()

            descriptions = {
                "text_column": "Column containing the text/context",
                "target_column": "Column containing the target labels/values",
                "tokens_column": "Column containing tokenized text (list of tokens)",
                "tags_column": "Column containing token labels (list of tags)",
                "question_column": "Column containing questions",
                "answer_column": "Column containing answers",
                "image_column": "Column containing images",
                "sentence1_column": "Column containing first sentence",
                "sentence2_column": "Column containing second sentence",
                "sentence3_column": "Column containing third sentence (optional)",
                "target_columns": "Comma-separated list of target column names",
                "categorical_columns": "Comma-separated list of categorical column names",
                "numerical_columns": "Comma-separated list of numerical column names",
                "prompt_text_column": "Column containing prompts/instructions",
            }

            desc = descriptions.get(col_name, "")
            if desc:
                print(f"\n{desc}")

            # Add specific help text for each column type
            help_texts = {
                "text_column": """The column containing your text data.

This is the main text that will be analyzed:
• For classification: the text to classify
• For NER: the text to extract entities from
• Common names: 'text', 'content', 'review', 'comment', 'sentence'""",
                "target_column": """The column containing labels or target values.

What the model should predict:
• For classification: the category/label (e.g., 'positive', 'negative')
• For regression: the numerical value to predict
• Common names: 'label', 'target', 'class', 'category', 'rating'""",
                "tokens_column": """The column containing tokenized text (list of words).

For token classification tasks:
• Should be a list of tokens/words
• Example: ["The", "cat", "sat", "on", "the", "mat"]
• Common names: 'tokens', 'words'""",
                "tags_column": """The column containing token labels (list of tags).

For NER/POS tagging:
• One label per token
• Example: ["O", "B-ANIMAL", "O", "O", "O", "B-OBJECT"]
• Common names: 'tags', 'labels', 'ner_tags'""",
                "question_column": """The column containing questions.

For Q&A tasks:
• The question being asked
• Example: "What is the capital of France?"
• Common names: 'question', 'query'""",
                "answer_column": """The column containing answers.

For Q&A tasks:
• The answer to the question
• Can be text spans or full answers
• Common names: 'answer', 'answers', 'text'""",
                "image_column": """The column containing images.

For vision tasks:
• File paths or image data
• Example: "images/cat_001.jpg"
• Common names: 'image', 'image_path', 'file_name'""",
                "sentence1_column": """The first sentence for comparison.

For sentence similarity:
• The primary sentence
• Example: "The weather is nice today"
• Common names: 'sentence1', 'text1', 'premise'""",
                "sentence2_column": """The second sentence for comparison.

For sentence similarity:
• The sentence to compare with
• Example: "It's a beautiful day"
• Common names: 'sentence2', 'text2', 'hypothesis'""",
                "sentence3_column": """The third sentence (optional).

For triplet loss training:
• An additional comparison sentence
• Used in advanced similarity learning
• Common names: 'sentence3', 'text3', 'negative'""",
                "target_columns": """Comma-separated list of columns to predict.

For multi-target prediction:
• Multiple values to predict at once
• Example: "age,income,risk_score"
• Each should be a column in your dataset""",
                "categorical_columns": """Comma-separated list of categorical columns.

For tabular data:
• Columns with categories (not numbers)
• Example: "color,size,brand,category"
• Will be encoded automatically""",
                "numerical_columns": """Comma-separated list of numerical columns.

For tabular data:
• Columns with numbers
• Example: "age,price,quantity,rating"
• Will be normalized automatically""",
                "prompt_text_column": """The column containing prompts or instructions.

The input prompt/instruction:
• What the user is asking
• Example: "Translate to French:"
• Common names: 'prompt', 'instruction', 'input'""",
            }

            help_text = help_texts.get(col_name, desc if desc else None)

            value = self._get_input(f"{display_name} [{default}]", default, help_text=help_text).strip()
            self.answers[col_name] = value

    def _validate_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Validate that a model exists (either on HuggingFace Hub or as a local path).

        Args:
            model_name: Model identifier (HF ID or local path)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not model_name or not model_name.strip():
            return False, "Model name cannot be empty"

        model_name = model_name.strip()

        # Check if it looks like a HuggingFace model (contains /)
        if (
            "/" in model_name
            and not os.path.isabs(model_name)
            and not model_name.startswith("./")
            and not model_name.startswith("../")
        ):
            # Likely a HuggingFace model - verify it exists
            try:
                from huggingface_hub import HfApi, model_info

                HfApi()

                # Try to get model info
                try:
                    info = model_info(model_name, token=self.answers.get("token"))
                    # If we get here, model exists
                    return True, ""
                except Exception as e:
                    error_msg = str(e)
                    if (
                        "404" in error_msg
                        or "Not Found" in error_msg
                        or "RepositoryNotFoundError" in str(type(e).__name__)
                    ):
                        # Suggest common typos
                        suggestions = []
                        if "gemma3" in model_name.lower() and "gemma-3" not in model_name:
                            suggestions.append(f"Did you mean '{model_name.replace('gemma3', 'gemma-3')}'?")
                        elif "gemma-3" not in model_name.lower() and "gemma" in model_name.lower():
                            suggestions.append(f"Did you mean '{model_name.replace('gemma', 'gemma-3')}'?")

                        suggestion_text = f"\n  Suggestions: {', '.join(suggestions)}" if suggestions else ""
                        return (
                            False,
                            f"Model '{model_name}' not found on HuggingFace Hub.{suggestion_text}\n  Check the model ID at https://huggingface.co/models",
                        )
                    else:
                        return False, f"Error checking HuggingFace model '{model_name}': {error_msg}"
            except ImportError:
                # huggingface_hub not available, skip validation
                logger.warning("huggingface_hub not available, skipping model validation")
                return True, ""
        else:
            # Likely a local path - verify it exists and contains model files
            if not os.path.exists(model_name):
                return False, f"Local model path does not exist: {model_name}\n  Please check the path and try again"

            if not os.path.isdir(model_name):
                return (
                    False,
                    f"Model path is not a directory: {model_name}\n  Please provide a directory containing the model files",
                )

            # Check for common model files
            required_files = ["config.json"]  # At minimum, should have config.json
            optional_files = ["tokenizer_config.json", "tokenizer.json", "model.safetensors", "pytorch_model.bin"]

            has_config = os.path.exists(os.path.join(model_name, "config.json"))
            has_tokenizer = any(os.path.exists(os.path.join(model_name, f)) for f in optional_files)

            if not has_config:
                return (
                    False,
                    f"Model directory does not contain 'config.json': {model_name}\n  This doesn't appear to be a valid model directory",
                )

            if not has_tokenizer:
                logger.warning(f"Model directory found but no tokenizer files detected in {model_name}")
                # Don't fail, just warn - some models might have tokenizer elsewhere

            return True, ""

    def _prompt_model_selection(self):
        """Prompt for model selection."""
        print("\n" + "=" * 60)
        print("📋 Step 3: Model Selection")
        print("=" * 60)

        # Check current model
        current_model = self.answers.get("model")
        if current_model in PLACEHOLDER_MODEL:
            current_model = None

        if not current_model:
            # Interactive selection
            while True:
                selected_model = self._run_catalog_interaction_loop(
                    "models", get_popular_models, filter_func=self._filter_models_by_size
                )
                if not selected_model:
                    break

                # Validate the selected model
                is_valid, error_msg = self._validate_model(selected_model)
                if is_valid:
                    self.answers["model"] = selected_model
                    print(f"✓ Model: {selected_model}")
                    break
                else:
                    print(f"\n❌ {error_msg}")
                    retry = self._get_yes_no("\nTry again with a different model? [Y/n]", default=True)
                    if not retry:
                        # User wants to skip validation or use anyway
                        self.answers["model"] = selected_model
                        print(f"⚠️  Using model '{selected_model}' without validation (may fail during training)")
                        break
        else:
            # Model already set - validate it
            is_valid, error_msg = self._validate_model(current_model)
            if is_valid:
                print(f"✓ Model: {current_model}")
            else:
                print(f"\n⚠️  Warning: {error_msg}")
                use_anyway = self._get_yes_no(
                    f"\nUse '{current_model}' anyway? (May fail during training) [y/N]", default=False
                )
                if not use_anyway:
                    # Clear and reprompt
                    self.answers["model"] = None
                    self._prompt_model_selection()

        # Suggest mixed precision for non-CUDA systems (MPS/CPU)
        self._suggest_mixed_precision()

    def _suggest_mixed_precision(self):
        """Suggest enabling mixed precision (bf16) for MPS/CPU systems after model selection."""
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False

            # Only suggest if CUDA is not available and we haven't already set mixed_precision
            if not cuda_available and self.answers.get("mixed_precision") is None:
                print("\n" + "=" * 60)
                print("⚡ Performance Suggestion")
                print("=" * 60)

                if mps_available:
                    print("\n💡 Apple Silicon GPU detected (MPS)")
                    print("   Enabling bf16 mixed precision can significantly speed up training (3-5x faster)!")
                    print("   • Current: float32 (slower)")
                    print("   • Recommended: bf16 (much faster on Apple Silicon)")
                else:
                    print("\n💡 No CUDA GPU detected")
                    print("   Enabling mixed precision can help with memory and speed on CPU/MPS")
                    print("   • Recommended: bf16 for better performance")

                enable = self._get_yes_no("\nEnable bf16 mixed precision? [Y/n]", default=True)
                if enable:
                    self.answers["mixed_precision"] = "bf16"
                    print("✓ Mixed precision set to: bf16")
                else:
                    print("⚠ Skipping mixed precision (training may be slower)")
        except ImportError:
            # torch not available, skip suggestion
            pass
        except Exception as e:
            # Don't fail wizard if this check fails
            logger.debug(f"Could not check for CUDA/MPS availability: {e}")

    def _filter_models_by_size(self, models: List[CatalogEntry], size_filter: str) -> List[CatalogEntry]:
        """Filter models based on parameter count."""
        if size_filter == "all":
            return models
        filtered = []
        for m in models:
            if m.params is None:
                # Include unknown sizes in 'all' but maybe exclude in specific filters?
                # Or treat as "unknown" and exclude from S/M/L?
                # If the user explicitly asks for "Small", they probably want guaranteed small.
                continue

            # definitions: Small < 3B, Medium 3-10B, Large > 10B
            if size_filter == "small" and m.params < 3e9:
                filtered.append(m)
            elif size_filter == "medium" and 3e9 <= m.params <= 10e9:
                filtered.append(m)
            elif size_filter == "large" and m.params > 10e9:
                filtered.append(m)
        return filtered

    def _print_catalog_interactive(
        self, label: str, entries: List[CatalogEntry], sort_by: str, size_filter: str, search_query: Optional[str]
    ) -> None:
        """Pretty-print catalog entries with interactive hints."""
        print(f"\nPopular {label} ({sort_by}):")
        if label == "models":
            print(f"  Sort: [T]rending [D]ownloads [L]ikes [R]ecent")
            print(f"  Filter size: [A]ll [S]mall(<3B) [M]edium(3-10B) [L]arge(>10B) (current: {size_filter})")
        else:
            print(f"  Sort: [T]rending [D]ownloads [L]ikes [R]ecent")

        if search_query:
            print(f"  Search: '{search_query}'")

        if not entries:
            print("  (No results found)")
        else:
            for idx, entry in enumerate(entries, 1):
                text = f"  {idx}. {entry.label}"
                if entry.params:
                    text += f" {format_params(entry.params)}"
                elif entry.description:
                    text += f" — {entry.description}"

                if entry.id != entry.label:
                    text += f" ({entry.id})"
                print(text)
        print("  (Type number, HF ID, or command: /sort, /filter, /search <query>, /refresh)")

    def _run_catalog_interaction_loop(self, item_type, fetch_func, filter_func=None) -> str:
        """Run the interactive command loop for catalog selection."""
        sort_by = "trending"
        search_query = None
        size_filter = "all"

        while True:
            try:
                # Fetch items (cache handled by function itself based on args)
                items = fetch_func(self.trainer_type, self.trainer, sort_by=sort_by, search_query=search_query)
            except Exception as e:
                logger.error(f"Error fetching {item_type}: {e}")
                items = []

            filtered_items = items
            if filter_func:
                filtered_items = filter_func(items, size_filter)

            # Display
            self._print_catalog_interactive(item_type, filtered_items[:20], sort_by, size_filter, search_query)

            # Prompt
            prompt = f"{item_type.capitalize().rstrip('s')} (number, HF ID, or command)"
            default_val = "1" if filtered_items else ""

            # Add help text for dataset/model selection
            if item_type == "datasets":
                help_text = """Choose a dataset for training.

You can:
• Type a number (1-20) to select from the list
• Type a HuggingFace dataset ID (e.g., 'tatsu-lab/alpaca')
• Type a local folder path (e.g., './my_data')
• Type /search <query> to find specific datasets
• Type /sort to change sorting
• Type /refresh to reload the list"""
            else:  # models
                help_text = """Choose a model to fine-tune.

You can:
• Type a number (1-20) to select from the list
• Type a HuggingFace model ID (e.g., 'meta-llama/Llama-2-7b')
• Type a local model path (e.g., './my_model')
• Type /search <query> to find specific models
• Type /sort to change sorting
• Type /filter to filter by size
• Type /refresh to reload the list"""

            choice = self._get_input(prompt, default_val, help_text=help_text).strip()
            if not choice:
                choice = default_val

            if choice.startswith("/"):
                parts = choice.split(maxsplit=1)
                cmd = parts[0]
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == "/search":
                    search_query = arg
                elif cmd == "/sort":
                    print(f"Sort options: [T]rending [D]ownloads [L]ikes [R]ecent")
                    help_text = """How to sort the results:

T = Trending - What's popular right now
D = Downloads - Most downloaded all-time
L = Likes - Most liked by the community
R = Recent - Newest additions"""
                    s = self._get_input("Sort by", "T", help_text=help_text).upper()
                    sort_map = {"T": "trending", "D": "downloads", "L": "likes", "R": "recent"}
                    sort_by = sort_map.get(s, "trending")
                elif cmd.startswith("/filter"):
                    if item_type == "models":
                        print(f"Filter size: [A]ll [S]mall(<3B) [M]edium(3-10B) [L]arge(>10B)")
                        help_text = """Filter models by size:

A = All - Show everything
S = Small (<3B parameters) - Fast, runs on most hardware
M = Medium (3-10B) - Good balance of quality and speed
L = Large (>10B) - Best quality, needs powerful GPU"""
                        s = self._get_input("Filter size", "A", help_text=help_text).upper()
                        size_map = {"A": "all", "S": "small", "M": "medium", "L": "large"}
                        size_filter = size_map.get(s, "all")
                    else:
                        print("Filters not available for datasets yet.")
                elif cmd == "/refresh":
                    try:
                        fetch_func.cache_clear()
                        print("Cache cleared.")
                    except AttributeError:
                        pass
                else:
                    print(f"Unknown command: {cmd}")
                continue

            # Selection
            resolved = self._resolve_catalog_choice(choice, filtered_items[:20])
            if resolved:
                return resolved

            # Manual entry?
            # If it looks like a HF ID (contains /) or user insists
            if choice and not choice.isdigit():
                return choice

            print("❌ Invalid selection. Please try again.")

    def _select_from_catalog(
        self,
        prompt: str,
        entries: List[CatalogEntry],
    ) -> Optional[str]:
        """
        Show an interactive select in the terminal when possible; otherwise return None to fall back.
        Returns the selected id or None.
        """
        if not entries:
            return None
        # Build display labels
        labels: List[str] = []
        for e in entries:
            text = e.label or e.id
            if e.description:
                text += f" — {e.description}"
            if e.id != e.label:
                text += f" ({e.id})"
            labels.append(text)
        # Add manual option
        manual_option = "Other (type manually)"
        labels.append(manual_option)

        # Try questionary
        try:
            import questionary  # type: ignore

            answer = questionary.select(
                message=prompt,
                choices=labels,
                qmark=">",
                instruction="Use ↑/↓ to select, Enter to confirm",
            ).unsafe_ask()
            if answer == manual_option:
                return None
            # Map back to id by index
            idx = labels.index(answer)
            return entries[idx].id if idx < len(entries) else None
        except Exception:
            pass

        # Try InquirerPy
        try:
            from InquirerPy import inquirer  # type: ignore

            answer = inquirer.select(
                message=prompt,
                choices=labels,
                instruction="(Arrow keys)",
                qmark=">",
                long_instruction="",
            ).execute()
            if answer == manual_option:
                return None
            idx = labels.index(answer)
            return entries[idx].id if idx < len(entries) else None
        except Exception:
            pass

        # No interactive library available
        return None

    def _prompt_advanced_params(self):
        """Prompt for advanced parameters organized by groups."""
        print("\n" + "=" * 60)
        print("📋 Step 5: Advanced Configuration (Optional)")
        print("=" * 60)

        print("\nWould you like to configure advanced parameters?")
        print("  • Training hyperparameters (learning rate, batch size, etc.)")
        if self.trainer_type in ["llm", "seq2seq", "vlm"]:
            print("  • PEFT/LoRA settings")
            print("  • Model quantization")
        print("  • And more...")

        configure_advanced = self._get_yes_no("\nConfigure advanced parameters? [y/N]", default=False)

        if not configure_advanced:
            print("\n✓ Using default settings for advanced parameters.")
            return

        # Get params class and field groups based on trainer type
        if self.trainer_type == "llm":
            params_class = LLMTrainingParams
            field_groups = LLM_FIELD_GROUPS
            field_scopes = LLM_FIELD_SCOPES
        else:
            metadata = TRAINER_METADATA.get(self.trainer_type)
            if not metadata:
                print("⚠️  No advanced configuration available for this trainer.")
                return
            params_class = metadata["params_class"]
            field_groups = metadata["field_groups"]
            field_scopes = None  # Non-LLM trainers don't have scopes

        # Get field info for the selected trainer
        field_info_list = get_field_info(params_class, field_groups, field_scopes, enforce_scope=False)

        # Organize fields by group
        fields_by_group = {}
        for field_info in field_info_list:
            group = field_info.get("group", "Other")
            scope = field_info.get("scope", ["all"])

            # Filter by trainer scope (only for LLM)
            if field_scopes and "all" not in scope and self.trainer not in scope:
                continue

            # Skip already configured fields and command-specific args
            field_name = field_info["arg"].replace("--", "").replace("-", "_")
            if field_name in self.answers and field_name != "token":
                continue
            if field_name in ["train", "deploy", "inference", "backend"]:
                continue

            if group not in fields_by_group:
                fields_by_group[group] = []
            fields_by_group[group].append(field_info)

        # Prompt by group in order
        for group in GROUP_ORDER:
            if group not in fields_by_group or group == "Basic":
                continue

            fields = fields_by_group[group]
            if not fields:
                continue

            print(f"\n{'─'*60}")
            print(f"⚙️  {group}")
            print(f"{'─'*60}")

            configure_group = self._get_yes_no(f"\nConfigure {group} parameters? [y/N]", default=False)

            if not configure_group:
                continue

            for field_info in fields:
                self._prompt_field(field_info)

    def _prompt_field(self, field_info: Dict[str, Any]):
        """Prompt for a single field."""
        field_name = field_info["arg"].replace("--", "").replace("-", "_")
        field_help = field_info.get("help", "")
        field_type = field_info.get("type")
        field_default = field_info.get("default")
        choices = field_info.get("choices")

        # Format default value for display
        default_str = str(field_default) if field_default is not None else "None"

        # Build prompt
        prompt = f"\n{field_name}"

        # Handle different field types
        if field_info.get("action") == "store_true":
            # Boolean flag
            value = self._get_yes_no(f"{prompt} [y/N]", default=False, help_text=field_help)
            self.answers[field_name] = value
        elif choices:
            # Choice field
            print(f"  Choices: {', '.join(str(c) for c in choices)}")
            value = self._get_input(f"{prompt} [{default_str}]", default_str, help_text=field_help).strip()
            if value and value in [str(c) for c in choices]:
                # Convert to appropriate type
                if isinstance(choices[0], bool):
                    self.answers[field_name] = value.lower() in ("true", "yes", "y", "1")
                else:
                    self.answers[field_name] = type(choices[0])(value)
        else:
            # General input
            value = self._get_input(f"{prompt} [{default_str}]", default_str, help_text=field_help).strip()
            if value and value != default_str:
                # Convert to appropriate type
                try:
                    if field_type == int:
                        self.answers[field_name] = int(value)
                    elif field_type == float:
                        self.answers[field_name] = float(value)
                    elif field_type == str:
                        self.answers[field_name] = value
                    else:
                        self.answers[field_name] = value
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {value} to {field_type}, using as string")
                    self.answers[field_name] = value

    def _show_summary_and_confirm(self) -> bool:
        """Show configuration summary and ask for confirmation."""
        print("\n" + "=" * 60)
        print("📋 Configuration Summary")
        print("=" * 60)

        # Group parameters for better readability
        summary = {
            "Basic Configuration": {},
            "Dataset": {},
            "Model & Training": {},
            "Logging": {},
            "Advanced": {},
        }

        basic_keys = ["project_name", "trainer", "task"]
        dataset_keys = [
            "data_path",
            "train_split",
            "valid_split",
            "text_column",
            "prompt_text_column",
            "rejected_text_column",
            "tokens_column",
            "tags_column",
            "question_column",
            "answer_column",
            "image_column",
            "target_column",
            "target_columns",
            "categorical_columns",
            "numerical_columns",
            "sentence1_column",
            "sentence2_column",
            "sentence3_column",
            "id_column",
        ]
        model_keys = ["model", "epochs", "batch_size", "lr", "peft", "quantization", "num_trials", "time_limit"]
        logging_keys = ["log", "wandb_visualizer", "wandb_project", "wandb_entity"]

        for key, value in self.answers.items():
            if key in basic_keys:
                summary["Basic Configuration"][key] = value
            elif key in dataset_keys:
                summary["Dataset"][key] = value
            elif key in model_keys:
                summary["Model & Training"][key] = value
            elif key in logging_keys:
                summary["Logging"][key] = value
            else:
                summary["Advanced"][key] = value

        # Print summary
        sensitive_fields = {"token", "wandb_token", "hf_token", "huggingface_token"}

        for section, params in summary.items():
            if not params:
                continue
            print(f"\n{section}:")
            for key, value in params.items():
                # Format value nicely
                if isinstance(value, bool):
                    value_str = "✓" if value else "✗"
                elif value is None:
                    value_str = "(not set)"
                else:
                    value_str = str(value)

                # Special formatting for logging fields
                if key == "log":
                    if value == "wandb":
                        value_str = "wandb ✓"
                    elif value == "none":
                        value_str = "none (disabled)"
                    elif value:
                        value_str = f"{value} ✓"
                elif key == "wandb_visualizer":
                    if value:
                        value_str = "✓ (LEET panel will open automatically)"
                    else:
                        value_str = "✗"

                if key in sensitive_fields and value and value_str != "(not set)":
                    value_str = "*****"
                print(f"  • {key}: {value_str}")

        # Try to validate with Pydantic
        print("\n" + "─" * 60)
        try:
            # Get the appropriate params class
            if self.trainer_type == "llm":
                params_class = LLMTrainingParams
            else:
                metadata = TRAINER_METADATA.get(self.trainer_type)
                if metadata:
                    params_class = metadata["params_class"]
                else:
                    params_class = None

            if params_class:
                _ = params_class(**self.answers)
                print("✓ Configuration is valid!")
            else:
                print("⚠️  Could not validate configuration (no params class found)")
        except ValidationError as e:
            print("⚠️  Configuration validation warnings:")
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                msg = error["msg"]
                print(f"  • {field}: {msg}")
            print("\n❓ You can fix these issues or proceed anyway.")

        # Confirm
        print("\n" + "=" * 60)
        confirmed = self._get_yes_no("\n🚀 Start training with this configuration? [Y/n]", default=True)

        if confirmed:
            # Offer to manually open wandb LEET panel if wandb logging is enabled
            self._prompt_manual_wandb_leet()

        return confirmed

    def _prompt_manual_wandb_leet(self):
        """Prompt user to manually open wandb LEET panel if wandb logging is enabled."""
        log_method = self.answers.get("log", "none")
        if log_method == "wandb":
            project_name = self.answers.get("project_name", "project-name")
            # Resolve project path - same logic as in run_training:
            # project_run_dir = os.path.abspath(params.project_name)
            project_path = os.path.abspath(project_name)

            print("\n" + "=" * 60)
            print("📊 W&B Visualizer (LEET)")
            print("=" * 60)
            print("\n💡 Open the W&B LEET panel to monitor training in real-time!")
            print(f"   Project directory: {project_path}")

            wandb_command = f'WANDB_DIR="{project_path}" wandb beta leet "{project_path}"'
            print(f"\n   Command: {wandb_command}")

            open_now = self._get_yes_no("\nOpen W&B LEET panel now? (in a new terminal) [Y/n]", default=True)

            if open_now:
                try:
                    import subprocess
                    import sys

                    # Create the command to run in a new terminal
                    # Keep terminal open after LEET exits (like run_training does)
                    if sys.platform == "darwin":
                        # Use osascript to open new Terminal window with the command
                        # Add 'read' at the end to keep terminal open after LEET exits
                        script = f"""
                        tell application "Terminal"
                            activate
                            do script "cd {os.getcwd()} && {wandb_command}; echo ''; echo 'LEET panel closed. Press Enter to close this window...'; read"
                        end tell
                        """
                        subprocess.run(["osascript", "-e", script], check=False)
                        print("✓ Opening W&B LEET panel in a new Terminal window...")
                        print("   (Terminal will stay open after LEET exits)")
                    else:
                        # For Linux/other systems, try to use xterm or gnome-terminal
                        # Fallback: just print the command
                        print(f"\n⚠️  Please run this command in a new terminal:")
                        print(f"   {wandb_command}")
                        print("   (Add '; read' at the end to keep terminal open)")
                except Exception as e:
                    print(f"\n⚠️  Could not auto-open terminal. Please run this command manually:")
                    print(f"   {wandb_command}")
                    logger.debug(f"Failed to open terminal: {e}")
            else:
                print(f"\n💡 You can open it later with:")
                print(f"   {wandb_command}")

    def _show_help(self, help_text: Optional[str]) -> None:
        """Display contextual help for the current prompt."""
        print("\nℹ️  Help")
        if help_text:
            print(f"  {help_text}")
        else:
            print("  Provide a value or choose from the suggestions shown.")
        print("  Commands: :back, :help, :exit\n")

    def _get_input(self, prompt: str, default: str, help_text: Optional[str] = None) -> str:
        """Get input from user with default value and command shortcuts."""
        while True:
            try:
                value = input(f"{prompt}: ").strip()
            except (EOFError, KeyboardInterrupt):
                raise WizardExitRequest()

            if value in (":exit", ":quit"):
                raise WizardExitRequest()
            if value == ":back":
                if self._current_step is None:
                    print("↩️  No previous step available yet.")
                    continue
                raise WizardBackRequest()
            if value in (":help", "?", ":h"):
                self._show_help(help_text)
                continue

            return value if value else default

    def _get_yes_no(self, prompt: str, default: bool, help_text: Optional[str] = None) -> bool:
        """Get yes/no input from user."""
        while True:
            default_choice = "y" if default else "n"
            value = self._get_input(prompt, default_choice, help_text=help_text).strip().lower()
            if not value:
                return default
            if value in ("y", "yes", "1", "true"):
                return True
            if value in ("n", "no", "0", "false"):
                return False
            print("❌ Please enter y or n.")


def run_wizard(initial_args: Optional[Dict[str, Any]] = None, trainer_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the interactive wizard.

    Args:
        initial_args: Initial arguments from CLI (if any)
        trainer_type: Trainer type (llm, text-classification, etc.). If None, wizard will prompt.

    Returns:
        Dictionary of configuration parameters ready for the appropriate params class
    """
    wizard = InteractiveWizard(initial_args, trainer_type=trainer_type)
    return wizard.run()
