# AITraining Terminal User Interface (TUI)

The AITraining TUI provides a full-screen, keyboard-driven interface for configuring and running AI model training. It offers an intuitive way to browse, filter, and edit parameters without memorizing command-line flags.

## Features

### 🎯 Core Features
- **Interactive Parameter Configuration**: Browse and edit all training parameters through an intuitive form interface
- **Trainer-Specific Filtering**: Automatically shows only relevant parameters for your selected trainer (SFT, DPO, ORPO, PPO, etc.)
- **Grouped Organization**: Parameters organized into logical groups for easy navigation
- **Real-time Command Preview**: See the exact CLI command that will be executed
- **Live Training Logs & LEET**: W&B logging is enabled by default (`log=wandb`), so you get streaming logs plus the LEET visualizer sidecar out of the box. Toggle to `none` if you need a silent run.
- **Catalog Browser**: A dedicated **Catalog** tab surfaces curated Hugging Face models and datasets—the same lists used in the web UI. Pick an entry and the form auto-fills the corresponding `model` or `data_path`.
- **Configuration Management**: Save and load parameter presets in JSON or YAML format

### 🎨 User Experience
- **Keyboard-Driven**: Full keyboard navigation - no mouse required
- **Theme Support**: Dark and light themes for comfortable viewing
- **Parameter Search**: Quickly find parameters with fuzzy search
- **Validation**: Real-time validation with helpful error messages
- **Context Help**: Detailed parameter descriptions and default values always visible
- **Hub & W&B Ready**: Fill in Hugging Face and W&B tokens directly from the *Hub Integration* group so downloads, pushes, and LEET replays work without leaving the TUI.

## Installation

The TUI requires the `textual` and `rich` packages, which are automatically installed with AITraining:

```bash
pip install -e .
```

Or install dependencies separately:

```bash
pip install textual==0.86.1 rich==13.9.4
```

## Usage

### Launching the TUI

```bash
aitraining tui
```

### Command-Line Options

```bash
aitraining tui [options]

Options:
  --theme {dark,light}  Color theme for the TUI (default: dark)
  --dry-run            Enable dry-run mode (commands won't execute)
  --config FILE        Load configuration from JSON/YAML file on startup
  --help               Show help message
```

### Examples

```bash
# Launch with dark theme (default)
aitraining tui

# Launch with light theme
aitraining tui --theme light

# Launch in dry-run mode for testing
aitraining tui --dry-run

# Launch with pre-loaded configuration
aitraining tui --config my_config.yaml
```

## Interface Layout

The TUI consists of three main panels:

### Left Panel: Navigation
- **Trainer Selector**: Choose between Default, SFT, DPO, ORPO, PPO, or Reward training modes
- **Parameter Groups**: Navigate through categorized parameters:
  - Basic (model, data paths, etc.)
  - Data Processing (includes `max_samples` for limiting dataset size during testing)
  - Training Configuration
  - Training Hyperparameters
  - PEFT/LoRA
  - DPO/ORPO (when applicable)
  - Reinforcement Learning (PPO)
  - Advanced Features

### Center Panel: Parameter Form
- Auto-generated widgets based on parameter types:
  - Text inputs for strings and paths
  - Numeric inputs for integers and floats
  - Checkboxes for boolean flags
  - Dropdowns for enumerated options
  - Multi-line editors for JSON parameters
- Modified values are highlighted with indicators

### Right Panel: Information Tabs
- **Context Tab**: Shows help text, default values, and validation for the selected parameter
- **Command Tab**: Displays the CLI command that will be executed
- **Logs Tab**: Shows live training output when running
- **Visualizer Tab**: Streams the W&B LEET dashboard whenever `log=wandb` and `wandb_visualizer=True`
- **Catalog Tab**: Browse curated model/dataset suggestions; selecting an item updates the form and command preview instantly

## Keyboard Shortcuts

### Navigation
| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Navigate between panels |
| `↑` / `↓` | Navigate lists and menus |
| `Enter` | Select/Activate |
| `Escape` | Clear search / Close dialogs |

### Actions
| Key | Action |
|-----|--------|
| `Ctrl+R` | Run training with current configuration |
| `Ctrl+S` | Save configuration to file |
| `Ctrl+L` | Load configuration from file |
| `Ctrl+P` | Preview command |
| `Ctrl+D` | Toggle dry run mode |
| `Ctrl+T` | Toggle theme (dark/light) |

### Other
| Key | Action |
|-----|--------|
| `/` | Search parameters |
| `F1` | Show help screen |
| `q` / `Ctrl+C` | Quit application |

## Configuration Files

### Saving Configuration

Press `Ctrl+S` to save the current configuration. The TUI supports both JSON and YAML formats:

```json
{
  "trainer": "sft",
  "parameters": {
    "model": "meta-llama/Llama-2-7b-hf",
    "project_name": "my-sft-model",
    "data_path": "./data",
    "lr": 2e-5,
    "epochs": 3,
    "batch_size": 4
  }
}
```

```yaml
trainer: sft
parameters:
  model: meta-llama/Llama-2-7b-hf
  project_name: my-sft-model
  data_path: ./data
  lr: 2e-5
  epochs: 3
  batch_size: 4
```

### Loading Configuration

Press `Ctrl+L` to load a saved configuration, or use the `--config` flag when launching:

```bash
aitraining tui --config my_training_config.yaml
```

## Trainer-Specific Parameters

The TUI automatically filters parameters based on the selected trainer:

- **Default**: Shows all general training parameters
- **SFT**: Includes supervised fine-tuning specific options
- **DPO/ORPO**: Shows preference optimization parameters like `model_ref` and `dpo_beta`
- **PPO**: Displays reinforcement learning parameters including reward model configuration
- **Reward**: Focuses on reward model training parameters

## Validation

The TUI provides several levels of validation:

1. **Type Validation**: Ensures numeric fields contain valid numbers
2. **JSON Validation**: Validates JSON syntax for complex parameters
3. **Required Fields**: Highlights missing required parameters
4. **Trainer-Specific**: Enforces trainer-specific requirements (e.g., PPO requires reward model)

Validation errors are displayed in the context panel with helpful messages.

## Dry Run Mode

Use dry run mode to test configurations without actually running training:

```bash
aitraining tui --dry-run
```

Or toggle it within the TUI with `Ctrl+D`.

In dry run mode:
- Commands are displayed but not executed
- Useful for testing configurations
- Helps understand what will be run

## Troubleshooting

### TTY Not Available

If you see an error about TTY not being available:

```
❌ AITraining TUI requires an interactive terminal (TTY).
```

This means you're trying to run the TUI in a non-interactive environment. The TUI requires a proper terminal and cannot run in:
- CI/CD pipelines
- Jupyter notebooks
- Non-terminal environments

**Solution**: Use the standard CLI interface instead:
```bash
aitraining llm --help
```

### Import Errors

If you see import errors for `textual` or `rich`:

```
❌ Failed to load TUI dependencies.
```

**Solution**: Install the required packages:
```bash
pip install textual rich
```

### Display Issues

If the TUI doesn't display correctly:

1. Ensure your terminal supports ANSI colors and Unicode
2. Try a different terminal emulator
3. Adjust terminal size (minimum 80x24 recommended)
4. Toggle theme with `Ctrl+T`

## Tips and Best Practices

1. **Start with a Template**: Save a working configuration as a template for future training runs
2. **Use Groups**: Navigate by parameter groups to quickly find related settings
3. **Check Validation**: Always check the context panel for validation errors before running
4. **Preview Commands**: Use `Ctrl+P` to see the exact command before execution
5. **Monitor Logs**: Switch to the Logs tab during training to monitor progress
6. **Dry Run First**: Test configurations with dry run mode before actual training

## Advanced Usage

### Programmatic Configuration

You can generate configuration files programmatically and load them in the TUI:

```python
import json

config = {
    "trainer": "sft",
    "parameters": {
        "model": "gpt2",
        "project_name": "automated-training",
        # ... other parameters
    }
}

with open("auto_config.json", "w") as f:
    json.dump(config, f, indent=2)
```

Then load in TUI:
```bash
aitraining tui --config auto_config.json
```

### Batch Processing

While the TUI is interactive, you can use saved configurations for batch processing:

```bash
# Create configurations in TUI
aitraining tui  # Save as config1.json, config2.json, etc.

# Run batch training with CLI
for config in config*.json; do
    aitraining llm --config "$config" --train
done
```

## Comparison with CLI

| Feature | TUI | CLI |
|---------|-----|-----|
| Parameter Discovery | ✅ Browse all options | ❌ Need to know flags |
| Validation | ✅ Real-time | ⚠️ On execution |
| Command Preview | ✅ Always visible | ❌ Manual construction |
| Configuration Management | ✅ Built-in save/load | ⚠️ Manual file editing |
| Scripting | ⚠️ Interactive only | ✅ Fully scriptable |
| CI/CD Compatible | ❌ Requires TTY | ✅ Works everywhere |

## Support

For issues, feature requests, or questions:
- GitHub Issues: [Report an issue](https://github.com/huggingface/autotrain-advanced/issues)
- Documentation: [AITraining Docs](https://huggingface.co/docs/autotrain)

## See Also

- [CLI Documentation](README.md) - Standard command-line interface
- [LLM Training Guide](../training/llm.md) - Detailed LLM training documentation
- [Parameter Reference](../reference/parameters.md) - Complete parameter documentation