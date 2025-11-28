# AITraining TUI (Terminal User Interface)

## ⚠️ EXPERIMENTAL FEATURE

**Status:** Under Active Development  
**Stability:** Experimental/Alpha  
**Recommended for:** Development and testing only  
**NOT recommended for:** Production use  

## Overview

The AITraining Portal TUI is an experimental interactive terminal interface for configuring and running AITraining jobs. It provides a visual way to configure parameters, select models and datasets, and monitor training progress.

## Current Status

### Working Features
- Basic parameter configuration
- Model and dataset selection via dropdowns
- Command preview
- Parameter groups navigation
- Trainer selection

### Known Issues
- TabbedContent widget is broken (tabs don't render content)
- Slow startup time in some environments
- Some visual elements may not render correctly
- Scrolling can be buggy in parameter lists
- Not all trainers have proper parameter filtering

### Not Yet Implemented
- Full catalog browsing
- Advanced parameter validation
- Training progress monitoring
- Result visualization

## Usage (Development Only)

```bash
# Launch the TUI
aitraining portal

# With options
aitraining portal --theme dark --dry-run
```

## Architecture

The TUI uses the Textual framework and consists of:
- **Left Panel:** Trainer selector and parameter groups
- **Center Panel:** Parameter form for selected group
- **Right Panel:** Command preview, model/dataset selectors, and logs

## Development

If you want to contribute or fix issues:

1. The main app is in `src/autotrain/cli/tui/app.py`
2. Custom widgets are in `src/autotrain/cli/tui/widgets/`
3. Styling is in `src/autotrain/cli/tui/theme.tcss`

## For Production Use

Please use the standard CLI instead:

```bash
aitraining llm --help
aitraining llm --model google/gemma-2-2b --data-path dataset --train
```

## Future Plans

This TUI is being developed as a more user-friendly interface for AITraining. Once stable, it will provide:
- Full parameter configuration with validation
- Real-time training monitoring
- Model evaluation and comparison
- Dataset preview and analysis

## Reporting Issues

If you encounter issues with the experimental TUI:
1. Note that this is expected as it's under development
2. Use the standard CLI for actual work
3. Report bugs with the `[TUI]` prefix in issue titles
