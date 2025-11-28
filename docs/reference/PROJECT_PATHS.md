# AITraining Project Path Management

## Overview
AITraining automatically manages project output directories to keep training artifacts organized and separate from the server directory.

## Default Behavior

When you specify a project name without a full path, AITraining will automatically create the project in a dedicated `trainings` directory:

```bash
# If you run:
aitraining llm --project-name my-model --model gpt2 ...

# Your project will be created at:
../trainings/my-model/
```

The `trainings` directory is created at the same level as the AITraining server directory:

```
parent-directory/
├── aitraining/     # Server runs here
└── trainings/              # All projects created here
    ├── my-model/
    ├── test-orpo/
    └── experiment-1/
```

## Using Absolute Paths

If you prefer to specify exactly where your project should be saved, you can provide an absolute path:

```bash
# This will save to your specified location
aitraining llm --project-name /home/user/my-projects/custom-model --model gpt2 ...
```

Absolute paths are always respected and used as-is.

## Environment Variable Override

You can customize the base directory for all projects using the `AUTOTRAIN_PROJECTS_DIR` environment variable:

```bash
# Set custom base directory
export AUTOTRAIN_PROJECTS_DIR=/data/ml-experiments

# Now projects will be created in your custom location
aitraining llm --project-name my-model --model gpt2 ...
# Creates: /data/ml-experiments/my-model/
```

## Path Resolution Examples

| Input | Environment Variable | Result |
|-------|---------------------|---------|
| `my-model` | Not set | `../trainings/my-model` |
| `my-model` | `/data/ml` | `/data/ml/my-model` |
| `/home/user/model` | Any | `/home/user/model` (unchanged) |
| `test-1` | Not set | `../trainings/test-1` |

## Benefits

1. **Clean Server Directory**: Training artifacts are kept separate from the server installation
2. **Organized Structure**: All training runs are in one place
3. **Flexible**: Use absolute paths or environment variables for custom locations
4. **Backward Compatible**: Existing scripts using absolute paths continue to work

## Migration from Previous Versions

If you have existing projects in the server directory from previous versions, you can:

1. Move them to the new `trainings` directory:
   ```bash
   mv aitraining/test-* ../trainings/
   ```

2. Or specify their absolute paths when referencing them:
   ```bash
   aitraining llm --project-name /path/to/aitraining/existing-project ...
   ```

## API Usage

When using the AITraining API, the same rules apply:

```python
from autotrain.trainers.clm.params import LLMTrainingParams

# Relative name - will be normalized
config = LLMTrainingParams(
    project_name="my-api-model",  # -> ../trainings/my-api-model
    ...
)

# Absolute path - used as-is
config = LLMTrainingParams(
    project_name="/data/models/production-model",  # -> /data/models/production-model
    ...
)
```

## Troubleshooting

**Q: My projects are still being created in the server directory**
A: Make sure you're using the latest version of AITraining. Check that the path normalization is active by looking for the log message: `"Project path normalized to: ..."`

**Q: I want all projects in my home directory**
A: Set the environment variable: `export AUTOTRAIN_PROJECTS_DIR=~/autotrain-projects`

**Q: Can I use relative paths like `./my-model` or `../other/model`?**
A: These will be treated as relative names and placed in the trainings directory. Use absolute paths if you need specific locations.