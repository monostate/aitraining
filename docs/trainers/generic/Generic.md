# Generic Trainer

## Overview

The Generic trainer provides a flexible framework for running custom training scripts and workflows that don't fit into AutoTrain's predefined trainer categories. It allows you to use AutoTrain's infrastructure while maintaining complete control over the training process.

## Use Cases

- **Custom Model Architectures:** Train models not supported by standard trainers
- **Research Experiments:** Run experimental training procedures
- **Multi-Stage Training:** Complex workflows with multiple steps
- **Custom Frameworks:** Use frameworks beyond Hugging Face
- **Legacy Code Integration:** Wrap existing training scripts
- **Specialized Preprocessing:** Custom data pipelines
- **Hybrid Training:** Combine multiple training approaches

## How It Works

The Generic trainer:
1. Pulls your dataset from Hugging Face Hub (optional)
2. Sets up the environment with your requirements
3. Executes your custom training script
4. Manages outputs and logging

## Data Format

The Generic trainer doesn't enforce any specific data format. Your custom script handles all data loading and preprocessing.

### Common Patterns

**Custom CSV/JSON:**
```python
# Your script handles any format
data = pd.read_csv("custom_format.csv")
data = json.load(open("data.json"))
```

**Hugging Face Datasets:**
```python
from datasets import load_dataset
dataset = load_dataset("your-username/your-dataset")
```

**Custom Loaders:**
```python
# Implement any loading logic
data = load_custom_format("data.xyz")
```

## Parameters

### Required Parameters
- `data_path`: Path to data or Hugging Face dataset ID
- `custom_script`: Path to your training script
- `output_dir`: Where to save outputs

### Optional Parameters
- `requirements_file`: Path to requirements.txt
- `pip_packages`: List of packages to install
- `custom_command`: Command to execute (overrides script)
- `env_vars`: Environment variables to set
- `username`: Hugging Face username
- `token`: Hugging Face token for private repos

## Directory Structure

```
project/
├── train.py           # Your custom training script
├── requirements.txt   # Python dependencies
├── config.yaml       # Optional configuration
├── utils/            # Supporting modules
│   └── helpers.py
└── data/             # Local data (optional)
```

## Command Line Usage

### Basic Custom Script
```bash
autotrain generic \
  --custom-script ./train.py \
  --data-path ./data \
  --output-dir ./output \
  --train
```

### With Hugging Face Dataset
```bash
autotrain generic \
  --data-path username/dataset-name \
  --custom-script ./custom_trainer.py \
  --requirements-file ./requirements.txt \
  --username your-username \
  --token $HF_TOKEN \
  --train
```

### Custom Command Execution
```bash
autotrain generic \
  --custom-command "python train.py --epochs 10 --lr 0.001" \
  --data-path ./data \
  --pip-packages "torch torchvision torchaudio" \
  --output-dir ./results \
  --train
```

## Python API Usage

```python
from autotrain.trainers.generic.params import GenericParams
from autotrain.trainers.generic import run

# Configure parameters
params = GenericParams(
    data_path="./custom_data",
    custom_script="./my_trainer.py",
    requirements_file="./requirements.txt",
    output_dir="./outputs",
    env_vars={
        "CUDA_VISIBLE_DEVICES": "0,1",
        "WANDB_PROJECT": "my-project"
    }
)

# Run custom training
run(params)
```

## Custom Script Examples

### Basic Training Script
```python
# train.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    # Your custom training logic
    model = YourCustomModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load data
    dataset = YourCustomDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=32)

    # Training loop
    for epoch in range(args.epochs):
        for batch in dataloader:
            # Training step
            loss = train_step(model, batch, optimizer)
            print(f"Epoch {epoch}, Loss: {loss}")

    # Save model
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
```

### Multi-Stage Training
```python
# complex_training.py
import os
import json

def stage1_preprocessing():
    """Custom preprocessing stage"""
    print("Running preprocessing...")
    # Your preprocessing code

def stage2_training():
    """Main training stage"""
    print("Training model...")
    # Your training code

def stage3_evaluation():
    """Evaluation stage"""
    print("Evaluating model...")
    # Your evaluation code

def main():
    # Run multi-stage pipeline
    stage1_preprocessing()
    stage2_training()
    stage3_evaluation()

    # Save results
    results = {"status": "completed", "metrics": {...}}
    with open("results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
```

### Integration with External Frameworks
```python
# sklearn_trainer.py
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

def train():
    # Load data
    df = pd.read_csv("data.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    # Save model
    joblib.dump(model, "model.joblib")

    print("Training completed!")

if __name__ == "__main__":
    train()
```

## Requirements Management

### Using requirements.txt
```txt
torch>=2.0.0
transformers>=4.30.0
datasets
numpy
pandas
scikit-learn
wandb
```

### Dynamic Installation
```python
# In your script
import subprocess
import sys

def install_requirements():
    """Install packages dynamically"""
    packages = ["numpy", "pandas", "torch"]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_requirements()
```

## Environment Variables

Set environment variables for your script:

```python
params = GenericParams(
    env_vars={
        "CUDA_VISIBLE_DEVICES": "0,1",
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_API_KEY": "your-key",
        "CUSTOM_CONFIG": "production"
    }
)
```

Access in your script:
```python
import os

gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
config_mode = os.environ.get("CUSTOM_CONFIG", "debug")
```

## Best Practices

1. **Script Structure:** Keep scripts modular and well-organized
2. **Error Handling:** Include proper error handling and logging
3. **Checkpointing:** Save intermediate checkpoints
4. **Reproducibility:** Set random seeds and log configurations
5. **Documentation:** Document your custom script thoroughly
6. **Testing:** Test locally before running with AutoTrain

## Logging and Monitoring

### Basic Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting training...")
logger.info(f"Parameters: {params}")
```

### Integration with Tracking Tools
```python
# Weights & Biases
import wandb
wandb.init(project="my-project")
wandb.log({"loss": loss, "accuracy": accuracy})

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar("Loss/train", loss, epoch)

# MLflow
import mlflow
mlflow.start_run()
mlflow.log_metric("accuracy", accuracy)
```

## Troubleshooting

### Script Not Found
- Ensure script path is correct
- Check file permissions
- Use absolute paths when possible

### Missing Dependencies
- List all dependencies in requirements.txt
- Test installation locally first
- Use specific version numbers

### Out of Memory
- Monitor resource usage
- Implement batch processing
- Use gradient checkpointing

### No Output
- Add print statements for debugging
- Check output directory permissions
- Ensure script completes successfully

## Example Projects

### Custom Vision Model
```python
# custom_vision.py
import torch
import torchvision
from custom_model import MyVisionModel

def main():
    # Load custom dataset
    dataset = load_custom_dataset("./images")

    # Initialize custom model
    model = MyVisionModel(num_classes=100)

    # Custom training loop
    for epoch in range(100):
        train_epoch(model, dataset)

    # Save in custom format
    save_custom_format(model, "model.custom")

if __name__ == "__main__":
    main()
```

### Ensemble Training
```python
# ensemble_trainer.py
def train_ensemble():
    models = []

    # Train multiple models
    for i in range(5):
        model = train_single_model(seed=i)
        models.append(model)

    # Create ensemble
    ensemble = EnsembleModel(models)

    # Evaluate ensemble
    evaluate(ensemble)

    # Save ensemble
    save_ensemble(ensemble, "ensemble_model")

if __name__ == "__main__":
    train_ensemble()
```

## Advanced Features

### Distributed Training
```python
# distributed_train.py
import torch.distributed as dist

def setup_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank

def main():
    rank = setup_distributed()
    model = Model().to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model)
    # Training code
```

### Custom Callbacks
```python
class CustomCallback:
    def on_epoch_end(self, epoch, logs):
        # Custom logic
        if logs['loss'] < 0.01:
            return "stop_training"

    def on_train_end(self):
        # Cleanup or final processing
        pass
```

## See Also

- [Tabular Trainer](../tabular/Tabular.md) - For structured data with standard algorithms
- [CLM Trainers](../clm/README.md) - For language model training
- [Text Classification](../nlp/TextClassification.md) - For standard text classification