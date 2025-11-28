# Token Classification Trainer

## Overview

The Token Classification trainer enables fine-tuning of transformer models for labeling individual tokens (words or subwords) in text sequences. This is essential for Named Entity Recognition (NER), Part-of-Speech (POS) tagging, and other token-level classification tasks where each word in a sentence needs its own label.

## Use Cases

- **Named Entity Recognition (NER):** Identify and classify entities like persons, organizations, locations, dates, and monetary values
- **Part-of-Speech Tagging:** Label words by grammatical category (noun, verb, adjective, etc.)
- **Chunking:** Identify syntactic constituents in sentences (noun phrases, verb phrases, etc.)
- **Slot Filling:** Extract structured information from conversational text for dialogue systems
- **Medical Entity Extraction:** Identify diseases, medications, symptoms, and procedures in clinical text
- **Legal Document Analysis:** Extract contract terms, parties, dates, and obligations
- **Resume Parsing:** Extract skills, experience, education, and contact information
- **Product Attribute Extraction:** Identify product features, brands, and specifications from descriptions

## Supported Models

Any AutoModelForTokenClassification compatible model from Hugging Face Hub:
- **BERT family:** bert-base-cased, bert-large-cased
- **RoBERTa:** roberta-base, roberta-large
- **DistilBERT:** distilbert-base-cased
- **ALBERT:** albert-base-v2, albert-large-v2
- **ELECTRA:** google/electra-base-discriminator
- **DeBERTa:** microsoft/deberta-base, microsoft/deberta-v3-base
- **XLM-RoBERTa:** xlm-roberta-base (multilingual)
- **Domain-specific models:** BioBERT, ClinicalBERT, LegalBERT

**Note:** Use cased models (not uncased) for better NER performance, as capitalization provides important signals for entity recognition.

## Data Format

Token classification requires aligned sequences of tokens and their corresponding labels.

### Dataset Structure

The data must have two columns (configurable via `tokens_column` and `tags_column`):

```python
{
    "tokens": ["John", "Smith", "works", "at", "Google", "in", "New", "York"],
    "tags": [1, 1, 0, 0, 3, 0, 2, 2]  # Numeric labels
}
```

### Label Schema

Labels are typically encoded as integers. Common schemes include:

#### BIO Tagging (Beginning, Inside, Outside)
```
0: O (Outside)
1: B-PER (Beginning of Person)
2: I-PER (Inside Person)
3: B-ORG (Beginning of Organization)
4: I-ORG (Inside Organization)
5: B-LOC (Beginning of Location)
6: I-LOC (Inside Location)
```

#### IOB2 Format Example
```json
{
    "tokens": ["Steve", "Jobs", "founded", "Apple", "Inc", "."],
    "tags": [1, 2, 0, 3, 4, 0]
}
```

Explanation:
- "Steve" (1) = B-PER
- "Jobs" (2) = I-PER
- "founded" (0) = O
- "Apple" (3) = B-ORG
- "Inc" (4) = I-ORG
- "." (0) = O

### CSV Format

```csv
tokens,tags
"['John', 'Smith', 'works', 'at', 'Google']","[1, 1, 0, 0, 3]"
"['Mary', 'visited', 'Paris', 'last', 'year']","[1, 0, 2, 0, 0]"
```

### Hugging Face Dataset

```python
from datasets import Dataset

data = {
    "tokens": [
        ["EU", "rejects", "German", "call"],
        ["Peter", "Blackburn"]
    ],
    "tags": [
        [3, 0, 7, 0],  # B-ORG, O, B-MISC, O
        [1, 1]         # B-PER, I-PER
    ]
}

dataset = Dataset.from_dict(data)
```

## Parameters

### Required Parameters
- **`data_path`**: Path to training data (local path or Hugging Face dataset ID)
- **`model`**: Pre-trained model name or path (default: "bert-base-uncased")
- **`project_name`**: Output directory for saved model

### Data Parameters
- **`tokens_column`**: Name of the tokens column (default: "tokens")
- **`tags_column`**: Name of the tags/labels column (default: "tags")
- **`train_split`**: Name of training split (default: "train")
- **`valid_split`**: Name of validation split (default: None)
- **`max_samples`**: Limit dataset size for testing/debugging (default: None)

### Training Parameters
- **`lr`**: Learning rate (default: 5e-5)
  - Recommended range: 1e-5 to 5e-5
  - Larger models may benefit from lower learning rates
- **`epochs`**: Number of training epochs (default: 3)
  - Typical range: 3-10 epochs
- **`batch_size`**: Training batch size per device (default: 8)
  - Adjust based on available memory
- **`max_seq_length`**: Maximum sequence length in tokens (default: 128)
  - Longer sequences capture more context but use more memory
- **`warmup_ratio`**: Proportion of training for learning rate warmup (default: 0.1)
- **`gradient_accumulation`**: Number of steps to accumulate gradients (default: 1)
  - Use to simulate larger batch sizes
- **`optimizer`**: Optimizer choice (default: "adamw_torch")
  - Options: adamw_torch, adamw_hf, sgd, adafactor
- **`scheduler`**: Learning rate scheduler (default: "linear")
  - Options: linear, cosine, constant, polynomial

### Model Parameters
- **`weight_decay`**: Weight decay for regularization (default: 0.0)
- **`max_grad_norm`**: Maximum gradient norm for clipping (default: 1.0)
- **`seed`**: Random seed for reproducibility (default: 42)

### Performance Parameters
- **`mixed_precision`**: Enable mixed precision training (default: None)
  - Options: "fp16", "bf16", None
  - Significantly speeds up training on modern GPUs
- **`auto_find_batch_size`**: Automatically find optimal batch size (default: False)
- **`logging_steps`**: Steps between logging (default: -1 for auto)
- **`eval_strategy`**: When to evaluate (default: "epoch")
  - Options: "no", "steps", "epoch"

### Early Stopping Parameters
- **`early_stopping_patience`**: Epochs to wait for improvement (default: 5)
- **`early_stopping_threshold`**: Minimum improvement threshold (default: 0.01)

### Hub Parameters
- **`push_to_hub`**: Whether to push model to Hugging Face Hub (default: False)
- **`username`**: Hugging Face username for Hub uploads
- **`token`**: Hugging Face API token for authentication
- **`save_total_limit`**: Maximum number of checkpoints to save (default: 1)

### Logging Parameters
- **`log`**: Experiment tracking backend (default: "none")
  - Options: "none", "tensorboard", "wandb", "mlflow"

## Command Line Usage

### Basic NER Training

```bash
autotrain token-classification \
  --train \
  --model bert-base-cased \
  --data-path ./ner_data \
  --tokens-column tokens \
  --tags-column ner_tags \
  --project-name my-ner-model \
  --lr 5e-5 \
  --epochs 5 \
  --batch-size 16
```

### Training with Validation and Early Stopping

```bash
autotrain token-classification \
  --train \
  --model roberta-base \
  --data-path conll2003 \
  --train-split train \
  --valid-split validation \
  --project-name conll-ner \
  --max-seq-length 256 \
  --epochs 10 \
  --batch-size 8 \
  --lr 2e-5 \
  --early-stopping-patience 3 \
  --mixed-precision fp16
```

### Training with Custom Column Names

```bash
autotrain token-classification \
  --train \
  --model distilbert-base-cased \
  --data-path ./custom_data.csv \
  --tokens-column words \
  --tags-column labels \
  --project-name custom-ner \
  --epochs 5 \
  --batch-size 16
```

### Training and Pushing to Hub

```bash
autotrain token-classification \
  --train \
  --model xlm-roberta-base \
  --data-path ./multilingual_ner \
  --project-name multilingual-ner-model \
  --epochs 5 \
  --batch-size 8 \
  --push-to-hub \
  --username your-username \
  --token $HF_TOKEN
```

## Python API Usage

### Basic Training

```python
from autotrain.trainers.token_classification.params import TokenClassificationParams
from autotrain.trainers.token_classification.__main__ import train

# Configure parameters
params = TokenClassificationParams(
    data_path="conll2003",
    model="bert-base-cased",
    project_name="./ner-model",
    tokens_column="tokens",
    tags_column="ner_tags",
    train_split="train",
    valid_split="validation",
    lr=5e-5,
    epochs=5,
    batch_size=16,
    max_seq_length=128,
    seed=42
)

# Train model
train(params)
```

### Advanced Training with All Options

```python
from autotrain.trainers.token_classification.params import TokenClassificationParams
from autotrain.trainers.token_classification.__main__ import train

params = TokenClassificationParams(
    # Data configuration
    data_path="./medical_ner_data",
    model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    project_name="./biomedical-ner",

    # Column mapping
    tokens_column="tokens",
    tags_column="entity_tags",

    # Data splits
    train_split="train",
    valid_split="validation",

    # Training hyperparameters
    lr=3e-5,
    epochs=10,
    batch_size=8,
    max_seq_length=256,
    warmup_ratio=0.1,
    gradient_accumulation=2,

    # Optimizer and scheduler
    optimizer="adamw_torch",
    scheduler="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Performance optimization
    mixed_precision="fp16",
    auto_find_batch_size=False,

    # Evaluation and logging
    eval_strategy="epoch",
    logging_steps=50,
    save_total_limit=2,

    # Early stopping
    early_stopping_patience=3,
    early_stopping_threshold=0.001,

    # Reproducibility
    seed=42,

    # Hub integration
    push_to_hub=True,
    username="my-org",
    token="hf_xxxxx",

    # Experiment tracking
    log="tensorboard"
)

train(params)
```

### Loading and Using Trained Model

```python
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)

# Load model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("./ner-model")
tokenizer = AutoTokenizer.from_pretrained("./ner-model")

# Create NER pipeline
ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"  # Groups B- and I- tags
)

# Predict entities
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
entities = ner_pipeline(text)

for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']} ({entity['score']:.3f})")
```

Output:
```
Apple Inc.: ORG (0.998)
Steve Jobs: PER (0.996)
Cupertino: LOC (0.991)
California: LOC (0.995)
```

## Data Preparation Tips

### 1. Tokenization Alignment

Ensure your labels align with tokenized text. The trainer handles subword tokenization automatically:

```python
from datasets import Dataset

# Your pre-tokenized data
data = {
    "tokens": ["John", "Smith", "works", "at", "Google"],
    "tags": [1, 1, 0, 0, 3]  # B-PER, I-PER, O, O, B-ORG
}

dataset = Dataset.from_dict(data)
```

The trainer will:
1. Tokenize "Google" → ["Google"] or ["Goo", "##gle"]
2. Assign the same label to all subword tokens
3. Use -100 for special tokens (ignored in loss calculation)

### 2. Converting CoNLL Format

```python
def read_conll_file(file_path):
    """Read CoNLL format file into tokens and tags."""
    sentences = []
    sentence_tokens = []
    sentence_tags = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Empty line marks sentence boundary
                if sentence_tokens:
                    sentences.append({
                        "tokens": sentence_tokens,
                        "tags": sentence_tags
                    })
                    sentence_tokens = []
                    sentence_tags = []
            else:
                parts = line.split()
                sentence_tokens.append(parts[0])  # Token
                sentence_tags.append(label2id[parts[-1]])  # Tag

    return sentences

# Convert to dataset
from datasets import Dataset
data = read_conll_file("train.conll")
dataset = Dataset.from_list(data)
```

### 3. Label Encoding

```python
# Define label mapping
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# Convert string labels to IDs
def encode_labels(example):
    example["tags"] = [label2id[tag] for tag in example["tag_strings"]]
    return example

dataset = dataset.map(encode_labels)
```

### 4. Creating Validation Split

```python
from datasets import load_dataset, DatasetDict

dataset = load_dataset("your-dataset")

# Split training data
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

dataset_dict = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"]
})

# Save for training
dataset_dict.save_to_disk("./ner_data")
```

## Evaluation Metrics

The trainer automatically computes the following metrics using seqeval:

### Core Metrics
- **Precision:** Percentage of predicted entities that are correct
- **Recall:** Percentage of actual entities that were found
- **F1 Score:** Harmonic mean of precision and recall
- **Accuracy:** Token-level accuracy (less important for NER)

### Entity-Level Evaluation

Metrics are computed at the entity level (not token level):
- An entity is correct only if both boundaries and type match exactly
- Partial matches are counted as errors

Example:
```
Gold:     [B-PER I-PER] works at [B-ORG]
Predicted: [B-PER] works at [B-ORG I-ORG]

Result: 0/2 correct entities (50% on first entity doesn't count)
```

### Per-Class Metrics

The trainer also reports metrics for each entity type:
```
PER: Precision: 0.95, Recall: 0.92, F1: 0.93
ORG: Precision: 0.88, Recall: 0.85, F1: 0.86
LOC: Precision: 0.91, Recall: 0.89, F1: 0.90
```

## Troubleshooting

### Out of Memory Errors

**Symptoms:** CUDA out of memory, allocation errors

**Solutions:**
```python
params = TokenClassificationParams(
    # Reduce batch size
    batch_size=4,  # Down from 8 or 16

    # Reduce sequence length
    max_seq_length=128,  # Down from 256 or 512

    # Enable gradient accumulation
    gradient_accumulation=4,  # Effective batch size = 4 * 4 = 16

    # Enable mixed precision
    mixed_precision="fp16",  # or "bf16" for newer GPUs

    # Use smaller model
    model="distilbert-base-cased"  # Instead of bert-large
)
```

### Poor Entity Recognition

**Symptoms:** Low F1 scores, many missed entities

**Solutions:**

1. **Use Cased Models:**
```python
# Good for NER
model="bert-base-cased"

# Bad for NER (loses capitalization info)
model="bert-base-uncased"
```

2. **Increase Sequence Length:**
```python
# If entities appear in long contexts
max_seq_length=256  # or 512
```

3. **Adjust Learning Rate:**
```python
# Try different learning rates
lr=2e-5  # Lower for stability
lr=5e-5  # Higher for faster learning
```

4. **Train Longer:**
```python
epochs=10  # NER often needs more epochs than classification
```

5. **Use Domain-Specific Models:**
```python
# For medical text
model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# For scientific text
model="allenai/scibert_scivocab_cased"

# For legal text
model="nlpaueb/legal-bert-base-uncased"
```

### Label Misalignment

**Symptoms:** Training crashes with index errors or -100 label warnings

**Solutions:**

1. **Check Label Indices:**
```python
# Labels must start at 0 and be continuous
# Good: [0, 1, 2, 3, 4]
# Bad: [1, 2, 3, 4, 5] or [0, 1, 3, 4]
```

2. **Verify Token-Tag Alignment:**
```python
# Each token list must have same length as tags list
assert len(tokens) == len(tags), "Mismatched lengths!"
```

3. **Handle Special Cases:**
```python
# Ensure all examples have at least one token
dataset = dataset.filter(lambda x: len(x["tokens"]) > 0)
```

### Slow Training

**Solutions:**

```python
params = TokenClassificationParams(
    # Enable mixed precision
    mixed_precision="fp16",

    # Reduce logging frequency
    logging_steps=100,  # Log less often

    # Reduce evaluation frequency
    eval_strategy="epoch",  # Instead of "steps"

    # Use faster model
    model="distilbert-base-cased"  # 40% faster than BERT
)
```

## Best Practices

### 1. Model Selection

- **Start with BERT-base-cased** for general NER
- **Use domain-specific models** when available (BioBERT, LegalBERT, etc.)
- **Choose cased models** to preserve capitalization signals
- **Consider multilingual models** (XLM-RoBERTa) for non-English or mixed-language text

### 2. Data Quality

- **Consistent labeling:** Use clear annotation guidelines
- **Sufficient examples:** Aim for 100+ examples per entity type
- **Balanced distribution:** Avoid extreme class imbalance
- **Clean boundaries:** Ensure entity spans are correct
- **Quality over quantity:** Better to have fewer high-quality annotations

### 3. Hyperparameter Tuning

Start with these defaults:
```python
lr=5e-5
epochs=5
batch_size=16
max_seq_length=128
warmup_ratio=0.1
```

Then tune in this order:
1. **Learning rate:** Try 2e-5, 3e-5, 5e-5
2. **Epochs:** Increase if underfitting
3. **Batch size:** Adjust based on memory
4. **Sequence length:** Increase for longer contexts

### 4. Monitoring Training

```python
params = TokenClassificationParams(
    # Enable validation
    valid_split="validation",

    # Enable early stopping
    early_stopping_patience=3,
    early_stopping_threshold=0.001,

    # Track experiments
    log="tensorboard",  # or "wandb"

    # Regular evaluation
    eval_strategy="epoch",
    logging_steps=50
)
```

### 5. Production Deployment

```python
# After training, test inference speed
from transformers import pipeline
import time

ner = pipeline("token-classification", model="./ner-model", device=0)

text = "Your test text here"
start = time.time()
result = ner(text)
end = time.time()

print(f"Inference time: {(end-start)*1000:.2f}ms")

# For faster inference, use ONNX or quantization
```

## Advanced Techniques

### 1. Handling Nested Entities

For nested entities (e.g., "New York City" containing "New York"):

```python
# Use BIO scheme with entity types
# Or consider span-based models
```

### 2. Few-Shot Learning

When training data is limited:

```python
params = TokenClassificationParams(
    # Use pre-trained NER model as base
    model="dslim/bert-base-NER",

    # Lower learning rate for fine-tuning
    lr=1e-5,

    # More epochs for limited data
    epochs=20,

    # Strong regularization
    weight_decay=0.01
)
```

### 3. Multi-Task Learning

Train on multiple datasets:

```python
from datasets import concatenate_datasets

# Load multiple datasets
dataset1 = load_dataset("dataset1")["train"]
dataset2 = load_dataset("dataset2")["train"]

# Combine
combined = concatenate_datasets([dataset1, dataset2])
```

## Example: Building a Complete NER Pipeline

```python
from datasets import Dataset, DatasetDict
from autotrain.trainers.token_classification.params import TokenClassificationParams
from autotrain.trainers.token_classification.__main__ import train

# 1. Prepare data
train_data = {
    "tokens": [
        ["John", "Smith", "works", "at", "Google"],
        ["Apple", "Inc", "is", "based", "in", "Cupertino"],
        # ... more examples
    ],
    "tags": [
        [1, 1, 0, 0, 3],  # B-PER, I-PER, O, O, B-ORG
        [3, 3, 0, 0, 0, 2],  # B-ORG, I-ORG, O, O, O, B-LOC
        # ... more labels
    ]
}

val_data = {
    "tokens": [["Mary", "Johnson", "visited", "Microsoft"]],
    "tags": [[1, 1, 0, 3]]
}

# Create datasets
dataset = DatasetDict({
    "train": Dataset.from_dict(train_data),
    "validation": Dataset.from_dict(val_data)
})

# Save to disk
dataset.save_to_disk("./ner_dataset")

# 2. Configure training
params = TokenClassificationParams(
    data_path="./ner_dataset",
    model="bert-base-cased",
    project_name="./my-ner-model",
    train_split="train",
    valid_split="validation",
    lr=5e-5,
    epochs=5,
    batch_size=16,
    max_seq_length=128,
    mixed_precision="fp16",
    early_stopping_patience=3
)

# 3. Train
train(params)

# 4. Evaluate
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="./my-ner-model",
    aggregation_strategy="simple"
)

test_text = "Barack Obama was the 44th President of the United States."
entities = ner(test_text)

for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']} (score: {entity['score']:.3f})")
```

## See Also

- [Text Classification](./TextClassification.md) - For sentence-level classification
- [Seq2Seq](./Seq2Seq.md) - For text generation tasks
- [Sentence Transformers](./SentenceTransformers.md) - For semantic similarity
