# Text Classification Trainer

## Overview

The Text Classification trainer enables fine-tuning of transformer models for classifying text into predefined categories. It supports binary classification, multi-class classification, and multi-label classification tasks.

## Use Cases

- **Sentiment Analysis:** Classify text as positive, negative, or neutral
- **Spam Detection:** Identify spam vs. legitimate messages
- **Topic Classification:** Categorize documents by subject matter
- **Intent Recognition:** Determine user intent in conversational AI
- **Content Moderation:** Detect toxic, offensive, or inappropriate content
- **Language Detection:** Identify the language of text
- **Customer Feedback:** Categorize support tickets or reviews

## Supported Models

Any AutoModelForSequenceClassification compatible model:
- BERT, RoBERTa, DistilBERT, ALBERT
- XLNet, ELECTRA, DeBERTa
- XLM-RoBERTa (multilingual)
- Custom models from Hugging Face Hub

## Data Format

### CSV Format
```csv
text,target
"This movie is fantastic!",positive
"Terrible service, would not recommend.",negative
"The product arrived on time.",neutral
```

### JSON Format
```json
{"text": "Great product!", "target": "positive"}
{"text": "Not worth the price.", "target": "negative"}
```

### Multi-label Format
```csv
text,target
"This is a technical blog about AI and ethics.","technology|ethics"
"Breaking news about sports and politics.","sports|politics|news"
```

## Parameters

### Required Parameters
- `model`: Pre-trained model name or path
- `data_path`: Path to training data
- `text_column`: Name of text column (default: "text")
- `target_column`: Name of target column (default: "target")

### Training Parameters
- `lr`: Learning rate (default: 5e-5)
- `epochs`: Number of training epochs (default: 3)
- `batch_size`: Training batch size (default: 8)
- `warmup_ratio`: Warmup proportion (default: 0.1)
- `gradient_accumulation`: Gradient accumulation steps (default: 1)
- `optimizer`: Optimizer choice (default: "adamw_torch")
- `scheduler`: Learning rate scheduler (default: "linear")

### Advanced Parameters
- `max_seq_length`: Maximum sequence length (default: 128)
- `mixed_precision`: Enable mixed precision ("fp16", "bf16", None)
- `early_stopping`: Enable early stopping
- `early_stopping_patience`: Patience for early stopping (default: 5)
- `early_stopping_threshold`: Minimum improvement threshold (default: 0.01)

## Command Line Usage

### Basic Binary Classification
```bash
autotrain text-classification \
  --model bert-base-uncased \
  --data-path ./data \
  --text-column text \
  --target-column label \
  --output-dir ./output \
  --train
```

### Multi-class Classification
```bash
autotrain text-classification \
  --model roberta-base \
  --data-path ./news_data.csv \
  --text-column article \
  --target-column category \
  --max-seq-length 256 \
  --epochs 5 \
  --batch-size 16 \
  --lr 2e-5 \
  --train
```

### Multi-label Classification
```bash
autotrain text-classification \
  --model bert-base-multilingual-cased \
  --data-path ./multilabel_data.csv \
  --text-column text \
  --target-column labels \
  --max-seq-length 512 \
  --epochs 10 \
  --mixed-precision fp16 \
  --train
```

## Python API Usage

```python
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_classification import train

# Configure parameters
params = TextClassificationParams(
    model="distilbert-base-uncased",
    data_path="./sentiment_data",
    text_column="review",
    target_column="sentiment",
    train_split="train",
    valid_split="validation",
    max_seq_length=128,
    epochs=3,
    batch_size=32,
    lr=5e-5,
    warmup_ratio=0.1,
    gradient_accumulation=1,
    optimizer="adamw_torch",
    scheduler="cosine",
    seed=42,
    output_dir="./models/sentiment_classifier",
    push_to_hub=True,
    hub_model_id="my-org/sentiment-model"
)

# Run training
train(params)
```

## Data Preparation Tips

### 1. Balanced Dataset
Ensure balanced representation across classes:
```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df['target'].value_counts())

# Balance if needed
balanced_df = df.groupby('target').sample(n=min_samples, random_state=42)
```

### 2. Text Preprocessing
While models handle most preprocessing, consider:
- Removing excessive whitespace
- Handling special characters consistently
- Normalizing URLs, emails, mentions

### 3. Train/Validation Split
```python
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['target'], random_state=42
)
```

## Advanced Features

### 1. Class Weights
Handle imbalanced datasets:
```python
params = TextClassificationParams(
    # ... other params
    auto_weights=True  # Automatically compute class weights
)
```

### 2. Custom Metrics
Track additional metrics during training:
```python
params = TextClassificationParams(
    # ... other params
    valid_split="validation",
    logging_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    metric_for_best_model="f1",
    greater_is_better=True
)
```

### 3. Model Quantization
Reduce model size for deployment:
```python
params = TextClassificationParams(
    # ... other params
    quantization="int8",  # or "int4"
    mixed_precision="fp16"
)
```

## Evaluation Metrics

The trainer automatically computes:
- **Accuracy:** Overall correctness
- **Precision:** Correct positive predictions ratio
- **Recall:** True positives found ratio
- **F1 Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** For detailed error analysis

For multi-label:
- **Micro/Macro/Weighted F1:** Different averaging strategies
- **Hamming Loss:** Fraction of wrong labels
- **Subset Accuracy:** Exact match ratio

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Decrease `max_seq_length`
- Enable `gradient_accumulation`
- Use `mixed_precision="fp16"`
- Try a smaller model

### Poor Performance
- Increase `epochs`
- Adjust `lr` (try 2e-5 to 5e-5 range)
- Increase `max_seq_length` if text is truncated
- Check data quality and labeling consistency
- Try different pre-trained models
- Enable early stopping

### Slow Training
- Enable mixed precision training
- Reduce validation frequency
- Use a smaller model
- Enable gradient checkpointing

## Best Practices

1. **Start Simple:** Begin with default parameters and a small model
2. **Iterate:** Gradually increase complexity based on results
3. **Monitor:** Use validation metrics to detect overfitting
4. **Experiment:** Try different models and hyperparameters
5. **Document:** Keep track of experiments and results

## Example Projects

### Sentiment Analysis Pipeline
```python
# 1. Load and prepare data
import pandas as pd
df = pd.read_csv("reviews.csv")

# 2. Configure training
params = TextClassificationParams(
    model="distilbert-base-uncased",
    data_path="reviews.csv",
    text_column="review_text",
    target_column="rating_category",
    max_seq_length=256,
    epochs=5,
    batch_size=16,
    output_dir="./sentiment_model"
)

# 3. Train model
train(params)

# 4. Load and use model
from transformers import pipeline
classifier = pipeline("text-classification", model="./sentiment_model")
result = classifier("This product exceeded my expectations!")
```

## See Also

- [Token Classification](./TokenClassification.md) - For NER and token-level tasks
- [Text Regression](./TextRegression.md) - For continuous value prediction
- [Sentence Transformers](./SentenceTransformers.md) - For semantic similarity