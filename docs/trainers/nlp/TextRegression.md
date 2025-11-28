# Text Regression Trainer

## Overview

The Text Regression trainer fine-tunes transformer models to predict continuous numerical values from text input. Unlike classification which predicts discrete categories, regression outputs continuous scores, ratings, or measurements.

## Use Cases

- **Sentiment Intensity:** Predict sentiment score (0-1 or -1 to 1)
- **Review Rating:** Predict star ratings (1.0-5.0)
- **Text Similarity:** Compute similarity scores between texts
- **Readability Scoring:** Estimate text complexity (Flesch score)
- **Quality Assessment:** Score content quality (0-100)
- **Price Prediction:** Estimate price from product descriptions
- **Risk Scoring:** Calculate risk levels from reports
- **Engagement Prediction:** Predict likes, shares, or views

## Supported Models

Any AutoModelForSequenceClassification compatible model (configured for regression):
- BERT, RoBERTa, ALBERT, DistilBERT
- XLNet, ELECTRA, DeBERTa
- XLM-RoBERTa (multilingual)
- Domain-specific models from Hugging Face Hub

## Data Format

### CSV Format
```csv
text,score
"This product exceeded my expectations! Highly recommend.",4.8
"Average quality, nothing special.",2.5
"Terrible experience, would not buy again.",0.3
```

### JSON Format
```json
{"text": "Excellent service!", "score": 9.5}
{"text": "Could be better.", "score": 5.0}
{"text": "Very disappointed.", "score": 1.2}
```

### Multiple Targets
```csv
text,readability,sentiment,quality
"Clear and concise explanation.",8.5,0.7,9.0
"Confusing and poorly written.",3.2,-0.5,2.0
```

## Parameters

### Required Parameters
- `model`: Pre-trained model name or path
- `data_path`: Path to training data
- `text_column`: Name of text column (default: "text")
- `target_column`: Name of target column (default: "target")

### Training Parameters
- `lr`: Learning rate (default: 5e-5)
- `epochs`: Number of epochs (default: 3)
- `batch_size`: Batch size (default: 8)
- `warmup_ratio`: Warmup proportion (default: 0.1)
- `gradient_accumulation`: Gradient accumulation steps (default: 1)
- `optimizer`: Optimizer choice (default: "adamw_torch")
- `scheduler`: Learning rate scheduler (default: "linear")

### Advanced Parameters
- `max_seq_length`: Maximum sequence length (default: 128)
- `loss_function`: Loss function (mse, mae, huber)
- `mixed_precision`: Enable mixed precision training
- `early_stopping`: Enable early stopping
- `metric`: Evaluation metric (rmse, mae, r2)

## Command Line Usage

### Basic Training
```bash
autotrain text-regression \
  --model bert-base-uncased \
  --data-path ./ratings_data.csv \
  --text-column review \
  --target-column rating \
  --output-dir ./rating_model \
  --train
```

### Advanced Configuration
```bash
autotrain text-regression \
  --model roberta-base \
  --data-path ./sentiment_scores.csv \
  --text-column text \
  --target-column sentiment_score \
  --max-seq-length 256 \
  --epochs 10 \
  --batch-size 16 \
  --lr 2e-5 \
  --loss-function huber \
  --metric rmse \
  --mixed-precision fp16 \
  --train
```

## Python API Usage

```python
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.text_regression import train

# Configure parameters
params = TextRegressionParams(
    model="distilbert-base-uncased",
    data_path="./review_ratings",
    text_column="review_text",
    target_column="rating_score",
    train_split="train",
    valid_split="validation",
    max_seq_length=128,
    epochs=5,
    batch_size=32,
    lr=5e-5,
    loss_function="mse",
    warmup_ratio=0.1,
    output_dir="./models/rating_predictor"
)

# Train model
train(params)
```

## Data Preparation

### Normalizing Targets
```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("data.csv")

# Normalize targets to [0, 1] range
scaler = MinMaxScaler()
df['normalized_score'] = scaler.fit_transform(df[['score']])

# Save scaler for inference
import joblib
joblib.dump(scaler, 'scaler.pkl')

# Save normalized data
df.to_csv("normalized_data.csv", index=False)
```

### Train/Validation Split
```python
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42
)

# Ensure similar target distribution
print(f"Train mean: {train_df['target'].mean():.3f}")
print(f"Val mean: {val_df['target'].mean():.3f}")
```

### Handling Outliers
```python
# Remove extreme outliers
q1 = df['target'].quantile(0.01)
q99 = df['target'].quantile(0.99)
df_clean = df[(df['target'] >= q1) & (df['target'] <= q99)]

# Or cap outliers
df['target_capped'] = df['target'].clip(lower=q1, upper=q99)
```

## Loss Functions

### Mean Squared Error (MSE)
- **Use when:** Standard regression, balanced data
- **Properties:** Penalizes large errors heavily

### Mean Absolute Error (MAE)
- **Use when:** Outliers present, robust predictions needed
- **Properties:** Linear penalty, outlier resistant

### Huber Loss
- **Use when:** Some outliers, want balance
- **Properties:** MSE for small errors, MAE for large

### Custom Loss
```python
# Example: Weighted loss for different ranges
def custom_loss(predictions, targets):
    errors = predictions - targets
    weights = torch.where(targets > 7, 2.0, 1.0)  # Weight high scores more
    return (weights * errors ** 2).mean()
```

## Evaluation Metrics

- **RMSE:** Root Mean Squared Error (standard)
- **MAE:** Mean Absolute Error (interpretable)
- **R² Score:** Variance explained (0-1)
- **Pearson Correlation:** Linear relationship
- **Spearman Correlation:** Rank correlation

## Best Practices

1. **Target Scaling:** Normalize targets to similar range as model outputs
2. **Text Length:** Consider impact of text length on predictions
3. **Data Balance:** Ensure good coverage of target range
4. **Validation:** Use proper metrics for continuous values
5. **Baseline:** Compare against simple baselines (mean, median)

## Troubleshooting

### Predictions All Similar
- Check target distribution
- Verify loss function choice
- Increase model complexity
- Add more diverse training data

### High Variance in Predictions
- Reduce learning rate
- Add regularization
- Increase batch size
- Use gradient clipping

### Poor Correlation
- Try different pre-trained models
- Increase sequence length
- Clean text data
- Check for label noise

## Example Projects

### Review Rating Predictor
```python
from autotrain.trainers.text_regression import train
from autotrain.trainers.text_regression.params import TextRegressionParams
import pandas as pd

# Prepare data
df = pd.read_csv("reviews.csv")
df['rating_normalized'] = df['rating'] / 5.0  # Normalize 1-5 to 0-1

# Configure training
params = TextRegressionParams(
    model="roberta-base",
    data_path="reviews.csv",
    text_column="review_text",
    target_column="rating_normalized",
    max_seq_length=256,
    epochs=10,
    batch_size=16,
    loss_function="huber",
    output_dir="./rating_model"
)

# Train
model = train(params)

# Inference
from transformers import pipeline

# Load model
predictor = pipeline("text-classification", model="./rating_model")

# Predict
text = "This product is amazing! Best purchase ever."
result = predictor(text)
rating = result[0]['score'] * 5.0  # Denormalize
print(f"Predicted rating: {rating:.1f}/5.0")
```

### Sentiment Intensity Scorer
```python
# Train sentiment intensity model (0-1 scale)
params = TextRegressionParams(
    model="distilbert-base-uncased-finetuned-sst-2-english",
    data_path="./sentiment_intensity.csv",
    text_column="text",
    target_column="intensity",  # 0 (very negative) to 1 (very positive)
    epochs=5,
    output_dir="./sentiment_scorer"
)

model = train(params)

# Use for ranking
texts = ["Great!", "Good", "OK", "Bad", "Terrible!"]
scores = [predictor(t)[0]['score'] for t in texts]
ranked = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)
```

## Advanced Techniques

### Multi-Target Regression
```python
# Predict multiple scores simultaneously
params = TextRegressionParams(
    target_columns=["quality", "readability", "relevance"],
    loss_function="mse",
    # ... other params
)
```

### Ensemble Predictions
```python
# Train multiple models and average
models = []
for seed in [42, 123, 456]:
    params.seed = seed
    model = train(params)
    models.append(model)

# Ensemble prediction
def predict_ensemble(text):
    predictions = [m.predict(text) for m in models]
    return np.mean(predictions)
```

### Fine-tuning on Domain Data
```python
# Two-stage training
# Stage 1: General data
params_general = TextRegressionParams(
    model="bert-base-uncased",
    data_path="./general_scores.csv",
    output_dir="./general_model"
)
train(params_general)

# Stage 2: Domain-specific
params_domain = TextRegressionParams(
    model="./general_model",  # Start from general model
    data_path="./domain_scores.csv",
    lr=2e-5,  # Lower learning rate
    output_dir="./domain_model"
)
train(params_domain)
```

## See Also

- [Text Classification](./TextClassification.md) - For discrete categories
- [Sentence Transformers](./SentenceTransformers.md) - For similarity scores
- [Seq2Seq](./Seq2Seq.md) - For text generation with scores