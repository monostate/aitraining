# Tabular Data Trainer

## Overview

The Tabular trainer provides state-of-the-art machine learning algorithms for structured/tabular data. It supports both classification and regression tasks with automatic hyperparameter optimization using Optuna.

## Use Cases

### Classification
- **Customer Churn Prediction:** Identify customers likely to cancel subscriptions
- **Credit Risk Assessment:** Evaluate loan default probability
- **Fraud Detection:** Identify fraudulent transactions
- **Disease Diagnosis:** Predict medical conditions from lab results
- **Marketing Response:** Predict customer response to campaigns
- **Quality Control:** Classify product defects

### Regression
- **Sales Forecasting:** Predict future sales volumes
- **Price Prediction:** Estimate real estate or product prices
- **Demand Forecasting:** Predict inventory requirements
- **Risk Scoring:** Calculate continuous risk scores
- **Performance Metrics:** Predict KPIs and business metrics

## Supported Algorithms

### Tree-based Models
- **XGBoost:** Gradient boosting with regularization
- **LightGBM:** Fast gradient boosting
- **CatBoost:** Handles categorical features natively
- **Random Forest:** Ensemble of decision trees
- **Extra Trees:** Extremely randomized trees

### Neural Networks
- **TabNet:** Attention-based deep learning for tabular data
- **Neural Network:** Custom MLP architectures

### Linear Models
- **Logistic/Linear Regression:** Baseline models
- **SVM:** Support Vector Machines

## Data Format

### CSV Format
```csv
feature1,feature2,feature3,target
1.5,23,high,0
2.3,45,medium,1
3.1,67,low,0
```

### Required Structure
- One row per sample
- Features in columns
- Target column(s) specified
- Headers required

### Multi-label Classification
```csv
feature1,feature2,label1,label2,label3
1.5,23,1,0,1
2.3,45,0,1,1
```

## Parameters

### Required Parameters
- `data_path`: Path to training data
- `model`: Algorithm choice (xgboost, lightgbm, catboost, etc.)
- `target_columns`: List of target column names
- `task`: Task type (classification or regression)

### Data Parameters
- `id_column`: Column to use as ID (optional)
- `train_split`: Training split name (default: "train")
- `valid_split`: Validation split name (optional)
- `categorical_columns`: List of categorical feature names
- `numerical_columns`: List of numerical feature names
- `datetime_columns`: List of datetime columns

### Training Parameters
- `num_trials`: Number of Optuna trials (default: 100)
- `time_limit`: Time limit for optimization in seconds
- `seed`: Random seed (default: 42)

### Model-specific Parameters
```python
# XGBoost
xgboost_params = {
    "n_estimators": [100, 500],
    "max_depth": [3, 10],
    "learning_rate": [0.01, 0.3],
    "subsample": [0.5, 1.0],
    "colsample_bytree": [0.5, 1.0]
}

# LightGBM
lightgbm_params = {
    "num_leaves": [31, 127],
    "learning_rate": [0.01, 0.3],
    "feature_fraction": [0.5, 1.0],
    "bagging_fraction": [0.5, 1.0]
}

# CatBoost
catboost_params = {
    "iterations": [100, 1000],
    "learning_rate": [0.01, 0.3],
    "depth": [4, 10],
    "l2_leaf_reg": [1, 10]
}
```

## Command Line Usage

### Basic Classification
```bash
autotrain tabular \
  --model xgboost \
  --data-path ./data.csv \
  --target-columns target \
  --task classification \
  --output-dir ./output \
  --train
```

### Multi-class Classification with Validation
```bash
autotrain tabular \
  --model lightgbm \
  --data-path ./train.csv \
  --valid-split ./valid.csv \
  --target-columns category \
  --categorical-columns "color,size,brand" \
  --numerical-columns "price,weight,rating" \
  --task classification \
  --num-trials 200 \
  --time-limit 3600 \
  --train
```

### Regression with Feature Engineering
```bash
autotrain tabular \
  --model catboost \
  --data-path ./sales_data.csv \
  --target-columns sales_amount \
  --datetime-columns "date" \
  --categorical-columns "store_id,product_category" \
  --task regression \
  --num-trials 150 \
  --train
```

## Python API Usage

```python
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.tabular import train

# Configure parameters
params = TabularParams(
    data_path="./customer_data.csv",
    model="xgboost",
    task="classification",
    target_columns=["churn"],
    categorical_columns=["plan_type", "region", "payment_method"],
    numerical_columns=["tenure", "monthly_charges", "total_charges"],
    train_split="train",
    valid_split="validation",
    num_trials=100,
    time_limit=1800,
    seed=42,
    output_dir="./models/churn_predictor"
)

# Run training
model = train(params)
```

## Feature Engineering

### Automatic Feature Processing
The trainer automatically handles:
- **Categorical Encoding:** Label encoding or one-hot encoding
- **Missing Values:** Imputation strategies by model
- **Scaling:** Normalization when needed
- **Datetime Features:** Extract year, month, day, weekday, etc.

### Manual Feature Engineering
```python
import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Create interaction features
df['feature_product'] = df['feature1'] * df['feature2']

# Binning continuous variables
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100])

# Aggregate features
df['total_purchases'] = df[['purchase1', 'purchase2', 'purchase3']].sum(axis=1)

# Save processed data
df.to_csv("processed_data.csv", index=False)
```

## Hyperparameter Optimization

### Optuna Integration
The trainer uses Optuna for automatic hyperparameter tuning:

```python
# Optimization happens automatically
# Best parameters are selected based on validation performance
# Trial history is saved for analysis

# Access optimization results
import optuna

study = optuna.load_study(
    study_name="autotrain_study",
    storage="sqlite:///optuna.db"
)

# Best parameters
print(study.best_params)

# Optimization history
print(study.trials_dataframe())
```

### Custom Parameter Ranges
```python
params = TabularParams(
    # ... other params
    custom_params={
        "n_estimators": [50, 1000],
        "max_depth": [3, 15],
        "min_samples_split": [2, 20]
    }
)
```

## Evaluation Metrics

### Classification Metrics
- **Accuracy:** Overall correctness
- **Precision/Recall/F1:** Per-class performance
- **AUC-ROC:** Discrimination ability
- **Log Loss:** Probabilistic accuracy
- **Confusion Matrix:** Detailed error analysis

### Regression Metrics
- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **R²:** Coefficient of determination
- **MAPE:** Mean Absolute Percentage Error

## Best Practices

### 1. Algorithm Selection
- **XGBoost:** General-purpose, good default choice
- **LightGBM:** Large datasets, faster training
- **CatBoost:** Many categorical features
- **Random Forest:** Interpretability needed
- **TabNet:** Complex patterns, deep learning

### 2. Feature Engineering
- Create domain-specific features
- Handle missing values appropriately
- Consider feature interactions
- Remove redundant features

### 3. Validation Strategy
- Use proper train/validation/test splits
- Consider time-based splits for temporal data
- Stratify splits for imbalanced datasets

### 4. Hyperparameter Tuning
- Start with fewer trials for quick iteration
- Increase trials for final model
- Set reasonable time limits
- Monitor overfitting

## Troubleshooting

### Memory Issues
- Reduce dataset size for initial experiments
- Use LightGBM for large datasets
- Enable categorical feature support in CatBoost
- Sample data for hyperparameter tuning

### Poor Performance
- Check for data leakage
- Verify feature engineering
- Increase number of trials
- Try different algorithms
- Check class imbalance

### Slow Training
- Reduce number of trials
- Set time limit
- Use early stopping
- Parallel trial execution

## Example Projects

### Customer Churn Prediction
```python
from autotrain.trainers.tabular import train
from autotrain.trainers.tabular.params import TabularParams
import pandas as pd

# Prepare data
df = pd.read_csv("telecom_churn.csv")

# Feature engineering
df['tenure_per_charge'] = df['tenure'] / (df['monthly_charges'] + 1)
df['total_services'] = df[service_columns].sum(axis=1)

# Configure training
params = TabularParams(
    data_path="telecom_churn.csv",
    model="catboost",  # Good with categorical features
    task="classification",
    target_columns=["churn"],
    categorical_columns=[
        "gender", "partner", "dependents",
        "phone_service", "internet_service"
    ],
    numerical_columns=[
        "tenure", "monthly_charges",
        "total_charges", "tenure_per_charge"
    ],
    num_trials=200,
    seed=42,
    output_dir="./churn_model"
)

# Train model
model = train(params)

# Make predictions
test_df = pd.read_csv("test_data.csv")
predictions = model.predict(test_df)
```

### Sales Forecasting
```python
params = TabularParams(
    data_path="sales_history.csv",
    model="lightgbm",
    task="regression",
    target_columns=["sales_amount"],
    datetime_columns=["date"],
    categorical_columns=["store_id", "product_id"],
    numerical_columns=[
        "price", "promotion", "temperature",
        "holiday", "weekend"
    ],
    num_trials=150,
    time_limit=7200,
    output_dir="./sales_model"
)

model = train(params)
```

## Advanced Features

### Multi-output Prediction
```python
# Predict multiple targets simultaneously
params = TabularParams(
    target_columns=["sales", "profit", "units_sold"],
    task="regression"
    # ... other params
)
```

### Custom Metrics
```python
from sklearn.metrics import make_scorer

def custom_metric(y_true, y_pred):
    # Custom business metric
    return business_value(y_true, y_pred)

params = TabularParams(
    # ... other params
    custom_metric=make_scorer(custom_metric, greater_is_better=True)
)
```

## See Also

- [Text Classification](../nlp/TextClassification.md) - For text-based classification
- [Image Classification](../vision/ImageClassification.md) - For image data
- [Generic Trainer](../generic/Generic.md) - For custom ML pipelines