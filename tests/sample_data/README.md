# Sample Test Data

This directory contains minimal sample data for testing various AutoTrain trainers.

## Files

- `train.jsonl` - Training data in chat format for LLM SFT training
- `validation.jsonl` - Validation data in chat format
- `train_dpo.jsonl` - Training data for DPO/ORPO with prompt/chosen/rejected columns
- `train.csv` - Training data for text classification with text/label columns

## Data Formats

### SFT Training (train.jsonl)
```json
{"text": "[{\"role\": \"user\", \"content\": \"...\"}, {\"role\": \"assistant\", \"content\": \"...\"}]"}
```

### DPO/ORPO Training (train_dpo.jsonl)
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

### Text Classification (train.csv)
```csv
text,label
"Sample text",label_name
```

## Note
This is minimal test data intended only for unit testing. Real training would require much larger datasets.