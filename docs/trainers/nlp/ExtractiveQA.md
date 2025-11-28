# Extractive Question Answering Trainer

## Overview

The Extractive Question Answering trainer fine-tunes transformer models to extract answer spans from given contexts. Unlike generative QA, this trainer identifies the exact text segment containing the answer within the provided passage.

## Use Cases

- **Reading Comprehension:** Extract answers from documents
- **Customer Support:** Find answers in knowledge bases
- **Information Retrieval:** Locate specific information in texts
- **Legal Document Analysis:** Extract relevant clauses
- **Medical Records:** Find specific patient information
- **FAQ Systems:** Automatic answer extraction
- **Educational Tools:** Reading comprehension assessment

## Supported Models

Any AutoModelForQuestionAnswering compatible model:
- BERT, RoBERTa, ALBERT (most common)
- DistilBERT (lightweight)
- XLM-RoBERTa (multilingual)
- DeBERTa (state-of-the-art)
- Longformer (long documents)

## Data Format

### SQuAD v1 Format (Answer Always Present)
```json
{
  "context": "The Eiffel Tower is located in Paris, France. It was built in 1889.",
  "question": "Where is the Eiffel Tower located?",
  "answers": {
    "text": ["Paris, France"],
    "answer_start": [31]
  }
}
```

### SQuAD v2 Format (May Have No Answer)
```json
{
  "context": "The Eiffel Tower was built in 1889.",
  "question": "Who designed the Statue of Liberty?",
  "answers": {
    "text": [],
    "answer_start": []
  },
  "is_impossible": true
}
```

### CSV Format
```csv
context,question,answer_text,answer_start
"Paris is the capital of France.",What is the capital of France?,Paris,0
"The event will be held on Monday.",When is the event?,Monday,26
```

## Parameters

### Required Parameters
- `model`: Pre-trained model name or path
- `data_path`: Path to training data
- `context_column`: Column containing context (default: "context")
- `question_column`: Column containing questions (default: "question")
- `answer_column`: Column containing answers (default: "answers")

### Training Parameters
- `lr`: Learning rate (default: 3e-5)
- `epochs`: Number of epochs (default: 3)
- `batch_size`: Batch size (default: 12)
- `max_seq_length`: Maximum sequence length (default: 384)
- `doc_stride`: Stride for splitting long documents (default: 128)
- `max_answer_length`: Maximum answer length (default: 30)

### Advanced Parameters
- `version_2`: Enable SQuAD v2 (no-answer questions)
- `n_best`: Number of best predictions to generate
- `null_score_diff_threshold`: Threshold for no-answer prediction
- `warmup_steps`: Number of warmup steps
- `gradient_accumulation`: Gradient accumulation steps

## Command Line Usage

### Basic Training
```bash
autotrain extractive-question-answering \
  --model bert-base-uncased \
  --data-path ./squad_data.json \
  --output-dir ./qa_model \
  --train
```

### SQuAD v2 with Custom Parameters
```bash
autotrain extractive-question-answering \
  --model roberta-base \
  --data-path ./squad_v2_data.json \
  --version-2 \
  --max-seq-length 512 \
  --doc-stride 128 \
  --epochs 5 \
  --batch-size 8 \
  --train
```

## Python API Usage

```python
from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
from autotrain.trainers.extractive_question_answering import train

# Configure parameters
params = ExtractiveQuestionAnsweringParams(
    model="bert-base-uncased",
    data_path="./qa_dataset",
    context_column="passage",
    question_column="query",
    answer_column="answer",
    max_seq_length=384,
    doc_stride=128,
    version_2=True,
    epochs=3,
    batch_size=16,
    lr=3e-5,
    output_dir="./qa_model"
)

# Train model
train(params)
```

## Data Preparation

### Converting to SQuAD Format
```python
import json

def create_squad_example(context, question, answer_text, answer_start):
    return {
        "context": context,
        "question": question,
        "answers": {
            "text": [answer_text],
            "answer_start": [answer_start]
        }
    }

# Create dataset
data = []
data.append(create_squad_example(
    "The company was founded in 2010 in San Francisco.",
    "When was the company founded?",
    "2010",
    28
))

# Save to file
with open("qa_data.json", "w") as f:
    json.dump(data, f, indent=2)
```

### Finding Answer Positions
```python
def find_answer_position(context, answer):
    """Find the character position of answer in context"""
    answer_start = context.lower().find(answer.lower())
    if answer_start == -1:
        # Answer not found exactly, may need fuzzy matching
        return None
    return answer_start

# Example
context = "The Eiffel Tower is 330 meters tall."
answer = "330 meters"
position = find_answer_position(context, answer)  # Returns 21
```

## Evaluation Metrics

- **Exact Match (EM):** Percentage of exact answer matches
- **F1 Score:** Token-level F1 between prediction and ground truth
- **HasAns EM/F1:** Metrics for answerable questions only
- **NoAns Accuracy:** Accuracy on unanswerable questions (v2)

## Best Practices

1. **Context Length:** Keep contexts concise but complete
2. **Answer Verification:** Ensure answers exist verbatim in context
3. **Question Quality:** Write clear, unambiguous questions
4. **Data Balance:** Mix different question types
5. **Validation Set:** Use proper train/val/test splits

## Troubleshooting

### Answer Not Found
- Verify answer exists exactly in context
- Check for whitespace/punctuation differences
- Consider lowercasing for matching

### Poor Performance
- Increase training epochs
- Try different base models
- Adjust max_seq_length and doc_stride
- Check data quality

### Out of Memory
- Reduce batch_size
- Decrease max_seq_length
- Use gradient accumulation
- Try smaller model

## Example Project

```python
# Complete QA System
import json
from transformers import pipeline

# 1. Prepare training data
training_data = [
    {
        "context": "AutoTrain is a tool for training models. It was created by Hugging Face.",
        "question": "Who created AutoTrain?",
        "answers": {"text": ["Hugging Face"], "answer_start": [61]}
    }
]

# 2. Save data
with open("train.json", "w") as f:
    json.dump(training_data, f)

# 3. Train model
from autotrain.trainers.extractive_question_answering import train
from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams

params = ExtractiveQuestionAnsweringParams(
    model="distilbert-base-uncased",
    data_path="train.json",
    epochs=3,
    output_dir="./my_qa_model"
)

train(params)

# 4. Use trained model
qa_pipeline = pipeline("question-answering", model="./my_qa_model")

result = qa_pipeline(
    question="Who created AutoTrain?",
    context="AutoTrain is a tool for training models. It was created by Hugging Face."
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['score']:.2f}")
```

## See Also

- [Seq2Seq](./Seq2Seq.md) - For generative QA
- [Text Classification](./TextClassification.md) - For question type classification
- [Token Classification](./TokenClassification.md) - For answer span labeling