# Seq2Seq Trainer

## Overview

The Seq2Seq (Sequence-to-Sequence) trainer enables fine-tuning of encoder-decoder transformer models for text generation tasks. These models excel at transforming input text into output text, making them ideal for translation, summarization, question answering, and other text-to-text generation tasks.

## Use Cases

- **Machine Translation:** Translate text between languages
- **Text Summarization:** Generate concise summaries of long documents
- **Question Answering:** Generate answers from context passages
- **Paraphrasing:** Rephrase text while preserving meaning
- **Code Generation:** Generate code from natural language descriptions
- **Data Augmentation:** Create synthetic training examples
- **Text Simplification:** Convert complex text to simpler versions
- **Dialogue Generation:** Generate conversational responses
- **Grammar Correction:** Fix grammatical errors in text
- **Title Generation:** Create titles or headlines from content
- **Email/Letter Writing:** Generate formal or informal correspondence
- **SQL Query Generation:** Convert natural language to SQL

## Supported Models

Any AutoModelForSeq2SeqLM compatible model from Hugging Face Hub:

### T5 Family (Text-to-Text Transfer Transformer)
- **google/flan-t5-small** (80M parameters) - Fast, efficient
- **google/flan-t5-base** (250M parameters) - Default, balanced
- **google/flan-t5-large** (780M parameters) - Higher quality
- **google/flan-t5-xl** (3B parameters) - Best quality
- **google/t5-v1_1-base** - Improved T5 variant

### BART Family (Bidirectional and Auto-Regressive Transformers)
- **facebook/bart-base** - Good for summarization
- **facebook/bart-large** - High-quality generation
- **facebook/bart-large-cnn** - Pre-trained on CNN/DailyMail

### mBART (Multilingual BART)
- **facebook/mbart-large-50** - 50 languages
- **facebook/mbart-large-cc25** - 25 languages

### Other Models
- **google/pegasus-xsum** - Extractive summarization
- **google/pegasus-cnn_dailymail** - Abstractive summarization
- **Helsinki-NLP/opus-mt-\*** - Translation models
- **microsoft/GODEL-v1_1-base-seq2seq** - Dialogue generation

## Data Format

Seq2Seq requires paired input and output text sequences.

### Dataset Structure

```python
{
    "text": "Input text to transform",
    "target": "Expected output text"
}
```

### CSV Format

```csv
text,target
"Translate to French: Hello, how are you?","Bonjour, comment allez-vous?"
"Summarize: The quick brown fox...","A summary of the text"
"Question: What is AI? Context: Artificial intelligence...","AI is a field of computer science..."
```

### JSON Lines Format

```json
{"text": "translate English to German: Hello", "target": "Hallo"}
{"text": "summarize: Long article text...", "target": "Brief summary"}
{"text": "answer question: What is...", "target": "The answer is..."}
```

### Task Prefixes

For models like T5/FLAN-T5, include task prefixes in input:

```python
# Translation
"translate English to French: Hello world"

# Summarization
"summarize: Long article text here..."

# Question Answering
"question: What is X? context: Information about X..."

# Paraphrasing
"paraphrase: Original sentence here"
```

## Parameters

### Required Parameters
- **`data_path`**: Path to training data (local or Hugging Face dataset)
- **`model`**: Pre-trained model name (default: "google/flan-t5-base")
- **`project_name`**: Output directory for saved model

### Data Parameters
- **`text_column`**: Input text column name (default: "text")
- **`target_column`**: Output text column name (default: "target")
- **`train_split`**: Training split name (default: "train")
- **`valid_split`**: Validation split name (default: None)
- **`max_samples`**: Limit dataset size for testing (default: None)

### Training Parameters
- **`lr`**: Learning rate (default: 5e-5)
  - T5: 1e-4 to 3e-4
  - BART: 3e-5 to 5e-5
- **`epochs`**: Number of training epochs (default: 3)
- **`batch_size`**: Training batch size (default: 2)
  - Seq2seq is memory-intensive; use smaller batches
- **`max_seq_length`**: Maximum input sequence length (default: 128)
- **`max_target_length`**: Maximum output sequence length (default: 128)
- **`warmup_ratio`**: Warmup proportion (default: 0.1)
- **`gradient_accumulation`**: Gradient accumulation steps (default: 1)
  - Increase to simulate larger batch sizes
- **`optimizer`**: Optimizer (default: "adamw_torch")
- **`scheduler`**: LR scheduler (default: "linear")

### Model Parameters
- **`weight_decay`**: Weight decay (default: 0.0)
- **`max_grad_norm`**: Gradient clipping (default: 1.0)
- **`seed`**: Random seed (default: 42)

### PEFT (Parameter-Efficient Fine-Tuning) Parameters
- **`peft`**: Enable LoRA fine-tuning (default: False)
  - Drastically reduces memory usage and training time
- **`quantization`**: Quantization mode (default: "int8")
  - Options: "int4", "int8", None
  - Reduces memory footprint
- **`lora_r`**: LoRA rank (default: 16)
  - Higher = more parameters = better quality but slower
- **`lora_alpha`**: LoRA alpha (default: 32)
  - Scaling factor, typically 2x lora_r
- **`lora_dropout`**: LoRA dropout (default: 0.05)
- **`target_modules`**: Modules to apply LoRA (default: "all-linear")

### Performance Parameters
- **`mixed_precision`**: Mixed precision training (default: None)
  - Options: "fp16", "bf16", None
- **`auto_find_batch_size`**: Auto-find batch size (default: False)
- **`logging_steps`**: Logging frequency (default: -1 for auto)
- **`eval_strategy`**: Evaluation strategy (default: "epoch")

### Early Stopping
- **`early_stopping_patience`**: Epochs to wait (default: 5)
- **`early_stopping_threshold`**: Minimum improvement (default: 0.01)

### Hub Parameters
- **`push_to_hub`**: Push to Hugging Face Hub (default: False)
- **`username`**: Hugging Face username
- **`token`**: Hugging Face API token
- **`save_total_limit`**: Max checkpoints to save (default: 1)

### Logging
- **`log`**: Experiment tracking (default: "none")
  - Options: "tensorboard", "wandb", "mlflow"

## Command Line Usage

### Basic Summarization

```bash
autotrain seq2seq \
  --train \
  --model google/flan-t5-base \
  --data-path ./summarization_data \
  --text-column article \
  --target-column summary \
  --project-name my-summarizer \
  --max-seq-length 512 \
  --max-target-length 128 \
  --epochs 3 \
  --batch-size 4 \
  --lr 1e-4
```

### Translation with LoRA

```bash
autotrain seq2seq \
  --train \
  --model google/flan-t5-large \
  --data-path ./translation_data \
  --text-column source \
  --target-column target \
  --project-name en-fr-translator \
  --peft \
  --quantization int8 \
  --lora-r 16 \
  --lora-alpha 32 \
  --epochs 5 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --mixed-precision bf16
```

### Question Answering

```bash
autotrain seq2seq \
  --train \
  --model google/flan-t5-base \
  --data-path squad_v2 \
  --train-split train \
  --valid-split validation \
  --project-name qa-model \
  --max-seq-length 512 \
  --max-target-length 64 \
  --epochs 3 \
  --batch-size 4 \
  --early-stopping-patience 3
```

### Push to Hub

```bash
autotrain seq2seq \
  --train \
  --model google/flan-t5-small \
  --data-path ./my_data \
  --project-name my-seq2seq-model \
  --epochs 3 \
  --push-to-hub \
  --username your-username \
  --token $HF_TOKEN
```

## Python API Usage

### Basic Summarization Training

```python
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.seq2seq.__main__ import train

# Configure parameters
params = Seq2SeqParams(
    data_path="cnn_dailymail",
    model="google/flan-t5-base",
    project_name="./summarization-model",

    # Data configuration
    text_column="article",
    target_column="highlights",
    train_split="train[:5000]",  # Use subset for faster training
    valid_split="validation[:500]",

    # Training parameters
    lr=1e-4,
    epochs=3,
    batch_size=4,
    max_seq_length=512,
    max_target_length=128,

    # Optimization
    mixed_precision="fp16",
    gradient_accumulation=2,

    seed=42
)

# Train model
train(params)
```

### Translation with PEFT

```python
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.seq2seq.__main__ import train

params = Seq2SeqParams(
    data_path="./translation_dataset",
    model="google/flan-t5-large",
    project_name="./translator-model",

    # Data
    text_column="en",
    target_column="fr",
    train_split="train",
    valid_split="validation",

    # PEFT configuration for efficient training
    peft=True,
    quantization="int8",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",

    # Training
    lr=2e-4,
    epochs=5,
    batch_size=2,
    gradient_accumulation=4,  # Effective batch size = 8
    max_seq_length=256,
    max_target_length=256,

    # Performance
    mixed_precision="bf16",
    eval_strategy="epoch",
    early_stopping_patience=3,

    # Logging
    log="tensorboard",

    seed=42
)

train(params)
```

### Question Answering System

```python
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.seq2seq.__main__ import train

params = Seq2SeqParams(
    data_path="squad",
    model="google/flan-t5-base",
    project_name="./qa-model",

    # Data configuration
    text_column="question_context",  # Combined question + context
    target_column="answer",
    train_split="train",
    valid_split="validation",

    # Model configuration
    max_seq_length=512,  # Long context
    max_target_length=64,  # Short answers

    # Training
    lr=1e-4,
    epochs=3,
    batch_size=4,
    warmup_ratio=0.1,
    weight_decay=0.01,

    # Optimization
    optimizer="adamw_torch",
    scheduler="cosine",
    mixed_precision="fp16",

    # Evaluation
    eval_strategy="epoch",
    logging_steps=100,

    seed=42
)

train(params)
```

### Using Trained Model for Inference

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("./summarization-model")
tokenizer = AutoTokenizer.from_pretrained("./summarization-model")

# Prepare input
text = "summarize: " + "Your long article text here..."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

# Generate
outputs = model.generate(
    inputs.input_ids,
    max_length=128,
    num_beams=4,  # Beam search
    early_stopping=True,
    temperature=0.7,
    top_p=0.9
)

# Decode
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

### Using Pipeline for Inference

```python
from transformers import pipeline

# Create pipeline
summarizer = pipeline(
    "summarization",
    model="./summarization-model",
    device=0  # GPU
)

# Generate summary
article = "Your long article text here..."
summary = summarizer(
    article,
    max_length=128,
    min_length=30,
    do_sample=True,
    temperature=0.7
)

print(summary[0]['summary_text'])
```

## Data Preparation Tips

### 1. Format Input Text with Task Prefixes

For T5/FLAN-T5 models, always include task prefixes:

```python
import pandas as pd

df = pd.read_csv("raw_data.csv")

# Add task prefix
df['text'] = df['source'].apply(lambda x: f"translate English to French: {x}")
df['target'] = df['french_translation']

df[['text', 'target']].to_csv("formatted_data.csv", index=False)
```

### 2. Prepare Summarization Data

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Format for seq2seq
def format_example(example):
    return {
        "text": f"summarize: {example['article']}",
        "target": example['highlights']
    }

formatted_dataset = dataset.map(format_example)
formatted_dataset.save_to_disk("./summarization_data")
```

### 3. Prepare Translation Data

```python
from datasets import load_dataset

# Load parallel corpus
dataset = load_dataset("wmt14", "de-en")

def format_translation(example):
    return {
        "text": f"translate German to English: {example['de']}",
        "target": example['en']
    }

formatted = dataset.map(format_translation)
formatted.save_to_disk("./translation_data")
```

### 4. Handle Long Documents

```python
# Split long documents into chunks
def chunk_text(text, max_length=512, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_length - overlap):
        chunk = ' '.join(words[i:i + max_length])
        chunks.append(chunk)

    return chunks

# Process long documents
df['text_chunks'] = df['long_text'].apply(chunk_text)
df = df.explode('text_chunks')
```

### 5. Create Validation Split

```python
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("data.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42
)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
```

## Evaluation Metrics

The trainer computes ROUGE metrics for evaluating generation quality:

### ROUGE Scores
- **ROUGE-1:** Unigram overlap (individual word matches)
- **ROUGE-2:** Bigram overlap (two consecutive word matches)
- **ROUGE-L:** Longest common subsequence
- **ROUGE-Lsum:** ROUGE-L with sentence-level normalization

Higher scores indicate better overlap with reference text.

### Generation Length
- **gen_len:** Average length of generated sequences
- Useful for detecting degenerate outputs (too short/long)

### Interpreting ROUGE Scores

```
ROUGE-1: 0.35 → 35% of reference unigrams appear in generated text
ROUGE-2: 0.15 → 15% of reference bigrams appear in generated text
ROUGE-L: 0.30 → 30% longest common subsequence overlap
```

**Typical ranges:**
- **Good:** ROUGE-1 > 40, ROUGE-2 > 20, ROUGE-L > 35
- **Acceptable:** ROUGE-1 > 30, ROUGE-2 > 15, ROUGE-L > 25
- **Poor:** Below acceptable thresholds

## Troubleshooting

### Out of Memory Errors

**Symptoms:** CUDA OOM, allocation failures

**Solutions:**

```python
params = Seq2SeqParams(
    # Reduce batch size (most effective)
    batch_size=1,  # or 2

    # Use gradient accumulation for larger effective batch
    gradient_accumulation=8,  # Effective batch = 1 * 8 = 8

    # Reduce sequence lengths
    max_seq_length=256,  # Down from 512
    max_target_length=64,  # Down from 128

    # Enable PEFT (dramatically reduces memory)
    peft=True,
    quantization="int8",

    # Use mixed precision
    mixed_precision="fp16",  # or "bf16"

    # Use smaller model
    model="google/flan-t5-small"  # Instead of base/large
)
```

### Poor Generation Quality

**Symptoms:** Repetitive text, nonsensical outputs, generic responses

**Solutions:**

1. **Improve Training Data:**
```python
# Ensure diverse, high-quality examples
# Remove duplicates
df = df.drop_duplicates(subset=['text', 'target'])

# Filter low-quality examples
df = df[df['target'].str.len() > 10]  # Remove very short targets
```

2. **Adjust Training Hyperparameters:**
```python
params = Seq2SeqParams(
    # Train longer
    epochs=5,  # Up from 3

    # Adjust learning rate
    lr=1e-4,  # T5 models prefer higher LR

    # Add weight decay for regularization
    weight_decay=0.01,

    # Use warmup
    warmup_ratio=0.1
)
```

3. **Improve Inference Settings:**
```python
outputs = model.generate(
    inputs,
    max_length=128,
    min_length=30,  # Prevent very short outputs

    # Beam search for better quality
    num_beams=4,
    early_stopping=True,

    # Sampling for diversity
    do_sample=True,
    temperature=0.7,  # Lower = more focused, higher = more random
    top_k=50,
    top_p=0.9,

    # Prevent repetition
    no_repeat_ngram_size=3,
    repetition_penalty=1.2
)
```

### Slow Training

**Solutions:**

```python
params = Seq2SeqParams(
    # Enable mixed precision
    mixed_precision="bf16",  # Faster on A100, H100

    # Reduce evaluation frequency
    eval_strategy="epoch",  # Not "steps"
    logging_steps=100,  # Log less frequently

    # Use smaller model
    model="google/flan-t5-small",

    # Enable PEFT
    peft=True,  # Faster convergence with LoRA

    # Use efficient optimizer
    optimizer="adafactor"  # Memory-efficient optimizer
)
```

### Model Not Learning

**Symptoms:** Loss not decreasing, poor validation metrics

**Solutions:**

```python
params = Seq2SeqParams(
    # Increase learning rate
    lr=2e-4,  # Higher for T5

    # Reduce batch size to increase updates
    batch_size=2,
    gradient_accumulation=2,

    # Add warmup
    warmup_ratio=0.1,

    # Check data format
    # Ensure task prefixes are included!
    # "translate English to French: hello" not just "hello"
)
```

## Best Practices

### 1. Model Selection Strategy

```python
# Start small for prototyping
model = "google/flan-t5-small"  # Fast iterations

# Scale up for production
model = "google/flan-t5-base"   # Good balance

# Use specialized models when available
# Summarization: "facebook/bart-large-cnn"
# Translation: "Helsinki-NLP/opus-mt-en-fr"
```

### 2. Optimal Hyperparameters by Model

**T5/FLAN-T5:**
```python
params = Seq2SeqParams(
    lr=1e-4,  # Higher LR
    batch_size=4,
    epochs=3,
    optimizer="adafactor",  # T5 was trained with Adafactor
    scheduler="constant"
)
```

**BART:**
```python
params = Seq2SeqParams(
    lr=3e-5,  # Lower LR
    batch_size=8,
    epochs=3,
    optimizer="adamw_torch",
    scheduler="linear"
)
```

### 3. Memory-Efficient Training

```python
params = Seq2SeqParams(
    # Enable PEFT
    peft=True,
    quantization="int8",
    lora_r=8,  # Lower rank = less memory

    # Optimize batch processing
    batch_size=1,
    gradient_accumulation=16,

    # Mixed precision
    mixed_precision="bf16",

    # Reduce sequence lengths
    max_seq_length=256,
    max_target_length=64
)
```

### 4. Production Inference Optimization

```python
# Load model in inference mode
model = AutoModelForSeq2SeqLM.from_pretrained(
    "./model",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Optimize generation
outputs = model.generate(
    inputs,
    max_new_tokens=128,  # Faster than max_length
    num_beams=4,
    early_stopping=True,
    use_cache=True,  # Enable KV cache
)

# Batch inference for throughput
texts = ["text1", "text2", "text3"]
inputs = tokenizer(texts, padding=True, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_new_tokens=128)
```

### 5. Evaluation Best Practices

```python
params = Seq2SeqParams(
    # Always use validation set
    valid_split="validation",

    # Enable early stopping
    early_stopping_patience=3,
    early_stopping_threshold=0.001,

    # Regular evaluation
    eval_strategy="epoch",

    # Save best model
    save_total_limit=2,
    load_best_model_at_end=True
)
```

## Advanced Techniques

### 1. Multi-Task Training

Train on multiple tasks simultaneously:

```python
# Combine datasets with task prefixes
summarization_data = df_sum['text'].apply(lambda x: f"summarize: {x}")
translation_data = df_trans['text'].apply(lambda x: f"translate: {x}")

combined_df = pd.concat([
    pd.DataFrame({'text': summarization_data, 'target': df_sum['target']}),
    pd.DataFrame({'text': translation_data, 'target': df_trans['target']})
])
```

### 2. Controlled Generation

Add control tokens for specific generation styles:

```python
# Length control
"summarize in 50 words: " + article

# Style control
"formal tone: " + informal_text
"simple language: " + complex_text

# Domain control
"medical summary: " + medical_article
```

### 3. Curriculum Learning

Train on increasingly difficult examples:

```python
# Sort by difficulty (e.g., length)
df['length'] = df['text'].str.len()
df = df.sort_values('length')

# Train in stages
# Stage 1: Short examples (epochs 1-2)
# Stage 2: Medium examples (epochs 3-4)
# Stage 3: All examples (epochs 5+)
```

## Example: Complete Summarization Pipeline

```python
from datasets import load_dataset, DatasetDict
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.seq2seq.__main__ import train
from transformers import pipeline

# 1. Prepare data
dataset = load_dataset("cnn_dailymail", "3.0.0")

def format_data(example):
    return {
        "text": f"summarize: {example['article']}",
        "target": example['highlights']
    }

formatted_dataset = dataset.map(format_data)

# Use subset for faster training
train_subset = formatted_dataset['train'].select(range(5000))
val_subset = formatted_dataset['validation'].select(range(500))

formatted_dataset = DatasetDict({
    'train': train_subset,
    'validation': val_subset
})

formatted_dataset.save_to_disk("./summarization_dataset")

# 2. Configure training with PEFT for efficiency
params = Seq2SeqParams(
    data_path="./summarization_dataset",
    model="google/flan-t5-base",
    project_name="./my-summarizer",

    text_column="text",
    target_column="target",
    train_split="train",
    valid_split="validation",

    # PEFT for efficient training
    peft=True,
    quantization="int8",
    lora_r=16,
    lora_alpha=32,

    # Training config
    lr=1e-4,
    epochs=3,
    batch_size=4,
    gradient_accumulation=2,
    max_seq_length=512,
    max_target_length=128,

    # Optimization
    mixed_precision="fp16",
    optimizer="adafactor",
    warmup_ratio=0.1,

    # Evaluation
    eval_strategy="epoch",
    early_stopping_patience=2,

    seed=42
)

# 3. Train
train(params)

# 4. Load and test
summarizer = pipeline(
    "summarization",
    model="./my-summarizer",
    device=0
)

test_article = """
Your long article text here. This should be a substantial
piece of text that needs to be summarized. The model will
generate a concise summary capturing the key points.
"""

summary = summarizer(
    test_article,
    max_length=128,
    min_length=30,
    do_sample=False,  # Deterministic for consistency
    num_beams=4
)

print("Summary:", summary[0]['summary_text'])
```

## See Also

- [Text Classification](./TextClassification.md) - For classification tasks
- [Token Classification](./TokenClassification.md) - For NER and tagging
- [Sentence Transformers](./SentenceTransformers.md) - For embeddings
