# Sentence Transformers Trainer

## Overview

The Sentence Transformers trainer enables fine-tuning of models for generating semantic embeddings - dense vector representations that capture the meaning of text. Unlike classification or generation tasks, these models transform text into fixed-size vectors where semantically similar texts have similar embeddings, enabling powerful similarity search, clustering, and retrieval applications.

## Use Cases

- **Semantic Search:** Find relevant documents based on meaning, not just keywords
- **Question Answering Systems:** Match questions to relevant answers or passages
- **Duplicate Detection:** Identify duplicate or near-duplicate content
- **Clustering:** Group similar documents, reviews, or user queries
- **Recommendation Systems:** Recommend similar products, articles, or content
- **Paraphrase Detection:** Identify text with similar meaning but different wording
- **Cross-lingual Retrieval:** Find similar content across languages
- **Zero-Shot Classification:** Classify text by comparing embeddings to class descriptions
- **Conversational AI:** Match user queries to intents or knowledge base entries
- **Academic Paper Matching:** Find related research papers
- **Code Search:** Find similar code snippets or documentation
- **Image-Text Matching:** Connect visual and textual content (with multimodal models)

## Supported Models

Any sentence-transformers compatible encoder model:

### General Purpose Models
- **microsoft/mpnet-base** (default) - Best quality for English
- **sentence-transformers/all-mpnet-base-v2** - Pre-trained sentence model
- **sentence-transformers/all-MiniLM-L6-v2** - Fast and efficient
- **sentence-transformers/all-MiniLM-L12-v2** - Better quality, slower
- **sentence-transformers/paraphrase-multilingual-mpnet-base-v2** - Multilingual

### Specialized Models
- **sentence-transformers/msmarco-distilbert-base-v4** - Search/retrieval
- **sentence-transformers/multi-qa-mpnet-base-dot-v1** - Question answering
- **sentence-transformers/gtr-t5-base** - T5-based embeddings
- **BAAI/bge-base-en-v1.5** - High-quality general embeddings
- **intfloat/e5-base-v2** - Strong performance across tasks

### Multilingual Models
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** - 50+ languages
- **sentence-transformers/LaBSE** - 109 languages
- **intfloat/multilingual-e5-base** - Strong multilingual performance

## Trainer Types and Data Formats

Sentence Transformers supports multiple training objectives, each requiring different data formats.

### 1. Pair Trainer (`pair`)

**Use Case:** Learn from positive pairs (similar sentences)

**Data Format:**
```csv
sentence1,sentence2
"The cat sat on the mat","A feline rested on the rug"
"Machine learning is fascinating","I love studying ML"
```

**Python:**
```python
{
    "sentence1": "What is AI?",
    "sentence2": "Define artificial intelligence"
}
```

**Training Objective:** MultipleNegativesRankingLoss - pulls similar pairs together

### 2. Pair Classification Trainer (`pair_class`)

**Use Case:** Binary similarity classification (similar/dissimilar)

**Data Format:**
```csv
sentence1,sentence2,target
"Python is a programming language","Python is a snake",0
"Python for coding","Python programming",1
```

**Labels:** 0 = dissimilar, 1 = similar

**Training Objective:** SoftmaxLoss with binary classification

### 3. Pair Score Trainer (`pair_score`)

**Use Case:** Regression on similarity scores

**Data Format:**
```csv
sentence1,sentence2,target
"The weather is nice","It's a beautiful day",0.8
"I love pizza","Pizza is terrible",0.2
"Hello world","Goodbye world",0.5
```

**Scores:** Float between 0 (dissimilar) and 1 (similar)

**Training Objective:** CoSENTLoss - optimizes for ranking and similarity scores

### 4. Triplet Trainer (`triplet`)

**Use Case:** Learn from anchor-positive-negative triplets

**Data Format:**
```csv
sentence1,sentence2,sentence3
"Query: best restaurants","Review: excellent food and service","Review: terrible movie, avoid"
```

Where:
- sentence1 = anchor
- sentence2 = positive (similar to anchor)
- sentence3 = negative (dissimilar to anchor)

**Training Objective:** MultipleNegativesRankingLoss with triplets

### 5. Question-Answer Trainer (`qa`)

**Use Case:** Match questions to answers

**Data Format:**
```csv
sentence1,sentence2
"What is the capital of France?","The capital of France is Paris"
"How do I reset my password?","To reset your password, click the forgot password link"
```

**Training Objective:** MultipleNegativesRankingLoss optimized for QA

## Parameters

### Required Parameters
- **`data_path`**: Path to training data
- **`model`**: Base model (default: "microsoft/mpnet-base")
- **`project_name`**: Output directory
- **`trainer`**: Training objective (default: "pair_score")
  - Options: "pair", "pair_class", "pair_score", "triplet", "qa"

### Data Parameters
- **`sentence1_column`**: First sentence column (default: "sentence1")
- **`sentence2_column`**: Second sentence column (default: "sentence2")
- **`sentence3_column`**: Third sentence column for triplets (default: None)
- **`target_column`**: Label/score column (default: None)
  - Required for: pair_class, pair_score
- **`train_split`**: Training split name (default: "train")
- **`valid_split`**: Validation split name (default: None)
- **`max_samples`**: Limit dataset size (default: None)

### Training Parameters
- **`lr`**: Learning rate (default: 3e-5)
  - Recommended range: 1e-5 to 5e-5
- **`epochs`**: Number of epochs (default: 3)
  - Typical range: 1-5 epochs
- **`batch_size`**: Training batch size (default: 8)
  - Larger batches = better negative sampling
- **`max_seq_length`**: Maximum sequence length (default: 128)
- **`warmup_ratio`**: Warmup proportion (default: 0.1)
- **`gradient_accumulation`**: Gradient accumulation steps (default: 1)
- **`optimizer`**: Optimizer (default: "adamw_torch")
- **`scheduler`**: LR scheduler (default: "linear")

### Model Parameters
- **`weight_decay`**: Weight decay (default: 0.0)
- **`max_grad_norm`**: Gradient clipping (default: 1.0)
- **`seed`**: Random seed (default: 42)

### Performance Parameters
- **`mixed_precision`**: Mixed precision training (default: None)
  - Options: "fp16", "bf16", None
- **`auto_find_batch_size`**: Auto-find batch size (default: False)
- **`logging_steps`**: Logging frequency (default: -1)
- **`eval_strategy`**: Evaluation strategy (default: "epoch")

### Early Stopping
- **`early_stopping_patience`**: Patience (default: 5)
- **`early_stopping_threshold`**: Threshold (default: 0.01)

### Hub Parameters
- **`push_to_hub`**: Push to Hub (default: False)
- **`username`**: Hugging Face username
- **`token`**: Hugging Face token
- **`save_total_limit`**: Max checkpoints (default: 1)

### Logging
- **`log`**: Experiment tracking (default: "none")
  - Options: "tensorboard", "wandb", "mlflow"

## Command Line Usage

### Semantic Similarity with Scores

```bash
autotrain sentence-transformers \
  --train \
  --model microsoft/mpnet-base \
  --data-path ./similarity_data.csv \
  --trainer pair_score \
  --sentence1-column text1 \
  --sentence2-column text2 \
  --target-column similarity_score \
  --project-name semantic-similarity-model \
  --epochs 3 \
  --batch-size 16 \
  --lr 2e-5
```

### Question-Answer Matching

```bash
autotrain sentence-transformers \
  --train \
  --model sentence-transformers/multi-qa-mpnet-base-dot-v1 \
  --data-path ./qa_pairs.csv \
  --trainer qa \
  --sentence1-column question \
  --sentence2-column answer \
  --project-name qa-retrieval-model \
  --max-seq-length 256 \
  --epochs 3 \
  --batch-size 32
```

### Triplet Training for Search

```bash
autotrain sentence-transformers \
  --train \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --data-path ./search_data \
  --trainer triplet \
  --sentence1-column query \
  --sentence2-column positive_doc \
  --sentence3-column negative_doc \
  --project-name search-model \
  --epochs 2 \
  --batch-size 16 \
  --mixed-precision fp16
```

### Binary Similarity Classification

```bash
autotrain sentence-transformers \
  --train \
  --model bert-base-uncased \
  --data-path ./paraphrase_data.csv \
  --trainer pair_class \
  --sentence1-column sentence_a \
  --sentence2-column sentence_b \
  --target-column is_paraphrase \
  --project-name paraphrase-detector \
  --epochs 4 \
  --batch-size 32
```

## Python API Usage

### Basic Semantic Similarity Training

```python
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.sent_transformers.__main__ import train

# Configure training
params = SentenceTransformersParams(
    data_path="./similarity_dataset",
    model="microsoft/mpnet-base",
    project_name="./similarity-model",

    # Training objective
    trainer="pair_score",

    # Column mapping
    sentence1_column="text1",
    sentence2_column="text2",
    target_column="score",

    # Data splits
    train_split="train",
    valid_split="validation",

    # Training parameters
    lr=2e-5,
    epochs=3,
    batch_size=16,
    max_seq_length=128,

    # Optimization
    warmup_ratio=0.1,
    mixed_precision="fp16",

    seed=42
)

# Train model
train(params)
```

### Advanced QA Retrieval Model

```python
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.sent_transformers.__main__ import train

params = SentenceTransformersParams(
    # Data configuration
    data_path="squad",
    model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    project_name="./qa-retrieval-model",

    # Training type
    trainer="qa",

    # Column names
    sentence1_column="question",
    sentence2_column="context",

    # Splits
    train_split="train",
    valid_split="validation",

    # Training hyperparameters
    lr=2e-5,
    epochs=3,
    batch_size=32,  # Larger batch = better contrastive learning
    max_seq_length=256,

    # Optimization
    optimizer="adamw_torch",
    scheduler="warmup_linear",
    warmup_ratio=0.1,
    weight_decay=0.01,

    # Performance
    mixed_precision="bf16",
    gradient_accumulation=2,

    # Evaluation
    eval_strategy="epoch",
    early_stopping_patience=2,

    # Logging
    log="tensorboard",

    # Hub
    push_to_hub=True,
    username="my-org",
    token="hf_xxxxx",

    seed=42
)

train(params)
```

### Triplet Training for Semantic Search

```python
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.sent_transformers.__main__ import train

params = SentenceTransformersParams(
    data_path="./search_triplets",
    model="sentence-transformers/all-MiniLM-L6-v2",
    project_name="./search-model",

    # Triplet training
    trainer="triplet",
    sentence1_column="query",
    sentence2_column="positive",
    sentence3_column="negative",

    train_split="train",
    valid_split="validation",

    # Training config
    lr=2e-5,
    epochs=2,
    batch_size=16,
    max_seq_length=128,

    # Optimization
    mixed_precision="fp16",
    warmup_ratio=0.1,

    # Early stopping
    early_stopping_patience=2,

    seed=42
)

train(params)
```

### Using Trained Model

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer("./similarity-model")

# Encode sentences
sentences = [
    "The cat sits on the mat",
    "A feline rests on a rug",
    "Dogs are great pets",
    "Python is a programming language"
]

embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")  # (4, 768)

# Compute similarities
from sentence_transformers.util import cos_sim

similarities = cos_sim(embeddings, embeddings)
print("Similarity matrix:")
print(similarities)

# Find most similar to query
query = "Cat on carpet"
query_embedding = model.encode(query)
query_similarities = cos_sim(query_embedding, embeddings)[0]

# Get top matches
top_k = 2
top_results = np.argsort(-query_similarities)[:top_k]

for idx in top_results:
    print(f"{sentences[idx]}: {query_similarities[idx]:.3f}")
```

### Semantic Search Application

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer("./search-model")

# Index your documents
documents = [
    "Python is a high-level programming language",
    "Machine learning is a subset of AI",
    "Neural networks are inspired by the brain",
    # ... thousands more documents
]

# Create embeddings for all documents
doc_embeddings = model.encode(documents, show_progress_bar=True)

# Save embeddings for reuse
np.save("doc_embeddings.npy", doc_embeddings)

# Search function
def search(query, top_k=5):
    query_embedding = model.encode(query)
    similarities = cos_sim(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(-similarities)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "document": documents[idx],
            "score": float(similarities[idx])
        })

    return results

# Perform search
query = "What is machine learning?"
results = search(query, top_k=3)

for i, result in enumerate(results, 1):
    print(f"{i}. {result['document']} (score: {result['score']:.3f})")
```

## Data Preparation Tips

### 1. Prepare Pair Score Data

```python
import pandas as pd

# Your raw data
data = {
    'text1': ['The weather is nice', 'I love pizza'],
    'text2': ['It is a beautiful day', 'Pizza is delicious'],
    'similarity': [0.85, 0.90]  # 0-1 scale
}

df = pd.DataFrame(data)

# Ensure scores are normalized
df['similarity'] = df['similarity'].clip(0, 1)

df.to_csv('similarity_data.csv', index=False)
```

### 2. Create Triplets from Positive Pairs

```python
import pandas as pd
import random

# You have positive pairs
df = pd.read_csv('positive_pairs.csv')

triplets = []
for idx, row in df.iterrows():
    # Take a random negative from other sentences
    negative_idx = random.choice([i for i in range(len(df)) if i != idx])

    triplets.append({
        'anchor': row['sentence1'],
        'positive': row['sentence2'],
        'negative': df.iloc[negative_idx]['sentence1']
    })

triplet_df = pd.DataFrame(triplets)
triplet_df.to_csv('triplets.csv', index=False)
```

### 3. Generate Synthetic Paraphrases

```python
from transformers import pipeline

# Use a paraphrase model to generate synthetic data
paraphraser = pipeline("text2text-generation", model="humarin/chatgpt_paraphraser_on_T5_base")

sentences = ["Python is a programming language", "I love machine learning"]

pairs = []
for sentence in sentences:
    paraphrases = paraphraser(f"paraphrase: {sentence}", max_length=60, num_return_sequences=3)

    for para in paraphrases:
        pairs.append({
            'sentence1': sentence,
            'sentence2': para['generated_text'],
            'target': 1  # They're paraphrases
        })

df = pd.DataFrame(pairs)
```

### 4. Hard Negative Mining

```python
# After initial training, find hard negatives
model = SentenceTransformer("./initial-model")

# Encode all sentences
embeddings = model.encode(sentences)
similarities = cos_sim(embeddings, embeddings)

# For each anchor, find hard negatives
# (high similarity but not the same sentence)
hard_negatives = []
for i in range(len(sentences)):
    # Get most similar sentences (excluding itself)
    sims = similarities[i]
    sims[i] = -1  # Exclude self

    # Find high similarity but wrong matches
    hard_neg_idx = sims.argmax()

    hard_negatives.append({
        'anchor': sentences[i],
        'positive': positive_sentences[i],
        'negative': sentences[hard_neg_idx]  # Hard negative
    })
```

### 5. Creating Validation Sets

```python
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')

# Stratified split for classification
if 'target' in df.columns and df['target'].dtype in [int, bool]:
    train_df, val_df = train_test_split(
        df, test_size=0.1, stratify=df['target'], random_state=42
    )
else:
    train_df, val_df = train_test_split(
        df, test_size=0.1, random_state=42
    )

train_df.to_csv('train.csv', index=False)
val_df.to_csv('validation.csv', index=False)
```

## Evaluation Metrics

### For pair_score Trainer
- **Spearman Correlation:** Measures rank correlation between predicted and actual similarity
- **Pearson Correlation:** Measures linear correlation
- **MSE:** Mean squared error between predictions and labels

### For pair_class Trainer
- **Accuracy:** Classification accuracy
- **F1 Score:** Harmonic mean of precision and recall
- **Precision/Recall:** For similarity detection

### For triplet/pair/qa Trainers
- **Retrieval Metrics:**
  - Mean Reciprocal Rank (MRR)
  - Mean Average Precision (MAP)
  - Recall@K

### Interpreting Results

```python
# Good performance indicators:
# pair_score: Spearman > 0.80, Pearson > 0.75
# pair_class: Accuracy > 0.85, F1 > 0.85
# qa/retrieval: MRR > 0.70, Recall@10 > 0.90
```

## Troubleshooting

### Poor Similarity Scores

**Symptoms:** Model assigns similar scores to all pairs

**Solutions:**

```python
params = SentenceTransformersParams(
    # Use larger batch size for better contrastive learning
    batch_size=32,  # Or higher

    # Train longer
    epochs=5,

    # Use appropriate trainer
    trainer="pair_score",  # For regression
    # or
    trainer="triplet",  # For ranking

    # Adjust learning rate
    lr=2e-5,

    # Add more diverse training data
    # Include both similar and dissimilar pairs
)
```

### Model Overfitting

**Symptoms:** Perfect training scores, poor validation

**Solutions:**

```python
params = SentenceTransformersParams(
    # Add regularization
    weight_decay=0.01,

    # Use early stopping
    valid_split="validation",
    early_stopping_patience=2,

    # Reduce epochs
    epochs=2,

    # Add more training data
    # or use data augmentation
)
```

### Slow Inference

**Solutions:**

```python
from sentence_transformers import SentenceTransformer
import torch

# Load in fp16
model = SentenceTransformer("./model")
model = model.half()  # Convert to fp16

# Use GPU
model = model.to('cuda')

# Batch encoding
embeddings = model.encode(
    sentences,
    batch_size=64,  # Larger batches = faster
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # For cosine similarity
)

# For very fast inference, use ONNX
```

### Out of Memory

**Solutions:**

```python
params = SentenceTransformersParams(
    # Reduce batch size
    batch_size=8,

    # Use gradient accumulation
    gradient_accumulation=4,

    # Reduce sequence length
    max_seq_length=64,

    # Enable mixed precision
    mixed_precision="fp16",

    # Use smaller model
    model="sentence-transformers/all-MiniLM-L6-v2"
)
```

## Best Practices

### 1. Choose the Right Trainer

```python
# For similarity scores (0-1 range)
trainer="pair_score"  # Best

# For binary similar/dissimilar
trainer="pair_class"

# For ranking/retrieval
trainer="triplet"  # or "qa"

# For positive pairs only
trainer="pair"
```

### 2. Batch Size Matters

```python
# Larger batches = better in-batch negatives = better performance
# Try to use the largest batch that fits in memory

params = SentenceTransformersParams(
    batch_size=32,  # Or 64, 128 if possible
    gradient_accumulation=1,  # Adjust based on memory
)
```

### 3. Data Quality Over Quantity

```python
# Better: 1000 high-quality diverse pairs
# Worse: 10000 similar/repetitive pairs

# Include:
# - Diverse vocabulary
# - Various similarity levels (0.1, 0.5, 0.9)
# - Different domains/topics
# - Both similar and dissimilar pairs
```

### 4. Normalize Embeddings for Cosine Similarity

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./model")

# Normalize embeddings
embeddings = model.encode(
    sentences,
    normalize_embeddings=True  # Important!
)

# Now use dot product instead of cosine
# (faster and equivalent for normalized vectors)
similarities = embeddings @ embeddings.T
```

### 5. Production Deployment

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer("./model")

# Encode corpus
corpus_embeddings = model.encode(corpus, normalize_embeddings=True)

# Use FAISS for fast similarity search
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product (for normalized vectors)
index.add(corpus_embeddings.astype('float32'))

# Search
def search(query, k=5):
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding.astype('float32'), k)
    return [(corpus[i], scores[0][j]) for j, i in enumerate(indices[0])]

results = search("machine learning", k=5)
```

## Advanced Techniques

### 1. Domain Adaptation

Fine-tune on domain-specific data:

```python
params = SentenceTransformersParams(
    # Start from general model
    model="sentence-transformers/all-mpnet-base-v2",

    # Use domain-specific data
    data_path="./medical_pairs.csv",

    # Lower learning rate for fine-tuning
    lr=1e-5,

    # Fewer epochs
    epochs=2
)
```

### 2. Multi-Task Learning

Combine multiple objectives:

```python
# Train on mixed data
df1 = pd.read_csv('similarity_scores.csv')  # pair_score data
df2 = pd.read_csv('paraphrase_labels.csv')  # pair_class data

# Train separately or combine with different loss functions
```

### 3. Cross-Lingual Models

```python
params = SentenceTransformersParams(
    # Use multilingual model
    model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",

    # Train on parallel data
    data_path="./parallel_corpus",

    # English-French, English-German, etc.
)
```

### 4. Knowledge Distillation

Create smaller, faster models:

```python
# 1. Train large model
# 2. Use it to label unlabeled data
# 3. Train small model on labeled data

large_model = SentenceTransformer("./large-model")
small_model_data = generate_soft_labels(large_model, unlabeled_data)

# Train smaller model
params = SentenceTransformersParams(
    model="sentence-transformers/all-MiniLM-L6-v2",  # Smaller
    data_path=small_model_data,
    # ...
)
```

## Example: Complete Semantic Search System

```python
import pandas as pd
from sentence_transformers import SentenceTransformer
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.sent_transformers.__main__ import train
import numpy as np

# 1. Prepare training data
data = {
    'sentence1': [
        "How do I reset my password?",
        "What is your return policy?",
        "Where can I track my order?",
        # ... more questions
    ],
    'sentence2': [
        "To reset password, click forgot password link",
        "You can return items within 30 days",
        "Track your order in the orders section",
        # ... corresponding answers
    ],
    'score': [1.0, 1.0, 1.0]  # All are relevant pairs
}

df = pd.DataFrame(data)
train_df, val_df = train_test_split(df, test_size=0.1)

train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)

# 2. Train model
params = SentenceTransformersParams(
    data_path="./",  # Directory containing train.csv and val.csv
    model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    project_name="./qa-search-model",

    trainer="pair_score",
    sentence1_column="sentence1",
    sentence2_column="sentence2",
    target_column="score",

    train_split="train",
    valid_split="validation",

    lr=2e-5,
    epochs=3,
    batch_size=32,
    max_seq_length=128,
    mixed_precision="fp16",

    seed=42
)

train(params)

# 3. Create search system
model = SentenceTransformer("./qa-search-model")

# Index your FAQ/knowledge base
faqs = [
    "To reset your password, go to settings and click 'Forgot Password'",
    "Our return policy allows returns within 30 days of purchase",
    "You can track your order in the 'My Orders' section of your account",
    # ... all your FAQs
]

faq_embeddings = model.encode(faqs, normalize_embeddings=True)

# Search function
def search_faq(query, top_k=3):
    query_embedding = model.encode(query, normalize_embeddings=True)
    similarities = query_embedding @ faq_embeddings.T
    top_indices = np.argsort(-similarities)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'answer': faqs[idx],
            'score': float(similarities[idx])
        })

    return results

# Use the system
user_query = "I forgot my password, how do I recover it?"
results = search_faq(user_query, top_k=3)

for i, result in enumerate(results, 1):
    print(f"{i}. {result['answer']}")
    print(f"   Confidence: {result['score']:.2%}\n")
```

## See Also

- [Text Classification](./TextClassification.md) - For classification tasks
- [Token Classification](./TokenClassification.md) - For NER
- [Seq2Seq](./Seq2Seq.md) - For text generation
