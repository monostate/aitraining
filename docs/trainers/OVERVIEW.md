# AutoTrain Advanced - Complete Trainer Overview

AutoTrain Advanced provides 14 different trainer categories covering the full spectrum of machine learning tasks. This document provides a comprehensive overview of all available trainers.

## Trainer Categories

### 1. Language Models (CLM)
**Path:** `src/autotrain/trainers/clm/`
**Purpose:** Training large language models for text generation, chat, and instruction-following

**Variants:**
- `default` - Standard causal language modeling
- `sft` - Supervised fine-tuning for instructions
- `dpo` - Direct preference optimization
- `orpo` - Odds ratio preference optimization
- `ppo` - Proximal policy optimization (RLHF)
- `reward` - Reward model training
- `distillation` - Prompt distillation

**Key Features:** PEFT/LoRA, quantization, chat templates, multi-GPU training

---

### 2. Text Classification
**Path:** `src/autotrain/trainers/text_classification/`
**Purpose:** Classify text into predefined categories

**Capabilities:**
- Binary classification (spam/ham, positive/negative)
- Multi-class classification (topic categorization)
- Multi-label classification (tag assignment)

**Models:** BERT, RoBERTa, DistilBERT, ALBERT, XLNet, etc.

---

### 3. Token Classification (NER)
**Path:** `src/autotrain/trainers/token_classification/`
**Purpose:** Named Entity Recognition and token-level classification

**Use Cases:**
- Named Entity Recognition (PER, LOC, ORG)
- Part-of-speech tagging
- Slot filling
- Chunking

**Models:** BERT-based models with token classification heads

---

### 4. Text Regression
**Path:** `src/autotrain/trainers/text_regression/`
**Purpose:** Predict continuous values from text

**Applications:**
- Sentiment intensity scoring
- Text similarity scoring
- Readability scoring
- Review rating prediction

**Models:** Transformer models adapted for regression

---

### 5. Sequence-to-Sequence (Seq2Seq)
**Path:** `src/autotrain/trainers/seq2seq/`
**Purpose:** Transform input sequences to output sequences

**Tasks:**
- Translation
- Summarization
- Question generation
- Text simplification
- Code generation

**Models:** T5, BART, mT5, Pegasus, MarianMT

---

### 6. Image Classification
**Path:** `src/autotrain/trainers/image_classification/`
**Purpose:** Classify images into categories

**Capabilities:**
- Single-label classification
- Multi-label classification
- Fine-grained recognition

**Models:** ViT, Swin, ResNet, EfficientNet, ConvNeXT

---

### 7. Image Regression
**Path:** `src/autotrain/trainers/image_regression/`
**Purpose:** Predict continuous values from images

**Applications:**
- Age estimation
- Quality scoring
- Depth estimation
- Pose estimation

**Models:** Vision transformers adapted for regression

---

### 8. Object Detection
**Path:** `src/autotrain/trainers/object_detection/`
**Purpose:** Detect and localize objects in images

**Features:**
- Bounding box prediction
- Class label assignment
- Multiple object detection

**Models:** DETR, YOLOS, Conditional DETR

---

### 9. Vision-Language Models (VLM)
**Path:** `src/autotrain/trainers/vlm/`
**Purpose:** Multi-modal understanding of images and text

**Variants:**
- `vqa` - Visual Question Answering
- `captioning` - Image Captioning

**Models:** BLIP, CLIP, Florence, LLaVA variants

---

### 10. Extractive Question Answering
**Path:** `src/autotrain/trainers/extractive_question_answering/`
**Purpose:** Extract answer spans from context

**Capabilities:**
- SQuAD v1 format (has answer)
- SQuAD v2 format (may have no answer)
- Context-based QA

**Models:** BERT, RoBERTa, ALBERT with QA heads

---

### 11. Sentence Transformers
**Path:** `src/autotrain/trainers/sent_transformers/`
**Purpose:** Generate semantic embeddings for sentences

**Training Types:**
- `pair` - Sentence pairs with similarity
- `pair_class` - Sentence pairs with labels
- `pair_score` - Sentence pairs with scores
- `triplet` - Anchor-positive-negative triplets
- `qa` - Question-answer pairs

**Applications:** Semantic search, clustering, duplicate detection

---

### 12. Tabular Data
**Path:** `src/autotrain/trainers/tabular/`
**Purpose:** Traditional ML on structured/tabular data

**Algorithms:**
- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Extra Trees
- Neural Networks (TabNet)

**Tasks:**
- Classification (binary, multi-class, multi-label)
- Regression (single-target, multi-target)

---

### 13. Generic Trainer
**Path:** `src/autotrain/trainers/generic/`
**Purpose:** Run custom training scripts

**Features:**
- Custom dataset pulling
- Dependency management
- Script execution
- Flexible configuration

---

### 14. Reinforcement Learning Components
**Path:** `src/autotrain/trainers/rl/`
**Purpose:** RL utilities and experimental implementations

**Note:** Production RL training uses TRL library (see CLM PPO trainer)

---

## Quick Selection Guide

### By Data Type

**Text Data:**
- Generation → CLM trainers
- Classification → Text Classification
- Token labeling → Token Classification
- Translation/Summarization → Seq2Seq
- Similarity/Search → Sentence Transformers
- QA → Extractive QA
- Scoring → Text Regression

**Image Data:**
- Classification → Image Classification
- Detection → Object Detection
- Scoring → Image Regression
- Captioning → VLM

**Tabular Data:**
- All tasks → Tabular trainer

**Multi-modal (Image + Text):**
- VQA/Captioning → VLM trainer

### By Task Type

**Classification:**
- Text → Text Classification
- Images → Image Classification
- Tokens → Token Classification
- Tabular → Tabular (classification mode)

**Regression:**
- Text → Text Regression
- Images → Image Regression
- Tabular → Tabular (regression mode)

**Generation:**
- Text → CLM trainers
- Translations → Seq2Seq
- Captions → VLM

**Extraction:**
- Entities → Token Classification
- Answers → Extractive QA

**Embeddings:**
- Sentences → Sentence Transformers

---

## Command Line Usage

Each trainer type has a corresponding CLI command:

```bash
# Language Models
autotrain llm --help

# Text Tasks
autotrain text-classification --help
autotrain text-regression --help
autotrain token-classification --help
autotrain seq2seq --help
autotrain sentence-transformers --help
autotrain extractive-question-answering --help

# Vision Tasks
autotrain image-classification --help
autotrain image-regression --help
autotrain object-detection --help
autotrain vlm --help

# Tabular Data
autotrain tabular --help

# Generic/Custom
autotrain generic --help
```

---

## Documentation Structure

```
docs/trainers/
├── OVERVIEW.md           # This file - complete overview
├── README.md             # CLM trainers index
├── clm/                  # Language model trainers
│   ├── Default.md
│   ├── SFT.md
│   ├── DPO.md
│   ├── ORPO.md
│   ├── PPO.md
│   ├── Reward.md
│   └── Distillation.md
├── nlp/                  # NLP trainers
│   ├── TextClassification.md
│   ├── TokenClassification.md
│   ├── TextRegression.md
│   ├── Seq2Seq.md
│   ├── ExtractiveQA.md
│   └── SentenceTransformers.md
├── vision/               # Computer vision trainers
│   ├── ImageClassification.md
│   ├── ImageRegression.md
│   ├── ObjectDetection.md
│   └── VLM.md
├── tabular/              # Tabular data
│   └── Tabular.md
└── generic/              # Custom scripts
    └── Generic.md
```

---

## Parameter Classes

Each trainer has a dedicated parameter class:

- `LLMTrainingParams` - Language models
- `TextClassificationParams` - Text classification
- `TokenClassificationParams` - Token classification
- `TextRegressionParams` - Text regression
- `Seq2SeqParams` - Sequence-to-sequence
- `ImageClassificationParams` - Image classification
- `ImageRegressionParams` - Image regression
- `ObjectDetectionParams` - Object detection
- `VLMTrainingParams` - Vision-language models
- `ExtractiveQuestionAnsweringParams` - QA
- `SentenceTransformersParams` - Sentence embeddings
- `TabularParams` - Tabular data
- `GenericParams` - Generic trainer

---

## Next Steps

1. **Choose Your Trainer:** Use the selection guide above
2. **Read Specific Documentation:** Navigate to the relevant trainer doc
3. **Prepare Your Data:** Follow the data format requirements
4. **Configure Parameters:** Use the parameter class reference
5. **Run Training:** Execute via CLI or Python API
6. **Deploy Model:** Push to Hugging Face Hub or export locally

---

## Support

- **GitHub Issues:** https://github.com/huggingface/autotrain-advanced/issues
- **Discord:** Hugging Face Discord #autotrain channel
- **Documentation:** https://huggingface.co/docs/autotrain