# AutoTrain Advanced Trainers Documentation

Welcome to the comprehensive documentation for all AutoTrain Advanced trainers. This index provides quick access to documentation for all 14 trainer categories.

## 📚 Quick Navigation

### 🤖 Language Models (LLMs)
Advanced training methods for large language models and text generation.

- [**Overview**](./clm/README.md) - Index of all CLM trainers
- [Default CLM](./clm/Default.md) - Standard causal language modeling
- [SFT](./clm/SFT.md) - Supervised fine-tuning for instructions
- [DPO](./clm/DPO.md) - Direct preference optimization
- [ORPO](./clm/ORPO.md) - Odds ratio preference optimization
- [PPO](./clm/PPO.md) - Reinforcement learning with PPO
- [Reward](./clm/Reward.md) - Training reward models for RLHF
- [Distillation](./clm/Distillation.md) - Prompt distillation

### 📝 Natural Language Processing
Comprehensive NLP tasks for text understanding and generation.

- [Text Classification](./nlp/TextClassification.md) - Sentiment, topics, intent
- [Token Classification](./nlp/TokenClassification.md) - NER, POS tagging
- [Text Regression](./nlp/TextRegression.md) - Scoring and ranking
- [Seq2Seq](./nlp/Seq2Seq.md) - Translation, summarization
- [Extractive QA](./nlp/ExtractiveQA.md) - Answer extraction
- [Sentence Transformers](./nlp/SentenceTransformers.md) - Semantic embeddings

### 🖼️ Computer Vision
State-of-the-art vision models for image understanding.

- [Image Classification](./vision/ImageClassification.md) - Categorize images
- [Image Regression](./vision/ImageRegression.md) - Predict values from images
- [Object Detection](./vision/ObjectDetection.md) - Locate objects in images
- [VLM](./vision/VLM.md) - Vision-language models (VQA, captioning)

### 📊 Structured Data
Machine learning for tabular and structured datasets.

- [Tabular](./tabular/Tabular.md) - XGBoost, LightGBM, CatBoost, Neural Networks

### 🛠️ Specialized
Custom and specialized training workflows.

- [Generic](./generic/Generic.md) - Custom training scripts

---

## 🎯 Trainer Selection Guide

### By Task Type

| Task | Trainer | Best For |
|------|---------|----------|
| **Text Generation** | CLM (SFT, DPO, etc.) | Chatbots, content creation |
| **Text → Categories** | Text Classification | Sentiment, spam detection |
| **Text → Numbers** | Text Regression | Scoring, rating prediction |
| **Text → Text** | Seq2Seq | Translation, summarization |
| **Text → Entities** | Token Classification | NER, POS tagging |
| **Text → Answer** | Extractive QA | Reading comprehension |
| **Text → Embeddings** | Sentence Transformers | Semantic search |
| **Image → Categories** | Image Classification | Object recognition |
| **Image → Numbers** | Image Regression | Age/quality estimation |
| **Image → Boxes** | Object Detection | Object localization |
| **Image+Text → Text** | VLM | VQA, image captioning |
| **Table → Predictions** | Tabular | Business analytics |

### By Data Type

| Data Type | Available Trainers |
|-----------|-------------------|
| **Plain Text** | CLM, Text Classification, Text Regression, Sentence Transformers |
| **Text Pairs** | Seq2Seq, Sentence Transformers, Extractive QA |
| **Labeled Tokens** | Token Classification |
| **Images** | Image Classification, Image Regression, Object Detection |
| **Images + Text** | VLM |
| **CSV/Tabular** | Tabular |
| **Custom Format** | Generic |

### By Model Architecture

| Architecture | Trainers |
|--------------|----------|
| **GPT/LLaMA Style** | CLM trainers |
| **BERT Style** | Text/Token Classification, Extractive QA |
| **T5/BART Style** | Seq2Seq |
| **Vision Transformers** | Image Classification, VLM |
| **DETR Style** | Object Detection |
| **Tree-based** | Tabular (XGBoost, LightGBM, etc.) |

---

## 🚀 Getting Started

### 1. Install AutoTrain Advanced
```bash
pip install autotrain-advanced
```

### 2. Choose Your Trainer
Use the selection guide above or browse the [Complete Overview](./OVERVIEW.md).

### 3. Prepare Your Data
Each trainer documentation includes detailed data format requirements and examples.

### 4. Configure Training
```bash
# Example: Text classification
autotrain text-classification \
  --model bert-base-uncased \
  --data-path ./data \
  --text-column text \
  --target-column label \
  --output-dir ./output \
  --train
```

### 5. Deploy Your Model
```bash
# Push to Hugging Face Hub
autotrain [trainer-type] \
  --push-to-hub \
  --hub-model-id username/model-name
```

---

## 📖 Documentation Structure

```
docs/trainers/
├── README.md              # This file - master index
├── OVERVIEW.md           # Complete overview of all trainers
│
├── clm/                  # Language Model Trainers
│   ├── README.md         # CLM index
│   ├── Default.md        # Standard CLM
│   ├── SFT.md           # Supervised Fine-tuning
│   ├── DPO.md           # Direct Preference Optimization
│   ├── ORPO.md          # Odds Ratio Preference Optimization
│   ├── PPO.md           # Proximal Policy Optimization
│   ├── Reward.md        # Reward Model Training
│   └── Distillation.md  # Prompt Distillation
│
├── nlp/                  # NLP Trainers
│   ├── TextClassification.md
│   ├── TokenClassification.md
│   ├── TextRegression.md
│   ├── Seq2Seq.md
│   ├── ExtractiveQA.md
│   └── SentenceTransformers.md
│
├── vision/              # Vision Trainers
│   ├── ImageClassification.md
│   ├── ImageRegression.md
│   ├── ObjectDetection.md
│   └── VLM.md
│
├── tabular/             # Tabular Data
│   └── Tabular.md
│
└── generic/             # Custom Scripts
    └── Generic.md
```

---

## 💡 Common Workflows

### Fine-tune an LLM for Chat
```bash
# Use SFT trainer with chat template
autotrain llm \
  --trainer sft \
  --model meta-llama/Llama-2-7b-hf \
  --data-path ./chat_data \
  --chat-template zephyr \
  --train
```

### Build a Sentiment Classifier
```bash
autotrain text-classification \
  --model distilbert-base-uncased \
  --data-path ./reviews.csv \
  --text-column review \
  --target-column sentiment \
  --train
```

### Train a Custom Object Detector
```bash
autotrain object-detection \
  --model facebook/detr-resnet-50 \
  --data-path ./coco_format_data \
  --image-column image_path \
  --objects-column objects \
  --train
```

### Create Semantic Search Embeddings
```bash
autotrain sentence-transformers \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --trainer pair \
  --data-path ./sentence_pairs.csv \
  --sentence1-column query \
  --sentence2-column document \
  --train
```

---

## 🔧 Advanced Features

### Multi-GPU Training
Most trainers support distributed training:
```bash
autotrain [trainer] \
  --distributed-backend deepspeed \
  --num-gpus 4 \
  ...
```

### Mixed Precision
Speed up training with mixed precision:
```bash
autotrain [trainer] \
  --mixed-precision fp16 \
  ...
```

### Hyperparameter Optimization
Tabular trainer includes automatic HPO:
```bash
autotrain tabular \
  --num-trials 200 \
  --time-limit 3600 \
  ...
```

### PEFT/LoRA
Efficient fine-tuning for large models:
```bash
autotrain llm \
  --peft \
  --lora-r 16 \
  --lora-alpha 32 \
  ...
```

---

## 📊 Performance Considerations

| Trainer | GPU Required | Typical Training Time | Memory Usage |
|---------|--------------|----------------------|--------------|
| CLM | Yes (recommended) | Hours to days | High (8-80GB) |
| Text Classification | Optional | Minutes to hours | Low-Medium |
| Token Classification | Optional | Minutes to hours | Low-Medium |
| Seq2Seq | Recommended | Hours | Medium-High |
| Image Classification | Yes | Hours | Medium-High |
| Object Detection | Yes | Hours to days | High |
| VLM | Yes | Hours to days | Very High |
| Tabular | No | Minutes to hours | Low |
| Sentence Transformers | Optional | Minutes to hours | Low-Medium |

---

## 🤝 Contributing

Found an issue or want to contribute?
- Report issues: [GitHub Issues](https://github.com/huggingface/autotrain-advanced/issues)
- Join discussion: [Hugging Face Discord](https://discord.gg/huggingface)
- Contribute: See [CONTRIBUTING.md](../../CONTRIBUTING.md)

---

## 📚 Additional Resources

- [AutoTrain Advanced GitHub](https://github.com/huggingface/autotrain-advanced)
- [Hugging Face Documentation](https://huggingface.co/docs/autotrain)
- [Model Hub](https://huggingface.co/models)
- [Datasets Hub](https://huggingface.co/datasets)

---

*Last updated: October 2024*