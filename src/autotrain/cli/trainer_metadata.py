"""
Metadata for all trainer types: FIELD_GROUPS and FIELD_SCOPES
This module centralizes parameter organization and scope information for the interactive wizard.
"""

from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_regression.params import ImageRegressionParams
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.token_classification.params import TokenClassificationParams
from autotrain.trainers.vlm.params import VLMTrainingParams


# ===== TEXT CLASSIFICATION =====

TEXT_CLASSIFICATION_FIELD_GROUPS = {
    "Basic": [
        "model",
        "project_name",
        "data_path",
        "train_split",
        "valid_split",
        "max_samples",
    ],
    "Data Processing": [
        "text_column",
        "target_column",
        "max_seq_length",
    ],
    "Training Configuration": [
        "log",
        "logging_steps",
        "eval_strategy",
        "auto_find_batch_size",
        "mixed_precision",
        "save_total_limit",
        "early_stopping_patience",
        "early_stopping_threshold",
    ],
    "Training Hyperparameters": [
        "lr",
        "epochs",
        "batch_size",
        "warmup_ratio",
        "gradient_accumulation",
        "optimizer",
        "scheduler",
        "weight_decay",
        "max_grad_norm",
        "seed",
    ],
    "Hub Integration": [
        "push_to_hub",
        "username",
        "token",
    ],
}

# ===== TOKEN CLASSIFICATION =====

TOKEN_CLASSIFICATION_FIELD_GROUPS = {
    "Basic": [
        "model",
        "project_name",
        "data_path",
        "train_split",
        "valid_split",
        "max_samples",
    ],
    "Data Processing": [
        "tokens_column",
        "tags_column",
        "max_seq_length",
    ],
    "Training Configuration": [
        "log",
        "logging_steps",
        "eval_strategy",
        "auto_find_batch_size",
        "mixed_precision",
        "save_total_limit",
        "early_stopping_patience",
        "early_stopping_threshold",
    ],
    "Training Hyperparameters": [
        "lr",
        "epochs",
        "batch_size",
        "warmup_ratio",
        "gradient_accumulation",
        "optimizer",
        "scheduler",
        "weight_decay",
        "max_grad_norm",
        "seed",
    ],
    "Hub Integration": [
        "push_to_hub",
        "username",
        "token",
    ],
}

# ===== TABULAR =====

TABULAR_FIELD_GROUPS = {
    "Basic": [
        "model",
        "project_name",
        "data_path",
        "train_split",
        "valid_split",
        "max_samples",
        "seed",
    ],
    "Data Processing": [
        "id_column",
        "target_columns",
        "categorical_columns",
        "numerical_columns",
        "task",
    ],
    "Training Configuration": [
        "num_trials",
        "time_limit",
        "categorical_imputer",
        "numerical_imputer",
        "numeric_scaler",
    ],
    "Hub Integration": [
        "push_to_hub",
        "username",
        "token",
    ],
}

# ===== IMAGE CLASSIFICATION =====

IMAGE_CLASSIFICATION_FIELD_GROUPS = {
    "Basic": [
        "model",
        "project_name",
        "data_path",
        "train_split",
        "valid_split",
        "max_samples",
    ],
    "Data Processing": [
        "image_column",
        "target_column",
    ],
    "Training Configuration": [
        "log",
        "logging_steps",
        "eval_strategy",
        "auto_find_batch_size",
        "mixed_precision",
        "save_total_limit",
        "early_stopping_patience",
        "early_stopping_threshold",
    ],
    "Training Hyperparameters": [
        "lr",
        "epochs",
        "batch_size",
        "warmup_ratio",
        "gradient_accumulation",
        "optimizer",
        "scheduler",
        "weight_decay",
        "max_grad_norm",
        "seed",
    ],
    "Hub Integration": [
        "push_to_hub",
        "username",
        "token",
    ],
}

# ===== IMAGE REGRESSION =====

IMAGE_REGRESSION_FIELD_GROUPS = {
    "Basic": [
        "model",
        "project_name",
        "data_path",
        "train_split",
        "valid_split",
        "max_samples",
    ],
    "Data Processing": [
        "image_column",
        "target_column",
    ],
    "Training Configuration": [
        "log",
        "logging_steps",
        "eval_strategy",
        "auto_find_batch_size",
        "mixed_precision",
        "save_total_limit",
        "early_stopping_patience",
        "early_stopping_threshold",
    ],
    "Training Hyperparameters": [
        "lr",
        "epochs",
        "batch_size",
        "warmup_ratio",
        "gradient_accumulation",
        "optimizer",
        "scheduler",
        "weight_decay",
        "max_grad_norm",
        "seed",
    ],
    "Hub Integration": [
        "push_to_hub",
        "username",
        "token",
    ],
}

# ===== SEQ2SEQ =====

SEQ2SEQ_FIELD_GROUPS = {
    "Basic": [
        "model",
        "project_name",
        "data_path",
        "train_split",
        "valid_split",
        "max_samples",
    ],
    "Data Processing": [
        "text_column",
        "target_column",
        "max_seq_length",
        "max_target_length",
    ],
    "Training Configuration": [
        "log",
        "logging_steps",
        "eval_strategy",
        "auto_find_batch_size",
        "mixed_precision",
        "save_total_limit",
        "early_stopping_patience",
        "early_stopping_threshold",
    ],
    "Training Hyperparameters": [
        "lr",
        "epochs",
        "batch_size",
        "warmup_ratio",
        "gradient_accumulation",
        "optimizer",
        "scheduler",
        "weight_decay",
        "max_grad_norm",
        "seed",
    ],
    "PEFT/LoRA": [
        "peft",
        "quantization",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
    ],
    "Hub Integration": [
        "push_to_hub",
        "username",
        "token",
    ],
}

# ===== EXTRACTIVE QA =====

EXTRACTIVE_QA_FIELD_GROUPS = {
    "Basic": [
        "model",
        "project_name",
        "data_path",
        "train_split",
        "valid_split",
        "max_samples",
    ],
    "Data Processing": [
        "text_column",
        "question_column",
        "answer_column",
        "max_seq_length",
        "max_doc_stride",
    ],
    "Training Configuration": [
        "log",
        "logging_steps",
        "eval_strategy",
        "auto_find_batch_size",
        "mixed_precision",
        "save_total_limit",
        "early_stopping_patience",
        "early_stopping_threshold",
    ],
    "Training Hyperparameters": [
        "lr",
        "epochs",
        "batch_size",
        "warmup_ratio",
        "gradient_accumulation",
        "optimizer",
        "scheduler",
        "weight_decay",
        "max_grad_norm",
        "seed",
    ],
    "Hub Integration": [
        "push_to_hub",
        "username",
        "token",
    ],
}

# ===== SENTENCE TRANSFORMERS =====

SENTENCE_TRANSFORMERS_FIELD_GROUPS = {
    "Basic": [
        "model",
        "project_name",
        "data_path",
        "train_split",
        "valid_split",
        "max_samples",
    ],
    "Data Processing": [
        "sentence1_column",
        "sentence2_column",
        "sentence3_column",
        "target_column",
        "max_seq_length",
        "trainer",
    ],
    "Training Configuration": [
        "log",
        "logging_steps",
        "eval_strategy",
        "auto_find_batch_size",
        "mixed_precision",
        "save_total_limit",
        "early_stopping_patience",
        "early_stopping_threshold",
    ],
    "Training Hyperparameters": [
        "lr",
        "epochs",
        "batch_size",
        "warmup_ratio",
        "gradient_accumulation",
        "optimizer",
        "scheduler",
        "weight_decay",
        "max_grad_norm",
        "seed",
    ],
    "Hub Integration": [
        "push_to_hub",
        "username",
        "token",
    ],
}

# ===== VLM (Vision Language Models) =====

VLM_FIELD_GROUPS = {
    "Basic": [
        "model",
        "project_name",
        "data_path",
        "train_split",
        "valid_split",
        "max_samples",
    ],
    "Data Processing": [
        "text_column",
        "image_column",
        "prompt_text_column",
        "trainer",
    ],
    "Training Configuration": [
        "log",
        "logging_steps",
        "eval_strategy",
        "save_total_limit",
        "auto_find_batch_size",
        "mixed_precision",
        "disable_gradient_checkpointing",
    ],
    "Training Hyperparameters": [
        "lr",
        "epochs",
        "batch_size",
        "warmup_ratio",
        "gradient_accumulation",
        "optimizer",
        "scheduler",
        "weight_decay",
        "max_grad_norm",
        "seed",
    ],
    "PEFT/LoRA": [
        "peft",
        "quantization",
        "target_modules",
        "merge_adapter",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
    ],
    "Hub Integration": [
        "push_to_hub",
        "username",
        "token",
    ],
}

# ===== TRAINER METADATA REGISTRY =====

TRAINER_METADATA = {
    "text-classification": {
        "params_class": TextClassificationParams,
        "field_groups": TEXT_CLASSIFICATION_FIELD_GROUPS,
        "display_name": "Text Classification",
        "description": "Classify text into predefined categories",
        "required_columns": ["text_column", "target_column"],
        "default_model": "bert-base-uncased",
        "starter_models": [
            "bert-base-uncased",
            "distilbert-base-uncased",
            "roberta-base",
            "albert-base-v2",
        ],
    },
    "token-classification": {
        "params_class": TokenClassificationParams,
        "field_groups": TOKEN_CLASSIFICATION_FIELD_GROUPS,
        "display_name": "Token Classification (NER)",
        "description": "Label individual tokens (Named Entity Recognition, POS tagging)",
        "required_columns": ["tokens_column", "tags_column"],
        "default_model": "bert-base-uncased",
        "starter_models": [
            "bert-base-uncased",
            "distilbert-base-uncased",
            "roberta-base",
        ],
    },
    "tabular": {
        "params_class": TabularParams,
        "field_groups": TABULAR_FIELD_GROUPS,
        "display_name": "Tabular",
        "description": "Train on structured/tabular data (classification or regression)",
        "required_columns": ["target_columns"],
        "default_model": "xgboost",
        "starter_models": [
            "xgboost",
            "random_forest",
            "lightgbm",
            "catboost",
            "logistic_regression",
            "ridge",
            "svm",
            "extra_trees",
        ],
    },
    "image-classification": {
        "params_class": ImageClassificationParams,
        "field_groups": IMAGE_CLASSIFICATION_FIELD_GROUPS,
        "display_name": "Image Classification",
        "description": "Classify images into predefined categories",
        "required_columns": ["image_column", "target_column"],
        "default_model": "google/vit-base-patch16-224",
        "starter_models": [
            "google/vit-base-patch16-224",
            "microsoft/resnet-50",
            "facebook/convnext-tiny-224",
        ],
    },
    "image-regression": {
        "params_class": ImageRegressionParams,
        "field_groups": IMAGE_REGRESSION_FIELD_GROUPS,
        "display_name": "Image Regression",
        "description": "Predict continuous values from images",
        "required_columns": ["image_column", "target_column"],
        "default_model": "google/vit-base-patch16-224",
        "starter_models": [
            "google/vit-base-patch16-224",
            "microsoft/resnet-50",
        ],
    },
    "seq2seq": {
        "params_class": Seq2SeqParams,
        "field_groups": SEQ2SEQ_FIELD_GROUPS,
        "display_name": "Seq2Seq",
        "description": "Sequence-to-sequence tasks (translation, summarization)",
        "required_columns": ["text_column", "target_column"],
        "default_model": "t5-small",
        "starter_models": [
            "t5-small",
            "t5-base",
            "google/flan-t5-small",
            "facebook/bart-base",
        ],
    },
    "extractive-qa": {
        "params_class": ExtractiveQuestionAnsweringParams,
        "field_groups": EXTRACTIVE_QA_FIELD_GROUPS,
        "display_name": "Extractive Question Answering",
        "description": "Extract answers from context passages",
        "required_columns": ["text_column", "question_column", "answer_column"],
        "default_model": "bert-base-uncased",
        "starter_models": [
            "bert-base-uncased",
            "distilbert-base-uncased",
            "roberta-base",
        ],
    },
    "sent-transformers": {
        "params_class": SentenceTransformersParams,
        "field_groups": SENTENCE_TRANSFORMERS_FIELD_GROUPS,
        "display_name": "Sentence Transformers",
        "description": "Train embeddings for semantic similarity",
        "required_columns": ["sentence1_column"],
        "default_model": "sentence-transformers/all-MiniLM-L6-v2",
        "starter_models": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "bert-base-uncased",
        ],
    },
    "vlm": {
        "params_class": VLMTrainingParams,
        "field_groups": VLM_FIELD_GROUPS,
        "display_name": "Vision-Language Model",
        "description": "Train models that understand both images and text",
        "required_columns": ["text_column", "image_column"],
        "default_model": "HuggingFaceM4/idefics2-8b",
        "starter_models": [
            "HuggingFaceM4/idefics2-8b",
            "llava-hf/llava-1.5-7b-hf",
        ],
    },
}


def get_trainer_metadata(trainer_type):
    """Get metadata for a specific trainer type."""
    return TRAINER_METADATA.get(trainer_type)


def get_all_trainer_types():
    """Get list of all available trainer types."""
    return list(TRAINER_METADATA.keys())


def get_trainer_display_name(trainer_type):
    """Get display name for a trainer type."""
    metadata = TRAINER_METADATA.get(trainer_type)
    return metadata["display_name"] if metadata else trainer_type
