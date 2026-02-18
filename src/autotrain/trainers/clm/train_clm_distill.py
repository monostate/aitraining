"""
Prompt Distillation Trainer for AutoTrain Advanced
====================================================

Implements prompt distillation to train student models to internalize complex prompts.
This allows reducing inference costs by having the model internally learn prompt behaviors.

Based on Tinker's approach to knowledge distillation from teacher to student models.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from autotrain import logger
from autotrain.trainers.clm import utils
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.utils import get_model_loading_kwargs, maybe_move_to_mps


@dataclass
class PromptDistillationConfig:
    """Configuration for prompt distillation training."""

    # Required fields (no defaults)
    teacher_model_name: str
    student_model_name: str

    # Optional fields with defaults
    teacher_prompt_template: str = "{input}"  # Complex prompt for teacher
    student_prompt_template: Optional[str] = ""  # Simple/empty prompt for student

    # Distillation settings
    temperature: float = 3.0  # Distillation temperature
    alpha: float = 0.7  # Weight for distillation loss vs cross-entropy
    max_length: int = 512
    num_samples: int = 1000  # Number of samples to generate from teacher

    # Generation settings
    teacher_generation_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_new_tokens": 256,
            "temperature": 0.8,
            "top_p": 0.95,
            "do_sample": True,
        }
    )

    # Training settings
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100

    # PEFT settings
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None

    # HuggingFace authentication
    token: Optional[str] = None


class DistillationDataset(Dataset):
    """Dataset for prompt distillation containing teacher outputs."""

    def __init__(
        self,
        base_inputs: List[str],
        teacher_outputs: List[str],
        teacher_logits: Optional[List[torch.Tensor]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
        student_prompt_template: str = "",
    ):
        self.base_inputs = base_inputs
        self.teacher_outputs = teacher_outputs
        self.teacher_logits = teacher_logits
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.student_prompt_template = student_prompt_template

    def __len__(self):
        return len(self.base_inputs)

    def __getitem__(self, idx):
        base_input = self.base_inputs[idx]
        teacher_output = self.teacher_outputs[idx]

        # Create student input (with simple/no prompt)
        if self.student_prompt_template:
            student_input = self.student_prompt_template.format(input=base_input)
        else:
            student_input = base_input

        # Combine input and output for training
        full_text = f"{student_input}\n{teacher_output}"

        # Tokenize if tokenizer provided
        if self.tokenizer:
            encoded = self.tokenizer(
                full_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
            )

            item = {
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
                "labels": encoded["input_ids"].squeeze(),
            }

            # Add teacher logits if available
            if self.teacher_logits:
                item["teacher_logits"] = self.teacher_logits[idx]

            return item
        else:
            return {"text": full_text, "base_input": base_input, "teacher_output": teacher_output}


class PromptDistillationTrainer(Trainer):
    """Custom trainer for prompt distillation with KL divergence loss."""

    def __init__(self, *args, distill_config: PromptDistillationConfig = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_config = distill_config

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute combined cross-entropy and KL divergence loss."""

        # Get student model outputs
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )

        # Standard cross-entropy loss
        ce_loss = outputs.loss

        # If we have teacher logits, compute KL divergence
        if "teacher_logits" in inputs and self.distill_config:
            student_logits = outputs.logits
            teacher_logits = inputs["teacher_logits"]

            # Apply temperature scaling
            T = self.distill_config.temperature
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)

            # KL divergence loss
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T**2)

            # Combine losses
            alpha = self.distill_config.alpha
            loss = alpha * kl_loss + (1 - alpha) * ce_loss
        else:
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss


def generate_teacher_outputs(
    teacher_model: AutoModelForCausalLM,
    teacher_tokenizer: AutoTokenizer,
    base_inputs: List[str],
    prompt_template: str,
    generation_config: Dict[str, Any],
    batch_size: int = 8,
    return_logits: bool = True,
) -> Tuple[List[str], Optional[List[torch.Tensor]]]:
    """Generate outputs from teacher model with complex prompts."""

    logger.info(f"Generating {len(base_inputs)} outputs from teacher model...")

    teacher_outputs = []
    teacher_logits = [] if return_logits else None

    # Process in batches
    for i in tqdm(range(0, len(base_inputs), batch_size), desc="Teacher generation"):
        batch_inputs = base_inputs[i : i + batch_size]

        # Apply teacher prompt template
        prompted_inputs = [prompt_template.format(input=inp) for inp in batch_inputs]

        # Tokenize
        encoded = teacher_tokenizer(
            prompted_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Handle both BatchEncoding and dict returns from tokenizer
        if hasattr(encoded, "input_ids"):
            # BatchEncoding object
            input_length = encoded.input_ids.shape[1]
        elif isinstance(encoded, dict) and "input_ids" in encoded:
            # Plain dict (e.g., from mocked tokenizer)
            input_length = (
                encoded["input_ids"].shape[1]
                if hasattr(encoded["input_ids"], "shape")
                else len(encoded["input_ids"][0])
            )
        else:
            # Fallback for edge cases
            input_length = 0

        # Move tensors to device (can't call .to() directly on BatchEncoding)
        encoded = {k: v.to(teacher_model.device) if hasattr(v, "to") else v for k, v in encoded.items()}

        # Generate with teacher model
        with torch.no_grad():
            if return_logits:
                outputs = teacher_model.generate(
                    **encoded,
                    **generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                # Extract generated text
                generated_ids = outputs.sequences[:, input_length:]
                generated_texts = teacher_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                teacher_outputs.extend(generated_texts)

                # Stack logits from generation
                if outputs.scores:
                    # Convert tuple of tensors to single tensor
                    batch_logits = torch.stack(outputs.scores, dim=1)
                    teacher_logits.extend([logits for logits in batch_logits])
            else:
                generated_ids = teacher_model.generate(
                    **encoded,
                    **generation_config,
                )

                # Extract only generated part
                generated_ids = generated_ids[:, input_length:]
                generated_texts = teacher_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                teacher_outputs.extend(generated_texts)

    return teacher_outputs, teacher_logits


def train_prompt_distillation(
    config: PromptDistillationConfig,
    base_inputs: List[str],
    output_dir: str = "./distilled_model",
    validation_inputs: Optional[List[str]] = None,
):
    """Main training function for prompt distillation."""

    logger.info("Starting prompt distillation training...")

    # Load teacher model and tokenizer
    logger.info(f"Loading teacher model: {config.teacher_model_name}")

    # Device setup via central helper
    model_kwargs = get_model_loading_kwargs(
        token=getattr(config, "token", None),
        fp16_if_cuda=True,
        trust_remote_code=True,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model_name, **model_kwargs)
    teacher_model = maybe_move_to_mps(teacher_model, model_kwargs)

    teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)

    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # Generate teacher outputs
    teacher_outputs, teacher_logits = generate_teacher_outputs(
        teacher_model,
        teacher_tokenizer,
        base_inputs,
        config.teacher_prompt_template,
        config.teacher_generation_config,
        return_logits=(config.alpha > 0),
    )

    # Save teacher outputs for inspection
    teacher_data_path = os.path.join(output_dir, "teacher_outputs.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    with open(teacher_data_path, "w") as f:
        for inp, out in zip(base_inputs, teacher_outputs):
            f.write(json.dumps({"input": inp, "teacher_output": out}) + "\n")
    logger.info(f"Saved teacher outputs to {teacher_data_path}")

    # Free teacher model memory if not needed for KL loss
    if config.alpha == 0:
        del teacher_model
        torch.cuda.empty_cache()

    # Load student model and tokenizer
    logger.info(f"Loading student model: {config.student_model_name}")

    # Device setup for student model
    student_kwargs = get_model_loading_kwargs(
        token=getattr(config, "token", None),
        fp16_if_cuda=True,
        trust_remote_code=True,
    )
    student_model = AutoModelForCausalLM.from_pretrained(config.student_model_name, **student_kwargs)
    student_model = maybe_move_to_mps(student_model, student_kwargs)
    student_tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)

    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    # Apply PEFT if configured
    if config.use_peft:
        logger.info("Applying LoRA configuration...")
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=config.lora_target_modules,
        )
        student_model = get_peft_model(student_model, peft_config)
        student_model.print_trainable_parameters()

    # Note: Don't manually move PEFT models to device - Trainer handles this

    # Create datasets
    train_dataset = DistillationDataset(
        base_inputs=base_inputs,
        teacher_outputs=teacher_outputs,
        teacher_logits=teacher_logits if config.alpha > 0 else None,
        tokenizer=student_tokenizer,
        max_length=config.max_length,
        student_prompt_template=config.student_prompt_template,
    )

    # Validation dataset if provided
    eval_dataset = None
    if validation_inputs:
        # Generate validation teacher outputs
        val_teacher_outputs, val_teacher_logits = generate_teacher_outputs(
            teacher_model if config.alpha > 0 else None,
            teacher_tokenizer,
            validation_inputs,
            config.teacher_prompt_template,
            config.teacher_generation_config,
            return_logits=(config.alpha > 0),
        )

        eval_dataset = DistillationDataset(
            base_inputs=validation_inputs,
            teacher_outputs=val_teacher_outputs,
            teacher_logits=val_teacher_logits if config.alpha > 0 else None,
            tokenizer=student_tokenizer,
            max_length=config.max_length,
            student_prompt_template=config.student_prompt_template,
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        fp16=torch.cuda.is_available() and not torch.backends.mps.is_available(),
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="steps" if eval_dataset else "no",  # Updated API
        eval_steps=50 if eval_dataset else None,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        greater_is_better=False,
        report_to="none",
        push_to_hub=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=student_tokenizer,
        mlm=False,
    )

    # Create trainer
    trainer = PromptDistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=student_tokenizer,
        data_collator=data_collator,
        distill_config=config if config.alpha > 0 else None,
    )

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=utils.get_resume_checkpoint(config))

    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    student_tokenizer.save_pretrained(output_dir)

    # Save distillation config
    config_path = os.path.join(output_dir, "distillation_config.json")
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    logger.info("Prompt distillation training complete!")

    return trainer


def train(config: LLMTrainingParams):
    """Entry point for AutoTrain integration."""

    logger.info("Starting prompt distillation training...")

    # Validate distillation requirements
    if not config.teacher_model:
        raise ValueError("teacher_model must be specified for distillation training")

    # Convert AutoTrain config to distillation config
    # Determine max_length from block_size
    max_length = config.distill_max_teacher_length
    if isinstance(config.block_size, int) and config.block_size > 0:
        max_length = config.block_size
    elif isinstance(config.block_size, list) and len(config.block_size) > 0:
        max_length = config.block_size[0]

    distill_config = PromptDistillationConfig(
        teacher_model_name=config.teacher_model,
        teacher_prompt_template=config.teacher_prompt_template or "You are a helpful assistant. {input}",
        student_model_name=config.model,
        student_prompt_template=config.student_prompt_template or "{input}",
        temperature=config.distill_temperature,
        alpha=config.distill_alpha,
        max_length=max_length,
        learning_rate=config.lr,
        batch_size=config.batch_size,
        num_epochs=config.epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=int(config.warmup_ratio * 100),  # Convert ratio to steps
        use_peft=config.peft,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        token=config.token,
    )

    # Load training data
    train_data, valid_data = utils.process_input_data(config)

    # Validate required columns
    utils.validate_required_columns(train_data, [config.text_column], "Distill", "training")
    if valid_data is not None:
        utils.validate_required_columns(valid_data, [config.text_column], "Distill", "validation")

    # Extract base inputs from data
    if config.text_column in train_data.column_names:
        base_inputs = train_data[config.text_column]
    else:
        base_inputs = train_data["text"]

    validation_inputs = None
    if valid_data and config.text_column in valid_data.column_names:
        validation_inputs = valid_data[config.text_column][:100]  # Use subset for validation

    # Run distillation training
    trainer = train_prompt_distillation(
        config=distill_config,
        base_inputs=base_inputs,
        output_dir=config.project_name,
        validation_inputs=validation_inputs,
    )

    # Post-training steps (push to hub, etc.)
    utils.post_training_steps(config, trainer)

    return trainer
