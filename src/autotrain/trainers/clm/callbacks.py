import os

import torch
from peft import set_peft_model_state_dict
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")

        # Check for both .safetensors (new format) and .bin (legacy format)
        safetensors_path = os.path.join(state.best_model_checkpoint, "adapter_model.safetensors")
        bin_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")

        if os.path.exists(safetensors_path):
            # Use safetensors format (preferred)
            from safetensors.torch import load_file

            adapters_weights = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            # Fallback to legacy .bin format
            adapters_weights = torch.load(bin_path)
        else:
            raise FileNotFoundError(
                f"Could not find adapter weights in {state.best_model_checkpoint}. "
                f"Looked for adapter_model.safetensors and adapter_model.bin"
            )

        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control


class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.deepspeed)
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.accelerator.wait_for_everyone()
        return control
