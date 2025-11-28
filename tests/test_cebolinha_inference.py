"""Quick test of Cebolinha model inference with proper EOS handling."""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# Load model and tokenizer
model_path = "../trainings/cebolinha_merged_peft/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
)

# Get test jokes from dataset (some from training, some not)
ds = load_dataset("monostate/cebolinha-sft", split="train")
test_questions = [
    ds[0]["question"],  # From training
    ds[500]["question"],  # From training
    "Por que o programador foi demitido?",  # Not in training - test generalization
    "O que você acha do clima hoje?",  # Simple question - test conversation
]

print("Testing Cebolinha model inference")
print("=" * 60)

# Get end_of_turn token ID for stopping
end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
eos_token_ids = [tokenizer.eos_token_id, end_of_turn_id]

for question in test_questions:
    print(f"\nQ: {question}")

    # Format with chat template
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Generate with proper stopping
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids,  # Stop on both EOS and end_of_turn
            repetition_penalty=1.1,
        )

    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the answer (after "model\n")
    if "model\n" in generated:
        answer = generated.split("model\n")[-1].strip()
    else:
        answer = generated.split(prompt)[-1].strip() if prompt in generated else generated

    print(f"A: {answer}")

    # Check if it has Cebolinha pattern (more 'l' than 'r')
    has_pattern = answer.lower().count("l") > answer.lower().count("r") and "l" in answer.lower()
    print(f"   {'✓' if has_pattern else '✗'} Cebolinha pattern detected")

print("\n" + "=" * 60)
print("Note: Model should apply r→l transformation in answers")
