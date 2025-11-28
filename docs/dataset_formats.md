# Dataset Formats and Conversion

AutoTrain supports automatic detection and conversion of various dataset formats for LLM training. This document describes the supported formats and conversion options.

## Supported Formats

### 1. Messages Format (Standard)
The canonical format with `role` and `content` keys:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is..."}
  ]
}
```

### 2. ShareGPT Format
Common format with `from` and `value` keys:
```json
{
  "conversations": [
    {"from": "human", "value": "Hello"},
    {"from": "gpt", "value": "Hi there!"}
  ]
}
```

### 3. Alpaca Format
Instruction-based format:
```json
{
  "instruction": "Translate to French",
  "input": "Hello world",
  "output": "Bonjour le monde"
}
```

### 4. Q&A Formats
Various question-answer patterns are auto-detected:
- `question`/`answer`
- `query`/`response`
- `prompt`/`completion`
- `human`/`bot`
- `input`/`target`
- `user`/`assistant`

### 5. DPO/ORPO Format
For preference learning:
```json
{
  "prompt": "Write a poem",
  "chosen": "Roses are red...",
  "rejected": "Poetry is boring..."
}
```

## Conversion Options

### 1. Automatic Conversion (Recommended)
```bash
autotrain llm --auto-convert-dataset
```

This will:
1. Detect your dataset format automatically
2. Convert to standard messages format
3. Apply the appropriate chat template

### 2. Runtime Mapping (ShareGPT Alternative)
For ShareGPT-like datasets, you can skip conversion and use runtime mapping:

```bash
autotrain llm \
  --use-sharegpt-mapping \
  --runtime-mapping '{"role": "sender", "content": "text", "user": "human", "assistant": "bot"}' \
  --map-eos-token
```

This follows [Unsloth's approach](https://github.com/unslothai/unsloth) of mapping keys during template application.

### 3. Manual Column Mapping
For completely custom formats:

```bash
autotrain llm \
  --auto-convert-dataset \
  --column-mapping '{"user_col": "my_question", "assistant_col": "my_answer"}'
```

Or for Alpaca-style:
```bash
--column-mapping '{"instruction_col": "task", "input_col": "context", "output_col": "result"}'
```

## Chat Templates

AutoTrain includes 32+ chat templates from Unsloth:
- `llama3`, `llama3.1`, `llama3.2`
- `gemma`, `gemma2`, `gemma3`
- `qwen2.5`
- `chatml` (ChatML format)
- `deepseek3`, `deepseek2.5`
- `phi3`, `phi4`
- And many more...

### Using Templates

```bash
# Use model's built-in template
--chat-template tokenizer

# Use specific template
--chat-template llama3

# With runtime mapping
--chat-template chatml --map-eos-token
```

### Map EOS Token

The `--map-eos-token` flag maps template end tokens (like `<|im_end|>`) to the model's EOS token. This is recommended for:
- ChatML templates
- Gemma templates
- Qwen templates

This helps models learn when to stop generating without additional training.

## Conversation Extension

Convert single-turn examples to multi-turn conversations:

```bash
--conversation-extension 3  # Merge 3 single-turn examples into one conversation
```

This is useful for teaching models to handle multi-turn dialogues.

## Interactive Wizard

The CLI wizard will guide you through:
1. Dataset format detection
2. Conversion options
3. Column mapping (if needed)
4. Chat template selection
5. Runtime mapping configuration

```bash
autotrain llm --interactive
```

## Examples

### Example 1: Standard Alpaca Dataset
```bash
autotrain llm \
  --data-path my_alpaca_dataset \
  --auto-convert-dataset \
  --chat-template llama3 \
  --conversation-extension 2
```

### Example 2: Custom ShareGPT with Runtime Mapping
```bash
autotrain llm \
  --data-path custom_sharegpt \
  --use-sharegpt-mapping \
  --runtime-mapping '{"role": "speaker", "content": "message", "user": "person", "assistant": "ai"}' \
  --chat-template chatml \
  --map-eos-token
```

### Example 3: Unknown Format with Manual Mapping
```bash
autotrain llm \
  --data-path custom_dataset.csv \
  --auto-convert-dataset \
  --column-mapping '{"user_col": "question_text", "assistant_col": "answer_text"}' \
  --chat-template gemma3
```

## Technical Details

The conversion pipeline follows this flow:

1. **Detection**: Analyze dataset columns and content
2. **Normalization**: Convert to canonical messages format
3. **Extension** (optional): Merge single-turn to multi-turn
4. **Template Application**: Apply model-specific chat template
5. **Training**: Use formatted text for training

Or with runtime mapping:
1. **Detection**: Identify ShareGPT-like format
2. **Runtime Mapping**: Apply mapping during template application (no conversion)
3. **Training**: Use formatted text directly

This approach is compatible with [Unsloth's methodology](https://docs.unslothai.com/), ensuring optimal performance and compatibility.