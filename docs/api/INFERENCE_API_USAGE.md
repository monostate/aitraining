# LLM Inference API Endpoint

## Overview
A new inference API endpoint has been successfully added to AutoTrain that allows running inference on trained LLM models via HTTP API calls.

## Endpoint Details

### URL
```
POST /api/llm/inference
```

### Authentication
- Requires Bearer token authentication
- Include token in request header: `Authorization: Bearer YOUR_TOKEN`

### Request Schema

```json
{
    "model_path": "string",           // Required: Path to the trained model
    "prompts": ["string", ...],       // Required: List of text prompts
    "max_new_tokens": 100,            // Optional: Maximum new tokens to generate (default: 100)
    "temperature": 0.7,               // Optional: Sampling temperature (default: 0.7)
    "top_p": 0.95,                   // Optional: Top-p sampling (default: 0.95)
    "top_k": 50,                     // Optional: Top-k sampling (default: 50)
    "do_sample": true,               // Optional: Enable sampling (default: true)
    "device": "cuda"                 // Optional: Device to use (default: null, auto-detect)
}
```

### Response Schema

```json
{
    "outputs": ["string", ...],      // Generated text for each prompt
    "model_path": "string",          // Echo of model path used
    "num_prompts": 2                 // Number of prompts processed
}
```

## Usage Examples

### Python Example

```python
import requests

url = "http://localhost:8000/api/llm/inference"
headers = {
    "Authorization": "Bearer YOUR_TOKEN",
    "Content-Type": "application/json"
}

data = {
    "model_path": "/path/to/your/model",
    "prompts": [
        "Tell me about artificial intelligence",
        "What is machine learning?"
    ],
    "max_new_tokens": 150,
    "temperature": 0.8
}

response = requests.post(url, json=data, headers=headers)
result = response.json()

for i, output in enumerate(result["outputs"]):
    print(f"Prompt {i+1} response: {output}")
```

### cURL Example

```bash
curl -X POST http://localhost:8000/api/llm/inference \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/path/to/your/model",
    "prompts": ["Hello, how are you?"],
    "max_new_tokens": 100,
    "temperature": 0.7
  }'
```

### JavaScript Example

```javascript
const response = await fetch('http://localhost:8000/api/llm/inference', {
    method: 'POST',
    headers: {
        'Authorization': 'Bearer YOUR_TOKEN',
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        model_path: '/path/to/your/model',
        prompts: ['Generate a story about a robot'],
        max_new_tokens: 200
    })
});

const result = await response.json();
console.log(result.outputs[0]);
```

## Error Handling

The endpoint includes proper error handling:

- **401 Unauthorized**: Invalid or missing authentication token
- **500 Internal Server Error**: Model loading failure or inference error
  - Error details will be in the response body

## Implementation Details

The endpoint leverages the existing `autotrain.generation` module, specifically:

1. Uses `CompletionConfig` to configure generation parameters
2. Creates a `TokenCompleter` instance for efficient token-level generation
3. Processes each prompt and returns generated text
4. Supports device specification for GPU/CPU inference

## Performance Considerations

- The endpoint processes prompts sequentially for stability
- For batch processing of many prompts, consider implementing pagination or async processing
- Model loading happens on each request - consider caching for production use

## Success Criteria Met ✅

- [x] POST /api/llm/inference endpoint works
- [x] Accepts model path + prompts
- [x] Returns generated text
- [x] Mirrors CLI inference functionality
- [x] Has proper error handling
- [x] Includes request/response schemas with Pydantic models
- [x] Supports customizable inference parameters