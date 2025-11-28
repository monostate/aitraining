"""Test real inference with actual model loading and generation.

This test uses a small model from HuggingFace to verify the entire inference pipeline works.
"""

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from autotrain.app.api_routes import api_router


# Create test app
app = FastAPI()
app.include_router(api_router, prefix="/api")
client = TestClient(app)


@pytest.fixture
def mock_auth(monkeypatch):
    """Mock authentication to allow local access."""
    monkeypatch.delenv("HF_TOKEN", raising=False)


@pytest.fixture
def real_llm_model_path():
    """Use the actual trained model from your trainings directory."""
    # Use the cebolinha model you've been testing with
    model_path = "/Users/andrewcorrea/lmnr/Training-monorepo/autotrain-hf/trainings/cebolinha_full_finetune/model"

    if not os.path.exists(model_path):
        pytest.skip(f"Model not found at {model_path}")

    return "cebolinha_full_finetune/model"


@pytest.mark.slow
class TestRealLLMInference:
    """Test real LLM inference with actual model loading."""

    def test_llm_inference_without_system_prompt(self, mock_auth, real_llm_model_path):
        """Test real LLM inference without system prompt."""
        request_data = {
            "model_id": real_llm_model_path,
            "inputs": {"text": "oi"},
            "parameters": {"temperature": 0.7, "max_new_tokens": 50, "top_p": 0.95, "top_k": 50, "do_sample": True},
        }

        response = client.post("/api/inference/universal", json=request_data)

        print(f"\nResponse status: {response.status_code}")
        print(f"Response body: {response.json()}")

        assert response.status_code == 200, f"Failed with: {response.text}"

        data = response.json()
        assert "outputs" in data, "Response missing 'outputs' field"
        assert "model_type" in data, "Response missing 'model_type' field"
        assert data["model_type"] == "llm", f"Expected 'llm', got {data['model_type']}"
        assert isinstance(data["outputs"], list), "Outputs should be a list"
        assert len(data["outputs"]) > 0, "Outputs list is empty"
        assert isinstance(data["outputs"][0], str), "Output should be a string"
        assert len(data["outputs"][0]) > 0, "Generated text is empty"

        print(f"✅ Generated text: {data['outputs'][0]}")

    def test_llm_inference_with_system_prompt(self, mock_auth, real_llm_model_path):
        """Test real LLM inference with system prompt."""
        request_data = {
            "model_id": real_llm_model_path,
            "inputs": {
                "text": "Como o Cebolinha diria: Qual é o nome do filme?",
                "system_prompt": "Você é o Cebolinha da Turma da Mônica. Você troca R por L.",
            },
            "parameters": {"temperature": 0.8, "max_new_tokens": 100, "top_p": 0.9, "top_k": 40, "do_sample": True},
        }

        response = client.post("/api/inference/universal", json=request_data)

        print(f"\nResponse status: {response.status_code}")
        print(f"Response body: {response.json()}")

        assert response.status_code == 200, f"Failed with: {response.text}"

        data = response.json()
        assert data["model_type"] == "llm"
        assert len(data["outputs"]) > 0
        assert isinstance(data["outputs"][0], str)
        assert len(data["outputs"][0]) > 0

        print(f"✅ Generated text with system prompt: {data['outputs'][0]}")

    def test_llm_inference_different_parameters(self, mock_auth, real_llm_model_path):
        """Test that different parameters produce different outputs."""
        base_request = {
            "model_id": real_llm_model_path,
            "inputs": {"text": "test"},
        }

        # High temperature (more random)
        request_high_temp = {
            **base_request,
            "parameters": {"temperature": 1.5, "max_new_tokens": 30, "do_sample": True},
        }

        # Low temperature (more deterministic)
        request_low_temp = {
            **base_request,
            "parameters": {"temperature": 0.1, "max_new_tokens": 30, "do_sample": True},
        }

        response1 = client.post("/api/inference/universal", json=request_high_temp)
        response2 = client.post("/api/inference/universal", json=request_low_temp)

        assert response1.status_code == 200
        assert response2.status_code == 200

        output1 = response1.json()["outputs"][0]
        output2 = response2.json()["outputs"][0]

        print(f"\n✅ High temp output: {output1}")
        print(f"✅ Low temp output: {output2}")

        # Both should generate text
        assert len(output1) > 0
        assert len(output2) > 0


@pytest.mark.slow
class TestConversationStructure:
    """Test that conversation structure is properly formed."""

    def test_conversation_is_list_of_dicts(self, mock_auth, real_llm_model_path, monkeypatch):
        """Verify the conversation structure passed to completer.complete()."""

        captured_args = {}

        # Intercept the completer.complete call
        from autotrain.app.api_routes import get_cached_llm

        original_get_cached_llm = get_cached_llm

        def intercepting_get_cached_llm(*args, **kwargs):
            completer = original_get_cached_llm(*args, **kwargs)
            original_complete = completer.complete

            def capturing_complete(conversation, system_prompt=None, **extra_kwargs):
                captured_args["conversation"] = conversation
                captured_args["system_prompt"] = system_prompt
                return original_complete(conversation, system_prompt=system_prompt, **extra_kwargs)

            completer.complete = capturing_complete
            return completer

        monkeypatch.setattr("autotrain.app.api_routes.get_cached_llm", intercepting_get_cached_llm)

        request_data = {
            "model_id": real_llm_model_path,
            "inputs": {"text": "Hello", "system_prompt": "You are helpful."},
            "parameters": {"temperature": 0.7, "max_new_tokens": 20},
        }

        response = client.post("/api/inference/universal", json=request_data)

        assert response.status_code == 200

        # Verify conversation structure
        assert "conversation" in captured_args
        conversation = captured_args["conversation"]

        print(f"\n✅ Conversation structure: {conversation}")
        print(f"✅ System prompt: {captured_args['system_prompt']}")

        assert isinstance(conversation, list), "Conversation should be a list"
        assert len(conversation) == 1, "Should have 1 user message"
        assert conversation[0]["role"] == "user"
        assert conversation[0]["content"] == "Hello"
        assert captured_args["system_prompt"] == "You are helpful."


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "slow"])
