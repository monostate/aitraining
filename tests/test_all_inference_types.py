"""Test all inference types to verify API structure and functionality."""

import os
from unittest.mock import MagicMock, Mock, patch

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
def mock_validate_model_path():
    """Mock validate_model_path to return test path."""

    def _validate(model_id):
        return f"/fake/path/{model_id}"

    return _validate


class TestLLMInference:
    """Test LLM inference with MessageCompleter."""

    def test_llm_with_system_prompt(self, mock_auth, monkeypatch):
        """Test that LLM inference uses MessageCompleter and system prompt."""
        # Mock detect_model_type
        monkeypatch.setattr("autotrain.app.api_routes.detect_model_type", lambda x: "llm")
        monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", lambda x: f"/fake/{x}")

        # Mock MessageCompleter
        mock_result = Mock()
        mock_result.text = "Hello! How can I help you today?"

        mock_completer = Mock()
        mock_completer.complete = Mock(return_value=mock_result)

        # Mock get_cached_llm to return our mock completer
        monkeypatch.setattr("autotrain.app.api_routes.get_cached_llm", lambda *args, **kwargs: mock_completer)

        # Make request with system prompt
        request_data = {
            "model_id": "test-llm/model",
            "inputs": {"text": "Hello", "system_prompt": "You are a helpful assistant."},
            "parameters": {"temperature": 0.8, "max_new_tokens": 150, "top_p": 0.9, "top_k": 40, "do_sample": True},
        }

        response = client.post("/api/inference/universal", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "llm"
        assert len(data["outputs"]) == 1
        assert data["outputs"][0] == "Hello! How can I help you today?"

        # Verify the completer.complete was called with correct arguments
        mock_completer.complete.assert_called_once()
        call_args = mock_completer.complete.call_args

        # First arg should be conversation
        conversation = call_args[0][0]
        assert isinstance(conversation, list)
        assert len(conversation) == 1
        assert conversation[0]["role"] == "user"
        assert conversation[0]["content"] == "Hello"

        # System prompt should be passed as kwarg
        assert call_args[1]["system_prompt"] == "You are a helpful assistant."

    def test_llm_without_system_prompt(self, mock_auth, monkeypatch):
        """Test that LLM inference works without system prompt."""
        monkeypatch.setattr("autotrain.app.api_routes.detect_model_type", lambda x: "llm")
        monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", lambda x: f"/fake/{x}")

        mock_result = Mock()
        mock_result.text = "Response without system prompt"

        mock_completer = Mock()
        mock_completer.complete = Mock(return_value=mock_result)

        monkeypatch.setattr("autotrain.app.api_routes.get_cached_llm", lambda *args, **kwargs: mock_completer)

        request_data = {"model_id": "test-llm/model", "inputs": {"text": "Test"}, "parameters": {}}

        response = client.post("/api/inference/universal", json=request_data)

        assert response.status_code == 200
        call_args = mock_completer.complete.call_args

        # System prompt should be None when not provided
        assert call_args[1]["system_prompt"] is None


class TestSeq2SeqInference:
    """Test Seq2Seq inference (T5, BART, etc.)."""

    def test_seq2seq_with_parameters(self, mock_auth, monkeypatch):
        """Test seq2seq inference with generation parameters."""
        monkeypatch.setattr("autotrain.app.api_routes.detect_model_type", lambda x: "seq2seq")
        monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", lambda x: f"/fake/{x}")

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"generated_text": "Translated output"}]

        monkeypatch.setattr("autotrain.app.api_routes.get_cached_pipeline", lambda *args, **kwargs: mock_pipeline)
        monkeypatch.setattr("autotrain.app.api_routes.torch", Mock(cuda=Mock(is_available=Mock(return_value=False))))

        request_data = {
            "model_id": "test-t5/model",
            "inputs": {"text": "translate English to French: Hello"},
            "parameters": {"temperature": 0.5, "max_new_tokens": 50, "top_p": 0.9, "top_k": 30, "do_sample": True},
        }

        response = client.post("/api/inference/universal", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "seq2seq"
        assert data["outputs"][0] == "Translated output"

        # Verify pipeline was called with generation kwargs
        mock_pipeline.assert_called_once()
        call_args = mock_pipeline.call_args
        assert call_args[0][0] == "translate English to French: Hello"
        kwargs = call_args[1]
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_new_tokens"] == 50
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 30
        assert kwargs["do_sample"] == True


class TestTextClassification:
    """Test text classification inference."""

    def test_text_classification(self, mock_auth, monkeypatch):
        """Test text classification returns proper structure."""
        monkeypatch.setattr("autotrain.app.api_routes.detect_model_type", lambda x: "text-classification")
        monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", lambda x: f"/fake/{x}")

        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}, {"label": "NEGATIVE", "score": 0.1}]

        monkeypatch.setattr("autotrain.app.api_routes.get_cached_pipeline", lambda *args, **kwargs: mock_pipeline)
        monkeypatch.setattr("autotrain.app.api_routes.torch", Mock(cuda=Mock(is_available=Mock(return_value=False))))

        request_data = {"model_id": "test-classifier/model", "inputs": {"text": "This is great!"}}

        response = client.post("/api/inference/universal", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "text-classification"
        assert len(data["outputs"]) == 2


class TestQuestionAnswering:
    """Test question answering inference."""

    def test_qa_requires_question_and_context(self, mock_auth, monkeypatch):
        """Test that QA requires both question and context."""
        monkeypatch.setattr("autotrain.app.api_routes.detect_model_type", lambda x: "extractive-question-answering")
        monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", lambda x: f"/fake/{x}")

        # Missing question
        request_data = {"model_id": "test-qa/model", "inputs": {"text": "Context here"}}

        response = client.post("/api/inference/universal", json=request_data)
        assert response.status_code == 400
        assert "question" in response.json()["detail"].lower()

    def test_qa_successful(self, mock_auth, monkeypatch):
        """Test successful QA inference."""
        monkeypatch.setattr("autotrain.app.api_routes.detect_model_type", lambda x: "extractive-question-answering")
        monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", lambda x: f"/fake/{x}")

        mock_pipeline = Mock()
        mock_pipeline.return_value = {"answer": "Paris", "score": 0.95, "start": 10, "end": 15}

        monkeypatch.setattr("autotrain.app.api_routes.get_cached_pipeline", lambda *args, **kwargs: mock_pipeline)
        monkeypatch.setattr("autotrain.app.api_routes.torch", Mock(cuda=Mock(is_available=Mock(return_value=False))))

        request_data = {
            "model_id": "test-qa/model",
            "inputs": {"text": "The capital of France is Paris.", "question": "What is the capital of France?"},
        }

        response = client.post("/api/inference/universal", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "extractive-question-answering"
        assert data["outputs"][0] == "Paris"

        # Verify pipeline was called correctly
        call_args = mock_pipeline.call_args
        assert call_args[1]["question"] == "What is the capital of France?"
        assert call_args[1]["context"] == "The capital of France is Paris."


class TestParameterPassing:
    """Test that parameters are correctly passed for different model types."""

    def test_llm_config_parameters(self, mock_auth, monkeypatch):
        """Verify LLM CompletionConfig receives correct parameters."""
        monkeypatch.setattr("autotrain.app.api_routes.detect_model_type", lambda x: "llm")
        monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", lambda x: f"/fake/{x}")

        captured_completer = None

        def capture_completer(model_path, config):
            nonlocal captured_completer
            mock_completer = Mock()
            # Create a mock config that can have attributes set
            mock_config = Mock()
            mock_config.max_new_tokens = 100
            mock_config.temperature = 0.7
            mock_config.top_p = 0.95
            mock_config.top_k = 50
            mock_config.do_sample = True
            mock_completer.config = mock_config
            mock_completer.complete = Mock(return_value=Mock(text="test"))
            captured_completer = mock_completer
            return mock_completer

        monkeypatch.setattr("autotrain.app.api_routes.get_cached_llm", capture_completer)

        request_data = {
            "model_id": "test/model",
            "inputs": {"text": "test"},
            "parameters": {"temperature": 1.5, "max_new_tokens": 200, "top_p": 0.85, "top_k": 60, "do_sample": False},
        }

        response = client.post("/api/inference/universal", json=request_data)

        assert response.status_code == 200
        assert captured_completer is not None
        # Verify config was updated with request parameters
        assert captured_completer.config.temperature == 1.5
        assert captured_completer.config.max_new_tokens == 200
        assert captured_completer.config.top_p == 0.85
        assert captured_completer.config.top_k == 60
        assert captured_completer.config.do_sample == False


class TestUnsupportedModelTypes:
    """Test handling of unsupported model types."""

    def test_unsupported_type_returns_400(self, mock_auth, monkeypatch):
        """Test that unsupported model types return proper error."""
        monkeypatch.setattr("autotrain.app.api_routes.detect_model_type", lambda x: "unknown")
        monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", lambda x: f"/fake/{x}")

        request_data = {"model_id": "test/model", "inputs": {"text": "test"}}

        response = client.post("/api/inference/universal", json=request_data)

        assert response.status_code == 400
        assert "Unsupported model type" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
