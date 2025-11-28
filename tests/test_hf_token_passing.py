"""Test that HF tokens are properly passed to model loading functions.

This ensures users can load private HuggingFace models via the web UI.
"""

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


class TestHFTokenPassing:
    """Test that user-provided HF tokens are passed to model loading."""

    def test_token_passed_to_pipeline_for_seq2seq(self, mock_auth, monkeypatch):
        """Verify token from Authorization header is passed to pipeline()."""
        monkeypatch.setattr("autotrain.app.api_routes.detect_model_type", lambda x: "seq2seq")
        monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", lambda x: x)

        captured_token = {}

        # Mock pipeline to capture the token parameter
        def mock_pipeline(task, model, device, token=None):
            captured_token["value"] = token
            mock_pipe = Mock()
            mock_pipe.return_value = [{"generated_text": "test output"}]
            return mock_pipe

        monkeypatch.setattr("autotrain.app.api_routes.pipeline", mock_pipeline)
        monkeypatch.setattr("autotrain.app.api_routes.torch", Mock(cuda=Mock(is_available=Mock(return_value=False))))

        # Make request with Authorization header containing HF token
        test_token = "hf_test_private_model_token_12345"
        request_data = {
            "model_id": "username/private-seq2seq-model",
            "inputs": {"text": "test"},
            "parameters": {"max_new_tokens": 50},
        }

        response = client.post(
            "/api/inference/universal", json=request_data, headers={"Authorization": f"Bearer {test_token}"}
        )

        print(f"\n{'='*80}")
        print("TEST: HF Token Passing to pipeline()")
        print("=" * 80)
        print(f"Sent token in header: {test_token}")
        print(f"Token received by pipeline(): {captured_token.get('value')}")
        print(f"Response status: {response.status_code}")

        assert response.status_code == 200
        assert captured_token["value"] == test_token, f"Expected {test_token}, got {captured_token.get('value')}"

        print("✅ Token successfully passed to pipeline()")
        print("=" * 80)

    def test_token_passed_to_sentence_transformer(self, mock_auth, monkeypatch):
        """Verify token is passed to SentenceTransformer()."""
        monkeypatch.setattr("autotrain.app.api_routes.detect_model_type", lambda x: "sentence-transformers")
        monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", lambda x: x)

        captured_token = {}

        # Mock SentenceTransformer
        class MockSentenceTransformer:
            def __init__(self, model_path, token=None):
                captured_token["value"] = token

            def to(self, device):
                return self

            def encode(self, texts):
                import numpy as np

                if isinstance(texts, str):
                    texts = [texts]
                return np.random.rand(len(texts), 384)

        # Patch at the actual import location (sentence_transformers module)
        monkeypatch.setattr("sentence_transformers.SentenceTransformer", MockSentenceTransformer)

        test_token = "hf_test_private_embeddings_token_67890"
        request_data = {"model_id": "username/private-embeddings-model", "inputs": {"texts": "test sentence"}}

        response = client.post(
            "/api/inference/universal", json=request_data, headers={"Authorization": f"Bearer {test_token}"}
        )

        print(f"\n{'='*80}")
        print("TEST: HF Token Passing to SentenceTransformer()")
        print("=" * 80)
        print(f"Sent token in header: {test_token}")
        print(f"Token received by SentenceTransformer(): {captured_token.get('value')}")
        print(f"Response status: {response.status_code}")

        assert response.status_code == 200
        assert captured_token["value"] == test_token

        print("✅ Token successfully passed to SentenceTransformer()")
        print("=" * 80)

    def test_token_passed_to_question_answering(self, mock_auth, monkeypatch):
        """Verify token is passed to question-answering pipeline."""
        monkeypatch.setattr("autotrain.app.api_routes.detect_model_type", lambda x: "extractive-question-answering")
        monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", lambda x: x)

        captured_token = {}

        def mock_pipeline(task, model, device, token=None):
            captured_token["value"] = token
            mock_pipe = Mock()
            mock_pipe.return_value = {"answer": "test answer", "score": 0.9}
            return mock_pipe

        monkeypatch.setattr("autotrain.app.api_routes.pipeline", mock_pipeline)
        monkeypatch.setattr("autotrain.app.api_routes.torch", Mock(cuda=Mock(is_available=Mock(return_value=False))))

        test_token = "hf_test_private_qa_model_token_99999"
        request_data = {
            "model_id": "username/private-qa-model",
            "inputs": {"text": "Context here", "question": "What is the answer?"},
        }

        response = client.post(
            "/api/inference/universal", json=request_data, headers={"Authorization": f"Bearer {test_token}"}
        )

        print(f"\n{'='*80}")
        print("TEST: HF Token Passing to QA pipeline()")
        print("=" * 80)
        print(f"Sent token in header: {test_token}")
        print(f"Token received by pipeline(): {captured_token.get('value')}")

        assert response.status_code == 200
        assert captured_token["value"] == test_token

        print("✅ Token successfully passed to QA pipeline()")
        print("=" * 80)

    def test_no_token_still_works(self, mock_auth, monkeypatch):
        """Verify inference still works without token (public models)."""
        monkeypatch.setattr("autotrain.app.api_routes.detect_model_type", lambda x: "seq2seq")
        monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", lambda x: x)

        captured_token = {}

        def mock_pipeline(task, model, device, token=None):
            captured_token["value"] = token
            mock_pipe = Mock()
            mock_pipe.return_value = [{"generated_text": "public model output"}]
            return mock_pipe

        monkeypatch.setattr("autotrain.app.api_routes.pipeline", mock_pipeline)
        monkeypatch.setattr("autotrain.app.api_routes.torch", Mock(cuda=Mock(is_available=Mock(return_value=False))))

        request_data = {"model_id": "t5-small", "inputs": {"text": "test"}, "parameters": {"max_new_tokens": 50}}

        # No Authorization header - public model
        response = client.post("/api/inference/universal", json=request_data)

        print(f"\n{'='*80}")
        print("TEST: Public Model (No Token)")
        print("=" * 80)
        print(f"No Authorization header sent")
        print(f"Token received by pipeline(): {captured_token.get('value')}")

        assert response.status_code == 200
        assert captured_token["value"] is None  # Should be None for public models

        print("✅ Public models work without token")
        print("=" * 80)

    def test_token_extraction_from_header(self, mock_auth, monkeypatch):
        """Test that token is correctly extracted from Authorization header."""
        monkeypatch.setattr("autotrain.app.api_routes.detect_model_type", lambda x: "text-classification")
        monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", lambda x: x)

        captured_token = {}

        def mock_pipeline(task, model, device, token=None):
            captured_token["value"] = token
            mock_pipe = Mock()
            mock_pipe.return_value = [{"label": "POSITIVE", "score": 0.99}]
            return mock_pipe

        monkeypatch.setattr("autotrain.app.api_routes.pipeline", mock_pipeline)
        monkeypatch.setattr("autotrain.app.api_routes.torch", Mock(cuda=Mock(is_available=Mock(return_value=False))))

        # Test various token formats
        test_cases = [
            ("Bearer hf_abc123", "hf_abc123"),
            ("Bearer hf_XYZ789_long_token_here", "hf_XYZ789_long_token_here"),
        ]

        for auth_header, expected_token in test_cases:
            # Clear the model cache to ensure pipeline is called for each test case
            import autotrain.app.api_routes

            autotrain.app.api_routes.MODEL_CACHE.clear()

            # Reset the captured token for each test case
            captured_token["value"] = None

            request_data = {"model_id": "test-model", "inputs": {"text": "test"}}

            response = client.post(
                "/api/inference/universal", json=request_data, headers={"Authorization": auth_header}
            )

            print(f"\nAuth header: {auth_header}")
            print(f"Extracted token: {captured_token.get('value')}")

            assert response.status_code == 200
            assert (
                captured_token["value"] == expected_token
            ), f"Expected {expected_token}, got {captured_token.get('value')}"

        print("✅ Token extraction works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
