"""Test real inference for newly added task types with actual HuggingFace models.

This test downloads small models from HF Hub and runs real inference.
NO MOCKS - everything is real.
"""

import base64
import os
from io import BytesIO

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from autotrain.app.api_routes import api_router


# Create test app
app = FastAPI()
app.include_router(api_router, prefix="/api")
client = TestClient(app)


@pytest.fixture
def mock_auth(monkeypatch):
    """Mock authentication to allow local access."""
    monkeypatch.delenv("HF_TOKEN", raising=False)


@pytest.mark.slow
@pytest.mark.integration
class TestSeq2SeqRealInference:
    """Test Seq2Seq inference with real T5 model from HuggingFace."""

    def test_t5_translation(self, mock_auth):
        """Test T5 small model for translation."""
        request_data = {
            "model_id": "t5-small",  # Download from HF Hub
            "inputs": {"text": "translate English to German: Hello, how are you?"},
            "parameters": {"temperature": 1.0, "max_new_tokens": 50, "top_p": 0.9, "top_k": 50, "do_sample": False},
        }

        print("\n" + "=" * 80)
        print("Testing Seq2Seq (T5-small) - Translation Task")
        print("=" * 80)
        print(f"Model: t5-small (downloading from HuggingFace Hub...)")
        print(f"Input: {request_data['inputs']['text']}")
        print(f"Parameters: {request_data['parameters']}")

        response = client.post("/api/inference/universal", json=request_data)

        print(f"\nResponse status: {response.status_code}")

        assert response.status_code == 200, f"Failed with: {response.text}"

        data = response.json()
        print(f"Response data: {data}")

        assert "outputs" in data
        assert "model_type" in data
        assert data["model_type"] == "seq2seq"
        assert isinstance(data["outputs"], list)
        assert len(data["outputs"]) > 0
        assert isinstance(data["outputs"][0], str)
        assert len(data["outputs"][0]) > 0

        print(f"\n✅ Generated translation: '{data['outputs'][0]}'")
        print("=" * 80)

    def test_t5_summarization(self, mock_auth):
        """Test T5 small model for summarization."""
        text = (
            "summarize: The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, "
            "and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. "
            "During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest "
            "man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York "
            "City was finished in 1930."
        )

        request_data = {
            "model_id": "t5-small",
            "inputs": {"text": text},
            "parameters": {"max_new_tokens": 100, "do_sample": False},
        }

        print("\n" + "=" * 80)
        print("Testing Seq2Seq (T5-small) - Summarization Task")
        print("=" * 80)
        print(f"Input text length: {len(text)} characters")

        response = client.post("/api/inference/universal", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["model_type"] == "seq2seq"
        assert len(data["outputs"][0]) > 0

        print(f"\n✅ Generated summary: '{data['outputs'][0]}'")
        print("=" * 80)


@pytest.mark.slow
@pytest.mark.integration
class TestQuestionAnsweringRealInference:
    """Test Question Answering with real model from HuggingFace."""

    def test_distilbert_qa(self, mock_auth):
        """Test DistilBERT QA model."""
        context = (
            "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest "
            "in the Amazon biome that covers most of the Amazon basin of South America. This basin "
            "encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) "
            "are covered by the rainforest."
        )

        question = "How large is the Amazon rainforest?"

        request_data = {
            "model_id": "distilbert-base-cased-distilled-squad",  # Download from HF Hub
            "inputs": {"text": context, "question": question},
        }

        print("\n" + "=" * 80)
        print("Testing Question Answering (DistilBERT-SQuAD)")
        print("=" * 80)
        print(f"Model: distilbert-base-cased-distilled-squad (downloading from HuggingFace Hub...)")
        print(f"Context: {context[:100]}...")
        print(f"Question: {question}")

        response = client.post("/api/inference/universal", json=request_data)

        print(f"\nResponse status: {response.status_code}")

        assert response.status_code == 200, f"Failed with: {response.text}"

        data = response.json()
        print(f"Response data: {data}")

        assert "outputs" in data
        assert "model_type" in data
        assert data["model_type"] == "extractive-question-answering"
        assert isinstance(data["outputs"], list)
        assert len(data["outputs"]) > 0
        assert isinstance(data["outputs"][0], str)
        assert len(data["outputs"][0]) > 0

        print(f"\n✅ Extracted answer: '{data['outputs'][0]}'")
        print("=" * 80)

    def test_qa_missing_question(self, mock_auth):
        """Test that QA properly validates required fields."""
        request_data = {
            "model_id": "distilbert-base-cased-distilled-squad",
            "inputs": {
                "text": "Some context here"
                # Missing question
            },
        }

        print("\n" + "=" * 80)
        print("Testing QA Validation - Missing Question")
        print("=" * 80)

        response = client.post("/api/inference/universal", json=request_data)

        print(f"Response status: {response.status_code}")
        print(f"Response: {response.json()}")

        assert response.status_code == 400
        assert "question" in response.json()["detail"].lower()

        print("✅ Validation working correctly - rejected missing question")
        print("=" * 80)


@pytest.mark.slow
@pytest.mark.integration
class TestSentenceTransformersRealInference:
    """Test Sentence Transformers with real model from HuggingFace."""

    def test_sentence_embeddings_single(self, mock_auth):
        """Test sentence transformer with single text."""
        request_data = {
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",  # Download from HF Hub
            "inputs": {"texts": "This is a test sentence for embedding generation."},
        }

        print("\n" + "=" * 80)
        print("Testing Sentence Transformers - Single Text")
        print("=" * 80)
        print(f"Model: sentence-transformers/all-MiniLM-L6-v2 (downloading from HuggingFace Hub...)")
        print(f"Input: {request_data['inputs']['texts']}")

        response = client.post("/api/inference/universal", json=request_data)

        print(f"\nResponse status: {response.status_code}")

        assert response.status_code == 200, f"Failed with: {response.text}"

        data = response.json()

        assert "outputs" in data
        assert "model_type" in data
        assert data["model_type"] == "sentence-transformers"
        assert isinstance(data["outputs"], list)
        assert len(data["outputs"]) > 0

        # Should return a list of embeddings (one per input text)
        assert isinstance(data["outputs"][0], list)  # First text's embedding
        embedding_dim = len(data["outputs"][0])

        print(f"\n✅ Generated embedding dimension: {embedding_dim}")
        print(f"✅ Embedding preview (first 10 values): {data['outputs'][0][:10]}")
        print("=" * 80)

    def test_sentence_embeddings_batch(self, mock_auth):
        """Test sentence transformer with multiple texts."""
        texts = ["The weather is nice today.", "I love machine learning.", "Python is a great programming language."]

        request_data = {"model_id": "sentence-transformers/all-MiniLM-L6-v2", "inputs": {"texts": texts}}

        print("\n" + "=" * 80)
        print("Testing Sentence Transformers - Batch Processing")
        print("=" * 80)
        print(f"Number of texts: {len(texts)}")
        for i, text in enumerate(texts):
            print(f"  {i+1}. {text}")

        response = client.post("/api/inference/universal", json=request_data)

        assert response.status_code == 200

        data = response.json()

        assert data["model_type"] == "sentence-transformers"
        assert len(data["outputs"]) == len(texts), "Should return one embedding per input text"

        # All embeddings should have same dimension
        embedding_dims = [len(emb) for emb in data["outputs"]]
        assert all(dim == embedding_dims[0] for dim in embedding_dims), "All embeddings should have same dimension"

        print(f"\n✅ Generated {len(data['outputs'])} embeddings")
        print(f"✅ Embedding dimension: {embedding_dims[0]}")
        print(f"✅ All embeddings same size: {embedding_dims}")
        print("=" * 80)

    def test_sentence_similarity(self, mock_auth):
        """Test semantic similarity using embeddings."""
        texts = [
            "The cat sits on the mat.",
            "A feline rests on a rug.",  # Similar meaning
            "Python is a programming language.",  # Different meaning
        ]

        request_data = {"model_id": "sentence-transformers/all-MiniLM-L6-v2", "inputs": {"texts": texts}}

        print("\n" + "=" * 80)
        print("Testing Sentence Transformers - Semantic Similarity")
        print("=" * 80)
        print("Texts:")
        for i, text in enumerate(texts):
            print(f"  {i+1}. {text}")

        response = client.post("/api/inference/universal", json=request_data)

        assert response.status_code == 200

        data = response.json()
        embeddings = data["outputs"]

        # Calculate cosine similarity
        import numpy as np

        def cosine_similarity(a, b):
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_0_1 = cosine_similarity(embeddings[0], embeddings[1])  # Similar sentences
        sim_0_2 = cosine_similarity(embeddings[0], embeddings[2])  # Different sentences

        print(f"\n✅ Similarity between sentences 1 and 2 (similar meaning): {sim_0_1:.4f}")
        print(f"✅ Similarity between sentences 1 and 3 (different meaning): {sim_0_2:.4f}")
        print(f"✅ Similar sentences have higher similarity: {sim_0_1 > sim_0_2}")

        # Similar sentences should have higher similarity
        assert sim_0_1 > sim_0_2, "Similar sentences should have higher cosine similarity"

        print("=" * 80)


@pytest.mark.slow
@pytest.mark.integration
class TestImageRegressionRealInference:
    """Test Image Regression inference."""

    def test_image_regression_basic(self, mock_auth):
        """Test image regression with a real image."""
        # Create a simple test image
        img = Image.new("RGB", (224, 224), color="red")

        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        request_data = {
            "model_id": "google/vit-base-patch16-224",  # Image classification model
            "task_override": "image-regression",  # Use it for regression by extracting scores
            "inputs": {"image": f"data:image/png;base64,{img_str}"},
        }

        print("\n" + "=" * 80)
        print("Testing Image Regression")
        print("=" * 80)
        print(f"Model: google/vit-base-patch16-224")
        print(f"Image: 224x224 red image")

        response = client.post("/api/inference/universal", json=request_data)

        print(f"\nResponse status: {response.status_code}")

        assert response.status_code == 200, f"Failed with: {response.text}"

        data = response.json()
        print(f"Response data: {data}")

        assert "outputs" in data
        assert "model_type" in data
        assert data["model_type"] == "image-regression"
        assert isinstance(data["outputs"], list)

        print(f"\n✅ Image regression output: {data['outputs']}")
        print("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "slow"])
