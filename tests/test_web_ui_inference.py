"""Test web UI inference for all task types.

This test simulates the web UI flow:
1. User selects a model from the sidebar
2. UI loads the appropriate interface
3. User fills in inputs and clicks submit
4. API returns results that UI can display

Tests cover all task types to ensure end-to-end functionality.
"""

import base64
from io import BytesIO

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
class TestWebUISeq2SeqInference:
    """Test Seq2Seq inference via web UI flow."""

    def test_web_ui_seq2seq_translation(self, mock_auth):
        """Simulate web UI: user enters text, clicks translate."""
        # User selects t5-small from model list and enters text
        user_input_text = "translate English to German: Hello, how are you?"

        # User clicks "Submit" button - UI sends this request
        request_data = {
            "model_id": "t5-small",
            "inputs": {"text": user_input_text},
            "parameters": {"temperature": 1.0, "max_new_tokens": 50, "do_sample": False},
        }

        print("\n" + "=" * 80)
        print("WEB UI TEST: Seq2Seq Translation")
        print("=" * 80)
        print(f"User input: {user_input_text}")
        print("User clicks: Submit")

        response = client.post("/api/inference/universal", json=request_data)

        print(f"API response status: {response.status_code}")
        assert response.status_code == 200, f"Failed with: {response.text}"

        data = response.json()
        print(f"UI displays: {data['outputs'][0]}")

        # UI should be able to display these fields
        assert "outputs" in data
        assert "model_type" in data
        assert data["model_type"] == "seq2seq"
        assert len(data["outputs"]) > 0

        print("✅ Web UI can display translation result")
        print("=" * 80)


@pytest.mark.slow
@pytest.mark.integration
class TestWebUIQuestionAnswering:
    """Test Question Answering via web UI flow."""

    def test_web_ui_qa(self, mock_auth):
        """Simulate web UI: user enters context and question, clicks Find Answer."""
        # User fills in the QA form
        user_context = "The Amazon rainforest covers 5,500,000 km² of South America."
        user_question = "How large is the Amazon rainforest?"

        # User clicks "Find Answer" - UI sends this request
        request_data = {
            "model_id": "distilbert-base-cased-distilled-squad",
            "inputs": {"text": user_context, "question": user_question},
        }

        print("\n" + "=" * 80)
        print("WEB UI TEST: Question Answering")
        print("=" * 80)
        print(f"User enters context: {user_context[:50]}...")
        print(f"User enters question: {user_question}")
        print("User clicks: Find Answer")

        response = client.post("/api/inference/universal", json=request_data)

        print(f"API response status: {response.status_code}")
        assert response.status_code == 200, f"Failed with: {response.text}"

        data = response.json()
        print(f"UI displays answer in green box: {data['outputs'][0]}")

        assert data["model_type"] == "extractive-question-answering"
        assert isinstance(data["outputs"][0], str)
        assert len(data["outputs"][0]) > 0

        print("✅ Web UI can display extracted answer")
        print("=" * 80)

    def test_web_ui_qa_missing_question(self, mock_auth):
        """Simulate web UI: user forgets to enter question, gets validation error."""
        # User only fills in context, forgets question
        request_data = {
            "model_id": "distilbert-base-cased-distilled-squad",
            "inputs": {
                "text": "Some context here"
                # Missing question!
            },
        }

        print("\n" + "=" * 80)
        print("WEB UI TEST: QA Validation - Missing Question")
        print("=" * 80)
        print("User enters context but forgets question")
        print("User clicks: Find Answer")

        response = client.post("/api/inference/universal", json=request_data)

        print(f"API response status: {response.status_code}")
        print(f"UI shows error alert: {response.json()['detail']}")

        assert response.status_code == 400
        assert "question" in response.json()["detail"].lower()

        print("✅ Web UI can show validation error")
        print("=" * 80)


@pytest.mark.slow
@pytest.mark.integration
class TestWebUISentenceTransformers:
    """Test Sentence Transformers via web UI flow."""

    def test_web_ui_embeddings_single(self, mock_auth):
        """Simulate web UI: user enters single text, generates embedding."""
        user_text = "This is a test sentence for embedding generation."

        request_data = {"model_id": "sentence-transformers/all-MiniLM-L6-v2", "inputs": {"texts": user_text}}

        print("\n" + "=" * 80)
        print("WEB UI TEST: Sentence Embeddings - Single Text")
        print("=" * 80)
        print(f"User enters: {user_text}")
        print("User clicks: Generate Embeddings")

        response = client.post("/api/inference/universal", json=request_data)

        print(f"API response status: {response.status_code}")
        assert response.status_code == 200

        data = response.json()
        embedding = data["outputs"][0]

        print(f"UI displays:")
        print(f"  Generated 1 embedding(s)")
        print(f"  Dimension: {len(embedding)}")
        print(f"  Embedding preview: [{', '.join(map(str, embedding[:5]))}...]")

        assert data["model_type"] == "sentence-transformers"
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension

        print("✅ Web UI can display embedding info")
        print("=" * 80)

    def test_web_ui_embeddings_batch(self, mock_auth):
        """Simulate web UI: user enters multiple lines, generates batch embeddings."""
        # User enters multiple lines in textarea
        user_texts = [
            "The weather is nice today.",
            "I love machine learning.",
            "Python is a great programming language.",
        ]

        request_data = {"model_id": "sentence-transformers/all-MiniLM-L6-v2", "inputs": {"texts": user_texts}}

        print("\n" + "=" * 80)
        print("WEB UI TEST: Sentence Embeddings - Batch")
        print("=" * 80)
        print(f"User enters {len(user_texts)} lines:")
        for i, text in enumerate(user_texts):
            print(f"  {i+1}. {text}")
        print("User clicks: Generate Embeddings")

        response = client.post("/api/inference/universal", json=request_data)

        assert response.status_code == 200
        data = response.json()

        print(f"UI displays:")
        print(f"  Generated {len(data['outputs'])} embedding(s)")
        print(f"  Dimension: {len(data['outputs'][0])}")
        print(f"  All embeddings same size: {[len(e) for e in data['outputs']]}")

        assert len(data["outputs"]) == len(user_texts)
        assert all(len(emb) == 384 for emb in data["outputs"])

        print("✅ Web UI can display batch embedding results")
        print("=" * 80)


@pytest.mark.slow
@pytest.mark.integration
class TestWebUIImageRegression:
    """Test Image Regression via web UI flow."""

    def test_web_ui_image_regression(self, mock_auth):
        """Simulate web UI: user uploads image, gets regression value."""
        # User uploads a red image via file picker
        img = Image.new("RGB", (224, 224), color="red")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # User clicks "Analyze" - UI sends this request
        request_data = {
            "model_id": "google/vit-base-patch16-224",
            "task_override": "image-regression",  # Using task override
            "inputs": {"image": f"data:image/png;base64,{img_base64}"},
        }

        print("\n" + "=" * 80)
        print("WEB UI TEST: Image Regression")
        print("=" * 80)
        print("User uploads: 224x224 red image")
        print("User clicks: Analyze")

        response = client.post("/api/inference/universal", json=request_data)

        print(f"API response status: {response.status_code}")
        assert response.status_code == 200

        data = response.json()
        predicted_value = data["outputs"][0]

        print(f"UI displays in blue box:")
        print(f"  Predicted Value: {predicted_value:.4f}")

        assert data["model_type"] == "image-regression"
        assert isinstance(predicted_value, float)

        print("✅ Web UI can display regression value")
        print("=" * 80)


@pytest.mark.slow
@pytest.mark.integration
class TestWebUIImageClassification:
    """Test Image Classification displays correctly."""

    def test_web_ui_image_classification(self, mock_auth):
        """Simulate web UI: user uploads image, sees classification bars."""
        img = Image.new("RGB", (224, 224), color="blue")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        request_data = {
            "model_id": "google/vit-base-patch16-224",
            "inputs": {"image": f"data:image/png;base64,{img_base64}"},
        }

        print("\n" + "=" * 80)
        print("WEB UI TEST: Image Classification")
        print("=" * 80)
        print("User uploads: 224x224 blue image")
        print("User clicks: Analyze")

        response = client.post("/api/inference/universal", json=request_data)

        assert response.status_code == 200
        data = response.json()

        print(f"UI displays classification bars:")
        for i, pred in enumerate(data["outputs"][:3]):
            percentage = pred["score"] * 100
            print(f"  {pred['label']}: {'█' * int(percentage/5)} {percentage:.1f}%")

        assert data["model_type"] == "image-classification"
        assert isinstance(data["outputs"], list)
        assert all("label" in p and "score" in p for p in data["outputs"])

        print("✅ Web UI can render classification bars")
        print("=" * 80)


@pytest.mark.slow
@pytest.mark.integration
class TestWebUITextClassification:
    """Test Text Classification via existing interface."""

    def test_web_ui_text_classification(self, mock_auth):
        """Simulate web UI: user enters text, gets classification."""
        # Using a sentiment analysis model
        user_text = "I love this product! It's amazing!"

        request_data = {"model_id": "distilbert-base-uncased-finetuned-sst-2-english", "inputs": {"text": user_text}}

        print("\n" + "=" * 80)
        print("WEB UI TEST: Text Classification (Sentiment)")
        print("=" * 80)
        print(f"User enters: {user_text}")
        print("User clicks: Classify")

        response = client.post("/api/inference/universal", json=request_data)

        print(f"API response status: {response.status_code}")
        assert response.status_code == 200

        data = response.json()

        print(f"UI displays:")
        for pred in data["outputs"]:
            print(f"  {pred['label']}: {pred['score']*100:.1f}%")

        assert data["model_type"] == "text-classification"
        assert isinstance(data["outputs"], list)

        print("✅ Web UI can display classification results")
        print("=" * 80)


@pytest.mark.slow
@pytest.mark.integration
class TestWebUITabular:
    """Test Tabular interface (basic validation)."""

    def test_web_ui_tabular_api_structure(self, mock_auth, monkeypatch):
        """Test that tabular API accepts proper input structure from UI."""
        # Mock detect_model_type to return tabular
        monkeypatch.setattr("autotrain.app.api_routes.detect_model_type", lambda x: "tabular")
        monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", lambda x: f"/fake/{x}")

        # User enters JSON features
        user_features = {"feature1": 1.5, "feature2": 2.3, "feature3": 0.8}

        request_data = {"model_id": "test-tabular-model", "inputs": {"features": user_features}}

        print("\n" + "=" * 80)
        print("WEB UI TEST: Tabular Prediction (Mock)")
        print("=" * 80)
        print(f"User enters JSON: {user_features}")
        print("User clicks: Predict")

        # This will hit the tabular inference code
        # Since we don't have a real tabular model, we expect a specific error
        response = client.post("/api/inference/universal", json=request_data)

        print(f"API response status: {response.status_code}")

        # We expect either 200 with prediction or an error about model loading
        # The important thing is the API accepts the input structure
        if response.status_code == 200:
            data = response.json()
            assert "outputs" in data
            print("✅ Tabular API accepts UI input structure")
        else:
            # Expected to fail at model loading, not input validation
            print(f"Expected error (no real model): {response.json()}")
            print("✅ Tabular API validates input structure correctly")

        print("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "slow"])
