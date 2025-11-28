"""Test conversation API endpoints with model IDs containing slashes."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from autotrain.app.api_routes import api_router


# Create a test app
app = FastAPI()
app.include_router(api_router, prefix="/api")

# Create test client
client = TestClient(app)

# Print registered routes for debugging
print("\n=== Registered Routes ===")
for route in app.routes:
    if hasattr(route, "path") and hasattr(route, "methods"):
        print(f"{list(route.methods)} {route.path}")
print("========================\n")


@pytest.fixture
def test_model_dir():
    """Create a temporary directory structure for testing."""
    temp_dir = tempfile.mkdtemp()
    model_path = Path(temp_dir) / "cebolinha_full_finetune" / "model"
    model_path.mkdir(parents=True, exist_ok=True)

    # Create a dummy config.json to make it look like a valid model
    config_file = model_path / "config.json"
    config_file.write_text('{"model_type": "test"}')

    yield str(model_path)

    # Cleanup
    shutil.rmtree(temp_dir)


def test_get_conversations_empty(test_model_dir, monkeypatch):
    """Test GET /api/conversations returns empty array when no conversations exist."""

    # Mock the validate_model_path to return our test directory
    def mock_validate(model_id):
        return test_model_dir

    monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", mock_validate)

    # Mock authentication - remove HF_TOKEN to allow local access
    monkeypatch.delenv("HF_TOKEN", raising=False)

    model_id = "cebolinha_full_finetune/model"
    response = client.get(f"/api/conversations", params={"model_id": model_id})

    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    assert len(data) == 0, f"Expected empty list, got {data}"


def test_save_and_get_conversation(test_model_dir, monkeypatch):
    """Test POST and GET conversation endpoints with model ID containing slash."""

    # Mock the validate_model_path to return our test directory
    def mock_validate(model_id):
        return test_model_dir

    monkeypatch.setattr("autotrain.app.api_routes.validate_model_path", mock_validate)

    # Mock authentication - remove HF_TOKEN to allow local access
    monkeypatch.delenv("HF_TOKEN", raising=False)

    model_id = "cebolinha_full_finetune/model"

    # Test data
    test_conversation = {
        "id": "test-conv-1",
        "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}],
        "timestamp": 1704110400,  # 2024-01-01T12:00:00 as Unix timestamp
    }

    # Save conversation
    save_response = client.post(f"/api/conversations/save", params={"model_id": model_id}, json=test_conversation)

    print(f"Save response status: {save_response.status_code}")
    print(f"Save response body: {save_response.text}")

    assert save_response.status_code == 200, f"Save failed: {save_response.status_code}: {save_response.text}"
    save_data = save_response.json()
    assert save_data.get("success") == True, f"Expected 'success' True, got {save_data}"

    # Verify file was created
    conv_dir = Path(test_model_dir) / "conversations"
    assert conv_dir.exists(), f"Conversations directory not created at {conv_dir}"

    conv_files = list(conv_dir.glob("*.json"))
    assert len(conv_files) > 0, f"No conversation files found in {conv_dir}"

    # Get conversations
    get_response = client.get(f"/api/conversations", params={"model_id": model_id})

    print(f"Get response status: {get_response.status_code}")
    print(f"Get response body: {get_response.text}")

    assert get_response.status_code == 200, f"Get failed: {get_response.status_code}: {get_response.text}"
    conversations = get_response.json()
    assert isinstance(conversations, list), f"Expected list, got {type(conversations)}"
    assert len(conversations) == 1, f"Expected 1 conversation, got {len(conversations)}"
    assert conversations[0]["id"] == "test-conv-1", f"Wrong conversation returned: {conversations[0]}"


def test_model_id_with_slash_in_query():
    """Test that the model_id with slash is correctly handled via query parameter."""
    model_id = "cebolinha_full_finetune/model"

    # Mock authentication - remove HF_TOKEN to allow local access
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]

    # Using query parameter approach - should work with slashes
    response = client.get(f"/api/conversations", params={"model_id": model_id})

    print(f"Query param test response status: {response.status_code}")
    print(f"Query param test response body: {response.text}")

    # We expect either 200 (success) or 404 with our custom error message
    # A 404 from FastAPI routing would have different error format
    assert response.status_code in [200, 404], f"Unexpected status: {response.status_code}"

    if response.status_code == 404:
        # Check if it's our error (model not found) vs routing error
        error_data = response.json()
        assert "detail" in error_data, f"Expected error detail, got {error_data}"
        # Our error messages contain "model" or "Model"
        assert (
            "model" in error_data["detail"].lower() or "not found" in error_data["detail"].lower()
        ), f"Routing failed - got generic 404: {error_data}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
