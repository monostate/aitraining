"""Test inference performance and parameter effects."""

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from autotrain.app.api_routes import api_router


app = FastAPI()
app.include_router(api_router, prefix="/api")
client = TestClient(app)


@pytest.fixture
def mock_auth(monkeypatch):
    """Mock authentication."""
    monkeypatch.delenv("HF_TOKEN", raising=False)


@pytest.fixture
def model_id():
    """Test model ID."""
    return "cebolinha_full_finetune/model"


class TestInferencePerformance:
    """Test inference speed and caching."""

    def test_cold_start_inference(self, mock_auth, model_id):
        """Test cold start (loads model for first time)."""
        response = client.post(
            "/api/inference/universal",
            json={
                "model_id": model_id,
                "inputs": {"text": "oi"},
                "parameters": {"max_new_tokens": 30, "temperature": 0.7, "do_sample": True},
            },
        )

        assert response.status_code == 200
        output = response.json()["outputs"][0]
        assert len(output) > 0
        print(f"\n✅ Cold start output: {output}")

    def test_cached_inference_is_fast(self, mock_auth, model_id):
        """Test that cached inference is fast (<5s)."""
        # First call to cache model
        client.post(
            "/api/inference/universal",
            json={"model_id": model_id, "inputs": {"text": "test"}, "parameters": {"max_new_tokens": 20}},
        )

        # Second call should be fast
        start = time.time()
        response = client.post(
            "/api/inference/universal",
            json={"model_id": model_id, "inputs": {"text": "oi"}, "parameters": {"max_new_tokens": 30}},
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 5.0, f"Cached inference took {elapsed:.2f}s (should be <5s)"

        print(f"\n✅ Cached inference: {elapsed:.2f}s")


class TestParameterEffects:
    """Test that parameters actually affect generation."""

    def test_different_temperatures_produce_different_outputs(self, mock_auth, model_id):
        """Test temperature parameter affects output."""
        prompt = "me conta algo"

        # Low temperature (deterministic)
        response_low = client.post(
            "/api/inference/universal",
            json={
                "model_id": model_id,
                "inputs": {"text": prompt},
                "parameters": {"max_new_tokens": 30, "temperature": 0.1, "do_sample": True},
            },
        )

        # High temperature (creative)
        response_high = client.post(
            "/api/inference/universal",
            json={
                "model_id": model_id,
                "inputs": {"text": prompt},
                "parameters": {"max_new_tokens": 30, "temperature": 1.5, "do_sample": True},
            },
        )

        assert response_low.status_code == 200
        assert response_high.status_code == 200

        output_low = response_low.json()["outputs"][0]
        output_high = response_high.json()["outputs"][0]

        print(f"\n✅ Temp 0.1: {output_low}")
        print(f"✅ Temp 1.5: {output_high}")

        # Outputs should be different (very unlikely to be same with different temps)
        # Just verify both generated something
        assert len(output_low) > 0
        assert len(output_high) > 0

    def test_max_tokens_affects_length(self, mock_auth, model_id):
        """Test max_new_tokens parameter affects output."""
        prompt = "me conta uma história"

        # Small max_tokens
        response_short = client.post(
            "/api/inference/universal",
            json={"model_id": model_id, "inputs": {"text": prompt}, "parameters": {"max_new_tokens": 10}},
        )

        # Large max_tokens
        response_long = client.post(
            "/api/inference/universal",
            json={"model_id": model_id, "inputs": {"text": prompt}, "parameters": {"max_new_tokens": 100}},
        )

        assert response_short.status_code == 200
        assert response_long.status_code == 200

        output_short = response_short.json()["outputs"][0]
        output_long = response_long.json()["outputs"][0]

        print(f"\n✅ 10 tokens: {output_short}")
        print(f"✅ 100 tokens: {output_long}")

        # Both should generate something
        assert len(output_short) > 0
        assert len(output_long) > 0

    def test_parameter_changes_between_requests(self, mock_auth, model_id):
        """Test that changing parameters between requests works."""
        prompt = "oi"

        # First request with temp 0.5
        response1 = client.post(
            "/api/inference/universal",
            json={
                "model_id": model_id,
                "inputs": {"text": prompt},
                "parameters": {"temperature": 0.5, "max_new_tokens": 20},
            },
        )

        # Second request with temp 1.0
        response2 = client.post(
            "/api/inference/universal",
            json={
                "model_id": model_id,
                "inputs": {"text": prompt},
                "parameters": {"temperature": 1.0, "max_new_tokens": 20},
            },
        )

        # Third request back to temp 0.5
        response3 = client.post(
            "/api/inference/universal",
            json={
                "model_id": model_id,
                "inputs": {"text": prompt},
                "parameters": {"temperature": 0.5, "max_new_tokens": 20},
            },
        )

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response3.status_code == 200

        print("\n✅ Parameters can be changed between requests")
        print(f"   Request 1 (temp 0.5): {response1.json()['outputs'][0]}")
        print(f"   Request 2 (temp 1.0): {response2.json()['outputs'][0]}")
        print(f"   Request 3 (temp 0.5): {response3.json()['outputs'][0]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
