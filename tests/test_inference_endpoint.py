#!/usr/bin/env python3
"""
Test script for the new inference API endpoint.
This tests the implementation without needing a running server.
"""

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from autotrain.app.api_routes import InferenceRequest, InferenceResponse


def test_schemas():
    """Test that request and response schemas are properly defined."""

    # Test InferenceRequest with required fields
    request = InferenceRequest(
        model_path="/path/to/model", prompts=["Hello, how are you?", "What is the weather like?"]
    )

    assert request.model_path == "/path/to/model"
    assert len(request.prompts) == 2
    assert request.max_new_tokens == 100  # Default value
    assert request.temperature == 0.7  # Default value
    assert request.top_p == 0.95  # Default value
    assert request.top_k == 50  # Default value
    assert request.do_sample == True  # Default value
    assert request.device is None  # Default value

    # Test InferenceRequest with custom values
    request2 = InferenceRequest(
        model_path="/another/model",
        prompts=["Test prompt"],
        max_new_tokens=200,
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        do_sample=False,
        device="cuda",
    )

    assert request2.max_new_tokens == 200
    assert request2.temperature == 0.5
    assert request2.device == "cuda"

    # Test InferenceResponse
    response = InferenceResponse(
        outputs=["Generated text 1", "Generated text 2"], model_path="/path/to/model", num_prompts=2
    )

    assert len(response.outputs) == 2
    assert response.model_path == "/path/to/model"
    assert response.num_prompts == 2

    print("✓ All schema tests passed!")


def test_endpoint_signature():
    """Test that the endpoint is properly defined."""
    from autotrain.app.api_routes import api_router

    # Check that the endpoint exists in the router
    routes = [route.path for route in api_router.routes]
    assert "/llm/inference" in routes, f"Endpoint not found. Available routes: {routes}"

    # Find the inference endpoint
    for route in api_router.routes:
        if route.path == "/llm/inference":
            # Check it's a POST method
            assert "POST" in route.methods, f"Expected POST method, got: {route.methods}"
            # Check response model
            assert route.response_model == InferenceResponse
            print("✓ Endpoint signature tests passed!")
            break


if __name__ == "__main__":
    print("Testing inference API endpoint implementation...")
    print("-" * 50)

    test_schemas()
    test_endpoint_signature()

    print("-" * 50)
    print("✅ All tests passed successfully!")
    print("\nThe inference endpoint is available at: POST /api/llm/inference")
    print("It accepts:")
    print("  - model_path (str): Path to the trained model")
    print("  - prompts (List[str]): List of text prompts for generation")
    print("  - Optional parameters: max_new_tokens, temperature, top_p, top_k, do_sample, device")
    print("\nIt returns:")
    print("  - outputs (List[str]): Generated text for each prompt")
    print("  - model_path (str): Echo of the model path used")
    print("  - num_prompts (int): Number of prompts processed")
