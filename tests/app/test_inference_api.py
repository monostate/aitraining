import json
import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from autotrain.app.app import app


class TestInferenceAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Mock authentication if necessary, or assume it passes in tests if dependencies are overridden
        # The app uses api_auth dependency. We might need to override it or provide headers.
        self.headers = {"Authorization": "Bearer hf_test_token"}

    @patch("autotrain.app.api_routes.token_verification")
    @patch("autotrain.app.api_routes.get_models_dir")
    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("autotrain.app.api_routes.detect_model_type")
    @patch("autotrain.app.api_routes.get_model_metadata")
    def test_models_list(self, mock_metadata, mock_detect, mock_isdir, mock_listdir, mock_get_dir, mock_auth):
        mock_auth.return_value = {"name": "test_user"}
        mock_get_dir.return_value = "/tmp/models"
        mock_listdir.return_value = ["model1", "random_file"]
        mock_isdir.side_effect = lambda x: x.endswith("model1")
        mock_detect.return_value = "text-classification"
        mock_metadata.return_value = {"created_at": 1234567890}

        with patch("os.path.exists", return_value=True):
            response = self.client.get("/api/models/list", headers=self.headers)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], "model1")
        self.assertEqual(data[0]["type"], "text-classification")

    @patch("autotrain.app.api_routes.token_verification")
    @patch("autotrain.app.api_routes.validate_model_path")
    @patch("autotrain.app.api_routes.detect_model_type")
    @patch("autotrain.app.api_routes.get_cached_pipeline")
    def test_universal_inference_text(self, mock_pipeline, mock_detect, mock_validate, mock_auth):
        mock_auth.return_value = {"name": "test_user"}
        mock_validate.return_value = "/tmp/models/model1"
        mock_detect.return_value = "text-classification"

        mock_pipe_instance = MagicMock()
        mock_pipe_instance.return_value = [{"label": "POSITIVE", "score": 0.99}]
        mock_pipeline.return_value = mock_pipe_instance

        payload = {"model_id": "model1", "inputs": {"text": "I love AutoTrain"}}

        with patch("torch.cuda.is_available", return_value=False):
            response = self.client.post("/api/inference/universal", json=payload, headers=self.headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_type"], "text-classification")
        self.assertEqual(response.json()["outputs"][0]["label"], "POSITIVE")

    @patch("autotrain.app.api_routes.token_verification")
    @patch("autotrain.app.api_routes.validate_model_path")
    def test_conversation_endpoints(self, mock_validate, mock_auth):
        mock_auth.return_value = {"name": "test_user"}
        mock_validate.return_value = "/tmp/models/model1"

        # Test Save
        with patch("os.makedirs") as mock_makedirs, patch("builtins.open", unittest.mock.mock_open()) as mock_file:

            payload = {"timestamp": 123456, "messages": []}
            response = self.client.post("/api/conversations/save?model_id=model1", json=payload, headers=self.headers)
            self.assertEqual(response.status_code, 200)
            mock_file.assert_called_with("/tmp/models/model1/conversations/123456.json", "w")

    @patch("autotrain.app.api_routes.token_verification")
    def test_path_traversal(self, mock_auth):
        mock_auth.return_value = {"name": "test_user"}

        payload = {"model_id": "../../../etc/passwd", "inputs": {"text": "test"}}
        response = self.client.post("/api/inference/universal", json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 400)  # Should fail validation


if __name__ == "__main__":
    unittest.main()
