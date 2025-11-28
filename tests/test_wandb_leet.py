import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.utils import run_training


class TestWandbLeet(unittest.TestCase):
    @patch("autotrain.utils.subprocess.Popen")
    @patch("sys.stdout.isatty")
    @patch("autotrain.commands.launch_command")
    def test_run_training_launches_leet(self, mock_launch, mock_isatty, mock_popen):
        mock_isatty.return_value = True
        mock_launch.return_value = ["accelerate", "launch", "script.py"]

        params = LLMTrainingParams(
            model="gpt2",
            project_name="test_project_leet",
            data_path="data",
            log="wandb",
            wandb_visualizer=True,
            token="test_token",
        )

        # Mock process
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_process.pid = 123
        mock_popen.return_value = mock_process

        # Run training
        run_training(params.model_dump_json(), task_id=9, wait=True)

        # Verify Popen calls
        # Filter calls to find leet
        leet_calls = [
            call
            for call in mock_popen.call_args_list
            if isinstance(call[0][0], list) and len(call[0][0]) >= 3 and call[0][0][:3] == ["wandb", "beta", "leet"]
        ]

        self.assertTrue(len(leet_calls) > 0, "W&B LEET should be launched")
        # Project path is normalized to absolute path in trainings/ dir
        # leet_calls[0][0] is positional args tuple, leet_calls[0][0][0] is the command list
        self.assertTrue(
            leet_calls[0][0][0][-1].endswith("test_project_leet"),
            f"Expected path ending in test_project_leet, got {leet_calls[0][0][0][-1]}",
        )

    @patch("autotrain.utils.subprocess.Popen")
    @patch("sys.stdout.isatty")
    @patch("autotrain.commands.launch_command")
    def test_run_training_no_leet_if_disabled(self, mock_launch, mock_isatty, mock_popen):
        mock_isatty.return_value = True
        mock_launch.return_value = ["accelerate", "launch", "script.py"]

        params = LLMTrainingParams(
            model="gpt2",
            project_name="test_project_no_leet",
            data_path="data",
            log="wandb",
            wandb_visualizer=False,  # Disabled explicitly
            token="test_token",
        )

        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        run_training(params.model_dump_json(), task_id=9, wait=True)

        leet_calls = [
            call
            for call in mock_popen.call_args_list
            if isinstance(call[0][0], list) and len(call[0][0]) >= 3 and call[0][0][:3] == ["wandb", "beta", "leet"]
        ]

        self.assertEqual(len(leet_calls), 0, "W&B LEET should NOT be launched")

    @patch("autotrain.utils.subprocess.Popen")
    @patch("sys.stdout.isatty")
    @patch("autotrain.commands.launch_command")
    def test_run_training_leet_default_logic(self, mock_launch, mock_isatty, mock_popen):
        # Test that if wandb_visualizer is not provided (None), it defaults to True if log=wandb
        mock_isatty.return_value = True
        mock_launch.return_value = ["accelerate", "launch", "script.py"]

        # Not providing wandb_visualizer explicitly
        params = LLMTrainingParams(
            model="gpt2", project_name="test_project_default", data_path="data", log="wandb", token="test_token"
        )

        # The validator logic runs on instantiation of LLMTrainingParams
        self.assertTrue(params.wandb_visualizer)

        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        run_training(params.model_dump_json(), task_id=9, wait=True)

        leet_calls = [
            call
            for call in mock_popen.call_args_list
            if isinstance(call[0][0], list) and len(call[0][0]) >= 3 and call[0][0][:3] == ["wandb", "beta", "leet"]
        ]

        self.assertTrue(len(leet_calls) > 0, "W&B LEET should be launched by default for log=wandb")

    @patch("autotrain.utils.subprocess.Popen")
    @patch("sys.stdout.isatty")
    @patch("autotrain.commands.launch_command")
    def test_wandb_env_and_command_hint(self, mock_launch, mock_isatty, mock_popen):
        mock_isatty.return_value = False  # force manual mode (no auto LEET launch)
        mock_launch.return_value = ["accelerate", "launch", "script.py"]

        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_process.pid = 456
        mock_popen.return_value = mock_process

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = os.path.join(tmpdir, "wandb_env_project")
            params = LLMTrainingParams(
                model="gpt2",
                project_name=project_dir,
                data_path="data",
                log="wandb",
                wandb_visualizer=True,
                wandb_token="super_secret",
                token="hf_token",
            )

            run_training(params.model_dump_json(), task_id=9, wait=True)

            # Training process should see WANDB env vars
            training_env = mock_popen.call_args_list[0][1]["env"]
            self.assertEqual(training_env.get("WANDB_DIR"), os.path.abspath(project_dir))
            self.assertEqual(training_env.get("WANDB_API_KEY"), "super_secret")

            # Log file should contain the rewatch command hint
            log_path = os.path.join(project_dir, "autotrain.log")
            with open(log_path, "r", encoding="utf-8") as handle:
                log_contents = handle.read()
            self.assertIn("wandb beta leet", log_contents)
            self.assertIn(os.path.abspath(project_dir), log_contents)

    @patch("autotrain.utils._terminate_process")
    @patch("autotrain.utils.subprocess.Popen")
    @patch("sys.stdout.isatty")
    @patch("autotrain.commands.launch_command")
    def test_run_training_keyboard_interrupt_cleans_processes(
        self, mock_launch, mock_isatty, mock_popen, mock_terminate
    ):
        mock_isatty.return_value = True
        mock_launch.return_value = ["accelerate", "launch", "script.py"]

        training_proc = MagicMock()
        training_proc.wait.side_effect = [KeyboardInterrupt(), 0]
        training_proc.pid = 999
        training_proc.poll.return_value = None

        leet_proc = MagicMock()
        leet_proc.wait.return_value = 0
        leet_proc.poll.return_value = None

        mock_popen.side_effect = [training_proc, leet_proc]

        params = LLMTrainingParams(
            model="gpt2",
            project_name="interrupt_project",
            data_path="data",
            log="wandb",
            wandb_visualizer=True,
            token="test_token",
        )

        with self.assertRaises(KeyboardInterrupt):
            run_training(params.model_dump_json(), task_id=9, wait=True)

        mock_terminate.assert_any_call(training_proc)
        mock_terminate.assert_any_call(leet_proc)
