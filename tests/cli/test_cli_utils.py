import sys
from types import SimpleNamespace

import pytest

from autotrain.cli.utils import should_launch_wizard


def test_should_not_launch_wizard_when_model_provided(monkeypatch: pytest.MonkeyPatch):
    """Ensure CLI skips wizard when critical args are provided explicitly."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aitraining",
            "llm",
            "--train",
            "--project-name",
            "mygemma",
            "--data-path",
            "tatsu-lab/alpaca",
            "--model",
            "google/gemma-3-270m",
        ],
    )

    args = SimpleNamespace(
        interactive=False,
        train=True,
        project_name="mygemma",
        data_path="tatsu-lab/alpaca",
        model="google/gemma-3-270m",
    )

    assert should_launch_wizard(args, "llm") is False


def test_should_launch_wizard_when_defaults_used(monkeypatch: pytest.MonkeyPatch):
    """Wizard should launch when required params are missing or still placeholders."""
    monkeypatch.setattr(sys, "argv", ["aitraining", "llm", "--train"])

    args = SimpleNamespace(
        interactive=False,
        train=True,
        project_name="project-name",
        data_path=None,
        model="google/gemma-3-270m",
    )

    assert should_launch_wizard(args, "llm") is True
