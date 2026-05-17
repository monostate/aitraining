"""Tests for the AUTOTRAIN_FORCE_NUM_PROCESSES env var and its legacy alias."""

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Strip both env names and reset the deprecation latch before each test."""
    monkeypatch.delenv("AUTOTRAIN_FORCE_NUM_PROCESSES", raising=False)
    monkeypatch.delenv("AUTOTRAIN_FORCE_NUM_GPUS", raising=False)
    import autotrain.commands as commands_mod

    commands_mod._legacy_force_num_processes_warned = False


class TestGetForcedNumProcesses:
    def test_returns_none_when_unset(self):
        from autotrain.commands import get_forced_num_processes

        assert get_forced_num_processes() is None

    def test_reads_new_env_var(self, monkeypatch):
        from autotrain.commands import get_forced_num_processes

        monkeypatch.setenv("AUTOTRAIN_FORCE_NUM_PROCESSES", "3")
        assert get_forced_num_processes() == "3"

    def test_falls_back_to_legacy_env_var(self, monkeypatch):
        from autotrain.commands import get_forced_num_processes

        monkeypatch.setenv("AUTOTRAIN_FORCE_NUM_GPUS", "2")
        assert get_forced_num_processes() == "2"

    def test_new_var_wins_over_legacy(self, monkeypatch):
        from autotrain.commands import get_forced_num_processes

        monkeypatch.setenv("AUTOTRAIN_FORCE_NUM_PROCESSES", "4")
        monkeypatch.setenv("AUTOTRAIN_FORCE_NUM_GPUS", "1")
        assert get_forced_num_processes() == "4"

    def test_legacy_emits_one_time_deprecation_warning(self, monkeypatch):
        import autotrain.commands as commands_mod

        monkeypatch.setenv("AUTOTRAIN_FORCE_NUM_GPUS", "1")
        with patch.object(commands_mod.logger, "warning") as mock_warn:
            commands_mod.get_forced_num_processes()
            commands_mod.get_forced_num_processes()  # second call should NOT re-warn

        assert mock_warn.call_count == 1
        assert "deprecated" in mock_warn.call_args[0][0].lower()
        assert "AUTOTRAIN_FORCE_NUM_GPUS" in mock_warn.call_args[0][0]

    def test_new_var_does_not_emit_warning(self, monkeypatch):
        import autotrain.commands as commands_mod

        monkeypatch.setenv("AUTOTRAIN_FORCE_NUM_PROCESSES", "2")
        with patch.object(commands_mod.logger, "warning") as mock_warn:
            commands_mod.get_forced_num_processes()

        mock_warn.assert_not_called()

    def test_accepts_explicit_env_dict(self):
        from autotrain.commands import get_forced_num_processes

        env = {"AUTOTRAIN_FORCE_NUM_PROCESSES": "5"}
        assert get_forced_num_processes(env=env) == "5"

    def test_explicit_env_dict_falls_back_to_legacy(self):
        from autotrain.commands import get_forced_num_processes

        env = {"AUTOTRAIN_FORCE_NUM_GPUS": "7"}
        assert get_forced_num_processes(env=env) == "7"


class TestLaunchCommandHonorsBothNames:
    """End-to-end check that launch_command reads either env name."""

    def _make_config(self):
        from autotrain.trainers.clm.params import LLMTrainingParams

        return LLMTrainingParams(model="hf-internal-testing/tiny-random-LlamaForCausalLM", trainer="sft")

    def test_new_var_drives_num_processes(self, monkeypatch):
        from autotrain.commands import launch_command

        monkeypatch.setenv("AUTOTRAIN_FORCE_NUM_PROCESSES", "1")
        cmd = launch_command(self._make_config())
        # num_processes==1 takes the SINGLE_GPU path → no --multi_gpu
        assert "--multi_gpu" not in cmd

    def test_legacy_var_still_drives_num_processes(self, monkeypatch):
        from autotrain.commands import launch_command

        monkeypatch.setenv("AUTOTRAIN_FORCE_NUM_GPUS", "1")
        cmd = launch_command(self._make_config())
        assert "--multi_gpu" not in cmd
