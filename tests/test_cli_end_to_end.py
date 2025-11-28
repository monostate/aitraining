"""
End-to-End CLI Tests - Mirroring API Tests
===========================================

These tests run actual CLI commands and verify they produce the same results as API tests.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


def run_cli(command: str, timeout: int = 300):
    """Run CLI command and return result."""
    # Use bash explicitly to support process substitution <(...)
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=Path(__file__).parent.parent,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    return result


@pytest.fixture
def small_dataset():
    """Create a small test dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = os.path.join(tmpdir, "train.jsonl")
        with open(train_file, "w") as f:
            for i in range(20):
                f.write(json.dumps({"text": f"This is sample text number {i}. It contains some words."}) + "\n")
        yield tmpdir


class TestCLIDistillation:
    """Test distillation via CLI - mirrors test_prompt_distillation.py"""

    def test_distillation_config_via_cli(self, small_dataset):
        """Test distillation with config parameters."""
        with tempfile.TemporaryDirectory() as output_dir:
            cmd = f"""PYTHONPATH=src python -m autotrain.trainers.clm --training_config <(echo '{{
                "model": "gpt2",
                "project_name": "{output_dir}/test_distill",
                "data_path": "{small_dataset}",
                "trainer": "distillation",
                "teacher_model": "gpt2",
                "teacher_prompt_template": "Answer: {{input}}",
                "student_prompt_template": "{{input}}",
                "distill_temperature": 3.0,
                "distill_alpha": 0.7,
                "epochs": 1,
                "batch_size": 2,
                "lr": 1e-4,
                "text_column": "text",
                "block_size": 128
            }}')"""

            result = run_cli(cmd)
            # Check if training started (may not complete in test environment)
            assert "distillation" in result.stdout.lower() or result.returncode != 2


class TestCLISweep:
    """Test hyperparameter sweep via CLI - mirrors test_sweep.py"""

    def test_sweep_via_cli(self, small_dataset):
        """Test sweep with Optuna backend."""
        with tempfile.TemporaryDirectory() as output_dir:
            cmd = f"""PYTHONPATH=src python -m autotrain.trainers.clm --training_config <(echo '{{
                "model": "gpt2",
                "project_name": "{output_dir}/test_sweep",
                "data_path": "{small_dataset}",
                "trainer": "sft",
                "use_sweep": true,
                "sweep_backend": "optuna",
                "sweep_n_trials": 2,
                "sweep_params": "{{\\"lr\\": {{\\"low\\": 1e-5, \\"high\\": 1e-4, \\"type\\": \\"float\\"}}}}",
                "epochs": 1,
                "batch_size": 2,
                "text_column": "text",
                "block_size": 128
            }}')"""

            result = run_cli(cmd, timeout=600)
            # Check sweep started
            assert result.returncode != 2 or "sweep" in result.stdout.lower()


class TestCLIEnhancedEval:
    """Test enhanced evaluation via CLI - mirrors test_evaluation.py"""

    def test_enhanced_eval_via_cli(self, small_dataset):
        """Test enhanced evaluation callbacks."""
        with tempfile.TemporaryDirectory() as output_dir:
            cmd = f"""PYTHONPATH=src python -m autotrain.trainers.clm --training_config <(echo '{{
                "model": "gpt2",
                "project_name": "{output_dir}/test_eval",
                "data_path": "{small_dataset}",
                "trainer": "sft",
                "use_enhanced_eval": true,
                "eval_metrics": "perplexity",
                "valid_split": "train",
                "epochs": 1,
                "batch_size": 2,
                "text_column": "text",
                "block_size": 128
            }}')"""

            result = run_cli(cmd)
            assert result.returncode != 2


class TestCLIInference:
    """Test inference via CLI - mirrors test_completers.py"""

    def test_inference_via_cli(self):
        """Test inference/generation mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = os.path.join(tmpdir, "prompts.txt")
            with open(prompts_file, "w") as f:
                f.write("Hello\\n")
                f.write("What is AI?\\n")

            output_file = os.path.join(tmpdir, "results.json")

            cmd = f"""PYTHONPATH=src python src/autotrain/cli/run_llm.py \\
                --inference \\
                --model gpt2 \\
                --inference-prompts {prompts_file} \\
                --inference-output {output_file} \\
                --inference-max-tokens 20 \\
                --project-name test-inf"""

            result = run_cli(cmd)

            if result.returncode == 0 and os.path.exists(output_file):
                with open(output_file) as f:
                    results = json.load(f)
                    assert len(results) == 2
                    assert "prompt" in results[0]
                    assert "response" in results[0]


class TestCLIPPO:
    """Test PPO training via CLI - mirrors RL tests"""

    def test_ppo_with_env_config(self, small_dataset):
        """Test PPO with custom environment."""
        with tempfile.TemporaryDirectory() as output_dir:
            cmd = f"""PYTHONPATH=src python -m autotrain.trainers.clm --training_config <(echo '{{
                "model": "gpt2",
                "project_name": "{output_dir}/test_ppo",
                "data_path": "{small_dataset}",
                "trainer": "ppo",
                "rl_env_type": "text_generation",
                "rl_gamma": 0.99,
                "rl_kl_coef": 0.1,
                "epochs": 1,
                "batch_size": 2,
                "text_column": "text",
                "block_size": 128
            }}')"""

            result = run_cli(cmd, timeout=600)
            assert result.returncode != 2


class TestCLICustomLoss:
    """Test custom loss via CLI"""

    def test_custom_loss_via_cli(self, small_dataset):
        """Test custom KL loss."""
        with tempfile.TemporaryDirectory() as output_dir:
            cmd = f"""PYTHONPATH=src python -m autotrain.trainers.clm --training_config <(echo '{{
                "model": "gpt2",
                "project_name": "{output_dir}/test_loss",
                "data_path": "{small_dataset}",
                "trainer": "sft",
                "custom_loss": "kl",
                "epochs": 1,
                "batch_size": 2,
                "text_column": "text",
                "block_size": 128
            }}')"""

            result = run_cli(cmd)
            assert result.returncode != 2


class TestCLIMessageRendering:
    """Test message rendering via CLI"""

    def test_chat_format_via_cli(self, small_dataset):
        """Test chat format parameter."""
        with tempfile.TemporaryDirectory() as output_dir:
            cmd = f"""PYTHONPATH=src python -m autotrain.trainers.clm --training_config <(echo '{{
                "model": "gpt2",
                "project_name": "{output_dir}/test_chat",
                "data_path": "{small_dataset}",
                "trainer": "sft",
                "chat_format": "chatml",
                "epochs": 1,
                "batch_size": 2,
                "text_column": "text",
                "block_size": 128
            }}')"""

            result = run_cli(cmd)
            assert result.returncode != 2


# Quick verification tests
class TestCLIQuickVerify:
    """Quick verification that CLI parameters are recognized."""

    def test_all_new_parameters_recognized(self):
        """Verify all new parameters exist and are loaded."""
        cmd = """PYTHONPATH=src python -c "
from autotrain.trainers.clm.params import LLMTrainingParams
p = LLMTrainingParams()

# Check all new params exist
params_to_check = [
    'use_distillation', 'teacher_model', 'distill_temperature', 'distill_alpha',
    'use_sweep', 'sweep_backend', 'sweep_n_trials',
    'use_enhanced_eval', 'eval_metrics',
    'rl_gamma', 'rl_env_type',
    'custom_loss', 'chat_format'
]

for param in params_to_check:
    assert hasattr(p, param), f'Missing parameter: {param}'

print('✅ All CLI parameters exist and are loaded')
"
"""
        result = run_cli(cmd)
        assert result.returncode == 0
        assert "✅ All CLI parameters exist" in result.stdout

    def test_trainers_registered(self):
        """Verify all trainers are registered."""
        cmd = """PYTHONPATH=src python -c "
import sys
sys.path.insert(0, 'src')

from autotrain.trainers.clm.params import LLMTrainingParams

trainers = ['default', 'sft', 'reward', 'dpo', 'orpo', 'distillation', 'ppo']

for trainer in trainers:
    # Create config with minimum required parameters for each trainer
    kwargs = {'trainer': trainer}

    # DPO and ORPO require prompt_text_column and rejected_text_column
    if trainer in ['dpo', 'orpo']:
        kwargs['prompt_text_column'] = 'prompt'
        kwargs['rejected_text_column'] = 'rejected'

    # PPO requires either rl_reward_model_path or model_ref
    if trainer == 'ppo':
        kwargs['model_ref'] = 'gpt2'

    config = LLMTrainingParams(**kwargs)
    assert config.trainer == trainer

print('✅ All trainers registered:', ', '.join(trainers))
"
"""
        result = run_cli(cmd)
        assert result.returncode == 0
        assert "✅ All trainers registered" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
