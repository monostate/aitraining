#!/usr/bin/env python
"""Test script for parameter validators in LLMTrainingParams."""

import os
import sys


sys.path.insert(0, "/code/src")

from autotrain.trainers.clm.params import LLMTrainingParams


def test_ppo_params_warning():
    """Test 1: Should WARN when RL params are used on non-PPO trainer."""
    print("\n" + "=" * 60)
    print("TEST 1: RL params on non-PPO trainer (should warn)")
    print("=" * 60)
    try:
        config = LLMTrainingParams(
            trainer="sft", model="gpt2", data_path="data", rl_gamma=0.95  # PPO-specific param on SFT trainer
        )
        print("✓ Config created with warning (check logs above)")
        print(f"  trainer={config.trainer}, rl_gamma={config.rl_gamma}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    return True


def test_ppo_missing_reward_model():
    """Test 2: Should RAISE when PPO trainer is used without reward model."""
    print("\n" + "=" * 60)
    print("TEST 2: PPO without reward model (should raise)")
    print("=" * 60)
    try:
        config = LLMTrainingParams(
            trainer="ppo",
            model="gpt2",
            data_path="data",
            # Missing rl_reward_model_path
        )
        print("✗ Should have raised an error!")
        return False
    except ValueError as e:
        print("✓ Correctly raised ValueError:")
        print(f"  {str(e).split(chr(10))[1]}")  # Print first line of error
        return True
    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}")
        return False


def test_distillation_missing_teacher():
    """Test 3: Should RAISE when distillation is enabled without teacher model."""
    print("\n" + "=" * 60)
    print("TEST 3: Distillation without teacher (should raise)")
    print("=" * 60)
    try:
        config = LLMTrainingParams(
            trainer="sft",
            model="gpt2",
            data_path="data",
            use_distillation=True,
            # Missing teacher_model
        )
        print("✗ Should have raised an error!")
        return False
    except ValueError as e:
        print("✓ Correctly raised ValueError:")
        print(f"  {e}")
        return True
    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}")
        return False


def test_distillation_params_warning():
    """Test 4: Should WARN when distillation params are set without use_distillation."""
    print("\n" + "=" * 60)
    print("TEST 4: Distillation params without use_distillation (should warn)")
    print("=" * 60)
    try:
        config = LLMTrainingParams(
            trainer="sft",
            model="gpt2",
            data_path="data",
            use_distillation=False,
            distill_temperature=4.0,  # Non-default value
        )
        print("✓ Config created with warning (check logs above)")
        print(f"  use_distillation={config.use_distillation}, distill_temperature={config.distill_temperature}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    return True


def test_valid_ppo_config():
    """Test 5: Should work fine with valid PPO config."""
    print("\n" + "=" * 60)
    print("TEST 5: Valid PPO config with reward model")
    print("=" * 60)
    try:
        # Create a dummy reward model path
        import os

        os.makedirs("/tmp/reward_model", exist_ok=True)

        config = LLMTrainingParams(
            trainer="ppo",
            model="gpt2",
            data_path="data",
            rl_reward_model_path="/tmp/reward_model",
            rl_gamma=0.95,
            rl_kl_coef=0.2,
        )
        print("✓ Valid PPO config accepted")
        print(f"  trainer={config.trainer}, rl_reward_model_path={config.rl_reward_model_path}")
        print(f"  rl_gamma={config.rl_gamma}, rl_kl_coef={config.rl_kl_coef}")

        # Clean up
        os.rmdir("/tmp/reward_model")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    return True


def test_valid_distillation_config():
    """Test 6: Should work fine with valid distillation config."""
    print("\n" + "=" * 60)
    print("TEST 6: Valid distillation config")
    print("=" * 60)
    try:
        config = LLMTrainingParams(
            trainer="sft",
            model="gpt2",
            data_path="data",
            use_distillation=True,
            teacher_model="gpt2-large",
            distill_temperature=3.5,
        )
        print("✓ Valid distillation config accepted")
        print(f"  use_distillation={config.use_distillation}, teacher_model={config.teacher_model}")
        print(f"  distill_temperature={config.distill_temperature}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    return True


def test_dpo_with_model_ref():
    """Test 7: DPO should accept model_ref without warning."""
    print("\n" + "=" * 60)
    print("TEST 7: DPO with model_ref (should work)")
    print("=" * 60)
    try:
        config = LLMTrainingParams(trainer="dpo", model="gpt2", data_path="data", model_ref="gpt2-medium")
        print("✓ DPO config with model_ref accepted")
        print(f"  trainer={config.trainer}, model_ref={config.model_ref}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    return True


def main():
    """Run all validation tests."""
    print("\n" + "#" * 60)
    print("# PARAMETER VALIDATION TESTS")
    print("#" * 60)

    tests = [
        test_ppo_params_warning,
        test_ppo_missing_reward_model,
        test_distillation_missing_teacher,
        test_distillation_params_warning,
        test_valid_ppo_config,
        test_valid_distillation_config,
        test_dpo_with_model_ref,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            results.append(False)

    print("\n" + "#" * 60)
    print("# TEST SUMMARY")
    print("#" * 60)
    passed = sum(results)
    total = len(results)
    print(f"\nPassed: {passed}/{total} tests")

    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
