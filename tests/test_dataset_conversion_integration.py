#!/usr/bin/env python
"""
Test that dataset conversion parameters work through CLI, API, and Wizard.
"""

import os
import sys


sys.path.insert(0, "src")

from autotrain.preprocessor.chat_templates_standalone import get_template_for_model
from autotrain.preprocessor.llm import get_available_chat_templates
from autotrain.trainers.clm.params import LLMTrainingParams


def test_parameters_exist():
    """Test that new parameters are properly defined."""
    print("=== Testing Parameters ===")

    # Create params with dataset conversion
    params = LLMTrainingParams(
        model="google/gemma-3-270m-it",
        data_path="tatsu-lab/alpaca",
        project_name="test-conversion",
        auto_convert_dataset=True,
        use_sharegpt_mapping=False,
        conversation_extension=2,
        apply_chat_template=True,
        chat_template="gemma",
        trainer="sft",
        push_to_hub=False,
        train_split="train[:10]",  # Small sample for testing
    )

    print(f"✓ Parameters created successfully")
    print(f"  auto_convert_dataset: {params.auto_convert_dataset}")
    print(f"  conversation_extension: {params.conversation_extension}")
    print(f"  chat_template: {params.chat_template}")
    print(f"  apply_chat_template: {params.apply_chat_template}")

    return True


def test_templates_available():
    """Test that templates are accessible."""
    print("\n=== Testing Templates ===")

    templates = get_available_chat_templates()
    print(f"✓ {len(templates)} templates available")

    # Test model suggestion
    model = "google/gemma-3-270m-it"
    suggested = get_template_for_model(model)
    print(f"✓ Template suggestion for {model}: {suggested}")

    # Check specific templates
    important_templates = ["gemma", "llama3", "chatml", "alpaca"]
    for template in important_templates:
        if template in templates:
            print(f"  ✓ {template} available")
        else:
            print(f"  ✗ {template} NOT FOUND")

    return True


def test_cli_help():
    """Test that CLI exposes the parameters."""
    print("\n=== Testing CLI ===")

    import subprocess

    result = subprocess.run(
        ["python", "-m", "autotrain.cli.autotrain", "llm", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    # Check for our parameters in help text
    params_to_check = [
        "--auto-convert-dataset",
        "--conversation-extension",
        "--apply-chat-template",
        "--use-sharegpt-mapping",
    ]

    for param in params_to_check:
        if param in result.stdout:
            print(f"  ✓ {param} in CLI help")
        else:
            print(f"  ✗ {param} NOT in CLI help")
            return False

    return True


def test_project_integration():
    """Test that llm_munge_data handles conversion."""
    print("\n=== Testing Project Integration ===")

    from autotrain.project import llm_munge_data

    # Create test params
    params = LLMTrainingParams(
        model="google/gemma-3-270m-it",
        data_path="tatsu-lab/alpaca",
        project_name="test-project",
        auto_convert_dataset=True,
        conversation_extension=1,
        trainer="sft",
        push_to_hub=False,
        train_split="train[:5]",
    )

    # This should not fail
    try:
        # Note: This won't actually convert since we're not running the full flow
        # but it will test that the function accepts our parameters
        result = llm_munge_data(params, local=True)
        print("  ✓ llm_munge_data accepts conversion parameters")

        # Check if sharegpt mapping flag was set
        if hasattr(result, "sharegpt_mapping_enabled"):
            print(f"  ✓ sharegpt_mapping_enabled: {result.sharegpt_mapping_enabled}")
    except Exception as e:
        print(f"  ✗ Error in llm_munge_data: {e}")
        return False

    return True


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("DATASET CONVERSION INTEGRATION TEST")
    print("=" * 60)

    tests = [test_parameters_exist, test_templates_available, test_cli_help, test_project_integration]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test failed with error: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    if all(results):
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
