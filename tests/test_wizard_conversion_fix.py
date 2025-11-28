#!/usr/bin/env python
"""
Test to verify the wizard dataset conversion fix.

This test simulates the wizard flow where:
1. User accepts dataset conversion (auto_convert_dataset=True)
2. llm_munge_data is called during project creation
3. Conversion should create 'text' column and update data_path
"""

import sys
import os
import json
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.project import llm_munge_data


def test_wizard_conversion_flow():
    """Test that wizard conversion flow works correctly."""
    print("=" * 60)
    print("TEST: Wizard Dataset Conversion Flow")
    print("=" * 60)
    
    # Create a test project directory
    test_project_name = "test_wizard_conversion"
    test_project_path = Path(test_project_name)
    
    # Clean up any existing test project
    if test_project_path.exists():
        print(f"Cleaning up existing test project: {test_project_path}")
        shutil.rmtree(test_project_path)
    
    try:
        # Simulate wizard answers - user accepts conversion
        print("\n1. Creating params with auto_convert_dataset=True (simulating wizard)...")
        params = LLMTrainingParams(
            model="google/gemma-3-270m-it",
            data_path="monostate/cebolinha-sft",  # Q&A format dataset
            project_name=test_project_name,
            auto_convert_dataset=True,  # Wizard sets this to True
            trainer="sft",
            train_split="train[:10]",  # Small sample for testing
            valid_split=None,
            push_to_hub=False,
            chat_template="tokenizer",  # Use model's default template
        )
        
        print(f"   ✓ Params created")
        print(f"   ✓ auto_convert_dataset: {params.auto_convert_dataset}")
        print(f"   ✓ data_path: {params.data_path}")
        print(f"   ✓ text_column: {params.text_column}")
        
        # Simulate what happens in AutoTrainProject.create() -> _process_params_data()
        print("\n2. Calling llm_munge_data() (simulating project creation)...")
        result_params = llm_munge_data(params, local=True)
        
        # Verify conversion happened
        print("\n3. Verifying conversion results...")
        
        # Check that data_path was updated
        converted_dir = Path(result_params.project_name) / "data_converted"
        assert converted_dir.exists(), f"Converted directory not found: {converted_dir}"
        print(f"   ✓ Converted directory exists: {converted_dir}")
        
        # Check that data_path points to converted files
        assert result_params.data_path == str(converted_dir), \
            f"data_path not updated! Expected: {converted_dir}, Got: {result_params.data_path}"
        print(f"   ✓ data_path updated to: {result_params.data_path}")
        
        # Check that text_column is set to "text"
        assert result_params.text_column == "text", \
            f"text_column not set correctly! Expected: 'text', Got: '{result_params.text_column}'"
        print(f"   ✓ text_column set to: {result_params.text_column}")
        
        # Check that converted files exist
        train_file = converted_dir / f"{result_params.train_split}.jsonl"
        assert train_file.exists(), f"Converted training file not found: {train_file}"
        print(f"   ✓ Converted training file exists: {train_file}")
        
        # Verify the converted file has 'text' column
        print("\n4. Verifying converted file structure...")
        with open(train_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            first_sample = json.loads(first_line)
            
            # Check that 'text' column exists
            assert 'text' in first_sample, \
                f"'text' column missing in converted file! Columns: {list(first_sample.keys())}"
            print(f"   ✓ 'text' column exists in converted file")
            
            # Check that 'text' column has content
            assert first_sample['text'] and len(first_sample['text']) > 0, \
                f"'text' column is empty!"
            print(f"   ✓ 'text' column has content (length: {len(first_sample['text'])})")
            
            # Show sample
            text_preview = first_sample['text'][:100] + "..." if len(first_sample['text']) > 100 else first_sample['text']
            print(f"   ✓ Sample text preview: {text_preview}")
            
            # Optionally check for messages column (should also exist)
            if 'messages' in first_sample:
                print(f"   ✓ 'messages' column also exists (for reference)")
        
        # Check training_params.json if it exists
        params_file = Path(result_params.project_name) / "training_params.json"
        if params_file.exists():
            print("\n5. Checking training_params.json...")
            with open(params_file, 'r') as f:
                saved_params = json.load(f)
                
            # Note: auto_convert_dataset might be False in saved params (that's okay, conversion already happened)
            # But data_path should point to converted files
            if 'data_path' in saved_params:
                print(f"   ✓ data_path in saved params: {saved_params['data_path']}")
            if 'text_column' in saved_params:
                print(f"   ✓ text_column in saved params: {saved_params['text_column']}")
        
        print("\n" + "=" * 60)
        print("✅ ALL CHECKS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print(f"  • Conversion successful")
        print(f"  • 'text' column created")
        print(f"  • data_path updated to: {result_params.data_path}")
        print(f"  • text_column set to: {result_params.text_column}")
        print(f"  • Converted files ready for training")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Optionally clean up (comment out to inspect files)
        # if test_project_path.exists():
        #     print(f"\nCleaning up test project: {test_project_path}")
        #     shutil.rmtree(test_project_path)
        pass


if __name__ == "__main__":
    success = test_wizard_conversion_flow()
    sys.exit(0 if success else 1)

