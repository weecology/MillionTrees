#!/usr/bin/env python3
"""
Test script to validate the HiPerGator test release functionality.
This simulates some aspects of what would happen on HiPerGator.
"""

import subprocess
import tempfile
import os
import sys
from pathlib import Path

def test_report_generation():
    """Test that the dataset report generation works correctly."""
    print("Testing dataset report generation...")
    
    # Test with different scenarios
    test_cases = [
        {"args": [], "expected_exit": 0},
        {"args": ["--test-exit-code", "0"], "expected_exit": 0},
        {"args": ["--test-exit-code", "1"], "expected_exit": 0},
    ]
    
    for i, case in enumerate(test_cases):
        print(f"  Test case {i+1}: {case['args']}")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_output = f.name
        
        try:
            args = ["python", "generate_dataset_report.py", "--output", temp_output] + case["args"]
            result = subprocess.run(args, capture_output=True, text=True)
            
            if result.returncode != case["expected_exit"]:
                print(f"    ❌ Expected exit code {case['expected_exit']}, got {result.returncode}")
                print(f"    stdout: {result.stdout}")
                print(f"    stderr: {result.stderr}")
                return False
            
            # Check that output file was created and has content
            if not os.path.exists(temp_output):
                print(f"    ❌ Output file not created")
                return False
            
            with open(temp_output, 'r') as f:
                content = f.read()
            
            if len(content) < 100:  # Should have substantial content
                print(f"    ❌ Output file too small ({len(content)} chars)")
                return False
            
            # Check for expected content
            required_content = [
                "# MillionTrees Dataset Release Report",
                "TreePolygons",
                "TreePoints", 
                "TreeBoxes",
                "Dataset Summary"
            ]
            
            for req in required_content:
                if req not in content:
                    print(f"    ❌ Missing required content: {req}")
                    return False
            
            print(f"    ✅ Test case passed")
            
        finally:
            # Clean up
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    return True


def test_script_structure():
    """Test that the SLURM script has proper structure."""
    print("Testing SLURM script structure...")
    
    script_path = "submit_test_release.sh"
    if not os.path.exists(script_path):
        print(f"  ❌ Script not found: {script_path}")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for required SLURM directives
    required_slurm = [
        "#SBATCH --job-name=",
        "#SBATCH --account=",
        "#SBATCH --time=",
        "#SBATCH --output=",
        "#SBATCH --error=",
    ]
    
    for req in required_slurm:
        if req not in content:
            print(f"  ❌ Missing SLURM directive: {req}")
            return False
    
    # Check for required script components
    required_components = [
        "pytest tests/test_release.py",
        "generate_dataset_report.py",
        "PYTEST_EXIT_CODE=",
        "exit $PYTEST_EXIT_CODE"
    ]
    
    for req in required_components:
        if req not in content:
            print(f"  ❌ Missing script component: {req}")
            return False
    
    print("  ✅ SLURM script structure is valid")
    return True


def test_documentation():
    """Test that documentation files exist and have content."""
    print("Testing documentation...")
    
    doc_files = [
        "HIPERGATOR_TESTING.md",
        "dataset_release_report.md"
    ]
    
    for doc_file in doc_files:
        if not os.path.exists(doc_file):
            print(f"  ❌ Documentation file not found: {doc_file}")
            return False
        
        with open(doc_file, 'r') as f:
            content = f.read()
        
        if len(content) < 100:
            print(f"  ❌ Documentation file too small: {doc_file}")
            return False
        
        print(f"  ✅ {doc_file} exists and has content")
    
    return True


def main():
    """Run all tests."""
    print("Running HiPerGator test release validation tests...\n")
    
    # Change to the repository root
    repo_root = Path(__file__).parent
    os.chdir(repo_root)
    
    tests = [
        test_report_generation,
        test_script_structure,
        test_documentation,
    ]
    
    all_passed = True
    for test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"  ❌ Test {test_func.__name__} failed with exception: {e}")
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All tests passed! The HiPerGator test release scripts are ready.")
        return 0
    else:
        print("❌ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())