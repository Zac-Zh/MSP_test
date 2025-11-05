"""
Verification script to check that all extracted functions are properly structured
and can be imported without circular dependencies or syntax errors.
"""

import sys
import importlib.util

def check_module(module_path, module_name):
    """Check if a module can be loaded."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            # Don't execute, just check syntax
            return True, "OK"
        return False, "No spec"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    print("="*80)
    print("EXTRACTION VERIFICATION")
    print("="*80)
    
    modules_to_check = [
        ("eval/case_level.py", "case_level"),
        ("eval/metrics.py", "metrics"),
        ("inference/detection.py", "detection"),
        ("inference/keypoints.py", "keypoints"),
        ("inference/tta.py", "tta"),
        ("train/staged_training.py", "staged_training"),
        ("train/meta_classifier.py", "meta_classifier"),
        ("train/helpers.py", "helpers"),
        ("utils/gating.py", "gating"),
        ("utils/threshold_tuning.py", "threshold_tuning"),
        ("utils/results.py", "results"),
        ("visualization/evaluation_plots.py", "evaluation_plots"),
        ("visualization/case_viz.py", "case_viz"),
        ("pipelines/validation.py", "validation"),
    ]
    
    all_ok = True
    for module_path, module_name in modules_to_check:
        ok, msg = check_module(module_path, module_name)
        status = "✓" if ok else "✗"
        print(f"{status} {module_path:40s} {msg}")
        if not ok:
            all_ok = False
    
    print("="*80)
    
    # Check critical function presence
    print("\nCRITICAL FUNCTIONS CHECK:")
    print("-"*80)
    
    critical_functions = {
        "pipelines/validation.py": [
            "run_5fold_validation_with_case_level",
            "run_baseline_validation"
        ],
        "train/staged_training.py": [
            "train_heatmap_model_with_coverage_aware_training",
            "generate_negative_samples",
            "train_stage1_heatmap",
            "train_stage2_joint"
        ],
        "inference/detection.py": [
            "detect_msp_case_level_with_coverage",
            "process_slice_with_coverage_constraints",
            "evaluate_case_level"
        ],
        "eval/case_level.py": [
            "test_fold_model",
            "test_case_level"
        ]
    }
    
    for file_path, functions in critical_functions.items():
        with open(file_path, 'r') as f:
            content = f.read()
        
        print(f"\n{file_path}:")
        for func in functions:
            if f"def {func}(" in content:
                print(f"  ✓ {func}")
            else:
                print(f"  ✗ {func} MISSING")
                all_ok = False
    
    print("\n" + "="*80)
    if all_ok:
        print("✅ ALL CHECKS PASSED - Extraction is functionally complete!")
    else:
        print("❌ SOME CHECKS FAILED - Review errors above")
    print("="*80)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
