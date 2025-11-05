"""
Evaluation module for MSP detection.

Provides metrics computation, threshold optimization, and performance evaluation
at both slice and case levels.
"""

from .metrics import (
    scan_slice_threshold_youden,
    collect_and_store_roc_data,
    evaluate_case_level,
    compute_optimal_case_threshold,
    find_optimal_case_threshold,
    adaptive_threshold_search,
)
from .case_level import (
    test_fold_model,
    test_case_level,
)

__all__ = [
    'scan_slice_threshold_youden',
    'collect_and_store_roc_data',
    'evaluate_case_level',
    'compute_optimal_case_threshold',
    'find_optimal_case_threshold',
    'adaptive_threshold_search',
    'test_fold_model',
    'test_case_level',
]
