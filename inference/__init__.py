"""
Inference module for MSP detection.

Provides functions for test-time augmentation and model inference pipelines.
"""

from .tta import apply_tta_horizontal_flip
from .detection import (
    detect_msp_case_level_with_coverage,
    process_slice_with_coverage_constraints,
    evaluate_case_level,
    combine_slice_probability,
)
from .keypoints import detect_two_keypoints

__all__ = [
    'apply_tta_horizontal_flip',
    'detect_msp_case_level_with_coverage',
    'process_slice_with_coverage_constraints',
    'evaluate_case_level',
    'combine_slice_probability',
    'detect_two_keypoints',
]
