"""
Pipeline orchestration module for MSP detection.

Provides complete research pipelines including validation, baseline training,
and automated evaluation workflows.
"""

from .validation import (
    run_5fold_validation_with_case_level,
    run_baseline_validation,
)

__all__ = [
    'run_5fold_validation_with_case_level',
    'run_baseline_validation',
]
