"""
Training module for MSP detection.

Provides training loops, pipelines, and helper utilities for model training.
"""

from .helpers import (
    prepare_patient_grouped_datasets,
    load_model_with_correct_architecture,
)
from .staged_training import (
    generate_negative_samples,
    train_stage1_heatmap,
    train_stage2_joint,
    train_heatmap_model_with_coverage_aware_training,
)
from .meta_classifier import (
    collect_features_and_labels,
    train_meta_classifier_full,
)

__all__ = [
    'prepare_patient_grouped_datasets',
    'load_model_with_correct_architecture',
    'generate_negative_samples',
    'train_stage1_heatmap',
    'train_stage2_joint',
    'train_heatmap_model_with_coverage_aware_training',
    'collect_features_and_labels',
    'train_meta_classifier_full',
]
