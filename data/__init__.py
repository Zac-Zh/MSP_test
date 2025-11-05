"""
Data handling module for MSP detection.

This module provides utilities for loading, preprocessing, and batching
medical imaging data for training and inference.
"""

from .loaders import load_nifti_data, load_nifti_data_cached
from .preprocessing import (
    extract_slice,
    normalize_slice,
    generate_brain_mask_from_image,
    preprocess_and_cache,
    create_target_heatmap_with_distance_transform,
    mask_to_distancemap,
    get_transforms,
    remap_small_structures_to_parent,
)
from .datasets import HeatmapDataset
from .samplers import (
    CaseAwareBalancedBatchSampler,
    BalancedBatchSampler,
    create_balanced_dataloader,
)

__all__ = [
    'load_nifti_data',
    'load_nifti_data_cached',
    'extract_slice',
    'normalize_slice',
    'generate_brain_mask_from_image',
    'preprocess_and_cache',
    'create_target_heatmap_with_distance_transform',
    'mask_to_distancemap',
    'get_transforms',
    'remap_small_structures_to_parent',
    'HeatmapDataset',
    'CaseAwareBalancedBatchSampler',
    'BalancedBatchSampler',
    'create_balanced_dataloader',
]
