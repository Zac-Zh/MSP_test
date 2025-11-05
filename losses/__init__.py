"""
Loss functions for MSP detection training.

Provides Dice loss, Focal loss, and combined losses for multi-task learning.
"""

from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .combined_loss import (
    compute_heatmap_loss_with_brain_constraint,
    compute_coverage_aware_combined_loss,
    compute_keypoint_constrained_loss
)

__all__ = [
    'DiceLoss',
    'FocalLoss',
    'compute_heatmap_loss_with_brain_constraint',
    'compute_coverage_aware_combined_loss',
    'compute_keypoint_constrained_loss'
]
