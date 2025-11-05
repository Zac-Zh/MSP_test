"""
Neural network models for MSP detection.

Provides UNet-based architectures for heatmap regression and classification.
"""

from .unet_heatmap import UNetHeatmap
from .unet_with_cls import UNetWithCls
from .unet_dual_heads import (
    UNetWithDualHeads,
    CriterionCombined,
    compute_slice_coverage_label,
    combine_slice_probability
)

__all__ = [
    'UNetHeatmap',
    'UNetWithCls',
    'UNetWithDualHeads',
    'CriterionCombined',
    'compute_slice_coverage_label',
    'combine_slice_probability'
]
