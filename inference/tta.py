"""
Test-Time Augmentation (TTA) for MSP detection.

Provides functions for applying test-time augmentation strategies
to improve model predictions during inference.
"""

import torch


def apply_tta_horizontal_flip(img_tensor, model_func, *args, **kwargs):
    """Applies horizontal flip Test-Time Augmentation, supporting UNetWithDualHeads."""
    device = img_tensor.device

    # Original inference
    with torch.no_grad():
        original_output = model_func(img_tensor, *args, **kwargs)

    # Horizontally flipped inference
    img_flipped = torch.flip(img_tensor, dims=[3])  # Horizontal flip
    with torch.no_grad():
        flipped_output = model_func(img_flipped, *args, **kwargs)

    # Process based on output type, supporting the three-output of UNetWithDualHeads
    if isinstance(original_output, tuple) and len(original_output) == 3:
        # UNetWithDualHeads output: (heatmaps, cls_logit, cov_logits)
        orig_heatmaps, orig_cls, orig_cov = original_output
        flip_heatmaps, flip_cls, flip_cov = flipped_output

        # Un-flip the heatmaps and take the maximum
        flip_heatmaps_unflipped = torch.flip(flip_heatmaps, dims=[3])
        enhanced_heatmaps = 0.5 * (orig_heatmaps + flip_heatmaps_unflipped)
        enhanced_cls     = 0.5 * (orig_cls + flip_cls)
        enhanced_cov     = 0.5 * (orig_cov + flip_cov)

        return enhanced_heatmaps, enhanced_cls, enhanced_cov

    elif isinstance(original_output, tuple) and len(original_output) == 2:
        # UNetWithCls output: (heatmaps, cls_logit)
        orig_heatmaps, orig_cls = original_output
        flip_heatmaps, flip_cls = flipped_output

        # Un-flip the heatmaps and take the maximum
        flip_heatmaps_unflipped = torch.flip(flip_heatmaps, dims=[3])
        enhanced_heatmaps = 0.5 * (orig_heatmaps + flip_heatmaps_unflipped)
        enhanced_cls     = 0.5 * (orig_cls + flip_cls)

        return enhanced_heatmaps, enhanced_cls
    else:
        # UNetHeatmap output: heatmaps only
        flip_heatmaps_unflipped = torch.flip(flipped_output, dims=[3])
        enhanced_heatmaps = 0.5 * (original_output + flip_heatmaps_unflipped)

        return enhanced_heatmaps
