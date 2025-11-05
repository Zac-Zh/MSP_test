"""
Combined loss functions for MSP detection training.

Includes:
- Heatmap loss with brain constraints
- Coverage-aware combined loss
- Keypoint constrained loss
"""

import torch
import torch.nn.functional as F


def compute_keypoint_constrained_loss(p4, p5, gt4, gt5, gt_seg, epoch, total_epochs=400,
                                      lambda_out_max=0.5, warmup_epochs=5):
    """
    Keypoint loss function: out-of-bounds penalty + warm-up.

    Args:
        p4/p5: (B,H,W) in [0,1] after sigmoid - predicted keypoint heatmaps
        gt4/gt5: Gaussian keypoint maps - ground truth
        gt_seg: (B,H,W) integer semantic GT (0-5) - semantic segmentation
        epoch: Current epoch number
        total_epochs: Total number of epochs
        lambda_out_max: Maximum penalty weight for out-of-bounds predictions
        warmup_epochs: Number of epochs before applying out-of-bounds penalty

    Returns:
        total_loss: Combined BCE + out-of-bounds penalty
        loss_dict: Dictionary with detailed loss components
    """
    bce4 = F.binary_cross_entropy(p4, gt4)
    bce5 = F.binary_cross_entropy(p5, gt5)

    if epoch >= warmup_epochs:
        # Align seg with p4/p5 resolution
        gt_seg = F.interpolate(gt_seg.unsqueeze(1).float(),
                               size=p4.shape[-2:],
                               mode='nearest').squeeze(1)
        mask2 = (gt_seg == 2).float()
        mask3 = (gt_seg == 3).float()

        # Uniform penalty for probability falling outside of structures 2/3
        out4 = torch.mean(p4 * (1 - mask2))
        out5 = torch.mean(p5 * (1 - mask3))

        # Progressive penalty weight
        lam = lambda_out_max * (epoch - warmup_epochs + 1) / (total_epochs - warmup_epochs + 1)
        lam = min(lam, lambda_out_max)  # Ensure it doesn't exceed the max
    else:
        out4 = out5 = lam = 0.0

    total_loss = bce4 + bce5 + lam * (out4 + out5)

    return total_loss, {
        'bce4': bce4.item(),
        'bce5': bce5.item(),
        'out4': out4.item() if hasattr(out4, 'item') else out4,
        'out5': out5.item() if hasattr(out5, 'item') else out5,
        'lambda_out': lam,
        'total': total_loss.item()
    }


def compute_heatmap_loss_with_brain_constraint(
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        brain_mask: torch.Tensor | None = None,
        per_channel_weights: list[float] | None = None,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        outside_brain_penalty: float = 0.1,
        eps: float = 1e-6,
        # Keypoint constraint parameters
        apply_keypoint_constraints: bool = False,
        gt_semantic_seg: torch.Tensor = None,
        current_epoch: int = 0,
        total_epochs: int = 400
):
    """
    Computes the heatmap loss, integrating a penalty for predictions outside the brain
    and constraints for keypoints.

    Args:
        pred_logits: (B, C, H, W) - predicted heatmap logits
        target: (B, C, H, W) - target heatmaps
        brain_mask: (B, 1, H, W) or None - brain region mask
        per_channel_weights: Optional weights for each channel
        focal_gamma: Focal loss gamma parameter
        focal_alpha: Focal loss alpha parameter
        outside_brain_penalty: Weight for outside-brain penalty
        eps: Small epsilon for numerical stability
        apply_keypoint_constraints: Whether to apply keypoint constraints
        gt_semantic_seg: (B, H, W) - semantic segmentation for keypoint constraints
        current_epoch: Current training epoch
        total_epochs: Total number of training epochs

    Returns:
        loss_reg: Combined regression loss
        loss_dict: Dictionary with detailed loss components
    """
    B, C, H, W = pred_logits.shape
    device = pred_logits.device

    if per_channel_weights is None:
        per_channel_weights = [1.0] * C
    w = torch.tensor(per_channel_weights, device=device)

    if brain_mask is None:
        brain_mask = torch.ones((B, 1, H, W), device=device, dtype=pred_logits.dtype)
    else:
        brain_mask = brain_mask.float()

    loss_bce = loss_dice = loss_focal = loss_outside = loss_keypoint = 0.0

    for i in range(C):
        li = pred_logits[:, i]  # logits
        ti = target[:, i]  # target
        mi = brain_mask.squeeze(1)  # mask
        outside_mi = 1.0 - mi  # outside-brain mask

        # BCE loss (within brain region)
        bce_map = F.binary_cross_entropy_with_logits(li, ti, reduction="none")
        bce_i = (bce_map * mi).sum() / (mi.sum() + eps)

        # Dice loss (within brain region)
        pi = torch.sigmoid(li) * mi
        ti_m = ti * mi
        inter = (pi * ti_m).sum(dim=(1, 2))
        dice_i = 1.0 - (2 * inter + eps) / (pi.sum(dim=(1, 2)) + ti_m.sum(dim=(1, 2)) + eps)
        dice_i = dice_i.mean()

        # Focal loss (within brain region)
        focal_map = - focal_alpha * (1 - pi) ** focal_gamma * ti_m * torch.log(pi + eps) \
                    - (1 - focal_alpha) * pi ** focal_gamma * (1 - ti_m) * torch.log(1 - pi + eps)
        focal_i = focal_map.sum() / (mi.sum() + eps)

        # Penalty for outside-brain region
        outside_pred = torch.sigmoid(li) * outside_mi
        outside_penalty_i = (outside_pred ** 2).sum() / (outside_mi.sum() + eps)

        loss_bce += w[i] * bce_i
        loss_dice += w[i] * dice_i
        loss_focal += w[i] * focal_i
        loss_outside += w[i] * outside_penalty_i

    # Keypoint constraints (if enabled and channels are sufficient)
    if apply_keypoint_constraints and C >= 4 and gt_semantic_seg is not None:
        # Assuming the first 4 channels correspond to structures 2,3,4,5
        p2 = torch.sigmoid(pred_logits[:, 0])  # struct 2
        p3 = torch.sigmoid(pred_logits[:, 1])  # struct 3
        p4 = torch.sigmoid(pred_logits[:, 2])  # struct 4 (keypoint)
        p5 = torch.sigmoid(pred_logits[:, 3])  # struct 5 (keypoint)

        # Create virtual GT for keypoint channels (from semantic segmentation)
        gt4 = (gt_semantic_seg == 4).float()
        gt5 = (gt_semantic_seg == 5).float()

        keypoint_loss, _ = compute_keypoint_constrained_loss(
            p4, p5, gt4, gt5, gt_semantic_seg,
            current_epoch, total_epochs
        )
        loss_keypoint = keypoint_loss

    loss_reg = 0.5 * (loss_bce + loss_dice) + loss_focal + outside_brain_penalty * loss_outside + loss_keypoint

    loss_dict = {
        "bce": loss_bce.detach().item(),
        "dice": loss_dice.detach().item(),
        "focal": loss_focal.detach().item(),
        "outside": loss_outside.detach().item(),
        "keypoint": loss_keypoint.detach().item() if hasattr(loss_keypoint, 'detach') else float(loss_keypoint),
        "total": loss_reg.detach().item(),
    }

    return loss_reg, loss_dict


def compute_coverage_aware_combined_loss(pred_heatmaps_logits, cls_logit, cov_logits,
                                         target_heatmaps, is_msp_labels_for_cls, cov_labels, has_regression_targets,
                                         brain_masks=None, lambda_reg=1.0, lambda_cls=0.5, lambda_cov=0.3,
                                         outside_brain_penalty_for_reg=0.1, config=None,
                                         current_epoch=0, total_epochs=400, cls_pos_weight=None):
    """
    Calculates the combined loss, supporting all features including coverage loss.

    Args:
        pred_heatmaps_logits: (B, C, H, W) - predicted heatmap logits
        cls_logit: (B, 1) - classification logits (MSP vs non-MSP)
        cov_logits: (B, 3) - coverage classification logits (3 classes)
        target_heatmaps: (B, C, H, W) - target heatmaps
        is_msp_labels_for_cls: (B,) - binary labels for classification
        cov_labels: (B,) - coverage labels (0, 1, or 2)
        has_regression_targets: (B,) - boolean mask for slices with regression targets
        brain_masks: (B, 1, H, W) or None - brain region masks
        lambda_reg: Weight for regression loss
        lambda_cls: Weight for classification loss
        lambda_cov: Weight for coverage loss
        outside_brain_penalty_for_reg: Penalty weight for outside-brain predictions
        config: Configuration dictionary
        current_epoch: Current epoch number
        total_epochs: Total number of epochs
        cls_pos_weight: Positive class weight for classification loss

    Returns:
        total_loss: Combined weighted loss
        loss_dict: Dictionary with detailed loss components
    """
    device = pred_heatmaps_logits.device
    focal_gamma = config.get("FOCAL_GAMMA_REG", 2.0) if config else 2.0
    focal_alpha = config.get("FOCAL_ALPHA_REG", 0.25) if config else 0.25

    # 1) Regression loss (including keypoint constraints)
    mask_for_regression = has_regression_targets

    if mask_for_regression.any():
        pred_reg = pred_heatmaps_logits[mask_for_regression]
        target_reg = target_heatmaps[mask_for_regression]

        brain_reg = None
        if brain_masks is not None:
            if brain_masks.dim() == 3:
                brain_masks_expanded = brain_masks.unsqueeze(1)
            elif brain_masks.dim() == 4:
                brain_masks_expanded = brain_masks
            else:
                brain_masks_expanded = None
            if brain_masks_expanded is not None:
                brain_reg = brain_masks_expanded[mask_for_regression]

        loss_reg, reg_details_dict = compute_heatmap_loss_with_brain_constraint(
            pred_reg, target_reg, brain_reg,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            outside_brain_penalty=outside_brain_penalty_for_reg,
            apply_keypoint_constraints=True,
            current_epoch=current_epoch,
            total_epochs=total_epochs
        )
    else:
        loss_reg = torch.tensor(0.0, device=device, requires_grad=True)
        reg_details_dict = {'bce': 0.0, 'dice': 0.0, 'focal': 0.0, 'outside': 0.0, 'keypoint': 0.0, 'total': 0.0}

    # 2) Classification loss with class weighting
    pos_weight_tensor = torch.tensor([cls_pos_weight], device=device) if cls_pos_weight is not None else None
    loss_cls = F.binary_cross_entropy_with_logits(cls_logit, is_msp_labels_for_cls,
                                                  pos_weight=pos_weight_tensor,
                                                  reduction='mean')

    # 3) Coverage Loss (using Cross Entropy)
    loss_cov = F.cross_entropy(cov_logits, cov_labels, reduction='mean')

    # 4) Total loss
    total_loss = lambda_reg * loss_reg + lambda_cls * loss_cls + lambda_cov * loss_cov

    # 5) Detailed loss dictionary
    loss_dict = {
        'total_loss': total_loss.item(),
        'reg_loss_weighted': (lambda_reg * loss_reg).item(),
        'cls_loss_weighted': (lambda_cls * loss_cls).item(),
        'cov_loss_weighted': (lambda_cov * loss_cov).item(),
        'reg_loss_raw': loss_reg.item(),
        'cls_loss_raw': loss_cls.item(),
        'cov_loss_raw': loss_cov.item(),
        'lambda_reg': lambda_reg,
        'lambda_cls': lambda_cls,
        'lambda_cov': lambda_cov,
        'n_slices_for_reg': mask_for_regression.sum().item(),
        'n_total_slices_in_batch': len(is_msp_labels_for_cls),
        'loss_type': 'hard_label_with_coverage_and_constraints',
        'current_epoch': current_epoch,
        'enhancements_active': {
            'keypoint_constraints': True,
            'coverage_classification': True,
            'dynamic_penalties': current_epoch >= 5
        }
    }

    # Add regression details
    for k, v in reg_details_dict.items():
        loss_dict[f"reg_{k}"] = v

    return total_loss, loss_dict
