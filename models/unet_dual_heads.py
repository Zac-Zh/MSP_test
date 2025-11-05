"""
UNet with dual heads (classification + coverage).

Implements UNet architecture with three outputs:
- Heatmap regression for anatomical structure localization
- Binary classification for slice-level MSP prediction
- Coverage prediction (none/partial/full) for structure visibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .unet_heatmap import UNetHeatmap


def compute_slice_coverage_label(mask_slice: np.ndarray, struct_lbl: int, thresh: float = 0.7) -> int:
    total_pixels = (mask_slice == struct_lbl).sum()
    if total_pixels == 0:
        return 0
    coverage_ratio = total_pixels / mask_slice.size
    return 2 if coverage_ratio >= thresh else 1


class UNetWithDualHeads(nn.Module):
    """UNet backbone + heatmap head + binary-classification head + coverage-head"""

    def __init__(self, in_channels: int = 1, feat_channels: int = 64):
        super().__init__()
        self.unet = UNetHeatmap(n_channels=in_channels, n_classes=4, bilinear=True)

        self.cls_head = nn.Linear(feat_channels * 8, 1)
        self.coverage_head = nn.Linear(feat_channels * 8, 3)

    def forward(self, x):
        # Manually execute the UNet encoding process to get intermediate features
        x1 = self.unet.inc(x)  # [B, 64, H, W]
        x2 = self.unet.down1(x1)  # [B, 128, H/2, W/2]
        x3 = self.unet.down2(x2)  # [B, 256, H/4, W/4]
        x4 = self.unet.down3(x3)  # [B, 512, H/8, W/8]

        # Classification branch: use the deepest features from the encoder
        pooled = F.adaptive_avg_pool2d(x4, (1, 1)).view(x4.size(0), -1)  # [B, 512]
        cls_logits = self.cls_head(pooled)  # [B, 1]
        cov_logits = self.coverage_head(pooled)  # [B, 3]
        # Complete the UNet decoding to get the heatmap
        x = self.unet.up1(x4, x3)
        x = self.unet.up2(x, x2)
        x = self.unet.up3(x, x1)
        heatmaps_logits = self.unet.outc(x)

        return heatmaps_logits, cls_logits, cov_logits


class CriterionCombined(nn.Module):
    """heatmap BCE/Dice + binary BCE + coverage CE"""

    def __init__(self, lambda_heat=1.0, lambda_cls=1.0, lambda_cov=1.0):
        super().__init__()
        self.lambda_heat = lambda_heat
        self.lambda_cls = lambda_cls
        self.lambda_cov = lambda_cov
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.ce_cov = nn.CrossEntropyLoss()

    def forward(self, pred_heat, tgt_heat, pred_cls, tgt_cls, pred_cov, tgt_cov):
        loss_heat = self.bce_logits(pred_heat, tgt_heat)
        loss_cls = self.bce_logits(pred_cls, tgt_cls.float())
        loss_cov = self.ce_cov(pred_cov, tgt_cov)
        return (self.lambda_heat * loss_heat +
                self.lambda_cls * loss_cls +
                self.lambda_cov * loss_cov), {
            "loss_heat": loss_heat.detach(),
            "loss_cls": loss_cls.detach(),
            "loss_cov": loss_cov.detach()
        }


def combine_slice_probability(cov_logits: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """softmax(2)=full, softmax(1)=partial; slice_prob = full + alpha * partial"""
    probs = torch.softmax(cov_logits, dim=1)
    return probs[:, 2] + alpha * probs[:, 1]
