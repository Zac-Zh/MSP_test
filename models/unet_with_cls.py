"""
UNet with classification head.

Implements UNet architecture with dual outputs: heatmap regression
and slice-level MSP classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_heatmap import UNetHeatmap


class UNetWithCls(nn.Module):
    """UNet with both heatmap regression and slice-level classification."""

    def __init__(self, n_channels=1, n_classes=4, bilinear_unet=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_heatmap_classes = n_classes

        # UNet for heatmap regression
        self.unet = UNetHeatmap(n_channels=n_channels, n_classes=self.n_heatmap_classes, bilinear=bilinear_unet)

        # Use encoder features instead of decoder output
        # The deepest layer of the encoder has 512 features
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),  # Changed from n_heatmap_classes to 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Reduced from 0.5 to 0.3
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Reduced from 0.4 to 0.3
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Reduced from 0.3 to 0.2
            nn.Linear(64, 1)
        )
        self._initialize_cls_weights()

    def _initialize_cls_weights(self):
        """Initializes weights using a better method."""
        for m in self.cls_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Use Kaiming initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Uses deep features from the UNet encoder for classification.
        """
        # 1) Manually execute the UNet encoding process to get intermediate features
        x1 = self.unet.inc(x)  # [B, 64, H, W]
        x2 = self.unet.down1(x1)  # [B, 128, H/2, W/2]
        x3 = self.unet.down2(x2)  # [B, 256, H/4, W/4]
        x4 = self.unet.down3(x3)  # [B, 512, H/8, W/8]

        # 2) Classification branch: use the deepest encoder features
        cls_features = F.adaptive_avg_pool2d(x4, (1, 1))  # [B, 512, 1, 1]
        cls_features = cls_features.view(cls_features.size(0), -1)  # [B, 512]
        cls_logit = self.cls_head(cls_features)  # [B, 1]

        # 3) Complete the UNet decoding to get the heatmap
        x = self.unet.up1(x4, x3)
        x = self.unet.up2(x, x2)
        x = self.unet.up3(x, x1)
        heatmaps_logits = self.unet.outc(x)

        return heatmaps_logits, cls_logit
