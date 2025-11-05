"""
UNet architecture for heatmap regression.

Implements the base UNet model for multi-channel anatomical structure
heatmap prediction.
"""

import torch
import torch.nn as nn
from .unet_base import DoubleConv, Down, Up


class UNetHeatmap(nn.Module):
    """A UNet that only outputs a heatmap (bilinear can be changed by default)."""

    def __init__(self, n_channels=1, n_classes=4, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes  # Number of heatmap channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1  # Originally for determining expected skip_channels, but now we pass the true channels

        # The skip_channels here directly use the true channel counts from the skip connections (x3->256, x2->128, x1->64)
        self.up1 = Up(512, 256, 256, bilinear)
        self.up2 = Up(256, 128, 128, bilinear)
        self.up3 = Up(128, 64, 64, bilinear)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)  # [B, 64, H, W]
        x2 = self.down1(x1)  # [B, 128, H/2, W/2]
        x3 = self.down2(x2)  # [B, 256, H/4, W/4]
        x4 = self.down3(x3)  # [B, 512, H/8, W/8]

        x = self.up1(x4, x3)  # Concatenated channels = 512//2 + 256 = 512 -> DoubleConv -> out 256
        x = self.up2(x, x2)  # Concatenated channels = 256//2 + 128 = 256 -> DoubleConv -> out 128
        x = self.up3(x, x1)  # Concatenated channels = 128//2 + 64  = 128 -> DoubleConv -> out 64

        logits = self.outc(x)  # [B, n_classes, H, W]
        return logits
