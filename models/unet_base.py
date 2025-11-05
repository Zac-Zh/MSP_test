"""
Base UNet building blocks.

Provides fundamental convolutional blocks used across all UNet variants:
- DoubleConv: Two consecutive Conv->BatchNorm->ReLU blocks
- Down: Downsampling with MaxPool + DoubleConv
- Up: Upsampling with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling module: MaxPool + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling module: if bilinear, use Upsample then 1x1 conv to reduce channels;
    otherwise, use ConvTranspose2d directly. Then concatenate and apply DoubleConv.
    """

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            # 1. Upsample, keeping channels the same
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 2. Use 1x1 convolution to reduce channels from in_channels to in_channels//2
            self.channel_reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
            # 3. Channels after concatenation: in_channels//2 + skip_channels
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)
        else:
            # ConvTranspose2d directly reduces channels from in_channels -> in_channels//2
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # Channels after concatenation: in_channels//2 + skip_channels
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: Deep features from the next level (channels = in_channels)
        x2: Shallow features from the skip connection (channels = skip_channels)
        """
        if self.bilinear:
            x1 = self.up(x1)  # Spatial upsampling, no channel change
            x1 = self.channel_reduce(x1)  # Halve the channels -> in_channels//2
        else:
            x1 = self.up(x1)  # ConvTranspose2d already halved channels during upsampling

        # Align spatial dimensions (maintaining original padding logic)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate: x1 channels = in_channels//2, x2 channels = skip_channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
