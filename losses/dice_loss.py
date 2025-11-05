"""
Dice loss for segmentation.
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice coefficient loss for segmentation tasks.

    Measures overlap between prediction and ground truth.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_probs: Predicted probabilities after sigmoid
            target: Binary ground truth

        Returns:
            Dice loss value
        """
        intersection = (pred_probs * target).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
