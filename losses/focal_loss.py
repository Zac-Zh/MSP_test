"""
Focal loss for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss to address class imbalance by down-weighting easy examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: Raw logits from model
            target: Binary ground truth

        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_term = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_term.mean()
        elif self.reduction == 'sum':
            return focal_term.sum()
        return focal_term
