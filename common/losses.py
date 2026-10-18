"""
Cross entropy plus Dice, which counters the class imbalance.

94 percent of all pixels are background and three quarters of the building
pixels are undamaged, so plain cross entropy settles on predicting background.
"""

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """Cross entropy plus Dice on the foreground class."""

    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        target_float = target.float()
        intersection = (probs * target_float).sum()
        dice = 1 - (2 * intersection + 1) / (probs.sum() + target_float.sum() + 1)
        return self.ce(logits, target) + self.dice_weight * dice


class MultiClassDiceLoss(nn.Module):
    """Cross entropy plus mean Dice over all classes."""

    def __init__(self, n_classes=3, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.n_classes = n_classes
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        dice_total = 0
        for c in range(self.n_classes):
            probs_c, target_c = probs[:, c], (target == c).float()
            intersection = (probs_c * target_c).sum()
            dice_total += 1 - (2 * intersection + 1) / (probs_c.sum() + target_c.sum() + 1)

        return self.ce(logits, target) + self.dice_weight * dice_total / self.n_classes


class WeightedMultiClassDiceLoss(nn.Module):
    """MultiClassDiceLoss with class weights in the cross entropy term.

    The default weights lift minor and major, the two rarest damage classes.
    """

    def __init__(self, n_classes=5, dice_weight=0.5, class_weights=None, device="cpu"):
        super().__init__()
        self.n_classes = n_classes
        self.dice_weight = dice_weight
        if class_weights is None:
            class_weights = torch.tensor([1.0, 1.0, 2.0, 1.5, 1.0])
        self.ce = nn.CrossEntropyLoss(weight=class_weights.to(device))

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        dice_total = 0
        for c in range(self.n_classes):
            probs_c, target_c = probs[:, c], (target == c).float()
            intersection = (probs_c * target_c).sum()
            dice_total += 1 - (2 * intersection + 1) / (probs_c.sum() + target_c.sum() + 1)

        return self.ce(logits, target) + self.dice_weight * dice_total / self.n_classes
