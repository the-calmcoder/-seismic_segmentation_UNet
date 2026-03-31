"""
loss.py — Composite Loss Function for Seismic Segmentation
===========================================================
Combines Weighted Cross-Entropy and Dice Loss to handle the extreme
class imbalance inherent in seismic data (>90% background pixels).

Why two losses?
    • Weighted CE:  Provides strong per-pixel gradients and penalizes
                    misclassification of rare classes more heavily.
    • Dice Loss:    Directly optimizes the overlap (IoU-like) between
                    predicted and true masks, ignoring class frequency.

The composite loss = α × WCE + (1 − α) × Dice, where α defaults to 0.5.

Usage:
    criterion = CompositeLoss(num_classes=9)
    loss = criterion(logits, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Class Weights
# ─────────────────────────────────────────────────────────────────────────────

# Default weights: heavily penalize misclassifying geological features.
# Background (0) gets low weight because it dominates (>90% of pixels).
# Faults (1) get the highest weight because they are rarest and most critical.
# Horizons (2–8) get moderate-high weight.
DEFAULT_CLASS_WEIGHTS = torch.tensor([
    0.10,   # 0: Background   — abundant, low penalty
    3.00,   # 1: Fault        — very rare, very high penalty
    1.50,   # 2: FS4
    1.50,   # 3: MFS4
    1.50,   # 4: FS6
    1.50,   # 5: FS7
    1.50,   # 6: FS8
    1.50,   # 7: Shallow
    1.50,   # 8: Top Foresets
], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Weighted Cross-Entropy Loss
# ─────────────────────────────────────────────────────────────────────────────

class WeightedCrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy loss with per-class weighting.

    Higher weights for rare classes force the model to pay more attention
    to faults and horizons rather than defaulting to "predict background."

    Parameters
    ----------
    weights : torch.Tensor
        Per-class weight vector of length num_classes.
    """

    def __init__(self, weights: torch.Tensor = DEFAULT_CLASS_WEIGHTS) -> None:
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Raw predictions of shape (B, C, H, W).
        targets : torch.Tensor
            Ground truth labels of shape (B, H, W) with values in [0, C-1].

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        return F.cross_entropy(logits, targets, weight=self.weights)


# ─────────────────────────────────────────────────────────────────────────────
# Dice Loss
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice Loss computed per-class and averaged.

    Dice coefficient measures the overlap between prediction and ground
    truth, ranging from 0 (no overlap) to 1 (perfect overlap). The loss
    is 1 − Dice.

    Advantages over CE:
        • Directly optimizes spatial overlap (what IoU measures)
        • Naturally handles class imbalance without explicit weights

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes (default: 9).
    smooth : float
        Smoothing constant to prevent division by zero and stabilize
        gradients when a class has very few pixels (default: 1.0).
    """

    def __init__(self, num_classes: int = 9, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor, shape (B, C, H, W)
        targets : torch.Tensor, shape (B, H, W)

        Returns
        -------
        torch.Tensor
            Scalar: mean (1 − Dice) across all classes.
        """
        # Convert logits to probabilities via softmax
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets → (B, C, H, W)
        targets_one_hot = F.one_hot(
            targets, num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()  # Move class dim to channel position

        # Compute per-class Dice coefficient
        dice_sum = torch.tensor(0.0, device=logits.device)

        for c in range(self.num_classes):
            pred_c = probs[:, c]            # (B, H, W)
            true_c = targets_one_hot[:, c]  # (B, H, W)

            intersection = (pred_c * true_c).sum()
            cardinality = pred_c.sum() + true_c.sum()

            dice_c = (2.0 * intersection + self.smooth) / (
                cardinality + self.smooth
            )
            dice_sum += dice_c

        # Average Dice across classes, return as loss (1 - Dice)
        mean_dice = dice_sum / self.num_classes
        return 1.0 - mean_dice


# ─────────────────────────────────────────────────────────────────────────────
# Composite Loss
# ─────────────────────────────────────────────────────────────────────────────

class CompositeLoss(nn.Module):
    """
    Combined loss: α × WeightedCE + (1 − α) × Dice.

    This dual-loss strategy leverages the strengths of both:
        • WeightedCE provides stable per-pixel gradients early in training
        • DiceLoss pushes for better spatial overlap as training progresses

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes.
    alpha : float
        Weighting factor. α=0.5 gives equal contribution (default).
    class_weights : torch.Tensor or None
        Per-class weights for cross-entropy. Uses defaults if None.
    """

    def __init__(
        self,
        num_classes: int = 9,
        alpha: float = 0.5,
        class_weights: torch.Tensor = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha

        weights = class_weights if class_weights is not None else DEFAULT_CLASS_WEIGHTS
        self.wce = WeightedCrossEntropyLoss(weights)
        self.dice = DiceLoss(num_classes=num_classes)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the composite loss.

        Parameters
        ----------
        logits : torch.Tensor, shape (B, C, H, W)
        targets : torch.Tensor, shape (B, H, W)

        Returns
        -------
        torch.Tensor
            Scalar loss value: α × WCE + (1 − α) × Dice.
        """
        loss_wce = self.wce(logits, targets)
        loss_dice = self.dice(logits, targets)

        return self.alpha * loss_wce + (1.0 - self.alpha) * loss_dice
