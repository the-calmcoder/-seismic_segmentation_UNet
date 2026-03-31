"""
train.py — Training Loop for Seismic Segmentation
==================================================
Trains the U-Net model on extracted seismic patches using:
    • Adam optimizer with learning rate 1e-4
    • Cosine Annealing LR scheduler
    • FP16 mixed precision (AMP) for efficient GPU memory usage
    • Best-model checkpointing based on validation Dice score

Designed for: Google Colab T4 (15GB VRAM) — batch_size=16 default.

Usage:
    python src/train.py --data_dir data/ --epochs 50
    python src/train.py --data_dir /content/data --batch_size 16 --epochs 80
"""

import os
import time
import argparse
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.model import UNet, model_summary
from src.loss import CompositeLoss
from src.dataset import build_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# Metric Computation
# ─────────────────────────────────────────────────────────────────────────────

NUM_CLASSES = 9

CLASS_NAMES = [
    "Background", "Fault", "FS4", "MFS4", "FS6",
    "FS7", "FS8", "Shallow", "Top Foresets",
]


def compute_dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute per-class Dice scores and mean Dice.

    Dice = 2 × |P ∩ T| / (|P| + |T|)

    Parameters
    ----------
    predictions : torch.Tensor, shape (B, H, W)
        Predicted class labels (argmax of logits).
    targets : torch.Tensor, shape (B, H, W)
        Ground truth class labels.
    num_classes : int
        Total number of classes.
    smooth : float
        Smoothing to handle empty predictions.

    Returns
    -------
    dict[str, float]
        Keys: class names + "mean_dice" (excluding background).
    """
    scores = {}
    dice_sum = 0.0
    valid_classes = 0

    for c in range(num_classes):
        pred_c = (predictions == c).float()
        true_c = (targets == c).float()

        intersection = (pred_c * true_c).sum().item()
        cardinality = pred_c.sum().item() + true_c.sum().item()

        dice = (2.0 * intersection + smooth) / (cardinality + smooth)
        scores[CLASS_NAMES[c]] = dice

        # Mean Dice excludes background (class 0)
        if c > 0:
            dice_sum += dice
            valid_classes += 1

    scores["mean_dice"] = dice_sum / max(valid_classes, 1)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Training & Validation Epochs
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Run one training epoch with mixed precision.

    Returns
    -------
    avg_loss : float
        Mean loss over all batches.
    dice_scores : dict
        Per-class Dice scores computed over the epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass with AMP
        with autocast():
            logits = model(images)
            loss = criterion(logits, masks)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Collect predictions for Dice computation
        preds = logits.argmax(dim=1)  # (B, H, W)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    # Compute epoch-level Dice scores
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    dice_scores = compute_dice_score(all_preds, all_targets)

    avg_loss = running_loss / len(loader)
    return avg_loss, dice_scores


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Run validation with no gradient computation.

    Returns
    -------
    avg_loss : float
    dice_scores : dict
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        with autocast():
            logits = model(images)
            loss = criterion(logits, masks)

        running_loss += loss.item()
        all_preds.append(logits.argmax(dim=1).cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    dice_scores = compute_dice_score(all_preds, all_targets)

    avg_loss = running_loss / len(loader)
    return avg_loss, dice_scores


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_dir: str,
    checkpoint_dir: str = "checkpoints",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    num_workers: int = 2,
) -> str:
    """
    Complete training pipeline.

    Workflow:
        1. Detect GPU and configure device
        2. Build DataLoaders for train/val splits
        3. Initialize model, loss, optimizer, scheduler
        4. Train for `epochs` with AMP and checkpoint best model
        5. Print final summary

    Parameters
    ----------
    data_dir : str
        Root data directory with slices/ and masks/ subdirectories.
    checkpoint_dir : str
        Directory to save model checkpoints.
    epochs : int
        Number of training epochs (default: 50).
    batch_size : int
        Batch size (default: 16, tuned for Colab T4 15GB).
    lr : float
        Initial learning rate for Adam optimizer.
    num_workers : int
        DataLoader worker processes (default: 2 for Colab).

    Returns
    -------
    str
        Path to the best model checkpoint.
    """
    # ── Device setup ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═' * 60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'═' * 60}")
    print(f"  Device:      {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU:         {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Data dir:    {data_dir}")

    # ── DataLoaders ──
    train_loader, val_loader, _ = build_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ── Model ──
    model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    model_summary(model)

    # ── Loss, Optimizer, Scheduler ──
    criterion = CompositeLoss(num_classes=NUM_CLASSES).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()

    # ── Checkpointing ──
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_model.pth")
    best_dice = 0.0

    # ── Training loop ──
    print(f"\n{'═' * 60}")
    print(f"STARTING TRAINING")
    print(f"{'═' * 60}")

    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Track history
        epoch_dice = val_dice["mean_dice"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(epoch_dice)

        elapsed = time.time() - t0

        # Checkpoint best model
        is_best = epoch_dice > best_dice
        if is_best:
            best_dice = epoch_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "history": history,
            }, best_path)

        # Log epoch results
        star = " ★ BEST" if is_best else ""
        print(
            f"  Epoch {epoch:>3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {epoch_dice:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{elapsed:.1f}s{star}"
        )

        # Detailed Dice breakdown every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            print(f"    Per-class Dice scores:")
            for name in CLASS_NAMES[1:]:  # Skip background
                print(f"      {name:>14s}: {val_dice[name]:.4f}")

    # ── Final summary ──
    print(f"\n{'═' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Best Validation Dice: {best_dice:.4f}")
    print(f"  Best model saved to:  {best_path}")
    print(f"{'═' * 60}\n")

    return best_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse arguments and launch training."""
    parser = argparse.ArgumentParser(
        description="Train U-Net for seismic fault & horizon segmentation",
    )
    parser.add_argument(
        "--data_dir", default="data",
        help="Path to extracted data directory (default: data/)",
    )
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints",
        help="Directory for model checkpoints (default: checkpoints/)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size — 16 for Colab T4, 4 for 6GB GPUs (default: 16)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Initial learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="DataLoader workers (default: 2)",
    )

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
