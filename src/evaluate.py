"""
evaluate.py — Inference, Metrics & Visualization
=================================================
Loads the best trained U-Net checkpoint, runs inference on the test set,
computes quantitative metrics, and generates visual overlay images.

Outputs:
    • Per-class IoU, Dice, Precision, Recall (printed + saved as CSV)
    • Visual overlays: grayscale seismic + colored fault/horizon masks
      (Red = Fault, unique colors for each horizon type)
    • Saved to the outputs/ directory

Usage:
    python src/evaluate.py --data_dir data/ --checkpoint checkpoints/best_model.pth
"""

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab/headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.model import UNet
from src.dataset import SeismicPatchDataset


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

NUM_CLASSES = 9

CLASS_NAMES = [
    "Background", "Fault", "FS4", "MFS4", "FS6",
    "FS7", "FS8", "Shallow", "Top Foresets",
]

# Color map for visualization overlays (RGBA, alpha applied separately)
CLASS_COLORS = {
    0: (0.0, 0.0, 0.0),       # Background — transparent
    1: (1.0, 0.0, 0.0),       # Fault — red
    2: (0.0, 0.5, 1.0),       # FS4 — blue
    3: (1.0, 0.65, 0.0),      # MFS4 — orange
    4: (0.0, 1.0, 0.5),       # FS6 — green
    5: (1.0, 1.0, 0.0),       # FS7 — yellow
    6: (0.5, 0.0, 1.0),       # FS8 — purple
    7: (0.0, 1.0, 1.0),       # Shallow — cyan
    8: (1.0, 0.4, 0.7),       # Top Foresets — pink
}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, Dict[str, float]]:
    """
    Compute IoU, Dice, Precision, and Recall for each class.

    Parameters
    ----------
    all_preds : np.ndarray
        Flattened predicted labels.
    all_targets : np.ndarray
        Flattened ground truth labels.
    num_classes : int
        Number of segmentation classes.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: class_name → {iou, dice, precision, recall}.
    """
    metrics = {}

    for c in range(num_classes):
        pred_c = (all_preds == c)
        true_c = (all_targets == c)

        tp = np.logical_and(pred_c, true_c).sum()
        fp = np.logical_and(pred_c, ~true_c).sum()
        fn = np.logical_and(~pred_c, true_c).sum()

        # IoU = TP / (TP + FP + FN)
        iou = tp / max(tp + fp + fn, 1)

        # Dice = 2TP / (2TP + FP + FN)
        dice = 2 * tp / max(2 * tp + fp + fn, 1)

        # Precision = TP / (TP + FP)
        precision = tp / max(tp + fp, 1)

        # Recall = TP / (TP + FN)
        recall = tp / max(tp + fn, 1)

        metrics[CLASS_NAMES[c]] = {
            "iou": float(iou),
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
        }

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def create_overlay_image(
    amplitude: np.ndarray,
    mask: np.ndarray,
    title: str = "Seismic Overlay",
    save_path: str = None,
) -> None:
    """
    Create a publication-quality overlay of seismic amplitude with
    colored segmentation mask.

    The seismic section is displayed in grayscale, with geological
    features overlaid as semi-transparent colored regions.

    Parameters
    ----------
    amplitude : np.ndarray, shape (H, W)
        Normalized seismic amplitude.
    mask : np.ndarray, shape (H, W)
        Class label mask with values in [0, 8].
    title : str
        Plot title.
    save_path : str or None
        If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=120)

    # Panel 1: Raw seismic
    axes[0].imshow(amplitude, cmap="gray", aspect="auto")
    axes[0].set_title("Seismic Amplitude", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Crossline")
    axes[0].set_ylabel("Sample (TWT)")

    # Panel 2: Segmentation mask
    colored_mask = np.zeros((*mask.shape, 3))
    for c in range(NUM_CLASSES):
        color = CLASS_COLORS[c]
        colored_mask[mask == c] = color

    axes[1].imshow(colored_mask, aspect="auto")
    axes[1].set_title("Predicted Segmentation", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Crossline")

    # Panel 3: Overlay (seismic + semi-transparent mask)
    axes[2].imshow(amplitude, cmap="gray", aspect="auto")

    overlay = np.zeros((*mask.shape, 4))  # RGBA
    for c in range(1, NUM_CLASSES):  # Skip background
        r, g, b = CLASS_COLORS[c]
        region = (mask == c)
        overlay[region] = [r, g, b, 0.5]  # 50% opacity

    axes[2].imshow(overlay, aspect="auto")
    axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Crossline")

    # Legend for non-background classes
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
        for c in range(1, NUM_CLASSES)
    ]
    axes[2].legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=8,
        framealpha=0.8,
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved overlay: {save_path}")

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Full-Slice Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_full_slice(
    model: torch.nn.Module,
    amplitude: np.ndarray,
    device: torch.device,
    patch_size: int = 256,
    overlap: int = 64,
) -> np.ndarray:
    """
    Run inference on a full-resolution inline slice using sliding window.

    Splits the slice into overlapping patches, predicts each, then
    merges the results using majority voting in overlap regions.

    Parameters
    ----------
    model : nn.Module
        Trained U-Net model.
    amplitude : np.ndarray, shape (H, W)
        Normalized seismic amplitude slice.
    device : torch.device
        Computation device.
    patch_size : int
        Size of each prediction patch.
    overlap : int
        Overlap between adjacent patches (in pixels).

    Returns
    -------
    np.ndarray, shape (H, W)
        Predicted class labels for the entire slice.
    """
    model.eval()
    h, w = amplitude.shape
    stride = patch_size - overlap

    # Pad the slice to fit integer number of patches
    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride
    padded = np.pad(amplitude, ((0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = padded.shape

    # Accumulate class votes for each pixel
    votes = np.zeros((NUM_CLASSES, ph, pw), dtype=np.float32)

    for y in range(0, ph - patch_size + 1, stride):
        for x in range(0, pw - patch_size + 1, stride):
            patch = padded[y : y + patch_size, x : x + patch_size]
            tensor = torch.from_numpy(
                patch[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
            ).float().to(device)

            logits = model(tensor)  # (1, C, H, W)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # (C, H, W)

            votes[:, y : y + patch_size, x : x + patch_size] += probs

    # Take argmax of accumulated votes → final prediction
    prediction = votes[:, :h, :w].argmax(axis=0).astype(np.uint8)
    return prediction


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    data_dir: str,
    checkpoint_path: str,
    output_dir: str = "outputs",
    max_overlays: int = 10,
) -> None:
    """
    Run full evaluation pipeline.

    Steps:
        1. Load best model checkpoint
        2. Run inference on test set patches → compute metrics
        3. Generate visual overlays for sample slices
        4. Print and save metric summary

    Parameters
    ----------
    data_dir : str
        Data directory with slices/ and masks/ subdirectories.
    checkpoint_path : str
        Path to the saved model checkpoint.
    output_dir : str
        Directory to save overlay images and metrics.
    max_overlays : int
        Maximum number of overlay images to generate.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    print(f"\n{'═' * 60}")
    print(f"EVALUATION")
    print(f"{'═' * 60}")
    print(f"  Device:     {device}")
    print(f"  Checkpoint: {checkpoint_path}")

    model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_dice = checkpoint.get("best_dice", "N/A")
    best_epoch = checkpoint.get("epoch", "N/A")
    print(f"  Best Dice:  {best_dice}")
    print(f"  Epoch:      {best_epoch}")

    # ── Patch-level metrics on test set ──
    print(f"\n  Computing test set metrics...")
    test_ds = SeismicPatchDataset(
        data_dir, split="test", patch_size=256,
        patches_per_slice=4, augment=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    all_preds = []
    all_targets = []

    for images, masks in test_loader:
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds.flatten())
        all_targets.append(masks.numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_metrics(all_preds, all_targets)

    # ── Print metrics table ──
    print(f"\n  {'Class':>14s} | {'IoU':>6s} | {'Dice':>6s} | "
          f"{'Precision':>9s} | {'Recall':>6s}")
    print(f"  {'─' * 52}")

    mean_iou_sum = 0.0
    valid_classes = 0

    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        m = metrics[name]
        print(
            f"  {name:>14s} | {m['iou']:6.4f} | {m['dice']:6.4f} | "
            f"{m['precision']:9.4f} | {m['recall']:6.4f}"
        )
        if c > 0:
            mean_iou_sum += m["iou"]
            valid_classes += 1

    mean_iou = mean_iou_sum / max(valid_classes, 1)
    print(f"  {'─' * 52}")
    print(f"  {'Mean IoU':>14s} | {mean_iou:6.4f} |")

    # ── Save metrics to CSV ──
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("class,iou,dice,precision,recall\n")
        for name, m in metrics.items():
            f.write(
                f"{name},{m['iou']:.6f},{m['dice']:.6f},"
                f"{m['precision']:.6f},{m['recall']:.6f}\n"
            )
    print(f"\n  Metrics saved to: {csv_path}")

    # ── Generate overlay images on full slices ──
    print(f"\n  Generating overlay images...")

    slices_dir = os.path.join(data_dir, "slices")
    masks_dir = os.path.join(data_dir, "masks")

    test_slices = sorted([
        f for f in os.listdir(slices_dir) if "test" in f and f.endswith(".npy")
    ])

    for i, fname in enumerate(test_slices[:max_overlays]):
        amp = np.load(os.path.join(slices_dir, fname))
        mask_fname = fname.replace(".npy", "_mask.npy")
        gt_mask = np.load(os.path.join(masks_dir, mask_fname))

        # Run sliding-window inference on full slice
        pred_mask = predict_full_slice(model, amp, device)

        inline_id = fname.split("_")[1]

        # Ground truth overlay
        create_overlay_image(
            amp, gt_mask,
            title=f"Inline {inline_id} — Ground Truth",
            save_path=os.path.join(output_dir, f"inline_{inline_id}_gt.png"),
        )

        # Prediction overlay
        create_overlay_image(
            amp, pred_mask,
            title=f"Inline {inline_id} — Prediction",
            save_path=os.path.join(output_dir, f"inline_{inline_id}_pred.png"),
        )

    print(f"\n{'═' * 60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Results saved to: {output_dir}/")
    print(f"{'═' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained U-Net on test seismic data",
    )
    parser.add_argument(
        "--data_dir", default="data",
        help="Path to data directory (default: data/)",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best_model.pth",
        help="Path to model checkpoint (default: checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--output_dir", default="outputs",
        help="Directory for overlay images and metrics (default: outputs/)",
    )
    parser.add_argument(
        "--max_overlays", type=int, default=10,
        help="Maximum overlay images to generate (default: 10)",
    )

    args = parser.parse_args()
    evaluate(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        max_overlays=args.max_overlays,
    )


if __name__ == "__main__":
    main()
