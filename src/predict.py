"""
predict.py — Single-Image Inference for Seismic Segmentation
=============================================================
Run the trained U-Net on ANY seismic image — no test set required.

Supported input formats:
    • .npy   — Pre-processed 2D numpy array (H, W) of seismic amplitudes
    • .sgy/.segy — Raw SEG-Y file (extracts all inlines, predicts each)

Outputs:
    • Colored overlay PNG (seismic + predicted faults/horizons)
    • Predicted mask as .npy file (optional, with --save_mask)

Usage Examples:
    # Predict on a single .npy slice:
    python -m src.predict --input data/slices/inline_0700_test.npy

    # Predict on a .npy slice with a ground-truth mask for comparison:
    python -m src.predict --input data/slices/inline_0700_test.npy --mask data/masks/inline_0700_test_mask.npy

    # Predict on a raw SEG-Y file (processes all inlines):
    python -m src.predict --input path/to/seismic.sgy --max_inlines 5

    # Save the predicted mask as .npy:
    python -m src.predict --input data/slices/inline_0700_test.npy --save_mask
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.model import UNet


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

NUM_CLASSES = 9

CLASS_NAMES = [
    "Background", "Fault", "FS4", "MFS4", "FS6",
    "FS7", "FS8", "Shallow", "Top Foresets",
]

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
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_amplitude(amp: np.ndarray) -> np.ndarray:
    """Z-score normalize a 2D amplitude array."""
    std = amp.std()
    if std < 1e-10:
        return np.zeros_like(amp, dtype=np.float32)
    return ((amp - amp.mean()) / std).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Sliding-Window Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_slice(
    model: torch.nn.Module,
    amplitude: np.ndarray,
    device: torch.device,
    patch_size: int = 256,
    overlap: int = 64,
) -> np.ndarray:
    """
    Run inference on a full-resolution 2D seismic slice using sliding window.

    Parameters
    ----------
    model : nn.Module
        Trained U-Net model.
    amplitude : np.ndarray, shape (H, W)
        Normalized seismic amplitude.
    device : torch.device
        Computation device (cuda/cpu).
    patch_size : int
        Size of each prediction patch.
    overlap : int
        Overlap between adjacent patches.

    Returns
    -------
    np.ndarray, shape (H, W)
        Predicted class labels for the entire slice.
    """
    model.eval()
    h, w = amplitude.shape
    stride = patch_size - overlap

    # Pad to fit integer number of patches
    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride
    padded = np.pad(amplitude, ((0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = padded.shape

    # Accumulate class votes
    votes = np.zeros((NUM_CLASSES, ph, pw), dtype=np.float32)

    for y in range(0, ph - patch_size + 1, stride):
        for x in range(0, pw - patch_size + 1, stride):
            patch = padded[y : y + patch_size, x : x + patch_size]
            tensor = torch.from_numpy(
                patch[np.newaxis, np.newaxis, :, :]
            ).float().to(device)

            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            votes[:, y : y + patch_size, x : x + patch_size] += probs

    prediction = votes[:, :h, :w].argmax(axis=0).astype(np.uint8)
    return prediction


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def create_prediction_overlay(
    amplitude: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray = None,
    title: str = "Prediction",
    save_path: str = None,
) -> None:
    """
    Create an overlay visualization. If gt_mask is provided, shows 4 panels
    (seismic, ground truth, prediction, overlay). Otherwise shows 3 panels.
    """
    has_gt = gt_mask is not None
    n_panels = 4 if has_gt else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), dpi=120)

    # Panel 1: Raw seismic
    axes[0].imshow(amplitude, cmap="gray", aspect="auto")
    axes[0].set_title("Seismic Amplitude", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Crossline")
    axes[0].set_ylabel("Sample (TWT)")

    panel_idx = 1

    # Panel 2 (optional): Ground truth
    if has_gt:
        gt_colored = np.zeros((*gt_mask.shape, 3))
        for c in range(NUM_CLASSES):
            gt_colored[gt_mask == c] = CLASS_COLORS[c]
        axes[panel_idx].imshow(gt_colored, aspect="auto")
        axes[panel_idx].set_title("Ground Truth", fontsize=12, fontweight="bold")
        axes[panel_idx].set_xlabel("Crossline")
        panel_idx += 1

    # Next panel: Prediction mask
    pred_colored = np.zeros((*pred_mask.shape, 3))
    for c in range(NUM_CLASSES):
        pred_colored[pred_mask == c] = CLASS_COLORS[c]
    axes[panel_idx].imshow(pred_colored, aspect="auto")
    axes[panel_idx].set_title("Predicted Segmentation", fontsize=12, fontweight="bold")
    axes[panel_idx].set_xlabel("Crossline")
    panel_idx += 1

    # Last panel: Overlay
    axes[panel_idx].imshow(amplitude, cmap="gray", aspect="auto")
    overlay = np.zeros((*pred_mask.shape, 4))
    for c in range(1, NUM_CLASSES):
        r, g, b = CLASS_COLORS[c]
        region = (pred_mask == c)
        overlay[region] = [r, g, b, 0.5]
    axes[panel_idx].imshow(overlay, aspect="auto")
    axes[panel_idx].set_title("Overlay", fontsize=12, fontweight="bold")
    axes[panel_idx].set_xlabel("Crossline")

    # Legend
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
        for c in range(1, NUM_CLASSES)
    ]
    axes[panel_idx].legend(
        handles=legend_patches, loc="lower right",
        fontsize=8, framealpha=0.8,
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved: {save_path}")

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Input Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_npy_input(filepath: str, max_inlines: int = None) -> list:
    """
    Load a .npy file. Handles both 2D slices and 3D volumes.

    - 2D array (H, W): treated as a single slice
    - 3D array (N, H, W): treated as N inline slices

    Returns list of (name, amplitude) tuples.
    """
    amp = np.load(filepath).astype(np.float32)
    name = Path(filepath).stem

    if amp.ndim == 2:
        return [(name, amp)]

    elif amp.ndim == 3:
        n_slices = amp.shape[0]
        print(f"  3D volume detected: {amp.shape} ({n_slices} slices)")

        if max_inlines and max_inlines < n_slices:
            indices = np.linspace(0, n_slices - 1, max_inlines, dtype=int)
        else:
            indices = range(n_slices)

        slices = []
        for i in indices:
            slices.append((f"{name}_slice_{i:04d}", amp[i]))

        print(f"  ✓ Processing {len(slices)} slices")
        return slices

    else:
        raise ValueError(
            f"Expected 2D (H,W) or 3D (N,H,W) array, got shape {amp.shape}."
        )


def load_segy_input(filepath: str, max_inlines: int = None) -> list:
    """
    Load a SEG-Y file and extract inline slices.
    Returns list of (name, amplitude) tuples.
    """
    try:
        import segyio
    except ImportError:
        raise ImportError(
            "segyio is required for SEG-Y files. Install with: pip install segyio"
        )

    print(f"  Loading SEG-Y: {filepath}")
    print(f"  Size: {os.path.getsize(filepath) / 1e6:.0f} MB")

    slices = []

    with segyio.open(filepath, "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples = len(f.samples)

        print(f"  {n_traces:,} traces × {n_samples} samples")

        # Bulk read
        all_traces = f.trace.raw[:]
        il_nums = np.array(
            f.attributes(segyio.TraceField.INLINE_3D)[:], dtype=np.int32
        )
        xl_nums = np.array(
            f.attributes(segyio.TraceField.CROSSLINE_3D)[:], dtype=np.int32
        )

    unique_il = np.sort(np.unique(il_nums))
    unique_xl = np.sort(np.unique(xl_nums))
    n_xl = len(unique_xl)
    xl_to_idx = {int(v): j for j, v in enumerate(unique_xl)}

    if max_inlines and max_inlines < len(unique_il):
        # Evenly sample inlines
        indices = np.linspace(0, len(unique_il) - 1, max_inlines, dtype=int)
        unique_il = unique_il[indices]

    print(f"  Processing {len(unique_il)} inlines...")

    for il in unique_il:
        il_int = int(il)
        trace_mask = (il_nums == il)
        il_xl = xl_nums[trace_mask]
        il_data = all_traces[trace_mask]

        s = np.zeros((n_samples, n_xl), dtype=np.float32)
        for k, xv in enumerate(il_xl):
            s[:, xl_to_idx[int(xv)]] = il_data[k]

        slices.append((f"inline_{il_int:04d}", s))

    print(f"  ✓ {len(slices)} slices loaded")
    return slices


# ─────────────────────────────────────────────────────────────────────────────
# 3D Volume Composite Visualization
# ─────────────────────────────────────────────────────────────────────────────

def create_volume_composite(
    volume: np.ndarray,
    pred_volume: np.ndarray,
    name: str,
    save_path: str,
) -> None:
    """
    Create a single composite image from a 3D seismic volume and its
    predicted segmentation, showing 3 orthogonal cross-sections:
        - Inline view   (slice along axis 0)
        - Crossline view (slice along axis 2)
        - Depth/time slice (slice along axis 1)

    Each view shows seismic + overlay side-by-side.

    Parameters
    ----------
    volume : np.ndarray, shape (N_inlines, N_samples, N_crosslines)
        The original seismic amplitude volume.
    pred_volume : np.ndarray, shape (N_inlines, N_samples, N_crosslines)
        Predicted class labels for the entire volume.
    name : str
        Name for the title.
    save_path : str
        Where to save the composite image.
    """
    n_il, n_samp, n_xl = volume.shape

    # Pick center slices for each view
    il_idx = n_il // 2
    xl_idx = n_xl // 2
    depth_idx = n_samp // 2

    # Extract cross-sections
    inline_amp = volume[il_idx, :, :]           # (N_samples, N_crosslines)
    inline_pred = pred_volume[il_idx, :, :]

    crossline_amp = volume[:, :, xl_idx]        # (N_inlines, N_samples)
    crossline_pred = pred_volume[:, :, xl_idx]

    depth_amp = volume[:, depth_idx, :]         # (N_inlines, N_crosslines)
    depth_pred = pred_volume[:, depth_idx, :]

    views = [
        (inline_amp, inline_pred,
         f"Inline {il_idx}", "Crossline", "Time (sample)"),
        (crossline_amp, crossline_pred,
         f"Crossline {xl_idx}", "Inline", "Time (sample)"),
        (depth_amp, depth_pred,
         f"Depth Slice {depth_idx}", "Crossline", "Inline"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 18), dpi=120)

    for row, (amp, pred, title, xlabel, ylabel) in enumerate(views):
        # Normalize for display
        amp_disp = normalize_amplitude(amp)

        # Left: seismic amplitude
        axes[row, 0].imshow(amp_disp, cmap="gray", aspect="auto")
        axes[row, 0].set_title(f"{title} — Seismic", fontsize=12, fontweight="bold")
        axes[row, 0].set_xlabel(xlabel)
        axes[row, 0].set_ylabel(ylabel)

        # Right: seismic + colored overlay
        axes[row, 1].imshow(amp_disp, cmap="gray", aspect="auto")

        overlay = np.zeros((*pred.shape, 4))
        for c in range(1, NUM_CLASSES):
            r, g, b = CLASS_COLORS[c]
            region = (pred == c)
            overlay[region] = [r, g, b, 0.6]

        axes[row, 1].imshow(overlay, aspect="auto")
        axes[row, 1].set_title(
            f"{title} — Prediction Overlay", fontsize=12, fontweight="bold"
        )
        axes[row, 1].set_xlabel(xlabel)

    # Add legend to the last panel
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
        for c in range(1, NUM_CLASSES)
    ]
    axes[2, 1].legend(
        handles=legend_patches, loc="lower right",
        fontsize=9, framealpha=0.9,
    )

    # Summarize detected classes
    all_classes = np.unique(pred_volume)
    detected = [CLASS_NAMES[c] for c in all_classes if c > 0]
    detected_str = ", ".join(detected) if detected else "Background only"

    fig.suptitle(
        f"{name}\n"
        f"Volume: {n_il} inlines × {n_samp} samples × {n_xl} crosslines\n"
        f"Detected: {detected_str}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ Composite image saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Prediction Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    input_path: str,
    checkpoint_path: str = "checkpoints/best_model.pth",
    output_dir: str = "outputs/predictions",
    mask_path: str = None,
    save_mask: bool = False,
    max_inlines: int = None,
) -> None:
    """
    Run prediction on a single input file.

    For 2D inputs: produces a single per-slice overlay image.
    For 3D volumes: predicts ALL slices and produces one composite image
    showing inline, crossline, and depth cross-sections.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    print(f"\n{'═' * 60}")
    print(f"SINGLE-IMAGE PREDICTION")
    print(f"{'═' * 60}")
    print(f"  Device:     {device}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Input:      {input_path}")

    model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_dice = checkpoint.get("best_dice", "N/A")
    print(f"  Model Dice: {best_dice}")

    # ── Detect input type ──
    ext = Path(input_path).suffix.lower()
    raw_data = None

    if ext == ".npy":
        raw_data = np.load(input_path).astype(np.float32)
    elif ext in (".sgy", ".segy"):
        pass  # handled below
    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. Supported: .npy, .sgy, .segy"
        )

    # ── 3D Volume Path ──
    is_volume = (raw_data is not None and raw_data.ndim == 3)

    if is_volume:
        name = Path(input_path).stem
        n_slices, n_samp, n_xl = raw_data.shape
        print(f"  3D volume: {raw_data.shape} ({n_slices} slices)")

        # Predict ALL slices
        print(f"\n  Predicting all {n_slices} slices...")
        print(f"{'─' * 60}")

        pred_volume = np.zeros_like(raw_data, dtype=np.uint8)

        for i in range(n_slices):
            amp_norm = normalize_amplitude(raw_data[i])
            pred_volume[i] = predict_slice(model, amp_norm, device)

            if (i + 1) % 20 == 0 or (i + 1) == n_slices:
                print(f"  [{i+1:>4d}/{n_slices}] {100*(i+1)/n_slices:.0f}%")

        # Summary of detected classes
        all_classes = np.unique(pred_volume)
        detected = [CLASS_NAMES[c] for c in all_classes if c > 0]
        print(f"\n  Classes detected: "
              f"{', '.join(detected) if detected else 'Background only'}")

        # Create single composite image
        composite_path = os.path.join(output_dir, f"{name}_composite.png")
        create_volume_composite(raw_data, pred_volume, name, composite_path)

        # Save full 3D predicted mask
        if save_mask:
            mask_path_out = os.path.join(output_dir, f"{name}_pred_volume.npy")
            np.save(mask_path_out, pred_volume)
            print(f"  ✓ 3D mask saved: {mask_path_out}")

    # ── 2D Slice Path (single .npy or SEG-Y) ──
    else:
        if ext == ".npy":
            slices_to_predict = [(Path(input_path).stem, raw_data)]
        else:
            slices_to_predict = load_segy_input(input_path, max_inlines)

        # Load optional ground-truth mask
        gt_mask = None
        if mask_path:
            gt_mask = np.load(mask_path).astype(np.uint8)
            print(f"  GT Mask:    {mask_path}")

        print(f"\n  Predicting {len(slices_to_predict)} slice(s)...")
        print(f"{'─' * 60}")

        for i, (name, amplitude) in enumerate(slices_to_predict):
            print(f"  [{i+1}/{len(slices_to_predict)}] {name}"
                  f" — shape {amplitude.shape}")

            amp_norm = normalize_amplitude(amplitude)
            pred_mask = predict_slice(model, amp_norm, device)

            unique_classes = np.unique(pred_mask)
            detected = [CLASS_NAMES[c] for c in unique_classes if c > 0]
            print(f"         Detected: "
                  f"{', '.join(detected) if detected else 'Background only'}")

            overlay_path = os.path.join(output_dir, f"{name}_prediction.png")
            slice_gt = gt_mask if gt_mask is not None else None

            create_prediction_overlay(
                amp_norm, pred_mask,
                gt_mask=slice_gt,
                title=f"{name} — Prediction",
                save_path=overlay_path,
            )

            if save_mask:
                mask_save_path = os.path.join(
                    output_dir, f"{name}_pred_mask.npy"
                )
                np.save(mask_save_path, pred_mask)
                print(f"         Mask saved: {mask_save_path}")

    print(f"\n{'═' * 60}")
    print(f"PREDICTION COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Results saved to: {output_dir}/")
    print(f"{'═' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse arguments and run prediction."""
    parser = argparse.ArgumentParser(
        description="Run seismic segmentation on any .npy or .segy file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on a .npy slice:
  python -m src.predict --input data/slices/inline_0700_test.npy

  # Predict on a 3D volume (single composite output):
  python -m src.predict --input test1_seismic.npy --save_mask

  # Predict with ground-truth comparison:
  python -m src.predict --input data/slices/inline_0700_test.npy \\
                        --mask data/masks/inline_0700_test_mask.npy

  # Predict on SEG-Y file:
  python -m src.predict --input seismic_data.sgy
        """,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input file (.npy or .sgy/.segy)",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best_model.pth",
        help="Path to model checkpoint (default: checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--output_dir", default="outputs/predictions",
        help="Directory for output images (default: outputs/predictions/)",
    )
    parser.add_argument(
        "--mask", default=None,
        help="Optional ground-truth mask .npy for comparison overlay",
    )
    parser.add_argument(
        "--save_mask", action="store_true",
        help="Save predicted mask as .npy file",
    )
    parser.add_argument(
        "--max_inlines", type=int, default=None,
        help="For SEG-Y: max inlines to process (default: all)",
    )

    args = parser.parse_args()
    predict(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        mask_path=args.mask,
        save_mask=args.save_mask,
        max_inlines=args.max_inlines,
    )


if __name__ == "__main__":
    main()
