"""
dataset.py — PyTorch Dataset for Seismic Segmentation
=====================================================
Loads paired .npy files (amplitude slices + masks) and yields 256×256
patches with optional augmentation for training.

Design notes:
    • Patches are extracted on-the-fly to maximize data diversity
    • Augmentations are geophysically valid (flips, noise — no rotation)
    • Memory efficient: loads one slice at a time from disk

Usage:
    from src.dataset import SeismicPatchDataset, build_dataloaders
    train_loader, val_loader, test_loader = build_dataloaders("data/")
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────────────────────────────────────

class SeismicPatchDataset(Dataset):
    """
    PyTorch Dataset that extracts 256×256 patches from seismic inline slices.

    Each item is a (image, mask) pair where:
        • image: FloatTensor of shape (1, H, W) — normalized amplitude
        • mask:  LongTensor  of shape (H, W)   — class labels [0..8]

    Parameters
    ----------
    data_dir : str
        Root data directory containing slices/ and masks/ subdirectories.
    split : str
        One of 'train', 'val', 'test'. Filters files by filename suffix.
    patch_size : int
        Height and width of extracted patches (default: 256).
    patches_per_slice : int
        Number of random patches to extract from each slice (default: 8).
    augment : bool
        Whether to apply geometric and noise augmentations (default: False).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        patch_size: int = 256,
        patches_per_slice: int = 8,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patches_per_slice = patches_per_slice
        self.augment = augment

        # Discover slice/mask file pairs for the requested split
        slices_dir = os.path.join(data_dir, "slices")
        masks_dir = os.path.join(data_dir, "masks")

        self.slice_paths: List[str] = []
        self.mask_paths: List[str] = []

        for fname in sorted(os.listdir(slices_dir)):
            if not fname.endswith(".npy") or split not in fname:
                continue

            # Construct matching mask filename
            mask_fname = fname.replace(".npy", "_mask.npy")
            mask_path = os.path.join(masks_dir, mask_fname)

            if os.path.exists(mask_path):
                self.slice_paths.append(os.path.join(slices_dir, fname))
                self.mask_paths.append(mask_path)

        if len(self.slice_paths) == 0:
            raise FileNotFoundError(
                f"No '{split}' slice files found in {slices_dir}. "
                f"Run data_pipeline.py first."
            )

        print(
            f"  SeismicPatchDataset [{split:>5s}]: "
            f"{len(self.slice_paths)} slices × "
            f"{patches_per_slice} patches = "
            f"{len(self)} items"
        )

    def __len__(self) -> int:
        """Total number of patches across all slices."""
        return len(self.slice_paths) * self.patches_per_slice

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a slice and extract a random patch.

        Returns
        -------
        image : torch.FloatTensor
            Shape (1, patch_size, patch_size).
        mask : torch.LongTensor
            Shape (patch_size, patch_size).
        """
        # Determine which slice & load it
        slice_idx = idx // self.patches_per_slice
        amplitude = np.load(self.slice_paths[slice_idx])  # (H, W)
        mask = np.load(self.mask_paths[slice_idx])          # (H, W)

        # Extract a random patch
        image_patch, mask_patch = self._extract_random_patch(amplitude, mask)

        # Apply augmentations (training only)
        if self.augment:
            image_patch, mask_patch = self._apply_augmentations(
                image_patch, mask_patch
            )

        # Convert to tensors
        image_tensor = torch.from_numpy(
            image_patch[np.newaxis, :, :]  # Add channel dim: (1, H, W)
        ).float()
        mask_tensor = torch.from_numpy(mask_patch).long()

        return image_tensor, mask_tensor

    # ── Patch Extraction ──────────────────────────────────────────────────

    def _extract_random_patch(
        self, amplitude: np.ndarray, mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a random patch_size × patch_size crop from the slice.

        If the slice is smaller than patch_size in either dimension,
        the image is padded with zeros and the mask with background (0).

        Returns
        -------
        image_patch : np.ndarray, shape (patch_size, patch_size)
        mask_patch  : np.ndarray, shape (patch_size, patch_size)
        """
        h, w = amplitude.shape
        ps = self.patch_size

        # Pad if necessary
        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            amplitude = np.pad(amplitude, ((0, pad_h), (0, pad_w)), mode="constant")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
            h, w = amplitude.shape

        # Random crop origin
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)

        return (
            amplitude[top : top + ps, left : left + ps].copy(),
            mask[top : top + ps, left : left + ps].copy(),
        )

    # ── Augmentations ─────────────────────────────────────────────────────

    def _apply_augmentations(
        self, image: np.ndarray, mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply geophysically valid augmentations.

        Augmentations applied (each with 50% probability):
            1. Horizontal flip — valid for seismic (survey direction invariant)
            2. Vertical flip   — valid (time reversal symmetry)
            3. Gaussian noise  — simulates acquisition noise

        Note: Rotations are NOT applied because they can create unrealistic
        geological configurations (e.g., slanted horizontal reflectors).

        Parameters
        ----------
        image : np.ndarray, shape (H, W)
        mask  : np.ndarray, shape (H, W)

        Returns
        -------
        Augmented image and mask (same shapes).
        """
        # Horizontal flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Vertical flip
        if random.random() > 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        # Additive Gaussian noise (σ = 0.05, subtle)
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.05, image.shape).astype(np.float32)
            image = image + noise

        return image, mask


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    patch_size: int = 256,
    patches_per_slice: int = 8,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from extracted .npy files.

    Parameters
    ----------
    data_dir : str
        Root data directory (containing slices/ and masks/).
    batch_size : int
        Batch size for all loaders (default: 16, tuned for Colab T4).
    patch_size : int
        Spatial size of each patch (default: 256).
    patches_per_slice : int
        Random patches extracted per full-sized inline slice.
    num_workers : int
        DataLoader worker processes (default: 2 for Colab).

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader, test_loader)
    """
    print(f"\nBuilding DataLoaders from: {data_dir}")
    print(f"  batch_size={batch_size}, patch_size={patch_size}")

    train_ds = SeismicPatchDataset(
        data_dir, split="train",
        patch_size=patch_size,
        patches_per_slice=patches_per_slice,
        augment=True,
    )
    val_ds = SeismicPatchDataset(
        data_dir, split="val",
        patch_size=patch_size,
        patches_per_slice=patches_per_slice,
        augment=False,
    )
    test_ds = SeismicPatchDataset(
        data_dir, split="test",
        patch_size=patch_size,
        patches_per_slice=patches_per_slice,
        augment=False,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    print(
        f"  Batches per epoch — "
        f"train: {len(train_loader)}, "
        f"val: {len(val_loader)}, "
        f"test: {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader
