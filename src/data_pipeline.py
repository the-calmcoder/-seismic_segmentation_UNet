"""
data_pipeline.py — Seismic Data Extraction & Label Rasterization
================================================================
FULLY OPTIMIZED: Bulk I/O + KDTree coordinate mapping.

Performance Profile (Colab T4):
    - SEG-Y bulk load:     ~2 min
    - Header bulk read:    ~10 sec
    - KDTree coord mapping: ~5 sec
    - Rasterize 631 slices: ~2 min
    - TOTAL:               ~5 min
"""

import os
import gc
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

NUM_CLASSES = 9
CLASS_NAMES = [
    "Background", "Fault", "FS4", "MFS4", "FS6",
    "FS7", "FS8", "Shallow", "Top Foresets",
]
HORIZON_FILE_MAP = {
    "F3-Horizon-MFS4.txt":         3,
    "F3-Horizon-FS6.txt":          4,
    "F3-Horizon-FS7.txt":          5,
    "F3-Horizon-FS8.txt":          6,
    "F3-Horizon-Shallow.txt":      7,
    "F3-Horizon-Top-Foresets.txt":  8,
}
RASTER_HALF_THICKNESS = 2

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# STEP 1: Bulk SEG-Y Loader
# ─────────────────────────────────────────────────────────────────

def load_segy_bulk(segy_path: str) -> dict:
    """Load SEG-Y in bulk. Returns geometry + all inline slices."""
    import segyio

    logger.info(f"Loading SEG-Y: {segy_path}")
    logger.info(f"  Size: {os.path.getsize(segy_path) / 1e6:.0f} MB")

    with segyio.open(segy_path, "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples = len(f.samples)
        sample_rate = f.bin[segyio.BinField.Interval] / 1000.0
        t_start = float(f.samples[0])
        t_end = float(f.samples[-1])

        logger.info(f"  {n_traces:,} traces × {n_samples} samples")
        logger.info(f"  Time: {t_start:.1f}–{t_end:.1f} ms (dt={sample_rate:.2f})")

        # Bulk read ALL traces
        logger.info("  Bulk-reading trace data...")
        all_traces = f.trace.raw[:]
        logger.info(f"  ✓ Trace data: {all_traces.nbytes / 1e6:.0f} MB")

        # Bulk read ALL headers
        logger.info("  Bulk-reading headers...")
        il_nums = np.array(f.attributes(segyio.TraceField.INLINE_3D)[:], dtype=np.int32)
        xl_nums = np.array(f.attributes(segyio.TraceField.CROSSLINE_3D)[:], dtype=np.int32)
        cdp_xs = np.array(f.attributes(segyio.TraceField.CDP_X)[:], dtype=np.float64)
        cdp_ys = np.array(f.attributes(segyio.TraceField.CDP_Y)[:], dtype=np.float64)
        scalars = np.array(f.attributes(segyio.TraceField.SourceGroupScalar)[:], dtype=np.float64)
        logger.info("  ✓ Headers loaded")

    # Apply coordinate scalars
    neg = scalars < 0
    scalars[neg] = -1.0 / scalars[neg]
    scalars[scalars == 0] = 1.0
    cdp_xs *= scalars
    cdp_ys *= scalars

    # Build grid
    unique_il = np.sort(np.unique(il_nums))
    unique_xl = np.sort(np.unique(xl_nums))
    n_il, n_xl = len(unique_il), len(unique_xl)

    il_to_idx = {int(v): i for i, v in enumerate(unique_il)}
    xl_to_idx = {int(v): j for j, v in enumerate(unique_xl)}

    logger.info(f"  Grid: {n_il} inlines × {n_xl} crosslines")

    # Build coordinate grids
    cdp_x_grid = np.zeros((n_il, n_xl), dtype=np.float64)
    cdp_y_grid = np.zeros((n_il, n_xl), dtype=np.float64)

    il_idxs = np.array([il_to_idx[int(v)] for v in il_nums])
    xl_idxs = np.array([xl_to_idx[int(v)] for v in xl_nums])
    cdp_x_grid[il_idxs, xl_idxs] = cdp_xs
    cdp_y_grid[il_idxs, xl_idxs] = cdp_ys

    logger.info(f"  X: [{cdp_xs.min():.1f}, {cdp_xs.max():.1f}]")
    logger.info(f"  Y: [{cdp_ys.min():.1f}, {cdp_ys.max():.1f}]")

    # Assemble inline slices
    logger.info("  Assembling inline slices...")
    inline_slices = {}
    for il in unique_il:
        il_int = int(il)
        trace_mask = (il_nums == il)
        il_xl = xl_nums[trace_mask]
        il_data = all_traces[trace_mask]

        s = np.zeros((n_samples, n_xl), dtype=np.float32)
        for k, xv in enumerate(il_xl):
            s[:, xl_to_idx[int(xv)]] = il_data[k]
        inline_slices[il_int] = s

    del all_traces
    gc.collect()
    logger.info(f"  ✓ {len(inline_slices)} slices ready")

    return {
        "inlines": unique_il, "crosslines": unique_xl,
        "n_samples": n_samples, "sample_rate": sample_rate,
        "t_start": t_start, "t_end": t_end,
        "cdp_x": cdp_x_grid, "cdp_y": cdp_y_grid,
        "il_to_idx": il_to_idx, "xl_to_idx": xl_to_idx,
        "slices": inline_slices,
    }


# ─────────────────────────────────────────────────────────────────
# STEP 2: KDTree-based Coordinate Mapping (FAST)
# ─────────────────────────────────────────────────────────────────

def build_kdtree(cdp_x, cdp_y):
    """
    Build a KDTree from the coordinate grid for O(log N) lookups
    instead of O(N) brute-force per point.
    """
    from scipy.spatial import cKDTree

    n_il, n_xl = cdp_x.shape
    coords = np.column_stack([cdp_x.ravel(), cdp_y.ravel()])
    tree = cKDTree(coords)

    # Compute grid spacing for distance threshold
    if n_xl > 1:
        dx = cdp_x[0, 1] - cdp_x[0, 0]
        dy = cdp_y[0, 1] - cdp_y[0, 0]
        spacing = max(np.sqrt(dx**2 + dy**2), 1.0)
    else:
        spacing = 50.0

    return tree, spacing


def map_points_to_grid(points_xy, tree, spacing, n_il, n_xl):
    """
    Map an array of (X, Y) points to (inline_idx, crossline_idx)
    using the KDTree. Returns arrays of (il_idx, xl_idx) or -1 for
    points outside the grid.

    This is VECTORIZED — maps ALL points in one call.
    """
    dists, indices = tree.query(points_xy, k=1)

    il_idxs = indices // n_xl
    xl_idxs = indices % n_xl

    # Reject points too far from grid
    outside = dists > 1.5 * spacing
    il_idxs[outside] = -1
    xl_idxs[outside] = -1

    return il_idxs, xl_idxs


# ─────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────

def parse_horizon_file(filepath):
    name = Path(filepath).name
    logger.info(f"  Parsing: {name}")
    data = np.loadtxt(filepath, delimiter="\t", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    logger.info(f"    → {len(data):,} points")
    return data[:, :3]


def parse_fault_sticks(filepath):
    logger.info(f"  Parsing: {Path(filepath).name}")
    data = np.loadtxt(filepath, delimiter="\t", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    sticks = {}
    for row in data:
        sticks.setdefault(int(row[3]), []).append([row[0], row[1], row[2]])
    result = {s: np.array(p) for s, p in sticks.items()}
    total = sum(len(v) for v in result.values())
    logger.info(f"    → {len(result)} sticks, {total} points")
    return result


# ─────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────

def normalize_slice(amp):
    std = amp.std()
    if std < 1e-10:
        return np.zeros_like(amp, dtype=np.float32)
    return ((amp - amp.mean()) / std).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# Pre-compute ALL label positions (vectorized)
# ─────────────────────────────────────────────────────────────────

def precompute_labels(horizons, faults, tree, spacing, n_il, n_xl,
                      t_start, sample_rate, n_samples):
    """
    Convert ALL horizon/fault points from (X,Y,TWT) to
    (inline_idx, crossline_idx, sample_idx) in one vectorized pass.

    Returns dict: inline_idx → list of (crossline_idx, sample_idx, class_id)
    """
    half_w = RASTER_HALF_THICKNESS

    # labels_by_inline[i] = [(xl_idx, sample_idx, class_id), ...]
    labels_by_inline = defaultdict(list)

    # ── Process horizons ──
    for class_id, points in horizons.items():
        xy = points[:, :2]  # (N, 2)
        twt = points[:, 2]  # (N,)

        # Vectorized KDTree lookup — ALL points at once
        il_idxs, xl_idxs = map_points_to_grid(xy, tree, spacing, n_il, n_xl)

        # Convert TWT to sample indices (vectorized)
        sample_idxs = np.round((twt - t_start) / sample_rate).astype(np.int32)
        sample_idxs = np.clip(sample_idxs, 0, n_samples - 1)

        # Group by inline
        valid = il_idxs >= 0
        for idx in np.where(valid)[0]:
            il = int(il_idxs[idx])
            xl = int(xl_idxs[idx])
            s = int(sample_idxs[idx])

            for ds in range(-half_w, half_w + 1):
                si = s + ds
                if 0 <= si < n_samples:
                    labels_by_inline[il].append((xl, si, class_id))

        logger.info(f"    {CLASS_NAMES[class_id]}: {valid.sum():,} mapped")

    # ── Process faults ──
    if faults:
        all_fault_pts = np.vstack(list(faults.values()))
        xy = all_fault_pts[:, :2]
        depths = all_fault_pts[:, 2]

        il_idxs, xl_idxs = map_points_to_grid(xy, tree, spacing, n_il, n_xl)
        sample_idxs = np.round((depths - t_start) / sample_rate).astype(np.int32)
        sample_idxs = np.clip(sample_idxs, 0, n_samples - 1)

        valid = il_idxs >= 0
        for idx in np.where(valid)[0]:
            il = int(il_idxs[idx])
            xl = int(xl_idxs[idx])
            s = int(sample_idxs[idx])

            for ds in range(-half_w, half_w + 1):
                si = s + ds
                if 0 <= si < n_samples:
                    # Class 1 (fault) — will overwrite horizons
                    labels_by_inline[il].append((xl, si, 1))

        logger.info(f"    Fault: {valid.sum():,} mapped")

    return labels_by_inline


# ─────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────

def run_pipeline(segy_path, f3_demo_dir, output_dir="data",
                 train_end=450, val_end=550):
    slices_dir = os.path.join(output_dir, "slices")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(slices_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # ── Step 1: Bulk load SEG-Y ──
    logger.info("═" * 50)
    logger.info("STEP 1: BULK-LOADING SEG-Y")
    logger.info("═" * 50)
    geo = load_segy_bulk(segy_path)

    # ── Step 2: Build KDTree ──
    logger.info("═" * 50)
    logger.info("STEP 2: BUILDING SPATIAL INDEX")
    logger.info("═" * 50)
    tree, spacing = build_kdtree(geo["cdp_x"], geo["cdp_y"])
    n_il = len(geo["inlines"])
    n_xl = len(geo["crosslines"])
    logger.info(f"  ✓ KDTree built ({n_il}×{n_xl} = {n_il*n_xl:,} nodes)")

    # ── Step 3: Parse interpretations ──
    logger.info("═" * 50)
    logger.info("STEP 3: PARSING INTERPRETATIONS")
    logger.info("═" * 50)

    surface_dir = os.path.join(f3_demo_dir, "Rawdata", "Surface_data")
    horizons = {}
    for fname, cid in HORIZON_FILE_MAP.items():
        fp = os.path.join(surface_dir, fname)
        if os.path.exists(fp):
            horizons[cid] = parse_horizon_file(fp)

    fault_path = os.path.join(f3_demo_dir, "Rawdata", "Faults", "FaultA.txt")
    faults = parse_fault_sticks(fault_path) if os.path.exists(fault_path) else {}

    # ── Step 4: Vectorized label mapping ──
    logger.info("═" * 50)
    logger.info("STEP 4: MAPPING LABELS TO GRID (KDTree)")
    logger.info("═" * 50)

    labels_by_inline = precompute_labels(
        horizons, faults, tree, spacing, n_il, n_xl,
        geo["t_start"], geo["sample_rate"], geo["n_samples"],
    )
    logger.info(f"  ✓ Labels mapped for {len(labels_by_inline)} inlines")

    # ── Step 5: Rasterize & save ──
    logger.info("═" * 50)
    logger.info("STEP 5: RASTERIZING & SAVING")
    logger.info("═" * 50)

    inlines = geo["inlines"]
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for i, il in enumerate(inlines):
        il_int = int(il)
        amp = normalize_slice(geo["slices"][il_int])

        mask = np.zeros((geo["n_samples"], n_xl), dtype=np.uint8)

        # Paint all pre-computed labels for this inline
        if i in labels_by_inline:
            # First pass: horizons (class 2-8, don't overwrite)
            for xl_idx, s_idx, cid in labels_by_inline[i]:
                if cid != 1 and mask[s_idx, xl_idx] == 0:
                    mask[s_idx, xl_idx] = cid

            # Second pass: faults (class 1, always overwrite)
            for xl_idx, s_idx, cid in labels_by_inline[i]:
                if cid == 1:
                    mask[s_idx, xl_idx] = 1

        for c in range(NUM_CLASSES):
            counts[c] += np.sum(mask == c)

        split = "train" if i < train_end else ("val" if i < val_end else "test")
        tag = f"inline_{il_int:04d}_{split}"
        np.save(os.path.join(slices_dir, f"{tag}.npy"), amp)
        np.save(os.path.join(masks_dir, f"{tag}_mask.npy"), mask)

        if (i + 1) % 100 == 0 or (i + 1) == n_il:
            logger.info(f"  [{i+1:>4d}/{n_il}] {100*(i+1)/n_il:.1f}%")

    del geo
    gc.collect()

    # Stats
    logger.info("═" * 50)
    logger.info("DONE — CLASS DISTRIBUTION")
    logger.info("═" * 50)
    total = counts.sum()
    for c in range(NUM_CLASSES):
        n = counts[c]
        logger.info(f"  {c} {CLASS_NAMES[c]:>14s}: {n:>12,} ({100*n/total:.3f}%)")
    logger.info(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segy_path",
                        default=os.path.join("F3_Demo_2023", "Rawdata", "Seismic_data.sgy"))
    parser.add_argument("--f3_demo_dir", default="F3_Demo_2023")
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()
    run_pipeline(args.segy_path, args.f3_demo_dir, args.output_dir)


if __name__ == "__main__":
    main()
