"""
Dataset loading and preprocessing for gravitational lens cutouts.

Expected format: directory of .npz files, each containing:
    - 'image': 2D array (101x101)
    - 'label': int (0 or 1)
    - 'params': dict with keys including 'ra', 'dec'

Normalization: clip to [p1, p99], then (img - median) / (p99 - p1).
"""

import glob
from pathlib import Path

import numpy as np


def normalize_cutout(img):
    """Normalize a single cutout image."""
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    denom = p99 - p1
    if denom < 1e-10:
        denom = 1e-10
    return (img - np.median(img)) / denom


def load_dataset(directory):
    """
    Load dataset from a directory of .npz files.

    Returns:
        images: np.ndarray (N, H, W) float32
        labels: np.ndarray (N,) int
        params: list of dicts
    """
    directory = Path(directory)

    pos_files = sorted(glob.glob(str(directory / "pos_*.npz")))
    neg_files = sorted(glob.glob(str(directory / "neg_*.npz")))
    all_files = pos_files + neg_files

    if len(all_files) == 0:
        raise FileNotFoundError(f"No .npz files found in {directory}")

    images = []
    labels = []
    params = []

    for fpath in all_files:
        z = np.load(fpath, allow_pickle=True)
        img = z["image"].astype(np.float32)
        img = normalize_cutout(img)
        images.append(img)
        labels.append(int(z["label"]))
        params.append(z["params"].item() if "params" in z else {})

    return np.array(images, dtype=np.float32), np.array(labels, dtype=int), params


def bg_group_from_params(params_list, precision=4):
    """
    Assign background group IDs from (RA, Dec) coordinates.

    Groups are defined by rounding RA and Dec to `precision` decimal
    places and hashing the result.  All images from the same parent
    galaxy cutout share the same group ID.
    """
    groups = []
    for p in params_list:
        ra = round(float(p.get("ra", 0.0)), precision)
        dec = round(float(p.get("dec", 0.0)), precision)
        groups.append(hash((ra, dec)) & 0xFFFFFFFF)
    return groups
