"""
Metrics for false-positive concentration and attribution analysis.
"""

import numpy as np


def gini(x):
    """Gini coefficient via the sorted-cumulative-sum formula."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * cumx.sum() / cumx[-1]) / n)


def lorenz(x):
    """
    Lorenz curve coordinates.

    Returns:
        cum_groups: np.ndarray, cumulative fraction of groups
        cum_values: np.ndarray, cumulative fraction of values
    """
    x = np.sort(np.asarray(x, dtype=float))
    cumx = np.cumsum(x)
    cumx = np.insert(cumx, 0, 0.0)
    cumx = cumx / cumx[-1]
    cum_groups = np.arange(len(cumx)) / (len(cumx) - 1)
    return cum_groups, cumx


def attribution_metrics(img, cam, radius_source=15, annulus_inner=10, annulus_outer=35):
    """
    Compute three attribution metrics for one image + Grad-CAM pair.

    Args:
        img: np.ndarray (H, W) or (1, H, W)
        cam: np.ndarray (H, W) Grad-CAM heatmap in [0, 1]
        radius_source: pixels, attribution concentration radius
        annulus_inner: inner radius for peak outer attribution
        annulus_outer: outer radius for peak outer attribution

    Returns:
        concentration: fraction of attribution near brightest pixel
        asymmetry: off-center / center attribution ratio
        peak_outer: max attribution in annulus
    """
    h, w = img.shape[-2], img.shape[-1]
    cy, cx = h // 2, w // 2

    flat_img = img.ravel() if img.ndim == 2 else img[0].ravel()
    by, bx = np.unravel_index(flat_img.argmax(), (h, w))
    yy, xx = np.mgrid[:h, :w]

    # 1. Concentration near brightest pixel
    dist_bright = np.sqrt((yy - by) ** 2 + (xx - bx) ** 2)
    mask_source = dist_bright <= radius_source
    total_attr = cam.sum()
    concentration = cam[mask_source].sum() / total_attr if total_attr > 0 else 0.0

    # 2. Asymmetry: off-center / center
    dist_center = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_half = min(h, w) / 4
    center_mask = dist_center <= r_half
    attr_center = cam[center_mask].sum()
    attr_outer = cam[~center_mask].sum()
    if attr_center > 0:
        asymmetry = attr_outer / attr_center
    else:
        asymmetry = float(attr_outer > 0) * 10.0

    # 3. Peak attribution outside core
    annulus = (dist_center >= annulus_inner) & (dist_center <= annulus_outer)
    peak_outer = float(cam[annulus].max()) if annulus.any() else 0.0

    return concentration, asymmetry, peak_outer
