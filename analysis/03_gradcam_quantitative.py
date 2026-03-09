#!/usr/bin/env python3
"""
03_gradcam_quantitative.py

Quantitative Grad-CAM attribution comparison: all catastrophic groups
(FPR=1.0) vs a matched sample of clean groups (FPR=0.0).

Uses out-of-fold checkpoints to ensure no group is evaluated on its
training model.

Metrics per negative image:
  1. Attribution concentration: fraction within 15px of brightest pixel
  2. Attribution asymmetry: off-center vs center ratio
  3. Peak attribution outside core: max in annulus r=10..35

Requires:
    - Locked fold checkpoints
    - Full dataset
    - Locked test_predictions.npz per fold

Output:
    results/gradcam_summary_stats.txt
    figures/gradcam_summary_boxplot.pdf
"""

import sys
from pathlib import Path

print("[INFO] This script requires locked fold checkpoints and the full dataset.")
print("[INFO] See models/README.md and data/README.md for setup instructions.")
print("[INFO] Skeleton provided for reference; adapt paths to your local setup.")
