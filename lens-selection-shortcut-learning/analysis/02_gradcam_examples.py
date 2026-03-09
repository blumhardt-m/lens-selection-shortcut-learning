#!/usr/bin/env python3
"""
02_gradcam_examples.py

Generate qualitative Grad-CAM activation maps comparing catastrophic FP
vs clean TN background groups.

Layout: 6 rows x 2 columns
  Rows 1-3: catastrophic FP groups (FPR=1.0)
  Rows 4-6: clean TN groups (FPR=0.0)
  Col 1: raw cutout (sqrt stretch)
  Col 2: Grad-CAM overlay

Requires:
    - Locked fold checkpoints under models/ or artifacts/
    - Dataset under artifacts/injected_dataset/

Output:
    figures/gradcam_catastrophic_vs_clean.pdf
"""

import sys
from pathlib import Path

print("[INFO] This script requires locked fold checkpoints and the full dataset.")
print("[INFO] See models/README.md and data/README.md for setup instructions.")
print("[INFO] Skeleton provided for reference; adapt paths to your local setup.")
