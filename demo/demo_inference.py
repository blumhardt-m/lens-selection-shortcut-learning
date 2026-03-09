#!/usr/bin/env python3
"""
demo_inference.py

Run the ProtocolCNN lens classifier on a single FITS cutout.
Prints the positive-class logit and sigmoid probability.

Usage:
    python demo/demo_inference.py
    python demo/demo_inference.py --fits path/to/cutout.fits
    python demo/demo_inference.py --checkpoint path/to/model.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_fits(path):
    """Load a single-channel FITS cutout as a 2D numpy array."""
    try:
        from astropy.io import fits
    except ImportError:
        sys.exit("[ERROR] astropy is required: pip install astropy")

    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
    if data.ndim > 2:
        data = data[0]
    return data


def normalize(img):
    """Normalize cutout: clip [p1, p99], then (img - median) / (p99 - p1)."""
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    denom = max(p99 - p1, 1e-10)
    return (img - np.median(img)) / denom


def main():
    ap = argparse.ArgumentParser(description="Run lens classifier on a FITS cutout")
    ap.add_argument("--fits", default=str(ROOT / "demo/example_cutout.fits"),
                    help="Path to input FITS file (default: demo/example_cutout.fits)")
    ap.add_argument("--checkpoint", default=str(ROOT / "models/fold00_model.pt"),
                    help="Path to model checkpoint (default: models/fold00_model.pt)")
    args = ap.parse_args()

    # Check inputs
    if not Path(args.fits).exists():
        print(f"[ERROR] FITS file not found: {args.fits}")
        print("  Place a 101x101 single-channel FITS cutout at this path,")
        print("  or use --fits to specify a different file.")
        sys.exit(1)

    if not Path(args.checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("  Place a ProtocolCNN state_dict checkpoint at this path,")
        print("  or use --checkpoint to specify a different file.")
        print("  See models/README.md for details.")
        sys.exit(1)

    # Load and preprocess
    print(f"Loading: {args.fits}")
    img = load_fits(args.fits)
    print(f"  Shape: {img.shape}  dtype: {img.dtype}")
    img = normalize(img)

    # Load model
    try:
        import torch
    except ImportError:
        sys.exit("[ERROR] PyTorch is required: pip install torch")

    from src.model import build_model, load_checkpoint
    model = build_model()
    load_checkpoint(model, args.checkpoint)
    print(f"Model loaded: {args.checkpoint}")

    # Inference
    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logit = model(x).squeeze().item()
    prob = 1.0 / (1.0 + np.exp(-logit))

    print(f"\nResult:")
    print(f"  Logit:       {logit:+.4f}")
    print(f"  Probability: {prob:.4f}")
    if prob > 0.5:
        print(f"  Prediction:  LENS CANDIDATE")
    else:
        print(f"  Prediction:  NON-LENS")


if __name__ == "__main__":
    main()
