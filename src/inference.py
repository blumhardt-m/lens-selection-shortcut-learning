"""
Inference utilities for the ProtocolCNN lens classifier.
"""

import numpy as np
import torch

from .model import ProtocolCNN, load_checkpoint


def predict_single(model, image, device="cpu"):
    """
    Run inference on a single image.

    Args:
        model: ProtocolCNN instance (eval mode)
        image: np.ndarray (H, W) or (1, H, W), already normalized
        device: torch device

    Returns:
        logit: float, raw positive-class logit
        prob: float, sigmoid probability
    """
    x = torch.tensor(image, dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        logit = model(x).squeeze().item()
    prob = 1.0 / (1.0 + np.exp(-logit))
    return logit, prob


def predict_batch(model, images, indices, batch_size=64, device="cpu"):
    """
    Run inference on a batch of images at given indices.

    Args:
        model: ProtocolCNN instance (eval mode)
        images: np.ndarray (N, H, W)
        indices: array-like of indices into images
        batch_size: int
        device: torch device

    Returns:
        scores: np.ndarray of logits, shape (len(indices),)
    """
    model.eval()
    scores = []
    for start in range(0, len(indices), batch_size):
        idx_batch = indices[start : start + batch_size]
        batch = torch.tensor(
            images[idx_batch], dtype=torch.float32
        ).unsqueeze(1).to(device)
        with torch.no_grad():
            logits = model(batch).squeeze(-1).cpu().numpy()
        scores.append(logits)
    return np.concatenate(scores)
