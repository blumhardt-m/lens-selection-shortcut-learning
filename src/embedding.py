"""
Embedding extraction and Grad-CAM utilities.
"""

import numpy as np
import torch
from scipy.ndimage import zoom as scipy_zoom


def extract_embeddings(model, images, indices, batch_size=64, device="cpu"):
    """
    Extract 128-dim penultimate-layer embeddings for images at indices.

    Returns:
        embeddings: np.ndarray (len(indices), 128)
        logits: np.ndarray (len(indices),)
    """
    model.eval()
    all_emb = []
    all_logit = []

    for start in range(0, len(indices), batch_size):
        idx_batch = indices[start : start + batch_size]
        batch = torch.tensor(
            images[idx_batch], dtype=torch.float32
        ).unsqueeze(1).to(device)

        with torch.no_grad():
            emb = model.embed(batch).cpu().numpy()
            logit = model(batch).squeeze(-1).cpu().numpy()
        all_emb.append(emb)
        all_logit.append(logit)

    return np.concatenate(all_emb), np.concatenate(all_logit)


def gradcam(model, x_tensor, target_layer):
    """
    Grad-CAM for a single image tensor (1, 1, H, W).

    Returns heatmap (H, W) in [0, 1].
    """
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["val"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["val"] = grad_out[0].detach()

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    model.eval()
    model.zero_grad()
    x_tensor = x_tensor.clone().requires_grad_(True)
    logit = model(x_tensor)
    logit.backward()

    h_fwd.remove()
    h_bwd.remove()

    act = activations["val"]
    grad = gradients["val"]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    cam = cam.squeeze().cpu().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()
    h, w = x_tensor.shape[2], x_tensor.shape[3]
    cam = scipy_zoom(cam, (h / cam.shape[0], w / cam.shape[1]), order=1)
    return np.clip(cam, 0, 1)


def disable_inplace_relu(model):
    """Disable inplace ReLU (required for Grad-CAM backward hooks)."""
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            m.inplace = False
