"""
ProtocolCNN: fixed architecture for gravitational lens classification.

4-block CNN: Conv2d(1->16->32->64->128) + BN + ReLU + MaxPool/AvgPool
Embedding: 128-dim penultimate layer (Linear(2048->128) + ReLU)
Head: Linear(128->1)
Total parameters: 360,033

This architecture is fixed across all experiments. Do not modify it.
"""

import torch
import torch.nn as nn


class ProtocolCNN(nn.Module):
    """
    Protocol-compliant CNN for gravitational lens classification.

    Input: (batch, 1, 101, 101) single-channel galaxy cutout
    Output: (batch, 1) positive-class logit
    """

    def __init__(self, in_channels=1):
        super().__init__()

        def block(c_in, c_out, pool):
            layers = [
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            if pool == "max":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.AdaptiveAvgPool2d(4))
            return layers

        self.features = nn.Sequential(
            *block(in_channels, 16, "max"),
            *block(16, 32, "max"),
            *block(32, 64, "max"),
            *block(64, 128, "avg"),
        )
        self.embed_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(128, 1)

    def embed(self, x):
        """Return 128-dim penultimate-layer embedding."""
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.embed_layer(self.features(x))

    def forward(self, x):
        return self.head(self.embed(x))


def build_model(in_channels=1):
    """Build the locked ProtocolCNN architecture."""
    return ProtocolCNN(in_channels=in_channels)


def load_checkpoint(model, checkpoint_path, device="cpu"):
    """Load a saved state_dict into the model."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
