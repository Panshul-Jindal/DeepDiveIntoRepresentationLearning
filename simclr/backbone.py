"""
CIFAR-10 adapted ResNet-18 backbone.

Modifications from standard ResNet-18 (following SimCLR CIFAR practice):
  - conv1: 7x7 s2 p3  →  3x3 s1 p1  (preserves 32x32 spatial resolution)
  - maxpool            →  Identity   (avoids aggressive downsampling)
  - fc                 →  Identity   (removed; projection/clustering heads attach here)

The backbone outputs a 512-dim feature vector (avgpool output).
"""

import torch
import torch.nn as nn
from torchvision import models


def build_cifar_resnet18(zero_init_residual: bool = False) -> nn.Module:
    """Return a CIFAR-10-friendly ResNet-18 with the classifier head stripped."""
    backbone = models.resnet18(weights=None, zero_init_residual=zero_init_residual)

    # 1. Replace first conv: 7x7 s2 p3 → 3x3 s1 p1
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # 2. Remove the first max-pool (would halve 32→16 prematurely)
    backbone.maxpool = nn.Identity()

    # 3. Remove the classification head – outputs raw 512-d pooled features
    backbone.fc = nn.Identity()

    return backbone


class CIFARResNet18(nn.Module):
    """
    Thin wrapper around the CIFAR-adapted ResNet-18 so it can be used as a
    self-contained module with a consistent `.feature_dim` attribute.
    """

    feature_dim: int = 512

    def __init__(self, zero_init_residual: bool = False):
        super().__init__()
        self.encoder = build_cifar_resnet18(zero_init_residual=zero_init_residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled feature vector, shape (B, 512)."""
        return self.encoder(x)
