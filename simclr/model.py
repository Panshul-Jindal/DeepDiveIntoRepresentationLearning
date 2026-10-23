"""
SimCLR model: CIFAR-ResNet-18 encoder + MLP projection head.

Architecture follows Chen et al. 2020 (SimCLR v1):
  - Backbone  : CIFAR-ResNet-18 → 512-dim pooled feature  (h)
  - Projector : 2-layer MLP with BN & ReLU                (z)
               512 → proj_hidden_dim → proj_out_dim
               Default: 512 → 2048 → 128   (scaled down from paper's 2048-backbone
               version, but keeps the same relative ratio for ResNet-18)

Loss: NT-Xent  (Normalized Temperature-scaled Cross Entropy)
  - Operates on L2-normalised z vectors
  - Temperature τ (default 0.5, SimCLR CIFAR-10 setting)

Reference: https://arxiv.org/abs/2002.05709
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure simclr directory is on sys.path for direct or module execution
_SIMCLR_DIR = str(Path(__file__).resolve().parent)
if _SIMCLR_DIR not in sys.path:
    sys.path.insert(0, _SIMCLR_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import CIFARResNet18


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head as used in SimCLR.

    Architecture:
        Linear(in_dim, hidden_dim) → BN → ReLU → Linear(hidden_dim, out_dim)

    Note: No activation / BN on the final layer (per SimCLR v1).
    SimCLR v2 adds a third layer; v1 uses 2 layers — we follow v1 here.
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 2048,
        out_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross-Entropy loss.

    Given a batch of N samples with 2 views each (2N total):
      - Positive pair  : (z_i, z_j) from the same image
      - Negative pairs : all other 2(N-1) samples in the batch

    Implementation follows Algorithm 1 in Chen et al. 2020.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (N, D) projected embeddings for view 1
            z2: (N, D) projected embeddings for view 2
        Returns:
            scalar loss
        """
        N = z1.size(0)
        device = z1.device

        # L2-normalise
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate: [z1; z2]  shape (2N, D)
        z = torch.cat([z1, z2], dim=0)

        # Similarity matrix (2N, 2N), scaled by temperature
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Mask out diagonal (self-similarity)
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive pair indices
        # For view1[i] → positive is view2[i], i.e. index i+N
        # For view2[i] → positive is view1[i], i.e. index i
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),  # positives for first N
            torch.arange(0, N,     device=device),  # positives for second N
        ])  # (2N,)

        loss = F.cross_entropy(sim, labels)
        return loss


# ---------------------------------------------------------------------------
# Full SimCLR model
# ---------------------------------------------------------------------------

class SimCLR(nn.Module):
    """
    SimCLR = CIFARResNet18 backbone + ProjectionHead.

    During pre-training, forward() returns projected embeddings (z1, z2).
    For downstream evaluation, use .encoder directly to obtain h.
    """

    def __init__(
        self,
        proj_hidden_dim: int = 2048,
        proj_out_dim: int = 128,
        zero_init_residual: bool = True,
    ):
        super().__init__()
        self.encoder = CIFARResNet18(zero_init_residual=zero_init_residual)
        self.projector = ProjectionHead(
            in_dim=self.encoder.feature_dim,  # 512
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim,
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x1: (N, C, H, W) first view
            x2: (N, C, H, W) second view
        Returns:
            z1, z2: (N, proj_out_dim) projected embeddings
        """
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return z1, z2

    @torch.no_grad()
    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features h (no projection) for evaluation."""
        return self.encoder(x)
