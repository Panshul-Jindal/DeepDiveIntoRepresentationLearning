"""
Linear evaluation protocol for SimCLR.

Accepts a live in-memory backbone (no checkpoint file needed).
All logging goes through the same logger as train.py, so output appears
in both stdout and the run's .log file.

Protocol:
  1. Freeze backbone encoder.
  2. Extract and cache all features once (single pass over dataset).
  3. Train a single linear layer on top for `epochs` epochs.
  4. Report top-1 accuracy on CIFAR-10 test set every `log_every` epochs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SIMCLR_DIR = str(Path(__file__).resolve().parent)
if _SIMCLR_DIR not in sys.path:
    sys.path.insert(0, _SIMCLR_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

from augmentations import build_eval_transform


# ---------------------------------------------------------------------------
# Linear classifier head
# ---------------------------------------------------------------------------

class LinearClassifier(nn.Module):
    def __init__(self, feature_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> "tuple[torch.Tensor, torch.Tensor]":
    backbone.eval()
    feats, labels = [], []
    for x, y in loader:
        feats.append(backbone(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def linear_eval(
    backbone: nn.Module,    # live in-memory encoder; will be frozen, not modified
    device: torch.device,
    data_root: str = "./data",
    batch_size: int = 512,
    lr: float = 0.1,
    weight_decay: float = 0.0,
    epochs: int = 100,
    wandb_run=None,         # active W&B run object; None → skip W&B logging
    logger=None,            # logging.Logger from train.py → stdout + .log file
    log_every: int = 10,    # log accuracy every N epochs
) -> float:
    """
    Run frozen linear probe. Returns best top-1 accuracy (%) on test set.
    All print output mirrors to the same .log file via `logger`.
    """
    # Use logger if given, else fall back to plain print
    emit = logger.info if logger is not None else print

    # ── Freeze backbone ───────────────────────────────────────────────────
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    # ── Datasets ─────────────────────────────────────────────────────────
    eval_tf  = build_eval_transform()
    train_ds = CIFAR10(data_root, train=True,  download=True, transform=eval_tf)
    test_ds  = CIFAR10(data_root, train=False, download=True, transform=eval_tf)

    raw_train = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    raw_test  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

    # ── Extract features once ─────────────────────────────────────────────
    emit("  Extracting train features ...")
    train_feats, train_labels = extract_features(backbone, raw_train, device)
    emit("  Extracting test features ...")
    test_feats, test_labels   = extract_features(backbone, raw_test,  device)

    train_feat_loader = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=512, shuffle=True, num_workers=0,
    )
    test_feat_loader = DataLoader(
        TensorDataset(test_feats, test_labels),
        batch_size=512, shuffle=False, num_workers=0,
    )

    # ── Linear head ───────────────────────────────────────────────────────
    classifier = LinearClassifier(feature_dim=train_feats.shape[1]).to(device)
    optimizer  = optim.SGD(classifier.parameters(), lr=lr,
                           momentum=0.9, weight_decay=weight_decay)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion  = nn.CrossEntropyLoss()

    emit(f"  {'Epoch':>6}  {'Top-1 Acc':>10}")
    emit("  " + "-" * 20)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Train linear head
        classifier.train()
        for feat, lbl in train_feat_loader:
            feat, lbl = feat.to(device), lbl.to(device)
            optimizer.zero_grad()
            criterion(classifier(feat), lbl).backward()
            optimizer.step()
        scheduler.step()

        # Evaluate
        classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for feat, lbl in test_feat_loader:
                feat, lbl = feat.to(device), lbl.to(device)
                correct += (classifier(feat).argmax(1) == lbl).sum().item()
                total   += lbl.size(0)

        acc      = 100.0 * correct / total
        best_acc = max(best_acc, acc)

        if wandb_run is not None:
            wandb_run.log({
                "linear_eval/epoch":    epoch,
                "linear_eval/top1_acc": acc,
            })

        if epoch % log_every == 0 or epoch == epochs:
            emit(f"  {epoch:>6d}  {acc:>9.2f}%")

    emit(f"  Best Top-1 Accuracy: {best_acc:.2f}%")
    return best_acc
