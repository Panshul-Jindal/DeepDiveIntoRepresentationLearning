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

# Use fast CDN mirror (HuggingFace) instead of slow University of Toronto server
CIFAR10.url = "https://huggingface.co/datasets/xingslong/cifar-10-batches-py/resolve/main/cifar-10-python.tar.gz"

from augmentations import load_cifar10_gpu, CIFAR10_MEAN, CIFAR10_STD


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
    Run frozen linear probe. Returns final-epoch top-1 accuracy (%) on test set.
    All print output mirrors to the same .log file via `logger`.
    """
    # Use logger if given, else fall back to plain print
    emit = logger.info if logger is not None else print

    # ── Freeze backbone ───────────────────────────────────────────────────
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    # ── Load datasets into GPU VRAM (instant, zero CPU DataLoader) ────────
    emit("  Loading datasets into GPU VRAM ...")
    train_data, train_labels = load_cifar10_gpu(data_root, train=True, device=device)
    test_data,  test_labels  = load_cifar10_gpu(data_root, train=False, device=device)

    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD,  device=device).view(1, 3, 1, 1)

    train_norm = (train_data.float() / 255.0 - mean) / std
    test_norm  = (test_data.float()  / 255.0 - mean) / std

    # ── Extract features once directly on GPU (~2 seconds) ────────────────
    emit("  Extracting train features on GPU ...")
    train_feats = []
    with torch.no_grad():
        for i in range(0, train_norm.shape[0], 1024):
            train_feats.append(backbone(train_norm[i : i + 1024]))
    train_feats = torch.cat(train_feats, dim=0)

    emit("  Extracting test features on GPU ...")
    test_feats = []
    with torch.no_grad():
        for i in range(0, test_norm.shape[0], 1024):
            test_feats.append(backbone(test_norm[i : i + 1024]))
    test_feats = torch.cat(test_feats, dim=0)

    # ── Linear head ───────────────────────────────────────────────────────
    classifier = LinearClassifier(feature_dim=train_feats.shape[1]).to(device)
    optimizer  = optim.SGD(classifier.parameters(), lr=lr,
                           momentum=0.9, weight_decay=weight_decay)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion  = nn.CrossEntropyLoss()

    emit(f"  {'Epoch':>6}  {'Top-1 Acc':>10}")
    emit("  " + "-" * 20)

    num_train = train_feats.shape[0]

    for epoch in range(1, epochs + 1):
        # Train linear head directly on GPU
        classifier.train()
        perm = torch.randperm(num_train, device=device)
        for i in range(0, num_train, 512):
            batch_idx = perm[i : i + 512]
            optimizer.zero_grad(set_to_none=True)
            criterion(classifier(train_feats[batch_idx]), train_labels[batch_idx]).backward()
            optimizer.step()
        scheduler.step()

        # Evaluate directly on GPU
        classifier.eval()
        with torch.no_grad():
            preds = classifier(test_feats).argmax(dim=1)
            correct = (preds == test_labels).sum().item()
            total = test_labels.size(0)

        acc      = 100.0 * correct / total

        if wandb_run is not None:
            wandb_run.log({
                "linear_eval/epoch":    epoch,
                "linear_eval/top1_acc": acc,
            })

        if epoch % log_every == 0 or epoch == epochs:
            emit(f"  {epoch:>6d}  {acc:>9.2f}%")

    emit(f"  Final Top-1 Accuracy: {acc:.2f}%")
    return acc


# ---------------------------------------------------------------------------
# Evaluate directly from a saved .pt checkpoint
# ---------------------------------------------------------------------------

def eval_checkpoint(
    ckpt_path: str | Path,
    device: torch.device | None = None,
    data_root: str = "./data",
    eval_epochs: int = 100,
    eval_lr: float = 0.1,
    wandb_run=None,
    logger=None,
    eval_file: Path | None = None,
) -> float:
    """
    Load a saved SimCLR checkpoint (.pt), extract ONLY its backbone (stripping
    the projection head), freeze it, and run linear evaluation on CIFAR-10.
    """
    from model import SimCLR

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = SimCLR().to(device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    backbone = model.encoder  # Strip projection head, retain only ResNet-18 backbone

    acc = linear_eval(
        backbone=backbone,
        device=device,
        data_root=data_root,
        lr=eval_lr,
        epochs=eval_epochs,
        wandb_run=wandb_run,
        logger=logger,
    )

    if eval_file is not None:
        eval_file = Path(eval_file)
        eval_file.parent.mkdir(parents=True, exist_ok=True)
        summary_lines = [
            "=" * 65,
            "  Linear Evaluation Summary",
            "=" * 65,
            f"  Checkpoint      : {ckpt_path.name}",
            f"  Eval epochs     : {eval_epochs}",
            f"  Eval LR         : {eval_lr}",
            f"",
            f"  >>> Final Top-1 Accuracy: {acc:.2f}%",
            "=" * 65,
        ]
        eval_file.write_text("\n".join(summary_lines) + "\n")

    return acc


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Evaluate a SimCLR checkpoint via linear probe")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--eval_epochs", type=int, default=100)
    p.add_argument("--eval_lr", type=float, default=0.1)
    p.add_argument("--eval_file", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    eval_checkpoint(
        ckpt_path=args.ckpt,
        device=dev,
        data_root=args.data_root,
        eval_epochs=args.eval_epochs,
        eval_lr=args.eval_lr,
        eval_file=Path(args.eval_file) if args.eval_file else None,
    )
