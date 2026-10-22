"""
SimCLR pre-training script for CIFAR-10.

Logging:
  - Every epoch is printed to stdout AND written to logs/{run_name}.log
  - The log file starts with a full config header (hyperparams + aug settings)
  - Final checkpoint saved to logs/checkpoints/{run_name}_final.pt
  - Linear eval summary saved to logs/{run_name}_eval.txt

Usage:
    python train.py --aug_preset full_aug
    python ablation.py [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure simclr directory is on sys.path
_SIMCLR_DIR = str(Path(__file__).resolve().parent)
if _SIMCLR_DIR not in sys.path:
    sys.path.insert(0, _SIMCLR_DIR)

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import wandb

from augmentations import (
    TwoViewTransform,
    build_simclr_transform,
    get_augmentation_config,
    AUGMENTATION_REGISTRY,
)
from model import SimCLR, NTXentLoss
from linear_eval import linear_eval


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SimCLR pre-training on CIFAR-10")

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument("--data_root",   type=str, default="./data")

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--proj_hidden_dim", type=int, default=2048)
    p.add_argument("--proj_out_dim",    type=int, default=128)

    # ── Loss ──────────────────────────────────────────────────────────────
    p.add_argument("--temperature", type=float, default=0.07)

    # ── Augmentation ──────────────────────────────────────────────────────
    p.add_argument(
        "--aug_preset", type=str, default="full_aug",
        choices=list(AUGMENTATION_REGISTRY.keys()),
    )

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--epochs",        type=int,   default=500)
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--lr",            type=float, default=None,
                   help="Base LR. Default: 0.3 × batch_size/256")
    p.add_argument("--weight_decay",  type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int,   default=10)
    p.add_argument("--optimizer",     type=str,   default="adam",
                   choices=["adam", "sgd"])

    # ── AMP ───────────────────────────────────────────────────────────────
    p.add_argument("--amp",    action="store_true",  default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")

    # ── System ────────────────────────────────────────────────────────────
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--seed",        type=int, default=42)

    # ── Logging ───────────────────────────────────────────────────────────
    p.add_argument("--log_dir", type=str, default="./logs",
                   help="Directory for .log files, checkpoints, eval summaries")

    # ── Linear eval ───────────────────────────────────────────────────────
    p.add_argument("--eval_epochs", type=int,   default=100)
    p.add_argument("--eval_lr",     type=float, default=0.1)
    p.add_argument("--skip_eval",   action="store_true")

    # ── W&B ───────────────────────────────────────────────────────────────
    p.add_argument("--wandb_project", type=str, default="simclr-cifar10")
    p.add_argument("--wandb_entity",  type=str, default=None)
    p.add_argument("--run_name",      type=str, default=None)
    p.add_argument("--offline",       action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# LR helpers
# ---------------------------------------------------------------------------

def get_base_lr(args: argparse.Namespace) -> float:
    if args.lr is not None:
        return args.lr
    if args.optimizer == "adam":
        return 1e-3 * (args.batch_size / 256.0)
    return 0.3 * args.batch_size / 256.0


def make_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    import math

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Logger setup — writes to stdout AND a plain .log file simultaneously
# ---------------------------------------------------------------------------

def setup_logger(log_path: Path) -> logging.Logger:
    """
    Returns a logger that writes every message to:
      1. stdout  (so you see it live in your terminal)
      2. logs/{run_name}.log  (plain text, same content)
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    logger = logging.getLogger("simclr")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # avoid duplicate handlers on reruns

    # stdout handler
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file handler  (.log, plain text)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Augmentation config ───────────────────────────────────────────────
    aug_cfg = get_augmentation_config(args.aug_preset)

    # ── Run name ──────────────────────────────────────────────────────────
    if args.run_name is None:
        args.run_name = f"simclr_{aug_cfg.tag}"

    # ── Paths ─────────────────────────────────────────────────────────────
    log_dir   = Path(args.log_dir)
    log_file  = log_dir / f"{args.run_name}.log"
    ckpt_dir  = log_dir / "checkpoints"
    eval_file = log_dir / f"{args.run_name}_eval.txt"

    # ── Logger ────────────────────────────────────────────────────────────
    log = setup_logger(log_file)

    # ── Model ─────────────────────────────────────────────────────────────
    model = SimCLR(
        proj_hidden_dim=args.proj_hidden_dim,
        proj_out_dim=args.proj_out_dim,
        zero_init_residual=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.encoder.parameters())
    projector_params = sum(p.numel() for p in model.projector.parameters())

    criterion = NTXentLoss(temperature=args.temperature)

    # ── Config header in the log ──────────────────────────────────────────
    base_lr = get_base_lr(args)
    log.info("=" * 65)
    log.info(f"  SimCLR Pre-training — CIFAR-10")
    log.info("=" * 65)
    log.info(f"  Run name       : {args.run_name}")
    log.info(f"  Device         : {device}")
    log.info("")
    log.info("  ── Model Architecture & Parameters ────────────────────")
    log.info(f"  Backbone       : CIFAR-ResNet-18 (3x3 conv1, no maxpool)")
    log.info(f"  Proj head      : 512 → {args.proj_hidden_dim} → {args.proj_out_dim}")
    log.info(f"  Total params   : {total_params:,} ({total_params / 1e6:.2f}M)")
    log.info(f"    Backbone     : {backbone_params:,} ({backbone_params / 1e6:.2f}M)")
    log.info(f"    Projector    : {projector_params:,} ({projector_params / 1e6:.2f}M)")
    log.info(f"  Trainable      : {trainable_params:,}")
    log.info("")
    log.info("  ── Loss ───────────────────────────────────────────────")
    log.info(f"  Loss           : NT-Xent")
    log.info(f"  Temperature τ  : {args.temperature}")
    log.info("")
    log.info("  ── Training ───────────────────────────────────────────")
    log.info(f"  Epochs         : {args.epochs}")
    log.info(f"  Batch size     : {args.batch_size}")
    log.info(f"  Optimizer      : {args.optimizer}")
    log.info(f"  Base LR        : {base_lr:.6f}")
    log.info(f"  Weight decay   : {args.weight_decay}")
    log.info(f"  Warmup epochs  : {args.warmup_epochs}")
    log.info(f"  AMP            : {args.amp}")
    log.info(f"  Seed           : {args.seed}")
    log.info("")
    log.info("  ── Augmentation ───────────────────────────────────────")
    log.info(f"  Preset         : {args.aug_preset}  (tag: {aug_cfg.tag})")
    log.info(f"  Random crop    : scale={aug_cfg.crop_scale}, size={aug_cfg.crop_size}")
    log.info(f"  Horiz flip     : {aug_cfg.use_hflip}")
    log.info(f"  Color jitter   : {aug_cfg.use_color_jitter}"
             + (f"  (strength s={aug_cfg.color_jitter_strength})" if aug_cfg.use_color_jitter else ""))
    log.info(f"    brightness   : {aug_cfg.jitter_brightness * aug_cfg.color_jitter_strength:.2f}")
    log.info(f"    contrast     : {aug_cfg.jitter_contrast   * aug_cfg.color_jitter_strength:.2f}")
    log.info(f"    saturation   : {aug_cfg.jitter_saturation * aug_cfg.color_jitter_strength:.2f}")
    log.info(f"    hue          : {aug_cfg.jitter_hue        * aug_cfg.color_jitter_strength:.2f}")
    log.info(f"    apply prob   : {aug_cfg.jitter_prob}")
    log.info(f"  Grayscale      : {aug_cfg.use_grayscale}"
             + (f"  (p={aug_cfg.grayscale_prob})" if aug_cfg.use_grayscale else ""))
    log.info(f"  Gaussian blur  : {aug_cfg.use_blur}"
             + (f"  (kernel={aug_cfg.blur_kernel_size}, σ={aug_cfg.blur_sigma})" if aug_cfg.use_blur else ""))
    log.info("")
    log.info("  ── Linear Eval ────────────────────────────────────────")
    log.info(f"  Eval epochs    : {args.eval_epochs}")
    log.info(f"  Eval LR        : {args.eval_lr}")
    log.info("=" * 65)
    log.info("")

    # ── W&B ───────────────────────────────────────────────────────────────
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        mode="offline" if args.offline else "online",
        config={
            "backbone":              "cifar_resnet18",
            "total_params":          total_params,
            "backbone_params":       backbone_params,
            "projector_params":      projector_params,
            "proj_hidden_dim":       args.proj_hidden_dim,
            "proj_out_dim":          args.proj_out_dim,
            "temperature":           args.temperature,
            "aug_preset":            args.aug_preset,
            "aug_tag":               aug_cfg.tag,
            "crop_scale":            list(aug_cfg.crop_scale),
            "use_hflip":             aug_cfg.use_hflip,
            "use_color_jitter":      aug_cfg.use_color_jitter,
            "color_jitter_strength": aug_cfg.color_jitter_strength,
            "use_grayscale":         aug_cfg.use_grayscale,
            "use_blur":              aug_cfg.use_blur,
            "epochs":                args.epochs,
            "batch_size":            args.batch_size,
            "optimizer":             args.optimizer,
            "base_lr":               base_lr,
            "weight_decay":          args.weight_decay,
            "warmup_epochs":         args.warmup_epochs,
            "amp":                   args.amp,
            "seed":                  args.seed,
        },
        tags=["simclr", aug_cfg.tag],
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    train_ds = CIFAR10(
        args.data_root, train=True, download=True,
        transform=TwoViewTransform(build_simclr_transform(aug_cfg)),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = (
        optim.Adam(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
        if args.optimizer == "adam"
        else optim.SGD(model.parameters(), lr=base_lr,
                       momentum=0.9, weight_decay=args.weight_decay)
    )
    scheduler = make_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler    = GradScaler(enabled=args.amp)

    wandb.watch(model, log="gradients", log_freq=500)

    # ── Pre-training loop ─────────────────────────────────────────────────
    log.info(f"Starting pre-training ...")
    log.info(f"{'Epoch':>6}  {'Loss':>10}  {'LR':>12}  {'Ep time':>9}  {'Elapsed':>9}  {'ETA':>9}")
    log.info("-" * 65)

    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for (x1, x2), _ in train_loader:
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(enabled=args.amp):
                z1, z2 = model(x1, x2)
                loss   = criterion(z1, z2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()

        avg_loss  = epoch_loss / len(train_loader)
        cur_lr    = scheduler.get_last_lr()[0]
        epoch_t   = time.time() - t0
        elapsed   = time.time() - total_start
        eta       = elapsed / epoch * (args.epochs - epoch)

        # ── Log every epoch to stdout + .log file ─────────────────────────
        log.info(
            f"{epoch:>6d}  {avg_loss:>10.4f}  {cur_lr:>12.2e}"
            f"  {epoch_t:>7.1f}s  {elapsed/60:>7.1f}m  {eta/60:>7.1f}m"
        )

        # ── W&B ───────────────────────────────────────────────────────────
        wandb.log({
            "train/loss":         avg_loss,
            "train/lr":           cur_lr,
            "train/epoch_time_s": epoch_t,
            "train/elapsed_min":  elapsed / 60,
            "train/eta_min":      eta / 60,
        }, step=epoch)

    total_time = time.time() - total_start
    log.info("-" * 65)
    log.info(f"Pre-training complete. Total time: {total_time/60:.1f} min")
    log.info("")

    # ── Save final checkpoint ─────────────────────────────────────────────
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = ckpt_dir / f"{args.run_name}_final.pt"
    torch.save(
        {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch":                args.epochs,
            "args":                 vars(args),
        },
        final_ckpt,
    )
    log.info(f"Final checkpoint saved: {final_ckpt}")
    log.info("")

    # ── Linear evaluation (in-memory model) ──────────────────────────────
    if not args.skip_eval:
        log.info("Starting linear evaluation ...")
        log.info(f"  Epochs: {args.eval_epochs}  LR: {args.eval_lr}")
        log.info("")

        best_acc = linear_eval(
            backbone=model.encoder,
            device=device,
            data_root=args.data_root,
            batch_size=512,
            lr=args.eval_lr,
            epochs=args.eval_epochs,
            wandb_run=run,
            logger=log,         # pass logger so eval also writes to the same .log
        )

        # ── Save plain-text eval summary ─────────────────────────────────
        summary_lines = [
            "=" * 65,
            "  Linear Evaluation Summary",
            "=" * 65,
            f"  Run name        : {args.run_name}",
            f"  Aug preset      : {args.aug_preset}  ({aug_cfg.tag})",
            f"  Total params    : {total_params:,} ({total_params / 1e6:.2f}M)",
            f"    Backbone      : {backbone_params:,} ({backbone_params / 1e6:.2f}M)",
            f"    Projector     : {projector_params:,} ({projector_params / 1e6:.2f}M)",
            f"  Pretrain epochs : {args.epochs}",
            f"  Batch size      : {args.batch_size}",
            f"  Temperature     : {args.temperature}",
            f"  Eval epochs     : {args.eval_epochs}",
            f"  Eval LR         : {args.eval_lr}",
            f"",
            f"  >>> Best Top-1 Accuracy: {best_acc:.2f}%",
            "=" * 65,
        ]
        eval_file.parent.mkdir(parents=True, exist_ok=True)
        eval_file.write_text("\n".join(summary_lines) + "\n")

        for line in summary_lines:
            log.info(line)

        wandb.summary["linear_eval/best_top1_acc"] = best_acc
        wandb.log({"linear_eval/best_top1_acc": best_acc})

    wandb.finish()
    log.info("Done.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
