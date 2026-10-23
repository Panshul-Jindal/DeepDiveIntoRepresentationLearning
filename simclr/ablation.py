"""
SimCLR augmentation ablation sweep for CIFAR-10.

Runs one experiment per augmentation preset. All other hyperparameters
(batch size, epochs, temperature, seed) are fixed at the base config.
Epochs default to 100, matching the paper's augmentation ablations (Sec. 2.3).

Each W&B run name is:  simclr_{aug_tag}
e.g.  simclr_full_aug
      simclr_color_only

Usage:
    # Dry-run: print all commands without running
    python ablation.py --dry_run

    # Run all augmentation ablations sequentially
    python ablation.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixed base config (held constant across all ablations)
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "epochs":      100,
    "batch_size":  512,
    "temperature": 0.5,
    "eval_epochs": 100,
    "seed":        42,
    "data_root":   "./data",
    "log_dir":     "./logs",
}

# ---------------------------------------------------------------------------
# Augmentation presets to sweep (tag → aug_preset arg)
# ---------------------------------------------------------------------------
# Tags are intentionally descriptive of what is present/absent so W&B
# run names are self-explanatory at a glance.

AUG_PRESETS = [
    "full_aug",           # baseline: crop + flip + color jitter (s=0.5) + grayscale
    "no_color_jitter",    # remove color jitter
    "no_grayscale",       # remove random grayscale
    "no_hflip",           # remove horizontal flip
    "crop_only",          # only random resized crop, nothing else
    "weak_crop",          # narrow crop scale (0.8–1.0) instead of (0.08–1.0)
    "strong_color_s1",    # color jitter with doubled strength (s=1.0)
    "with_blur",          # add Gaussian blur (ImageNet-style variant)
    # composition (paper Fig. 5): crop vs color alone and together
    "no_aug",             # floor: no augmentation, identical views
    "color_only",         # jitter + grayscale, no crop, no flip
    "no_crop",            # full minus crop
]


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_command(aug_preset: str, cfg: dict) -> list[str]:
    """Build the train.py command for a single augmentation preset."""
    run_name = f"simclr_{aug_preset}"   # e.g. simclr_no_color_jitter

    return [
        sys.executable,
        str(Path(__file__).parent / "train.py"),
        f"--aug_preset={aug_preset}",
        f"--batch_size={cfg['batch_size']}",
        f"--epochs={cfg['epochs']}",
        f"--temperature={cfg['temperature']}",
        f"--eval_epochs={cfg['eval_epochs']}",
        f"--data_root={cfg['data_root']}",
        f"--log_dir={cfg['log_dir']}",
        f"--seed={cfg['seed']}",
        f"--run_name={run_name}",
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(cfg: dict, dry_run: bool = False):
    print(f"\n{'─'*60}")
    print(f"SimCLR augmentation ablation — {len(AUG_PRESETS)} runs")
    print(f"Fixed: bs={cfg['batch_size']}, "
          f"ep={cfg['epochs']}, "
          f"τ={cfg['temperature']}, "
          f"seed={cfg['seed']}")
    print(f"{'─'*60}\n")

    for i, preset in enumerate(AUG_PRESETS, 1):
        cmd = build_command(preset, cfg)
        run_name = f"simclr_{preset}"
        print(f"[{i}/{len(AUG_PRESETS)}]  run: {run_name}")
        print(f"           cmd: {' '.join(cmd)}\n")

        if not dry_run:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  !! Run failed with code {result.returncode}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="SimCLR augmentation ablation sweep launcher"
    )
    p.add_argument("--epochs", type=int, default=BASE_CONFIG["epochs"],
                   help=f"Pre-training epochs per run (default: {BASE_CONFIG['epochs']})")
    p.add_argument("--batch_size", type=int, default=BASE_CONFIG["batch_size"],
                   help=f"Batch size per run (default: {BASE_CONFIG['batch_size']})")
    p.add_argument("--eval_epochs", type=int, default=BASE_CONFIG["eval_epochs"],
                   help=f"Linear eval epochs per run (default: {BASE_CONFIG['eval_epochs']})")
    p.add_argument("--data_root", type=str, default=BASE_CONFIG["data_root"])
    p.add_argument("--log_dir", type=str, default=BASE_CONFIG["log_dir"])
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = p.parse_args()

    cfg = {
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "temperature": BASE_CONFIG["temperature"],
        "eval_epochs": args.eval_epochs,
        "seed":        BASE_CONFIG["seed"],
        "data_root":   args.data_root,
        "log_dir":     args.log_dir,
    }

    run_all(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
