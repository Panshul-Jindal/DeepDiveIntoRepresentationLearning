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
import time
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
        f"--aug_backend={cfg['aug_backend']}",
        f"--run_name={run_name}",
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(cfg: dict, presets: list[str], parallel: int = 1, dry_run: bool = False):
    print(f"\n{'─'*60}")
    print(f"SimCLR augmentation ablation — {len(presets)} runs")
    print(f"Fixed: bs={cfg['batch_size']}, "
          f"ep={cfg['epochs']}, "
          f"τ={cfg['temperature']}, "
          f"seed={cfg['seed']}, parallel={parallel}")
    print(f"{'─'*60}\n")

    pending = []

    for i, preset in enumerate(presets, 1):
        cmd = build_command(preset, cfg)
        run_name = f"simclr_{preset}"
        print(f"[{i}/{len(presets)}]  run: {run_name}")

        # train.py writes {run_name}_eval.txt only after a run fully finishes
        if (Path(cfg["log_dir"]) / f"{run_name}_eval.txt").exists():
            print(f"           skip: already finished ({run_name}_eval.txt exists)\n")
            continue
        print(f"           cmd: {' '.join(cmd)}\n")
        pending.append((run_name, cmd))

    if dry_run or not pending:
        return

    # Download once up front so concurrent runs don't race on the same files
    from augmentations import CIFAR10
    CIFAR10(cfg["data_root"], train=True, download=True)
    CIFAR10(cfg["data_root"], train=False, download=True)

    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    running: list[tuple[str, subprocess.Popen]] = []

    def reap():
        """Drop finished runs from `running` and report their status."""
        for run_name, proc in list(running):
            if proc.poll() is not None:
                running.remove((run_name, proc))
                status = "done" if proc.returncode == 0 else f"!! failed (code {proc.returncode})"
                print(f"  {run_name}: {status}", flush=True)

    for run_name, cmd in pending:
        reap()
        while len(running) >= parallel:
            time.sleep(10)
            reap()
        if parallel == 1:
            stdout = None   # stream live to the terminal
        else:
            # Interleaved output would be unreadable; each run gets its own file
            stdout = open(log_dir / f"{run_name}.out", "w")
            print(f"  {run_name}: started (output → {log_dir / (run_name + '.out')})", flush=True)
        running.append((run_name, subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT)))

    while running:
        time.sleep(10)
        reap()


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
    p.add_argument("--presets", type=str, nargs="+", default=AUG_PRESETS,
                   choices=AUG_PRESETS,
                   help="Subset of presets to run (default: all)")
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of runs to train concurrently on the same GPU (default: 1)")
    p.add_argument("--aug_backend", type=str, default="auto",
                   choices=["auto", "cpu", "gpu"],
                   help="cpu = torchvision, gpu = Kornia, auto = by CPU count (default)")
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
        "aug_backend": args.aug_backend,
        "data_root":   args.data_root,
        "log_dir":     args.log_dir,
    }

    run_all(cfg, args.presets, parallel=args.parallel, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
