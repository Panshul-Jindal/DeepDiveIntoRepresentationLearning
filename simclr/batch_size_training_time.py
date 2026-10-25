"""
SimCLR Batch Size Sweep & Training Time (Epochs) Sweep (Base LR = 1.0).

Experiments executed:
  1. Batch Size 128:  Pre-train 100 epochs (LR = 0.50) -> Linear eval (100 ep)
  2. Batch Size 256:  Pre-train 100 epochs (LR = 1.00) -> Linear eval (100 ep)
  3. Batch Size 512:  Pre-train 200 epochs (LR = 2.00) -> Linear eval at epochs 100, 150, 200
  4. Batch Size 1024: Pre-train 200 epochs (LR = 4.00) -> Linear eval at epochs 100, 150, 200

Learning Rate Scaling:
  Base LR = 1.0 (from paper's CIFAR-10 grid {0.5, 1.0, 1.5})
  Actual LR = Base LR * (Batch Size / 256.0)
    - BS 128:  LR = 0.50
    - BS 256:  LR = 1.00
    - BS 512:  LR = 2.00
    - BS 1024: LR = 4.00

Run Names:
  sweep_base_lr_1.0_bs128, sweep_base_lr_1.0_bs256,
  sweep_base_lr_1.0_bs512, sweep_base_lr_1.0_bs1024

Usage:
    # Dry run: view planned commands and scaled LRs
    python batch_size_sweep.py --dry_run

    # Run full sweep
    python batch_size_sweep.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Ensure simclr directory is on sys.path
_SIMCLR_DIR = Path(__file__).resolve().parent
if str(_SIMCLR_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMCLR_DIR))


BASE_CONFIG = {
    "base_lr":     1.0,
    "temperature": 0.5,
    "eval_epochs": 100,
    "eval_lr":     0.1,
    "aug_preset":  "full_aug",
    "seed":        42,
    "data_root":   "./data",
    "log_dir":     "./logs",
    "aug_backend": "auto",
    "offline":     False,
}

MULTI_EPOCH_CHECKPOINTS = [100, 150, 200]


def parse_eval_accuracy(eval_file: Path) -> float | None:
    """Extract final top-1 accuracy from an eval summary file."""
    if not eval_file.exists():
        return None
    for line in eval_file.read_text().splitlines():
        if "Final Top-1 Accuracy:" in line or "Best Top-1 Accuracy:" in line:
            parts = line.split(":")[-1].strip().replace("%", "")
            try:
                return float(parts)
            except ValueError:
                pass
    return None


def run_pretrain(cmd: list[str], run_name: str, log_dir: Path, dry_run: bool = False):
    """Execute pre-training subprocess."""
    print(f"\n{'='*65}")
    print(f"  Starting Pre-training: {run_name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*65}\n", flush=True)

    if dry_run:
        return

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"!! {run_name} pre-training failed with return code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"\nPre-training {run_name} completed in {elapsed/60:.1f} minutes.\n", flush=True)


def evaluate_saved_checkpoints(
    run_name: str,
    epochs_list: list[int],
    args,
    log_dir: Path,
    ckpt_dir: Path,
):
    """Evaluate checkpoints at epochs in epochs_list for a run."""
    print(f"\n{'='*65}")
    print(f"  Evaluating {run_name} checkpoints at epochs {epochs_list}")
    print(f"{'='*65}\n", flush=True)

    final_ckpt = ckpt_dir / f"{run_name}_final.pt"

    for ep in epochs_list:
        ep_ckpt = ckpt_dir / f"{run_name}_ep{ep}.pt"
        if not ep_ckpt.exists() and ep == 200 and final_ckpt.exists():
            ep_ckpt = final_ckpt

        eval_file = log_dir / f"{run_name}_ep{ep}_eval.txt"

        if eval_file.exists():
            acc = parse_eval_accuracy(eval_file)
            print(f"  [Epoch {ep:>3d}] Already evaluated: Top-1 Acc = {acc:.2f}% ({eval_file.name})")
            continue

        if args.dry_run:
            print(f"  [Epoch {ep:>3d}] Would evaluate checkpoint: {ep_ckpt.name} -> {eval_file.name}")
            continue

        if not ep_ckpt.exists():
            print(f"  !! Checkpoint for epoch {ep} not found at {ep_ckpt}. Skipping.")
            continue

        print(f"\n--- Running Linear Eval: {run_name}, Pretrained Epochs={ep} ---")
        wb_run = None
        if not args.offline:
            import wandb
            wb_run = wandb.init(
                project="simclr-cifar10",
                name=f"{run_name}_ep{ep}_eval",
                mode="online",
                config={
                    "run_name":        run_name,
                    "pretrain_epochs": ep,
                    "eval_epochs":     args.eval_epochs,
                    "eval_lr":         args.eval_lr,
                },
                tags=["simclr", "training_time_sweep", f"ep{ep}"],
            )

        from linear_eval import eval_checkpoint
        acc = eval_checkpoint(
            ckpt_path=ep_ckpt,
            data_root=args.data_root,
            eval_epochs=args.eval_epochs,
            eval_lr=args.eval_lr,
            wandb_run=wb_run,
            eval_file=eval_file,
        )

        if wb_run is not None:
            wb_run.summary["linear_eval/final_top1_acc"] = acc
            wb_run.finish()

        print(f"--- [Epoch {ep:>3d}] Finished. Top-1 Accuracy: {acc:.2f}% ---\n", flush=True)


def main():
    p = argparse.ArgumentParser(description="SimCLR Batch Size & Training Time Sweep (Base LR = 1.0)")
    p.add_argument("--base_lr", type=float, default=BASE_CONFIG["base_lr"],
                   help=f"Base LR for pre-training (scaled as base_lr * batch_size / 256, default: {BASE_CONFIG['base_lr']})")
    p.add_argument("--data_root", type=str, default=BASE_CONFIG["data_root"])
    p.add_argument("--log_dir", type=str, default=BASE_CONFIG["log_dir"])
    p.add_argument("--eval_epochs", type=int, default=BASE_CONFIG["eval_epochs"])
    p.add_argument("--eval_lr", type=float, default=BASE_CONFIG["eval_lr"])
    p.add_argument("--aug_backend", type=str, default=BASE_CONFIG["aug_backend"],
                   choices=["auto", "cpu", "gpu"])
    p.add_argument("--offline", action="store_true", default=BASE_CONFIG["offline"],
                   help="Run W&B in offline mode without network syncing")
    p.add_argument("--online", dest="offline", action="store_false",
                   help="Run W&B in online mode (default: online)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print plan without running")
    p.add_argument("--skip_pretrain", action="store_true",
                   help="Skip pre-training and only run linear evaluation on existing checkpoints")
    args = p.parse_args()

    def get_lr(bs: int) -> float:
        return args.base_lr * (bs / 256.0)

    log_dir = Path(args.log_dir)
    ckpt_dir = log_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_script = str(_SIMCLR_DIR / "train.py")
    save_epochs_str = [str(ep) for ep in MULTI_EPOCH_CHECKPOINTS]

    print("\n" + "=" * 65)
    print("  SimCLR Scaled Batch Size & Training Time Experiment Plan")
    print(f"  Base LR: {args.base_lr} (Formula: LR = BaseLR * BatchSize / 256)")
    print(f"    - BS  128 Scaled LR: {get_lr(128):.4f}")
    print(f"    - BS  256 Scaled LR: {get_lr(256):.4f}")
    print(f"    - BS  512 Scaled LR: {get_lr(512):.4f}")
    print(f"    - BS 1024 Scaled LR: {get_lr(1024):.4f}")
    print("=" * 65 + "\n")

    # -----------------------------------------------------------------------
    # Step 1: Pre-training Run: Batch Size 128 (100 epochs)
    # -----------------------------------------------------------------------
    bs128_run_name = "sweep_base_lr_1.0_bs128"
    bs128_ckpt = ckpt_dir / f"{bs128_run_name}_final.pt"
    bs128_eval_file = log_dir / f"{bs128_run_name}_eval.txt"
    bs128_lr = get_lr(128)

    bs128_cmd = [
        sys.executable, train_script,
        "--batch_size=128",
        "--epochs=100",
        f"--lr={bs128_lr:.4f}",
        "--aug_preset=full_aug",
        f"--temperature={BASE_CONFIG['temperature']}",
        f"--eval_epochs={args.eval_epochs}",
        f"--eval_lr={args.eval_lr}",
        f"--data_root={args.data_root}",
        f"--log_dir={args.log_dir}",
        f"--aug_backend={args.aug_backend}",
        f"--run_name={bs128_run_name}",
        "--skip_eval",  # Pre-training only; linear eval is run once below via eval_checkpoint
    ]
    if args.offline:
        bs128_cmd.append("--offline")
    else:
        bs128_cmd.append("--online")

    if not args.skip_pretrain:
        if bs128_eval_file.exists():
            print(f"[{bs128_run_name}] already completed ({bs128_eval_file} exists). Skipping.")
        else:
            run_pretrain(bs128_cmd, bs128_run_name, log_dir, dry_run=args.dry_run)
            if not args.dry_run and bs128_ckpt.exists():
                print(f"Running linear eval for {bs128_run_name} from {bs128_ckpt} ...")
                from linear_eval import eval_checkpoint
                eval_checkpoint(bs128_ckpt, data_root=args.data_root, eval_epochs=args.eval_epochs,
                                eval_lr=args.eval_lr, eval_file=bs128_eval_file)
    elif not bs128_eval_file.exists() and bs128_ckpt.exists() and not args.dry_run:
        print(f"Running linear eval for {bs128_run_name} from {bs128_ckpt} ...")
        from linear_eval import eval_checkpoint
        eval_checkpoint(bs128_ckpt, data_root=args.data_root, eval_epochs=args.eval_epochs,
                        eval_lr=args.eval_lr, eval_file=bs128_eval_file)

    # -----------------------------------------------------------------------
    # Step 2: Pre-training Run: Batch Size 256 (100 epochs)
    # -----------------------------------------------------------------------
    bs256_run_name = "sweep_base_lr_1.0_bs256"
    bs256_ckpt = ckpt_dir / f"{bs256_run_name}_final.pt"
    bs256_eval_file = log_dir / f"{bs256_run_name}_eval.txt"
    bs256_lr = get_lr(256)

    bs256_cmd = [
        sys.executable, train_script,
        "--batch_size=256",
        "--epochs=100",
        f"--lr={bs256_lr:.4f}",
        "--aug_preset=full_aug",
        f"--temperature={BASE_CONFIG['temperature']}",
        f"--eval_epochs={args.eval_epochs}",
        f"--eval_lr={args.eval_lr}",
        f"--data_root={args.data_root}",
        f"--log_dir={args.log_dir}",
        f"--aug_backend={args.aug_backend}",
        f"--run_name={bs256_run_name}",
        "--skip_eval",  # Pre-training only; linear eval is run once below via eval_checkpoint
    ]
    if args.offline:
        bs256_cmd.append("--offline")
    else:
        bs256_cmd.append("--online")

    if not args.skip_pretrain:
        if bs256_eval_file.exists():
            print(f"[{bs256_run_name}] already completed ({bs256_eval_file} exists). Skipping.")
        else:
            run_pretrain(bs256_cmd, bs256_run_name, log_dir, dry_run=args.dry_run)
            if not args.dry_run and bs256_ckpt.exists():
                print(f"Running linear eval for {bs256_run_name} from {bs256_ckpt} ...")
                from linear_eval import eval_checkpoint
                eval_checkpoint(bs256_ckpt, data_root=args.data_root, eval_epochs=args.eval_epochs,
                                eval_lr=args.eval_lr, eval_file=bs256_eval_file)
    elif not bs256_eval_file.exists() and bs256_ckpt.exists() and not args.dry_run:
        print(f"Running linear eval for {bs256_run_name} from {bs256_ckpt} ...")
        from linear_eval import eval_checkpoint
        eval_checkpoint(bs256_ckpt, data_root=args.data_root, eval_epochs=args.eval_epochs,
                        eval_lr=args.eval_lr, eval_file=bs256_eval_file)

    # -----------------------------------------------------------------------
    # Step 3: Pre-training Run: Batch Size 512 (200 epochs)
    # Saves checkpoints at 100, 150, 200 epochs
    # -----------------------------------------------------------------------
    bs512_run_name = "sweep_base_lr_1.0_bs512"
    bs512_final_ckpt = ckpt_dir / f"{bs512_run_name}_final.pt"
    bs512_lr = get_lr(512)

    bs512_cmd = [
        sys.executable, train_script,
        "--batch_size=512",
        "--epochs=200",
        f"--lr={bs512_lr:.4f}",
        "--aug_preset=full_aug",
        f"--temperature={BASE_CONFIG['temperature']}",
        f"--eval_epochs={args.eval_epochs}",
        f"--eval_lr={args.eval_lr}",
        f"--data_root={args.data_root}",
        f"--log_dir={args.log_dir}",
        f"--aug_backend={args.aug_backend}",
        f"--run_name={bs512_run_name}",
        "--save_epochs", *save_epochs_str,
        "--skip_eval",  # Pre-training only; evaluate_saved_checkpoints evaluates 100, 150, 200
    ]
    if args.offline:
        bs512_cmd.append("--offline")
    else:
        bs512_cmd.append("--online")

    if not args.skip_pretrain:
        if bs512_final_ckpt.exists() or (ckpt_dir / f"{bs512_run_name}_ep200.pt").exists():
            print(f"[{bs512_run_name}] 200 epochs already pre-trained. Skipping pre-training.")
        else:
            run_pretrain(bs512_cmd, bs512_run_name, log_dir, dry_run=args.dry_run)

    # Evaluate BS 512 checkpoints: 100, 150, 200
    evaluate_saved_checkpoints(bs512_run_name, MULTI_EPOCH_CHECKPOINTS, args, log_dir, ckpt_dir)

    # -----------------------------------------------------------------------
    # Step 4: Pre-training Run: Batch Size 1024 (200 epochs)
    # Saves checkpoints at 100, 150, 200 epochs
    # -----------------------------------------------------------------------
    bs1024_run_name = "sweep_base_lr_1.0_bs1024"
    bs1024_final_ckpt = ckpt_dir / f"{bs1024_run_name}_final.pt"
    bs1024_lr = get_lr(1024)

    bs1024_cmd = [
        sys.executable, train_script,
        "--batch_size=1024",
        "--epochs=200",
        f"--lr={bs1024_lr:.4f}",
        "--aug_preset=full_aug",
        f"--temperature={BASE_CONFIG['temperature']}",
        f"--eval_epochs={args.eval_epochs}",
        f"--eval_lr={args.eval_lr}",
        f"--data_root={args.data_root}",
        f"--log_dir={args.log_dir}",
        f"--aug_backend={args.aug_backend}",
        f"--run_name={bs1024_run_name}",
        "--save_epochs", *save_epochs_str,
        "--skip_eval",  # Pre-training only; evaluate_saved_checkpoints evaluates 100, 150, 200
    ]
    if args.offline:
        bs1024_cmd.append("--offline")
    else:
        bs1024_cmd.append("--online")

    if not args.skip_pretrain:
        if bs1024_final_ckpt.exists() or (ckpt_dir / f"{bs1024_run_name}_ep200.pt").exists():
            print(f"[{bs1024_run_name}] 200 epochs already pre-trained. Skipping pre-training.")
        else:
            run_pretrain(bs1024_cmd, bs1024_run_name, log_dir, dry_run=args.dry_run)

    # Evaluate BS 1024 checkpoints: 100, 150, 200
    evaluate_saved_checkpoints(bs1024_run_name, MULTI_EPOCH_CHECKPOINTS, args, log_dir, ckpt_dir)

    # -----------------------------------------------------------------------
    # Step 5: Print Summary Tables (Batch Size Sweep + Training Time Sweep)
    # -----------------------------------------------------------------------
    if args.dry_run:
        print("\nDry run completed successfully.")
        return

    # Accuracy retrieval
    bs128_acc = parse_eval_accuracy(bs128_eval_file)
    bs256_acc = parse_eval_accuracy(bs256_eval_file)
    bs512_ep100_acc = parse_eval_accuracy(log_dir / f"{bs512_run_name}_ep100_eval.txt")
    bs1024_ep100_acc = parse_eval_accuracy(log_dir / f"{bs1024_run_name}_ep100_eval.txt")

    summary_text = []
    summary_text.append("\n" + "=" * 65)
    summary_text.append(f"  EXPERIMENT RESULTS SUMMARY (Base LR = {args.base_lr:.1f}, Scaled per Batch Size)")
    summary_text.append("=" * 65)

    summary_text.append("\n1. BATCH SIZE SWEEP (Fixed at 100 Pre-training Epochs)")
    summary_text.append("-" * 65)
    summary_text.append(f"{'Batch Size':>12} | {'Pretrain Epochs':>16} | {'Scaled LR':>11} | {'Top-1 Accuracy':>15}")
    summary_text.append("-" * 65)
    summary_text.append(f"{128:>12} | {100:>16} | {get_lr(128):>11.4f} | {f'{bs128_acc:.2f}%' if bs128_acc is not None else 'N/A':>15}")
    summary_text.append(f"{256:>12} | {100:>16} | {get_lr(256):>11.4f} | {f'{bs256_acc:.2f}%' if bs256_acc is not None else 'N/A':>15}")
    summary_text.append(f"{512:>12} | {100:>16} | {get_lr(512):>11.4f} | {f'{bs512_ep100_acc:.2f}%' if bs512_ep100_acc is not None else 'N/A':>15}")
    summary_text.append(f"{1024:>12} | {100:>16} | {get_lr(1024):>11.4f} | {f'{bs1024_ep100_acc:.2f}%' if bs1024_ep100_acc is not None else 'N/A':>15}")
    summary_text.append("-" * 65)

    summary_text.append("\n2. TRAINING TIME SWEEP (Pre-training Epochs vs Batch Sizes)")
    summary_text.append("-" * 65)
    summary_text.append(f"{'Pretrain Epochs':>16} | {f'BS 512 (LR {get_lr(512):.1f})':>16} | {f'BS 1024 (LR {get_lr(1024):.1f})':>17}")
    summary_text.append("-" * 65)
    for ep in MULTI_EPOCH_CHECKPOINTS:
        f512 = log_dir / f"{bs512_run_name}_ep{ep}_eval.txt"
        f1024 = log_dir / f"{bs1024_run_name}_ep{ep}_eval.txt"
        a512 = parse_eval_accuracy(f512)
        a1024 = parse_eval_accuracy(f1024)
        s512 = f"{a512:.2f}%" if a512 is not None else "N/A"
        s1024 = f"{a1024:.2f}%" if a1024 is not None else "N/A"
        summary_text.append(f"{ep:>16} | {s512:>16} | {s1024:>17}")
    summary_text.append("-" * 65 + "\n")

    full_summary = "\n".join(summary_text)
    print(full_summary)

    # Save to distinct summary file
    summary_file = log_dir / "sweep_base_lr_1.0_summary.txt"
    summary_file.write_text(full_summary + "\n")
    print(f"Summary written to: {summary_file}")

    # Log summary tables to W&B if online
    if not args.offline:
        import wandb
        summary_run = wandb.init(
            project="simclr-cifar10",
            name="simclr_sweep_base_lr_1.0_summary",
            mode="online",
            tags=["simclr", "sweep_base_lr_1.0_summary"],
        )
        bs_table = wandb.Table(columns=["Batch Size", "Pretrain Epochs", "Scaled LR", "Top-1 Acc"])
        if bs128_acc is not None:
            bs_table.add_data(128, 100, get_lr(128), bs128_acc)
        if bs256_acc is not None:
            bs_table.add_data(256, 100, get_lr(256), bs256_acc)
        if bs512_ep100_acc is not None:
            bs_table.add_data(512, 100, get_lr(512), bs512_ep100_acc)
        if bs1024_ep100_acc is not None:
            bs_table.add_data(1024, 100, get_lr(1024), bs1024_ep100_acc)
        summary_run.log({"batch_size_sweep": bs_table})

        time_table = wandb.Table(columns=["Pretrain Epochs", f"BS 512 (LR {get_lr(512):.1f})", f"BS 1024 (LR {get_lr(1024):.1f})"])
        for ep in MULTI_EPOCH_CHECKPOINTS:
            f512 = log_dir / f"{bs512_run_name}_ep{ep}_eval.txt"
            f1024 = log_dir / f"{bs1024_run_name}_ep{ep}_eval.txt"
            a512 = parse_eval_accuracy(f512)
            a1024 = parse_eval_accuracy(f1024)
            time_table.add_data(ep, a512, a1024)
        summary_run.log({"training_time_sweep": time_table})
        summary_run.finish()


if __name__ == "__main__":
    main()
