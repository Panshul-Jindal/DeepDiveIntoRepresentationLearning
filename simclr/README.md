# SimCLR — CIFAR-10

Official SimCLR implementation adapted for CIFAR-10 with ResNet-18 backbone.

## Directory structure

```
simclr/
├── backbone.py          # CIFAR-adapted ResNet-18 (3×3 conv1, no maxpool, no fc)
├── augmentations.py     # All SimCLR augmentations + ablation presets + registry
├── model.py             # SimCLR model (backbone + projection head) + NT-Xent loss
├── train.py             # Pre-training script with W&B logging
├── linear_eval.py       # Linear probe evaluation (frozen backbone)
├── ablation.py          # Ablation sweep launcher
├── configs/
│   ├── simclr_base.yaml       # Full 100-epoch run
│   └── simclr_quick_test.yaml # 5-epoch smoke test
└── README.md
```

## Backbone modifications for CIFAR-10

| Layer | Standard ResNet-18 | CIFAR ResNet-18 |
|-------|-------------------|-----------------|
| `conv1` | 7×7, stride 2, pad 3 | **3×3, stride 1, pad 1** |
| `maxpool` | 3×3, stride 2 | **Identity** (removed) |
| `fc` | Linear(512, 1000) | **Identity** (removed; head attaches) |

This keeps the spatial resolution at 32×32 through the first stage instead of
aggressively downsampling it to 8×8 before the residual blocks.

## SimCLR projection head

```
h (512-d)  →  Linear(512, 2048)  →  BN  →  ReLU  →  Linear(2048, 128)
                                                                              ↓
                                                                        z (128-d)
```

Loss: **NT-Xent** with temperature τ = 0.5 (default).

## Running experiments

### 1. Install dependencies

```bash
cd /path/to/DeepDiveIntoRepresentationLearning
uv add wandb tqdm pyyaml        # add to pyproject.toml
```

### 2. Quick smoke test

```bash
cd simclr
python train.py \
    --epochs 5 \
    --batch_size 256 \
    --aug_preset full_aug \
    --eval_epochs 5 \
    --offline \
    --run_name simclr_smoke_test
```

### 3. Full base run (100 epochs)

```bash
python train.py \
    --epochs 100 \
    --batch_size 512 \
    --aug_preset full_aug \
    --run_name simclr_full_aug_bs512_ep100
```

### 4. Run ablation sweeps

```bash
# Print all commands first (dry run)
python ablation.py --dry_run

# Run augmentation ablations only
python ablation.py --axis aug

# Run batch-size ablations
python ablation.py --axis batch_size

# Run training-duration ablations
python ablation.py --axis epochs

# Run everything
python ablation.py
```

## Augmentation presets

| Tag | Description |
|-----|-------------|
| `full_aug` | Crop + Flip + ColorJitter(s=0.5) + Grayscale |
| `no_color_jitter` | Remove color jitter |
| `no_grayscale` | Remove random grayscale |
| `no_hflip` | Remove horizontal flip |
| `crop_only` | Only random resized crop |
| `weak_crop` | Narrow crop scale (0.8–1.0) |
| `strong_color_s1` | Color jitter with s=1.0 |
| `with_blur` | Add Gaussian blur (ImageNet-style) |
| `no_aug` | No augmentation (floor baseline; views identical) |
| `color_only` | ColorJitter + Grayscale only (no crop, no flip) |
| `no_crop` | Full minus random resized crop |

Ablations run 100 epochs (paper Sec. 2.3) per preset with a fixed seed (42).

## W&B project layout

All runs go under project `simclr-cifar10`. Run names follow the pattern:

```
simclr_aug-{tag}_bs{batch_size}_ep{epochs}_t{temperature}
```

Example: `simclr_aug-no_color_jitter_bs512_ep100_t0.5`

Key metrics logged:
- `train/loss` — NT-Xent loss per epoch
- `train/lr` — current learning rate
- `train/epoch_time_s` — seconds per epoch
- `linear_eval/top1_acc` — linear probe accuracy per epoch
- `linear_eval/final_top1_acc` — summary metric for comparison

## Ablation analysis guide

After all runs complete, go to the W&B project and:

1. **Augmentation axis**: Compare `linear_eval/final_top1_acc` across aug presets
   → shows which augmentations contribute most to representation quality.

2. **Batch size axis**: Plot loss curves and final accuracy vs. batch size
   → SimCLR is notoriously batch-size sensitive (more negatives = better).

3. **Training duration axis**: Plot accuracy vs. epoch checkpoint
   → shows diminishing returns and minimum epochs needed.

