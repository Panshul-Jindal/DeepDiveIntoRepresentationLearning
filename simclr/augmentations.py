"""
SimCLR augmentation pipeline for CIFAR-10 (32x32 images).

Official SimCLR augmentations (Chen et al. 2020, Appendix A):
  1. Random crop + resize                  (s = 1.0 for same-resolution)
  2. Random horizontal flip
  3. Color jitter  (s = strength scalar)
  4. Random grayscale
  5. Gaussian blur                         (paper uses it for ImageNet;
                                            kept here as optional since CIFAR is tiny)
  6. Normalize with CIFAR-10 channel stats

Two augmented views of each image are produced independently.

Ablation variants are exposed via factory functions so every experiment
can be logged to W&B with a descriptive augmentation tag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torchvision.transforms as T


# ---------------------------------------------------------------------------
# CIFAR-10 dataset statistics
# ---------------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# ---------------------------------------------------------------------------
# Augmentation config dataclass (drives ablations)
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """All knobs exposed for the augmentation ablation study."""

    # Crop
    crop_size: int = 32
    crop_scale: Tuple[float, float] = (0.2, 1.0)  # SimCLR default

    # Flip
    use_hflip: bool = True

    # Color jitter
    use_color_jitter: bool = True
    color_jitter_strength: float = 1.0          # SimCLR: s=1 for CIFAR
    jitter_brightness: float = 0.8              # scaled by s inside factory
    jitter_contrast: float = 0.8
    jitter_saturation: float = 0.8
    jitter_hue: float = 0.2
    jitter_prob: float = 0.8                    # probability of applying jitter

    # Grayscale
    use_grayscale: bool = True
    grayscale_prob: float = 0.2

    # Gaussian blur (optional for CIFAR; disabled by default)
    use_blur: bool = False
    blur_kernel_size: int = 3                   # must be odd; 3 is fine for 32x32
    blur_sigma: Tuple[float, float] = (0.1, 2.0)

    # Normalize
    mean: Tuple[float, float, float] = CIFAR10_MEAN
    std:  Tuple[float, float, float] = CIFAR10_STD

    # Human-readable tag for W&B
    tag: str = "full"


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_simclr_transform(cfg: AugmentationConfig) -> T.Compose:
    """
    Build a single-view SimCLR augmentation pipeline from *cfg*.
    Call it twice independently to get the two contrastive views.
    """
    s = cfg.color_jitter_strength
    transforms: List[Callable] = []

    # 1. Random resized crop
    transforms.append(
        T.RandomResizedCrop(
            size=cfg.crop_size,
            scale=cfg.crop_scale,
            interpolation=T.InterpolationMode.BILINEAR,
        )
    )

    # 2. Horizontal flip
    if cfg.use_hflip:
        transforms.append(T.RandomHorizontalFlip(p=0.5))

    # 3. Color jitter
    if cfg.use_color_jitter:
        jitter = T.ColorJitter(
            brightness=cfg.jitter_brightness * s,
            contrast=cfg.jitter_contrast * s,
            saturation=cfg.jitter_saturation * s,
            hue=cfg.jitter_hue * s,
        )
        transforms.append(T.RandomApply([jitter], p=cfg.jitter_prob))

    # 4. Random grayscale
    if cfg.use_grayscale:
        transforms.append(T.RandomGrayscale(p=cfg.grayscale_prob))

    # 5. Gaussian blur (optional)
    if cfg.use_blur:
        blur = T.GaussianBlur(
            kernel_size=cfg.blur_kernel_size,
            sigma=cfg.blur_sigma,
        )
        transforms.append(T.RandomApply([blur], p=0.5))

    # 6. To tensor + normalize
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=cfg.mean, std=cfg.std))

    return T.Compose(transforms)


# ---------------------------------------------------------------------------
# Two-view wrapper dataset transform
# ---------------------------------------------------------------------------

class TwoViewTransform:
    """
    Wraps a single-view transform and returns two independently augmented
    views of the same image.  Used as the `transform` argument to a dataset.

    Usage::
        transform = TwoViewTransform(build_simclr_transform(cfg))
        dataset   = CIFAR10(..., transform=transform)
        x1, x2   = dataset[i]          # two views
    """

    def __init__(self, base_transform: T.Compose):
        self.transform = base_transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# ---------------------------------------------------------------------------
# Validation / linear-probe transform (no augmentation beyond normalize)
# ---------------------------------------------------------------------------

def build_eval_transform(
    crop_size: int = 32,
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std:  Tuple[float, float, float] = CIFAR10_STD,
) -> T.Compose:
    """Deterministic transform for linear evaluation / test set."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Pre-built ablation configs
# ---------------------------------------------------------------------------

def get_full_augmentation() -> AugmentationConfig:
    """Full SimCLR pipeline (crop + flip + jitter + grayscale)."""
    return AugmentationConfig(tag="full_aug")


def get_no_color_jitter() -> AugmentationConfig:
    """Ablation: remove color jitter."""
    return AugmentationConfig(use_color_jitter=False, tag="no_color_jitter")


def get_no_grayscale() -> AugmentationConfig:
    """Ablation: remove random grayscale."""
    return AugmentationConfig(use_grayscale=False, tag="no_grayscale")


def get_no_flip() -> AugmentationConfig:
    """Ablation: remove horizontal flip."""
    return AugmentationConfig(use_hflip=False, tag="no_hflip")


def get_crop_only() -> AugmentationConfig:
    """Ablation: crop only (no color, no grayscale, no flip)."""
    return AugmentationConfig(
        use_hflip=False,
        use_color_jitter=False,
        use_grayscale=False,
        tag="crop_only",
    )


def get_weak_crop() -> AugmentationConfig:
    """Ablation: narrow crop scale range (less aggressive)."""
    return AugmentationConfig(crop_scale=(0.8, 1.0), tag="weak_crop")


def get_with_blur() -> AugmentationConfig:
    """Variant: add Gaussian blur (ImageNet-style)."""
    return AugmentationConfig(use_blur=True, tag="with_blur")


def get_weak_color() -> AugmentationConfig:
    """Ablation: halved color jitter strength (s=0.5)."""
    return AugmentationConfig(color_jitter_strength=0.5, tag="weak_color_s05")


# Lookup table used by train.py / configs
AUGMENTATION_REGISTRY: dict[str, Callable[[], AugmentationConfig]] = {
    "full_aug":          get_full_augmentation,
    "no_color_jitter":   get_no_color_jitter,
    "no_grayscale":      get_no_grayscale,
    "no_hflip":          get_no_flip,
    "crop_only":         get_crop_only,
    "weak_crop":         get_weak_crop,
    "with_blur":         get_with_blur,
    "weak_color_s05":    get_weak_color,
}


def get_augmentation_config(name: str) -> AugmentationConfig:
    """Retrieve an AugmentationConfig by registry name."""
    if name not in AUGMENTATION_REGISTRY:
        raise ValueError(
            f"Unknown augmentation preset '{name}'. "
            f"Available: {list(AUGMENTATION_REGISTRY)}"
        )
    return AUGMENTATION_REGISTRY[name]()
