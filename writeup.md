# Deep Dive into Representation Learning: SimCLR on CIFAR-10

## 1. Executive Summary

Key thing is for linear evaluation we dont use augmentations, takes enough amount of time, so removed, wont affect much of performance.

---

## 2. Core Architecture & Engineering Implementation

### 2.1 CIFAR-Adapted ResNet-18 Backbone
We remove the first MAxPool layer and also change the 7*7 conv's to 3*3.

```python
# From simclr/backbone.py
def build_cifar_resnet18(zero_init_residual: bool = False) -> nn.Module:
    backbone = models.resnet18(weights=None, zero_init_residual=zero_init_residual)
    # 1. Replace first conv: 7x7 s2 p3 -> 3x3 s1 p1 (preserves 32x32 resolution)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # 2. Remove max-pool (avoids premature downsampling 32 -> 16)
    backbone.maxpool = nn.Identity()
    # 3. Strip classification head (outputs raw 512-d pooled features)
    backbone.fc = nn.Identity()
    return backbone
```

### 2.2 SimCLR Projection Head
The projection head

```python
# From simclr/model.py
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),  # No BN/ReLU on final output
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)
```

### 2.3 Vectorized NT-Xent Loss
Given batch size $N$, we generate two views per sample ($2N$ total). For each positive pair $(z_i, z_j)$, the other $2(N-1)$ samples serve as negative pairs.

The Normalized Temperature-scaled Cross Entropy (NT-Xent) loss is:
$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \ne i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

We implement this fully vectorized on GPU by computing the $(2N \times 2N)$ cosine similarity matrix, masking the diagonal ($-\infty$), and evaluating cross-entropy against aligned target indices:

```python
# From simclr/model.py
class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.size(0)
        device = z1.device

        # L2-normalise embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)  # (2N, D)

        # Pairwise cosine similarity scaled by temperature
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Mask self-similarity (diagonal)
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, float("-inf"))

        # Ground truth positive targets: view1[i] <-> view2[i]
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N,     device=device),
        ])

        return F.cross_entropy(sim, labels)
```

### 2.4 High-Performance GPU Data Augmentations with Kornia
Standard `torchvision.transforms` runs on CPU per-image within `DataLoader` workers. On cloud GPU environments (e.g., Kaggle, Colab) with restricted CPU allocations (2–4 vCPUs), decoding and augmenting 50,000 images per epoch creates a severe CPU bottleneck, leaving the GPU idle. So using Kornia helps with this, this does transformations fast as it makes them similar to torch layers, and can be applied to batch aswell!. (Didnt go through it fully).

About Kornia:
* Random parameters are sampled independently per image in the batch across GPU threads.
* Generates two stochastic views $(x_1, x_2)$ directly on CUDA in ~3 milliseconds per batch.
* Reduced pre-training epoch duration from **~55 seconds to ~14.8 seconds** (~3.7x speedup).

```python
# From simclr/augmentations.py
def build_simclr_transform_gpu(cfg: AugmentationConfig):
    import kornia.augmentation as K
    s = cfg.color_jitter_strength
    transforms = []

    if cfg.use_crop:
        transforms.append(K.RandomResizedCrop(
            size=(cfg.crop_size, cfg.crop_size),
            scale=cfg.crop_scale,
            cropping_mode="resample",  # vectorized grid_sample across batch
        ))
    if cfg.use_hflip:
        transforms.append(K.RandomHorizontalFlip(p=0.5))
    if cfg.use_color_jitter:
        transforms.append(K.ColorJitter(
            brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s, hue=0.2 * s,
            p=cfg.jitter_prob,
        ))
    if cfg.use_grayscale:
        transforms.append(K.RandomGrayscale(p=cfg.grayscale_prob))
    transforms.append(K.Normalize(mean=torch.tensor(cfg.mean), std=torch.tensor(cfg.std)))

    return nn.Sequential(*transforms)
```

### 2.7 High-Speed Cached Linear Evaluation Protocol
In standard linear evaluation, the pre-trained backbone is frozen, and a linear classifier (`Linear(512, 10)`) is trained for 100 epochs on top of the representations $h$.
* Instead of running forward passes through the frozen ResNet backbone on every epoch, our pipeline extracts and caches the deterministic representation features in GPU VRAM once (~2 seconds).
* The linear classifier is then trained directly on the cached GPU embeddings in ~10 seconds.
* This achieves consistent, deterministic linear probe evaluation with a 40x speedup.

---

## 3. Experiment 1: Data Augmentation Ablations

### 3.1 Experimental Configuration
* **Dataset:** CIFAR-10 (50,000 train, 10,000 test)
* **Epochs:** 100
* **Batch Size:** 512
* **Base LR:** 0.3 (Scaled to $\text{LR} = 0.60$ for BS 512)
* **Temperature:** $\tau = 0.5$
* **Optimizer:** LARS (momentum 0.9, weight decay $10^{-6}$)

### 3.2 Empirical Results

| Augmentation Preset | Transformations Included | Top-1 Linear Eval Accuracy | $\Delta$ vs. Full Aug |
| :--- | :--- | :---: | :---: |
| **`full_aug`** | Random Crop + Flip + Color Jitter ($s=0.5$) + Grayscale | **78.09%** | **Baseline** |
| **`no_color_jitter`** | Crop + Flip + Grayscale (No Color Jitter) | **73.18%** | -4.91% |
| **`weak_crop`** | Mild Crop (scale $0.8 \dots 1.0$) + Flip + Color Jitter + Grayscale | **72.62%** | -5.47% |
| **`color_only`** | Color Jitter + Grayscale (No Spatial Crop, No Flip) | **59.93%** | -18.16% |
| **`crop_only`** | Random Crop Only (No Flip, No Color Jitter, No Grayscale) | **59.65%** | -18.44% |
| **`no_aug`** | None (Identical normalized views) | **41.76%** | -36.33% |

### 3.3 Key Findings & Analysis
1. **The Compositionality Principle (Crop + Color Jitter):**
   * Neither spatial cropping alone (**59.65%**) nor color distortion alone (**59.93%**) is sufficient for representation learning.
   * Combining spatial cropping with color jitter yields **78.09%**—a dramatic **+18.4% leap**.
   * *Mechanism:* When random crops are taken from the same image without color distortion, the two cropped patches share nearly identical color histograms. The network easily solves the contrastive task through the "color shortcut" (matching color histograms) without learning high-level semantic shapes. Adding strong color jitter breaks this color shortcut, forcing the network to extract invariant structural features.
2. **Aggressive vs. Weak Cropping:**
   * Restricting the crop scale to $[0.8, 1.0]$ (`weak_crop`) degraded performance by **-5.47%** (72.62% vs 78.09%). Contrastive learning heavily relies on small, aggressive crops ($[0.08, 1.0]$) to force the model to identify global context from local parts.
3. **Failure of Identity Views (`no_aug`):**
   * Without data augmentation, the two views are identical. The network achieves only **41.76%** top-1 accuracy, demonstrating that contrastive learning collapses into trivial representation spaces without stochastic view perturbation.

---

## 4. Experiment 2: Batch Size & Training Time Sweep (Constant Learning Rate)

### 4.1 Experimental Configuration
* **Batch Sizes:** $128, 256, 512, 1024$
* **Pre-training Epochs:** 100 epochs (all batch sizes); 150 & 200 epochs for BS 512 and BS 1024.
* **Learning Rate:** Fixed at $\text{LR} = 1.0$ across all batch sizes (no linear scaling).
* **Augmentation:** `full_aug`
* **Temperature:** $\tau = 0.5$

### 4.2 Empirical Results

#### Part A: Batch Size Sweep (Fixed at 100 Epochs)

| Batch Size | Pre-train Epochs | Learning Rate | Top-1 Linear Eval Accuracy | Total Updates |
| :---: | :---: | :---: | :---: | :---: |
| **128** | 100 | 1.00 | **82.37%** | 39,062 |
| **256** | 100 | 1.00 | **81.68%** | 19,531 |
| **512** | 100 | 1.00 | **81.67%** | 9,765 |
| **1024** | 100 | 1.00 | **80.76%** | 4,882 |

#### Part B: Training Time Sweep (BS 512 vs. BS 1024 Across Epochs)

| Pre-train Epochs | Batch Size 512 Top-1 Acc | Batch Size 1024 Top-1 Acc |
| :---: | :---: | :---: |
| **100** | 81.67% | 80.76% |
| **150** | 83.04% | 81.57% |
| **200** | **83.27%** | **82.11%** |

### 4.3 Key Findings & Analysis
1. **Constant LR Failure at Large Batch Sizes:**
   * Holding the learning rate constant while increasing batch size caused performance at 100 epochs to decline from **82.37%** (BS 128) down to **80.76%** (BS 1024).
   * At BS 1024, an epoch contains only 48 iterations. Across 100 epochs, the model only took 4,882 optimizer steps. With a fixed small learning rate, the model traveled too small a distance in parameter space to converge.
2. **Benefit of Extended Training:**
   * Increasing training time from 100 to 200 epochs partially compensated for the step deficit:
     * BS 512 improved from **81.67% $\rightarrow$ 83.27%** (+1.60%).
     * BS 1024 improved from **80.76% $\rightarrow$ 82.11%** (+1.35%).
   * However, neither caught up to its full potential due to under-scaling the learning rate.

---

## 5. Experiment 3: Batch Size & Training Time Sweep (Scaled LR, Base LR = 2.0)

### 5.1 Experimental Configuration
To fix the under-optimization observed in Experiment 2, we applied linear learning rate scaling per Goyal et al. (2017) and Chen et al. (2020):
$$\text{LR} = \text{Base LR} \times \frac{\text{Batch Size}}{256}$$
With $\text{Base LR} = 2.0$:
* **BS 128:** $\text{LR} = 2.0 \times \frac{128}{256} = \mathbf{1.00}$
* **BS 256:** $\text{LR} = 2.0 \times \frac{256}{256} = \mathbf{2.00}$
* **BS 512:** $\text{LR} = 2.0 \times \frac{512}{256} = \mathbf{4.00}$
* **BS 1024:** $\text{LR} = 2.0 \times \frac{1024}{256} = \mathbf{8.00}$

### 5.2 Empirical Results

#### Part A: Scaled Batch Size Sweep (Fixed at 100 Epochs)

| Batch Size | Pre-train Epochs | Scaled LR | Top-1 Linear Eval Accuracy | Total Updates |
| :---: | :---: | :---: | :---: | :---: |
| **128** | 100 | 1.00 | **82.56%** | 39,062 |
| **256** | 100 | 2.00 | **83.06%** | 19,531 |
| **512** | 100 | 4.00 | **82.98%** | 9,765 |
| **1024** | 100 | 8.00 | **82.24%** | 4,882 |

#### Part B: Scaled Training Time Sweep (BS 512 vs. BS 1024 Across Epochs)

| Pre-train Epochs | BS 512 (LR = 4.00) Top-1 Acc | BS 1024 (LR = 8.00) Top-1 Acc |
| :---: | :---: | :---: |
| **100** | 82.98% | 82.24% |
| **150** | 85.30% (+2.32%) | 84.92% (+2.68%) |
| **200** | **85.99%** (+3.01%) | **85.92%** (+3.68%) |

### 5.3 Comparative Analysis: Constant LR vs. Scaled LR

```
  Top-1 Accuracy (%) vs. Training Epochs (BS 512 & BS 1024)

  86.0% ─────────────────────────────────────────── ★ 85.99% (BS 512, Scaled LR)
                                                    ★ 85.92% (BS 1024, Scaled LR)
  85.0% ────────────────────────── ● 85.30%
                                   ● 84.92%
  84.0% ─────────────────────────────────────────── ○ 83.27% (BS 512, Constant LR)
  83.0% ────────── ■ 82.98%
                   ■ 82.24% ────── ○ 83.04% ─────── ○ 82.11% (BS 1024, Constant LR)
  82.0% ────────── △ 81.67%
  81.0% ────────── △ 80.76% ────── ○ 81.57%
          100 Epochs              150 Epochs              200 Epochs
```

1. **Massive Gains from Extended Training Under Scaled LR:**
   * Under scaled LR, moving from 100 to 200 epochs yielded substantial performance surges:
     * **BS 512:** $82.98\% \rightarrow \mathbf{85.99\%}$ (**+3.01%**)
     * **BS 1024:** $82.24\% \rightarrow \mathbf{85.92\%}$ (**+3.68%**)
   * In contrast, with constant LR, training from 100 to 200 epochs only yielded $+1.6\%$ and $+1.35\%$.
2. **Why Batch Size Advantage Requires Longer Training:**
   * At 100 epochs, BS 1024 (82.24%) was slightly below BS 256 (83.06%).
   * *Reason:* Contrastive learning requires sufficient gradient updates to distribute representations uniformly across the unit hypersphere. At 100 epochs, BS 1024 received only **4,882 steps** compared to **39,062 steps** for BS 128 (8x fewer steps).
   * Once training was extended to 200 epochs (9,765 steps for BS 1024), the model converged, jumping to **85.92%**, demonstrating that large batch sizes require extended epoch counts to unleash their performance benefits.
3. **Stability of LARS at High Learning Rates:**
   * Despite scaling the learning rate up to **$\text{LR} = 8.00$** for BS 1024, training remained completely stable without loss divergence. This confirms the efficacy of the layer-wise trust ratio in LARS preventing gradient explosion.

---

## 6. Experiment 4: Batch Size & Training Time Sweep (Scaled LR, Base LR = 1.0)

### 6.1 Experimental Configuration
Following the paper's CIFAR-10 grid search recommendation ($\text{Base LR} \in \{0.5, 1.0, 1.5\}$), we evaluated linear learning rate scaling with a more conservative base learning rate $\text{Base LR} = 1.0$:
$$\text{LR} = \text{Base LR} \times \frac{\text{Batch Size}}{256}$$
* **BS 128:** $\text{LR} = 1.0 \times \frac{128}{256} = \mathbf{0.50}$
* **BS 256:** $\text{LR} = 1.0 \times \frac{256}{256} = \mathbf{1.00}$
* **BS 512:** $\text{LR} = 1.0 \times \frac{512}{256} = \mathbf{2.00}$
* **BS 1024:** $\text{LR} = 1.0 \times \frac{1024}{256} = \mathbf{4.00}$

### 6.2 Empirical Results

#### Part A: Scaled Batch Size Sweep (Fixed at 100 Epochs)

| Batch Size | Pre-train Epochs | Scaled LR | Top-1 Linear Eval Accuracy | Total Updates |
| :---: | :---: | :---: | :---: | :---: |
| **128** | 100 | 0.50 | **81.52%** | 39,062 |
| **256** | 100 | 1.00 | **81.68%** | 19,531 |
| **512** | 100 | 2.00 | **83.01%** | 9,765 |
| **1024** | 100 | 4.00 | **83.50%** | 4,882 |

#### Part B: Scaled Training Time Sweep (BS 512 vs. BS 1024 Across Epochs)

| Pre-train Epochs | BS 512 (LR = 2.00) Top-1 Acc | BS 1024 (LR = 4.00) Top-1 Acc |
| :---: | :---: | :---: |
| **100** | 83.01% | 83.50% |
| **150** | 84.52% (+1.51%) | 84.83% (+1.33%) |
| **200** | **85.15%** (+2.14%) | **85.29%** (+1.79%) |

### 6.3 Key Findings & Analysis

1. **Strictly Monotonic Accuracy Scaling with Batch Size:**
   * Unlike Base LR = 2.0 (where BS 1024 ran at an aggressive $\text{LR} = 8.00$ and scored 82.24% at 100 epochs), Base LR = 1.0 exhibits **strictly monotonic performance gains as batch size increases**, even at 100 epochs:
     $$\mathbf{81.52\%} \text{ (BS 128)} < \mathbf{81.68\%} \text{ (BS 256)} < \mathbf{83.01\%} \text{ (BS 512)} < \mathbf{83.50\%} \text{ (BS 1024)}$$
   * At $\text{LR} = 4.00$, BS 1024 enjoys both sufficient learning rate and more negative pairs per step without initial optimization instability.
2. **BS 1024 Dominates at Every Training Horizon:**
   * At all epoch milestones (100, 150, 200 epochs), **BS 1024 consistently outperforms BS 512**:
     * At 100 ep: **83.50%** vs. 83.01% (+0.49%)
     * At 150 ep: **84.83%** vs. 84.52% (+0.31%)
     * At 200 ep: **85.29%** vs. 85.15% (+0.14%)
3. **Comparison between Base LR = 1.0 vs. Base LR = 2.0:**
   * **At 100 Epochs:** Base LR = 1.0 is superior for BS 1024 (**83.50%** vs. 82.24%) because $\text{LR} = 4.00$ provides smoother early-stage convergence than $\text{LR} = 8.00$.
   * **At 200 Epochs:** Base LR = 2.0 reaches a slightly higher absolute peak (**85.99%** / **85.92%** vs. 85.15% / 85.29%). With longer training schedules, a higher initial learning rate explores parameter space more broadly before cosine annealing converges to zero.

---

## 7. Synthesis and Conclusion

| Hypothesis / Question | Finding | Empirical Evidence |
| :--- | :--- | :--- |
| **Is one augmentation enough?** | **No.** Spatial crop and color distortion must be combined to break color shortcuts. | `crop_only` (59.65%) + `color_only` (59.93%) $\rightarrow$ `full_aug` (**78.09%**). |
| **Can we keep LR constant when scaling batch size?** | **No.** Fixed LR causes step starvation and accuracy degradation at large batch sizes. | Constant LR drops from 82.37% (BS 128) to 80.76% (BS 1024). |
| **Does linear LR scaling fix large batch pre-training?** | **Yes.** Scaled LR allows models to achieve higher accuracy ceilings. | BS 512 reached **85.99%**; BS 1024 reached **85.92%** at 200 epochs. |
| **Does Base LR = 1.0 scale monotonically?** | **Yes.** Moderate scaled LR produces clean monotonic improvements across all batch sizes. | BS 128 (81.52%) $\rightarrow$ BS 256 (81.68%) $\rightarrow$ BS 512 (83.01%) $\rightarrow$ BS 1024 (**83.50%**). |

