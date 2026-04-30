# CS-InfoBatch: Class-Stratified Data Pruning for Dead Tree Segmentation

Built on top of the [TreeFinder](https://github.com/zhwang0/treefinder) baseline (NeurIPS 2025 Datasets & Benchmarks). This project investigates whether intelligent training data pruning can reduce compute cost while maintaining segmentation quality for aerial tree detection.

The key contribution is **CS-InfoBatch (Class-Stratified InfoBatch)**, a pruning strategy that extends vanilla InfoBatch with per-class difficulty thresholds and a class-balanced loss signal, designed for the severe class imbalance present in tree canopy segmentation (~0.37% foreground pixels).

---

## What This Project Does

Standard training uses every sample every epoch. Data pruning removes "easy" samples (those the model has already learned) to reduce total training time. This project benchmarks three pruning strategies against a full-data baseline across two model architectures:

| Strategy | Description |
|----------|-------------|
| **CS-InfoBatch** | Our method. Per-class pruning thresholds + EMA scoring + class-balanced loss weighting |
| **InfoBatch** | Vanilla adaptive pruning with a global difficulty threshold |
| **Random Pruning** | Fixed random subset, chosen once before training |
| **Baseline** | Full dataset, no pruning |

Each pruning strategy is evaluated at five pruning ratios: 0.3, 0.5, 0.7, 0.8, 0.9 (fraction of easy samples removed each epoch).

---

## Models

### Mask2Former (Swin-Tiny backbone)
Loaded from `facebook/mask2former-swin-tiny-ade-semantic`. The Swin patch embedding is replaced to accept 5 input channels (RGB + NIR + NDVI). The classifier head is replaced for binary (tree/background) segmentation.

### DeepLabV3+ (ResNet50 backbone)
Loaded from torchvision with ImageNet pretrained weights. The first Conv2d is replaced to accept 5 input channels; extra channel weights are initialized as the mean of the pretrained RGB weights (following the treefinder baseline). The auxiliary classifier is disabled.

Both models receive the same 5-band input `(B, 5, 224, 224)` and produce `(B, 1, 224, 224)` logits. The training loop, pruning strategies, and evaluation code are fully model-agnostic.

---

## Input Data

5-band 224×224 NAIP GeoTIFF tiles from the [TreeFinder dataset](https://www.kaggle.com/datasets/zhihaow/tree-finder):

| Band | Content |
|------|---------|
| 1 | Red (R) |
| 2 | Green (G) |
| 3 | Blue (B) |
| 4 | Near-Infrared (NIR) |
| 5 | Label mask (0=background, 1=tree, 255=no-data) |

NDVI is computed on-the-fly: `NDVI = (NIR − R) / (NIR + R + ε)`. The 5 input channels passed to the model are: R, G, B, NIR, NDVI.

---

## Repository Structure

```
CS-Infobatch/
├── main.py                        # Entry point: train + evaluate one experiment
│
├── configs/                       # YAML config files (one per model × pruner combination)
│   ├── base.yaml                  # Shared defaults
│   ├── baseline.yaml              # Mask2Former, no pruning
│   ├── cs_infobatch.yaml          # Mask2Former + CS-InfoBatch
│   ├── infobatch.yaml             # Mask2Former + InfoBatch
│   ├── random_prune.yaml          # Mask2Former + random pruning
│   ├── baseline_deeplabv3.yaml    # DeepLabV3+, no pruning
│   ├── cs_infobatch_deeplabv3.yaml
│   ├── infobatch_deeplabv3.yaml
│   └── random_prune_deeplabv3.yaml
│
├── models/
│   ├── mask2former.py             # Mask2Former wrapper (5-channel input)
│   └── deeplabv3.py               # DeepLabV3+ wrapper (5-channel input)
│
├── pruning/
│   ├── cs_infobatch_pruner.py     # CS-InfoBatch: class-stratified + EMA scoring
│   ├── infobatch_pruner.py        # Vanilla InfoBatch
│   ├── random_pruner.py           # Fixed random subset
│   └── no_pruner.py               # Baseline (no pruning)
│
├── training/
│   ├── trainer.py                 # Main training loop (model-agnostic)
│   ├── losses.py                  # per_sample_bce, per_sample_cs_score, DiceLoss
│   └── metrics.py                 # val_class1_f1 for early stopping
│
├── data/
│   ├── random_split.py            # TileDataset (loads GeoTIFFs, computes NDVI)
│   ├── dataloader.py              # DataLoader factory
│   └── utils.py                   # load_tile, augment_tile
│
├── evaluation/
│   └── evaluator.py               # Test-set evaluation: IoU, F1, precision, recall
│
├── evaluate/                      # Post-training analysis scripts
│   ├── generate_probs.py          # Inference pass → saves per-pixel .npz prob files
│   ├── analyze_results.py         # Generates all figures and LaTeX tables
│   └── compare_models.py          # Side-by-side Mask2Former vs DeepLabV3+ comparison
│
├── utils/
│   ├── split_cache.py             # Dataset split generation and caching
│   ├── tools.py                   # Config loading, arg parsing, logging, seeding
│   └── gdrive_upload.py           # Auto-upload results to Google Drive
│
└── scripts/
    └── setup_gdrive.py            # One-time Google Drive OAuth setup
```

---

## CS-InfoBatch: How It Works

CS-InfoBatch addresses two weaknesses of vanilla InfoBatch when applied to class-imbalanced segmentation:

**Problem 1: Global threshold:** InfoBatch uses a single difficulty threshold across all samples. For tree segmentation, negative tiles (no trees) are inherently easier than positive tiles. A global threshold prunes too many positive tiles, including the rare, hard, informative ones.

**Solution:** Compute separate difficulty thresholds for positive tiles (tiles containing tree pixels) and negative tiles. Pruning decisions are made within each class independently.

**Problem 2: Loss signal ignores class imbalance:** Standard BCE weights every pixel equally. With 0.37% foreground pixels, the loss is dominated by background; easy negative tiles look just as "hard" as positive ones.

**Solution:** Replace the per-sample BCE score with a class-balanced score:

```
H(z_i) = w_fg × L_fg(z_i) × has_fg(z_i) + w_bg × L_bg(z_i)
```

where `w_fg ≈ 134` and `w_bg ≈ 0.5` are the inverse class frequencies. This makes foreground loss dominate, so positive tiles with unlearned tree pixels always score as hard.

**Training Phases:**

| Phase | Epochs | Behaviour |
|-------|--------|-----------|
| Warmup | 0 → 10% | Full dataset, no pruning. Scores initialise. |
| Pruning | 10% → 87.5% | Ramped pruning. Rate increases linearly from 0 to `prune_ratio`. |
| Annealing | 87.5% → end | Full dataset again. Weights reset to 1.0. Model converges cleanly. |

Scores are updated via EMA: `score = 0.7 × score + 0.3 × new_score`.

---

## Experiment Naming Convention

Experiment names are auto-generated by `main.py` as:

- `baseline_{id}` for no-pruning runs
- `{pruner_type}_r{prune_ratio}_{id}` for pruning runs

| Config | Experiment Name |
|--------|----------------|
| `baseline.yaml` | `baseline_baseline` |
| `cs_infobatch.yaml --prune_ratio 0.5` | `cs_infobatch_r0.5_cs_infobatch` |
| `baseline_deeplabv3.yaml` | `baseline_deeplabv3` |
| `infobatch_deeplabv3.yaml --prune_ratio 0.7` | `infobatch_r0.7_deeplabv3` |

---

## Loss Function

Total loss per batch:

```
L_total = L_BCE + w_dice × L_Dice
```

- `L_BCE`: per-sample BCE averaged over valid (non-no-data) pixels, then weighted by the pruner and averaged over the batch
- `L_Dice`: batch-level Dice loss: `1 − (2·TP + ε) / (2·TP + FP + FN + ε)`
- `w_dice = 1.0` by default

The CS-InfoBatch score `H(z_i)` is used **only** for pruning decisions, never for gradient computation.

---

## Hyperparameters

| Parameter | Mask2Former | DeepLabV3+ | Reason |
|-----------|-------------|------------|--------|
| Learning rate | 1e-5 | 1e-4 | Transformer vs CNN backbone |
| Optimizer | AdamW | AdamW | |
| Weight decay | 0.01 | 0.01 | |
| Scheduler | ExponentialLR γ=0.95 | ExponentialLR γ=0.95 | |
| Epochs | 100 | 100 | |
| Batch size | 32 | 32 | |
| Early stopping patience | 15 epochs | 15 epochs | Monitor: val_class1_f1 |
| EMA beta (CS-IB) | 0.7 | 0.7 | |
| Warmup fraction | 0.1 | 0.1 | 10% of epochs |
| Delta (annealing start) | 0.875 | 0.875 | 87.5% of epochs |

---

## Quick Start

See [SETUP.md](SETUP.md) for the full installation guide. Short version:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset at datasets/tiles224_v3/ and datasets/tile_info224_v3.csv

# 3. Run Mask2Former baseline
python main.py --config configs/baseline.yaml

# 4. Run CS-InfoBatch at ratio 0.5
python main.py --config configs/cs_infobatch.yaml --prune_ratio 0.5

# 5. Run DeepLabV3+ baseline
python main.py --config configs/baseline_deeplabv3.yaml

# 6. Analyze results (after training)
python evaluate/analyze_results.py

# 7. Compare models
python evaluate/compare_models.py
```

---

## Acknowledgements

This project is built on top of the [TreeFinder](https://github.com/zhwang0/treefinder) baseline by Zhihao Wang et al. (NeurIPS 2025 Datasets & Benchmarks). The CS-InfoBatch pruning strategy extends [InfoBatch](https://arxiv.org/abs/2303.04947) (Qin et al., 2023).

This work was completed as part of CSC 722: Advanced Topics in Machine Learning at North Carolina State University (Spring 2026), under the supervision of Dr. Ranga Raju Vatsavai.
