# CS-InfoBatch: Class-Stratified Data Pruning for Dead Tree Segmentation

This project investigates whether intelligent training data pruning can reduce compute cost while maintaining segmentation quality for aerial tree detection.

The key contribution is **CS-InfoBatch (Class-Stratified InfoBatch)**, a pruning strategy that extends vanilla InfoBatch with per-class difficulty thresholds and a class-balanced loss signal, designed for the severe class imbalance present in tree canopy segmentation (~0.37% foreground pixels). Built on top of the [TreeFinder](https://github.com/zhwang0/treefinder) baseline (NeurIPS 2025 Datasets & Benchmarks).

---

## What This Project Does

Standard training uses every sample every epoch. Data pruning removes "easy" samples (those the model has already learned) to reduce total training time. This project benchmarks three pruning strategies against a full-data baseline across two model architectures:

| Strategy | Description |
|----------|-------------|
| **CS-InfoBatch** | Our method. Per-class pruning thresholds, EMA scoring, class-balanced loss weighting |
| **InfoBatch** | Vanilla adaptive pruning with a global difficulty threshold |
| **Random Pruning** | Fixed random subset, chosen once before training |
| **Baseline** | Full dataset, no pruning |

Each pruning strategy is evaluated at five pruning ratios: 0.3, 0.5, 0.7, 0.8, 0.9 (fraction of easy samples removed each epoch).

---

## Models

### Mask2Former (Swin-Tiny backbone)
Loaded from `facebook/mask2former-swin-tiny-ade-semantic`. The Swin patch embedding is replaced to accept 5 input channels (RGB + NIR + NDVI). The classifier head is replaced for binary (tree/background) segmentation.

### DeepLabV3+ (ResNet50 backbone)
Loaded from torchvision with ImageNet pretrained weights. The first Conv2d is replaced to accept 5 input channels; extra channel weights are initialized as the mean of the pretrained RGB weights. The auxiliary classifier is disabled.

Both models receive the same 5-band input `(B, 5, 224, 224)` and produce `(B, 1, 224, 224)` logits. The training loop, pruning strategies, and evaluation code are fully model-agnostic.

---

## Datasets

Two datasets are supported. See [SETUP.md](SETUP.md#step-5--dataset-setup) for download and extraction steps.

### TreeFinder (primary)

5-band 224x224 NAIP GeoTIFF tiles from the [TreeFinder dataset](https://www.kaggle.com/datasets/zhihaow/tree-finder) (~15.5K tiles, 0.6 m/pixel, severe class imbalance: 0.37% dead-tree pixels).

NDVI is computed on-the-fly: `NDVI = (NIR - R) / (NIR + R + ε)`. The 5 input channels passed to the model are: R, G, B, NIR, NDVI.

### Deforestation (balanced-regime control, `dataset2` configs)

2,601 Sentinel-2 tiles from the [Amazon Rainforest Deforestation Dataset](https://www.kaggle.com/datasets/akhilchibber/deforestation-detection-dataset) (ground truth from PRODES/INPE), near-balanced (~61% foreground), used as a causal control to check that CS-InfoBatch's advantage is specific to class imbalance rather than pruning in general. RGB + NIR tiles are paired with separate ground-truth tiles; the same 5-channel (R, G, B, NIR, NDVI) contract is used, so both datasets run through the same pipeline unmodified.

---

## Repository Structure

```
CS-Infobatch/
├── main.py                        # Entry point: train + evaluate one experiment
│
├── configs/                       # YAML configs: one per model x pruner x dataset
│   ├── base.yaml                  # Shared defaults
│   ├── baseline*.yaml             # No pruning (Mask2Former / DeepLabV3+)
│   ├── cs_infobatch*.yaml         # CS-InfoBatch (cs_infobatch_short_ramp.yaml: ramp ends at epoch 20)
│   ├── infobatch*.yaml            # Vanilla InfoBatch
│   ├── random_prune*.yaml         # Random pruning
│   └── *_dataset2.yaml            # Deforestation (balanced-regime control) variants
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
│   └── utils.py                   # load_tile, augment_tile (both dataset layouts)
│
├── evaluation/
│   └── evaluator.py               # Test-set evaluation: IoU, F1, precision, recall
│
├── evaluate/                      # Post-training analysis
│   ├── generate_probs.py          # Inference pass -> per-pixel .npz prob files
│   ├── analyze_results.py         # Figures + tables (TreeFinder)
│   ├── analyze_results_dataset2.py# Figures + tables (Deforestation control)
│   ├── compare_models.py          # Mask2Former vs DeepLabV3+ side-by-side
│   ├── plot_deeplab_variance.py   # 5-seed variance bands (CS-IB vs InfoBatch)
│   └── plot_mask2former_multiseed.py # 3-seed F1 vs ratio + matched-budget figures
│
├── utils/
│   ├── split_cache.py             # Dataset split generation and caching
│   ├── tools.py                   # Config loading, arg parsing, logging, seeding
│   └── gdrive_upload.py           # Auto-upload results to Google Drive
│
├── scripts/
│   ├── setup_gdrive.py            # One-time Google Drive OAuth setup
│   ├── run_mask2former_multiseed.sh # Mask2Former 3-seed study (main + matched-budget sets)
│   ├── build_tables.py            # Regenerate results tables from metrics.yaml
│   ├── verify_probs_consistency.py# Sanity-check .npz files against metrics.yaml
│   ├── inspect_dataset2.py        # EDA for the Deforestation dataset
│   └── make_dataset2_info.py      # Builds the Deforestation tile_info.csv
│
├── results/                       # Per-run epoch logs + metrics.yaml
├── results_analysis/              # Generated figures, comparison tables, probs/
└── splits/                        # Cached train/val/test split JSONs
```

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

# 6. Multi-seed (seeds 42, 123) and matched-budget Mask2Former runs
bash scripts/run_mask2former_multiseed.sh --set main
bash scripts/run_mask2former_multiseed.sh --set short_ramp

# 7. Analyze results (after training)
python evaluate/analyze_results.py

# 8. Compare models
python evaluate/compare_models.py
```

---

## Acknowledgements

This project is built on top of the [TreeFinder](https://github.com/zhwang0/treefinder) baseline by Zhihao Wang et al. (NeurIPS 2025 Datasets & Benchmarks). The CS-InfoBatch pruning strategy extends [InfoBatch](https://arxiv.org/abs/2303.04947) (Qin et al., 2023).

This work was completed as part of CSC 722: Advanced Topics in Machine Learning at North Carolina State University (Spring 2026), under the supervision of Dr. Ranga Raju Vatsavai.

---

## Citation

If you use this code, please cite the datasets and the method it builds on:

```bibtex
@article{treefinder,
  title={TreeFinder: A US-Scale Benchmark Dataset for Individual Tree Mortality Monitoring Using High-Resolution Aerial Imagery},
  author={Wang, Zhihao and Li, Cooper and Wang, Ruichen and Ma, Lei and Hurtt, George and Jia, Xiaowei and Mai, Gengchen and Li, Zhili and Xie, Yiqun},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2026}
}

@misc{deforestation,
  title={Amazon Rainforest Deforestation Dataset},
  author={Chhibber, Akhil},
  year={2021},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/akhilchibber/deforestation-detection-dataset}
}

@inproceedings{infobatch,
  title={Infobatch: Lossless training speed up by unbiased dynamic data pruning},
  author={Qin, Ziheng and Wang, Kai and Zheng, Zangwei and Gu, Jianyang and Peng, Xiangyu and Xu, Zhaopan and Daquan, Zhou and Shang, Lei and Sun, Baigui and Xie, Xuansong and others},
  booktitle={International Conference on Learning Representations},
  volume={2024},
  pages={39723--39744},
  year={2024}
}
```