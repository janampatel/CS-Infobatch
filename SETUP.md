# Setup & Run Guide

Complete step-by-step instructions to install, configure, and run every experiment in this repository. Follow these steps in order.

---

## Requirements

| Requirement | Version |
|-------------|---------|
| NVIDIA GPU | 8 GB VRAM minimum (16 GB recommended) |
| NVIDIA Driver | 570.x or newer |
| CUDA | 12.8 |
| Python | 3.10 - 3.12 |
| OS | Ubuntu 20.04+ (or any Linux with GDAL available) |

Check your driver and CUDA version:
```bash
nvidia-smi
nvcc --version
```

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/janampatel/CS-Infobatch.git
cd CS-Infobatch
```

---

## Step 2: Install System Dependency (GDAL)

`rasterio` (used to read GeoTIFF tiles) requires the GDAL system library. Install it before pip:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y libgdal-dev gdal-bin

# Verify
gdal-config --version
```

---

## Step 3: Create a Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 4: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs PyTorch 2.11.0 with CUDA 12.8, torchvision, HuggingFace Transformers (for Mask2Former), rasterio, and all other dependencies.

Verify the install:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
python -c "import rasterio; print('rasterio:', rasterio.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
```

All three commands must succeed before continuing.

---

## Step 5: Dataset Setup

See [README.md](README.md#datasets) for dataset details and citations. Download each dataset from Kaggle and extract into `datasets/` with this layout:

**TreeFinder** (required): [Kaggle](https://www.kaggle.com/datasets/zhihaow/tree-finder)
```
datasets/
├── tile_info224_v3.csv
└── tiles224_v3/
    └── *.tif
```

**Deforestation** (optional, only for `*_dataset2.yaml` configs): [Kaggle](https://www.kaggle.com/datasets/akhilchibber/deforestation-detection-dataset)
```
datasets/
├── rgb-tiles/cf-s2-kbd-full-raster/
│   └── *.tif                       # 4-band float32: R, G, B, NIR
└── gt-tiles/raster/
    └── *.tif                       # 1-band uint8 ground truth, same filenames
```
Then generate its metadata CSV:
```bash
python scripts/inspect_dataset2.py
```

The first training run auto-generates and caches the train/val/test split under `splits/`; subsequent runs reuse it.

---

## Step 6 (Optional): Google Drive Auto-Upload

If you want training results auto-uploaded to Google Drive after each run, complete this step. Otherwise skip to Step 7, the code works fully without it.

**6a. Create Google Drive credentials**

1. Go to [https://console.cloud.google.com](https://console.cloud.google.com)
2. Create or select a project
3. Enable the **Google Drive API**
4. Go to **APIs & Services > Credentials > Create Credentials > OAuth 2.0 Client ID**
5. Application type: **Desktop app**
6. Download the JSON file, rename it to `credentials.json`, place it in the project root

**6b. Run the one-time auth script**

Run this on a machine with a browser (can be your local machine):
```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
python scripts/setup_gdrive.py
```

A browser window opens. Sign in with Google and grant permissions. This saves `token.json` in the project root.

If running on a remote server without a browser, run `setup_gdrive.py` locally first, then copy both files to the server:
```bash
scp credentials.json token.json <user>@<server>:/path/to/CS-Infobatch/
```

A `token.json` is already included in this repo for convenience; it is `.gitignore`d, so it stays local and is never pushed. If it expires or you want to auth with your own account, delete it and re-run `setup_gdrive.py`.

**6c. Enable upload in configs**

Google Drive upload is already enabled by default (`gdrive.enabled: true`) in all config files. To disable it, set `gdrive.enabled: false` in the config.

---

## Step 7: Run Experiments

All commands run from the **project root**. Results, checkpoints, and logs are saved under `results/`, `checkpoints/`, and `logs/`.

### Mask2Former

```bash
# Baseline (no pruning)
python main.py --config configs/baseline.yaml

# CS-InfoBatch / InfoBatch / Random Pruning, at each ratio
for r in 0.3 0.5 0.7 0.8 0.9; do
  python main.py --config configs/cs_infobatch.yaml --prune_ratio $r
  python main.py --config configs/infobatch.yaml --prune_ratio $r
  python main.py --config configs/random_prune.yaml --prune_ratio $r
done
```
Expected result: `results/baseline_baseline/baseline_baseline_metrics.yaml`

### DeepLabV3+

```bash
# Baseline (no pruning)
python main.py --config configs/baseline_deeplabv3.yaml

# CS-InfoBatch / InfoBatch / Random Pruning, at each ratio
for r in 0.3 0.5 0.7 0.8 0.9; do
  python main.py --config configs/cs_infobatch_deeplabv3.yaml --prune_ratio $r
  python main.py --config configs/infobatch_deeplabv3.yaml --prune_ratio $r
  python main.py --config configs/random_prune_deeplabv3.yaml --prune_ratio $r
done
```
Expected result: `results/baseline_deeplabv3/baseline_deeplabv3_metrics.yaml`

Swap in `*_dataset2.yaml` configs to run the same sweeps on the Deforestation dataset.

### Multi-seed and matched-budget runs (TreeFinder)

The runs above use seed 2025. The multi-seed results add seeds 42 and 123; `--seed` also
selects the split, so each seed is a fresh train/val/test split and initialization, and
methods are compared within a seed. Seed-labelled runs are saved as `..._s1/_s2/_s3`
(s1 = 2025, s2 = 42, s3 = 123).

```bash
# Mask2Former: baseline, InfoBatch and CS-InfoBatch for seeds 42 and 123 (all ratios)
bash scripts/run_mask2former_multiseed.sh --set main

# Mask2Former: matched per-epoch budget. CS-InfoBatch with the ramp ending at epoch 20
# (configs/cs_infobatch_short_ramp.yaml), r = 0.9 / 0.5 / 0.7, seeds 2025 / 42 / 123
bash scripts/run_mask2former_multiseed.sh --set short_ramp

# DeepLabV3+: CS-InfoBatch and InfoBatch over five seeds
seeds=(2025 42 123 7 99)
for cfg in cs_infobatch_deeplabv3 infobatch_deeplabv3; do
  for r in 0.3 0.5 0.7 0.8 0.9; do
    for i in 0 1 2 3 4; do
      python main.py --config configs/$cfg.yaml --prune_ratio $r \
        --seed ${seeds[$i]} --exp_id deeplabv3_s$((i+1))
    done
  done
done
```
Finished runs are skipped by the Mask2Former script, so it can be re-run after an interruption.
It ends by printing the per-seed and mean ± std test F1 table. `python evaluate/plot_mask2former_multiseed.py`
plots the Mask2Former study and `python evaluate/plot_deeplab_variance.py` the DeepLabV3+ study.

### CLI Overrides

| Flag | Description | Example |
|------|-------------|---------|
| `--config` | Config YAML path | `--config configs/baseline.yaml` |
| `--prune_ratio` | Pruning ratio (0-1) | `--prune_ratio 0.5` |
| `--exp_id` | Override experiment ID | `--exp_id my_run` |
| `--gpu_id` | GPU device index | `--gpu_id 1` |
| `--pruner_type` | Override pruner | `--pruner_type cs_infobatch` |
| `--seed` | Random seed (also selects the split) | `--seed 42` |

---

## Step 8 (Optional): Generate Per-Pixel Probabilities

Needed for ROC-AUC, calibration, and probability-distribution figures (fig8-10). Runs inference on the test set from each saved checkpoint.

```bash
python evaluate/generate_probs.py                      # all experiments
python evaluate/generate_probs.py --only baseline_baseline cs_infobatch_r0.5_cs_infobatch
```

Output: `results_analysis/probs/{exp_name}_probs.npz`. Runtime: ~5-10 minutes per experiment on GPU.

---

## Step 9: Analyze Results

No GPU required, only needs `results/`.

```bash
python evaluate/analyze_results.py
```

Output saved to `results_analysis/`: `results_table.csv`/`.tex`, and figures for F1 vs. ratio, training time, Pareto frontier, learning curves, samples/epoch, degradation heatmap, efficiency, and (if `generate_probs.py` was run) ROC-AUC, calibration, and probability distributions.

---

## Step 10: Compare Models (Mask2Former vs DeepLabV3+)

```bash
python evaluate/compare_models.py                    # all ratios
python evaluate/compare_models.py --ratios 0.5       # ratio 0.5 only
python evaluate/compare_models.py --ratios 0.5 0.7   # multiple ratios
```

Output saved to `results_analysis/`.

---

## Output Directory Structure

After running all experiments, the directory structure will look like this:

```
CS-Infobatch/
├── checkpoints/
│   ├── baseline_baseline/
│   │   └── baseline_baseline_best.pth
│   ├── cs_infobatch_r0.3_cs_infobatch/
│   │   └── cs_infobatch_r0.3_cs_infobatch_best.pth
│   └── ...
├── results/
│   ├── baseline_baseline/
│   │   ├── baseline_baseline_metrics.yaml
│   │   └── baseline_baseline_epoch_log.csv
│   └── ...
├── logs/
│   ├── baseline_baseline.log
│   └── ...
├── splits/
│   └── random_seed2025_train0.8_val0.1_test0.2.json
└── results_analysis/
    ├── results_table.csv
    ├── results_table.tex
    ├── fig1_f1_vs_ratio.png
    └── ...
```

---
