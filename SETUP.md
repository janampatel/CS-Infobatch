# Setup & Run Guide

Complete step-by-step instructions to install, configure, and run every experiment in this repository. Follow these steps in order.

---

## Requirements

| Requirement | Version |
|-------------|---------|
| NVIDIA GPU | 8 GB VRAM minimum (16 GB recommended) |
| NVIDIA Driver | 570.x or newer |
| CUDA | 12.8 |
| Python | 3.10 – 3.12 |
| OS | Ubuntu 20.04+ (or any Linux with GDAL available) |

Check your driver and CUDA version:
```bash
nvidia-smi
nvcc --version
```

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/janampatel/CS-Infobatch.git
cd CS-Infobatch
```

---

## Step 2 — Install System Dependency (GDAL)

`rasterio` (used to read GeoTIFF tiles) requires the GDAL system library. Install it before pip:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y libgdal-dev gdal-bin

# Verify
gdal-config --version
```

---

## Step 3 — Create a Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 4 — Install Python Dependencies

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

## Step 5 — Dataset Setup

The dataset is the 224×224 NAIP tile dataset (`tiles224_v3`) from the [TreeFinder dataset](https://www.kaggle.com/datasets/zhihaow/tree-finder). Each tile is a 5-band GeoTIFF: bands 1-4 are R, G, B, NIR and band 5 is the label mask (0=background, 1=tree, 255=no-data).

Place the dataset in the following structure relative to the project root:

```
CS-Infobatch/
└── datasets/
    ├── tile_info224_v3.csv       ← metadata CSV (tile filenames, label sizes, etc.)
    └── tiles224_v3/
        ├── tile_0001.tif
        ├── tile_0002.tif
        └── ...                   ← all .tif tile files
```

Verify the dataset path:
```bash
python -c "
import pandas as pd
df = pd.read_csv('datasets/tile_info224_v3.csv')
print(f'Tiles in CSV: {len(df)}')
print(df.head(3))
"
```

The first time you run any training command, the split file is auto-generated and cached at:
```
splits/random_seed2025_train0.8_val0.1_test0.2.json
```
This takes a few minutes (reads every train tile to compute foreground statistics). All subsequent runs reuse this file instantly.

---

## Step 6 — (Optional) Google Drive Auto-Upload

If you want training results auto-uploaded to Google Drive after each run, complete this step. Otherwise skip to Step 7 — the code works fully without it.

**6a. Create Google Drive credentials**

1. Go to [https://console.cloud.google.com](https://console.cloud.google.com)
2. Create or select a project
3. Enable the **Google Drive API**
4. Go to **APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID**
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

**6c. Enable upload in configs**

Google Drive upload is already enabled by default (`gdrive.enabled: true`) in all config files. To disable it for a specific run, pass nothing — or set `gdrive.enabled: false` in the config.

---

## Step 7 — Run Experiments

All commands are run from the **project root**. Results, checkpoints, and logs are automatically saved under `results/`, `checkpoints/`, and `logs/`.

### Mask2Former Experiments

#### Baseline (no pruning)
```bash
python main.py --config configs/baseline.yaml
```
Expected result: `results/baseline_baseline/baseline_baseline_metrics.yaml`

#### CS-InfoBatch (class-stratified pruning)
```bash
python main.py --config configs/cs_infobatch.yaml --prune_ratio 0.3
python main.py --config configs/cs_infobatch.yaml --prune_ratio 0.5
python main.py --config configs/cs_infobatch.yaml --prune_ratio 0.7
python main.py --config configs/cs_infobatch.yaml --prune_ratio 0.8
python main.py --config configs/cs_infobatch.yaml --prune_ratio 0.9
```

#### InfoBatch (vanilla adaptive pruning)
```bash
python main.py --config configs/infobatch.yaml --prune_ratio 0.3
python main.py --config configs/infobatch.yaml --prune_ratio 0.5
python main.py --config configs/infobatch.yaml --prune_ratio 0.7
python main.py --config configs/infobatch.yaml --prune_ratio 0.8
python main.py --config configs/infobatch.yaml --prune_ratio 0.9
```

#### Random Pruning (fixed subset baseline)
```bash
python main.py --config configs/random_prune.yaml --prune_ratio 0.3
python main.py --config configs/random_prune.yaml --prune_ratio 0.5
python main.py --config configs/random_prune.yaml --prune_ratio 0.7
python main.py --config configs/random_prune.yaml --prune_ratio 0.8
python main.py --config configs/random_prune.yaml --prune_ratio 0.9
```

---

### DeepLabV3+ Experiments

#### Baseline (no pruning)
```bash
python main.py --config configs/baseline_deeplabv3.yaml
```
Expected result: `results/baseline_deeplabv3/baseline_deeplabv3_metrics.yaml`

#### CS-InfoBatch
```bash
python main.py --config configs/cs_infobatch_deeplabv3.yaml --prune_ratio 0.3
python main.py --config configs/cs_infobatch_deeplabv3.yaml --prune_ratio 0.5
python main.py --config configs/cs_infobatch_deeplabv3.yaml --prune_ratio 0.7
python main.py --config configs/cs_infobatch_deeplabv3.yaml --prune_ratio 0.8
python main.py --config configs/cs_infobatch_deeplabv3.yaml --prune_ratio 0.9
```

#### InfoBatch
```bash
python main.py --config configs/infobatch_deeplabv3.yaml --prune_ratio 0.3
python main.py --config configs/infobatch_deeplabv3.yaml --prune_ratio 0.5
python main.py --config configs/infobatch_deeplabv3.yaml --prune_ratio 0.7
python main.py --config configs/infobatch_deeplabv3.yaml --prune_ratio 0.8
python main.py --config configs/infobatch_deeplabv3.yaml --prune_ratio 0.9
```

#### Random Pruning
```bash
python main.py --config configs/random_prune_deeplabv3.yaml --prune_ratio 0.3
python main.py --config configs/random_prune_deeplabv3.yaml --prune_ratio 0.5
python main.py --config configs/random_prune_deeplabv3.yaml --prune_ratio 0.7
python main.py --config configs/random_prune_deeplabv3.yaml --prune_ratio 0.8
python main.py --config configs/random_prune_deeplabv3.yaml --prune_ratio 0.9
```

---

### CLI Overrides

Any config value can be overridden at runtime:

| Flag | Description | Example |
|------|-------------|---------|
| `--config` | Config YAML path | `--config configs/baseline.yaml` |
| `--prune_ratio` | Pruning ratio (0–1) | `--prune_ratio 0.5` |
| `--exp_id` | Override experiment ID | `--exp_id my_run` |
| `--gpu_id` | GPU device index | `--gpu_id 1` |
| `--pruner_type` | Override pruner | `--pruner_type cs_infobatch` |

---

## Step 8 — Generate Per-Pixel Probabilities (Optional, for ROC/Calibration plots)

After training, run inference on the test set to save probability files. This is required for figures 8–10 (ROC-AUC, calibration curves, probability distributions).

```bash
# All experiments (skips any with missing checkpoints)
python evaluate/generate_probs.py

# Specific experiments only
python evaluate/generate_probs.py --only baseline_baseline cs_infobatch_r0.5_cs_infobatch baseline_deeplabv3
```

Output: `results_analysis/probs/{exp_name}_probs.npz`

Runtime: ~5–10 minutes per experiment on GPU.

---

## Step 9 — Analyze Results

Generate all figures and tables. Only needs the `results/` directory (no GPU required).

```bash
python evaluate/analyze_results.py
```

Output saved to `results_analysis/`:
- `results_table.csv` and `results_table.tex` — full metrics table
- `fig1_f1_vs_ratio.png` — F1 vs pruning ratio
- `fig2_time_vs_ratio.png` — training time vs pruning ratio
- `fig3_pareto.png` — quality-efficiency Pareto frontier
- `fig4_learning_curves.png` — val F1 convergence per ratio
- `fig5_samples_per_epoch.png` — adaptive pruning dynamics
- `fig6_degradation_heatmap.png` — ΔF1 vs baseline heatmap
- `fig7_efficiency.png` — F1 per training hour
- `fig8_roc_auc.png` — ROC curves (requires `generate_probs.py`)
- `fig9_calibration.png` — calibration/reliability diagrams (requires `generate_probs.py`)
- `fig10_prob_dist.png` — predicted probability distributions (requires `generate_probs.py`)

---

## Step 10 — Compare Models (Mask2Former vs DeepLabV3+)

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

## Common Errors

**`rasterio` import fails or GDAL not found**
```
Solution: sudo apt install libgdal-dev gdal-bin
          Then pip install rasterio --no-binary rasterio
```

**`CUDA out of memory`**
```
Solution: Reduce batch_size in the config (e.g., 16 instead of 32)
          Or use --gpu_id to select a different GPU
```

**`FileNotFoundError: datasets/tile_info224_v3.csv`**
```
Solution: Verify the dataset is placed at datasets/ relative to the project root.
          Run the verification command in Step 5.
```

**`transformers` raises a model download error on first run**
```
Solution: The Mask2Former weights are downloaded from HuggingFace on first use.
          Ensure the server has internet access, or pre-download with:
          python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('facebook/mask2former-swin-tiny-ade-semantic')"
```

**`Unknown model: 'deeplabv3'`**
```
Solution: Ensure models/__init__.py contains the deeplabv3 branch.
          git pull origin main to get the latest code.
```
