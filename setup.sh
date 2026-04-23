#!/bin/bash
# =============================================================================
# CS-InfoBatch — Full Server Setup Script
# Run once after each 24h factory reset: bash setup.sh
#
# What it does:
#   1. Installs Miniconda (if not present)
#   2. Creates/activates the conda environment
#   3. Clones / pulls the code from GitHub
#   4. Downloads data from cloud storage (rclone / gdown / aws s3)
#   5. Downloads pretrained model weights
# =============================================================================

set -e  # exit on first error

# ── Configuration ────────────────────────────────────────────────────────────
REPO_URL="https://github.com/pranshavpatel/cs-infobatch.git"   # TODO: set your repo URL
REPO_DIR="$(pwd)"
ENV_NAME="cs-infobatch"
DATA_DIR="$REPO_DIR/datasets"

# Data source — pick ONE and configure it:
# Option A: AWS S3
S3_BUCKET="s3://your-bucket-name/tiles224_v3"
# Option B: Google Drive (get folder ID from Drive URL)
GDRIVE_FOLDER_ID="1Z4F1QhygQ9DwJ_oXP52mdTzLJw8dd_m7"
# Option C: rclone remote (works with S3, GDrive, B2, etc.)
RCLONE_REMOTE="myremote:cs-infobatch-data/tiles224_v3"

# ── Step 1: Miniconda ──────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    if [ -d "$HOME/miniconda3" ]; then
        echo "[setup] Miniconda directory found. Activating..."
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    else
        echo "[setup] Installing Miniconda..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
        conda init bash
        source ~/.bashrc
    fi
else
    echo "[setup] Conda already installed: $(conda --version)"
    eval "$(conda shell.bash hook)"
fi

# ── Step 2: Conda environment ──────────────────────────────────────────────────
if conda env list | grep -q "^$ENV_NAME "; then
    echo "[setup] Conda env '$ENV_NAME' already exists — updating..."
    conda env update -n "$ENV_NAME" -f "$REPO_DIR/environment.yml" --prune
else
    echo "[setup] Creating conda env '$ENV_NAME'..."
    conda env create -f "$REPO_DIR/environment.yml"
fi
conda activate "$ENV_NAME"

# ── Step 3: Code ───────────────────────────────────────────────────────────────
# if [ -d "$REPO_DIR/.git" ]; then
#     echo "[setup] Pulling latest code..."
#     cd "$REPO_DIR" && git pull
# else
#     echo "[setup] Cloning repo..."
#     git clone "$REPO_URL" "$REPO_DIR"
#     cd "$REPO_DIR"
# fi

# ── Step 4: Data download ──────────────────────────────────────────────────────
mkdir -p "$DATA_DIR/tiles224_v3"

# --- Option A: AWS S3 (fastest on cloud servers) ---
# Uncomment if using S3:
# echo "[setup] Downloading data from S3..."
# aws s3 sync "$S3_BUCKET" "$DATA_DIR/tiles224_v3/" --no-progress

# --- Option B: rclone (universal — works with GDrive, S3, B2, Wasabi, etc.) ---
# First time only: run `rclone config` interactively on the server
# Then copy ~/.config/rclone/rclone.conf to a persistent location or pass via env
# Uncomment if using rclone:
# echo "[setup] Downloading data via rclone..."
# rclone copy "$RCLONE_REMOTE" "$DATA_DIR/tiles224_v3/" --progress --transfers 8

# --- Option C: Google Drive via gdown ---
# Uncomment if using Google Drive:
# pip install gdown
# echo "[setup] Downloading data from Google Drive..."
# gdown --folder "https://drive.google.com/drive/folders/$GDRIVE_FOLDER_ID" \
#        --output "$DATA_DIR/"

# # Also download the CSV metadata file:
# # aws s3 cp s3://your-bucket/tile_info224_v3.csv "$DATA_DIR/"
# # or
# # gdown "https://drive.google.com/uc?id=YOUR_CSV_FILE_ID" -O "$DATA_DIR/tile_info224_v3.csv"

# echo "[setup] Data directory contents:"
# ls -lh "$DATA_DIR/" 2>/dev/null || echo "  (empty — configure data download above)"

# ── Step 5: HuggingFace model cache ───────────────────────────────────────────
# Pre-download the Mask2Former weights so training doesn't stall on first run
echo "[setup] Caching HuggingFace model weights..."
python -c "
from transformers import Mask2FormerForUniversalSegmentation, AutoConfig
print('Downloading facebook/mask2former-swin-tiny-ade-semantic...')
AutoConfig.from_pretrained('facebook/mask2former-swin-tiny-ade-semantic')
Mask2FormerForUniversalSegmentation.from_pretrained(
    'facebook/mask2former-swin-tiny-ade-semantic', ignore_mismatched_sizes=True
)
print('Done.')
" 2>/dev/null || echo "  (skipped — no internet or model already cached)"

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Setup complete!"
echo " Activate env: conda activate $ENV_NAME"
echo " Run baseline: cd $REPO_DIR && make baseline"
echo " Or directly:  python main.py --config configs/baseline.yaml"
echo "=========================================="
