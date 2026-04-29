#!/usr/bin/env python3
"""
generate_probs.py — Inference-only pass that saves per-pixel probabilities.

NOT retraining. Loads each saved checkpoint, runs a forward pass on the
shared test set, and saves (probs, labels) as a compressed .npz file.
These files are then read by analyze_results.py for ROC-AUC and calibration plots.

Runtime: ~5-10 min per experiment on GPU (vs hours for training).

Usage (run from project root on the GPU server):
  python generate_probs.py              # process all known experiments
  python generate_probs.py --only baseline_baseline cs_infobatch_r0.5_cs_infobatch
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Ensure imports resolve from project root
PROJECT_ROOT = Path(__file__).parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.tools import load_config, set_seed
from data.random_split import TileDataset
from utils.split_cache import get_or_create_split
from models import get_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Cap pixels per run to stay within memory budget (~100 MB at float16)
MAX_PIXELS = 3_000_000

PROBS_DIR = PROJECT_ROOT / "results_analysis" / "probs"

# exp_name -> (config_file, prune_ratio_override)
# prune_ratio_override=None means use whatever is already in the config
EXPERIMENTS = {
    "baseline_baseline":               ("configs/baseline.yaml",     None),
    "cs_infobatch_r0.3_cs_infobatch":  ("configs/cs_infobatch.yaml", 0.3),
    "cs_infobatch_r0.5_cs_infobatch":  ("configs/cs_infobatch.yaml", 0.5),
    "cs_infobatch_r0.7_cs_infobatch":  ("configs/cs_infobatch.yaml", 0.7),
    "cs_infobatch_r0.8_cs_infobatch":  ("configs/cs_infobatch.yaml", 0.8),
    "cs_infobatch_r0.9_cs_infobatch":  ("configs/cs_infobatch.yaml", 0.9),
    "infobatch_r0.3_infobatch":        ("configs/infobatch.yaml",    0.3),
    "infobatch_r0.5_infobatch":        ("configs/infobatch.yaml",    0.5),
    "infobatch_r0.7_infobatch":        ("configs/infobatch.yaml",    0.7),
    "infobatch_r0.8_infobatch":        ("configs/infobatch.yaml",    0.8),
    "infobatch_r0.9_infobatch":        ("configs/infobatch.yaml",    0.9),
    "random_r0.3_random_prune":        ("configs/random_prune.yaml", 0.3),
    "random_r0.5_random_prune":        ("configs/random_prune.yaml", 0.5),
    "random_r0.7_random_prune":        ("configs/random_prune.yaml", 0.7),
    "random_r0.8_random_prune":        ("configs/random_prune.yaml", 0.8),
    "random_r0.9_random_prune":        ("configs/random_prune.yaml", 0.9),
    # --- DeepLabV3 entries ---
    "baseline_deeplabv3":                        ("configs/baseline_deeplabv3.yaml",      None),
    "cs_infobatch_r0.3_deeplabv3":               ("configs/cs_infobatch_deeplabv3.yaml",  0.3),
    "cs_infobatch_r0.5_deeplabv3":               ("configs/cs_infobatch_deeplabv3.yaml",  0.5),
    "cs_infobatch_r0.7_deeplabv3":               ("configs/cs_infobatch_deeplabv3.yaml",  0.7),
    "cs_infobatch_r0.8_deeplabv3":               ("configs/cs_infobatch_deeplabv3.yaml",  0.8),
    "cs_infobatch_r0.9_deeplabv3":               ("configs/cs_infobatch_deeplabv3.yaml",  0.9),
    "infobatch_r0.3_deeplabv3":                  ("configs/infobatch_deeplabv3.yaml",     0.3),
    "infobatch_r0.5_deeplabv3":                  ("configs/infobatch_deeplabv3.yaml",     0.5),
    "infobatch_r0.7_deeplabv3":                  ("configs/infobatch_deeplabv3.yaml",     0.7),
    "infobatch_r0.8_deeplabv3":                  ("configs/infobatch_deeplabv3.yaml",     0.8),
    "infobatch_r0.9_deeplabv3":                  ("configs/infobatch_deeplabv3.yaml",     0.9),
    "random_r0.3_deeplabv3":                     ("configs/random_prune_deeplabv3.yaml",  0.3),
    "random_r0.5_deeplabv3":                     ("configs/random_prune_deeplabv3.yaml",  0.5),
    "random_r0.7_deeplabv3":                     ("configs/random_prune_deeplabv3.yaml",  0.7),
    "random_r0.8_deeplabv3":                     ("configs/random_prune_deeplabv3.yaml",  0.8),
    "random_r0.9_deeplabv3":                     ("configs/random_prune_deeplabv3.yaml",  0.9),
}


def run_inference(exp_name: str, cfg: dict) -> bool:
    """
    Load checkpoint for exp_name, run on the shared test set, save .npz.
    Returns True on success, False if checkpoint is missing.
    """
    out_path = PROBS_DIR / f"{exp_name}_probs.npz"
    if out_path.exists():
        logger.info(f"  Already exists, skipping: {out_path.name}")
        return True

    ckpt = Path(cfg["output"]["checkpoint_dir"]) / exp_name / f"{exp_name}_best.pth"
    if not ckpt.exists():
        logger.warning(f"  Checkpoint not found: {ckpt}  (skipping)")
        return False

    gpu_id = cfg["experiment"].get("gpu_id", 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}  |  checkpoint: {ckpt.name}")

    set_seed(cfg["experiment"]["seed"])

    # Re-use the cached split (same test set for all experiments)
    split = get_or_create_split(cfg)
    test_ds = TileDataset(split["test_tiles"], cfg, split="test")

    bs = cfg["training"]["batch_size"]
    nw = min(cfg["experiment"].get("num_workers", 4), 4)
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0),
    )
    logger.info(f"  Test tiles: {len(test_ds)}")

    model = get_model(cfg["model"])
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    probs_list, labels_list = [], []
    n_pixels = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  inference", leave=False, ncols=80):
            imgs    = batch["image"].to(device)          # [B, C, H, W]
            gt      = batch["label"].cpu().numpy()       # [B, H, W]   uint8
            no_data = batch["no_data_mask"].cpu().numpy().astype(bool)  # [B, H, W]

            logits = model(imgs)                                 # [B, 1, H, W]
            p = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # [B, H, W]   float32

            for i in range(p.shape[0]):
                valid = ~no_data[i]
                pv = p[i][valid].astype(np.float16)
                lv = gt[i][valid].astype(np.uint8)
                probs_list.append(pv)
                labels_list.append(lv)
                n_pixels += int(valid.sum())

    all_probs  = np.concatenate(probs_list)
    all_labels = np.concatenate(labels_list)

    if n_pixels > MAX_PIXELS:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_pixels, size=MAX_PIXELS, replace=False)
        idx.sort()
        all_probs  = all_probs[idx]
        all_labels = all_labels[idx]
        logger.info(f"  Subsampled {n_pixels:,} -> {MAX_PIXELS:,} valid pixels")
    else:
        logger.info(f"  Valid pixels: {n_pixels:,}")

    np.savez_compressed(out_path, probs=all_probs, labels=all_labels)
    pos_rate = all_labels.mean() * 100
    logger.info(f"  Saved {out_path.name}  (tree pixel rate: {pos_rate:.3f}%)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", nargs="+", metavar="EXP_NAME", default=None,
        help="Process only these experiment names (default: all)"
    )
    args = parser.parse_args()

    PROBS_DIR.mkdir(parents=True, exist_ok=True)

    targets = args.only if args.only else list(EXPERIMENTS.keys())
    ok, skipped = 0, 0

    for exp_name in targets:
        if exp_name not in EXPERIMENTS:
            logger.warning(f"Unknown experiment name: {exp_name!r}  (check EXPERIMENTS dict)")
            continue

        cfg_file, ratio_override = EXPERIMENTS[exp_name]
        cfg = load_config(cfg_file)
        if ratio_override is not None:
            cfg["pruner"]["prune_ratio"] = ratio_override

        logger.info(f"=== {exp_name} ===")
        success = run_inference(exp_name, cfg)
        if success:
            ok += 1
        else:
            skipped += 1

    print(f"\nDone: {ok} generated / {skipped} skipped (missing checkpoint).")
    print(f"Prob files in: {PROBS_DIR}")
    print("Next: python analyze_results.py")


if __name__ == "__main__":
    main()
