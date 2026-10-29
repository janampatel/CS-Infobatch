"""Single source of truth for train/val/test splits. The split JSON is created
on first run and reused by all experiments."""
import os
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.utils import load_label_band

logger = logging.getLogger(__name__)

SPLITS_DIR = "splits"


def _split_key(cfg: dict) -> str:
    seed = cfg["experiment"]["seed"]
    sp = cfg["data"]["split"]["random"]
    train_r = sp["train_ratio"]
    val_r = sp["train_val_ratio"]
    test_r = sp["test_ratio"]
    name = cfg["data"].get("dataset_name", "")  # prefix so datasets never share a split file
    prefix = f"{name}_" if name else ""
    return f"{prefix}random_seed{seed}_train{train_r}_val{val_r}_test{test_r}"


def _build_tile_paths(tile_filenames: list, cfg: dict) -> list:
    base = os.path.join(cfg["data"]["root_dir"], cfg["data"]["dataset_dir"])
    return [os.path.join(base, fn) for fn in tile_filenames]


def _compute_cls_labels_and_fg_fraction(tile_paths: list, no_data_value: int) -> tuple:
    """Returns cls_labels (bool per tile: has any tree pixel) and mean fg_pixel_fraction."""
    cls_labels = []
    fg_fracs = []
    n = len(tile_paths)
    for i, path in enumerate(tile_paths):
        if i % 500 == 0:
            logger.info(f"Computing cls_labels: {i}/{n}")
        try:
            label, valid = load_label_band(path, no_data_value)
            n_valid = valid.sum()
            n_tree = label.sum()
            has_tree = bool(n_tree > 0)
            cls_labels.append(has_tree)
            fg_fracs.append(float(n_tree) / max(n_valid, 1))
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            cls_labels.append(False)
            fg_fracs.append(0.0)
    fg_pixel_fraction = float(np.mean(fg_fracs)) if fg_fracs else 0.003722
    return cls_labels, fg_pixel_fraction


def _generate_split(cfg: dict) -> dict:
    seed = cfg["experiment"]["seed"]
    sp = cfg["data"]["split"]["random"]
    train_ratio = sp["train_ratio"]
    test_ratio = sp["test_ratio"]
    train_val_ratio = sp["train_val_ratio"]
    shuffle_by_tile = sp.get("shuffle_by_tile", True)
    pos_threshold = cfg["data"]["split"].get("pos_threshold", 0)
    pos_frac = cfg["data"]["split"].get("pos_frac", 0.0)
    no_data_value = cfg["data"]["no_data_value"]

    info_path = os.path.join(cfg["data"]["root_dir"], cfg["data"]["dataset_info"])
    info = pd.read_csv(info_path)
    shuffle_att = "FileName" if shuffle_by_tile else "ImageRawPath"

    all_tiles = info[shuffle_att].unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(all_tiles)

    n_total = len(all_tiles)
    n_test = int(test_ratio * n_total)
    test_tiles_raw = all_tiles[:n_test]
    test_df = info[info[shuffle_att].isin(test_tiles_raw)]
    test_tiles = test_df["FileName"].unique().tolist()

    n_trainval = int(train_ratio * n_total)
    trainval_tiles = all_tiles[n_test:n_test + n_trainval]
    trainval_df = info[info[shuffle_att].isin(trainval_tiles)]

    is_pos = trainval_df["LabelSize"] >= pos_threshold
    pos_tiles = trainval_df[is_pos]["FileName"].unique().tolist()
    neg_tiles = trainval_df[~is_pos]["FileName"].unique().tolist()
    rng.shuffle(pos_tiles)
    rng.shuffle(neg_tiles)

    n_trainval_count = len(trainval_df)
    if pos_frac > 0:
        n_pos = min(int(pos_frac * n_trainval_count), len(pos_tiles))
        n_neg = min(int(n_pos / pos_frac - n_pos), len(neg_tiles))
    else:
        n_pos = len(pos_tiles)
        n_neg = len(neg_tiles)

    selected_pos = pos_tiles[:n_pos]
    selected_neg = neg_tiles[:n_neg]
    balanced_tiles = selected_pos + selected_neg
    rng.shuffle(balanced_tiles)

    n_val = int(train_val_ratio * len(balanced_tiles))
    val_tiles = balanced_tiles[:n_val]
    train_tiles = balanced_tiles[n_val:]

    base = os.path.join(cfg["data"]["root_dir"], cfg["data"]["dataset_dir"])
    train_paths = [os.path.join(base, fn) for fn in train_tiles]
    val_paths = [os.path.join(base, fn) for fn in val_tiles]
    test_paths = [os.path.join(base, fn) for fn in test_tiles]

    logger.info(f"Split: {len(train_paths)} train / {len(val_paths)} val / {len(test_paths)} test tiles")
    logger.info("Computing cls_labels and fg_pixel_fraction for train set (one-time cost)...")
    cls_labels, fg_pixel_fraction = _compute_cls_labels_and_fg_fraction(train_paths, no_data_value)
    logger.info(f"fg_pixel_fraction={fg_pixel_fraction:.6f}, "
                f"pos_tiles={sum(cls_labels)}/{len(cls_labels)}")

    return {
        "key": _split_key(cfg),
        "created": datetime.utcnow().isoformat(),
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "n_test": len(test_paths),
        "train_tiles": train_paths,
        "val_tiles": val_paths,
        "test_tiles": test_paths,
        "cls_labels_train": cls_labels,
        "fg_pixel_fraction": fg_pixel_fraction,
        "random_prune_subsets": {},
    }


def get_or_create_split(cfg: dict) -> dict:
    """Returns the canonical split dict, creating and caching to JSON on first call."""
    os.makedirs(SPLITS_DIR, exist_ok=True)
    key = _split_key(cfg)
    path = os.path.join(SPLITS_DIR, f"{key}.json")

    if os.path.exists(path):
        logger.info(f"Loading existing split from {path}")
        with open(path, "r") as f:
            return json.load(f)

    logger.info(f"Generating new split (key={key})")
    split = _generate_split(cfg)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
    logger.info(f"Split saved to {path}")
    return split


def get_or_save_random_subset(split: dict, prune_ratio: float, seed: int,
                               split_path: str) -> list:
    """Fixed random subset of train indices for RandomPruner; generated once, cached in split JSON."""
    key = f"ratio{prune_ratio}_seed{seed}"
    subsets = split.get("random_prune_subsets", {})
    if key in subsets:
        return subsets[key]

    n = split["n_train"]
    keep = int(n * (1 - prune_ratio))
    rng = np.random.RandomState(seed)
    indices = rng.choice(n, keep, replace=False).tolist()

    split.setdefault("random_prune_subsets", {})[key] = indices
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    logger.info(f"Random subset (ratio={prune_ratio}) saved: {keep}/{n} tiles")
    return indices


def get_split_path(cfg: dict) -> str:
    key = _split_key(cfg)
    return os.path.join(SPLITS_DIR, f"{key}.json")
