#!/usr/bin/env python3
"""
qualitative_comparison.py — render a 4-panel side-by-side test-tile comparison:
    Original (RGB) | Ground Truth | InfoBatch (r=R) | CS-InfoBatch (r=R)

Every eligible test tile is scored by
    gap = IoU(CS-InfoBatch) - IoU(InfoBatch)
and the top candidate(s) are rendered, so the example is reproducible rather than hand-picked.

Usage:
  python scripts/qualitative_comparison.py --ratio 0.8 --top 5
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils.tools import load_config, set_seed
from utils.split_cache import get_or_create_split
from data.utils import load_tile
from models import get_model

OUT_DIR = PROJECT_ROOT / "results_analysis" / "qualitative"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(exp_name: str, config_path: str):
    cfg = load_config(config_path)
    model = get_model(cfg["model"])
    ckpt = PROJECT_ROOT / "checkpoints" / exp_name / f"{exp_name}_best.pth"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def iou(pred, gt, valid):
    pred, gt = pred[valid], gt[valid]
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--top", type=int, default=5, help="number of tiles to render")
    ap.add_argument("--candidates", type=int, default=200,
                     help="test tiles (highest label pixel count) to actually score")
    args = ap.parse_args()

    r = args.ratio
    set_seed(42)

    cs_model = load_model(f"cs_infobatch_r{r}_cs_infobatch", "configs/cs_infobatch.yaml")
    ib_model = load_model(f"infobatch_r{r}_infobatch",       "configs/infobatch.yaml")

    base_cfg = load_config("configs/baseline.yaml")
    split = get_or_create_split(base_cfg)
    test_paths = split["test_tiles"]
    test_names = {Path(p).stem for p in test_paths}

    info = pd.read_csv(PROJECT_ROOT / "datasets" / "tile_info224_v3.csv")
    info = info[info["FileName"].str.replace(".tif", "", regex=False).isin(test_names)]
    info = info.sort_values("LabelSize", ascending=False).head(args.candidates)

    path_by_name = {Path(p).stem: p for p in test_paths}

    rows = []
    with torch.no_grad():
        for _, row in info.iterrows():
            name = row["FileName"].replace(".tif", "")
            path = path_by_name[name]
            img, label, no_data = load_tile(
                path, no_data_value=base_cfg["data"]["no_data_value"],
                normalize=base_cfg["data"]["normalize"],
            )
            x = torch.from_numpy(img).float().unsqueeze(0)
            valid = ~no_data
            gt = label.astype(bool)

            cs_pred = (torch.sigmoid(cs_model(x)).squeeze().numpy() > 0.5)
            ib_pred = (torch.sigmoid(ib_model(x)).squeeze().numpy() > 0.5)

            cs_iou = iou(cs_pred, gt, valid)
            ib_iou = iou(ib_pred, gt, valid)
            rows.append({"name": name, "path": path, "cs_iou": cs_iou, "ib_iou": ib_iou,
                          "gap": cs_iou - ib_iou, "fg_pixels": int(gt.sum())})

    scored = pd.DataFrame(rows).sort_values("gap", ascending=False)
    print(scored.head(15)[["name", "cs_iou", "ib_iou", "gap", "fg_pixels"]].to_string(index=False))

    gt_cmap = ListedColormap(["none", "#00FF00"])
    pr_cmap = ListedColormap(["none", "#FF6600"])

    for _, row in scored.head(args.top).iterrows():
        name, path = row["name"], row["path"]
        img, label, no_data = load_tile(
            path, no_data_value=base_cfg["data"]["no_data_value"],
            normalize=base_cfg["data"]["normalize"],
        )
        rgb = np.clip(np.transpose(img[:3], (1, 2, 0)), 0, 1)
        gt = label.astype(bool)
        x = torch.from_numpy(img).float().unsqueeze(0)
        with torch.no_grad():
            cs_pred = (torch.sigmoid(cs_model(x)).squeeze().numpy() > 0.5)
            ib_pred = (torch.sigmoid(ib_model(x)).squeeze().numpy() > 0.5)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.6),
                                  gridspec_kw={"wspace": 0.02})
        titles = ["Original (RGB)", "Ground Truth",
                  f"InfoBatch (r={r})", f"CS-InfoBatch (r={r})"]
        overlays = [None, gt, ib_pred, cs_pred]
        cmaps = [None, gt_cmap, pr_cmap, pr_cmap]

        for ax, title, ov, cmap in zip(axes, titles, overlays, cmaps):
            ax.imshow(rgb)
            if ov is not None:
                ax.imshow(ov, cmap=cmap, vmin=0, vmax=1, alpha=0.75)
            ax.set_title(title, fontsize=12, fontweight="bold", pad=4)
            ax.axis("off")

        fig.suptitle(f"Tile: {name}   |   IoU  InfoBatch={row['ib_iou']:.3f}  "
                      f"CS-InfoBatch={row['cs_iou']:.3f}", fontsize=11, y=0.995)
        handles = [plt.Rectangle((0, 0), 1, 1, color="#00FF00", label="Ground Truth"),
                   plt.Rectangle((0, 0), 1, 1, color="#FF6600", label="Prediction")]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True,
                    bbox_to_anchor=(0.5, 0.0), fontsize=9)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.08, wspace=0.02)
        out_path = OUT_DIR / f"qualitative_r{r}_{name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
