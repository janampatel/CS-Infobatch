"""
EDA script for dataset_2 (paired RGB + GT tiles).

Run on the GPU server where rasterio is available:
  python scripts/inspect_dataset2.py

Outputs:
  - Console: key stats and config values to paste into YAML
  - datasets/tile_info_dataset2.csv  — required by split_cache
  - results_analysis/dataset2_eda/   — histogram, pie chart, CDF
"""
import os
import sys
import csv
import glob
import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
RGB_DIR = "datasets/rgb-tiles/cf-s2-kbd-full-raster"
GT_DIR  = "datasets/gt-tiles/raster"
OUT_CSV = "datasets/tile_info_dataset2.csv"
PLOT_DIR = "results_analysis/dataset2_eda"

GT_FG_VALUE   = 2   # pixel value treated as foreground
GT_BG_VALUE   = 1   # pixel value treated as background
GT_NODATA     = 0   # pixel value to exclude


def _try_import_plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", default=RGB_DIR)
    parser.add_argument("--gt-dir",  default=GT_DIR)
    parser.add_argument("--out-csv", default=OUT_CSV)
    parser.add_argument("--plot-dir", default=PLOT_DIR)
    args = parser.parse_args()

    try:
        import rasterio
    except ImportError:
        logger.error("rasterio not installed. Run on the GPU server.")
        sys.exit(1)

    rgb_files = sorted(glob.glob(os.path.join(args.rgb_dir, "*.tif")))
    if not rgb_files:
        logger.error(f"No .tif files found in {args.rgb_dir}")
        sys.exit(1)

    logger.info(f"Found {len(rgb_files)} RGB tiles in {args.rgb_dir}")

    # Per-tile statistics
    rows = []          # for tile_info CSV
    fg_fracs = []      # fg / (fg + bg) per tile
    tile_has_fg = []   # bool per tile
    n_bands_set = set()

    total_fg_px = 0
    total_bg_px = 0
    total_nd_px = 0
    bad_pairs    = 0

    for i, rgb_path in enumerate(rgb_files):
        fname = os.path.basename(rgb_path)
        gt_path = os.path.join(args.gt_dir, fname)

        if not os.path.exists(gt_path):
            logger.warning(f"No GT for {fname} — skipping")
            bad_pairs += 1
            continue

        with rasterio.open(rgb_path) as src:
            n_bands_set.add(src.count)

        with rasterio.open(gt_path) as src:
            raw = src.read(1).astype(np.int32)

        fg = int((raw == GT_FG_VALUE).sum())
        bg = int((raw == GT_BG_VALUE).sum())
        nd = int((raw == GT_NODATA).sum())
        valid = fg + bg

        total_fg_px += fg
        total_bg_px += bg
        total_nd_px += nd

        frac = fg / max(valid, 1)
        fg_fracs.append(frac)
        tile_has_fg.append(fg > 0)

        rows.append({
            "FileName":   fname,
            "LabelSize":  fg,           # matches split_cache expectation
            "FGPixels":   fg,
            "BGPixels":   bg,
            "NDPixels":   nd,
            "FGFraction": round(frac, 6),
            "ImageRawPath": rgb_path,   # split_cache may read this col
        })

        if (i + 1) % 200 == 0:
            logger.info(f"  processed {i+1}/{len(rgb_files)}")

    # Aggregate stats
    n_paired = len(rows)
    n_pos_tiles = sum(tile_has_fg)
    fg_pixel_fraction = float(np.mean(fg_fracs))
    total_valid = total_fg_px + total_bg_px
    pixel_ratio  = total_bg_px / max(total_fg_px, 1)
    tile_pos_rate = n_pos_tiles / max(n_paired, 1)

    w_fg = 1.0 / (2.0 * fg_pixel_fraction)
    w_bg = 1.0 / (2.0 * (1.0 - fg_pixel_fraction))

    logger.info("=" * 60)
    logger.info("DATASET_2 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total tiles (paired)   : {n_paired}")
    logger.info(f"  Unpaired (missing GT)  : {bad_pairs}")
    logger.info(f"  RGB band counts seen   : {n_bands_set}")
    logger.info(f"  Tiles with foreground  : {n_pos_tiles}/{n_paired}  "
                f"({100*tile_pos_rate:.1f}%)")
    logger.info(f"  Total valid pixels     : {total_valid:,}")
    logger.info(f"  Total FG pixels        : {total_fg_px:,}")
    logger.info(f"  Total BG pixels        : {total_bg_px:,}")
    logger.info(f"  Total no-data pixels   : {total_nd_px:,}")
    logger.info(f"  Pixel imbalance (BG:FG): {pixel_ratio:.1f}:1")
    logger.info(f"  fg_pixel_fraction      : {fg_pixel_fraction:.6f}")
    logger.info(f"  CS-IB w_fg             : {w_fg:.2f}")
    logger.info(f"  CS-IB w_bg             : {w_bg:.4f}")
    logger.info("")
    logger.info("Paste these values into your dataset_2 YAML configs:")
    logger.info(f"  pruner.fg_pixel_fraction : {fg_pixel_fraction:.6f}")
    logger.info(f"  training.criterion.w_pos : {pixel_ratio:.1f}  "
                "(or 0 to rely on Dice alone)")

    # Compare with TreeFinder
    logger.info("")
    logger.info("TreeFinder reference:")
    logger.info("  pixel_ratio      : ~268:1")
    logger.info("  fg_pixel_fraction: 0.003722")
    logger.info("  CS-IB w_fg       : ~134")

    # Write tile_info CSV
    os.makedirs(os.path.dirname(args.out_csv) if os.path.dirname(args.out_csv) else ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"\ntile_info CSV written to: {args.out_csv}")
    logger.info(f"  (use data.dataset_info: tile_info_dataset2.csv in your YAML)")

    # Plots
    plt = _try_import_plot()
    if plt is None:
        return

    os.makedirs(args.plot_dir, exist_ok=True)
    fg_arr = np.array(fg_fracs)

    # 1. Histogram of per-tile fg fraction
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(fg_arr[fg_arr > 0], bins=50, color="#2ca02c", edgecolor="white", alpha=0.85)
    ax.axvline(fg_pixel_fraction, color="crimson", linewidth=1.5, linestyle="--",
               label=f"mean fg_frac = {fg_pixel_fraction:.4f}")
    ax.set_xlabel("FG pixel fraction per tile (FG tiles only)")
    ax.set_ylabel("Count")
    ax.set_title("Dataset 2 — Per-tile FG fraction distribution")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(args.plot_dir, "fig1_fg_fraction_hist.png")
    fig.savefig(path, dpi=150)
    logger.info(f"Saved: {path}")
    plt.close(fig)

    # 2. Pie: tile positivity
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Left: tile positivity
    axes[0].pie(
        [n_pos_tiles, n_paired - n_pos_tiles],
        labels=[f"Has FG\n({n_pos_tiles})", f"BG only\n({n_paired-n_pos_tiles})"],
        colors=["#2ca02c", "#d62728"],
        autopct="%1.1f%%", startangle=90,
    )
    axes[0].set_title("Tile positivity")
    # Right: pixel distribution
    axes[1].pie(
        [total_fg_px, total_bg_px, total_nd_px],
        labels=[f"FG\n({total_fg_px:,})", f"BG\n({total_bg_px:,})", f"No-data\n({total_nd_px:,})"],
        colors=["#2ca02c", "#aec7e8", "#c7c7c7"],
        autopct="%1.2f%%", startangle=90,
    )
    axes[1].set_title("Pixel distribution")
    fig.suptitle(f"Dataset 2 — Class Imbalance  (BG:FG = {pixel_ratio:.1f}:1)")
    fig.tight_layout()
    path = os.path.join(args.plot_dir, "fig2_class_imbalance_pie.png")
    fig.savefig(path, dpi=150)
    logger.info(f"Saved: {path}")
    plt.close(fig)

    # 3. CDF of per-tile fg fraction
    sorted_fg = np.sort(fg_arr)
    cdf = np.arange(1, len(sorted_fg) + 1) / len(sorted_fg)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sorted_fg, cdf, color="#1f77b4")
    ax.axvline(fg_pixel_fraction, color="crimson", linestyle="--",
               label=f"mean = {fg_pixel_fraction:.4f}")
    ax.set_xlabel("FG pixel fraction")
    ax.set_ylabel("CDF")
    ax.set_title("Dataset 2 — CDF of per-tile FG fraction")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(args.plot_dir, "fig3_fg_fraction_cdf.png")
    fig.savefig(path, dpi=150)
    logger.info(f"Saved: {path}")
    plt.close(fig)

    # 4. Comparison bar: dataset2 vs TreeFinder
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    datasets = ["TreeFinder", "Dataset 2"]
    pixel_ratios = [268.0, pixel_ratio]
    fg_fracs_comp = [0.003722, fg_pixel_fraction]
    w_fgs = [1.0 / (2.0 * f) for f in fg_fracs_comp]

    axes[0].bar(datasets, pixel_ratios, color=["#1f77b4", "#ff7f0e"])
    axes[0].set_ylabel("BG:FG pixel ratio")
    axes[0].set_title("Pixel Imbalance Ratio")
    for i, v in enumerate(pixel_ratios):
        axes[0].text(i, v + 1, f"{v:.1f}:1", ha="center", fontsize=10, fontweight="bold")

    axes[1].bar(datasets, w_fgs, color=["#1f77b4", "#ff7f0e"])
    axes[1].set_ylabel("CS-IB w_fg")
    axes[1].set_title("CS-InfoBatch FG Weight")
    for i, v in enumerate(w_fgs):
        axes[1].text(i, v + 1, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Dataset Comparison: TreeFinder vs Dataset 2")
    fig.tight_layout()
    path = os.path.join(args.plot_dir, "fig4_dataset_comparison.png")
    fig.savefig(path, dpi=150)
    logger.info(f"Saved: {path}")
    plt.close(fig)

    logger.info(f"\nAll plots saved to {args.plot_dir}/")


if __name__ == "__main__":
    main()
