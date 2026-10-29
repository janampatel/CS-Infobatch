"""
One-time generator for dataset_2/tile_info.csv.

Produces the same schema as datasets/tile_info224_v3.csv so that
utils/split_cache._generate_split works unchanged:
  FileName, FilePath, ImageRawID, ImageRawPath, State, ImageIndex,
  TileIndex, LabelSize, NoDataSize

LabelSize  = number of foreground pixels (GT value == 2)
NoDataSize = number of no-data pixels   (GT value == 0)

Usage (from repo root):
  python scripts/make_dataset2_info.py
"""
import csv
import os
import re
import sys

import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.utils import PAIRED_FG_VALUE  # noqa: E402

ROOT = "dataset_2"
RGB_DIR = os.path.join(ROOT, "rgb-tiles", "cf-s2-kbd-full-raster")
GT_DIR = os.path.join(ROOT, "gt-tiles", "raster")
OUT_CSV = os.path.join(ROOT, "tile_info.csv")
NO_DATA_VALUE = 0


def main():
    files = sorted(f for f in os.listdir(GT_DIR) if f.endswith(".tif"))
    rows = []
    for i, fn in enumerate(files):
        if i % 500 == 0:
            print(f"{i}/{len(files)}")
        with rasterio.open(os.path.join(GT_DIR, fn)) as src:
            gt = src.read(1)
        m = re.match(r"(.+)_patch_(\d+)\.tif$", fn)
        raw_id, tile_idx = (m.group(1), int(m.group(2))) if m else (fn, i)
        rows.append({
            "FileName": fn,
            "FilePath": os.path.join(RGB_DIR, fn),
            "ImageRawID": raw_id,
            "ImageRawPath": f"{raw_id}.tif",
            "State": "",
            "ImageIndex": 0,
            "TileIndex": tile_idx,
            "LabelSize": int((gt == PAIRED_FG_VALUE).sum()),
            "NoDataSize": int((gt == NO_DATA_VALUE).sum()),
        })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
