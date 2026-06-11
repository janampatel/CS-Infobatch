"""
Tile loading and augmentation utilities.
Rewritten from treefinder-main/data_loader/utils.py — identical logic, no imports from that repo.
"""
import numpy as np
import rasterio

# dataset_2 layout: image and label live in separate, identically-named files.
# RGB tile:  dataset_2/rgb-tiles/cf-s2-kbd-full-raster/img1_patch_N.tif  (4 bands float32: R, G, B, NIR)
# GT tile:   dataset_2/gt-tiles/raster/img1_patch_N.tif                  (1 band uint8: 1 / 2)
PAIRED_RGB_DIR = "rgb-tiles/cf-s2-kbd-full-raster"
PAIRED_GT_DIR = "gt-tiles/raster"
PAIRED_FG_VALUE = 2          # GT pixel value treated as foreground (class 1)
PAIRED_REFLECTANCE_SCALE = 10000.0  # Sentinel-2 reflectance -> [0, 1]


def gt_path_for(path: str) -> str:
    """Map a dataset_2 RGB tile path to its ground-truth tile path."""
    return path.replace(PAIRED_RGB_DIR, PAIRED_GT_DIR)


def is_paired_tile(path: str) -> bool:
    return PAIRED_RGB_DIR in path


def load_label_band(path: str, no_data_value: int):
    """
    Read a tile's label as binary {0, 1} plus its valid-pixel mask,
    for either dataset layout. Used by split_cache for cls_labels.
    """
    if is_paired_tile(path):
        with rasterio.open(gt_path_for(path)) as src:
            raw = src.read(1)
        valid = raw != no_data_value
        label = ((raw == PAIRED_FG_VALUE) & valid).astype(np.int64)
    else:
        with rasterio.open(path) as src:
            raw = src.read(5)  # 5th band is the label
        valid = raw != no_data_value
        label = ((raw == 1) & valid).astype(np.int64)
    return label, valid


def _load_paired_tile(path: str, no_data_value: int, normalize: bool):
    """
    dataset_2 loader. Returns the same (img, label, no_data_mask) contract
    as load_tile: img is float32 (5, H, W) = [R, G, B, NIR, NDVI].
    """
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    r, g, b, nir = data[0], data[1], data[2], data[3]

    with rasterio.open(gt_path_for(path)) as src:
        raw_label = src.read(1)

    no_data_mask = (raw_label == no_data_value)
    label = (raw_label == PAIRED_FG_VALUE).astype(np.int64)
    label[no_data_mask] = 0

    ndvi = (nir - r) / (nir + r + 1e-10)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    if normalize:
        r = np.clip(r / PAIRED_REFLECTANCE_SCALE, 0.0, 1.0)
        g = np.clip(g / PAIRED_REFLECTANCE_SCALE, 0.0, 1.0)
        b = np.clip(b / PAIRED_REFLECTANCE_SCALE, 0.0, 1.0)
        nir = np.clip(nir / PAIRED_REFLECTANCE_SCALE, 0.0, 1.0)
        ndvi = (ndvi + 1.0) / 2.0  # [-1,1] → [0,1]

    img = np.stack([r, g, b, nir, ndvi], axis=0).astype(np.float32)
    return img, label, no_data_mask


def load_tile(path: str, no_data_value: int = 255, normalize: bool = True):
    """
    Load a 5-band NAIP GeoTIFF tile.

    File bands: [R, G, B, NIR, Label]
    Returns:
      img: np.float32 (5, H, W)  — [R, G, B, NIR, NDVI]
      label: np.int64  (H, W)    — 0 or 1 (clipped; 255 excluded via no_data_mask)
      no_data_mask: np.bool_  (H, W) — True where label == no_data_value

    dataset_2 tiles (paired RGB/GT files) are dispatched to _load_paired_tile,
    which returns the exact same contract.
    """
    if is_paired_tile(path):
        return _load_paired_tile(path, no_data_value, normalize)

    with rasterio.open(path) as src:
        data = src.read()  # (bands, H, W)

    r, g, b, nir = data[0].astype(np.float32), data[1].astype(np.float32), \
                   data[2].astype(np.float32), data[3].astype(np.float32)
    raw_label = data[4]

    no_data_mask = (raw_label == no_data_value)
    label = raw_label.copy().astype(np.int64)
    label[no_data_mask] = 0  # map 255 → 0, not 1

    ndvi = (nir - r) / (nir + r + 1e-10)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    if normalize:
        r   = r   / 255.0
        g   = g   / 255.0
        b   = b   / 255.0
        nir = nir / 255.0
        ndvi = (ndvi + 1.0) / 2.0  # [-1,1] → [0,1]

    img = np.stack([r, g, b, nir, ndvi], axis=0).astype(np.float32)
    return img, label, no_data_mask


def augment_tile(img: np.ndarray, label: np.ndarray, no_data_mask: np.ndarray,
                 aug_cfg: dict):
    """
    Apply random augmentation to a tile. Identical to treefinder augment_tile.

    img: (C, H, W), label: (H, W), no_data_mask: (H, W)
    aug_cfg: {'random_flip': bool, 'rotation': {'type': '90'}}
    """
    if aug_cfg.get("random_flip", False):
        if np.random.rand() < 0.5:  # horizontal flip
            img = np.flip(img, axis=2).copy()
            label = np.flip(label, axis=1).copy()
            no_data_mask = np.flip(no_data_mask, axis=1).copy()
        if np.random.rand() < 0.5:  # vertical flip
            img = np.flip(img, axis=1).copy()
            label = np.flip(label, axis=0).copy()
            no_data_mask = np.flip(no_data_mask, axis=0).copy()

    rot_cfg = aug_cfg.get("rotation", {})
    if rot_cfg.get("type") == "90":
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k, axes=(1, 2)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()
            no_data_mask = np.rot90(no_data_mask, k, axes=(0, 1)).copy()

    return img, label, no_data_mask
