"""
Tile loading and augmentation utilities.
Rewritten from treefinder-main/data_loader/utils.py — identical logic, no imports from that repo.
"""
import numpy as np
import rasterio


def load_tile(path: str, no_data_value: int = 255, normalize: bool = True):
    """
    Load a 5-band NAIP GeoTIFF tile.

    File bands: [R, G, B, NIR, Label]
    Returns:
      img: np.float32 (5, H, W)  — [R, G, B, NIR, NDVI]
      label: np.int64  (H, W)    — 0 or 1 (clipped; 255 excluded via no_data_mask)
      no_data_mask: np.bool_  (H, W) — True where label == no_data_value
    """
    with rasterio.open(path) as src:
        data = src.read()  # (bands, H, W)

    r, g, b, nir = data[0].astype(np.float32), data[1].astype(np.float32), \
                   data[2].astype(np.float32), data[3].astype(np.float32)
    raw_label = data[4]

    no_data_mask = (raw_label == no_data_value)
    label = np.clip(raw_label, 0, 1).astype(np.int64)

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
