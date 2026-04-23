import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import load_tile, augment_tile


class TileDataset(Dataset):
    """
    Dataset over a pre-computed list of tile paths.
    Does NOT generate splits — paths come from split_cache.get_or_create_split().

    split: 'train' | 'val' | 'test'  — controls whether augmentation is applied.
    """

    def __init__(self, tile_paths: list, cfg: dict, split: str = "train"):
        self.paths = tile_paths
        self.cfg = cfg
        self.split = split

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        img, label, no_data_mask = load_tile(
            path,
            no_data_value=self.cfg["data"]["no_data_value"],
            normalize=self.cfg["data"]["normalize"],
        )

        if self.split == "train":
            img, label, no_data_mask = augment_tile(
                img, label, no_data_mask, self.cfg["data"]["augmentation"]
            )

        cls_label = torch.tensor(int((label > 0).any()), dtype=torch.long)
        return {
            "image": torch.from_numpy(img).float(),
            "label": torch.from_numpy(label).long(),
            "no_data_mask": torch.from_numpy(no_data_mask.astype(np.uint8)).bool(),
            "cls_label": cls_label,
        }
