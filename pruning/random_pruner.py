import json
import logging
import torch
import numpy as np
from torch.utils.data import Subset
from .base import PrunerBase

logger = logging.getLogger(__name__)


class RandomPruner(PrunerBase):
    """Fixed random subset, selected once before training, never changes."""

    needs_ib_sampler = False
    is_random = True

    def __init__(self, train_dataset, prune_ratio: float, seed: int,
                 split: dict, split_path: str):
        key = f"ratio{prune_ratio}_seed{seed}"
        subsets = split.get("random_prune_subsets", {})

        if key in subsets:
            indices = subsets[key]
            logger.info(f"Loaded random subset (key={key}): {len(indices)} tiles")
        else:
            n = len(train_dataset)
            keep = int(n * (1 - prune_ratio))
            rng = np.random.RandomState(seed)
            indices = rng.choice(n, keep, replace=False).tolist()
            split.setdefault("random_prune_subsets", {})[key] = indices
            with open(split_path, "w") as f:
                json.dump(split, f, indent=2)
            logger.info(f"Generated random subset (ratio={prune_ratio}): {keep}/{n} tiles")

        self._dataset = Subset(train_dataset, indices)
        self.prune_ratio = prune_ratio
        self.n_kept = len(indices)

    @property
    def dataset(self):
        return self._dataset

    def update(self, per_sample_bce: torch.Tensor,
               per_sample_score: torch.Tensor = None) -> torch.Tensor:
        return per_sample_bce.mean()
