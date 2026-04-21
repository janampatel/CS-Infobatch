"""
Vanilla InfoBatch pruner.
Rewritten from Infobatch-master/InfoBatch-master/infobatch/infobatch.py.
Key differences from original:
  - IBSampler.__iter__ does NOT call reset() — reset is called manually in the training loop
  - update() signature accepts (per_sample_bce, per_sample_score=None)
  - Only BCE is used for scoring (not class-balanced score)
"""
import logging
import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data.dataloader import _BaseDataLoaderIter

from .base import PrunerBase

logger = logging.getLogger(__name__)

_IB_PATCH_APPLIED = False


def _apply_ib_monkey_patch():
    """
    Patches _BaseDataLoaderIter.__next__ to intercept InfoBatch datasets.
    The patched __next__ unpacks (indices, data) when the dataset is an
    InfoBatchDataset, calls set_active_indices(), then returns just data.
    Applied once at import time when an IB pruner is created.
    """
    global _IB_PATCH_APPLIED
    if _IB_PATCH_APPLIED:
        return

    def _ib_next(self):
        if self._sampler_iter is None:
            self._reset()
        data = self._next_data()
        if isinstance(self._dataset, InfoBatchDataset):
            indices, data = data
            self._dataset.set_active_indices(indices)
        self._num_yielded += 1
        return data

    _BaseDataLoaderIter.__next__ = _ib_next
    _IB_PATCH_APPLIED = True
    logger.debug("InfoBatch _BaseDataLoaderIter monkey-patch applied")


class InfoBatchDataset:
    """
    Wraps a TileDataset to return (index, sample_dict) from __getitem__.
    DataLoader collates this into (indices_tensor, dict_of_batched_tensors).
    The monkey-patched __next__ unpacks this and calls set_active_indices().
    """

    def __init__(self, dataset, num_epochs: int, prune_ratio: float = 0.5,
                 delta: float = 0.875):
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.keep_ratio = min(1.0, max(0.1, 1.0 - prune_ratio))
        self.delta = delta
        n = len(dataset)
        self.scores = torch.ones(n) * 3.0   # init high — all samples trained early
        self.weights = torch.ones(n)
        self.num_pruned_samples = 0
        self.cur_batch_index: torch.Tensor = None

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return idx, self.dataset[idx]

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    # ------------------------------------------------------------------
    # InfoBatch API
    # ------------------------------------------------------------------

    def set_active_indices(self, indices: torch.Tensor):
        self.cur_batch_index = indices

    def update(self, per_sample_bce: torch.Tensor,
               per_sample_score: torch.Tensor = None) -> torch.Tensor:
        """
        Update per-sample scores and apply gradient rescaling.
        per_sample_bce: (B,) — used for both scoring and gradient path
        Returns: weighted mean scalar for loss.backward()
        """
        indices = self.cur_batch_index
        # Direct score replacement (no EMA) — as in original InfoBatch
        self.scores[indices] = per_sample_bce.detach().cpu()
        weights = self.weights[indices].to(per_sample_bce.device)
        return (per_sample_bce * weights).mean()

    def prune(self) -> list:
        """Select samples for the next epoch. Returns list of indices to train on."""
        mean_score = self.scores.mean().item()
        well_learned = (self.scores < mean_score).nonzero(as_tuple=True)[0]
        hard = (self.scores >= mean_score).nonzero(as_tuple=True)[0]

        # Reset all weights before selecting subset
        self.weights.fill_(1.0)

        if len(well_learned) > 0:
            n_keep = max(1, int(self.keep_ratio * len(well_learned)))
            perm = torch.randperm(len(well_learned))
            kept_mask = perm[:n_keep]
            kept_indices = well_learned[kept_mask]
            self.weights[kept_indices] = 1.0 / self.keep_ratio
            self.num_pruned_samples += len(well_learned) - n_keep
            remained = torch.cat([hard, kept_indices])
        else:
            remained = hard

        perm = torch.randperm(len(remained))
        return remained[perm].tolist()

    def no_prune(self) -> list:
        """Return all indices shuffled (used after pruning phase ends)."""
        self.weights.fill_(1.0)
        return torch.randperm(len(self.dataset)).tolist()

    @property
    def stop_prune_epoch(self) -> float:
        return self.num_epochs * self.delta


class IBSampler(Sampler):
    """
    Custom sampler for InfoBatch.

    IMPORTANT: __iter__ does NOT call reset(). The training loop must call
    sampler.reset() manually BEFORE iter(train_loader) each epoch.
    This prevents double-firing of prune() per epoch.
    """

    def __init__(self, ib_dataset: InfoBatchDataset):
        self.ib_dataset = ib_dataset
        self.epoch = 0
        self.sample_indices: list = []
        self.pruned_this_epoch: int = 0
        # Initialise with full dataset so len() is valid before first reset()
        self.sample_indices = list(range(len(ib_dataset)))

    def reset(self):
        """Compute sample indices for the coming epoch. Called once per epoch."""
        np.random.seed(self.epoch)  # reproducible pruning decisions
        n_before = len(self.ib_dataset)

        if self.epoch >= self.ib_dataset.stop_prune_epoch:
            # Phase 3: full dataset, weights already 1.0
            if self.epoch == int(self.ib_dataset.stop_prune_epoch) + 1:
                self.ib_dataset.weights.fill_(1.0)
            self.sample_indices = self.ib_dataset.no_prune()
            self.pruned_this_epoch = 0
        else:
            self.sample_indices = self.ib_dataset.prune()
            self.pruned_this_epoch = n_before - len(self.sample_indices)

        self.epoch += 1
        logger.debug(f"IBSampler reset: epoch={self.epoch}, "
                     f"samples={len(self.sample_indices)}, "
                     f"pruned={self.pruned_this_epoch}")

    def __iter__(self):
        # Do NOT call reset() here — training loop handles it
        return iter(self.sample_indices)

    def __len__(self) -> int:
        return len(self.sample_indices)


class InfoBatchPruner(PrunerBase):
    """Vanilla InfoBatch pruner wrapper."""

    needs_ib_sampler = True
    is_random = False

    def __init__(self, train_dataset, num_epochs: int, prune_ratio: float,
                 delta: float = 0.875):
        _apply_ib_monkey_patch()
        self._ib_dataset = InfoBatchDataset(train_dataset, num_epochs, prune_ratio, delta)
        self._sampler = IBSampler(self._ib_dataset)
        self.prune_ratio = prune_ratio

    @property
    def dataset(self) -> InfoBatchDataset:
        return self._ib_dataset

    @property
    def sampler(self) -> IBSampler:
        return self._sampler

    def update(self, per_sample_bce: torch.Tensor,
               per_sample_score: torch.Tensor = None) -> torch.Tensor:
        return self._ib_dataset.update(per_sample_bce, per_sample_score)
