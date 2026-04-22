"""
CS-InfoBatch pruner — our contribution.
Class-stratified InfoBatch with:
  - Separate pruning threshold per class (pos tiles vs neg tiles)
  - EMA-based scoring (not direct replacement)
  - Class-balanced score H(z_i) with w_fg >> w_bg
  - Warmup phase (no pruning), pruning phase (ramped ratio), annealing phase (full)
"""
import logging
import math
import numpy as np
import torch
from torch.utils.data import Sampler

from .base import PrunerBase
from .infobatch_pruner import (
    InfoBatchDataset, _apply_ib_monkey_patch, _IB_PATCH_APPLIED
)

logger = logging.getLogger(__name__)


class CSInfoBatchDataset(InfoBatchDataset):
    """
    Extends InfoBatchDataset with:
      - EMA score update (beta=0.7)
      - Class-stratified pruning thresholds (separate mean per class)
      - Scores initialised to 0.0 (not 3.0)
    """

    def __init__(self, dataset, cls_labels: list, num_epochs: int,
                 prune_ratio: float, delta: float = 0.875,
                 warmup_fraction: float = 0.1, ema_beta: float = 0.7,
                 fg_pixel_fraction: float = 0.003722):
        super().__init__(dataset, num_epochs, prune_ratio, delta)

        # Reinit scores to 0.0 for EMA to settle correctly
        n = len(dataset)
        self.scores = torch.zeros(n)

        self.cls_labels = torch.tensor(cls_labels, dtype=torch.bool)  # (N,)
        self.ema_beta = ema_beta

        warmup_stop = math.floor(warmup_fraction * num_epochs)
        prune_stop = math.floor(delta * num_epochs)
        self.warmup_stop = warmup_stop
        self.prune_stop = prune_stop

        f = fg_pixel_fraction
        self.w_fg = 1.0 / (2.0 * f)            # ~134 for f≈0.003722
        self.w_bg = 1.0 / (2.0 * (1.0 - f))    # ~0.502

        logger.info(f"CS-InfoBatch: w_fg={self.w_fg:.2f}, w_bg={self.w_bg:.4f}, "
                    f"warmup_stop={warmup_stop}, prune_stop={prune_stop}")

    def update(self, per_sample_bce: torch.Tensor,
               per_sample_score: torch.Tensor) -> torch.Tensor:
        """
        EMA update on scores using the class-balanced score.
        Gradient rescaling uses uniform per_sample_bce * weights.
        """
        indices = self.cur_batch_index
        score_cpu = per_sample_score.detach().cpu()
        self.scores[indices] = (self.ema_beta * self.scores[indices]
                                + (1.0 - self.ema_beta) * score_cpu)
        weights = self.weights[indices].to(per_sample_bce.device)
        return (per_sample_bce * weights).mean()

    def prune_stratified(self, effective_ratio: float) -> list:
        """
        Class-stratified pruning: separate threshold per class.
        Returns list of sample indices to use this epoch.
        """
        self.weights.fill_(1.0)
        remained = []

        pos_idx = self.cls_labels.nonzero(as_tuple=True)[0]   # positive tile indices
        neg_idx = (~self.cls_labels).nonzero(as_tuple=True)[0]  # negative tile indices

        for group_idx in [pos_idx, neg_idx]:
            if len(group_idx) == 0:
                continue
            group_scores = self.scores[group_idx]
            threshold = group_scores.mean().item()

            easy_mask = group_scores < threshold
            hard_mask = ~easy_mask
            easy = group_idx[easy_mask]
            hard = group_idx[hard_mask]

            # Keep all hard samples
            remained.extend(hard.tolist())

            # Keep (1 - effective_ratio) fraction of easy samples, upscale weights
            if len(easy) > 0:
                n_keep = max(1, int((1.0 - effective_ratio) * len(easy)))
                perm = torch.randperm(len(easy))
                kept_idx = easy[perm[:n_keep]]
                self.weights[kept_idx] = 1.0 / max(1.0 - effective_ratio, 1e-6)
                remained.extend(kept_idx.tolist())
                self.num_pruned_samples += len(easy) - n_keep

        # Shuffle the full remained list
        np.random.shuffle(remained)
        return remained

    def no_prune(self) -> list:
        self.weights.fill_(1.0)
        return torch.randperm(len(self.dataset)).tolist()


class CSIBSampler(Sampler):
    """
    IBSampler variant for CS-InfoBatch with phase logic:
      Phase 1 (epoch <= warmup_stop):  full dataset, no pruning
      Phase 2 (warmup < epoch <= prune_stop): ramped pruning
      Phase 3 (epoch > prune_stop):   full dataset, weights reset

    reset() must be called manually before each epoch.
    """

    def __init__(self, ib_dataset: CSInfoBatchDataset, prune_ratio: float):
        self.ib_dataset = ib_dataset
        self.prune_ratio = prune_ratio
        self.epoch = 0
        self.sample_indices: list = list(range(len(ib_dataset)))
        self.pruned_this_epoch: int = 0
        self._in_phase3 = False

    def reset(self):
        """Compute sample_indices for the coming epoch."""
        np.random.seed(self.epoch)
        n_before = len(self.ib_dataset)
        ws = self.ib_dataset.warmup_stop
        ps = self.ib_dataset.prune_stop

        if self.epoch < ws:
            # Phase 1: warmup — no pruning
            self.ib_dataset.weights.fill_(1.0)
            self.sample_indices = self.ib_dataset.no_prune()
            self.pruned_this_epoch = 0
            phase = 1
        elif self.epoch <= ps:
            # Phase 2: ramped pruning
            ramp = (self.epoch - ws) / max(ps - ws, 1)
            effective_ratio = self.prune_ratio * ramp
            self.sample_indices = self.ib_dataset.prune_stratified(effective_ratio)
            self.pruned_this_epoch = n_before - len(self.sample_indices)
            phase = 2
        else:
            # Phase 3: annealing — full dataset
            if not self._in_phase3:
                self.ib_dataset.weights.fill_(1.0)
                self._in_phase3 = True
            self.sample_indices = self.ib_dataset.no_prune()
            self.pruned_this_epoch = 0
            phase = 3

        self.epoch += 1
        logger.debug(f"CSIBSampler reset: epoch={self.epoch}, phase={phase}, "
                     f"samples={len(self.sample_indices)}, "
                     f"pruned={self.pruned_this_epoch}")

    def __iter__(self):
        return iter(self.sample_indices)

    def __len__(self) -> int:
        return len(self.sample_indices)


class CSInfoBatchPruner(PrunerBase):
    """Class-stratified InfoBatch pruner — our contribution."""

    needs_ib_sampler = True
    is_random = False

    def __init__(self, train_dataset, cls_labels: list, num_epochs: int,
                 prune_ratio: float, delta: float = 0.875,
                 warmup_fraction: float = 0.1, ema_beta: float = 0.7,
                 fg_pixel_fraction: float = 0.003722):
        _apply_ib_monkey_patch()
        self._ib_dataset = CSInfoBatchDataset(
            train_dataset, cls_labels, num_epochs, prune_ratio,
            delta, warmup_fraction, ema_beta, fg_pixel_fraction
        )
        self._sampler = CSIBSampler(self._ib_dataset, prune_ratio)
        self.prune_ratio = prune_ratio
        self.w_fg = self._ib_dataset.w_fg
        self.w_bg = self._ib_dataset.w_bg

    @property
    def dataset(self) -> CSInfoBatchDataset:
        return self._ib_dataset

    @property
    def sampler(self) -> CSIBSampler:
        return self._sampler

    def update(self, per_sample_bce: torch.Tensor,
               per_sample_score: torch.Tensor = None) -> torch.Tensor:
        if per_sample_score is None:
            per_sample_score = per_sample_bce
        return self._ib_dataset.update(per_sample_bce, per_sample_score)
