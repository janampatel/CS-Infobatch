import torch
from .base import PrunerBase


class NoPruner(PrunerBase):
    """Baseline — no pruning, full dataset, standard DataLoader."""

    needs_ib_sampler = False
    is_random = False

    def __init__(self, train_dataset):
        self._dataset = train_dataset

    @property
    def dataset(self):
        return self._dataset

    def update(self, per_sample_bce: torch.Tensor,
               per_sample_score: torch.Tensor = None) -> torch.Tensor:
        return per_sample_bce.mean()
