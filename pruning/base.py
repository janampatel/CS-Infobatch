from abc import ABC, abstractmethod
import torch


class PrunerBase(ABC):
    """Abstract base for all pruning strategies."""

    needs_ib_sampler: bool = False  # True for InfoBatch / CS-InfoBatch
    is_random: bool = False          # True for RandomPruner (uses Subset + shuffle)

    @abstractmethod
    def update(self, per_sample_bce: torch.Tensor,
               per_sample_score: torch.Tensor = None) -> torch.Tensor:
        """
        Called once per training batch.
        per_sample_bce: (B,) mean valid-pixel BCE loss per sample
        per_sample_score: (B,) class-balanced score for CS-IB; None otherwise

        Returns a scalar loss ready for .backward().
        """
        ...

    @property
    @abstractmethod
    def dataset(self):
        """The (possibly wrapped) training dataset passed to DataLoader."""
        ...
