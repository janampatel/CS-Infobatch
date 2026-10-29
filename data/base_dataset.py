from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base for tile datasets. __getitem__ returns a dict with
    'image' (5,H,W) float, 'label' (H,W) long, 'no_data_mask' (H,W) bool,
    'cls_label' scalar long (1 if tile has any tree pixels)."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict: ...
