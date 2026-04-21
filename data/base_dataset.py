from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Abstract base for all tile datasets.
    Subclasses must set self.paths and self.split, and optionally self.cfg.

    __getitem__ returns:
      {
        'image':        FloatTensor (5, H, W),
        'label':        LongTensor  (H, W),
        'no_data_mask': BoolTensor  (H, W),
        'cls_label':    LongTensor  scalar — 1 if tile has any tree pixels
      }
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict: ...
