"""
Public API for data loading.
Returns (train_loader, val_loader, test_loader, pruner).
All callers get the same canonical split — loaded from splits/{key}.json.
"""
import logging
from torch.utils.data import DataLoader

from .random_split import TileDataset
from utils.split_cache import get_or_create_split, get_split_path
from pruning import get_pruner

logger = logging.getLogger(__name__)


def get_dataloader(cfg: dict):
    """
    Returns: (train_loader, val_loader, test_loader, pruner)

    The split is loaded from (or created and saved to) splits/{key}.json.
    Pruner wraps the train dataset as needed.
    """
    split = get_or_create_split(cfg)
    split_path = get_split_path(cfg)

    train_ds = TileDataset(split["train_tiles"], cfg, split="train")
    val_ds   = TileDataset(split["val_tiles"],   cfg, split="val")
    test_ds  = TileDataset(split["test_tiles"],  cfg, split="test")

    logger.info(f"Dataset sizes — train: {len(train_ds)}, "
                f"val: {len(val_ds)}, test: {len(test_ds)}")

    pruner = get_pruner(cfg, train_ds, split, split_path)

    bs = cfg["training"]["batch_size"]
    nw = cfg["experiment"].get("num_workers", 4)
    persistent = nw > 0

    if pruner.needs_ib_sampler:
        train_loader = DataLoader(
            pruner.dataset,
            batch_size=bs,
            sampler=pruner.sampler,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=persistent,
            drop_last=True,
            collate_fn=_ib_collate_fn,
        )
    elif pruner.is_random:
        train_loader = DataLoader(
            pruner.dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=persistent,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            pruner.dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=persistent,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader, pruner


def _ib_collate_fn(batch):
    """
    Custom collate for InfoBatch datasets.
    Each item from InfoBatchDataset.__getitem__ is (index, sample_dict).
    We collate into (stacked_indices, batched_dict).
    The monkey-patched DataLoader.__next__ then unpacks this.
    """
    import torch
    indices = torch.tensor([item[0] for item in batch], dtype=torch.long)
    dicts   = [item[1] for item in batch]

    import torch.utils.data.dataloader as dl
    batched = dl.default_collate(dicts)
    return indices, batched
