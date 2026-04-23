from .base import PrunerBase
from .no_pruner import NoPruner
from .random_pruner import RandomPruner
from .infobatch_pruner import InfoBatchPruner
from .cs_infobatch_pruner import CSInfoBatchPruner


def get_pruner(cfg: dict, train_dataset, split: dict,
               split_path: str = None) -> PrunerBase:
    """
    Factory function. Returns the appropriate pruner based on cfg['pruner']['type'].
    split: the full split dict from split_cache (for cls_labels and fg_pixel_fraction)
    split_path: path to split JSON (for updating random_prune_subsets in-place)
    """
    p_cfg = cfg.get("pruner", {})
    pruner_type = p_cfg.get("type", "none")
    num_epochs = cfg["training"]["epochs"]
    prune_ratio = float(p_cfg.get("prune_ratio", 0.5))
    seed = cfg["experiment"]["seed"]

    if pruner_type == "none":
        return NoPruner(train_dataset)

    elif pruner_type == "random":
        return RandomPruner(
            train_dataset, prune_ratio, seed, split, split_path
        )

    elif pruner_type == "infobatch":
        delta = float(p_cfg.get("delta", 0.875))
        return InfoBatchPruner(train_dataset, num_epochs, prune_ratio, delta)

    elif pruner_type == "cs_infobatch":
        cls_labels = split["cls_labels_train"]
        fg_pixel_fraction = split.get("fg_pixel_fraction",
                                      float(p_cfg.get("fg_pixel_fraction", 0.003722)))
        delta = float(p_cfg.get("delta", 0.875))
        warmup_fraction = float(p_cfg.get("warmup_fraction", 0.1))
        ema_beta = float(p_cfg.get("ema_beta", 0.7))
        return CSInfoBatchPruner(
            train_dataset, cls_labels, num_epochs, prune_ratio,
            delta, warmup_fraction, ema_beta, fg_pixel_fraction
        )

    else:
        raise ValueError(f"Unknown pruner type: '{pruner_type}'")
