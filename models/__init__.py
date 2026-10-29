from .mask2former import build_mask2former
from .deeplabv3 import build_deeplabv3


def get_model(cfg: dict):
    """Factory: cfg is cfg['model'] from the YAML. Supported: mask2former, deeplabv3."""
    name = cfg.get("name", "mask2former")
    if name == "mask2former":
        return build_mask2former(cfg)
    elif name == "deeplabv3":
        return build_deeplabv3(cfg)
    else:
        raise ValueError(f"Unknown model: '{name}'. Supported: mask2former, deeplabv3")
