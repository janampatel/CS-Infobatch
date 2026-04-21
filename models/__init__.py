from .mask2former import build_mask2former


def get_model(cfg: dict):
    """
    Factory. cfg is cfg['model'] from the YAML.
    Currently supports: mask2former
    """
    name = cfg.get("name", "mask2former")
    if name == "mask2former":
        return build_mask2former(cfg)
    else:
        raise ValueError(f"Unknown model: '{name}'. Supported: mask2former")
