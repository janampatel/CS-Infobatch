import os
import yaml
import argparse
import logging
import random
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="CS-InfoBatch Tree Segmentation Pipeline")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--exp_id", type=str, default=None, help="Override experiment id")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--prune_ratio", type=float, default=None)
    parser.add_argument("--pruner_type", type=str, default=None,
                        choices=["none", "random", "infobatch", "cs_infobatch"])
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(args, cfg: dict) -> dict:
    if args.exp_id is not None:
        cfg["experiment"]["id"] = args.exp_id
    if args.gpu_id is not None:
        cfg["experiment"]["gpu_id"] = args.gpu_id
    if args.prune_ratio is not None:
        cfg["pruner"]["prune_ratio"] = args.prune_ratio
    if args.pruner_type is not None:
        cfg["pruner"]["type"] = args.pruner_type
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    return cfg


def setup_logging(log_dir: str, log_name: str, level=logging.INFO) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_name}.log")
    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(formatter)
    fh.setLevel(level)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level)

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_results(metrics: dict, exp_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    metrics = {k: float(v) if isinstance(v, (np.generic, np.ndarray)) else v
               for k, v in metrics.items()}
    out_path = os.path.join(output_dir, f"{exp_name}_metrics.yaml")
    with open(out_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)
    return out_path
