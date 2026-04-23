"""
CS-InfoBatch research pipeline entry point.

Usage:
  python main.py --config configs/baseline.yaml
  python main.py --config configs/cs_infobatch.yaml --prune_ratio 0.5 --exp_id csib_r05

Gate 1: baseline must reach class1_f1 ∈ [0.50, 0.54] before running pruning experiments.
"""
import os
import sys
import logging

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.tools import parse_args, load_config, apply_overrides, setup_logging, set_seed, save_results
from utils.gdrive_upload import upload_experiment
from data.dataloader import get_dataloader
from models import get_model
from training.trainer import train
from evaluation.evaluator import evaluate_model

logger = logging.getLogger(__name__)


def build_exp_name(cfg: dict) -> str:
    pruner_type = cfg.get("pruner", {}).get("type", "none")
    prune_ratio = cfg.get("pruner", {}).get("prune_ratio", "")
    exp_id = cfg["experiment"]["id"]
    if pruner_type == "none":
        return f"baseline_{exp_id}"
    return f"{pruner_type}_r{prune_ratio}_{exp_id}"


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = apply_overrides(args, cfg)

    exp_name = build_exp_name(cfg)
    log_dir  = cfg["logging"]["log_dir"]
    setup_logging(log_dir, exp_name)

    logger.info(f"=== Experiment: {exp_name} ===")
    logger.info(f"Config: {args.config}")
    logger.info(f"Pruner: {cfg['pruner']['type']}  "
                f"prune_ratio: {cfg['pruner'].get('prune_ratio', 'N/A')}")

    # Seed FIRST — before any dataset/model construction
    set_seed(cfg["experiment"]["seed"])

    # Data
    logger.info("Building dataloaders...")
    train_loader, val_loader, test_loader, pruner = get_dataloader(cfg)

    # Model
    logger.info("Building model...")
    model = get_model(cfg["model"])
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {n_params:.1f}M")

    # Train
    logger.info("Starting training...")
    train_stats = train(model, train_loader, val_loader, pruner, cfg, exp_name)
    logger.info(f"Training done — best val_class1_f1: {train_stats['best_val_class1_f1']:.4f}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    metrics = evaluate_model(model, test_loader, cfg, exp_name, train_stats)

    # Save results
    results_dir = os.path.join(cfg["output"]["results_dir"], exp_name)
    out_path = save_results(metrics, exp_name, results_dir)
    logger.info(f"Results saved: {out_path}")

    # Print summary
    logger.info("=== Final Results ===")
    for key in ["class1_f1", "class1_iou", "class1_recall", "class1_precision",
                "train_time_seconds", "total_samples_seen", "epochs_trained"]:
        if key in metrics:
            v = metrics[key]
            logger.info(f"  {key}: {v:.4f}" if isinstance(v, float) else f"  {key}: {v}")

    # Upload to Google Drive
    gdrive_cfg = cfg.get("gdrive", {})
    if gdrive_cfg.get("enabled", False):
        logger.info("Uploading results to Google Drive...")
        upload_experiment(exp_name, cfg, gdrive_cfg)
    else:
        logger.info("Google Drive upload disabled (gdrive.enabled=false).")

    return metrics


if __name__ == "__main__":
    main()
