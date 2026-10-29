"""Full test-set evaluation."""
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def evaluate_model(model: torch.nn.Module, test_loader,
                   cfg: dict, exp_name: str,
                   train_stats: dict = None) -> dict:
    """Evaluate on the full test set. Loads best checkpoint automatically."""
    gpu_id = cfg["experiment"].get("gpu_id", 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load best checkpoint
    ckpt_path = Path(cfg["output"]["checkpoint_dir"]) / exp_name / f"{exp_name}_best.pth"
    if ckpt_path.exists():
        logger.info(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        logger.warning(f"No checkpoint at {ckpt_path}; evaluating current weights")

    model.to(device)
    model.eval()

    metric_names = cfg["evaluation"].get("metrics", [])

    # Accumulate per-class TP/FP/FN/TN
    stats = {
        0: {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        1: {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "correct": 0,
        "total": 0,
    }

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            imgs    = batch["image"].to(device)
            gt      = batch["label"].unsqueeze(1).cpu().numpy().astype(np.uint8)
            no_data = batch["no_data_mask"].unsqueeze(1).cpu().numpy()

            outputs = model(imgs)
            probs   = torch.sigmoid(outputs).cpu().numpy()
            preds   = (probs > 0.5).astype(np.uint8)

            B = preds.shape[0]
            for i in range(B):
                pred_i = preds[i, 0]
                gt_i   = gt[i, 0]
                valid  = ~no_data[i, 0]
                p_flat = pred_i[valid]
                g_flat = gt_i[valid]

                for c in [0, 1]:
                    stats[c]["tp"] += int(np.logical_and(p_flat == c, g_flat == c).sum())
                    stats[c]["fp"] += int(np.logical_and(p_flat == c, g_flat != c).sum())
                    stats[c]["fn"] += int(np.logical_and(p_flat != c, g_flat == c).sum())
                    stats[c]["tn"] += int(np.logical_and(p_flat != c, g_flat != c).sum())

                stats["correct"] += int((p_flat == g_flat).sum())
                stats["total"]   += int(p_flat.size)

    def safe_div(a, b):
        return float(a) / (float(b) + 1e-6)

    results = {}
    for c in [0, 1]:
        tp, fp, fn = stats[c]["tp"], stats[c]["fp"], stats[c]["fn"]
        if "precision" in metric_names:
            results[f"class{c}_precision"] = safe_div(tp, tp + fp)
        if "recall" in metric_names:
            results[f"class{c}_recall"] = safe_div(tp, tp + fn)
        if "f1" in metric_names:
            p = safe_div(tp, tp + fp)
            r = safe_div(tp, tp + fn)
            results[f"class{c}_f1"] = safe_div(2 * p * r, p + r)
        if "iou" in metric_names:
            results[f"class{c}_iou"] = safe_div(tp, tp + fp + fn)

    if "iou" in metric_names:
        results["res_iou_macro"] = 0.5 * (results["class0_iou"] + results["class1_iou"])
    if "f1" in metric_names:
        results["res_f1_macro"] = 0.5 * (results["class0_f1"] + results["class1_f1"])
    if "precision" in metric_names:
        results["res_precision_macro"] = 0.5 * (
            results["class0_precision"] + results["class1_precision"]
        )
    if "recall" in metric_names:
        results["res_recall_macro"] = 0.5 * (
            results["class0_recall"] + results["class1_recall"]
        )
    if "accuracy" in metric_names:
        results["overall_accuracy"] = safe_div(stats["correct"], stats["total"])

    # Merge efficiency stats from trainer
    if train_stats:
        results.update(train_stats)

    logger.info("Evaluation results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    return results
