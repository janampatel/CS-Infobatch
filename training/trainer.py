"""
Core training loop for CS-InfoBatch experiments.
Supports all four pruner modes: none, random, infobatch, cs_infobatch.
"""
import csv
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from pruning import InfoBatchPruner, CSInfoBatchPruner
from training.losses import per_sample_bce, per_sample_cs_score, DiceLoss
from training.metrics import compute_val_class1_f1

logger = logging.getLogger(__name__)


def train(model: nn.Module, train_loader, val_loader,
          pruner, cfg: dict, exp_name: str) -> dict:
    """
    Train the model.

    Returns dict with 'best_val_class1_f1' and efficiency stats.
    """
    gpu_id = cfg["experiment"].get("gpu_id", 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Training on {device}")

    tr_cfg = cfg["training"]
    num_epochs = tr_cfg["epochs"]
    w_dice = float(tr_cfg.get("w_dice", 1.0))
    log_interval = cfg["logging"].get("log_interval", 50)

    use_ib = isinstance(pruner, (InfoBatchPruner, CSInfoBatchPruner))
    use_cs_ib = isinstance(pruner, CSInfoBatchPruner)

    # ------------------------------------------------------------------ #
    # Optimiser & Scheduler
    # ------------------------------------------------------------------ #
    optimizer = getattr(optim, tr_cfg["optimizer"]["type"])(
        model.parameters(),
        lr=float(tr_cfg["learning_rate"]),
        weight_decay=float(tr_cfg["optimizer"].get("weight_decay", 0.0)),
    )

    scheduler = None
    sc = tr_cfg.get("scheduler", {})
    if sc:
        stype = sc.get("type", "ExponentialLR")
        if stype == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=float(sc.get("gamma", 0.95))
            )
        elif stype == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(sc.get("step_size", 10)),
                gamma=float(sc.get("gamma", 0.1)),
            )
        elif stype == "CosineAnnealingWarmRestarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(sc.get("T_0", 10)),
                T_mult=int(sc.get("T_mult", 1)),
                eta_min=float(sc.get("eta_min", 0.0)),
            )

    # ------------------------------------------------------------------ #
    # Early stopping
    # ------------------------------------------------------------------ #
    es_cfg = tr_cfg.get("early_stopping", {})
    es_enabled = es_cfg.get("enabled", True)
    patience = int(es_cfg.get("patience", 15))
    best_f1 = -float("inf")
    no_improve = 0

    # ------------------------------------------------------------------ #
    # Checkpointing & CSV logging
    # ------------------------------------------------------------------ #
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"]) / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"{exp_name}_best.pth"

    results_dir = Path(cfg["output"]["results_dir"]) / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{exp_name}_epoch_log.csv"

    csv_fields = [
        "epoch", "train_loss", "val_loss", "val_class1_f1",
        "samples_this_epoch", "pruned_this_epoch", "epoch_time_s",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # ------------------------------------------------------------------ #
    # Loss functions (created once outside loop)
    # ------------------------------------------------------------------ #
    bce_crit  = nn.BCEWithLogitsLoss(reduction="none")
    dice_crit = DiceLoss()

    # ------------------------------------------------------------------ #
    # Training statistics
    # ------------------------------------------------------------------ #
    total_samples_seen = 0
    epochs_trained = 0
    total_pruned = 0
    train_start = time.time()

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # --- InfoBatch phase reset (must be before iter()) ---
        if use_ib:
            pruner.sampler.reset()

        samples_this_epoch = len(pruner.sampler) if use_ib else len(train_loader.dataset)
        pruned_this_epoch  = pruner.sampler.pruned_this_epoch if use_ib else 0

        # ---- Training pass ----
        model.train()
        train_loss_sum = 0.0
        n_steps = 0

        for step, batch in enumerate(train_loader, 1):
            imgs       = batch["image"].to(device)
            labels     = batch["label"].unsqueeze(1).float().to(device)
            no_data    = batch["no_data_mask"].unsqueeze(1).to(device)
            valid_mask = (~no_data).float()

            # Log input shape on first step of first epoch
            if epoch == 1 and step == 1:
                logger.info(f"Input shape: {imgs.shape}  "
                            f"(expect B×5×224×224)")

            optimizer.zero_grad()
            outputs  = model(imgs)
            raw_loss = bce_crit(outputs, labels)  # (B, 1, H, W)

            ps_bce = per_sample_bce(raw_loss, valid_mask)  # (B,)

            if use_ib:
                if use_cs_ib:
                    score = per_sample_cs_score(
                        raw_loss, labels, valid_mask,
                        pruner.w_fg, pruner.w_bg,
                    )
                else:
                    score = ps_bce
                bce_loss = pruner.update(ps_bce, score)
            else:
                bce_loss = ps_bce.mean()

            dice = dice_crit(outputs, labels, valid_mask)
            total_loss = bce_loss + w_dice * dice
            total_loss.backward()
            optimizer.step()

            train_loss_sum += total_loss.item()
            n_steps += 1

            if step % log_interval == 0:
                avg = train_loss_sum / n_steps
                logger.info(
                    f"[Epoch {epoch}/{num_epochs}] "
                    f"Step {step}/{len(train_loader)} — loss: {avg:.4f}"
                )

        avg_train_loss = train_loss_sum / max(n_steps, 1)

        # ---- Validation loss (for logging) ----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs       = batch["image"].to(device)
                labels_v   = batch["label"].unsqueeze(1).float().to(device)
                no_data_v  = batch["no_data_mask"].unsqueeze(1).to(device)
                valid_v    = (~no_data_v).float()
                outputs_v  = model(imgs)
                raw_v      = bce_crit(outputs_v, labels_v)
                bce_v      = per_sample_bce(raw_v, valid_v).mean()
                dice_v     = dice_crit(outputs_v, labels_v, valid_v)
                val_loss_sum += (bce_v + w_dice * dice_v).item()
        avg_val_loss = val_loss_sum / max(len(val_loader), 1)

        # ---- Validation F1 for early stopping ----
        val_f1 = compute_val_class1_f1(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        total_samples_seen += samples_this_epoch
        total_pruned += pruned_this_epoch
        epochs_trained = epoch

        # ---- Logging ----
        logger.info(
            f"[Epoch {epoch}/{num_epochs}] "
            f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
            f"val_class1_f1={val_f1:.4f}  "
            f"samples={samples_this_epoch}  pruned={pruned_this_epoch}  "
            f"time={epoch_time/60:.1f}m"
        )

        # Log IB weight/score diagnostics every 10 epochs
        if use_ib and epoch % 10 == 0:
            ib_ds = pruner.dataset
            w = ib_ds.weights
            s = ib_ds.scores
            logger.info(
                f"  IB weights: mean={w.mean():.3f} max={w.max():.3f} "
                f"n_upscaled={(w > 1.01).sum()}"
            )
            if use_cs_ib:
                cls_labels = ib_ds.cls_labels
                pos_s = s[cls_labels].mean().item() if cls_labels.any() else 0.0
                neg_s = s[~cls_labels].mean().item() if (~cls_labels).any() else 0.0
                logger.info(
                    f"  CS-IB scores: pos_mean={pos_s:.4f} neg_mean={neg_s:.4f} "
                    f"(pos_mean >> neg_mean is expected)"
                )

        csv_writer.writerow({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "val_class1_f1": round(val_f1, 6),
            "samples_this_epoch": samples_this_epoch,
            "pruned_this_epoch": pruned_this_epoch,
            "epoch_time_s": round(epoch_time, 2),
        })
        csv_file.flush()

        # ---- Early stopping ----
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(f"  New best val_class1_f1={best_f1:.4f} — checkpoint saved")
            no_improve = 0
        else:
            no_improve += 1
            if es_enabled and no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch} "
                            f"(no improvement for {patience} epochs)")
                break

        if scheduler:
            scheduler.step()

    csv_file.close()
    train_time = time.time() - train_start
    logger.info(f"Training complete: {epochs_trained} epochs, "
                f"{total_samples_seen} total samples seen, "
                f"{train_time/3600:.2f}h")

    return {
        "best_val_class1_f1": best_f1,
        "train_time_seconds": round(train_time, 1),
        "total_samples_seen": total_samples_seen,
        "epochs_trained": epochs_trained,
        "total_pruned_samples": total_pruned,
        "best_ckpt_path": str(best_ckpt_path),
    }
