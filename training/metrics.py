import numpy as np
import torch


def compute_val_class1_f1(model: torch.nn.Module,
                           val_loader,
                           device: torch.device) -> float:
    """Quick per-epoch class1 (tree) F1 for early stopping, over the full val set."""
    tp = fp = fn = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            imgs   = batch["image"].to(device)
            gt     = batch["label"].unsqueeze(1).float().to(device)
            no_data = batch["no_data_mask"].unsqueeze(1).to(device)
            valid  = (~no_data).float()

            preds = (torch.sigmoid(model(imgs)) > 0.5).float()

            p_flat = (preds * valid).view(-1).long().cpu().numpy()
            g_flat = (gt    * valid).view(-1).long().cpu().numpy()
            m_flat = valid.view(-1).bool().cpu().numpy()

            pv = p_flat[m_flat]
            gv = g_flat[m_flat]

            tp += int(np.logical_and(pv == 1, gv == 1).sum())
            fp += int(np.logical_and(pv == 1, gv == 0).sum())
            fn += int(np.logical_and(pv == 0, gv == 1).sum())

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    return float(f1)
