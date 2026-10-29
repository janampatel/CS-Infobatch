#!/usr/bin/env python3
"""
Unified analysis script for CS-InfoBatch experiment results.
Generates per-architecture figures/tables plus cross-architecture overlays.

Run from project root:  python evaluate/analyze_results.py

Outputs: results_analysis/{mask2former,deeplabv3}/ and results_analysis/comparison/.
ROC-AUC, calibration, and probability-distribution figures additionally need
per-pixel .npz files in results_analysis/probs/ (from generate_probs.py); skipped
gracefully per run if absent.

Per-architecture outputs (fig1–10 + tables):
    fig1_f1_vs_ratio.png         — main quality curve
    fig2_time_vs_ratio.png       — training time curve
    fig3_pareto.png              — quality-efficiency Pareto frontier
    fig4_learning_curves.png     — val F1 convergence per ratio
    fig5_samples_per_epoch.png   — adaptive pruning dynamics
    fig6_degradation_heatmap.png — dF1 vs baseline heatmap
    fig7_efficiency.png          — F1 per training hour
    fig8_roc_auc.png             — ROC curves + AUC (requires probs)
    fig9_calibration.png         — reliability diagrams (requires probs)
    fig10_prob_dist.png          — probability distributions (requires probs)
    results_table.csv / .tex

Cross-architecture outputs:
    figC1_f1_vs_ratio_by_method.png — all method×arch lines on one plot
    figC2_joint_pareto.png          — joint quality-efficiency frontier
    figC3_arch_gap_heatmap.png      — (M2F − DeepLab) F1 gap heatmap
"""

import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATHS & STYLE
# ──────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_ROOT = Path(__file__).parent.parent / "results_analysis"
PROBS_DIR   = OUTPUT_ROOT / "probs"          # shared across architectures
OUTPUT_ROOT.mkdir(exist_ok=True)

METHODS = ["CS-InfoBatch", "InfoBatch", "Random Prune"]
RATIOS  = [0.3, 0.5, 0.7, 0.8, 0.9]

# Per-architecture configuration.
#   suffix       : trailing token in results/ dir names that identifies the arch
#   out_subdir   : where this arch's figures/tables are written
#   baseline_dir : the no-pruning run directory name
#   id_token     : the exp_id suffix used when building pruned-run dir names
# For Mask2Former the exp_id mirrors the pruner (cs_infobatch / infobatch /
# random_prune); for DeepLabV3+ every run shares the exp_id "deeplabv3".
ARCHES = {
    "mask2former": {
        "label":        "Mask2Former",
        "suffix":       None,                 # everything NOT ending in _deeplabv3
        "out_subdir":   "mask2former",
        "baseline_dir": "baseline_baseline",
        "id_token":     {"CS-InfoBatch": "cs_infobatch",
                         "InfoBatch":    "infobatch",
                         "Random Prune": "random_prune"},
    },
    "deeplabv3": {
        "label":        "DeepLabV3+",
        "suffix":       "_deeplabv3",
        "out_subdir":   "deeplabv3",
        "baseline_dir": "baseline_deeplabv3",
        "id_token":     {"CS-InfoBatch": "deeplabv3",
                         "InfoBatch":    "deeplabv3",
                         "Random Prune": "deeplabv3"},
    },
}

# pruner prefix used in the dir name (same for both architectures)
PRUNER_PREFIX = {"CS-InfoBatch": "cs_infobatch",
                 "InfoBatch":    "infobatch",
                 "Random Prune": "random"}

COLORS = {
    "CS-InfoBatch": "#1565C0",
    "InfoBatch":    "#E65100",
    "Random Prune": "#B71C1C",
    "Baseline":     "#2E7D32",
}
MARKERS = {
    "CS-InfoBatch": "o",
    "InfoBatch":    "s",
    "Random Prune": "^",
    "Baseline":     "*",
}
LINES = {
    "CS-InfoBatch": "-",
    "InfoBatch":    "--",
    "Random Prune": ":",
    "Baseline":     "-.",
}

# Colours used only in the cross-architecture overlay figures
ARCH_COLORS = {
    "Mask2Former": "#1565C0",
    "DeepLabV3+":  "#E65100",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        120,
    "savefig.bbox":      "tight",
    "savefig.dpi":       180,
})

# ──────────────────────────────────────────────────────────────────────────────
# ARCH HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def belongs_to_arch(dir_name: str, arch_cfg: dict) -> bool:
    """True if a results/ directory belongs to this architecture."""
    if "dataset2" in dir_name or dir_name.endswith("_ds2"):
        return False  # Dataset 2 runs are analyzed separately, by analyze_results_dataset2.py
    if arch_cfg["suffix"] is None:
        # _s2/_s3 seed reruns are summarised separately, like the DeepLabV3+ _s1.._s5 sweep
        return "deeplabv3" not in dir_name and not re.search(r"_s\d+$", dir_name)
    return dir_name.endswith(arch_cfg["suffix"])


def build_exp_names(arch_cfg: dict) -> dict:
    """
    Maps (method_label, ratio) -> exp_name == results/ dir name == prob-file stem.
    """
    mapping = {("Baseline", 0.0): arch_cfg["baseline_dir"]}
    for method in METHODS:
        for ratio in RATIOS:
            mapping[(method, ratio)] = (
                f"{PRUNER_PREFIX[method]}_r{ratio}_{arch_cfg['id_token'][method]}"
            )
    return mapping

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def _parse_dir(name: str):
    if name.startswith("baseline"):
        return "Baseline", 0.0
    m = re.search(r"_r([\d.]+)_", name)
    ratio = float(m.group(1)) if m else 0.0
    if "cs_infobatch" in name:   return "CS-InfoBatch", ratio
    if "infobatch"    in name:   return "InfoBatch",    ratio
    if "random"       in name:   return "Random Prune", ratio
    return None, None


def load_all_runs(arch_cfg: dict):
    rows, logs = [], {}
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir() or not belongs_to_arch(d.name, arch_cfg):
            continue
        method, ratio = _parse_dir(d.name)
        if method is None:
            continue
        yaml_files = list(d.glob("*_metrics.yaml"))
        if not yaml_files:
            continue
        with open(yaml_files[0]) as f:
            m = yaml.safe_load(f)
        rows.append({
            "method":           method,
            "prune_ratio":      ratio,
            "class1_f1":        m.get("class1_f1"),
            "class1_iou":       m.get("class1_iou"),
            "class1_precision": m.get("class1_precision"),
            "class1_recall":    m.get("class1_recall"),
            "class0_f1":        m.get("class0_f1"),
            "overall_acc":      m.get("overall_accuracy"),
            "macro_f1":         m.get("res_f1_macro"),
            "epochs_trained":   m.get("epochs_trained"),
            "train_time_h":     m.get("train_time_seconds", 0) / 3600,
            "samples_seen":     m.get("total_samples_seen"),
            "pruned_samples":   m.get("total_pruned_samples", 0),
            "best_val_f1":      m.get("best_val_class1_f1"),
        })
        csv_files = list(d.glob("*_epoch_log.csv"))
        if csv_files:
            logs[(method, ratio)] = pd.read_csv(csv_files[0])

    df = (pd.DataFrame(rows)
            .sort_values(["method", "prune_ratio"])
            .reset_index(drop=True))
    return df, logs


def load_probs(method: str, ratio: float, exp_map: dict):
    """
    Load pre-generated probability file. Returns (probs, labels) as float32/uint8,
    or (None, None) if the file does not exist.
    """
    key = (method, ratio)
    if key not in exp_map:
        return None, None
    path = PROBS_DIR / f"{exp_map[key]}_probs.npz"
    if not path.exists():
        return None, None
    data = np.load(path)
    return data["probs"].astype(np.float32), data["labels"].astype(np.uint8)

# ──────────────────────────────────────────────────────────────────────────────
# ROC / CALIBRATION helpers (no sklearn dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _roc_auc(labels: np.ndarray, probs: np.ndarray, max_points: int = 2000):
    """
    Rank-based ROC: every distinct predicted probability acts as a threshold.

    This previously swept a UNIFORM grid, np.linspace(1, 0, 300). Around 98% of
    predicted probabilities on this task fall below 0.001, so that grid placed
    only about three points across the whole region where the curve lives. The
    ROC collapsed into three straight segments and AUC came out 0.07-0.18 too
    low -- by an amount that depended on how saturated each model's outputs
    were, which changed the ranking between methods. Sorting once removes the
    grid entirely. Verified to match sklearn.metrics.roc_auc_score exactly,
    including under heavy ties.
    """
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    order = np.argsort(probs, kind="mergesort")[::-1]
    y = pos[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)

    # Collapse runs of tied scores to a single ROC vertex.
    p_sorted = probs[order]
    keep = np.r_[np.diff(p_sorted) != 0, True]
    tp, fp = tp[keep], fp[keep]

    tpr = np.r_[0.0, tp / n_pos, 1.0]
    fpr = np.r_[0.0, fp / n_neg, 1.0]
    auc_val = float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))

    # Thin for plotting only; AUC above uses the full curve.
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return fpr, tpr, auc_val


def _calibration_curve(labels: np.ndarray, probs: np.ndarray,
                        n_bins: int = 15, strategy: str = "uniform"):
    if strategy == "quantile":
        edges = np.percentile(probs, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    mean_pred, frac_pos = [], []
    for i in range(len(edges) - 1):
        mask = (probs >= edges[i]) & (probs < edges[i + 1])
        if mask.sum() >= 50:
            mean_pred.append(probs[mask].mean())
            frac_pos.append(labels[mask].mean())
    return np.array(mean_pred), np.array(frac_pos)

# ──────────────────────────────────────────────────────────────────────────────
# TABLES
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(df, arch_label):
    b_f1 = df.loc[df.method == "Baseline", "class1_f1"].values[0]

    hdr = (f"{'Method':<16} {'Ratio':>5} {'F1':>7} {'IoU':>7} {'Prec':>7} "
           f"{'Recall':>7} {'Epochs':>6} {'Time(h)':>8} {'dF1':>8} {'F1/h':>8}")
    sep = "=" * len(hdr)
    print(f"\n{sep}\n{arch_label.upper()} - RESULTS SUMMARY\n{sep}")
    print(hdr)
    print("-" * len(hdr))

    prev = None
    for _, r in df.iterrows():
        if prev is not None and r.method != "Baseline" and r.prune_ratio != prev:
            print("-" * len(hdr))
        prev = r.prune_ratio
        delta = f"{r.class1_f1 - b_f1:+.4f}" if r.method != "Baseline" else "  ---  "
        eff   = f"{r.class1_f1 / r.train_time_h:.4f}"
        print(f"{r.method:<16} {r.prune_ratio:>5} {r.class1_f1:>7.4f} "
              f"{r.class1_iou:>7.4f} {r.class1_precision:>7.4f} {r.class1_recall:>7.4f} "
              f"{int(r.epochs_trained):>6} {r.train_time_h:>8.2f} {delta:>8} {eff:>8}")
    print(sep)


def save_csv(df: pd.DataFrame, path: Path):
    out = df.copy()
    b_f1   = df.loc[df.method == "Baseline", "class1_f1"].values[0]
    b_seen = df.loc[df.method == "Baseline", "samples_seen"].values[0]
    out["delta_f1"]         = out["class1_f1"] - b_f1
    out["f1_per_hour"]      = out["class1_f1"] / out["train_time_h"]
    out["samples_seen_pct"] = out["samples_seen"] / b_seen * 100
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"CSV saved -> {path}")


def save_latex(df: pd.DataFrame, path: Path, arch_label: str):
    b_f1 = df.loc[df.method == "Baseline", "class1_f1"].values[0]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{" + arch_label + r": test metrics on NAIP aerial imagery "
        r"(tree segmentation). $\Delta$F1 = change vs. full-data baseline.}",
        r"\label{tab:main_results_" + arch_label.lower().replace("+", "plus").replace(" ", "_") + r"}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Method & $\rho$ & F1$\uparrow$ & IoU$\uparrow$ & Prec. & Recall "
        r"& Epochs & Time(h)$\downarrow$ & $\Delta$F1 \\",
        r"\midrule",
    ]
    b = df[df.method == "Baseline"].iloc[0]
    lines.append(
        f"Baseline & --- & {b.class1_f1:.4f} & {b.class1_iou:.4f} & "
        f"{b.class1_precision:.4f} & {b.class1_recall:.4f} & "
        f"{int(b.epochs_trained)} & {b.train_time_h:.2f} & --- \\\\"
    )
    lines.append(r"\midrule")
    prev_ratio = None
    for _, r in df[df.method != "Baseline"].sort_values(["prune_ratio", "method"]).iterrows():
        if prev_ratio is not None and r.prune_ratio != prev_ratio:
            lines.append(r"\midrule")
        prev_ratio = r.prune_ratio
        delta = r.class1_f1 - b_f1
        lines.append(
            f"{r.method} & {r.prune_ratio} & {r.class1_f1:.4f} & "
            f"{r.class1_iou:.4f} & {r.class1_precision:.4f} & "
            f"{r.class1_recall:.4f} & {int(r.epochs_trained)} & "
            f"{r.train_time_h:.2f} & {delta:+.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))
    print(f"LaTeX saved -> {path}")

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: shared line-plot logic
# ──────────────────────────────────────────────────────────────────────────────

def _baseline_hline(ax, df, col, fmt="{:.4f}", **kwargs):
    val = df.loc[df.method == "Baseline", col].values[0]
    kw  = dict(color=COLORS["Baseline"], linestyle="-.", linewidth=1.5, zorder=1)
    kw.update(kwargs)
    ax.axhline(val, **kw, label=f"Baseline ({fmt.format(val)})")
    return val


def _plot_method_lines(ax, df, col, annotate=True, ann_ofs=(5, 3)):
    for method in METHODS:
        sub = df[df.method == method].sort_values("prune_ratio")
        if sub.empty:
            continue
        ax.plot(sub.prune_ratio, sub[col],
                color=COLORS[method], marker=MARKERS[method],
                linestyle=LINES[method], linewidth=2, markersize=8,
                label=method, zorder=3)
        if annotate:
            for _, r in sub.iterrows():
                ax.annotate(f"{r[col]:.3f}",
                            (r.prune_ratio, r[col]),
                            textcoords="offset points", xytext=ann_ofs,
                            fontsize=8, color=COLORS[method])

# ──────────────────────────────────────────────────────────────────────────────
# FIGURES (no prob files required)
# ──────────────────────────────────────────────────────────────────────────────

def fig1_f1_vs_ratio(df, path, arch_label):
    fig, ax = plt.subplots(figsize=(7, 5))
    _baseline_hline(ax, df, "class1_f1", "{:.4f}")
    _plot_method_lines(ax, df, "class1_f1")
    ax.set_xlabel("Prune Ratio")
    ax.set_ylabel("Tree F1  (class 1)")
    ax.set_title(f"{arch_label}: Segmentation Quality vs Pruning Aggressiveness")
    ax.set_xticks([0.0] + RATIOS); ax.set_xlim(-0.05, max(RATIOS) + 0.05)
    ax.legend(loc="lower left"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"Fig 1 saved -> {path}")


def fig2_time_vs_ratio(df, path, arch_label):
    fig, ax = plt.subplots(figsize=(7, 5))
    _baseline_hline(ax, df, "train_time_h", "{:.2f}h")
    _plot_method_lines(ax, df, "train_time_h")
    ax.set_xlabel("Prune Ratio")
    ax.set_ylabel("Training Time (hours)")
    ax.set_title(f"{arch_label}: Training Time vs Pruning Aggressiveness")
    ax.set_xticks([0.0] + RATIOS); ax.set_xlim(-0.05, max(RATIOS) + 0.05)
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"Fig 2 saved -> {path}")


def fig3_pareto(df, path, arch_label):
    fig, ax = plt.subplots(figsize=(8, 6))
    b = df[df.method == "Baseline"].iloc[0]
    ax.scatter(b.train_time_h, b.class1_f1, color=COLORS["Baseline"],
               marker="*", s=300, zorder=5, label="Baseline",
               edgecolors="k", linewidths=0.6)
    ax.annotate(f"Baseline\n({b.class1_f1:.3f}, {b.train_time_h:.2f}h)",
                (b.train_time_h, b.class1_f1),
                textcoords="offset points", xytext=(6, -22),
                fontsize=9, color=COLORS["Baseline"])
    for method in METHODS:
        sub = df[df.method == method].sort_values("train_time_h")
        if sub.empty: continue
        ax.plot(sub.train_time_h, sub.class1_f1,
                color=COLORS[method], linestyle=LINES[method],
                linewidth=1.5, alpha=0.55, zorder=2)
        ax.scatter(sub.train_time_h, sub.class1_f1,
                   color=COLORS[method], marker=MARKERS[method],
                   s=120, zorder=4, label=method,
                   edgecolors="k", linewidths=0.5)
        for _, r in sub.iterrows():
            ax.annotate(f"r={r.prune_ratio}",
                        (r.train_time_h, r.class1_f1),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=8, color=COLORS[method])
    ax.set_xlabel("Training Time (hours)")
    ax.set_ylabel("Tree F1  (class 1)")
    ax.set_title(f"{arch_label}: Quality-Efficiency Pareto Frontier\n"
                 "(upper-left = faster and better)")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"Fig 3 saved -> {path}")


def fig4_learning_curves(df, logs, path, arch_label):
    b_f1 = df.loc[df.method == "Baseline", "class1_f1"].values[0]
    baseline_log = logs.get(("Baseline", 0.0))

    n = len(RATIOS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1: axes = [axes]
    fig.suptitle(f"{arch_label}: Validation F1 Learning Curves (by Prune Ratio)", fontsize=13)

    for j, ratio in enumerate(RATIOS):
        ax = axes[j]
        if baseline_log is not None:
            ax.plot(baseline_log.epoch, baseline_log.val_class1_f1,
                    color=COLORS["Baseline"], linestyle=LINES["Baseline"],
                    linewidth=1.5, alpha=0.6, label="Baseline")
        ax.axhline(b_f1, color=COLORS["Baseline"], linestyle=":", linewidth=0.8, alpha=0.35)
        for method in METHODS:
            log = logs.get((method, ratio))
            if log is None: continue
            ax.plot(log.epoch, log.val_class1_f1,
                    color=COLORS[method], linestyle=LINES[method],
                    linewidth=1.8, label=method)
        ax.set_title(f"Prune Ratio r={ratio}")
        ax.set_xlabel("Epoch")
        if j == 0: ax.set_ylabel("Val F1  (tree class)")
        ax.set_ylim(0.0, 0.72); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"Fig 4 saved -> {path}")


def fig5_samples_per_epoch(df, logs, path, arch_label):
    N_FULL = 11152
    n = len(RATIOS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1: axes = [axes]
    fig.suptitle(f"{arch_label}: Samples per Epoch During Training (Pruning Dynamics)", fontsize=13)

    for j, ratio in enumerate(RATIOS):
        ax = axes[j]
        ax.axhline(N_FULL, color="gray", linestyle="--",
                   linewidth=0.9, alpha=0.5, label="Full dataset")
        for method in METHODS:
            log = logs.get((method, ratio))
            if log is None: continue
            ax.plot(log.epoch, log.samples_this_epoch,
                    color=COLORS[method], linestyle=LINES[method],
                    linewidth=2, label=method, alpha=0.9)
            ax.fill_between(log.epoch, log.samples_this_epoch,
                            color=COLORS[method], alpha=0.08)
        target = int(N_FULL * (1 - ratio))
        ax.axhline(target, color="black", linestyle=":", linewidth=0.8,
                   alpha=0.35, label=f"Target ({target})")
        ax.set_title(f"Prune Ratio r={ratio}")
        ax.set_xlabel("Epoch")
        if j == 0: ax.set_ylabel("Samples / Epoch")
        ax.set_ylim(0, N_FULL * 1.15); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"Fig 5 saved -> {path}")


def fig6_degradation_heatmap(df, path, arch_label):
    b_f1 = df.loc[df.method == "Baseline", "class1_f1"].values[0]
    data = np.full((len(METHODS), len(RATIOS)), np.nan)
    for i, method in enumerate(METHODS):
        for j, ratio in enumerate(RATIOS):
            row = df[(df.method == method) & (df.prune_ratio == ratio)]
            if not row.empty:
                data[i, j] = row.class1_f1.values[0] - b_f1
    vmax = float(np.nanmax(np.abs(data))) if not np.all(np.isnan(data)) else 1.0

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(RATIOS))); ax.set_xticklabels([f"r={r}" for r in RATIOS])
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS, fontsize=10)
    for i in range(len(METHODS)):
        for j in range(len(RATIOS)):
            if np.isnan(data[i, j]):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=11, color="gray")
            else:
                fc = "black" if abs(data[i, j]) < vmax * 0.55 else "white"
                ax.text(j, i, f"{data[i,j]:+.4f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=fc)
    plt.colorbar(im, ax=ax, label="dF1 vs Baseline")
    ax.set_title(f"{arch_label}: F1 Degradation vs Baseline (F1={b_f1:.4f})  "
                 "[green = small loss, red = large loss]")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"Fig 6 saved -> {path}")


def fig7_efficiency(df, path, arch_label):
    fig, ax = plt.subplots(figsize=(7, 5))
    b = df[df.method == "Baseline"].iloc[0]
    b_eff = b.class1_f1 / b.train_time_h
    ax.axhline(b_eff, color=COLORS["Baseline"], linestyle=LINES["Baseline"],
               linewidth=1.5, label=f"Baseline ({b_eff:.3f} F1/h)", zorder=1)
    for method in METHODS:
        sub = df[df.method == method].sort_values("prune_ratio")
        if sub.empty: continue
        eff = sub.class1_f1 / sub.train_time_h
        ax.plot(sub.prune_ratio.values, eff.values,
                color=COLORS[method], marker=MARKERS[method],
                linestyle=LINES[method], linewidth=2, markersize=8,
                label=method, zorder=3)
        for (_, r), e in zip(sub.iterrows(), eff.values):
            ax.annotate(f"{e:.3f}",
                        (r.prune_ratio, e),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=8, color=COLORS[method])
    ax.set_xlabel("Prune Ratio")
    ax.set_ylabel("F1 per Training Hour")
    ax.set_title(f"{arch_label}: Training Efficiency  (F1/hour — higher is better)")
    ax.set_xticks([0.0] + RATIOS); ax.set_xlim(-0.05, max(RATIOS) + 0.05)
    ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"Fig 7 saved -> {path}")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURES (require generate_probs.py output) — skipped per-run if .npz missing
# ──────────────────────────────────────────────────────────────────────────────

def fig8_roc_auc(df, path, exp_map, arch_label):
    b_probs, b_labels = load_probs("Baseline", 0.0, exp_map)
    if not _any_probs_for(exp_map):
        print(f"Fig 8 (ROC-AUC) skipped — no probability files for {arch_label}.")
        return

    n = len(RATIOS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1: axes = [axes]
    fig.suptitle(f"{arch_label}: ROC Curves by Prune Ratio  (pixel-level, tree class)", fontsize=13)

    for j, ratio in enumerate(RATIOS):
        ax = axes[j]
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="Random (AUC=0.50)")

        if b_probs is not None:
            fpr, tpr, auc_val = _roc_auc(b_labels, b_probs)
            ax.plot(fpr, tpr, color=COLORS["Baseline"],
                    linestyle=LINES["Baseline"], linewidth=2,
                    label=f"Baseline (AUC={auc_val:.4f})")

        for method in METHODS:
            probs, labels = load_probs(method, ratio, exp_map)
            if probs is None:
                continue
            fpr, tpr, auc_val = _roc_auc(labels, probs)
            ax.plot(fpr, tpr, color=COLORS[method],
                    linestyle=LINES[method], linewidth=2,
                    label=f"{method} (AUC={auc_val:.4f})")

        ax.set_title(f"Prune Ratio r={ratio}")
        ax.set_xlabel("False Positive Rate")
        if j == 0: ax.set_ylabel("True Positive Rate")
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
        ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.25)

    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"Fig 8 saved -> {path}")


def fig9_calibration(df, path, exp_map, arch_label):
    if not _any_probs_for(exp_map):
        print(f"Fig 9 (Calibration) skipped — no probability files for {arch_label}.")
        return

    n = len(RATIOS)
    fig = plt.figure(figsize=(5.5 * n, 6))
    fig.suptitle(f"{arch_label}: Calibration (Reliability) Diagrams  (pixel-level, tree class)\n"
                 "Above diagonal = underconfident  |  Below = overconfident",
                 fontsize=12)

    outer_gs = gridspec.GridSpec(1, n, figure=fig, hspace=0.0, wspace=0.35)
    b_probs, b_labels = load_probs("Baseline", 0.0, exp_map)

    for j, ratio in enumerate(RATIOS):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[j], height_ratios=[3, 1], hspace=0.08
        )
        ax_cal = fig.add_subplot(inner_gs[0])
        ax_hist = fig.add_subplot(inner_gs[1], sharex=ax_cal)

        ax_cal.plot([0, 1], [0, 1], "k--", linewidth=0.9, alpha=0.4,
                    label="Perfect calibration")
        ax_cal.fill_between([0, 1], [0, 0], [0, 1], alpha=0.03, color="green")
        ax_cal.fill_between([0, 1], [0, 1], [1, 1], alpha=0.03, color="orange")

        if b_probs is not None:
            mp, fp = _calibration_curve(b_labels, b_probs, n_bins=15)
            ax_cal.plot(mp, fp, color=COLORS["Baseline"],
                        linestyle=LINES["Baseline"], linewidth=2,
                        marker="o", markersize=5, label="Baseline")
            ax_hist.hist(b_probs, bins=40, range=(0, 1),
                         color=COLORS["Baseline"], alpha=0.35,
                         density=True, histtype="stepfilled")

        for method in METHODS:
            probs, labels = load_probs(method, ratio, exp_map)
            if probs is None:
                continue
            mp, fp = _calibration_curve(labels, probs, n_bins=15)
            ax_cal.plot(mp, fp, color=COLORS[method],
                        linestyle=LINES[method], linewidth=2,
                        marker=MARKERS[method], markersize=5, label=method)
            ax_hist.hist(probs, bins=40, range=(0, 1),
                         color=COLORS[method], alpha=0.35,
                         density=True, histtype="stepfilled")

        ax_cal.set_title(f"Prune Ratio r={ratio}", fontsize=11)
        ax_cal.set_ylabel("Fraction of Positives" if j == 0 else "")
        ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
        ax_cal.legend(fontsize=7, loc="upper left"); ax_cal.grid(True, alpha=0.2)
        plt.setp(ax_cal.get_xticklabels(), visible=False)

        ax_hist.set_xlabel("Mean Predicted Probability")
        ax_hist.set_ylabel("Density" if j == 0 else "")
        ax_hist.set_xlim(0, 1); ax_hist.grid(True, alpha=0.2)
        ax_hist.set_yscale("log")

    fig.savefig(path); plt.close(fig)
    print(f"Fig 9 saved -> {path}")


def fig10_prob_dist(df, path, exp_map, arch_label):
    if not _any_probs_for(exp_map):
        print(f"Fig 10 (Prob distribution) skipped — no probability files for {arch_label}.")
        return

    n = len(RATIOS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1: axes = [axes]
    fig.suptitle(f"{arch_label}: Predicted Probability Distributions  (p(tree) per pixel)\n"
                 "X-axis clipped to [0, 0.3] — most mass is near 0 due to class imbalance",
                 fontsize=12)

    b_probs, _ = load_probs("Baseline", 0.0, exp_map)

    for j, ratio in enumerate(RATIOS):
        ax = axes[j]
        clip = 0.30

        if b_probs is not None:
            ax.hist(b_probs[b_probs <= clip], bins=60,
                    color=COLORS["Baseline"], alpha=0.45,
                    density=True, histtype="stepfilled",
                    label="Baseline", linewidth=0)
            ax.hist(b_probs[b_probs <= clip], bins=60,
                    color=COLORS["Baseline"], histtype="step",
                    density=True, linewidth=1.2)

        for method in METHODS:
            probs, _ = load_probs(method, ratio, exp_map)
            if probs is None:
                continue
            ax.hist(probs[probs <= clip], bins=60,
                    color=COLORS[method], alpha=0.35,
                    density=True, histtype="stepfilled",
                    label=method, linewidth=0)
            ax.hist(probs[probs <= clip], bins=60,
                    color=COLORS[method], histtype="step",
                    density=True, linewidth=1.2)

        ax.set_title(f"Prune Ratio r={ratio}")
        ax.set_xlabel("Predicted p(tree)")
        if j == 0: ax.set_ylabel("Density")
        ax.set_yscale("log")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"Fig 10 saved -> {path}")


def _any_probs_for(exp_map: dict) -> bool:
    """True if at least one prob .npz exists for this architecture's experiments."""
    if not PROBS_DIR.exists():
        return False
    return any((PROBS_DIR / f"{name}_probs.npz").exists() for name in exp_map.values())

# ──────────────────────────────────────────────────────────────────────────────
# CROSS-ARCHITECTURE OVERLAY FIGURES (Mask2Former vs DeepLabV3+)
#
# These complement evaluate/compare_models.py (which produces a grouped bar chart
# and an F1-vs-IoU scatter). Here we add the line/Pareto/gap views it lacks.
# ──────────────────────────────────────────────────────────────────────────────

def _baseline_f1(df):
    b = df[df.method == "Baseline"]
    return b.class1_f1.values[0] if not b.empty else float("nan")


def figC1_f1_vs_ratio_by_method(dfs: dict, path: Path):
    """
    Single plot: every (architecture, method) combination as its own line.
    Method is encoded by colour; architecture by line style + marker.
    Both per-architecture baselines are drawn as horizontal reference lines.
    """
    # Architecture -> (line style, marker)
    arch_style = {
        "Mask2Former": ("-",  "o"),
        "DeepLabV3+":  ("--", "s"),
    }
    base_ls = {"Mask2Former": "-", "DeepLabV3+": "--"}

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for arch_label, df in dfs.items():
        ls, marker = arch_style.get(arch_label, ("-", "o"))
        for method in METHODS:
            sub = df[df.method == method].sort_values("prune_ratio")
            if sub.empty:
                continue
            ax.plot(sub.prune_ratio, sub.class1_f1,
                    color=COLORS[method], linestyle=ls, marker=marker,
                    linewidth=2, markersize=7,
                    label=f"{arch_label} · {method}", zorder=3)

        # per-architecture baseline reference line
        b = _baseline_f1(df)
        if not np.isnan(b):
            ax.axhline(b, color="gray", linestyle=base_ls.get(arch_label, ":"),
                       linewidth=1.3, alpha=0.6,
                       label=f"{arch_label} baseline ({b:.3f})")

    ax.set_xlabel("Prune Ratio")
    ax.set_ylabel("Tree F1  (class 1)")
    ax.set_title("Mask2Former vs DeepLabV3+  —  Tree F1 vs Prune Ratio\n"
                 "(colour = method, solid/○ = Mask2Former, dashed/□ = DeepLabV3+)")
    ax.set_xticks(RATIOS); ax.set_xlim(min(RATIOS) - 0.05, max(RATIOS) + 0.05)
    ax.legend(fontsize=8, loc="lower left", ncol=2); ax.grid(True, alpha=0.25)

    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"Cross-arch Fig C1 saved -> {path}")


def figC2_joint_pareto(dfs: dict, path: Path):
    """Quality-efficiency frontier with both architectures on one set of axes."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for arch_label, df in dfs.items():
        acolor = ARCH_COLORS[arch_label]
        # baseline marker
        b = df[df.method == "Baseline"]
        if not b.empty:
            b = b.iloc[0]
            ax.scatter(b.train_time_h, b.class1_f1, color=acolor, marker="*",
                       s=320, zorder=5, edgecolors="k", linewidths=0.6,
                       label=f"{arch_label} baseline")
            ax.annotate(f"{arch_label}\nbaseline", (b.train_time_h, b.class1_f1),
                        textcoords="offset points", xytext=(6, -24),
                        fontsize=8, color=acolor)
        for method in METHODS:
            sub = df[df.method == method].sort_values("train_time_h")
            if sub.empty: continue
            ax.plot(sub.train_time_h, sub.class1_f1, color=acolor,
                    linestyle=LINES[method], linewidth=1.4, alpha=0.55, zorder=2)
            ax.scatter(sub.train_time_h, sub.class1_f1, color=acolor,
                       marker=MARKERS[method], s=110, zorder=4,
                       edgecolors="k", linewidths=0.4,
                       label=f"{arch_label} · {method}")
            for _, r in sub.iterrows():
                ax.annotate(f"r={r.prune_ratio}", (r.train_time_h, r.class1_f1),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=acolor)
    ax.set_xlabel("Training Time (hours)")
    ax.set_ylabel("Tree F1  (class 1)")
    ax.set_title("Joint Quality-Efficiency Pareto Frontier\n"
                 "(colour = architecture, marker = method; upper-left = faster + better)")
    ax.legend(fontsize=7, loc="lower right", ncol=2); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"Cross-arch Fig C2 saved -> {path}")


def figC3_arch_gap_heatmap(dfs: dict, path: Path):
    """Heatmap of (Mask2Former F1 − DeepLabV3+ F1) per method × ratio."""
    if "Mask2Former" not in dfs or "DeepLabV3+" not in dfs:
        print("Cross-arch Fig C3 skipped — need both architectures.")
        return
    m2f, dlv = dfs["Mask2Former"], dfs["DeepLabV3+"]

    data = np.full((len(METHODS), len(RATIOS)), np.nan)
    for i, method in enumerate(METHODS):
        for j, ratio in enumerate(RATIOS):
            a = m2f[(m2f.method == method) & (m2f.prune_ratio == ratio)]
            b = dlv[(dlv.method == method) & (dlv.prune_ratio == ratio)]
            if not a.empty and not b.empty:
                data[i, j] = a.class1_f1.values[0] - b.class1_f1.values[0]
    vmax = float(np.nanmax(np.abs(data))) if not np.all(np.isnan(data)) else 1.0

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, cmap="PuOr", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(RATIOS))); ax.set_xticklabels([f"r={r}" for r in RATIOS])
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS, fontsize=10)
    for i in range(len(METHODS)):
        for j in range(len(RATIOS)):
            if np.isnan(data[i, j]):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=11, color="gray")
            else:
                fc = "black" if abs(data[i, j]) < vmax * 0.55 else "white"
                ax.text(j, i, f"{data[i,j]:+.4f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=fc)
    plt.colorbar(im, ax=ax, label="F1 gap (M2F − DeepLab)")
    ax.set_title("Architecture Gap: Mask2Former − DeepLabV3+ Tree F1\n"
                 "[purple = Mask2Former better, orange = DeepLabV3+ better]")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"Cross-arch Fig C3 saved -> {path}")


def compare_architectures(dfs: dict):
    """Generate the cross-architecture overlay figures into results_analysis/comparison/."""
    dfs = {k: v for k, v in dfs.items() if v is not None and not v.empty}
    if len(dfs) < 2:
        print("\nCross-architecture comparison skipped — need >= 2 architectures with data.")
        return

    out_dir = OUTPUT_ROOT / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 70)
    print(f"# CROSS-ARCHITECTURE COMPARISON  ({' vs '.join(dfs.keys())})")
    print(f"#   output -> {out_dir}")
    print("#" * 70)

    figC1_f1_vs_ratio_by_method(dfs, out_dir / "figC1_f1_vs_ratio_by_method.png")
    figC2_joint_pareto         (dfs, out_dir / "figC2_joint_pareto.png")
    figC3_arch_gap_heatmap     (dfs, out_dir / "figC3_arch_gap_heatmap.png")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def analyze_arch(arch_key: str, arch_cfg: dict):
    arch_label = arch_cfg["label"]
    out_dir    = OUTPUT_ROOT / arch_cfg["out_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_map    = build_exp_names(arch_cfg)

    print("\n" + "#" * 70)
    print(f"# ARCHITECTURE: {arch_label}")
    print(f"#   output -> {out_dir}")
    print("#" * 70)

    df, logs = load_all_runs(arch_cfg)
    if df.empty or "Baseline" not in df.method.values:
        print(f"  No usable runs (or no baseline) for {arch_label} — skipping.")
        return None

    print(f"\nLoaded {len(df)} runs:")
    for _, r in df.iterrows():
        prob_tag = "[probs OK]" if load_probs(r.method, r.prune_ratio, exp_map)[0] is not None else ""
        print(f"  {r.method:<16} r={r.prune_ratio}  F1={r.class1_f1:.4f}  "
              f"time={r.train_time_h:.2f}h  epochs={int(r.epochs_trained)}  {prob_tag}")

    print_summary(df, arch_label)

    # Tables
    save_csv(df.copy(), out_dir / "results_table.csv")
    save_latex(df,      out_dir / "results_table.tex", arch_label)

    # Figures — always available
    fig1_f1_vs_ratio       (df,        out_dir / "fig1_f1_vs_ratio.png",         arch_label)
    fig2_time_vs_ratio     (df,        out_dir / "fig2_time_vs_ratio.png",       arch_label)
    fig3_pareto            (df,        out_dir / "fig3_pareto.png",              arch_label)
    fig4_learning_curves   (df, logs,  out_dir / "fig4_learning_curves.png",     arch_label)
    fig5_samples_per_epoch (df, logs,  out_dir / "fig5_samples_per_epoch.png",   arch_label)
    fig6_degradation_heatmap(df,       out_dir / "fig6_degradation_heatmap.png", arch_label)
    fig7_efficiency        (df,        out_dir / "fig7_efficiency.png",          arch_label)

    # Figures — require probs (skipped per-run if missing)
    fig8_roc_auc     (df, out_dir / "fig8_roc_auc.png",     exp_map, arch_label)
    fig9_calibration (df, out_dir / "fig9_calibration.png", exp_map, arch_label)
    fig10_prob_dist  (df, out_dir / "fig10_prob_dist.png",  exp_map, arch_label)

    return df


def main():
    print(f"Results dir : {RESULTS_DIR}")
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"Probs dir   : {PROBS_DIR}  "
          f"({'found' if PROBS_DIR.exists() else 'not found — figs 8-10 will be skipped'})")

    dfs = {}
    for arch_key, arch_cfg in ARCHES.items():
        df = analyze_arch(arch_key, arch_cfg)
        if df is not None:
            dfs[arch_cfg["label"]] = df

    # Cross-architecture overlay figures (Mask2Former vs DeepLabV3+)
    compare_architectures(dfs)

    print("\nDone. Outputs in:")
    for arch_cfg in ARCHES.values():
        print(f"  {OUTPUT_ROOT / arch_cfg['out_subdir']}")
    print(f"  {OUTPUT_ROOT / 'comparison'}  (cross-architecture overlays)")


if __name__ == "__main__":
    main()
