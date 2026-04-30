#!/usr/bin/env python3
"""
compare_models.py — Side-by-side comparison of Mask2Former vs DeepLabV3+ results.

Reads from results/ directory. Gracefully skips missing experiments (shows ---).

Usage:
    python compare_models.py                   # all available ratios
    python compare_models.py --ratios 0.5      # only prune_ratio=0.5
    python compare_models.py --ratios 0.5 0.7  # multiple ratios
"""

import argparse
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR  = Path(__file__).parent / "results_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

METHODS_ORDER = ["Baseline", "CS-InfoBatch", "InfoBatch", "Random Prune"]

COLORS = {
    "Mask2Former": "#1565C0",
    "DeepLabV3":   "#E65100",
}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "figure.dpi":      120,
    "savefig.dpi":     180,
    "savefig.bbox":    "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dir(name: str):
    """
    Returns (method, ratio, model_label) or None.
    model_label is 'DeepLabV3' for *_deeplabv3 dirs, 'Mask2Former' for all others.
    """
    if name.endswith("_deeplabv3"):
        model = "DeepLabV3"
    elif (name.endswith("_baseline") or name.endswith("_cs_infobatch")
          or name.endswith("_infobatch") or name.endswith("_random_prune")):
        model = "Mask2Former"
    else:
        return None

    if name.startswith("baseline"):
        return "Baseline", 0.0, model

    m = re.search(r"_r([\d.]+)_", name)
    ratio = float(m.group(1)) if m else 0.0

    if "cs_infobatch" in name:  return "CS-InfoBatch", ratio, model
    if "infobatch"    in name:  return "InfoBatch",    ratio, model
    if "random"       in name:  return "Random Prune", ratio, model
    return None


def load_results() -> pd.DataFrame:
    rows = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        parsed = _parse_dir(d.name)
        if parsed is None:
            continue
        method, ratio, model = parsed

        yaml_files = list(d.glob("*_metrics.yaml"))
        if not yaml_files:
            continue
        with open(yaml_files[0]) as f:
            m = yaml.safe_load(f)

        rows.append({
            "model":            model,
            "method":           method,
            "prune_ratio":      ratio,
            "class1_f1":        m.get("class1_f1",        float("nan")),
            "class1_iou":       m.get("class1_iou",       float("nan")),
            "class1_precision": m.get("class1_precision", float("nan")),
            "class1_recall":    m.get("class1_recall",    float("nan")),
            "train_time_h":     m.get("train_time_seconds", 0) / 3600,
            "epochs_trained":   m.get("epochs_trained",   0),
            "best_val_f1":      m.get("best_val_class1_f1", float("nan")),
        })

    return (pd.DataFrame(rows)
            .sort_values(["method", "prune_ratio", "model"])
            .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# Console table
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val, fmt=".4f"):
    return f"{val:{fmt}}" if not (isinstance(val, float) and np.isnan(val)) else "  ---  "


def print_comparison_table(df: pd.DataFrame, ratios=None):
    models = ["Mask2Former", "DeepLabV3"]

    # Baseline F1 per model (for dF1 column)
    b_f1 = {}
    for mdl in models:
        b = df[(df.method == "Baseline") & (df.model == mdl)]
        b_f1[mdl] = b["class1_f1"].values[0] if not b.empty else float("nan")

    # Filter rows to show
    show_ratios = set([0.0] + list(ratios)) if ratios else None
    show_df = df[df.prune_ratio.isin(show_ratios)] if show_ratios else df

    W = 104
    ratio_tag = f"ratios={ratios}" if ratios else "all available ratios"
    print(f"\n{'='*W}")
    print(f"  MASK2FORMER  vs  DEEPLABV3+   —  {ratio_tag}")
    print(f"{'='*W}")
    print(f"{'Method':<16} {'Ratio':>5}  "
          f"{'M2F F1':>8} {'M2F IoU':>8} {'M2F h':>6}  "
          f"{'DLV F1':>8} {'DLV IoU':>8} {'DLV h':>6}  "
          f"{'dF1 M2F':>8} {'dF1 DLV':>8}")
    print(f"{'-'*W}")

    for method in METHODS_ORDER:
        sub = show_df[show_df.method == method]
        if sub.empty:
            continue
        for ratio in sorted(sub.prune_ratio.unique()):
            line = f"{method:<16} {ratio:>5.1f}  "
            for mdl in models:
                r = sub[(sub.prune_ratio == ratio) & (sub.model == mdl)]
                if r.empty:
                    line += f"{'---':>8} {'---':>8} {'---':>6}  "
                else:
                    r = r.iloc[0]
                    line += (f"{_fmt(r.class1_f1):>8} "
                             f"{_fmt(r.class1_iou):>8} "
                             f"{r.train_time_h:>6.2f}  ")
            # dF1 per model
            for mdl in models:
                r = sub[(sub.prune_ratio == ratio) & (sub.model == mdl)]
                if r.empty or method == "Baseline" or np.isnan(b_f1.get(mdl, float("nan"))):
                    line += f"{'---':>8} "
                else:
                    delta = r.iloc[0].class1_f1 - b_f1[mdl]
                    line += f"{delta:>+8.4f} "
            print(line)

    print(f"{'='*W}")
    for mdl in models:
        if not np.isnan(b_f1.get(mdl, float("nan"))):
            print(f"  {mdl} baseline F1 = {b_f1[mdl]:.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _get_val(df, method, ratio, model, col):
    r = df[(df.method == method) & (df.prune_ratio == ratio) & (df.model == model)]
    return r[col].values[0] if not r.empty else float("nan")


def fig_f1_bars(df: pd.DataFrame, ratios=None, out_path: Path = None):
    """Grouped bar chart: F1 per (method, ratio) for both models."""
    show_ratios = set([0.0] + list(ratios)) if ratios else None
    show_df = df[df.prune_ratio.isin(show_ratios)] if show_ratios else df

    # Build x-axis groups: one per (method, ratio)
    groups = []
    for method in METHODS_ORDER:
        sub = show_df[show_df.method == method]
        for ratio in sorted(sub.prune_ratio.unique()):
            if method == "Baseline":
                groups.append(("Baseline", 0.0))
            else:
                groups.append((method, ratio))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for g in groups:
        if g not in seen:
            seen.add(g)
            unique.append(g)
    groups = unique

    x_labels = [g[0] if g[0] == "Baseline" else f"{g[0]}\nr={g[1]}" for g in groups]
    x = np.arange(len(groups))
    w = 0.35

    m2f_f1 = [_get_val(show_df, g[0], g[1], "Mask2Former", "class1_f1") for g in groups]
    dlv_f1 = [_get_val(show_df, g[0], g[1], "DeepLabV3",   "class1_f1") for g in groups]

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.4), 5))
    b1 = ax.bar(x - w/2, m2f_f1, w, label="Mask2Former",
                color=COLORS["Mask2Former"], alpha=0.85)
    b2 = ax.bar(x + w/2, dlv_f1, w, label="DeepLabV3+",
                color=COLORS["DeepLabV3"],   alpha=0.85)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Tree F1  (class 1)")
    ax.set_title("Mask2Former vs DeepLabV3+  —  Segmentation Quality")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    ylim_top = max(v for v in m2f_f1 + dlv_f1 if not np.isnan(v)) + 0.08
    ax.set_ylim(0, min(1.0, ylim_top))

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        print(f"Bar chart saved -> {out_path.name}")
    plt.close(fig)


def fig_f1_iou_scatter(df: pd.DataFrame, ratios=None, out_path: Path = None):
    """F1 vs IoU scatter — one point per experiment, coloured by model."""
    show_ratios = set([0.0] + list(ratios)) if ratios else None
    plot_df = df[df.prune_ratio.isin(show_ratios)] if show_ratios else df

    fig, ax = plt.subplots(figsize=(6, 5))
    for mdl, color in COLORS.items():
        sub = plot_df[plot_df.model == mdl]
        if sub.empty:
            continue
        ax.scatter(sub.class1_iou, sub.class1_f1, label=mdl,
                   color=color, s=70, alpha=0.85, zorder=3)
        for _, r in sub.iterrows():
            tag = "Base" if r.method == "Baseline" else f"r={r.prune_ratio}"
            ax.annotate(tag, (r.class1_iou, r.class1_f1),
                        textcoords="offset points", xytext=(4, 2),
                        fontsize=7, color=color)

    ax.set_xlabel("Tree IoU  (class 1)")
    ax.set_ylabel("Tree F1   (class 1)")
    ax.set_title("F1 vs IoU  —  Mask2Former vs DeepLabV3+")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        print(f"Scatter plot saved -> {out_path.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, ratios=None, out_path: Path = None):
    show_ratios = set([0.0] + list(ratios)) if ratios else None
    out = df[df.prune_ratio.isin(show_ratios)].copy() if show_ratios else df.copy()

    b_f1 = {}
    for mdl in ["Mask2Former", "DeepLabV3"]:
        b = out[(out.method == "Baseline") & (out.model == mdl)]
        b_f1[mdl] = b["class1_f1"].values[0] if not b.empty else float("nan")

    out["delta_f1"] = out.apply(
        lambda r: r.class1_f1 - b_f1.get(r.model, float("nan"))
        if r.method != "Baseline" else float("nan"),
        axis=1,
    )
    out["f1_per_hour"] = out.class1_f1 / out.train_time_h.replace(0, float("nan"))

    if out_path:
        out.to_csv(out_path, index=False, float_format="%.6f")
        print(f"CSV saved -> {out_path.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare Mask2Former vs DeepLabV3+ experiment results"
    )
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=None,
        help="Prune ratios to include, e.g. --ratios 0.5  (default: all available)",
    )
    args = parser.parse_args()

    df = load_results()
    if df.empty:
        print("No results found in results/")
        return

    models_found = sorted(df.model.unique())
    ratios_found = sorted(r for r in df.prune_ratio.unique() if r > 0)
    print(f"Loaded {len(df)} entries  |  models: {models_found}  |  "
          f"non-baseline ratios available: {ratios_found}")

    ratios = args.ratios

    # Table to console
    print_comparison_table(df, ratios)

    # Bar chart (F1)
    fig_f1_bars(
        df, ratios,
        out_path=OUTPUT_DIR / "fig_model_comparison_f1.png",
    )

    # F1 vs IoU scatter
    fig_f1_iou_scatter(
        df, ratios,
        out_path=OUTPUT_DIR / "fig_model_comparison_scatter.png",
    )

    # CSV
    save_csv(
        df, ratios,
        out_path=OUTPUT_DIR / "model_comparison.csv",
    )


if __name__ == "__main__":
    main()
