#!/usr/bin/env python3
"""
Plot F1 vs Prune Ratio and IoU vs Prune Ratio for DeepLabV3+ experiments,
showing mean ± std bands for CS-InfoBatch and InfoBatch (5 seeds each),
and single points for Random Prune and Baseline.

Run from project root:
    python evaluate/plot_deeplab_variance.py

Outputs (saved to results_analysis/deeplabv3/):
    fig_variance_f1.png   — Tree F1 vs prune ratio with variance bands
    fig_variance_iou.png  — Tree IoU vs prune ratio with variance bands
    fig_variance_both.png — F1 and IoU side by side

Expected results/ directory layout for multi-seed runs:
    cs_infobatch_r{ratio}_deeplabv3_s1/   (seed index 1..5)
    cs_infobatch_r{ratio}_deeplabv3_s2/
    ...
    infobatch_r{ratio}_deeplabv3_s1/
    ...
Single-seed runs (random prune, baseline) use their standard directory names.
"""

import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR  = Path(__file__).parent.parent / "results_analysis" / "deeplabv3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RATIOS  = [0.3, 0.5, 0.7, 0.8, 0.9]
N_SEEDS = 5

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
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def _load_metric(dir_path: Path, key: str):
    yaml_files = list(dir_path.glob("*_metrics.yaml"))
    if not yaml_files:
        return None
    with open(yaml_files[0]) as f:
        m = yaml.safe_load(f)
    return m.get(key)


def load_multiseed(method_prefix: str, ratio: float, metric: str):
    """
    Load metric values across all available seeds for one (method, ratio).
    method_prefix: e.g. 'cs_infobatch' or 'infobatch'
    Returns list of float values (one per seed found).
    """
    values = []
    for s in range(1, N_SEEDS + 1):
        dir_name = f"{method_prefix}_r{ratio}_deeplabv3_s{s}"
        d = RESULTS_DIR / dir_name
        if not d.is_dir():
            continue
        v = _load_metric(d, metric)
        if v is not None:
            values.append(float(v))
    return values


def load_single(dir_name: str, metric: str):
    """Load metric from a standard (single-seed) results directory."""
    d = RESULTS_DIR / dir_name
    if not d.is_dir():
        return None
    return _load_metric(d, metric)


def build_data(metric: str):
    """
    Returns a dict keyed by method label with raw per-seed values.
    Multi-seed methods: {"ratios": [...], "seed_vals": [[s1,s2,s3], ...]}
    Single-point methods: {"ratios": [...], "values": [...]}
    """
    data = {}

    # CS-InfoBatch — raw per-seed values
    cs_ratios, cs_seed_vals = [], []
    for r in RATIOS:
        vals = load_multiseed("cs_infobatch", r, metric)
        if vals:
            cs_ratios.append(r)
            cs_seed_vals.append(vals)
    data["CS-InfoBatch"] = {"ratios": cs_ratios, "seed_vals": cs_seed_vals}

    # InfoBatch — raw per-seed values
    ib_ratios, ib_seed_vals = [], []
    for r in RATIOS:
        vals = load_multiseed("infobatch", r, metric)
        if vals:
            ib_ratios.append(r)
            ib_seed_vals.append(vals)
    data["InfoBatch"] = {"ratios": ib_ratios, "seed_vals": ib_seed_vals}

    # Random Prune — single point per ratio
    rp_vals, rp_ratios = [], []
    for r in RATIOS:
        v = load_single(f"random_r{r}_deeplabv3", metric)
        if v is not None:
            rp_vals.append(float(v))
            rp_ratios.append(r)
    data["Random Prune"] = {"ratios": rp_ratios, "values": rp_vals}

    # Baseline
    b = load_single("baseline_deeplabv3", metric)
    data["Baseline"] = {"value": float(b) if b is not None else None}

    return data

# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def _draw_variance_plot(ax, data: dict, metric_label: str, title: str):
    """
    Draw one axes:
      - CS-InfoBatch & InfoBatch: vertical line (min→max) with a dot per seed,
        plus a dashed line connecting the means across ratios.
      - Random Prune: single scatter point per ratio, no line.
      - Baseline: horizontal reference line.
    """
    # Baseline horizontal line
    b_val = data["Baseline"]["value"]
    if b_val is not None:
        ax.axhline(b_val, color=COLORS["Baseline"], linestyle="-.",
                   linewidth=1.5, label=f"Baseline ({b_val:.4f})", zorder=1)

    for method in ["CS-InfoBatch", "InfoBatch"]:
        d = data[method]
        if not d["ratios"]:
            print(f"  No data for {method} — skipping.")
            continue

        color  = COLORS[method]
        marker = MARKERS[method]
        means  = []

        for xi, vals in zip(d["ratios"], d["seed_vals"]):
            vals = sorted(vals)
            mean = np.mean(vals)
            means.append(mean)

            # vertical spine from min to max
            ax.plot([xi, xi], [min(vals), max(vals)],
                    color=color, linewidth=1.8, zorder=3, solid_capstyle="round")

            # one dot per seed value
            ax.scatter([xi] * len(vals), vals,
                       color=color, marker=marker,
                       s=60, zorder=5, edgecolors="white", linewidths=0.6)

            # annotate mean
            ax.annotate(f"{mean:.3f}", (xi, mean),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=8, color=color)

        # dashed line connecting means across ratios
        ax.plot(d["ratios"], means,
                color=color, linestyle="--", linewidth=1.2,
                alpha=0.55, zorder=2, label=method)

    # Random Prune — points connected by a line
    rp = data["Random Prune"]
    if rp["ratios"]:
        ax.plot(rp["ratios"], rp["values"],
                color=COLORS["Random Prune"], linestyle=":",
                linewidth=1.8, zorder=3)
        ax.scatter(rp["ratios"], rp["values"],
                   color=COLORS["Random Prune"], marker=MARKERS["Random Prune"],
                   s=80, zorder=5, label="Random Prune")
        for xi, yi in zip(rp["ratios"], rp["values"]):
            ax.annotate(f"{yi:.3f}", (xi, yi),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=8, color=COLORS["Random Prune"])

    ax.set_xlabel("Prune Ratio")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.set_xticks(RATIOS)
    ax.set_xlim(min(RATIOS) - 0.05, max(RATIOS) + 0.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.25)


def plot_single(metric_key: str, metric_label: str, filename: str, title: str):
    data = build_data(metric_key)
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_variance_plot(ax, data, metric_label, title)
    out = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Check how many seed runs are already available
    print("Checking available multi-seed results...")
    for method_prefix in ["cs_infobatch", "infobatch"]:
        for r in RATIOS:
            found = []
            for s in range(1, N_SEEDS + 1):
                d = RESULTS_DIR / f"{method_prefix}_r{r}_deeplabv3_s{s}"
                if d.is_dir() and list(d.glob("*_metrics.yaml")):
                    found.append(s)
            status = f"seeds {found}" if found else "MISSING"
            print(f"  {method_prefix:20s} r={r}  {status}")

    print()

    plot_single("class1_f1",
                "Tree F1  (class 1)",
                "fig_variance_f1.png",
                "DeepLabV3+: F1 vs Prune Ratio\n"
                "(CS-IB & InfoBatch: mean ± std over 5 seeds; Random Prune: single point)")

    plot_single("class1_iou",
                "Tree IoU  (class 1)",
                "fig_variance_iou.png",
                "DeepLabV3+: IoU vs Prune Ratio\n"
                "(CS-IB & InfoBatch: mean ± std over 5 seeds; Random Prune: single point)")


    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
