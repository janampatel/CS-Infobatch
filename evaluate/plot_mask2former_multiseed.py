#!/usr/bin/env python3
"""
Mask2Former / TreeFinder multi-seed figures (seeds 2025, 42, 123), produced by
scripts/run_mask2former_multiseed.sh.

Run from project root:
    python evaluate/plot_mask2former_multiseed.py

Outputs (saved to results_analysis/mask2former/):
    fig_multiseed_f1.png    — Tree F1 vs prune ratio: CS-InfoBatch and InfoBatch, a dot per
                              seed with a min→max line; baseline mean ± std as a band
    fig_matched_budget.png  — Tree F1 vs measured samples/epoch: InfoBatch and CS-InfoBatch with
                              the short ramp (matched per-epoch budget), baseline at 1.0 N

Run directories (s1 = 2025 is the un-suffixed run for the main methods):
    baseline_baseline, baseline_baseline_s2, baseline_baseline_s3
    {cs_infobatch,infobatch}_r{ratio}_{cs_infobatch,infobatch}[_s2|_s3]
    cs_infobatch_r{ratio}_cs_infobatch_short_ramp_s{1,2,3}
"""

import csv
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR  = Path(__file__).parent.parent / "results_analysis" / "mask2former"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RATIOS      = [0.3, 0.5, 0.7, 0.8, 0.9]
RAMP_RATIOS = [0.5, 0.7, 0.9]
N_TRAIN     = 11152
RAMP_END    = 20   # epochs after warmup + short ramp; budget is averaged over later epochs

COLORS = {
    "CS-InfoBatch": "#1565C0",
    "InfoBatch":    "#E65100",
    "Baseline":     "#2E7D32",
}
MARKERS = {
    "CS-InfoBatch": "o",
    "InfoBatch":    "s",
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

def run_names(method: str, ratio=None):
    """Directory names for the three seeds of one setting."""
    if method == "Baseline":
        return ["baseline_baseline", "baseline_baseline_s2", "baseline_baseline_s3"]
    if method == "CS-InfoBatch (short ramp)":
        return [f"cs_infobatch_r{ratio}_cs_infobatch_short_ramp_s{s}" for s in (1, 2, 3)]
    pre = "cs_infobatch" if method == "CS-InfoBatch" else "infobatch"
    base = f"{pre}_r{ratio}_{pre}"
    return [base, f"{base}_s2", f"{base}_s3"]


def load_run(name: str):
    """Returns (test class1_f1, mean samples/epoch after RAMP_END as a fraction of N), or None."""
    d = RESULTS_DIR / name
    m, e = d / f"{name}_metrics.yaml", d / f"{name}_epoch_log.csv"
    if not (m.exists() and e.exists()):
        return None
    with open(m) as f:
        f1 = float(yaml.safe_load(f)["class1_f1"])
    with open(e) as f:
        rows = list(csv.DictReader(f))
    spe = np.mean([int(r["samples_this_epoch"]) for r in rows[RAMP_END:]]) / N_TRAIN
    return f1, spe


def load_setting(method: str, ratio=None):
    runs = [load_run(n) for n in run_names(method, ratio)]
    runs = [r for r in runs if r is not None]
    return np.array([r[0] for r in runs]), np.array([r[1] for r in runs])

# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_multiseed_f1():
    fig, ax = plt.subplots(figsize=(8, 6))

    b, _ = load_setting("Baseline")
    ax.axhspan(b.mean() - b.std(ddof=1), b.mean() + b.std(ddof=1),
               color=COLORS["Baseline"], alpha=0.10, zorder=0)
    ax.axhline(b.mean(), color=COLORS["Baseline"], linestyle="-.", linewidth=1.5, zorder=1,
               label=f"Baseline ({b.mean():.3f} ± {b.std(ddof=1):.3f})")

    # methods are dodged left/right of each ratio so their seeds never overlap
    for method, dx, label_xy, ha in [("CS-InfoBatch", -0.012, (-7, 3), "right"),
                                     ("InfoBatch",     0.012, (7, 3),  "left")]:
        color, marker = COLORS[method], MARKERS[method]
        xs, means = [], []
        for r in RATIOS:
            vals, _ = load_setting(method, r)
            if len(vals) == 0:
                continue
            x = r + dx
            xs.append(x)
            means.append(vals.mean())
            ax.plot([x, x], [vals.min(), vals.max()], color=color, linewidth=1.8,
                    zorder=3, solid_capstyle="round")
            ax.scatter([x] * len(vals), vals, color=color, marker=marker, s=60,
                       zorder=5, edgecolors="white", linewidths=0.6)
            ax.annotate(f"{vals.mean():.3f}", (x, vals.mean()), textcoords="offset points",
                        xytext=label_xy, ha=ha, fontsize=8, color=color)
        ax.plot(xs, means, color=color, linestyle="--", linewidth=1.2, alpha=0.55,
                zorder=2, label=f"{method} (mean)")

    ax.set_xlabel("Prune Ratio")
    ax.set_ylabel("Tree F1  (class 1)")
    ax.set_title("Mask2Former: F1 vs Prune Ratio\n"
                 "(a dot per seed — 2025, 42, 123; line spans min→max)")
    ax.set_xticks(RATIOS)
    ax.set_xlim(min(RATIOS) - 0.09, max(RATIOS) + 0.07)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.25)
    out = OUTPUT_DIR / "fig_multiseed_f1.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_matched_budget():
    fig, ax = plt.subplots(figsize=(8, 6))

    b, _ = load_setting("Baseline")
    ax.errorbar([1.0], [b.mean()], yerr=[b.std(ddof=1)], color=COLORS["Baseline"],
                marker=MARKERS["Baseline"], markersize=13, capsize=4, linestyle="none",
                zorder=5, label="Baseline (no pruning)")

    # ratio labels: InfoBatch to the right of its points, CS-InfoBatch to the left
    series = [
        ("InfoBatch",                 COLORS["InfoBatch"],    MARKERS["InfoBatch"], "full", "--",
         (7, -12), "left"),
        ("CS-InfoBatch (short ramp)", COLORS["CS-InfoBatch"], "D",                  "none", "-",
         (-8, -12), "right"),
    ]
    for label, color, marker, fill, ls, label_xy, ha in series:
        xs, ys = [], []
        for r in RAMP_RATIOS:
            f1, spe = load_setting(label, r)
            if len(f1) == 0:
                continue
            xs.append(spe.mean())
            ys.append(f1.mean())
            ax.errorbar([spe.mean()], [f1.mean()], yerr=[f1.std(ddof=1)], color=color,
                        marker=marker, markersize=8, capsize=4, linestyle="none", zorder=5,
                        markerfacecolor=color if fill == "full" else "white",
                        markeredgewidth=1.6)
            ax.annotate(f"r={r}", (spe.mean(), f1.mean()), textcoords="offset points",
                        xytext=label_xy, ha=ha, fontsize=8, color="#444444")
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], color=color, linestyle=ls,
                linewidth=1.2, alpha=0.6, zorder=2)
        ax.plot([], [], color=color, linestyle=ls, marker=marker, markersize=7,
                markerfacecolor=color if fill == "full" else "white", label=label)

    ax.set_xlabel("Measured samples per epoch after epoch 20  (fraction of N)")
    ax.set_ylabel("Tree F1  (class 1)")
    ax.set_title("Mask2Former: F1 at matched per-epoch budget\n"
                 "(mean ± std over seeds 2025, 42, 123)")
    ax.set_xlim(0.15, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)
    out = OUTPUT_DIR / "fig_matched_budget.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Checking available multi-seed results...")
    settings = [("Baseline", None)] + [(m, r) for m in ("CS-InfoBatch", "InfoBatch") for r in RATIOS] \
        + [("CS-InfoBatch (short ramp)", r) for r in RAMP_RATIOS]
    for method, r in settings:
        n = len(load_setting(method, r)[0])
        print(f"  {method:28s} r={r}  seeds found: {n}/3")
    print()
    plot_multiseed_f1()
    plot_matched_budget()
    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
