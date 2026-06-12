#!/usr/bin/env python3
"""
Analysis script for Dataset 2 (Sentinel-2 paired tiles) experiment results.

Sibling of evaluate/analyze_results.py (TreeFinder). It reuses that script's
generic figure/table functions and only overrides what differs for Dataset 2:
  * results/ directory naming  (see DS2_ARCHES below)
  * fig4 y-limits              (DS2 F1 sits near 0.94, not <0.72)
  * fig5 full-dataset size     (N_FULL = 1872, not TreeFinder's 11152)

Run from project root:   python evaluate/analyze_results_dataset2.py

Per-architecture outputs:
    Mask2Former  -> results_analysis/dataset2/mask2former/
    DeepLabV3+   -> results_analysis/dataset2/deeplabv3/
Cross-architecture:
    results_analysis/dataset2/comparison/

Dataset 2 results/ directory naming
-----------------------------------
  Mask2Former : baseline_baseline_ds2
                {cs_infobatch,infobatch,random}_r{ratio}_dataset2
  DeepLabV3+  : baseline_deeplabv3_ds2
                {cs_infobatch,infobatch,random}_r{ratio}_deeplabv3_ds2

NOTE on labels: the reused figures inherit "Tree F1 (class 1)" / "NAIP" wording
from the TreeFinder script. Dataset 2 is a different (balanced) Sentinel-2 task,
so read those axis labels as "foreground (class 1)".
"""

import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import the TreeFinder analysis module as a library of reusable functions.
sys.path.insert(0, str(Path(__file__).parent))
import analyze_results as ar  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# DATASET 2 CONFIG
# ──────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = ar.RESULTS_DIR
OUTPUT_ROOT = ar.OUTPUT_ROOT / "dataset2"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

METHODS = ar.METHODS          # ["CS-InfoBatch", "InfoBatch", "Random Prune"]
RATIOS  = ar.RATIOS           # [0.3, 0.5, 0.7, 0.8, 0.9]
N_FULL_DS2 = 1872             # train tiles/epoch for Dataset 2

# match(dir_name) -> bool : does this results/ dir belong to the architecture?
DS2_ARCHES = {
    "mask2former": {
        "label":        "Mask2Former (DS2)",
        "out_subdir":   "mask2former",
        "baseline_dir": "baseline_baseline_ds2",
        "id_token":     "dataset2",
        "match":        lambda n: n == "baseline_baseline_ds2" or n.endswith("_dataset2"),
    },
    "deeplabv3": {
        "label":        "DeepLabV3+ (DS2)",
        "out_subdir":   "deeplabv3",
        "baseline_dir": "baseline_deeplabv3_ds2",
        "id_token":     "deeplabv3_ds2",
        "match":        lambda n: n == "baseline_deeplabv3_ds2" or n.endswith("_deeplabv3_ds2"),
    },
}


def build_exp_names(arch_cfg: dict) -> dict:
    """Maps (method_label, ratio) -> results/ dir name (== prob-file stem)."""
    mapping = {("Baseline", 0.0): arch_cfg["baseline_dir"]}
    for method in METHODS:
        for ratio in RATIOS:
            mapping[(method, ratio)] = (
                f"{ar.PRUNER_PREFIX[method]}_r{ratio}_{arch_cfg['id_token']}"
            )
    return mapping


def load_all_runs(arch_cfg: dict):
    """Dataset-2 variant of ar.load_all_runs using arch_cfg['match']."""
    rows, logs = [], {}
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir() or not arch_cfg["match"](d.name):
            continue
        method, ratio = ar._parse_dir(d.name)
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


# ──────────────────────────────────────────────────────────────────────────────
# OVERRIDDEN FIGURES (the two that are hardcoded for TreeFinder)
# ──────────────────────────────────────────────────────────────────────────────

def fig4_learning_curves(df, logs, path, arch_label):
    """Val-F1 learning curves with Dataset-2-appropriate y-limits (auto-scaled)."""
    b_f1 = df.loc[df.method == "Baseline", "class1_f1"].values[0]
    baseline_log = logs.get(("Baseline", 0.0))

    # Auto y-range from all available val-F1 traces, with a little padding.
    all_vals = []
    for log in logs.values():
        if log is not None and "val_class1_f1" in log:
            all_vals.extend(log.val_class1_f1.dropna().tolist())
    if all_vals:
        lo = max(0.0, min(all_vals) - 0.03)
        hi = min(1.0, max(all_vals) + 0.02)
    else:
        lo, hi = 0.0, 1.0

    n = len(RATIOS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{arch_label}: Validation F1 Learning Curves (by Prune Ratio)", fontsize=13)

    for j, ratio in enumerate(RATIOS):
        ax = axes[j]
        if baseline_log is not None:
            ax.plot(baseline_log.epoch, baseline_log.val_class1_f1,
                    color=ar.COLORS["Baseline"], linestyle=ar.LINES["Baseline"],
                    linewidth=1.5, alpha=0.6, label="Baseline")
        ax.axhline(b_f1, color=ar.COLORS["Baseline"], linestyle=":", linewidth=0.8, alpha=0.35)
        for method in METHODS:
            log = logs.get((method, ratio))
            if log is None:
                continue
            ax.plot(log.epoch, log.val_class1_f1,
                    color=ar.COLORS[method], linestyle=ar.LINES[method],
                    linewidth=1.8, label=method)
        ax.set_title(f"Prune Ratio r={ratio}")
        ax.set_xlabel("Epoch")
        if j == 0:
            ax.set_ylabel("Val F1  (class 1)")
        ax.set_ylim(lo, hi)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Fig 4 saved -> {path}")


def fig5_samples_per_epoch(df, logs, path, arch_label):
    """Pruning dynamics with Dataset-2 full-dataset size N_FULL=1872."""
    N_FULL = N_FULL_DS2
    n = len(RATIOS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{arch_label}: Samples per Epoch During Training (Pruning Dynamics)", fontsize=13)

    for j, ratio in enumerate(RATIOS):
        ax = axes[j]
        ax.axhline(N_FULL, color="gray", linestyle="--",
                   linewidth=0.9, alpha=0.5, label="Full dataset")
        for method in METHODS:
            log = logs.get((method, ratio))
            if log is None:
                continue
            ax.plot(log.epoch, log.samples_this_epoch,
                    color=ar.COLORS[method], linestyle=ar.LINES[method],
                    linewidth=2, label=method, alpha=0.9)
            ax.fill_between(log.epoch, log.samples_this_epoch,
                            color=ar.COLORS[method], alpha=0.08)
        target = int(N_FULL * (1 - ratio))
        ax.axhline(target, color="black", linestyle=":", linewidth=0.8,
                   alpha=0.35, label=f"Target ({target})")
        ax.set_title(f"Prune Ratio r={ratio}")
        ax.set_xlabel("Epoch")
        if j == 0:
            ax.set_ylabel("Samples / Epoch")
        ax.set_ylim(0, N_FULL * 1.15)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Fig 5 saved -> {path}")


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
        print(f"  {r.method:<16} r={r.prune_ratio}  F1={r.class1_f1:.4f}  "
              f"time={r.train_time_h:.2f}h  epochs={int(r.epochs_trained)}")

    ar.print_summary(df, arch_label)

    # Tables (reused; LaTeX caption inherits TreeFinder wording — cosmetic)
    ar.save_csv(df.copy(), out_dir / "results_table.csv")
    ar.save_latex(df,      out_dir / "results_table.tex", arch_label)

    # Core figures — fig1/2/3/6/7 reused verbatim; fig4/5 overridden for DS2
    ar.fig1_f1_vs_ratio        (df,       out_dir / "fig1_f1_vs_ratio.png",         arch_label)
    ar.fig2_time_vs_ratio      (df,       out_dir / "fig2_time_vs_ratio.png",       arch_label)
    ar.fig3_pareto             (df,       out_dir / "fig3_pareto.png",              arch_label)
    fig4_learning_curves       (df, logs, out_dir / "fig4_learning_curves.png",     arch_label)
    fig5_samples_per_epoch     (df, logs, out_dir / "fig5_samples_per_epoch.png",   arch_label)
    ar.fig6_degradation_heatmap(df,       out_dir / "fig6_degradation_heatmap.png", arch_label)
    ar.fig7_efficiency         (df,       out_dir / "fig7_efficiency.png",          arch_label)

    # Prob-based figures — skipped gracefully if no .npz files exist for DS2
    ar.fig8_roc_auc    (df, out_dir / "fig8_roc_auc.png",     exp_map, arch_label)
    ar.fig9_calibration(df, out_dir / "fig9_calibration.png", exp_map, arch_label)
    ar.fig10_prob_dist (df, out_dir / "fig10_prob_dist.png",  exp_map, arch_label)

    return df


def main():
    print(f"Results dir : {RESULTS_DIR}")
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"Probs dir   : {ar.PROBS_DIR}  "
          f"({'found' if ar.PROBS_DIR.exists() else 'not found — figs 8-10 will be skipped'})")

    dfs = {}
    for arch_key, arch_cfg in DS2_ARCHES.items():
        df = analyze_arch(arch_key, arch_cfg)
        if df is not None:
            # Use clean labels for the cross-arch overlay (strip " (DS2)")
            clean = arch_cfg["label"].replace(" (DS2)", "")
            dfs[clean] = df

    # Cross-architecture overlays into results_analysis/dataset2/comparison/
    dfs = {k: v for k, v in dfs.items() if v is not None and not v.empty}
    if len(dfs) >= 2:
        out_dir = OUTPUT_ROOT / "comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "#" * 70)
        print(f"# CROSS-ARCHITECTURE COMPARISON  ({' vs '.join(dfs.keys())})")
        print(f"#   output -> {out_dir}")
        print("#" * 70)
        ar.figC1_f1_vs_ratio_by_method(dfs, out_dir / "figC1_f1_vs_ratio_by_method.png")
        ar.figC2_joint_pareto         (dfs, out_dir / "figC2_joint_pareto.png")
        ar.figC3_arch_gap_heatmap     (dfs, out_dir / "figC3_arch_gap_heatmap.png")
    else:
        print("\nCross-architecture comparison skipped — need >= 2 architectures with data.")

    print("\nDone. Outputs in:")
    for arch_cfg in DS2_ARCHES.values():
        print(f"  {OUTPUT_ROOT / arch_cfg['out_subdir']}")
    print(f"  {OUTPUT_ROOT / 'comparison'}  (cross-architecture overlays)")


if __name__ == "__main__":
    main()