#!/usr/bin/env python3
"""Regenerate table.md and table2.md from results/*_metrics.yaml.

Run from the repository root:
    python scripts/build_tables.py

The `Samples/Epoch` column is MEASURED (total_samples_seen / epochs_trained),
never inferred from prune_ratio. Only Random Prune actually removes
prune_ratio x N; InfoBatch applies it to the below-mean subset only, and
CS-InfoBatch additionally ramps it in over epochs, so at a nominal r=0.9 the
three methods train on roughly 0.10 N, 0.27 N and 0.86 N respectively.

Also emits the DeepLabV3+ 5-seed table with paired t-tests.
"""
import os, re, yaml
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
N_FULL = {"ds1": 11152, "ds2": 1872}

def load_all():
    rows = []
    for d in sorted(os.listdir(RES)):
        p = os.path.join(RES, d, f"{d}_metrics.yaml")
        if not os.path.exists(p):
            continue
        m = yaml.safe_load(open(p))
        ds = "ds2" if (d.endswith("_ds2") or d.endswith("_dataset2")) else "ds1"
        arch = "deeplabv3" if "deeplabv3" in d else "mask2former"
        seed = re.search(r"_s(\d)$", d)
        seed = int(seed.group(1)) if seed else None
        if d.startswith("baseline"):
            meth, ratio = "baseline", None
        else:
            meth = d.split("_r")[0]
            ratio = float(re.search(r"_r([0-9.]+)_", d).group(1))
        ep, tot = m.get("epochs_trained"), m.get("total_samples_seen")
        rows.append(dict(
            run=d, ds=ds, arch=arch, method=meth, ratio=ratio, seed=seed,
            f1=m["class1_f1"], iou=m["class1_iou"],
            rec=m["class1_recall"], prec=m["class1_precision"],
            c0f1=m.get("class0_f1"), c0iou=m.get("class0_iou"),
            c0rec=m.get("class0_recall"), c0prec=m.get("class0_precision"),
            acc=m.get("overall_accuracy"), macro=m.get("res_f1_macro"),
            epochs=ep, total=tot, hours=m["train_time_seconds"] / 3600,
            spe=tot / ep, frac=(tot / ep) / N_FULL[ds],
        ))
    return pd.DataFrame(rows)

LABEL = {"baseline": "Our baseline (no IB)", "random": "Random Prune",
         "infobatch": "InfoBatch", "cs_infobatch": "CS-InfoBatch"}
ORDER = {"random": 0, "infobatch": 1, "cs_infobatch": 2}


def representative(df, ds, arch):
    """One row per (method, ratio).

    Uses the un-suffixed runs everywhere. On DeepLabV3+/TreeFinder these are the
    original seed-2025 runs -- the ones with saved per-pixel predictions, so the
    tables and figs 8-10 describe the same run. The _s1.._s5 sweep is reported
    separately in the 5-seed section below.
    """
    sub = df[(df.ds == ds) & (df.arch == arch)]
    return sub[sub.seed.isna()]


def fg_block(df, ds, arch, paper=None):
    sub = representative(df, ds, arch)
    out = ["| prune_ratio | Mode | class1_F1 | class1_IoU | class1_Recall | class1_Precision "
           "| Train Time (h) | Samples/Epoch (measured) | Total Samples |",
           "|---|---|---|---|---|---|---|---|---|"]
    if paper:
        out.append(f"| — (100%) | TreeFinder paper (no IB) | {paper[0]} | {paper[1]} | "
                   f"{paper[2]} | {paper[3]} | — | — | — |")
    b = sub[sub.method == "baseline"]
    for _, r in b.iterrows():
        out.append(f"| — (100%) | {LABEL['baseline']} | {r.f1:.4f} | {r.iou:.4f} | {r.rec:.4f} | "
                   f"{r.prec:.4f} | {r.hours:.2f} | 1.00 N | {int(r.total):,} |")
    for ratio in [0.3, 0.5, 0.7, 0.8, 0.9]:
        blk = sub[sub.ratio == ratio].copy()
        blk["o"] = blk.method.map(ORDER)
        for _, r in blk.sort_values("o").iterrows():
            out.append(f"| {ratio} | {LABEL[r.method]} | {r.f1:.4f} | {r.iou:.4f} | {r.rec:.4f} | "
                       f"{r.prec:.4f} | {r.hours:.2f} | {r.frac:.2f} N | {int(r.total):,} |")
    return "\n".join(out)


def bg_block(df, ds, arch):
    sub = representative(df, ds, arch)
    out = ["| prune_ratio | Mode | class0_F1 | class0_IoU | class0_Recall | class0_Precision "
           "| Overall Acc | Macro F1 |", "|---|---|---|---|---|---|---|---|"]
    for _, r in sub[sub.method == "baseline"].iterrows():
        out.append(f"| — (100%) | {LABEL['baseline']} | {r.c0f1:.4f} | {r.c0iou:.4f} | {r.c0rec:.4f} | "
                   f"{r.c0prec:.4f} | {r.acc:.4f} | {r.macro:.4f} |")
    for ratio in [0.3, 0.5, 0.7, 0.8, 0.9]:
        blk = sub[sub.ratio == ratio].copy()
        blk["o"] = blk.method.map(ORDER)
        for _, r in blk.sort_values("o").iterrows():
            out.append(f"| {ratio} | {LABEL[r.method]} | {r.c0f1:.4f} | {r.c0iou:.4f} | {r.c0rec:.4f} | "
                       f"{r.c0prec:.4f} | {r.acc:.4f} | {r.macro:.4f} |")
    return "\n".join(out)


def prune_rate_block(df, ds):
    out = ["| Nominal prune_ratio | Random | InfoBatch | CS-InfoBatch |", "|---|---|---|---|"]
    sub = pd.concat([representative(df, ds, a) for a in ["mask2former", "deeplabv3"]])
    for ratio in [0.3, 0.5, 0.7, 0.8, 0.9]:
        cells = []
        for meth in ["random", "infobatch", "cs_infobatch"]:
            v = sub[(sub.ratio == ratio) & (sub.method == meth)].frac
            cells.append(f"{v.mean():.2f} N" if len(v) else "—")
        out.append(f"| {ratio} (claims {1-ratio:.1f} N) | " + " | ".join(cells) + " |")
    return "\n".join(out)


def seed_block(df):
    ms = df[(df.ds == "ds1") & (df.arch == "deeplabv3") & df.seed.notna()]
    out = ["| prune_ratio | CS-InfoBatch (mean ± std) | InfoBatch (mean ± std) | Δ (paired) | Seeds won | p |",
           "|---|---|---|---|---|---|"]
    for ratio in [0.3, 0.5, 0.7, 0.8, 0.9]:
        a = ms[(ms.method == "cs_infobatch") & (ms.ratio == ratio)].set_index("seed").f1.sort_index()
        b = ms[(ms.method == "infobatch") & (ms.ratio == ratio)].set_index("seed").f1.sort_index()
        common = a.index.intersection(b.index)
        a, b = a[common], b[common]
        d = a - b
        t, p = stats.ttest_rel(a, b)
        out.append(f"| {ratio} | {a.mean():.4f} ± {a.std(ddof=1):.4f} | {b.mean():.4f} ± {b.std(ddof=1):.4f} "
                   f"| {d.mean():+.4f} | {int((d>0).sum())}/{len(d)} | {p:.3f} |")
    return "\n".join(out)


def seed_raw_block(df):
    ms = df[(df.ds == "ds1") & (df.arch == "deeplabv3") & df.seed.notna()]
    piv = ms.pivot_table(index=["method", "ratio"], columns="seed", values="f1")
    out = ["| Method | r | s1 (2025) | s2 (42) | s3 (123) | s4 (7) | s5 (99) | spread |",
           "|---|---|---|---|---|---|---|---|"]
    for meth in ["cs_infobatch", "infobatch"]:
        for ratio in [0.3, 0.5, 0.7, 0.8, 0.9]:
            v = piv.loc[(meth, ratio)]
            cells = " | ".join(f"{v[s]:.4f}" for s in [1, 2, 3, 4, 5])
            out.append(f"| {LABEL[meth]} | {ratio} | {cells} | {v.max()-v.min():.4f} |")
    return "\n".join(out)


df = load_all()

# ── table.md ────────────────────────────────────────────────────────────────
t1 = f"""# Dataset 1 — TreeFinder (NAIP, severely imbalanced)

`N = 11,152` train tiles. Pixel imbalance ≈ 268:1 background:foreground
(`fg_pixel_fraction = 0.003722`), so `w_fg = 134.3`, `w_bg = 0.502`.

> **Read the `Samples/Epoch` column, not `prune_ratio`.** `prune_ratio` is a
> configuration knob, and the three methods respond to it very differently.
> Random Prune removes exactly `prune_ratio × N`. InfoBatch and CS-InfoBatch
> apply it only to samples scoring below the (per-class) mean, and CS-InfoBatch
> additionally ramps it in linearly from epoch 10 to epoch 87. Because early
> stopping fires long before epoch 87, CS-InfoBatch never reaches its nominal
> rate. `Samples/Epoch (measured)` = `total_samples_seen / epochs_trained`,
> divided by N.

## Measured pruning — what each method actually removes

{prune_rate_block(df, "ds1")}

*Only Random Prune matches its label. Comparing the three methods at a fixed
`prune_ratio` compares different compute budgets; compare at matched
`Total Samples` or `Train Time` instead.*

---

# Mask2Former

## Class 1 (foreground)

{fg_block(df, "ds1", "mask2former", paper=("0.519", "0.351", "0.494", "0.555"))}

## Class 0 (background)

{bg_block(df, "ds1", "mask2former")}

---

# DeepLabV3+

## Class 1 (foreground)

{fg_block(df, "ds1", "deeplabv3", paper=("0.497", "0.334", "0.499", "0.496"))}

## Class 0 (background)

{bg_block(df, "ds1", "deeplabv3")}

---

# DeepLabV3+ — 5-seed study

Seeds 2025, 42, 123, 7, 99 (`scripts/run_deeplab_multiseed.sh`). The seed drives
`_split_key()` as well as initialisation, so each seed is a fresh split *and* a
fresh init. CS-InfoBatch and InfoBatch are therefore paired within a seed.

{seed_block(df)}

*Paired two-sided t-test on class1_F1, n = 5.*

## Per-seed values

{seed_raw_block(df)}

**Seed variance dominates the method difference.** The seed-to-seed spread is
0.08–0.12 F1; the CS-IB − IB gap is 0.006–0.016 and changes sign twice. No prune
ratio reaches p < 0.09. Single-seed rows in the tables above should not be read
as method differences.

Baseline and Random Prune are single runs and have no error bars, so
"pruning beats the baseline" is not currently supported on DeepLabV3+.
"""

# ── table2.md ───────────────────────────────────────────────────────────────
t2 = f"""# Dataset 2 (Sentinel-2 paired tiles)

`N = 1,872` train tiles. Unlike TreeFinder (268:1 pixel imbalance), Dataset 2 is
near-balanced — `fg_pixel_fraction = 0.6118` (BG:FG ≈ 0.6:1), 98.3% of tiles
contain foreground. CS-InfoBatch's class-stratified weights therefore degenerate
(`w_fg = 0.82 < w_bg = 1.29`), so CS-IB is expected to track plain InfoBatch here.
This serves as a **balanced-regime control**: it shows the pruning methods do no
harm when imbalance is absent.

> **Same caveat as Dataset 1:** `prune_ratio` is a knob, not a measurement.

## Measured pruning — what each method actually removes

{prune_rate_block(df, "ds2")}

---

# Mask2Former

## Class 1 (foreground)

{fg_block(df, "ds2", "mask2former")}

## Class 0 (background)

{bg_block(df, "ds2", "mask2former")}

---

# DeepLabV3+

## Class 1 (foreground)

{fg_block(df, "ds2", "deeplabv3")}

## Class 0 (background)

{bg_block(df, "ds2", "deeplabv3")}

---

## Takeaways

- **Both architectures score high** (class1 F1 ≈ 0.87–0.96): the balanced dataset
  is far easier than TreeFinder, so absolute numbers are not comparable across
  datasets. DeepLabV3+ is the stronger model here (baseline 0.9598 vs
  Mask2Former 0.9386).
- **All three pruners are near-lossless on this dataset.** Even Random Prune at
  r=0.9 — a true 10% of the data — holds class1 F1 at 0.9523 on DeepLabV3+
  (−0.0075 vs baseline). The task is easy enough that pruning barely matters.
- **The CS-IB weighting correctly becomes inert**, as predicted from
  `w_fg = 0.82 < w_bg = 1.29`. This is the control working as designed.
- **CS-IB's apparent lead is a compute difference, not robustness.** At r=0.9 on
  DeepLabV3+, CS-IB trained on 0.95 N per epoch, InfoBatch on 0.47 N, Random on
  0.10 N — and the F1 ordering is exactly that ordering. Compare at matched
  `Total Samples` before claiming robustness.
- **Random Prune degrades monotonically** with prune ratio on both models — the
  intended lower-bound control, and the only method whose x-axis label is
  literally true.
"""

for path, txt in [(os.path.join(ROOT, "table.md"), t1), (os.path.join(ROOT, "table2.md"), t2)]:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    print("wrote", path, len(txt), "chars")
