#!/usr/bin/env python3
"""Check that every saved prediction file agrees with its run's metrics.yaml.
Guards against a retrained run's results/ being updated while a stale .npz
from generate_probs.py (which skips runs that already have one) survives.

Run AFTER generate_probs.py:  python scripts/verify_probs_consistency.py

Recomputes class1_f1 from each .npz at threshold 0.5 and compares against
metrics.yaml (default tolerance 0.02). Exits non-zero if anything is
inconsistent; modifies no files.
"""
import argparse
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBS = os.path.join(ROOT, "results_analysis", "probs")
RESULTS = os.path.join(ROOT, "results")


def f1_from_npz(path):
    z = np.load(path)
    p = z["probs"].astype(np.float32)
    y = z["labels"].astype(np.int8)
    pred = p > 0.5
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn + 1e-9), len(p), float(y.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.02,
                    help="max |F1(predictions) - F1(metrics.yaml)| (default 0.02)")
    args = ap.parse_args()

    if not os.path.isdir(PROBS):
        print(f"No probs directory at {PROBS}")
        return 0

    runs = sorted(d for d in os.listdir(RESULTS)
                  if os.path.exists(os.path.join(RESULTS, d, f"{d}_metrics.yaml")))

    ok, bad, orphan, absent = [], [], [], []
    for run in runs:
        npz = os.path.join(PROBS, f"{run}_probs.npz")
        if not os.path.exists(npz):
            absent.append(run)
            continue
        reported = yaml.safe_load(
            open(os.path.join(RESULTS, run, f"{run}_metrics.yaml"))).get("class1_f1")
        got, n, pos = f1_from_npz(npz)
        (ok if abs(got - reported) <= args.tol else bad).append((run, reported, got, got - reported))

    known = set(runs)
    for f in sorted(os.listdir(PROBS)):
        if f.endswith("_probs.npz") and f[:-len("_probs.npz")] not in known:
            orphan.append(f)

    print(f"predictions checked : {len(ok) + len(bad)}")
    print(f"  consistent        : {len(ok)}")
    print(f"  INCONSISTENT      : {len(bad)}")
    print(f"runs with no .npz   : {len(absent)}")
    print(f"orphaned .npz       : {len(orphan)}")

    if bad:
        print(f"\nINCONSISTENT (|delta| > {args.tol}) -- do not use these in figures:")
        print(f"  {'run':40s} {'metrics.yaml':>13s} {'predictions':>12s} {'delta':>9s}")
        for run, rep, got, d in sorted(bad, key=lambda r: -abs(r[3])):
            print(f"  {run:40s} {rep:13.4f} {got:12.4f} {d:+9.4f}")
        print("\n  Fix: delete the .npz and re-run")
        print("       python evaluate/generate_probs.py --only <run>")
        print("  (it regenerates only files that are absent)")

    if orphan:
        print("\nOrphaned .npz (no matching results/ directory):")
        for f in orphan:
            print(f"  {f}")

    if absent:
        print(f"\nRuns without predictions ({len(absent)}) -- figs 8-10 will skip them:")
        for r in absent[:15]:
            print(f"  {r}")
        if len(absent) > 15:
            print(f"  ... and {len(absent) - 15} more")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
