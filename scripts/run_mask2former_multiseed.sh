#!/usr/bin/env bash
# Mask2Former / TreeFinder multi-seed study (seeds 2025, 42, 123).
#
#   --set main        baseline, InfoBatch and CS-InfoBatch for seeds 42 (s2) and 123 (s3).
#                     Seed 2025 (s1) is the default config seed, i.e. the un-suffixed runs
#                     from the single-seed sweep in SETUP.md Step 7.
#   --set short_ramp  CS-InfoBatch with the linear ramp ending at epoch 20
#                     (configs/cs_infobatch_short_ramp.yaml) for seeds s1-s3. The ramp then
#                     completes before early stopping, so CS-InfoBatch trains on the same
#                     per-epoch budget as InfoBatch at equal prune_ratio (matched budget).
#
# Same convention as the DeepLabV3+ 5-seed study: s1 = 2025, s2 = 42, s3 = 123, and --seed
# also selects the split (utils/split_cache._split_key). Results land in results/<run name>/.
#
# Jobs are in priority order. A run counts as done once results/<name>/<name>_metrics.yaml
# exists; done runs are skipped, so on a new reservation (with the finished results/ copied
# back in) the script continues with the remaining runs. A run cut off mid-training restarts
# from epoch 1. The summary table at the end covers both sets.
#
# Usage (from project root):
#   bash scripts/run_mask2former_multiseed.sh                    # --set main, one job at a time
#   bash scripts/run_mask2former_multiseed.sh --set short_ramp
#   bash scripts/run_mask2former_multiseed.sh --parallel 2       # two jobs share the GPU

GPU=0
PARALLEL=1
SET=main
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)      GPU=$2;      shift 2 ;;
        --parallel) PARALLEL=$2; shift 2 ;;
        --set)      SET=$2;      shift 2 ;;
        *) shift ;;
    esac
done
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

# config | prune_ratio (- = none) | run name prefix | seed | seed label
JOBS=()
if [[ $SET == main ]]; then
    for S in "42 s2" "123 s3"; do
        set -- $S
        JOBS+=("configs/baseline.yaml - baseline_baseline $1 $2")
    done
    for R in 0.9 0.5 0.7 0.3 0.8; do              # priority: headline ratios first
        for S in "42 s2" "123 s3"; do
            set -- $S
            JOBS+=("configs/cs_infobatch.yaml $R cs_infobatch_r${R}_cs_infobatch $1 $2")
            JOBS+=("configs/infobatch.yaml    $R infobatch_r${R}_infobatch       $1 $2")
        done
    done
elif [[ $SET == short_ramp ]]; then
    for R in 0.9 0.5 0.7; do
        for S in "2025 s1" "42 s2" "123 s3"; do
            set -- $S
            JOBS+=("configs/cs_infobatch_short_ramp.yaml $R cs_infobatch_r${R}_cs_infobatch_short_ramp $1 $2")
        done
    done
else
    echo "unknown --set '$SET' (expected: main | short_ramp)"; exit 1
fi

run_job() {
    read -r cfg ratio prefix seed slabel <<< "$1"
    local id; id=$(python -c "import yaml;print(yaml.safe_load(open('$cfg'))['experiment']['id'])")
    local name="${prefix}_${slabel}"
    if [[ -f "results/${name}/${name}_metrics.yaml" ]]; then echo "[skip] $name"; return; fi
    local args=(--config "$cfg" --exp_id "${id}_${slabel}" --seed "$seed" --gpu_id "$GPU")
    [[ $ratio != - ]] && args+=(--prune_ratio "$ratio")
    echo "[start $(date +%H:%M)] $name"
    python main.py "${args[@]}" > "logs/${name}.stdout" 2>&1
    echo "[end   $(date +%H:%M)] $name rc=$?"
}
export -f run_job
export GPU

# Create the seed-42/123 split files once, before any parallel job could race on them.
python - <<'EOF' || { echo "split prep failed"; exit 1; }
import sys; sys.path.insert(0, ".")
from utils.tools import load_config
from utils.split_cache import get_or_create_split
cfg = load_config("configs/baseline.yaml")
for seed in (42, 123):
    cfg["experiment"]["seed"] = seed
    get_or_create_split(cfg)
EOF

printf '%s\n' "${JOBS[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'run_job "$1"' _ {}

# Test dead-tree F1 per seed, mean ± std, and measured samples/epoch after the ramp (epoch > 20).
python - <<'EOF'
import csv, os, yaml, numpy as np
N = 11152
def load(name):
    p = f"results/{name}/{name}_metrics.yaml"
    if not os.path.exists(p):
        return None, None
    rows = list(csv.DictReader(open(f"results/{name}/{name}_epoch_log.csv")))
    spe = np.mean([int(r["samples_this_epoch"]) for r in rows[20:]] or [np.nan]) / N
    return yaml.safe_load(open(p))["class1_f1"], spe
rows = [("Baseline", ["baseline_baseline", "baseline_baseline_s2", "baseline_baseline_s3"])]
for r in (0.3, 0.5, 0.7, 0.8, 0.9):
    for lab, pre in (("InfoBatch", "infobatch"), ("CS-InfoBatch", "cs_infobatch")):
        base = f"{pre}_r{r}_{pre}"
        rows.append((f"{lab} r={r}", [base, base + "_s2", base + "_s3"]))
    sr = f"cs_infobatch_r{r}_cs_infobatch_short_ramp"
    rows.append((f"CS-IB short ramp r={r}", [sr + "_s1", sr + "_s2", sr + "_s3"]))
print(f"\n{'Setting':26s} {'s1(2025)':>9s} {'s2(42)':>9s} {'s3(123)':>9s}   {'mean ± std':17s} samples/epoch (ep>20)")
for lab, names in rows:
    got = [load(n) for n in names]
    f1 = [g[0] for g in got]
    if all(x is None for x in f1):
        continue
    vals = [x for x in f1 if x is not None]
    cells = " ".join(f"{x:9.4f}" if x is not None else f"{'-':>9s}" for x in f1)
    ms = f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}" if len(vals) > 1 else "-"
    spe = np.nanmean([g[1] for g in got if g[1] is not None])
    print(f"{lab:26s} {cells}   {ms:17s} {spe:.2f}N")
EOF
