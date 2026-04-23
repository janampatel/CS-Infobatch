# CS-InfoBatch — convenience commands
# Usage: make <target>
# Requires: conda activate cs-infobatch first

PYTHON   = python
CONFIG   = configs/baseline.yaml
GPU      = 0

# ── Single experiment shortcuts ────────────────────────────────────────────────

baseline:
	$(PYTHON) main.py --config configs/baseline.yaml --exp_id baseline_001

random_r05:
	$(PYTHON) main.py --config configs/random_prune.yaml --prune_ratio 0.5 --exp_id random_r05

ib_r05:
	$(PYTHON) main.py --config configs/infobatch.yaml --prune_ratio 0.5 --exp_id ib_r05

csib_r05:
	$(PYTHON) main.py --config configs/cs_infobatch.yaml --prune_ratio 0.5 --exp_id csib_r05

# ── Full experiment sweep (all 16 runs) ───────────────────────────────────────
# Run in tmux: tmux new -s sweep; make sweep_all

sweep_csib:
	for r in 0.3 0.5 0.7 0.8 0.9; do \
		$(PYTHON) main.py --config configs/cs_infobatch.yaml --prune_ratio $$r --exp_id csib_r$$r; \
	done

sweep_ib:
	for r in 0.3 0.5 0.7 0.8 0.9; do \
		$(PYTHON) main.py --config configs/infobatch.yaml --prune_ratio $$r --exp_id ib_r$$r; \
	done

sweep_random:
	for r in 0.3 0.5 0.7 0.8 0.9; do \
		$(PYTHON) main.py --config configs/random_prune.yaml --prune_ratio $$r --exp_id random_r$$r; \
	done

sweep_all: baseline sweep_random sweep_ib sweep_csib

# ── Utilities ─────────────────────────────────────────────────────────────────

# Show all results
results:
	find results/ -name "*_metrics.yaml" | sort | xargs -I{} sh -c 'echo "=== {} ==="; cat {}'

# Show per-epoch F1 progress for a given exp (usage: make progress EXP=csib_r0.5_cs_infobatch)
progress:
	@if [ -z "$(EXP)" ]; then echo "Usage: make progress EXP=<exp_name>"; exit 1; fi
	python -c "import csv, sys; \
	path = 'results/$(EXP)/$(EXP)_epoch_log.csv'; \
	f=open(path); \
	[print(f'epoch={row[\"epoch\"]:>3}  f1={float(row[\"val_class1_f1\"]):.4f}  samples={row[\"samples_this_epoch\"]}') for row in csv.DictReader(f)]; \
	f.close()"

# Clean pycache
clean:
	find . -type d -name __pycache__ -not -path "*/treefinder-main/*" -not -path "*/Infobatch-master/*" | xargs rm -rf

# Verify split JSON exists and is consistent
check_split:
	python -c "import json, glob; \
	files = glob.glob('splits/*.json'); \
	(print('No split found — run baseline first') if not files else \
	[print(f'Split: {s[\"n_train\"]} train / {s[\"n_val\"]} val / {s[\"n_test\"]} test\nfg_pixel_fraction: {s[\"fg_pixel_fraction\"]:.6f}\npos_tiles: {sum(s[\"cls_labels_train\"])}/{len(s[\"cls_labels_train\"])}') for s in [json.load(open(files[0]))]])"

.PHONY: baseline random_r05 ib_r05 csib_r05 sweep_csib sweep_ib sweep_random sweep_all results clean check_split
