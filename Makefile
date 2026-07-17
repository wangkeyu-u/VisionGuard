PYTHON := .venv/bin/python
BOOTSTRAP_PYTHON ?= python3.12
DATA ?=
OUTPUT_DIR ?= outputs/reports
VIS_OUTPUT_DIR ?= outputs/visualization
REMAP_OUTPUT_DIR ?= training/datasets/safety_clean
FINAL_OUTPUT_DIR ?= training/datasets/safety_final
FINALIZE_FLAGS ?=
SAMPLES_PER_SPLIT ?= 8
SEED ?= 42
IMGSZ ?= 512
EPOCHS ?= 30
MODEL ?= yolo11n.pt
EVAL_BATCH ?= 1
ERROR_BATCH ?= 1
ERROR_DEVICE ?= mps
EXPERIMENT ?= baseline_yolo11n_512
EXPERIMENT_DIR ?= outputs/experiments/$(EXPERIMENT)
FINAL_DATA ?= training/datasets/safety_final/data.yaml
BEST_MODEL ?= $(EXPERIMENT_DIR)/weights/best.pt
VLM_SPLIT ?= train
VLM_MODE ?= vlm
VLM_LIMIT ?= 1
VLM_PREDICTIONS ?= vlm/outputs/$(VLM_SPLIT)_$(VLM_MODE).jsonl
VLM_DEVICE ?= auto
VLM_YOLO_DEVICE ?= cpu
VLM_SCHEMA_RETRIES ?= 1

.PHONY: setup install environment validate remap finalize visualize train resume evaluate benchmark errors report comparison demo vlm-install vlm-validate vlm-candidates vlm-baseline vlm-evaluate vlm-train baseline test lint compile check

setup:
	$(BOOTSTRAP_PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt
	.venv/bin/python -m pip install --no-deps -e .

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps -e .

environment:
	$(PYTHON) scripts/check_environment.py

validate:
	@test -n "$(DATA)" || (echo "Error: provide DATA, for example: make validate DATA=/path/to/data.yaml"; exit 2)
	$(PYTHON) scripts/validate_dataset.py --data "$(DATA)" --output-dir "$(OUTPUT_DIR)"

remap:
	@test -n "$(DATA)" || (echo "Error: provide DATA, for example: make remap DATA=training/datasets/safety/data.yaml"; exit 2)
	$(PYTHON) scripts/remap_dataset_classes.py --data "$(DATA)" --output-dir "$(REMAP_OUTPUT_DIR)"

finalize:
	@test -n "$(DATA)" || (echo "Error: provide DATA, for example: make finalize DATA=training/datasets/safety_clean/data.yaml"; exit 2)
	$(PYTHON) scripts/finalize_dataset.py --data "$(DATA)" --output-dir "$(FINAL_OUTPUT_DIR)" --seed "$(SEED)" $(FINALIZE_FLAGS)

visualize:
	@test -n "$(DATA)" || (echo "Error: provide DATA, for example: make visualize DATA=/path/to/data.yaml"; exit 2)
	$(PYTHON) scripts/visualize_dataset.py --data "$(DATA)" --output-dir "$(VIS_OUTPUT_DIR)" \
		--samples-per-split "$(SAMPLES_PER_SPLIT)" --seed "$(SEED)"

train:
	$(PYTHON) scripts/train_yolo.py --data "$(FINAL_DATA)" --model $(MODEL) --epochs $(EPOCHS) \
		--imgsz $(IMGSZ) --batch 4 --patience 8 --workers 0 --seed 42 --device mps \
		--project outputs/experiments --name "$(EXPERIMENT)"

resume:
	$(PYTHON) scripts/resume_yolo.py --checkpoint "$(EXPERIMENT_DIR)/weights/last.pt" \
		--data "$(FINAL_DATA)" --device mps

evaluate:
	$(PYTHON) scripts/evaluate_yolo.py --model "$(BEST_MODEL)" --data "$(FINAL_DATA)" \
		--output-dir "$(EXPERIMENT_DIR)/test_evaluation" --imgsz $(IMGSZ) --batch $(EVAL_BATCH) --workers 0 --device mps

benchmark:
	$(PYTHON) scripts/benchmark_yolo.py --model "$(BEST_MODEL)" --data "$(FINAL_DATA)" \
		--output-dir "$(EXPERIMENT_DIR)/performance" --imgsz $(IMGSZ) --device mps --samples 100 --warmup 5

errors:
	$(PYTHON) scripts/analyze_yolo_errors.py --model "$(BEST_MODEL)" --data "$(FINAL_DATA)" \
		--output-dir "$(EXPERIMENT_DIR)/error_analysis" --imgsz $(IMGSZ) --device $(ERROR_DEVICE) \
		--batch $(ERROR_BATCH)

report:
	$(PYTHON) scripts/generate_experiment_report.py --run-dir "$(EXPERIMENT_DIR)" \
		--data "$(FINAL_DATA)" --dataset-report training/datasets/safety_final/finalization_report.json

comparison:
	$(PYTHON) scripts/generate_comparison_report.py --experiments \
		outputs/experiments/baseline_yolo11n_512 \
		outputs/experiments/exp2_yolo11n_640 \
		outputs/experiments/exp3_yolo11s_512 \
		outputs/experiments/exp4_yolo11s_512_e50 \
		--output-dir outputs/reports/experiment_comparison

demo:
	$(PYTHON) scripts/run_demo.py --model "$(BEST_MODEL)" --imgsz $(IMGSZ) --device auto

vlm-install:
	uv pip install --python $(PYTHON) -r requirements-vlm.txt

vlm-validate:
	$(PYTHON) vlm/src/validate_data.py

vlm-candidates:
	$(PYTHON) vlm/src/build_review_manifest.py --per-split 40

vlm-baseline:
	$(PYTHON) vlm/src/run_baseline.py --split $(VLM_SPLIT) --mode $(VLM_MODE) \
		--limit $(VLM_LIMIT) --device $(VLM_DEVICE) --yolo-device $(VLM_YOLO_DEVICE) \
		--schema-retries $(VLM_SCHEMA_RETRIES) --output $(VLM_PREDICTIONS)

vlm-evaluate:
	$(PYTHON) vlm/src/evaluate.py --gold vlm/data/$(VLM_SPLIT).jsonl \
		--predictions $(VLM_PREDICTIONS)

vlm-train:
	$(PYTHON) vlm/src/train_lora.py

baseline: train evaluate benchmark errors report

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src scripts tests vlm/src

compile:
	$(PYTHON) -m compileall -q src scripts tests vlm/src

check: environment lint compile test
