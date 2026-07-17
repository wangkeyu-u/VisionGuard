# VisionGuard VLM Extension

This directory adds a detector-grounded vision-language-model experiment without treating generated text as a safety decision. It is an evaluation-first extension: manually reviewed gold data, strict JSON output, three zero-shot baselines, and an optional CUDA-only LoRA stage.

## Experiment contract

The model receives an image and returns `findings`, `uncertainties`, and a review recommendation conforming to `schema.json`. A finding must associate `no_helmet` or `no_vest` with a normalized `0–1000` person box. Outputs are review candidates, not enforcement decisions.

The central ablation is:

| Mode | Input |
| --- | --- |
| `yolo` | YOLO detections converted to the common JSON schema |
| `vlm` | Original image and fixed prompt only |
| `grounded` | Image plus YOLO `person`/`helmet`/`vest` boxes |

The grounded mode deliberately excludes YOLO `no_helmet` and `no_vest` detections from the VLM prompt and annotated image. Otherwise the target label would leak into the input.

## Data rules

- `data/train.jsonl`, `dev.jsonl`, and `test.jsonl` contain only manually reviewed gold records.
- A source group may occur in only one split.
- `review_log.jsonl` records exclusions and ambiguous samples.
- `candidates.jsonl` is generated and ignored by Git. It is never treated as gold data.
- Do not inspect or tune on test predictions until the prompt/model/configuration is frozen.

Validate the current gold data:

```bash
make vlm-validate
```

Build a balanced review queue from the frozen YOLO dataset:

```bash
make vlm-candidates
```

Every candidate starts with `review_status=pending`. Review the image, assign left-to-right person IDs, draw visible person boxes, and either create a gold record or add an exclusion decision.

## Environment

Install optional dependencies into the existing environment with `uv`:

```bash
uv pip install --python .venv/bin/python -r requirements-vlm.txt
```

For a standalone environment:

```bash
uv venv --python 3.12 .venv-vlm
uv pip install --python .venv-vlm/bin/python -r requirements-vlm.txt
```

The default VLM is `Qwen/Qwen3-VL-2B-Instruct`. First use downloads several gigabytes of model files. Mac MPS is supported as a smoke-test path; CUDA is the required LoRA path.

## Baselines

Run one reviewed training image as a smoke test:

```bash
.venv/bin/python vlm/src/run_baseline.py \
  --split train --limit 1 --mode yolo --yolo-device cpu

.venv/bin/python vlm/src/run_baseline.py \
  --split train --limit 1 --mode vlm --device mps

.venv/bin/python vlm/src/run_baseline.py \
  --split train --limit 1 --mode grounded --device mps --yolo-device cpu
```

For a real comparison, populate dev data and run each mode without `--limit`, using distinct `--output` paths. Each JSONL row retains every raw model attempt, parsed JSON, bounded schema-repair count, latency, and detector context. On Apple Silicon, keep YOLO on CPU in grounded mode so YOLO and Qwen do not compete for MPS memory.

Evaluate one prediction file:

```bash
.venv/bin/python vlm/src/evaluate.py \
  --gold vlm/data/dev.jsonl \
  --predictions vlm/outputs/dev_vlm.jsonl
```

Metrics include JSON parse/schema rates, IoU-grounded precision/recall/F1 per violation, hallucination rate on clear images, bounded schema-repair success, runtime errors, and latency.

## LoRA

`src/train_lora.py` refuses to train unless CUDA is available and at least 20 train / 10 dev gold records exist. That guard prevents presenting a one-example overfit as a model experiment.

```bash
.venv/bin/python vlm/src/train_lora.py \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --epochs 3 --lora-rank 16 --lora-alpha 32
```

Use `--qlora` only in a CUDA environment with `bitsandbytes` installed. Freeze the final environment, seed, model revision, prompt, gold manifests, and adapter before running the held-out test once.

## Current status

The pipeline currently contains one reviewed training record and three reviewed development records. The test set remains empty and uninspected until the prompt, model, and configuration are frozen. See [`SMOKE_RESULTS.md`](SMOKE_RESULTS.md) for the deliberately small development smoke test; those numbers verify the experimental machinery and are not evidence of generalization.
