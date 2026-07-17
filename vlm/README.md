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

Every candidate starts with `review_status=pending`. Review the image, assign left-to-right person IDs,
draw boxes around the complete visible person (not only a head, helmet, torso, or vest), and either
create a gold record or add an exclusion decision.

Start the local human-review desk instead of editing JSONL by hand:

```bash
make vlm-review VLM_REVIEWER="Your Name"
```

Open [http://127.0.0.1:7861](http://127.0.0.1:7861). The desk shows the balanced queue and Train/Dev/Test targets, overlays YOLO-format source boxes, seeds editable person boxes when available, and supports:

- drawing, moving, resizing, and deleting normalized person boxes;
- confirming `no_helmet` and `no_vest` per person with evidence and confidence;
- marking compliant samples and recording uncertainty or quality tags;
- preserving every reviewed person box, including people with no violation finding;
- requiring an explicit full-visible-person contract confirmation after every geometry edit;
- excluding dirty data with a required audit reason;
- server-side target/person-registry consistency validation and atomic writes to gold JSONL.

Test candidates are always shown in blind-review mode. The UI hides source categories and all YOLO boxes so the held-out set is built from the original image without exposing model-derived signals.

### Test v2 recovery protocol

Test v1 is consumed and immutable. A separate 40-image Test v2 queue has been generated from source
groups absent from every prior gold record and review decision:

```bash
make vlm-test-v2-candidates
make vlm-review-v2 VLM_REVIEWER="Your Name"
# Open http://127.0.0.1:7862 and blind-review the Test queue.
make vlm-test-v2-gate
```

The gate blocks evaluation unless Test v2 has at least 20 records, human reviewer provenance,
`full-visible-person-v2` metadata, a consistent person-box registry, no source-group reuse, and a
median person-box area reasonably aligned with reviewed Dev geometry. The current gate is expected
to report `BLOCKED` while `data_v2/test.jsonl` is empty. Only a `PASS` permits freezing and running
the already selected configuration once via `make vlm-test-v2-once VLM_YOLO_DEVICE=cpu`. The runner
writes `RUN_MANIFEST.json` before inference and refuses any retry or overwrite. See
[`data_v2/README.md`](data_v2/README.md).

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

The pipeline now contains 41 Train, 36 Dev, and 40 blind-reviewed Test records. The frozen formal
Dev ablation selected YOLO-only (F1 0.600) over Qwen3-VL-only (0.362) and YOLO-grounded
Qwen3-VL (0.235). See
[`outputs/dev_formal/DEV_ABLATION_REPORT.md`](outputs/dev_formal/DEV_ABLATION_REPORT.md).

The selected YOLO-only configuration was run exactly once on Test. Image-level violation presence
F1 was 0.735, but strict person-grounding F1 was 0.027 because Test v1 gold boxes were typically
PPE-part boxes rather than complete visible-person boxes. Test v1 remains frozen and was not rerun.
See [`outputs/test_final/FINAL_REPORT.md`](outputs/test_final/FINAL_REPORT.md) for the full diagnosis
and the Test v2 protocol. The replacement Test v2 queue contains 40 pending, source-independent
candidates; it has not been evaluated.
