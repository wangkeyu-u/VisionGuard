# VLM Development Smoke Test

This is an engineering smoke test, not a model-quality claim. It uses only three manually reviewed development images: two violation images containing three gold findings, plus one clear compliant image.

## Results

| Mode | Precision | Recall | F1 | Clear-image hallucination | Schema repairs | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO-only | 1.000 | 0.667 | 0.800 | 0/1 | 0/3 | 292 ms |
| Qwen3-VL-only | 0.400 | 0.667 | 0.500 | 1/1 | 2/3 | 19,175 ms |
| YOLO-grounded Qwen3-VL | 0.250 | 0.333 | 0.286 | 1/1 | 3/3 | 13,293 ms |

At IoU 0.3, YOLO-only missed one `no_helmet` finding. The VLM-only run found two of three findings but added three false positives. Grounding did not improve this tiny slice: it found one finding and added three false positives. All repaired records retain both the initial invalid text and corrected text, so the repair rate is visible rather than hidden.

## Configuration

- Date: 2026-07-17
- Hardware: MacBook Air, Apple M5, 16 GB unified memory
- VLM: `Qwen/Qwen3-VL-2B-Instruct`, deterministic decoding, 384 maximum new tokens
- Detector: selected YOLO11s experiment checkpoint, image size 512
- Devices: MPS for Qwen; MPS for YOLO-only; CPU for YOLO in grounded mode
- Evaluator: class match plus person-box IoU >= 0.3
- Schema correction: at most one text-only retry; every attempt is logged

Reproduce after installing `requirements-vlm.txt`:

```bash
.venv/bin/python vlm/src/run_baseline.py \
  --split dev --mode yolo --yolo-device mps \
  --output vlm/outputs/dev_yolo.jsonl

.venv/bin/python vlm/src/run_baseline.py \
  --split dev --mode vlm --device mps --max-new-tokens 384 \
  --output vlm/outputs/dev_vlm.jsonl

.venv/bin/python vlm/src/run_baseline.py \
  --split dev --mode grounded --device mps --yolo-device cpu \
  --max-new-tokens 384 --output vlm/outputs/dev_grounded.jsonl
```

Evaluate each output with `vlm/src/evaluate.py`. Do not select prompts or checkpoints from these numbers and then describe the same three records as an unbiased test. A reportable experiment requires a larger adjudicated development set, a frozen configuration, and one untouched held-out test run.
