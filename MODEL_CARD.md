# VisionGuard Model Card

> Evidence warning: all performance and latency values below are historical project records, not
> independently verified in the current checkout. The checkpoint, four training `results.csv`
> files, test metrics JSON, and performance JSON are missing. Run `make evidence-gate`; until it
> passes, these values are `implemented_unverified` rather than verified evidence.

## Model

- Architecture: Ultralytics YOLO11s object detector
- Input size: 512×512
- Training schedule: 50 epochs maximum; early stopping completed after epoch 48
- Selected checkpoint: best validation checkpoint at epoch 40
- Classes: `person`, `helmet`, `vest`, `gloves`, `boots`, `no_helmet`, `no_vest`
- Training device: Apple Silicon MPS
- Dataset: frozen 70/20/10 split described in `DATASET_CARD.md`

Weights are intentionally excluded from Git. The expected local path is `outputs/experiments/exp4_yolo11s_512_e50/weights/best.pt`.

## Real Qwen3-VL diagnostic boundary

A real `Qwen/Qwen3-VL-2B-Instruct` diagnostic was run at pinned revision
`89644892e4d85e24eaac8bacfd4f463576704203` on 36 recovered historical Dev records. The resumed
evidence pass used 34 cached predictions and retained two strict JSON parse failures. Recomputing
from record-level outputs at IoU 0.5 gives precision 0.1786, recall 0.2778, and F1 0.2174.

This is not the resume's claimed three-model ablation: the selected project YOLO checkpoint and the
grounded-Qwen run are absent, and the recovered labels mix named human, Codex-assisted, and missing
reviewer provenance. It therefore cannot support Dev F1=0.600. The machine-readable report records
the pinned model revision, local weight hash, source-label hash, raw predictions, parse failures,
and the exact recomputation boundary in `docs/evidence/qwen_real_historical_result.json`.

## Selection and evaluation

Four configurations were compared. Selection used validation mAP@50–95 only. The independent test set was not used to choose the checkpoint.

| Metric | Validation | Independent test |
| --- | ---: | ---: |
| Precision | 0.7214 | 0.7027 |
| Recall | 0.7111 | 0.6683 |
| mAP@50 | 0.6986 | 0.6615 |
| mAP@50–95 | 0.4923 | 0.4602 |

Test class metrics:

| Class | Precision | Recall | AP@50 | AP@50–95 |
| --- | ---: | ---: | ---: | ---: |
| person | 0.718 | 0.888 | 0.858 | 0.775 |
| helmet | 0.884 | 0.780 | 0.849 | 0.536 |
| vest | 0.912 | 0.916 | 0.945 | 0.690 |
| gloves | 0.499 | 0.504 | 0.395 | 0.261 |
| boots | 0.916 | 0.804 | 0.860 | 0.601 |
| no_helmet | 0.495 | 0.347 | 0.336 | 0.139 |
| no_vest | 0.495 | 0.440 | 0.388 | 0.220 |

The recorded batch-1 MPS benchmark is 23.61 ms/image mean and 31.02 ms/image P95 after five warm-up images on 100 sorted test images. This excludes upload, decoding, rendering, network, and queueing time.

## Intended use

- Offline analysis of construction/PPE imagery
- Human-reviewed safety-screening prototypes
- Education and portfolio demonstrations of end-to-end CV engineering
- A detector/grounding component in a future multimodal research pipeline

## Out-of-scope use

- Safety certification or a claim that a scene is safe
- Unsupervised enforcement, disciplinary, employment, or legal decisions
- Identity recognition or worker tracking
- Deployment on an unseen site without local validation and threshold calibration

## Output interpretation

The demo uses a configurable confidence threshold, default 0.25. Image counts are detected boxes. Video counts are detection events accumulated across frames; they are not unique people because the system does not track identities. `no_violation_detected` means only that no `no_helmet` or `no_vest` box exceeded the threshold.

## Risks and limitations

- Violation recall is low: the test recall is 0.347 for `no_helmet` and 0.440 for `no_vest`.
- Each experiment has one seed, so improvements are not yet separated from training variance.
- Minority test metrics have high uncertainty because the relevant classes occur in few images.
- Occlusion, small objects, unusual PPE, lighting, camera angle, blur, and domain shift can cause errors.
- Positive equipment detections and negative violation detections are modeled as separate boxes; the detector does not explicitly reason about person–equipment ownership.
- No probability calibration, fairness audit, privacy review, robustness suite, or target-device acceptance test has been completed.

## Production gates

Before deployment: define cost-weighted safety metrics; collect representative site data; adjudicate labels; repeat training across seeds; calibrate thresholds per class; evaluate person–PPE association; test video tracking; add abstention and human escalation; measure end-to-end latency; monitor drift; and complete legal, privacy, and worker-impact reviews.
