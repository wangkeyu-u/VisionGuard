# VLM Extension Guide: Your Next Stage

This stage is intentionally a guide rather than an implemented feature. The goal is for you to build and defend the multimodal part yourself in an interview.

## The project you should build

Turn VisionGuard from “boxes and class names” into a **grounded safety inspector**:

```text
image/video frame
    → YOLO detections and person/PPE crops
    → vision-language model
    → strict JSON finding
    → human-readable evidence and recommendation
```

The VLM should answer questions the detector cannot: Which person is missing which PPE? What visual evidence supports the finding? Is the evidence uncertain or occluded? It must not invent violations that are not visually grounded.

Recommended first scope: images only, English output, `no_helmet` and `no_vest`, one JSON schema, and no real-time claim.

## Phase 0 — Define the contract (half day)

Use one output schema from day one:

```json
{
  "scene_summary": "Two workers near construction equipment.",
  "findings": [
    {
      "person_box": [0.12, 0.08, 0.41, 0.92],
      "violation": "no_helmet",
      "evidence": "Head is visible and no helmet is present.",
      "confidence": "medium"
    }
  ],
  "uncertainties": ["Worker at right is heavily occluded."],
  "recommended_action": "Request human review."
}
```

Write a JSON Schema and reject/repair invalid output. Treat `person_box` as normalized `xyxy`. Allow an empty `findings` list and require explicit uncertainty when evidence is weak.

## Phase 1 — Build an evaluation set before fine-tuning (1–2 days)

Do not reuse the detector test set for prompt/model selection. From training/validation data, create a small VLM development set; reserve a new untouched multimodal test set from newly collected or licensed images.

Start with about 200–300 carefully reviewed examples balanced across:

- compliant workers;
- `no_helmet` and `no_vest`;
- both violations on one person;
- multiple people with different PPE states;
- occlusion, blur, small persons, unusual colors, and empty scenes;
- hard negatives such as hats, machinery, or vests held rather than worn.

For every example record: image ID, person box, violation labels, short evidence, uncertainty, source/license, and reviewer status. Have a second person review at least the final test portion if possible.

## Phase 2 — Establish three baselines (1–2 days)

Choose one compact open-weight instruct VLM that fits your available GPU and whose license permits your use. Model ecosystems change quickly, so record the exact model ID, revision, library versions, precision, and prompt.

Compare:

1. **VLM only** — full image plus task prompt.
2. **YOLO rules only** — current detector output converted to the same schema.
3. **YOLO-grounded VLM** — full image, annotated image or crops, and serialized detector boxes/classes.

This ablation is your interview story: it shows whether grounding actually reduces hallucination or improves minority violations.

## Phase 3 — Build detector-grounded inputs (2–3 days)

Add a script that exports one JSONL row per image:

```json
{"image":".../frame.jpg","detections":[{"class":"person","conf":0.91,"xyxy":[...]},{"class":"no_vest","conf":0.62,"xyxy":[...]}],"target":{...}}
```

Create person crops with 10–15% context padding. Associate PPE/violation boxes to people using a simple, documented baseline such as center-in-person plus IoU, then manually inspect its errors. Keep the raw full image available because a crop can remove context.

Do not silently use ground-truth boxes at inference time. If you use ground-truth boxes during training, label that condition clearly and evaluate with predicted YOLO boxes.

## Phase 4 — Fine-tune with LoRA only if the baseline justifies it (2–4 days)

Suggested stack: PyTorch, Transformers, PEFT, TRL, and the chosen model's official processor. Use a CUDA GPU for practical LoRA/QLoRA training; Apple MPS is fine for parts of preprocessing and small inference tests but is not the path I would choose for this fine-tuning run.

Training discipline:

- Freeze the test set before training.
- Fine-tune on structured assistant responses matching the JSON schema.
- Use LoRA/QLoRA, gradient accumulation, mixed precision, and early stopping.
- Mask prompt tokens if the trainer expects completion-only supervision.
- Save data/model/config revisions and seeds.
- Plot train/validation loss, but select using task metrics—not loss alone.
- Run at least two seeds for the final comparison if budget allows.

If 200 examples are insufficient, do not manufacture confidence with a large epoch count. Add adjudicated data through error-driven collection.

## Phase 5 — Evaluate the actual multimodal behavior (1–2 days)

Report more than prose quality:

| Dimension | Metric |
| --- | --- |
| Structure | valid JSON rate, schema-valid rate |
| Safety finding | micro/macro precision, recall, F1 per violation |
| Grounding | person-box IoU or correct-person association rate |
| Hallucination | unsupported finding rate on compliant/empty scenes |
| Uncertainty | abstention coverage and selective risk |
| Explanation | human-rated evidence correctness, not writing style |
| System | detector + VLM end-to-end P50/P95 latency and peak memory |

Use bootstrap confidence intervals where possible. Publish a table comparing VLM-only, YOLO-only, grounded VLM, and grounded VLM + LoRA. Include 20–30 categorized failure cases.

## Phase 6 — Connect it to the demo (half day)

Only after offline evaluation, add a second action to the existing demo: **Generate safety explanation**. Keep detector and VLM output visibly separate. Show evidence boxes, the raw JSON, model/version, and a “human review required” state. Never turn uncertain generated text into an automatic enforcement action.

## A realistic two-week schedule

| Day | Deliverable |
| --- | --- |
| 1 | schema, task definition, data policy |
| 2–3 | 200–300-example reviewed evaluation/development set |
| 4 | three zero-shot baselines |
| 5–6 | detector-to-person grounding and JSONL exporter |
| 7–9 | LoRA run plus one controlled retry |
| 10–11 | ablation metrics and confidence intervals |
| 12 | failure taxonomy and manual review |
| 13 | optional demo integration |
| 14 | README, model card, 3-minute interview walkthrough |

## What to say in an interview

Use this structure:

1. **Problem:** detector labels do not express person–equipment relationships or uncertainty.
2. **Hypothesis:** detector grounding reduces VLM hallucination and improves correct-person attribution.
3. **Method:** fixed schema, frozen test set, three baselines, LoRA only after zero-shot evidence.
4. **Evidence:** class metrics, grounding accuracy, unsupported-finding rate, latency, and ablations.
5. **Failure:** show one convincing miss and what data/system change it motivated.
6. **Boundary:** this is decision support with human review, not autonomous safety enforcement.

The strongest version of this work is not “I called a VLM API.” It is “I framed a measurable multimodal hypothesis, controlled leakage, grounded outputs, quantified hallucination, and knew where the system must abstain.”

## Definition of done

- Reproducible environment and exact model revision
- Versioned schema and prompt
- Licensed, reviewed train/dev/test manifest
- YOLO-only, VLM-only, and grounded-VLM baselines
- LoRA ablation if it materially helps
- Per-class, grounding, hallucination, latency, and memory metrics
- Failure gallery with an error taxonomy
- Model card updated with new limitations
- Short demo and a three-minute explanation you can deliver without notes
