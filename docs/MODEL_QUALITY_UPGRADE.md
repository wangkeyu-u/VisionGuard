# Model quality upgrade: Exp5 safety tuning

## Why this experiment exists

The selected Exp4 detector reaches 0.4923 Validation mAP@50–95 and 0.4602 Test mAP@50–95, but the operational safety classes remain substantially weaker. On the independent Test evaluation, `no_helmet` AP@50–95 is 0.1392 and `no_vest` is 0.2197. The next experiment therefore optimizes the safety bottleneck rather than changing model size without a hypothesis.

## Train-only data intervention

The derived Exp5 dataset is generated from the verified frozen source fingerprint. It preserves every Validation and Test image and label byte-for-byte. Only Train is resampled:

| Train statistic | Frozen source | Exp5 derived |
| --- | ---: | ---: |
| Images | 6,125 | 7,311 |
| `no_helmet` annotations | 701 (2.29%) | 2,696 (7.25%) |
| `no_vest` annotations | 953 (3.11%) | 3,589 (9.65%) |

Rare violation images receive a 3× multiplier. Images with a rare box that is small, geometrically overlapped, or in a dense scene receive 4×. This is deterministic resampling, not new semantic content, and should be described as such.

## Training recipe

- Initialize from Exp4 `best.pt`, not from a new random/COCO start.
- Fine-tune for 15 epochs at 640 px.
- Use `cls_pw=0.5`, the square-root inverse-frequency class weighting supported by the installed Ultralytics runtime.
- Save every third epoch so checkpoint selection is not limited to global `best.pt` and `last.pt`.
- Keep seed 42 and all unrelated augmentation defaults unchanged.

The ablation switches are:

| Recipe | Rare resampling | `cls_pw=0.5` | 640 px |
| --- | --- | --- | --- |
| `rare_only` | yes | no | no |
| `weighted_only` | no | yes | no |
| `resolution_only` | no | no | yes |
| `safety_tune` | yes | yes | yes |

## Selection contract

Checkpoint selection uses Validation only. The primary metric is the harmonic mean of `no_helmet` and `no_vest` AP@50–95, which prevents a model from winning by improving one violation class while collapsing the other. A run is eligible only if overall Validation mAP@50–95 regresses by no more than 0.01 from Exp4.

The Exp4 baseline already demonstrates why task-aligned selection matters:

| Checkpoint | Overall mAP | `no_helmet` AP | `no_vest` AP | Safety H-mean |
| --- | ---: | ---: | ---: | ---: |
| `best.pt` | 0.4920 | 0.2133 | 0.3013 | 0.2498 |
| `last.pt` | 0.4887 | 0.2129 | 0.3184 | **0.2552** |

`last.pt` is weaker on global mAP but better for the balanced safety objective. It is copied to `best_safety.pt`; this does not change the previously published Test result, which still belongs to Exp4 `best.pt`.

## Completed Exp5 run

The combined `safety_tune` recipe completed all 15 epochs on Apple Silicon MPS with batch size 2.
Training took 10,669.7 seconds (2 h 57 min 50 s), and the derived dataset fingerprint verified all
9,948 files before training. Seven saved checkpoints were then evaluated on the unchanged Validation
split; Test was not used.

| Run / selected checkpoint | Overall mAP | `no_helmet` AP | `no_vest` AP | Safety H-mean | Guardrail |
| --- | ---: | ---: | ---: | ---: | --- |
| Exp4 `last.pt` | **0.4887** | **0.2129** | **0.3184** | **0.2552** | pass |
| Exp5 `epoch3.pt` | 0.4543 | 0.1855 | 0.3052 | 0.2308 | fail |

Exp5 changed the safety H-mean by **−0.0244** and overall mAP@50–95 by **−0.0344** relative to
Exp4. It therefore fails the predeclared maximum overall regression of 0.01 and is rejected. Exp4
remains the selected detector, and no new Test evaluation is permitted or needed for Exp5.

This result does not establish which of resampling, class weighting, or 640 px input caused the
regression because the combined recipe changes all three. It does show that repeating rare images is
not equivalent to collecting new rare-class evidence and that a safety-specific selection metric plus
an overall-quality guardrail prevented promotion of a worse model.

## Reproduction

```bash
make quality-train QUALITY_DEVICE=0 QUALITY_RECIPE=safety_tune
make quality-select QUALITY_DEVICE=0 QUALITY_RUN=outputs/experiments/exp5_safety_tune
make quality-report QUALITY_RUN=outputs/experiments/exp5_safety_tune
```

Run the three isolated ablations before making a causal claim about which intervention caused the
regression. Do not evaluate rejected recipes on Test.

## Interview framing

The defensible claim is: the system identified a rare-class bottleneck, built a leakage-safe
Train-only intervention, completed the combined experiment, and rejected it using a predeclared
task-aligned metric and global-quality guardrail. Do not claim that oversampling improved the model.
The useful engineering outcome is that the evaluation contract caught a plausible but harmful recipe
before Test or deployment.
