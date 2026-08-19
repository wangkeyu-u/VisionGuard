# Qwen3-VL historical Dev diagnostic failures

> Real-model diagnostic only. Mixed-provenance historical labels prevent this result from verifying the resume's three-model ablation or Dev F1=0.600 statement.

## Recomputed result

- Examples: 36
- Cache hits on the evidence rerun: 34
- Strict JSON parse failures: 2
- IoU-grounded F1 at 0.5 IoU: 0.2174
- Unsupported finding rate: 0.8214
- Negative-scene hallucination rate: 0.8462

## Strict JSON parse failures

- `00000239_jpg.rf.a41d71f80533740f61d5c59248364e77`
- `033896729-construction-workers-install-r_jpg.rf.5da6ce192b56bc573a5bb364214aa2c7`

The evaluator retains these failures as failed records instead of dropping them, so they reduce coverage and cannot improve the metric.
