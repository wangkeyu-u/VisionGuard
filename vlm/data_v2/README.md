# Test v2 human-review workspace

This directory is intentionally independent from `vlm/data`. It exists to replace the consumed
Test v1 benchmark whose person boxes did not consistently follow the full-visible-person contract.

1. Run `make vlm-test-v2-candidates` to select 40 source groups absent from all prior gold and review history.
2. Run `make vlm-review-v2 VLM_REVIEWER="Your Name"` and blind-review every Test candidate.
3. Draw one box around each complete visible person. Do not draw only the head, helmet, torso, or vest.
4. Run `make vlm-test-v2-gate`. Model evaluation remains blocked until this command reports `PASS`.
5. Run `make vlm-test-v2-once VLM_YOLO_DEVICE=cpu`. It reuses the frozen Dev-selected YOLO-only
   configuration and writes a consumption manifest before inference, so even a failed run cannot be retried.

`candidates.jsonl` is a queue, not gold data. Only `test.jsonl` becomes Test v2 gold after review.
