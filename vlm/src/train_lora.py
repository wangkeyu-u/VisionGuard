#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.vlm import read_jsonl, validate_inspection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3-VL on reviewed VisionGuard data.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--train-data", type=Path, default=PROJECT_ROOT / "vlm/data/train.jsonl")
    parser.add_argument("--dev-data", type=Path, default=PROJECT_ROOT / "vlm/data/dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "vlm/outputs/qwen3_vl_2b_lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Use 4-bit loading; requires bitsandbytes and a supported CUDA GPU.",
    )
    return parser.parse_args()


def _load_gold(path: Path) -> list[dict[str, Any]]:
    records, issues = read_jsonl(path.resolve())
    if issues:
        raise ValueError("\n".join(str(issue) for issue in issues))
    for index, record in enumerate(records, start=1):
        if "image" not in record or "target" not in record:
            raise ValueError(f"{path}:{index}: image and target are required")
        target_errors = validate_inspection(record["target"])
        if target_errors:
            raise ValueError(f"{path}:{index}: {'; '.join(target_errors)}")
    return records


def _training_rows(records: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        image_path = (PROJECT_ROOT / record["image"]).resolve()
        rows.append(
            {
                "images": [str(image_path)],
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "completion": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    record["target"],
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            }
                        ],
                    },
                ],
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"PyTorch is required. Original error: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "LoRA training is intentionally CUDA-only in this project. Use the Mac for data/baselines "
            "and run this entry point on a CUDA machine."
        )
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoProcessor
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "VLM training dependencies are missing. Install requirements-vlm.txt in a CUDA environment. "
            f"Original error: {exc}"
        ) from exc
    train_records = _load_gold(args.train_data)
    dev_records = _load_gold(args.dev_data)
    if len(train_records) < 20 or len(dev_records) < 10:
        raise SystemExit(
            f"Refusing an uninformative fine-tune: found {len(train_records)} train and "
            f"{len(dev_records)} dev records; require at least 20/10 reviewed records."
        )

    prompt = (PROJECT_ROOT / "vlm/prompts/safety_inspection.txt").read_text(encoding="utf-8").strip()
    train_dataset = Dataset.from_list(_training_rows(train_records, prompt))
    dev_dataset = Dataset.from_list(_training_rows(dev_records, prompt))
    processor = AutoProcessor.from_pretrained(args.model_id)
    model_init_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    }
    if args.qlora:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit("--qlora requires bitsandbytes-compatible Transformers support") from exc
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_init_kwargs["dtype"],
            bnb_4bit_use_double_quant=True,
        )

    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    training = SFTConfig(
        output_dir=str(args.output_dir.resolve()),
        model_init_kwargs=model_init_kwargs,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        max_length=None,
        completion_only_loss=True,
        assistant_only_loss=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=args.model_id,
        args=training,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=processor,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir.resolve()))
    processor.save_pretrained(str(args.output_dir.resolve()))
    print(f"Saved LoRA adapter and processor to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
