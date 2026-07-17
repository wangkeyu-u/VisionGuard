from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any


def iter_batched_predictions(
    model: Any,
    image_paths: Sequence[Path],
    batch_size: int,
    **predict_kwargs: Any,
) -> Iterator[Any]:
    """Run prediction in bounded batches instead of one in-memory image list."""
    if batch_size < 1:
        raise ValueError("Prediction batch size must be at least 1.")

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        results = model.predict(
            source=[str(path) for path in batch_paths],
            batch=len(batch_paths),
            stream=True,
            **predict_kwargs,
        )
        yielded = 0
        for result in results:
            yielded += 1
            yield result
        if yielded != len(batch_paths):
            raise RuntimeError(
                f"Prediction returned {yielded} results for a batch of {len(batch_paths)} images."
            )


def release_device_cache(device: str) -> None:
    """Release cached accelerator memory between inference stages."""
    if device == "mps":
        import torch

        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
