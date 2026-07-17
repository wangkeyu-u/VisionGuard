from pathlib import Path

import pytest

from visionguard.inference import iter_batched_predictions


class FakeModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict(self, *, source: list[str], batch: int, stream: bool, **_: object):
        assert stream is True
        assert batch == len(source)
        self.batch_sizes.append(len(source))
        return iter(source)


def test_predictions_are_split_into_bounded_batches() -> None:
    model = FakeModel()
    paths = [Path(f"image-{index}.jpg") for index in range(5)]

    results = list(iter_batched_predictions(model, paths, 2, device="cpu"))

    assert results == [str(path) for path in paths]
    assert model.batch_sizes == [2, 2, 1]


def test_prediction_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list(iter_batched_predictions(FakeModel(), [Path("image.jpg")], 0))
