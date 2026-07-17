from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class EnvironmentInfo:
    python_version: str
    platform: str
    machine: str
    pytorch_version: str
    torchvision_version: str
    ultralytics_version: str
    mps_built: bool
    mps_available: bool
    selected_device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_module(module_name: str) -> tuple[ModuleType | None, str]:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None, "not installed"
    except Exception as exc:  # Environment diagnostics must survive broken native packages.
        return None, f"import failed ({type(exc).__name__}: {exc})"
    return module, str(getattr(module, "__version__", "unknown"))


def collect_environment_info() -> EnvironmentInfo:
    """Inspect the local ML runtime without failing when optional packages are absent."""
    torch, pytorch_version = _load_module("torch")
    _, torchvision_version = _load_module("torchvision")
    _, ultralytics_version = _load_module("ultralytics")
    mps_built = False
    mps_available = False

    if torch is not None:
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None:
            mps_built = bool(mps_backend.is_built())
            mps_available = bool(mps_backend.is_available())

    return EnvironmentInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        pytorch_version=pytorch_version,
        torchvision_version=torchvision_version,
        ultralytics_version=ultralytics_version,
        mps_built=mps_built,
        mps_available=mps_available,
        selected_device="mps" if mps_available else "cpu",
    )


def print_environment_info(info: EnvironmentInfo) -> None:
    rows = (
        ("Python", info.python_version),
        ("Platform", info.platform),
        ("Machine", info.machine),
        ("PyTorch", info.pytorch_version),
        ("torchvision", info.torchvision_version),
        ("Ultralytics", info.ultralytics_version),
        ("MPS built", str(info.mps_built)),
        ("MPS available", str(info.mps_available)),
        ("Selected device", info.selected_device),
    )
    width = max(len(label) for label, _ in rows)
    print("VisionGuard environment")
    print("=" * 40)
    for label, value in rows:
        print(f"{label:<{width}} : {value}")


def main() -> int:
    info = collect_environment_info()
    print_environment_info(info)
    if info.pytorch_version == "not installed" or info.ultralytics_version == "not installed":
        print("\nWarning: install the missing ML dependencies before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
