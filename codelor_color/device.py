"""Runtime/platform helpers: device selection and Termux (Android) detection.

Training on a phone through Termux has a few constraints this module encodes:

* no CUDA, CPU only (Vulkan/NNAPI are not used for training);
* ``/dev/shm`` is missing, so ``DataLoader`` worker processes crash -> 0 workers;
* big.LITTLE CPUs: using every core is often *slower* than using the big ones;
* little RAM: small batches + gradient accumulation instead of large batches.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


def is_termux() -> bool:
    """Return ``True`` when running inside Termux on Android."""
    if os.environ.get("TERMUX_VERSION"):
        return True
    prefix = os.environ.get("PREFIX", "") + " " + sys.prefix
    return "com.termux" in prefix


def is_android() -> bool:
    """Return ``True`` on Android (Termux or another Python distribution)."""
    if sys.platform == "android" or is_termux():
        return True
    return Path("/system/build.prop").exists()


def has_shared_memory() -> bool:
    """Whether ``/dev/shm`` is usable (required by multi-process DataLoader workers)."""
    shm = Path("/dev/shm")
    return shm.is_dir() and os.access(shm, os.W_OK)


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Turn ``"auto"`` (or ``None``) into the best available device: cuda > mps > cpu."""
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def default_num_threads() -> int:
    """A sensible ``torch.set_num_threads`` value for the current machine.

    On Android, half of the cores (rounded up, max 4) usually maps to the big
    cores and gives the best throughput while keeping the phone responsive.
    """
    cores = os.cpu_count() or 1
    if is_android():
        return max(1, min(4, (cores + 1) // 2))
    return max(1, cores)


def default_num_workers() -> int:
    """DataLoader workers: 0 on Termux / without ``/dev/shm``, else up to 4."""
    if is_android() or not has_shared_memory():
        return 0
    return min(4, max(0, (os.cpu_count() or 1) - 1))


@dataclass(frozen=True)
class RuntimeProfile:
    """Recommended training settings for the detected platform."""

    platform: str
    device: str
    num_threads: int
    num_workers: int
    batch_size: int
    accum_steps: int
    img_size: int
    amp: bool


def recommended_profile(device: str | torch.device | None = "auto") -> RuntimeProfile:
    """Return training defaults adapted to the platform (Termux, CPU, GPU)."""
    dev = resolve_device(device)
    if is_android():
        return RuntimeProfile(
            platform="termux" if is_termux() else "android",
            device="cpu",
            num_threads=default_num_threads(),
            num_workers=0,
            batch_size=2,
            accum_steps=8,
            img_size=160,
            amp=False,
        )
    if dev.type == "cuda":
        return RuntimeProfile("cuda", "cuda", default_num_threads(), default_num_workers(),
                              16, 1, 256, True)
    return RuntimeProfile(dev.type, dev.type, default_num_threads(), default_num_workers(),
                          4, 4, 160, False)


def describe_environment() -> dict[str, object]:
    """Small diagnostic dictionary (used by ``codelor-color info``)."""
    try:
        import safetensors  # noqa: F401

        has_st = True
    except ImportError:
        has_st = False
    try:
        import PIL  # noqa: F401

        has_pil = True
    except ImportError:
        has_pil = False
    profile = recommended_profile()
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "termux": is_termux(),
        "android": is_android(),
        "device": profile.device,
        "cpu_count": os.cpu_count(),
        "shared_memory": has_shared_memory(),
        "safetensors": has_st,
        "pillow": has_pil,
        "profile": profile.__dict__,
    }
