from ._version import __version__
from .device import is_termux, recommended_profile, resolve_device
from .model import (
    DEFAULT_REF_RGB,
    TASKS,
    CodelorBackground,
    CodelorColor,
    build_model,
    load_from_pretrained,
    predict_mask,
    remove_background,
    remove_green,
    save_pretrained,
    transfer_weights,
)

__all__ = [
    "DEFAULT_REF_RGB",
    "TASKS",
    "CodelorBackground",
    "CodelorColor",
    "build_model",
    "is_termux",
    "load_from_pretrained",
    "predict_mask",
    "recommended_profile",
    "remove_background",
    "remove_green",
    "resolve_device",
    "save_pretrained",
    "transfer_weights",
    "__version__",
]
