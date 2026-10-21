from ._version import __version__
from .model import (
    DEFAULT_REF_RGB,
    CodelorColor,
    load_from_pretrained,
    predict_mask,
    remove_green,
)

__all__ = [
    "DEFAULT_REF_RGB",
    "CodelorColor",
    "load_from_pretrained",
    "predict_mask",
    "remove_green",
    "__version__",
]
