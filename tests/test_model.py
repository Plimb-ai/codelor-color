"""Test suite for the Codelor Color architecture."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from codelor_color import (
    DEFAULT_REF_RGB,
    CodelorColor,
    __version__,
    load_from_pretrained,
    predict_mask,
    remove_green,
)

ROOT = Path(__file__).resolve().parents[1]


def small_model(**kwargs) -> CodelorColor:
    params: Dict[str, Any] = dict(
        dim=32, depth=2, heads=4, tied_after=1, max_tokens=64, drop_path_rate=0.1
    )
    params.update(kwargs)
    return CodelorColor(**params)


# ---------------------------------------------------------------------------
# Version / packaging consistency
# ---------------------------------------------------------------------------

def test_version_single_source_of_truth() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    version_py = (ROOT / "codelor_color" / "_version.py").read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', version_py)
    assert match, "no __version__ in _version.py"
    assert match.group(1) == __version__
    assert 'attr = "codelor_color._version.__version__"' in pyproject


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def test_forward_shapes() -> None:
    model = small_model().eval()
    img = torch.rand(2, 3, 64, 64)
    ref = torch.rand(2, 3)
    with torch.no_grad():
        out = model(img, ref)
        main, aux = model(img, ref, return_aux=True)
    assert out.shape == (2, 1, 64, 64)
    assert main.shape == (2, 1, 64, 64)
    assert aux.shape == (2, 1, 16, 16)  # 1/4 resolution deep supervision


def test_forward_deterministic_in_eval(single_thread) -> None:
    model = small_model().eval()
    img = torch.rand(1, 3, 32, 32)
    ref = torch.rand(1, 3)
    with torch.no_grad():
        a = model(img, ref)
        b = model(img, ref)
    assert torch.equal(a, b)


def test_weight_tying_shares_modules() -> None:
    model = small_model(tied_after=1)
    assert len(model.blocks) == 1
    assert model._get(0)[0] is model._get(1)[0]
    assert model._get(0)[1] is model._get(1)[1]


def test_no_tying_when_disabled() -> None:
    model = small_model(tied_after=0, depth=2)
    assert len(model.blocks) == 2
    assert model._get(0)[0] is not model._get(1)[0]


def test_blur_weights_are_frozen() -> None:
    model = small_model()
    assert not model.stem.down8.blur.weight.requires_grad
    assert not model.stem.down16.blur.weight.requires_grad


def test_sobel_kernels_are_buffers() -> None:
    model = small_model()
    assert "decoder.edge.sx" in model.state_dict()
    assert "decoder.edge.sy" in model.state_dict()


def test_config_round_trip_rebuilds_model() -> None:
    model = small_model()
    rebuilt = CodelorColor(**model.config())
    assert rebuilt.config() == model.config()
    rebuilt.load_state_dict(model.state_dict())


def test_forward_rejects_bad_inputs() -> None:
    model = small_model().eval()
    with pytest.raises(ValueError):
        model(torch.rand(3, 32, 32), torch.rand(1, 3))
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, 32, 32), torch.rand(1, 4))
    with pytest.raises(ValueError):
        model(torch.rand(2, 3, 32, 32), torch.rand(1, 3))


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        CodelorColor(dim=30, heads=4)
    with pytest.raises(ValueError):
        CodelorColor(depth=0)


def test_max_tokens_overflow_warns_once() -> None:
    model = small_model(max_tokens=4).eval()
    img = torch.rand(1, 3, 160, 160)
    ref = torch.rand(1, 3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            model(img, ref)
            model(img, ref)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime) == 1
    assert "max_tokens" in str(runtime[0].message)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def test_predict_mask_single_and_batch() -> None:
    model = small_model().eval()
    mask, prob = predict_mask(model, torch.rand(3, 32, 32))
    assert mask.shape == (32, 32) and prob.shape == (32, 32)
    assert mask.dtype == np.bool_ and prob.dtype == np.float32
    mask_b, prob_b = predict_mask(model, torch.rand(2, 3, 32, 32))
    assert mask_b.shape == (2, 32, 32) and prob_b.shape == (2, 32, 32)


def test_predict_mask_default_color() -> None:
    model = small_model().eval()
    _, prob_default = predict_mask(model, torch.rand(3, 16, 16))
    _, prob_explicit = predict_mask(model, torch.rand(3, 16, 16), color_rgb=DEFAULT_REF_RGB)
    assert prob_default.shape == prob_explicit.shape


def test_predict_mask_validates_inputs() -> None:
    model = small_model().eval()
    with pytest.raises(ValueError):
        predict_mask(model, torch.rand(16, 16))
    with pytest.raises(TypeError):
        predict_mask(model, torch.randint(0, 255, (3, 16, 16), dtype=torch.uint8))
    with pytest.raises(ValueError):
        predict_mask(model, torch.rand(1, 1, 16, 16))
    with pytest.raises(ValueError):
        predict_mask(model, torch.rand(3, 16, 16), threshold=2.0)
    with pytest.raises(ValueError):
        predict_mask(model, torch.rand(3, 16, 16), color_rgb=(0.1, 0.2))


def test_predict_mask_sets_eval_mode() -> None:
    model = small_model()  # starts in train mode
    predict_mask(model, torch.rand(3, 16, 16))
    assert not model.training


def test_remove_green_single_and_batch(single_thread) -> None:
    model = small_model().eval()
    img = torch.rand(3, 32, 32)
    rgba = remove_green(img, model)
    assert rgba.shape == (4, 32, 32)
    _, prob = predict_mask(model, img)
    assert torch.allclose(rgba[3], torch.from_numpy(1.0 - prob), atol=1e-6)
    assert torch.allclose(rgba[:3], img)

    batch = torch.rand(2, 3, 32, 32)
    rgba_b = remove_green(batch, model)
    assert rgba_b.shape == (2, 4, 32, 32)


def test_remove_green_rejects_uint8() -> None:
    model = small_model().eval()
    with pytest.raises(TypeError):
        remove_green(torch.randint(0, 255, (3, 16, 16), dtype=torch.uint8), model)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _write_checkpoint(tmp_path: Path, model: CodelorColor, nested: bool) -> None:
    cfg = {"hyperparameters": model.config()} if nested else model.config()
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))


@pytest.mark.parametrize("nested", [True, False])
def test_load_from_pretrained_local(tmp_path: Path, nested: bool) -> None:
    model = small_model()
    _write_checkpoint(tmp_path, model, nested=nested)
    loaded = load_from_pretrained(tmp_path)
    assert isinstance(loaded, CodelorColor)
    assert not loaded.training
    assert loaded.config() == model.config()
    assert torch.equal(
        loaded.state_dict()["stem.patch.0.weight"],
        model.state_dict()["stem.patch.0.weight"],
    )


def test_load_from_pretrained_ignores_unknown_keys(tmp_path: Path) -> None:
    model = small_model()
    (tmp_path / "config.json").write_text(
        json.dumps({"hyperparameters": {**model.config(), "training_note": "ignored"}})
    )
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    assert load_from_pretrained(tmp_path).config() == model.config()


def test_load_from_pretrained_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_from_pretrained(tmp_path)


def test_saved_weights_are_loadable_by_plain_state_dict(tmp_path: Path) -> None:
    model = small_model()
    path = tmp_path / "model.safetensors"
    save_file(model.state_dict(), str(path))
    reloaded = small_model()
    from safetensors.torch import load_file

    reloaded.load_state_dict(load_file(str(path)))
    assert torch.equal(reloaded.pos, model.pos)
