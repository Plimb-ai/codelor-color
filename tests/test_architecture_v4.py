"""v4 preset, guided-filter refinement and the new training options."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from codelor_color import build_model, load_from_pretrained, predict_mask, save_pretrained
from codelor_color.model import PRESETS, guided_filter, guided_upsample
from codelor_color.training import (
    SyntheticDataset,
    TrainConfig,
    Trainer,
    edge_map,
    segmentation_loss,
)

SMALL = dict(dim=32, depth=2, heads=4, tied_after=1, max_tokens=64)


def test_v3_architecture_unchanged() -> None:
    """Models trained with 2.3.0's v3 preset must keep loading."""
    model = build_model("color", preset="v3", **SMALL)
    keys = model.state_dict().keys()
    assert not any(k.startswith(("stem.extra", "color_ctx")) for k in keys)
    assert "stem_depth" not in model.config() and "global_context" not in model.config()


@pytest.mark.parametrize("task", ["color", "background"])
def test_v4_forward_and_round_trip(task: str, tmp_path: Path) -> None:
    model = build_model(task, preset="v4", **SMALL).eval()
    assert model.stem.extra4 is not None and model.stem.extra8 is not None
    with torch.no_grad():
        out = model(torch.rand(2, 3, 48, 80), torch.rand(2, 3))
    assert out.shape == (2, 1, 48, 80)
    save_pretrained(model, tmp_path)
    loaded = load_from_pretrained(tmp_path)
    assert loaded.config() == model.config()
    assert loaded.stem_depth == 2


def test_v4_color_context_starts_neutral() -> None:
    model = build_model("color", preset="v4", **SMALL).eval()
    assert model.color_ctx is not None
    tokens = torch.randn(2, 9, 32)
    ref = torch.rand(2, 3)
    with torch.no_grad():
        assert torch.equal(model._condition(ref, 2, tokens), model.color_mlp(ref))


def test_global_context_requires_film() -> None:
    with pytest.raises(ValueError):
        build_model("color", **SMALL, global_context=True)


def test_presets_listed() -> None:
    assert {"v2", "v3", "v4"} <= set(PRESETS)


def test_guided_filter_keeps_constant_and_follows_edges() -> None:
    const = torch.full((1, 1, 16, 16), 0.3)
    assert torch.allclose(guided_filter(torch.rand(1, 1, 16, 16), const), const, atol=1e-5)
    guide = torch.zeros(1, 1, 32, 32)
    guide[..., 16:] = 1.0                       # sharp vertical edge
    blurry = torch.linspace(0, 1, 32).view(1, 1, 1, 32).expand(1, 1, 32, 32)
    out = guided_filter(guide, blurry, radius=2, eps=1e-4)
    assert float(out[..., 4].mean()) < float(blurry[..., 4].mean()) + 0.05
    jump_before = float((blurry[..., 16] - blurry[..., 15]).abs().mean())
    jump_after = float((out[..., 16] - out[..., 15]).abs().mean())
    assert jump_after > 2 * jump_before  # the border got sharper, at the guide's edge


def test_guided_upsample_shape() -> None:
    out = guided_upsample(torch.rand(1, 1, 64, 96), torch.rand(1, 1, 16, 24))
    assert out.shape == (1, 1, 64, 96)


def test_predict_mask_refine_modes() -> None:
    model = build_model("color", preset="v4", **SMALL).eval()
    img = torch.rand(3, 70, 90)
    for kwargs in (dict(refine="guided"), dict(refine="guided", max_side=32), dict()):
        mask, prob = predict_mask(model, img, **kwargs)
        assert prob.shape == (70, 90) and 0 <= prob.min() and prob.max() <= 1
    with pytest.raises(ValueError):
        predict_mask(model, img, refine="magic")


def test_edge_map_and_weighted_loss() -> None:
    t = torch.zeros(1, 1, 20, 20)
    t[..., 5:15, 5:15] = 1
    e = edge_map(t)
    assert e[0, 0, 10, 10] == 0 and e[0, 0, 5, 10] == 1 and e[0, 0, 0, 0] == 0
    logits = torch.zeros(1, 1, 20, 20)
    aux = torch.zeros(1, 1, 5, 5)
    assert segmentation_loss(logits, aux, t, edge_weight=2.0) > segmentation_loss(logits, aux, t)


def test_synthetic_hard_differs() -> None:
    easy = SyntheticDataset("color", 2, 32, seed=1)[0][0]
    hard = SyntheticDataset("color", 2, 32, seed=1, hard=True)[0][0]
    assert not torch.equal(easy, hard)
    img, target, _ = SyntheticDataset("background", 2, 32, hard=True)[1]
    assert img.shape == (3, 32, 32) and target.shape == (1, 32, 32)


def test_trainer_multiscale_auto(tmp_path: Path) -> None:
    base = dict(task="color", out_dir=str(tmp_path / "r"), max_steps=2, batch_size=2,
                accum_steps=1, img_size=32, num_threads=1, num_workers=0, device="cpu",
                warmup_steps=1, log_every=1, save_every=100, val_split=0.0,
                synthetic_length=4, model=dict(SMALL), resume=False)
    trainer = Trainer(TrainConfig(preset="v4", **base), log=lambda _m: None)
    assert trainer.multiscale
    trainer.fit()
    assert trainer.step == 2
    off = Trainer(TrainConfig(preset="v4", multiscale=False, **base), log=lambda _m: None)
    assert not off.multiscale
