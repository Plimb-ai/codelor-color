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
    assert model.norm == "gn" and model.head == "light" and model.cond_mode == "film"


@pytest.mark.parametrize("task", ["color", "background"])
def test_v4_forward_and_round_trip(task: str, tmp_path: Path) -> None:
    model = build_model(task, preset="v4", **SMALL).eval()
    assert model.norm == "bn" and model.pos is None and model.cond_mode == "film"
    with torch.no_grad():
        out = model(torch.rand(2, 3, 48, 80), torch.rand(2, 3))
    assert out.shape == (2, 1, 48, 80)
    save_pretrained(model, tmp_path)
    loaded = load_from_pretrained(tmp_path)
    assert loaded.config() == model.config()


def test_v4_bn_is_fusable(single_thread) -> None:
    from codelor_color.model import optimize_for_inference

    model = build_model("color", preset="v4", **SMALL)
    model.train()(torch.rand(4, 3, 48, 48), torch.rand(4, 3))
    model.eval()
    fast = optimize_for_inference(model)
    img, ref = torch.rand(1, 3, 48, 48), torch.rand(1, 3)
    with torch.no_grad():
        assert torch.allclose(fast(img, ref), model(img, ref), atol=1e-4)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in fast.modules())


def test_presets_listed() -> None:
    assert {"v2", "v3", "v4", "v5"} <= set(PRESETS)


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


def test_deep_tied_model_cycles_shared_blocks() -> None:
    model = build_model("color", preset="v4", dim=32, depth=7, heads=4, tied_after=2).eval()
    assert len(model.blocks) == 2
    assert [model._get(i)[0] is model.blocks[i % 2] for i in range(7)] == [True] * 7
    with torch.no_grad():
        assert model(torch.rand(1, 3, 32, 32), torch.rand(1, 3)).shape == (1, 1, 32, 32)


@pytest.mark.parametrize("task", ["color", "background"])
@pytest.mark.parametrize("size", [(48, 80), (70, 90), (33, 41)])
def test_v5_quarter_head_any_size(task: str, size: tuple[int, int], tmp_path: Path) -> None:
    model = build_model(task, preset="v5", **SMALL).eval()
    assert model.head == "quarter" and model.decoder.head_stride == 4
    with torch.no_grad():
        out = model(torch.rand(1, 3, *size), torch.rand(1, 3))
    assert out.shape == (1, 1, *size)
    save_pretrained(model, tmp_path)
    assert load_from_pretrained(tmp_path).config() == model.config()


def test_v4_preset_unchanged() -> None:
    """Models trained with 2.4.0's default preset must keep loading."""
    assert PRESETS["v4"]["head"] == "light"


def test_trainer_defaults_to_v5(tmp_path: Path) -> None:
    cfg = TrainConfig(task="color", out_dir=str(tmp_path / "r"), max_steps=1, batch_size=2,
                      img_size=32, num_threads=1, num_workers=0, device="cpu",
                      val_split=0.0, synthetic_length=2, model=dict(SMALL), resume=False)
    assert Trainer(cfg, log=lambda _m: None).model.head == "quarter"
