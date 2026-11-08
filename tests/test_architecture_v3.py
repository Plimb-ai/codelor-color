"""Compatibility of the original architecture + the v3 preset and inference optimizations."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from codelor_color import (
    CodelorBackground,
    CodelorColor,
    build_model,
    load_from_pretrained,
    predict_mask,
)
from codelor_color.model import (
    HYPERPARAM_KEYS,
    ColorCrossAttention,
    color_cosine,
    despill,
    optimize_for_inference,
    rope_2d,
)
from codelor_color.training import TrainConfig, Trainer

SMALL = dict(dim=32, depth=2, heads=4, tied_after=1, max_tokens=64)


# ---------------------------------------------------------------------------
# Original architecture stays bit-compatible
# ---------------------------------------------------------------------------

def test_default_model_is_original_architecture() -> None:
    model = CodelorColor()
    assert set(model.config()) == set(HYPERPARAM_KEYS)  # no new keys in v2 configs
    keys = model.state_dict().keys()
    assert "pos" in keys and "color_mlp.0.weight" in keys
    assert "color_xattn.0.q.weight" in keys
    assert not any(k.startswith(("color_film", "decoder.refine")) for k in keys)
    # same tensor count as plimb/codelor-color-1.2 (params + BN statistics + Sobel buffers)
    assert sum(t.numel() for t in model.state_dict().values()) == 1_641_055


def test_load_hub_style_config(tmp_path: Path) -> None:
    """``plimb/codelor-color-1.2`` uses ``"task": "image-segmentation"`` and extra metadata."""
    model = CodelorColor(**SMALL)
    cfg = {
        "architectures": ["CodelorColor"],
        "model_type": "codelor-color",
        "task": "image-segmentation",
        "labels": ["background", "green"],
        "green_ref": [0.2, 0.85, 0.3],
        "hyperparameters": {k: v for k, v in model.config().items() if k != "drop_path_rate"},
        "training": {"epochs_done": 30},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    loaded = load_from_pretrained(tmp_path)
    assert type(loaded) is CodelorColor


def test_background_detected_from_architectures(tmp_path: Path) -> None:
    model = CodelorBackground(**SMALL)
    (tmp_path / "config.json").write_text(json.dumps(
        {"architectures": ["CodelorBackground"], "task": "image-segmentation",
         "hyperparameters": model.config()}))
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    assert type(load_from_pretrained(tmp_path)) is CodelorBackground


def _reference_rope(q, k, h, w, base=100.0):
    B, Hh, N, D = q.shape
    d4 = D // 4
    freqs = 1.0 / (base ** (torch.arange(0, d4).float() / d4))
    ax = torch.outer(torch.arange(w).float(), freqs)[None].expand(h, w, d4).reshape(h * w, d4)
    ay = torch.outer(torch.arange(h).float(), freqs)[:, None].expand(h, w, d4).reshape(h * w, d4)
    ang = torch.cat([ax, ay], -1)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]

    def rot(x):
        x1, x2 = x[..., : D // 2], x[..., D // 2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)

    return rot(q), rot(k)


def test_cached_rope_matches_reference() -> None:
    q, k = torch.randn(2, 4, 12, 16), torch.randn(2, 4, 12, 16)
    for got, ref in zip(rope_2d(q, k, 3, 4), _reference_rope(q, k, 3, 4), strict=True):
        assert torch.equal(got, ref)


def test_rope_cache_built_in_inference_mode_is_trainable() -> None:
    model = CodelorColor(**SMALL)
    with torch.inference_mode():
        model.eval()(torch.rand(1, 3, 48, 48), torch.rand(1, 3))
    model.train()(torch.rand(2, 3, 48, 48), torch.rand(2, 3)).sum().backward()


def test_single_token_cross_attention_fast_path() -> None:
    xa = ColorCrossAttention(32, 4)
    x, c = torch.randn(2, 10, 32), torch.randn(2, 1, 32)
    B, N, C = x.shape
    q = xa.q(x).reshape(B, N, 4, 8).transpose(1, 2)
    kv = xa.kv(c).reshape(B, 1, 2, 4, 8).permute(2, 0, 3, 1, 4)
    kk, v = kv.unbind(0)
    q, kk = xa.q_norm(q), xa.k_norm(kk)
    full = xa.proj(F.scaled_dot_product_attention(q, kk, v).transpose(1, 2).reshape(B, N, C))
    assert torch.allclose(xa(x, c), full, atol=1e-6)


def test_color_cosine_matches_torch() -> None:
    img, ref = torch.rand(2, 3, 9, 7), torch.rand(2, 3)
    img[0, :, 0, 0] = 0.0  # black pixel: zero norm
    expected = F.cosine_similarity(img, ref.view(2, 3, 1, 1).expand_as(img), dim=1, eps=1e-6)
    assert torch.allclose(color_cosine(img, ref)[:, 0], expected, atol=1e-6)


def test_pos_grid_identical_at_training_resolution() -> None:
    model = CodelorColor(**SMALL).eval()
    img, ref = torch.rand(1, 3, 64, 64), torch.rand(1, 3)
    with torch.no_grad():
        before = model(img, ref)
        model.set_pos_grid((4, 4))
        after = model(img, ref)
        assert torch.equal(before, after)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = model(torch.rand(1, 3, 160, 224), ref)  # 10x14 tokens > max_tokens=64
    assert out.shape == (1, 1, 160, 224)
    assert model.config()["pos_grid"] == [4, 4]
    assert CodelorColor(**model.config()).pos_mode == "interp"


# ---------------------------------------------------------------------------
# v3 preset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["color", "background"])
def test_v3_forward_any_resolution(task: str) -> None:
    model = build_model(task, preset="v3", **SMALL).eval()
    assert model.pos is None and model.cond_mode == "film"
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with torch.no_grad():
            for h, w in ((64, 64), (96, 160), (50, 70)):
                main, aux = model(torch.rand(2, 3, h, w), torch.rand(2, 3), return_aux=True)
                assert main.shape == (2, 1, h, w)


def test_v3_config_round_trip(tmp_path: Path) -> None:
    from codelor_color import save_pretrained

    model = build_model("color", preset="v3", **SMALL)
    save_pretrained(model, tmp_path)
    loaded = load_from_pretrained(tmp_path)
    assert loaded.config() == model.config()
    assert loaded.head == "light" and loaded.norm == "gn" and loaded.chroma_guide


def test_v3_film_starts_as_identity() -> None:
    model = build_model("color", preset="v3", **SMALL)
    film = model.color_film[0]
    x = torch.randn(2, 5, 32)
    assert torch.equal(film(x, torch.randn(2, 32)), x)


def test_v3_color_conditioning_changes_output() -> None:
    model = build_model("color", preset="v3", **SMALL)
    with torch.no_grad():
        for p in model.color_film.parameters():
            p.normal_(0, 0.5)
    model.eval()
    img = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        a = model(img, torch.tensor([[0.2, 0.85, 0.3]]))
        b = model(img, torch.tensor([[0.1, 0.2, 0.9]]))
    assert not torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Inference optimizations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["v2", "v3"])
def test_optimize_for_inference_matches(preset: str, single_thread) -> None:
    model = build_model("color", preset=preset, **SMALL)
    model.train()(torch.rand(4, 3, 64, 64), torch.rand(4, 3))  # non-trivial BN statistics
    model.eval()
    fast = optimize_for_inference(model)
    img, ref = torch.rand(2, 3, 64, 48), torch.rand(2, 3)
    with torch.no_grad():
        assert torch.allclose(fast(img, ref), model(img, ref), atol=1e-4)
    if preset == "v2":  # original untouched
        assert any(isinstance(m, nn.BatchNorm2d) for m in model.modules())


def test_optimize_int8_runs() -> None:
    if not torch.backends.quantized.supported_engines:
        pytest.skip("no quantized engine")
    model = CodelorColor(**SMALL).eval()
    q = optimize_for_inference(model, quantize=True)
    with torch.no_grad():
        out = q(torch.rand(1, 3, 32, 32), torch.rand(1, 3))
    assert out.shape == (1, 1, 32, 32)


def test_predict_mask_max_side_keeps_shape() -> None:
    model = CodelorColor(**SMALL).eval()
    mask, prob = predict_mask(model, torch.rand(3, 100, 300), max_side=64)
    assert prob.shape == (100, 300) and mask.shape == (100, 300)
    with pytest.raises(ValueError):
        predict_mask(model, torch.rand(3, 32, 32), max_side=4)


def test_despill_reduces_green_cast() -> None:
    rgb = torch.tensor([0.5, 0.9, 0.5]).view(3, 1, 1).expand(3, 4, 4).clone()
    out = despill(rgb, torch.full((4, 4), 0.5))
    assert float(out[1].max()) < 0.9
    assert torch.equal(out[0], rgb[0]) and torch.equal(out[2], rgb[2])


# ---------------------------------------------------------------------------
# Training additions
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path, **kw) -> TrainConfig:
    base = dict(task="color", out_dir=str(tmp_path / "run"), epochs=1, batch_size=2,
                accum_steps=2, img_size=32, num_threads=1, num_workers=0, device="cpu",
                warmup_steps=1, log_every=1, save_every=100, val_split=0.0,
                synthetic_length=6, model=dict(SMALL), resume=False)
    base.update(kw)
    return TrainConfig(**base)


def test_trainer_defaults_to_v3_and_steps_on_leftovers(tmp_path: Path) -> None:
    trainer = Trainer(_cfg(tmp_path), log=lambda _m: None)
    assert trainer.model.cond_mode == "film"
    trainer.fit()
    assert trainer.step == 2  # 3 micro-batches, accum 2 -> 1 full + 1 leftover step


def test_trainer_v2_records_pos_grid(tmp_path: Path) -> None:
    trainer = Trainer(_cfg(tmp_path, preset="v2", max_steps=1), log=lambda _m: None)
    assert trainer.model.pos_grid == (2, 2)  # 32px -> 2x2 tokens


def test_trainer_converts_v2_checkpoint_to_v3(tmp_path: Path) -> None:
    from codelor_color import save_pretrained

    src = CodelorColor(**SMALL)
    save_pretrained(src, tmp_path / "v2")
    trainer = Trainer(_cfg(tmp_path, init_from=str(tmp_path / "v2"), preset="v3", max_steps=1),
                      log=lambda _m: None)
    assert trainer.model.cond_mode == "film"
    assert torch.equal(trainer.model.blocks[0].attn.qkv.weight, src.blocks[0].attn.qkv.weight)


def test_trainer_freezes_bn_when_finetuning_small_batch(tmp_path: Path) -> None:
    from codelor_color import save_pretrained

    src = CodelorColor(**SMALL)
    save_pretrained(src, tmp_path / "v2")
    trainer = Trainer(_cfg(tmp_path, init_from=str(tmp_path / "v2"), max_steps=1),
                      log=lambda _m: None)
    assert trainer.freeze_bn
    mean_before = trainer.model.stem.patch[1].running_mean.clone()
    trainer.fit()
    assert torch.equal(trainer.model.stem.patch[1].running_mean, mean_before)


def test_trainer_exports_ema(tmp_path: Path) -> None:
    trainer = Trainer(_cfg(tmp_path, max_steps=2), log=lambda _m: None)
    trainer.fit()
    exported = load_from_pretrained(tmp_path / "run" / "final")
    raw = trainer.model.state_dict()
    ema = trainer.ema.model.state_dict()
    k = "blocks.0.attn.qkv.weight"
    assert torch.equal(exported.state_dict()[k], ema[k])
    assert not torch.equal(ema[k], raw[k])
