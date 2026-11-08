"""Tests for the background-removal model, the training loop, Termux helpers and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from codelor_color import (
    TASKS,
    CodelorBackground,
    CodelorColor,
    build_model,
    load_from_pretrained,
    predict_mask,
    remove_background,
    save_pretrained,
)
from codelor_color import device as dev
from codelor_color.cli import _parse_color, main
from codelor_color.training import (
    SyntheticDataset,
    TrainConfig,
    Trainer,
    iou,
    segmentation_loss,
)

SMALL = dict(dim=32, depth=2, heads=4, tied_after=1, max_tokens=64)


# ---------------------------------------------------------------------------
# Background model
# ---------------------------------------------------------------------------

def test_tasks_and_factory() -> None:
    assert TASKS == ("color", "background")
    assert type(build_model("color", **SMALL)) is CodelorColor
    assert type(build_model("background", **SMALL)) is CodelorBackground
    with pytest.raises(ValueError):
        build_model("sky", **SMALL)


def test_background_forward_without_color() -> None:
    model = CodelorBackground(**SMALL).eval()
    img = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = model(img)
        main_, aux = model(img, torch.rand(2, 3), return_aux=True)  # ref ignored
    assert out.shape == (2, 1, 64, 64)
    assert main_.shape == (2, 1, 64, 64) and aux.shape == (2, 1, 16, 16)
    assert "color_mlp.0.weight" not in model.state_dict()
    assert "bg_tokens" in model.state_dict()


def test_color_model_still_requires_color() -> None:
    model = CodelorColor(**SMALL).eval()
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, 32, 32))


def test_background_from_color_model_transfers_backbone() -> None:
    color = CodelorColor(**SMALL)
    bg = CodelorBackground.from_color_model(color)
    assert torch.equal(bg.stem.patch[0].weight, color.stem.patch[0].weight)
    assert torch.equal(bg.blocks[0].attn.qkv.weight, color.blocks[0].attn.qkv.weight)


def test_predict_and_remove_background() -> None:
    model = CodelorBackground(**SMALL).eval()
    mask, prob = predict_mask(model, torch.rand(3, 32, 32))
    assert mask.shape == prob.shape == (32, 32)
    rgba = remove_background(torch.rand(2, 3, 32, 32), model)
    assert rgba.shape == (2, 4, 32, 32)


@pytest.mark.parametrize("cls", [CodelorColor, CodelorBackground])
def test_save_and_load_pretrained_keeps_task(tmp_path: Path, cls: type[CodelorColor]) -> None:
    model = cls(**SMALL)
    save_pretrained(model, tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["task"] == model.task
    loaded = load_from_pretrained(tmp_path)
    assert type(loaded) is cls
    for k, v in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v)


def test_load_pretrained_torch_fallback(tmp_path: Path) -> None:
    model = CodelorBackground(**SMALL)
    (tmp_path / "config.json").write_text(json.dumps({"task": "background", **model.config()}))
    torch.save(model.state_dict(), tmp_path / "model.pt")
    assert isinstance(load_from_pretrained(tmp_path), CodelorBackground)


def test_grad_checkpointing_matches_plain_backward() -> None:
    torch.manual_seed(0)
    model = CodelorColor(**SMALL, drop_path_rate=0.0).train()
    img, ref = torch.rand(2, 3, 32, 32), torch.rand(2, 3)
    model(img, ref).sum().backward()
    g_plain = model.blocks[0].attn.qkv.weight.grad.clone()
    model.zero_grad()
    model.set_grad_checkpointing(True)
    model(img, ref).sum().backward()
    assert torch.allclose(model.blocks[0].attn.qkv.weight.grad, g_plain, atol=1e-5)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", TASKS)
def test_synthetic_dataset(task: str) -> None:
    ds = SyntheticDataset(task, length=3, img_size=32)
    img, target, ref = ds[0]
    assert img.shape == (3, 32, 32) and target.shape == (1, 32, 32) and ref.shape == (3,)
    assert 0 <= float(target.min()) and float(target.max()) <= 1
    assert torch.equal(ds[1][0], ds[1][0])  # deterministic


def test_loss_and_iou() -> None:
    target = (torch.rand(2, 1, 32, 32) > 0.5).float()
    logits = (target * 2 - 1) * 10
    loss = segmentation_loss(logits, torch.zeros(2, 1, 8, 8), target)
    assert torch.isfinite(loss)
    assert iou(logits, target) == pytest.approx(1.0)


def _cfg(tmp_path: Path, task: str, **kw) -> TrainConfig:
    base = dict(task=task, out_dir=str(tmp_path / "run"), epochs=1, max_steps=2,
                batch_size=2, accum_steps=2, img_size=32, num_threads=1, num_workers=0,
                device="cpu", warmup_steps=1, log_every=1, save_every=1, val_split=0.25,
                synthetic_length=12, model=dict(SMALL))
    base.update(kw)
    return TrainConfig(**base)


@pytest.mark.parametrize("task", TASKS)
def test_trainer_runs_exports_and_resumes(tmp_path: Path, task: str) -> None:
    logs: list[str] = []
    trainer = Trainer(_cfg(tmp_path, task), log=logs.append)
    model = trainer.fit()
    assert model.task == task
    assert trainer.step == 2
    run = tmp_path / "run"
    assert (run / "last.pt").is_file()
    assert type(load_from_pretrained(run / "final")).task == task

    resumed = Trainer(_cfg(tmp_path, task, max_steps=3), log=logs.append)
    assert resumed.step == 2
    resumed.fit()
    assert resumed.step == 3
    assert any("Resumed" in line for line in logs)


def test_trainer_background_from_color_checkpoint(tmp_path: Path) -> None:
    save_pretrained(CodelorColor(**SMALL), tmp_path / "color")
    cfg = _cfg(tmp_path, "background", init_from=str(tmp_path / "color"), max_steps=1)
    trainer = Trainer(cfg, log=lambda _m: None)
    assert isinstance(trainer.model, CodelorBackground)


def test_train_config_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"epochz": 3})


def test_folder_dataset(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from codelor_color.training import MaskFolderDataset, save_image

    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    img = torch.zeros(3, 20, 20)
    img[1] = 1.0  # green
    keep = torch.zeros(3, 20, 20)
    keep[:, 5:15, 5:15] = 1.0
    save_image(img, tmp_path / "images" / "a.png")
    save_image(keep, tmp_path / "masks" / "a.png")
    rgba = torch.rand(4, 20, 20)
    save_image(rgba, tmp_path / "images" / "b.png")  # no mask: alpha used
    ds = MaskFolderDataset(tmp_path, img_size=16, task="color", augment=False)
    assert len(ds) == 2
    x, target, ref = ds[0]
    assert x.shape == (3, 16, 16) and target.shape == (1, 16, 16)
    assert ref[1] > 0.9 and ref[0] < 0.1  # estimated key color = green


# ---------------------------------------------------------------------------
# Termux / device helpers
# ---------------------------------------------------------------------------

def test_termux_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERMUX_VERSION", "0.118.0")
    assert dev.is_termux() and dev.is_android()
    prof = dev.recommended_profile()
    assert prof.platform == "termux" and prof.device == "cpu"
    assert prof.num_workers == 0 and not prof.amp
    assert 1 <= prof.num_threads <= 4
    assert dev.default_num_workers() == 0


def test_resolve_device() -> None:
    assert dev.resolve_device("cpu").type == "cpu"
    assert dev.resolve_device("auto").type in ("cpu", "cuda", "mps")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_parse_color() -> None:
    assert _parse_color("#ff0000") == (1.0, 0.0, 0.0)
    assert _parse_color("0.2,0.85,0.3") == (0.2, 0.85, 0.3)
    assert _parse_color("255,0,0") == (1.0, 0.0, 0.0)


def test_cli_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    assert "termux" in json.loads(capsys.readouterr().out)


def test_cli_train_and_predict(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from codelor_color.training import save_image

    out = tmp_path / "run"
    assert main(["train", "--task", "background", "--out", str(out), "--max-steps", "1",
                 "--batch-size", "2", "--accum-steps", "1", "--img-size", "32",
                 "--threads", "1", "--workers", "0", "--device", "cpu",
                 "--dim", "32", "--depth", "2", "--heads", "4", "--tied-after", "1"]) == 0
    src = tmp_path / "in.png"
    save_image(torch.rand(3, 40, 30), src)
    assert main(["predict", "--model", str(out / "final"), str(src),
                 "-o", str(tmp_path / "out"), "--size", "32", "--device", "cpu"]) == 0
    from PIL import Image

    with Image.open(tmp_path / "out" / "in.png") as im:
        assert im.mode == "RGBA" and im.size == (30, 40)
