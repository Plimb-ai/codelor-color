#!/usr/bin/env python3
"""Train ``codelor-color-bg-1.2``: a background-removal model on real photos.

The images come from torchvision datasets that ship a foreground mask:

* ``pets``: Oxford-IIIT Pet (~7,300 photos of cats and dogs, pixel trimaps).
* ``voc``:  Pascal VOC 2012 segmentation (~2,900 photos, 20 kinds of objects:
  people, animals, vehicles, furniture...). Its download mirror is sometimes
  down; the script then goes on with the other datasets.

The model starts from ``plimb/codelor-color-1.2`` (stem and Transformer
weights are reused, the color-specific parts are retrained) with the most
recent architecture preset of the installed codelor-color.

Update codelor-color first (or run this script with ``--update``)::

    # PC / Colab / Linux
    pip install -U "codelor-color[hf]" torchvision
    # Termux (torch and torchvision come from pkg, never from pip)
    pkg install tur-repo && pkg install python-torch python-torchvision
    pip install --no-deps -U codelor-color huggingface_hub

Then::

    python train_codelor_color_bg.py                      # everything, default settings
    python train_codelor_color_bg.py --epochs 3           # shorter
    python train_codelor_color_bg.py --datasets pets      # one dataset only
    python train_codelor_color_bg.py --push-to plimb/codelor-color-bg-1.2
    python train_codelor_color_bg.py --export-to /content/drive/MyDrive/export  # Colab + Drive

Interrupting is safe (Ctrl+C, or Termux killed): run the same command again
and training resumes from the last checkpoint. The finished model is written
to ``<out>/codelor-color-bg-1.2`` and can be used with::

    codelor-color predict --model runs/codelor-color-bg-1.2/codelor-color-bg-1.2 photo.jpg -o out/
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_NAME = "codelor-color-bg-1.2"
BASE_MODEL = "plimb/codelor-color-1.2"
DATASETS = ("pets", "voc")
#: Photos are stored with their longest side at most this (smaller, faster to load).
STORE_SIDE = 384
#: Held-out photos (never trained on) used for the final score and the samples.
N_TEST = 200
#: Free disk space needed to download + extract each dataset (bytes).
PETS_SPACE = int(2.0e9)
VOC_SPACE = int(4.5e9)


def log(msg: str) -> None:
    print(f"[{MODEL_NAME}] {msg}", flush=True)


def is_termux() -> bool:
    return "TERMUX_VERSION" in os.environ or Path("/data/data/com.termux").is_dir()


# ---------------------------------------------------------------------------
# 0. Update codelor-color
# ---------------------------------------------------------------------------

def update_packages() -> None:
    pip = [sys.executable, "-m", "pip", "install", "-U"]
    if is_termux():
        # torch / torchvision are Termux packages: pip must not try to build them.
        cmds = [pip + ["--no-deps", "codelor-color", "huggingface_hub"]]
        log("Termux: torchvision comes from pkg -> pkg install python-torchvision")
    else:
        cmds = [pip + ["codelor-color[hf]", "torchvision"]]
    for cmd in cmds:
        log("$ " + " ".join(cmd))
        subprocess.check_call(cmd)
    log("Updated. Run the script again without --update to train.")


def check_versions() -> None:
    try:
        import codelor_color
        from codelor_color.model import PRESETS
    except ImportError:
        sys.exit("codelor-color is not installed: run this script with --update")
    if "v4" not in PRESETS:
        sys.exit(f"codelor-color {codelor_color.__version__} is too old (need >= 2.4.0): "
                 "run this script with --update")
    try:
        import torchvision  # noqa: F401
    except ImportError:
        sys.exit("torchvision is missing: pip install torchvision "
                 "(Termux: pkg install python-torchvision)")
    log(f"codelor-color {codelor_color.__version__}")


# ---------------------------------------------------------------------------
# 1. Real photos + masks from torchvision
# ---------------------------------------------------------------------------

def _resize(img, side: int):  # PIL image -> PIL image
    from PIL import Image

    w, h = img.size
    scale = side / max(w, h)
    if scale >= 1:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(size, Image.BILINEAR if img.mode == "L" else Image.LANCZOS)


def _to_mask(arr):
    from PIL import Image

    return Image.fromarray(arr.astype("uint8"), "L")


def pets_mask(trimap):
    """Oxford pet trimap (1 animal, 2 background, 3 border) -> keep mask (255/0/128)."""
    import numpy as np

    t = np.asarray(trimap)
    return _to_mask(np.where(t == 1, 255, np.where(t == 3, 128, 0)))


def voc_mask(label):
    """VOC labels (0 background, 1-20 objects, 255 outline) -> keep mask (0/255/128)."""
    import numpy as np

    t = np.asarray(label)
    return _to_mask(np.where(t == 0, 0, np.where(t == 255, 128, 255)))


def export_pairs(dataset, to_mask, dst: Path, prefix: str, test: Path | None = None,
                 n_test: int = 0) -> int:
    """Write ``images/<name>.jpg`` + ``masks/<name>.png``; the last ``n_test`` go to ``test``."""
    n = len(dataset)
    for i in range(n):
        root = test if (test is not None and i >= n - n_test) else dst
        name = f"{prefix}_{i:05d}"
        img_path = root / "images" / f"{name}.jpg"
        mask_path = root / "masks" / f"{name}.png"
        if img_path.is_file() and mask_path.is_file():
            continue
        img, target = dataset[i]
        mask = to_mask(target)
        if mask.size != img.size:
            continue
        _resize(img.convert("RGB"), STORE_SIDE).save(img_path, quality=92)
        _resize(mask, STORE_SIDE).save(mask_path)
        if (i + 1) % 500 == 0:
            log(f"  {prefix}: {i + 1}/{n}")
    return n


def _marker(data: Path, name: str) -> Path:
    return data / f".{name}_done"


def _exported(data: Path, name: str, raw_paths: list[Path]) -> bool:
    """True if ``name`` was exported (marker, or exported files and no raw download left)."""
    if _marker(data, name).is_file():
        return True
    has_files = any((data / "images").glob(f"{name}_*.jpg"))
    return has_files and not any(p.exists() for p in raw_paths)


def _remove(paths: list[Path]) -> None:
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()


def _done(data: Path, name: str, raw_paths: list[Path], keep_raw: bool) -> None:
    _marker(data, name).write_text("ok\n")
    if not keep_raw:
        _remove(raw_paths)  # photos are exported: the archives only take space
        log(f"  {name}: raw download deleted (use --keep-raw to keep it)")


def _enough_space(path: Path, need: int, label: str) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < need:
        log(f"{label} skipped: {free / 1e9:.1f} GB free, about {need / 1e9:.1f} GB needed")
        return False
    return True


def prepare_data(names: list[str], raw: Path, data: Path, test: Path,
                 keep_raw: bool = False) -> int:
    from torchvision import datasets as tvd

    for root in (data, test):
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "masks").mkdir(parents=True, exist_ok=True)
    total = 0
    if "pets" in names:
        pets_raw = [raw / "oxford-iiit-pet"]
        if _exported(data, "pets", pets_raw):
            log("Oxford-IIIT Pet: already prepared")
        elif _enough_space(raw, PETS_SPACE, "Oxford-IIIT Pet"):
            log("Oxford-IIIT Pet: download (~800 MB the first time) and export...")
            for split in ("trainval", "test"):
                ds = tvd.OxfordIIITPet(str(raw), split=split, target_types="segmentation",
                                       download=True)
                total += export_pairs(ds, pets_mask, data, f"pets_{split}",
                                      test if split == "test" else None, N_TEST)
            _done(data, "pets", pets_raw, keep_raw)
    if "voc" in names:
        voc_raw = [raw / "VOCdevkit", raw / "VOCtrainval_11-May-2012.tar"]
        if _exported(data, "voc", voc_raw):
            log("Pascal VOC 2012: already prepared")
        elif _enough_space(raw, VOC_SPACE, "Pascal VOC 2012"):
            log("Pascal VOC 2012: download (~2 GB the first time) and export...")
            try:
                ds = tvd.VOCSegmentation(str(raw), year="2012", image_set="trainval",
                                         download=True)
                total += export_pairs(ds, voc_mask, data, "voc")
                _done(data, "voc", voc_raw, keep_raw)
            except Exception as e:  # mirror down, disk full... keep the other datasets
                log(f"  VOC skipped ({type(e).__name__}: {e})")
                _remove(voc_raw)  # a half-extracted archive only wastes space
    n = len(list((data / "images").glob("*.jpg")))
    if n == 0:
        sys.exit("no training photos could be prepared")
    log(f"{n} training photos, {len(list((test / 'images').glob('*.jpg')))} held-out")
    return n


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------

def _available_ram() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def make_safe_trainer(base):
    """``Trainer`` that skips optimizer steps whose gradients are not finite.

    One bad batch (or an fp16 overflow on GPU) can otherwise turn every weight
    into NaN; the loss then prints ``nan`` forever and the run is lost.
    """
    import math

    import torch

    class SafeTrainer(base):
        skipped = 0

        def _optimizer_step(self) -> None:
            self.scaler.unscale_(self.opt)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            if not math.isfinite(float(norm)):
                self.skipped += 1
                if self.scaler.is_enabled():
                    self.scaler.step(self.opt)  # skipped by the scaler, which lowers its scale
                    self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
                log(f"step {self.step}: non-finite gradients, update skipped "
                    f"({self.skipped} so far)")
                bad = [n for n, b in self.model.named_buffers()
                       if b.is_floating_point() and not torch.isfinite(b).all()]
                if bad and (self.out / "last.pt").is_file():
                    log(f"  {len(bad)} statistics became NaN: reloading last checkpoint")
                    self.load_checkpoint(self.out / "last.pt")
                if self.skipped >= 50:
                    raise RuntimeError("too many non-finite steps: lower --lr")
                return
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.sched.step()
            if self.ema is not None:
                self.ema.update(self.model)
            self.step += 1
            if self.step % self.cfg.save_every == 0:
                self.save_checkpoint()

    return SafeTrainer


def train(args: argparse.Namespace, data: Path, n_photos: int):
    from codelor_color.device import recommended_profile
    from codelor_color.model import PRESETS
    from codelor_color.training import TrainConfig, Trainer

    prof = recommended_profile(args.device)
    img_size = args.img_size or prof.img_size
    # Keep decoded photos in RAM only if they fit comfortably.
    need = n_photos * 3 * int(img_size * 1.15) ** 2
    ram = _available_ram()
    cache = need < 1e9 if ram is None else need < 0.4 * ram
    preset = "v5" if "v5" in PRESETS else "v4"
    # Before 2.5.0, freezing BatchNorm also froze layers that are new after
    # the preset conversion: let them all train instead.
    freeze_bn = None if preset == "v5" else False
    device = prof.device if args.device == "auto" else args.device
    log(f"preset {preset}, {img_size}px, device {device}"
        f", RAM cache {'on' if cache else 'off'} (~{need / 1e9:.1f} GB)")
    cfg = TrainConfig(
        task="background",
        data_dir=str(data),
        out_dir=str(args.out),
        init_from=None if args.from_scratch else BASE_MODEL,
        preset=preset,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        img_size=img_size,
        batch_size=args.batch_size,
        device=args.device,
        val_split=0.05,
        cache=cache,
        freeze_bn=freeze_bn,
        save_every=100,
        log_every=10,
    )
    trainer = make_safe_trainer(Trainer)(cfg)
    trainer.fit()
    trainer.preset = preset
    return trainer


# ---------------------------------------------------------------------------
# 3. Export, held-out score, samples, optional upload
# ---------------------------------------------------------------------------

def finalize(args: argparse.Namespace, trainer, test: Path) -> Path:
    import torch

    from codelor_color import load_from_pretrained, predict_mask, save_pretrained
    from codelor_color.training import load_image, save_image

    model = load_from_pretrained(args.out / "final")
    dst = args.out / MODEL_NAME
    if dst.exists():
        shutil.rmtree(dst)
    score = None
    samples = args.out / "samples"
    samples.mkdir(exist_ok=True)
    images = sorted((test / "images").glob("*.jpg"))
    if images:
        log(f"Scoring on {len(images)} held-out photos...")
        inter = union = 0.0
        size = trainer.cfg.img_size
        for k, path in enumerate(images):
            img = load_image(path)
            keep = load_image(test / "masks" / f"{path.stem}.png", mode="L")[0] > 0.5
            _, prob = predict_mask(model, img, max_side=max(size, 256))
            removed = torch.from_numpy(prob) > 0.5
            inter += float((removed & ~keep).sum())
            union += float((removed | ~keep).sum())
            if k < 8:
                alpha = (1 - torch.from_numpy(prob))[None]
                save_image(torch.cat([img, alpha], 0), samples / f"{path.stem}.png")
        score = inter / max(union, 1.0)
        log(f"held-out IoU (background) = {score:.3f}; cut-outs in {samples}")
    save_pretrained(model, dst, name=MODEL_NAME, base_model=BASE_MODEL,
                    datasets=args.datasets, training=dict(
                        step=trainer.step, img_size=trainer.cfg.img_size,
                        heldout_iou=score))
    (dst / "README.md").write_text(model_card(args, trainer, score))
    single = export_safetensors(model, dst)
    log(f"Model ready: {dst}")
    log(f"Single-file weights: {single}")
    if args.export_to:
        export_folder(dst, single, Path(args.export_to))
    if args.push_to:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to, exist_ok=True)
        api.upload_folder(folder_path=str(dst), repo_id=args.push_to)
        log(f"Uploaded to https://huggingface.co/{args.push_to}")
    return dst


def export_safetensors(model, dst: Path) -> Path:
    """Write ``<dst>/../codelor-color-bg-1.2.safetensors``: weights + config in one file.

    The config (task and hyper-parameters) is stored in the safetensors
    metadata, so the file alone is enough to rebuild the model; the folder
    ``dst`` (config.json + model.safetensors) is what ``load_from_pretrained``
    and Hugging Face use.
    """
    import json

    try:
        from safetensors.torch import save_file
    except ImportError:
        sys.exit("safetensors is missing: pip install safetensors "
                 "(Termux: pkg install python-safetensors)")
    if not (dst / "model.safetensors").is_file():  # save_pretrained fell back to model.pt
        state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        save_file(state, str(dst / "model.safetensors"))
        (dst / "model.pt").unlink(missing_ok=True)
    out = dst.parent / f"{MODEL_NAME}.safetensors"
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    meta = {"name": MODEL_NAME, "format": "pt",
            "codelor_task": model.task, "hyperparameters": json.dumps(model.config())}
    save_file(state, str(out), metadata=meta)
    return out


def load_single_file(path: str | Path):
    """Rebuild a model from the single ``.safetensors`` file written above."""
    import json

    from safetensors import safe_open
    from safetensors.torch import load_file

    from codelor_color import build_model

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    model = build_model(meta["codelor_task"], **json.loads(meta["hyperparameters"]))
    model.load_state_dict(load_file(str(path)))
    return model.eval()


def export_folder(model_dir: Path, single: Path, target: Path) -> None:
    """Copy the model folder and the single file to ``target`` (e.g. Google Drive)."""
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_dir, target / model_dir.name, dirs_exist_ok=True)
    shutil.copy2(single, target / single.name)
    log(f"Exported to {target}")


def model_card(args: argparse.Namespace, trainer, score: float | None) -> str:
    sets = {"pets": "Oxford-IIIT Pet", "voc": "Pascal VOC 2012 segmentation"}
    used = ", ".join(sets[d] for d in args.datasets)
    score_line = f"- Held-out IoU of the removed background: **{score:.3f}**\n" if score else ""
    return f"""---
library_name: codelor-color
pipeline_tag: image-segmentation
base_model: {BASE_MODEL}
tags: [background-removal, codelor-color]
---

# {MODEL_NAME}

Background removal (keeps the main subject, makes the background transparent),
trained with [codelor-color](https://pypi.org/project/codelor-color/) from
`{BASE_MODEL}`.

- Task: `background`, architecture preset `{getattr(trainer, "preset", "?")}`,
  {sum(p.numel() for p in trainer.model.parameters()) / 1e6:.2f}M parameters
- Training photos: {used}
- Steps: {trainer.step}, resolution {trainer.cfg.img_size} px
{score_line}
## Usage

    pip install -U "codelor-color[hf]"
    codelor-color predict --model <this repo or folder> photo.jpg -o out/

```python
from codelor_color import load_from_pretrained, remove_background
from codelor_color.training import load_image, save_image

model = load_from_pretrained("<this repo or folder>")
rgba = remove_background(load_image("photo.jpg"), model, max_side=512)
save_image(rgba, "photo_cutout.png")
```

Check the licenses of the training datasets before any commercial use.
"""


# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Train {MODEL_NAME} (background removal)")
    p.add_argument("--update", action="store_true", help="update codelor-color and exit")
    p.add_argument("--datasets", default=",".join(DATASETS),
                   help="comma-separated: pets, voc (default: both)")
    p.add_argument("--root", type=Path, default=Path("data"),
                   help="where datasets are downloaded and exported")
    p.add_argument("--keep-raw", action="store_true",
                   help="keep the downloaded archives after export (default: delete)")
    p.add_argument("--out", type=Path, default=Path("runs") / MODEL_NAME)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, help="stop after this many steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img-size", type=int, help="default: 160 (256 on a CUDA GPU)")
    p.add_argument("--batch-size", type=int, help="default: picked for the device")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    p.add_argument("--from-scratch", action="store_true",
                   help=f"do not start from {BASE_MODEL}")
    p.add_argument("--export-to", help="copy the finished model (folder + single "
                                       ".safetensors file) to this folder, e.g. a mounted "
                                       "Google Drive")
    p.add_argument("--push-to", help="upload the result to this Hugging Face repo "
                                     "(log in first: huggingface-cli login)")
    args = p.parse_args(argv)
    args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    bad = set(args.datasets) - set(DATASETS)
    if bad or not args.datasets:
        p.error(f"--datasets must be among {DATASETS}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.update:
        update_packages()
        return 0
    check_versions()
    data, test = args.root / "photos", args.root / "heldout"
    n = prepare_data(args.datasets, args.root / "raw", data, test, args.keep_raw)
    trainer = train(args, data, n)
    finalize(args, trainer, test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
