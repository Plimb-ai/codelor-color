"""Training utilities for both Codelor tasks (``"color"`` and ``"background"``).

Designed to run everywhere, from a GPU workstation to a phone with Termux:

* CPU-friendly defaults picked by :func:`codelor_color.device.recommended_profile`
  (few threads, no DataLoader workers on Android, small batch + gradient accumulation);
* crash-safe checkpoints written atomically, automatic resume, and a final save
  on ``Ctrl+C`` / ``SIGTERM`` (Android may kill Termux in the background);
* optional gradient checkpointing to reduce RAM;
* only ``torch`` is required; ``Pillow`` is needed to read image folders
  (``pip install "codelor-color[train]"`` or ``pkg install python-pillow`` on Termux).

Mask convention (both tasks): a mask is the **alpha you want in the result**,
white = keep (subject), black = remove (key color or background). The model is
trained to predict the pixels to *remove*, i.e. ``1 - mask``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import signal
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from .device import recommended_profile, resolve_device
from .model import (
    DEFAULT_REF_RGB,
    TASKS,
    CodelorBackground,
    CodelorColor,
    build_model,
    load_from_pretrained,
    save_pretrained,
    transfer_weights,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

#: A training sample: ``(image (3,H,W), target (1,H,W) = pixels to remove, ref_rgb (3,))``.
Sample = tuple[Tensor, Tensor, Tensor]


# ============================================================
# IMAGE I/O
# ============================================================

def _require_pil() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Reading images requires Pillow: pip install \"codelor-color[train]\" "
            "(on Termux: pkg install python-pillow)"
        ) from exc
    return Image


def load_image(path: str | Path, size: int | None = None, mode: str = "RGB") -> Tensor:
    """Read an image as a float tensor ``(C, H, W)`` in ``[0, 1]``, optionally resized."""
    Image = _require_pil()
    with Image.open(path) as im:
        im = im.convert(mode)
        if size is not None:
            im = im.resize((size, size), Image.BILINEAR)
        data = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        t = data.view(im.size[1], im.size[0], len(mode)).permute(2, 0, 1)
    return t.float() / 255.0


def save_image(tensor: Tensor, path: str | Path) -> None:
    """Write a ``(3|4, H, W)`` float tensor in ``[0, 1]`` as an RGB/RGBA image."""
    Image = _require_pil()
    t = (tensor.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    arr = t.permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(arr, "RGBA" if t.shape[0] == 4 else "RGB").save(path)


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def _estimate_key_color(img: Tensor, remove: Tensor) -> Tensor:
    """Mean color of the pixels to remove (fallback: :data:`DEFAULT_REF_RGB`)."""
    w = remove[0]
    total = w.sum()
    if total < 1:
        return torch.tensor(DEFAULT_REF_RGB)
    return (img * w).flatten(1).sum(1) / total


# ============================================================
# DATASETS
# ============================================================

class MaskFolderDataset(Dataset[Sample]):
    """Folder dataset: ``root/images/*`` + ``root/masks/*`` (matched by file stem).

    * Masks are the desired alpha: white = keep, black = remove (use
      ``invert_masks=True`` for the opposite convention).
    * An RGBA image without a mask file uses its own alpha channel as mask.
    * For the ``"color"`` task the key color comes from ``root/colors.json``
      (``{"stem": [r, g, b]}``, values in ``[0, 1]``) or is estimated as the mean
      color of the removed pixels.
    """

    def __init__(
        self,
        root: str | Path,
        img_size: int = 160,
        task: str = "color",
        invert_masks: bool = False,
        augment: bool = True,
    ) -> None:
        self.root = Path(root)
        self.img_size = img_size
        self.task = task
        self.invert_masks = invert_masks
        self.augment = augment
        img_dir = self.root / "images"
        if not img_dir.is_dir():
            img_dir = self.root
        mask_dir = self.root / "masks"
        masks = {p.stem: p for p in _list_images(mask_dir)} if mask_dir.is_dir() else {}
        self.items: list[tuple[Path, Path | None]] = []
        for img in _list_images(img_dir):
            mask = masks.get(img.stem)
            if mask is None and img.suffix.lower() not in (".png", ".webp"):
                logger.warning("No mask for %s, skipped", img.name)
                continue
            self.items.append((img, mask))
        if not self.items:
            raise FileNotFoundError(f"No (image, mask) pairs found in {self.root}")
        colors_file = self.root / "colors.json"
        self.colors: dict[str, list[float]] = (
            json.loads(colors_file.read_text()) if colors_file.is_file() else {}
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Sample:
        img_path, mask_path = self.items[idx]
        if mask_path is not None:
            img = load_image(img_path, self.img_size)
            keep = load_image(mask_path, self.img_size, mode="L")
        else:
            rgba = load_image(img_path, self.img_size, mode="RGBA")
            img, keep = rgba[:3], rgba[3:]
        if self.invert_masks:
            keep = 1.0 - keep
        if self.augment and random.random() < 0.5:
            img, keep = img.flip(-1), keep.flip(-1)
        remove = 1.0 - keep
        if img_path.stem in self.colors:
            ref = torch.tensor(self.colors[img_path.stem], dtype=torch.float32)
        else:
            ref = _estimate_key_color(img, remove)
        return img, remove, ref


def _random_texture(size: int, gen: torch.Generator) -> Tensor:
    """Smooth random RGB texture (gradient + low-frequency noise)."""
    low = torch.rand(3, 4, 4, generator=gen)
    tex = F.interpolate(low[None], size=(size, size), mode="bicubic", align_corners=False)[0]
    return tex.clamp(0, 1)


def _random_shapes_mask(size: int, gen: torch.Generator, n_max: int = 3) -> Tensor:
    """Union of random ellipses -> ``(1, H, W)`` soft subject mask."""
    ys, xs = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    mask = torch.zeros(size, size)
    n = int(torch.randint(1, n_max + 1, (1,), generator=gen))
    for _ in range(n):
        cx, cy, rx, ry = torch.rand(4, generator=gen).tolist()
        rx, ry = 0.1 + 0.3 * rx, 0.1 + 0.3 * ry
        d = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
        mask = torch.maximum(mask, torch.sigmoid((1 - d) * 12))
    return mask[None]


class SyntheticDataset(Dataset[Sample]):
    """Procedurally generated samples, no files and no Pillow needed.

    * ``"color"``: subjects over a solid (slightly noisy) key color.
    * ``"background"``: subjects over random textured backgrounds.

    Good for smoke tests, benchmarking a device, or pre-training before
    fine-tuning on real data.
    """

    def __init__(self, task: str = "color", length: int = 256, img_size: int = 64,
                 seed: int = 0) -> None:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")
        self.task, self.length, self.img_size, self.seed = task, length, img_size, seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Sample:
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
        s = self.img_size
        keep = _random_shapes_mask(s, gen)
        fg = _random_texture(s, gen)
        if self.task == "color":
            base = torch.tensor(DEFAULT_REF_RGB)
            ref = (base + 0.1 * (torch.rand(3, generator=gen) - 0.5)).clamp(0, 1)
            bg = ref.view(3, 1, 1) + 0.03 * torch.randn(3, s, s, generator=gen)
        else:
            ref = torch.tensor(DEFAULT_REF_RGB)
            bg = _random_texture(s, gen)
        img = (fg * keep + bg * (1 - keep)).clamp(0, 1)
        return img, 1.0 - keep, ref


class CompositeDataset(Dataset[Sample]):
    """Composite RGBA foregrounds (``foregrounds/*.png``) onto backgrounds.

    * ``"color"``: the background is a solid key color (random around ``key_rgb``).
    * ``"background"``: the background is a random image from ``backgrounds_dir``
      or a random synthetic texture when none is given.

    This is the easiest way to build a background-removal dataset on a phone:
    a few cut-out PNGs plus any photos.
    """

    def __init__(
        self,
        foregrounds_dir: str | Path,
        task: str = "background",
        backgrounds_dir: str | Path | None = None,
        img_size: int = 160,
        length: int | None = None,
        key_rgb: Sequence[float] = DEFAULT_REF_RGB,
    ) -> None:
        self.fgs = [p for p in _list_images(Path(foregrounds_dir))
                    if p.suffix.lower() in (".png", ".webp")]
        if not self.fgs:
            raise FileNotFoundError(f"No RGBA .png/.webp foregrounds in {foregrounds_dir}")
        self.bgs = _list_images(Path(backgrounds_dir)) if backgrounds_dir else []
        self.task, self.img_size = task, img_size
        self.length = length or len(self.fgs) * 4
        self.key_rgb = torch.tensor(list(key_rgb), dtype=torch.float32)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Sample:
        gen = torch.Generator().manual_seed(idx + random.randint(0, 2**31))
        rgba = load_image(self.fgs[idx % len(self.fgs)], self.img_size, mode="RGBA")
        fg, keep = rgba[:3], rgba[3:]
        s = self.img_size
        if self.task == "color":
            ref = (self.key_rgb + 0.1 * (torch.rand(3, generator=gen) - 0.5)).clamp(0, 1)
            bg = ref.view(3, 1, 1).expand(3, s, s)
        else:
            ref = self.key_rgb
            if self.bgs:
                j = int(torch.randint(len(self.bgs), (1,), generator=gen))
                bg = load_image(self.bgs[j], s)
            else:
                bg = _random_texture(s, gen)
        img = fg * keep + bg * (1 - keep)
        if random.random() < 0.5:
            img, keep = img.flip(-1), keep.flip(-1)
        return img.clamp(0, 1), 1.0 - keep, ref


# ============================================================
# LOSS / METRICS
# ============================================================

def segmentation_loss(main: Tensor, aux: Tensor, target: Tensor,
                      aux_weight: float = 0.4) -> Tensor:
    """BCE + soft Dice on the full-res logits, plus BCE deep supervision on ``aux``."""
    bce = F.binary_cross_entropy_with_logits(main, target)
    prob = torch.sigmoid(main)
    inter = (prob * target).flatten(1).sum(1)
    union = prob.flatten(1).sum(1) + target.flatten(1).sum(1)
    dice = 1 - ((2 * inter + 1) / (union + 1)).mean()
    aux_t = F.interpolate(target, size=aux.shape[-2:], mode="area")
    aux_bce = F.binary_cross_entropy_with_logits(aux, aux_t)
    return bce + dice + aux_weight * aux_bce


@torch.no_grad()
def iou(logits: Tensor, target: Tensor, threshold: float = 0.5) -> float:
    """Mean IoU of the "removed" region over a batch."""
    pred = torch.sigmoid(logits) > threshold
    gt = target > 0.5
    inter = (pred & gt).flatten(1).sum(1).float()
    union = (pred | gt).flatten(1).sum(1).float()
    return float(((inter + 1) / (union + 1)).mean())


# ============================================================
# CONFIG + TRAINER
# ============================================================

@dataclass
class TrainConfig:
    """All training options. ``None`` means "use the platform profile default"."""

    task: str = "color"
    data_dir: str | None = None          # MaskFolderDataset root; None -> synthetic data
    foregrounds_dir: str | None = None   # CompositeDataset foregrounds (RGBA)
    backgrounds_dir: str | None = None   # CompositeDataset backgrounds (optional)
    out_dir: str = "runs/codelor"
    init_from: str | None = None         # local dir or HF repo to fine-tune from
    resume: bool = True                  # resume from out_dir/last.pt if present

    epochs: int = 10
    max_steps: int | None = None         # stop after this many optimizer steps
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    grad_clip: float = 1.0
    val_split: float = 0.1
    seed: int = 0
    invert_masks: bool = False
    synthetic_length: int = 512

    batch_size: int | None = None
    accum_steps: int | None = None
    img_size: int | None = None
    num_threads: int | None = None
    num_workers: int | None = None
    device: str = "auto"
    amp: bool | None = None
    grad_checkpointing: bool = False

    log_every: int = 10
    save_every: int = 200                # optimizer steps between checkpoints

    model: dict[str, Any] = field(default_factory=lambda: dict(
        dim=128, depth=8, heads=8, tied_after=4))

    def resolved(self) -> TrainConfig:
        """Return a copy where ``None`` fields are filled from the platform profile."""
        prof = recommended_profile(self.device)
        data = asdict(self)
        for key in ("batch_size", "accum_steps", "img_size", "num_threads",
                    "num_workers", "amp"):
            if data[key] is None:
                data[key] = getattr(prof, key)
        if self.device == "auto":
            data["device"] = prof.device
        return TrainConfig(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown training options: {sorted(unknown)}")
        return cls(**data)


def build_dataset(cfg: TrainConfig) -> Dataset[Sample]:
    """Pick the dataset implied by ``cfg`` (folder, composite or synthetic)."""
    size = cfg.img_size or 160
    if cfg.data_dir:
        return MaskFolderDataset(cfg.data_dir, size, cfg.task, invert_masks=cfg.invert_masks)
    if cfg.foregrounds_dir:
        return CompositeDataset(cfg.foregrounds_dir, cfg.task, cfg.backgrounds_dir, size)
    logger.warning("No data_dir / foregrounds_dir given: training on synthetic data")
    return SyntheticDataset(cfg.task, cfg.synthetic_length, size, cfg.seed)


def _atomic_save(obj: Any, path: Path) -> None:
    """Save then rename, so a killed process never leaves a corrupt checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Trainer:
    """Minimal, resumable training loop shared by both tasks."""

    def __init__(self, cfg: TrainConfig, dataset: Dataset[Sample] | None = None,
                 log: Callable[[str], None] = print) -> None:
        if cfg.task not in TASKS:
            raise ValueError(f"unknown task {cfg.task!r}, expected one of {TASKS}")
        self.cfg = cfg.resolved()
        self.log = log
        c = self.cfg
        torch.manual_seed(c.seed)
        random.seed(c.seed)
        if c.num_threads:
            torch.set_num_threads(int(c.num_threads))
        self.device = resolve_device(c.device)
        self.out = Path(c.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.model = self._build_model().to(self.device)
        self.model.set_grad_checkpointing(c.grad_checkpointing)

        ds = dataset if dataset is not None else build_dataset(c)
        n_val = int(len(ds) * c.val_split) if c.val_split > 0 and len(ds) > 1 else 0  # type: ignore[arg-type]
        train_ds: Dataset[Sample]
        val_ds: Dataset[Sample] | None
        if n_val:
            gen = torch.Generator().manual_seed(c.seed)
            train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=gen)  # type: ignore[arg-type]
        else:
            train_ds, val_ds = ds, None
        loader_kw: dict[str, Any] = dict(
            batch_size=c.batch_size, num_workers=c.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=bool(c.num_workers),
        )
        self.train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kw)
        self.val_loader = DataLoader(val_ds, shuffle=False, **loader_kw) if val_ds else None

        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=c.lr, weight_decay=c.weight_decay,
        )
        steps_per_epoch = math.ceil(len(self.train_loader) / max(1, int(c.accum_steps or 1)))
        self.total_steps = c.max_steps or max(1, steps_per_epoch * c.epochs)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_lambda)
        self.use_amp = bool(c.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.step = 0
        self.epoch = 0
        self.best_iou = -1.0
        self._stop = False
        if c.resume and (self.out / "last.pt").is_file():
            self.load_checkpoint(self.out / "last.pt")

    # -- setup -----------------------------------------------------------------

    def _build_model(self) -> CodelorColor:
        c = self.cfg
        if not c.init_from:
            return build_model(c.task, **c.model)
        src = load_from_pretrained(c.init_from).train()
        if src.task == c.task:
            return src
        if c.task == "background":
            self.log(f"Initializing background model from color checkpoint {c.init_from}")
            return CodelorBackground.from_color_model(src)
        model = build_model(c.task, **src.config())
        transfer_weights(src, model)
        return model

    def _lr_lambda(self, step: int) -> float:
        warm = max(1, self.cfg.warmup_steps)
        if step < warm:
            return (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, self.total_steps - warm))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    # -- checkpoints -----------------------------------------------------------

    def save_checkpoint(self, name: str = "last.pt") -> Path:
        path = self.out / name
        _atomic_save({
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "best_iou": self.best_iou,
            "task": self.model.task,
            "model_config": self.model.config(),
            "train_config": asdict(self.cfg),
        }, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("task", "color") != self.model.task:
            raise ValueError(f"checkpoint {path} is for task {ckpt.get('task')!r}, "
                             f"not {self.model.task!r}")
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self.sched.load_state_dict(ckpt["sched"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step, self.epoch = int(ckpt["step"]), int(ckpt["epoch"])
        self.best_iou = float(ckpt.get("best_iou", -1.0))
        self.log(f"Resumed from {path} (epoch {self.epoch}, step {self.step})")

    def export(self, name: str = "final") -> Path:
        """Write a ``load_from_pretrained``-compatible directory."""
        return save_pretrained(self.model, self.out / name,
                               training={"step": self.step, "epoch": self.epoch,
                                         "best_iou": self.best_iou})

    # -- loop ------------------------------------------------------------------

    def _batches(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        for img, target, ref in self.train_loader:
            yield (img.to(self.device, non_blocking=True),
                   target.to(self.device, non_blocking=True),
                   ref.to(self.device, non_blocking=True).float())

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.log(f"Signal {signum} received: saving checkpoint and stopping...")
        self._stop = True

    @torch.no_grad()
    def evaluate(self) -> float | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        scores = []
        for img, target, ref in self.val_loader:
            img, target, ref = img.to(self.device), target.to(self.device), ref.to(self.device)
            scores.append(iou(self.model(img, ref), target))
        self.model.train()
        return sum(scores) / max(1, len(scores))

    def fit(self) -> CodelorColor:
        c = self.cfg
        accum = max(1, int(c.accum_steps or 1))
        previous = None
        try:
            previous = signal.signal(signal.SIGTERM, self._handle_signal)
        except (ValueError, OSError):  # not in main thread
            previous = None
        self.log(f"Training task={self.model.task} on {self.device} | "
                 f"batch={c.batch_size}x{accum} img={c.img_size} threads={torch.get_num_threads()} "
                 f"workers={c.num_workers} steps={self.total_steps}")
        self.model.train()
        t0, running, micro = time.time(), 0.0, 0
        try:
            # With max_steps set, the step budget drives training (epochs is ignored).
            while ((c.max_steps is not None or self.epoch < c.epochs)
                   and self.step < self.total_steps and not self._stop):
                self.opt.zero_grad(set_to_none=True)
                for img, target, ref in self._batches():
                    with torch.autocast(self.device.type, enabled=self.use_amp):
                        main, aux = self.model(img, ref, return_aux=True)
                        loss = segmentation_loss(main.float(), aux.float(), target) / accum
                    self.scaler.scale(loss).backward()
                    running += float(loss.detach()) * accum
                    micro += 1
                    if micro % accum:
                        continue
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    self.sched.step()
                    self.step += 1
                    if self.step % c.log_every == 0:
                        dt = time.time() - t0
                        self.log(f"epoch {self.epoch} step {self.step}/{self.total_steps} "
                                 f"loss {running / (c.log_every * accum):.4f} "
                                 f"lr {self.sched.get_last_lr()[0]:.2e} {dt:.1f}s")
                        running, t0 = 0.0, time.time()
                    if self.step % c.save_every == 0:
                        self.save_checkpoint()
                    if self.step >= self.total_steps or self._stop:
                        break
                if self._stop:
                    break
                self.epoch += 1
                score = self.evaluate()
                if score is not None:
                    self.log(f"epoch {self.epoch} val IoU {score:.4f}")
                    if score > self.best_iou:
                        self.best_iou = score
                        self.export("best")
                self.save_checkpoint()
        except KeyboardInterrupt:
            self.log("Interrupted: saving checkpoint...")
        finally:
            self.save_checkpoint()
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)
        path = self.export("final")
        self.log(f"Model exported to {path}")
        self.model.eval()
        return self.model


def train(cfg: TrainConfig | None = None, dataset: Dataset[Sample] | None = None,
          **overrides: Any) -> CodelorColor:
    """Convenience wrapper: ``train(task="background", data_dir="data", epochs=5)``."""
    cfg = cfg or TrainConfig()
    if overrides:
        cfg = TrainConfig(**{**asdict(cfg), **overrides})
    return Trainer(cfg, dataset).fit()
