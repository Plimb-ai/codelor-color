"""Command line interface: ``codelor-color {info,train,predict}``.

Examples::

    codelor-color info
    codelor-color train --task color --data data/greenscreen --epochs 20
    codelor-color train --task background --foregrounds cutouts/ --backgrounds photos/ \\
        --init-from plimb/codelor-color-1.2
    codelor-color predict --model runs/codelor/final photo.jpg -o out/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._version import __version__


def _parse_color(value: str) -> tuple[float, float, float]:
    """Accept ``#22dd44``, ``22dd44`` or ``0.2,0.85,0.3`` / ``51,217,76``."""
    v = value.strip()
    if "," not in v:
        h = v.lstrip("#")
        if len(h) != 6:
            raise argparse.ArgumentTypeError(f"invalid color {value!r}")
        return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)
    parts = [float(x) for x in v.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"invalid color {value!r}")
    if any(x > 1 for x in parts):
        parts = [x / 255 for x in parts]
    return (parts[0], parts[1], parts[2])


def _cmd_info(_args: argparse.Namespace) -> int:
    from .device import describe_environment

    print(json.dumps({"codelor_color": __version__, **describe_environment()}, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .training import TrainConfig, Trainer

    data: dict[str, Any] = {}
    if args.config:
        data.update(json.loads(Path(args.config).read_text()))
    mapping = {
        "task": args.task, "data_dir": args.data, "foregrounds_dir": args.foregrounds,
        "backgrounds_dir": args.backgrounds, "out_dir": args.out, "init_from": args.init_from,
        "epochs": args.epochs, "max_steps": args.max_steps, "lr": args.lr,
        "batch_size": args.batch_size, "accum_steps": args.accum_steps,
        "img_size": args.img_size, "num_threads": args.threads, "num_workers": args.workers,
        "device": args.device, "val_split": args.val_split, "seed": args.seed,
        "save_every": args.save_every, "log_every": args.log_every,
        "preset": args.preset, "ema_decay": args.ema_decay,
    }
    data.update({k: v for k, v in mapping.items() if v is not None})
    if args.no_resume:
        data["resume"] = False
    if args.grad_checkpointing:
        data["grad_checkpointing"] = True
    if args.invert_masks:
        data["invert_masks"] = True
    if args.no_ema:
        data["ema_decay"] = None
    if args.freeze_bn is not None:
        data["freeze_bn"] = args.freeze_bn == "on"
    if args.no_cache:
        data["cache"] = False
    if args.no_augment:
        data["augment"] = False
    model_kw = dict(data.get("model", {"dim": 128, "depth": 8, "heads": 8, "tied_after": 4}))
    for key in ("dim", "depth", "heads", "tied_after"):
        if getattr(args, key) is not None:
            model_kw[key] = getattr(args, key)
    data["model"] = model_kw
    Trainer(TrainConfig.from_dict(data)).fit()
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F

    from .device import resolve_device
    from .model import (
        DEFAULT_REF_RGB,
        despill,
        load_from_pretrained,
        optimize_for_inference,
        remove_green,
    )
    from .training import load_image, save_image

    device = resolve_device(args.device)
    model = load_from_pretrained(args.model, device=device)
    if args.pos_grid:
        model.set_pos_grid(args.pos_grid)
    if not args.no_optimize:
        model = optimize_for_inference(model, quantize=args.int8)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in args.inputs:
        img = load_image(src)
        h, w = img.shape[-2:]
        x = img
        if args.size:
            x = F.interpolate(img[None], size=(args.size, args.size), mode="bilinear",
                              align_corners=False)[0]
        alpha = remove_green(x, model, color_rgb=args.color, threshold=args.threshold,
                             device=device, max_side=args.max_side)[3:]
        if alpha.shape[-2:] != (h, w):
            alpha = F.interpolate(alpha[None], size=(h, w), mode="bilinear",
                                  align_corners=False)[0]
        alpha = alpha.clamp(0, 1)
        rgb = img
        if args.despill > 0 and model.needs_color:
            rgb = despill(img, alpha, args.color or DEFAULT_REF_RGB, args.despill)
        dst = out_dir / (Path(src).stem + ".png")
        save_image(torch.cat([rgb, alpha], 0), dst)
        print(f"{src} -> {dst}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codelor-color", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show platform / Termux diagnostics and recommended settings")

    t = sub.add_parser("train", help="train a color-removal or background-removal model")
    t.add_argument("--config", help="JSON file with TrainConfig fields (CLI flags override it)")
    t.add_argument("--task", choices=["color", "background"],
                   help="'color' removes a key color, 'background' removes the background")
    t.add_argument("--data", help="folder with images/ and masks/ (white = keep)")
    t.add_argument("--foregrounds", help="folder of RGBA cut-outs to composite")
    t.add_argument("--backgrounds", help="folder of background photos (background task)")
    t.add_argument("--out", help="output directory (default runs/codelor)")
    t.add_argument("--init-from", help="checkpoint dir or HF repo id to fine-tune from")
    t.add_argument("--epochs", type=int)
    t.add_argument("--max-steps", type=int)
    t.add_argument("--lr", type=float)
    t.add_argument("--batch-size", type=int)
    t.add_argument("--accum-steps", type=int, help="gradient accumulation steps")
    t.add_argument("--img-size", type=int)
    t.add_argument("--threads", type=int, help="torch CPU threads")
    t.add_argument("--workers", type=int, help="DataLoader workers (0 on Termux)")
    t.add_argument("--device", help="auto, cpu, cuda, mps")
    t.add_argument("--val-split", type=float)
    t.add_argument("--seed", type=int)
    t.add_argument("--save-every", type=int)
    t.add_argument("--log-every", type=int)
    t.add_argument("--dim", type=int)
    t.add_argument("--depth", type=int)
    t.add_argument("--heads", type=int)
    t.add_argument("--tied-after", type=int)
    t.add_argument("--grad-checkpointing", action="store_true", help="lower RAM, slower")
    t.add_argument("--invert-masks", action="store_true", help="masks are white = remove")
    t.add_argument("--no-resume", action="store_true", help="ignore out/last.pt")
    t.add_argument("--preset", choices=["v2", "v3"],
                   help="architecture: v3 (default from scratch, faster, resolution-free) "
                        "or v2 (original). With --init-from: convert the checkpoint")
    t.add_argument("--ema-decay", type=float, help="weight EMA decay (default 0.999)")
    t.add_argument("--no-ema", action="store_true", help="export raw weights instead of EMA")
    t.add_argument("--freeze-bn", choices=["on", "off"],
                   help="freeze BatchNorm stats (default: on when fine-tuning with batch < 8)")
    t.add_argument("--no-cache", action="store_true", help="do not keep images in RAM")
    t.add_argument("--no-augment", action="store_true", help="disable crop/flip/jitter")

    p = sub.add_parser("predict", help="cut out images, writes RGBA PNGs")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--model", required=True, help="checkpoint dir or HF repo id")
    p.add_argument("-o", "--output", default="out")
    p.add_argument("--color", type=_parse_color, default=None,
                   help="key color for color models (#22dd44 or r,g,b)")
    p.add_argument("--size", type=int, default=None, help="inference resolution (square)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-side", type=int, default=None,
                   help="run the model with the longest side resized to this (faster)")
    p.add_argument("--despill", type=float, default=0.0,
                   help="color models: remove key-color fringes, strength 0..1")
    p.add_argument("--int8", action="store_true", help="int8 quantized Transformer (CPU/ARM)")
    p.add_argument("--no-optimize", action="store_true", help="skip BatchNorm fusion")
    p.add_argument("--pos-grid", type=int, nargs=2, metavar=("ROWS", "COLS"),
                   help="v2 models: token grid used in training (10 10 for 160px) to "
                        "fix positions at other resolutions")
    p.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    handlers = {"info": _cmd_info, "train": _cmd_train, "predict": _cmd_predict}
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
