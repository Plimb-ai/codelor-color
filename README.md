# Codelor Color Architecture 🌿🧠

This repository contains the pure PyTorch architecture for Codelor Color, a lightweight Transformer designed for Chroma Key (green screen) segmentation.

Since 2.3.0 the same backbone comes in two flavours, pick the one you need:

| Task | Class | Removes | Needs a color? |
|------|-------|---------|----------------|
| `"color"` | `CodelorColor` | a target color (chroma key) | yes (`ref_rgb`) |
| `"background"` | `CodelorBackground` | the background, whatever its color | no |

Both can be trained with the built-in trainer, on a GPU, a laptop CPU, or a phone with **Termux**.

Important note: The model weights (.safetensors), training scripts, and inference tools are available on the Hugging Face repository: plimb/codelor-color-1.2

Version 2.2.0 keeps the exact same weights as 2.1.0 (same parameter names and shapes), it only hardens the surrounding API. See [CHANGELOG.md](CHANGELOG.md).

## 📦 Installation

    pip install codelor-color

To download checkpoints directly from the Hugging Face Hub:

    pip install "codelor-color[hf]"

Development install (tests, linter, type checker):

    pip install -e ".[dev,hf]"

## 🚀 Usage

Load a pretrained checkpoint (local directory or Hub repo id):

    from codelor_color import load_from_pretrained, predict_mask, remove_green

    model = load_from_pretrained("plimb/codelor-color-1.2", device="cpu")

    import torch
    img = torch.rand(3, 160, 160)          # (3, H, W) float in [0, 1]

    mask, prob = predict_mask(model, img)  # (H, W) bool mask + (H, W) float probability
    rgba = remove_green(img, model)        # (4, H, W) RGBA tensor

Batches are supported: pass `(B, 3, H, W)` and you get `(B, H, W)` masks /
`(B, 4, H, W)` RGBA tensors back.

Instantiate the architecture directly and load your own weights:

    import torch
    from codelor_color import CodelorColor

    # Initialize the architecture
    model = CodelorColor(dim=128, depth=8, heads=8, tied_after=4)

    # Load your weights
    # state_dict = load_file("model.safetensors")
    # model.load_state_dict(state_dict)

    model.eval()

    # Forward pass
    img = torch.rand(1, 3, 160, 160)
    ref = torch.tensor([[0.20, 0.85, 0.30]]) # Target color (Green)

    with torch.no_grad():
        prob_mask = torch.sigmoid(model(img, ref))

The default reference color is available as `codelor_color.DEFAULT_REF_RGB`.

### Background removal

    from codelor_color import CodelorBackground, build_model, remove_background

    model = build_model("background", dim=128, depth=8, heads=8, tied_after=4)
    # or a trained one: model = load_from_pretrained("runs/codelor/final")
    rgba = remove_background(img, model)   # (4, H, W), no color needed

    # Start from a trained chroma-key checkpoint (stem + Transformer are reused):
    bg = CodelorBackground.from_color_model(load_from_pretrained("plimb/codelor-color-1.2"))

`load_from_pretrained` picks the right class from the `"task"` key of
`config.json` (missing = `"color"`, so all existing checkpoints load unchanged).
`save_pretrained(model, "dir")` writes a directory it can reload.

## 🏋️ Training

    pip install "codelor-color[train]"        # adds Pillow for image folders

    # Chroma key: data/images/*.jpg + data/masks/*.png
    codelor-color train --task color --data data --epochs 20

    # Background removal from image + mask pairs
    codelor-color train --task background --data data --epochs 20

    # Background removal from RGBA cut-outs composited on random photos,
    # fine-tuned from the color checkpoint
    codelor-color train --task background --foregrounds cutouts/ --backgrounds photos/ \
        --init-from plimb/codelor-color-1.2

    # Cut out images with any trained model
    codelor-color predict --model runs/codelor/final photo.jpg -o out/
    codelor-color predict --model plimb/codelor-color-1.2 --color "#22dd44" photo.jpg

Data conventions:

- **Masks are the alpha you want**: white = keep (subject), black = remove.
  Use `--invert-masks` for the opposite.
- An RGBA `.png` in `images/` without a mask uses its own alpha channel.
- For the color task, the key color comes from `colors.json`
  (`{"image_stem": [r, g, b]}` in `[0, 1]`) or is estimated from the removed pixels.
- Without any data the trainer uses procedurally generated samples (handy to
  benchmark a device).

Training is resumable: `runs/codelor/last.pt` is written atomically every
`--save-every` steps and on `Ctrl+C`/`SIGTERM`; re-running the same command
continues from it (`--no-resume` to start over). The result is exported to
`runs/codelor/final` (and `best` when a validation split is used).

Training extras: weight EMA (exported by default, `--no-ema`), BatchNorm
statistics frozen when fine-tuning with small batches, random crop / flip /
brightness-contrast augmentation, in-RAM image cache (fast epochs with 0
workers), L1 alpha loss for soft edges. New models default to `--preset v3`;
`--init-from` keeps the checkpoint's architecture unless `--preset` is given.

From Python:

    from codelor_color.training import TrainConfig, Trainer
    Trainer(TrainConfig(task="background", data_dir="data", epochs=5)).fit()

All options (`--batch-size`, `--accum-steps`, `--img-size`, `--threads`,
`--workers`, `--device`, `--grad-checkpointing`, `--dim`, ...) can also be
put in a JSON file passed with `--config`.

## 📱 Training on Termux (Android)

PyTorch has no Android wheels on PyPI, so install it from the Termux repos and
install codelor-color without dependencies:

    pkg install git && git clone https://github.com/Plimb-ai/codelor-color
    cd codelor-color && bash scripts/termux_setup.sh          # or --local for this checkout

or by hand:

    pkg install tur-repo
    pkg install python python-pip python-torch python-numpy python-pillow
    pip install --no-deps codelor-color
    pip install safetensors huggingface_hub   # optional (safetensors needs: pkg install rust)

Then `codelor-color info` should report `"termux": true`. On Termux the trainer
automatically uses:

- CPU, `num_workers=0` (no `/dev/shm` for DataLoader workers);
- half of the cores, max 4 (the big cores of big.LITTLE SoCs);
- batch 2 x 8 accumulation steps, 160 px images, no AMP.

Tips: run `termux-wake-lock` so Android does not kill the training, use
`--grad-checkpointing` if RAM is short, and start with `--init-from` a
pretrained checkpoint rather than from scratch. If `safetensors` is not
installed, weights are saved to `model.pt` and `load_from_pretrained` reads it.

## 🏗️ Architecture Details

- ConvNeXt-style Stem: Anti-aliased downsampling for crisp edges.
- Transformer Blocks: 8 blocks with 2D RoPE, QK-Norm, and weight sharing (tied_after).
- Color Cross-Attention: Injects the target RGB color at each block.
- Gated Decoder: FPN-like decoder with deep supervision and a Sobel edge head.
- Background variant: the color token is replaced by 4 learned background
  prototype tokens and the decoder drops the color-similarity channel.
- Optional gradient checkpointing (`model.set_grad_checkpointing(True)`) to train with less RAM.

### Presets: `v2` (original) and `v3` (improved)

`CodelorColor()` with default arguments is **exactly** the original
architecture (`plimb/codelor-color-1.2` loads and gives the same masks).
The `v3` preset fixes its structural weaknesses; it needs new weights
(train from scratch, or `--init-from` a v2 checkpoint with `--preset v3` to
reuse the stem and Transformer weights):

| | `v2` (default, compatible) | `v3` |
|---|---|---|
| Color conditioning | cross-attention to **one** token (the softmax is always 1, so it is just a constant bias) | FiLM scale + shift in every block |
| Positions | learned table indexed by flat token index (wrong when the resolution or aspect ratio changes, disabled above `max_tokens`) | RoPE only: any resolution |
| Normalization | BatchNorm (noisy with batch 2 on a phone) | GroupNorm (batch-size independent) |
| Decoder guides | RGB cosine to the key color | + CbCr chroma distance (ignores shadows) |
| Full-resolution head | two 3x3 convs on 69 channels per pixel (memory bound, most of the CPU time) | half-resolution head + 8-channel full-resolution refinement |
| Background model | 4 constant learned tokens | FiLM from a global context vector of the image |

    from codelor_color import build_model
    model = build_model("color", preset="v3")        # or "background"

For v2 models used at another resolution than their training one,
`model.set_pos_grid((10, 10))` (grid of a 160x160 training) interpolates the
learned positions in 2D: identical output at the training resolution, correct
positions elsewhere. The trainer records it automatically for new v2 models.

### Faster inference

    from codelor_color.model import optimize_for_inference
    fast = optimize_for_inference(model)                  # BatchNorm folded, same output
    fast = optimize_for_inference(model, quantize=True)   # + int8 Transformer (ARM/Termux)
    mask, prob = predict_mask(fast, img, max_side=512)    # big images: run at <= 512 px
    rgba = remove_green(img, fast, despill_strength=0.8)  # remove green fringes

Same weights as `codelor-color-1.2`, 4 CPU threads:

| Input | 2.2.2 | 2.3.0 | 2.3.0 + `optimize_for_inference` |
|---|---|---|---|
| 160x160 | 32 ms | 22 ms | 17 ms |
| 320x320 | 102 ms | 61 ms | 60 ms |
| 512x512 | 252 ms | 203 ms | 170 ms |

A `v3` model of the same size (1.38 M parameters) runs 320x320 in ~40 ms
and 512x512 in ~90 ms on the same machine.

Notes:

- The learned positional embedding table holds `max_tokens` tokens (default 1024).
  Larger inputs still run, but raise a `RuntimeWarning` because the positional
  embeddings are skipped beyond that capacity (use `set_pos_grid`, or `v3`).
- `predict_mask` and `remove_green` switch the model to eval mode and run under
  `torch.inference_mode()`.

## 🧪 Development

    pip install -e ".[dev,hf]"
    pytest                # tests
    ruff check codelor_color tests
    mypy codelor_color

CI runs the same checks on every push and pull request, plus a Termux-like
smoke training run and a package build (`python -m build` + `twine check`).

## 🚀 Automatic PyPI release

On every push to `main`, once tests and the build pass, the `publish` job uploads
the package to PyPI **if the version in `codelor_color/_version.py` is not on PyPI
yet**, then creates the `vX.Y.Z` GitHub release. To release: bump the version,
update the changelog, merge to `main`.

One-time setup (pick one):

1. **Trusted Publishing (recommended, no secret)**: on pypi.org > your project >
   *Publishing* > *Add a new publisher* > GitHub: owner `Plimb-ai`, repository
   `codelor-color`, workflow `ci.yml`, environment `pypi`.
2. **API token**: create a token on pypi.org (scope: project `codelor-color`) and
   add it in GitHub > Settings > Secrets and variables > Actions as
   `PYPI_API_TOKEN`.

## 📜 License

MIT License.
