# Changelog

All notable changes to this project are documented in this file.

## [2.5.0] - 2026-11-09

Nothing removed; `plimb/codelor-color-1.2` (v2), v3 and v4 checkpoints load
unchanged (1.2 re-checked: 100% identical masks).

### Added

- **Preset `v5`** (new default for models trained from scratch): v4 with the
  last decoder stage at 1/4 resolution (`head="quarter"`). Same accuracy as v4
  in the benchmark (README), ~20% faster inference at 320-512 px and 5-16%
  faster training steps.
- CLI: `--preset v5`.

### Fixed

- Weight tying with `depth > 2 * tied_after` (e.g. `depth=12, tied_after=4`)
  raised `IndexError`; shared blocks are now reused cyclically. Unchanged for
  `depth <= 2 * tied_after`, which covers every existing checkpoint.

## [2.4.0] - 2026-11-09

Nothing removed; `plimb/codelor-color-1.2` (v2) and v3 checkpoints from 2.3.0
load unchanged.

### Added

- **Preset `v4`** (new default for models trained from scratch): v3 with
  BatchNorm. Chosen by an ablation on harder synthetic data, see the README:
  same IoU as v2 for chroma key and background removal (also with Termux-sized
  batches of 2), 1.6-2.4x lower alpha error, correct at any resolution, ~2x
  faster than v2 at 320-512 px, and its BatchNorm folds at inference.
- `predict_mask(..., refine="guided")` / `predict --refine guided`: guided
  filter / guided upsampling of the alpha (chroma guide for color models).
  Works with any model; measured gain is small, off by default.
- Training: `edge_weight` (extra BCE weight on mask borders, default 1.0),
  multi-scale batches (`multiscale`, on by default for resolution-independent
  models: +1.9 IoU points at an unseen resolution), `synthetic_hard`.
- CLI: `--preset v4`, `--edge-weight`, `--multiscale`, `--synthetic-hard`,
  `predict --refine`.

### Fixed

- Synthetic background-removal data used the same kind of texture for the
  subject and the background, so models only learned "everything is
  background". Subjects are now in focus, shaded and cast a shadow over a
  blurred background. The v2/v3 background numbers published in the 2.3.0
  README were measured on that data and were meaningless; they are replaced.

### Changed

- The 2.3.0 claim "v3 matches v2" did not hold on harder data: v3 loses
  0.4-1.7 IoU points, which the ablation traced to GroupNorm. v3 is kept for
  compatibility; use v4 for new models.

## [2.3.0] - 2026-11-08

Nothing was removed: `CodelorColor` weights, parameter names and the existing
API are unchanged. `plimb/codelor-color-1.2` loads and gives the same masks
(max logit difference 4e-5, 100% identical mask pixels, checked against the
Hub's `modeling_codelor.py`).

### Fixed

- `load_from_pretrained` failed on Hub configs such as `plimb/codelor-color-1.2`
  (`"task": "image-segmentation"`): the task is now read from `codelor_task`,
  then `architectures`, then `task` only when it is a Codelor task.
- Trainer: `GradScaler` fallback for torch < 2.3; leftover gradient
  accumulation micro-batches at the end of an epoch are no longer dropped;
  validation uses non-augmented images.

### Performance (same weights, same outputs)

- `rope_2d` sin/cos tables cached per token grid.
- Single-token color cross-attention computed in closed form (softmax over one
  key is 1): no query projection / QK-norm / attention for the color path.
- Color cosine guide computed with plain reductions (`F.cosine_similarity` on an
  expanded map was 23% of the CPU time at 512 px).
- `optimize_for_inference()`: BatchNorm folded into convolutions, optional int8
  dynamic quantization (`quantize=True`, for ARM/Termux) and `channels_last`.
- Measured on 4 CPU threads, codelor-color-1.2: 160 px 32 -> 17 ms,
  320 px 102 -> 60 ms, 512 px 252 -> 170 ms.
- `predict_mask(..., max_side=N)` runs big images at a reduced resolution and
  upsamples the probability.

### Added: architecture preset "v3" (opt-in, new weights)

- `build_model(task, preset="v3")` / `--preset v3` (default when training from
  scratch; fine-tuning keeps the checkpoint's architecture unless `--preset`
  is given, in which case matching weights are transferred).
- FiLM color conditioning instead of the degenerate one-token cross-attention;
  for background removal, FiLM driven by a global context vector of the image.
- RoPE-only positions: any resolution / aspect ratio, no `max_tokens` limit.
- GroupNorm: independent of the batch size (Termux trains with batch 2).
- CbCr chroma-distance guide (robust to shadows on green screens).
- Light head: last decoder stage at half resolution + 8-channel full-resolution
  refinement; ~2x faster than v2 at 320-512 px.
- For original-architecture models: `set_pos_grid((rows, cols))` / `pos_grid`
  interpolates the learned positions in 2D, identical at the training
  resolution and correct at others; set automatically by the trainer.

### Added: training

- EMA of the weights (`ema_decay`, exported by default), BatchNorm statistics
  frozen when fine-tuning with small batches (`freeze_bn`), random resized
  crop + flip + brightness/contrast augmentation, in-RAM image cache, subject
  scale/position randomization for composites, L1 alpha loss term.
- `despill()` and `remove_green(..., despill_strength=...)` / `predict --despill`.
- CLI: `--preset`, `--no-ema`, `--freeze-bn`, `--no-cache`, `--no-augment`;
  `predict --max-side --despill --int8 --pos-grid`.

### Added

- **Background removal model** `CodelorBackground` (task `"background"`): same
  backbone, learned background tokens instead of a target color. `build_model(task)`,
  `TASKS`, `remove_background()`, `CodelorBackground.from_color_model()` and
  `transfer_weights()` to fine-tune from a chroma-key checkpoint.
- `save_pretrained()` (writes `codelor_task` + `architectures`);
  `load_from_pretrained()` falls back to `model.pt` when safetensors is not
  available.
- `codelor_color.training`: resumable `Trainer` / `TrainConfig`, folder,
  composite (RGBA cut-outs) and synthetic datasets, BCE + Dice + deep supervision loss.
- `codelor_color.device`: Termux/Android detection and per-platform training
  profiles (threads, workers, batch/accumulation, AMP).
- Termux support: `scripts/termux_setup.sh`, atomic checkpoints, `SIGTERM`
  handling, optional gradient checkpointing (`set_grad_checkpointing`).
- `codelor-color` command (`info`, `train`, `predict`) and `python -m codelor_color`.
- `train` extra (Pillow).
- CI: Termux-like smoke training, package build + `twine check`, and automatic
  PyPI publication + GitHub release on `main` when the version is new.

## [2.2.2] - 2026-10-21

Patch release fixing the remaining mypy error from CI.

### Fixed

- `SobelEdge` declares its `sx`/`sy` buffers with class annotations so mypy
  types them as `Tensor` (previously `Tensor | Module`, which broke
  `F.conv2d`). No behaviour change: buffers and `state_dict` keys are identical.

## [2.2.1] - 2026-10-21

Patch release: the 2.2.0 files already on PyPI cannot be replaced (PyPI
releases are immutable), so the CI/lint fixes ship as 2.2.1. No behaviour
change; weights and API are identical to 2.2.0.

### Changed

- Python 3.8/3.9 support dropped: `requires-python` is now `>=3.10`
  (matching current PyTorch). The 2.2.0 metadata on PyPI still advertises
  `>=3.8`.
- Annotations use modern syntax (`tuple[...]`, `X | Y`,
  `collections.abc.Sequence`) required by current ruff (UP006/UP007/UP045).

### Fixed

- Lint errors reported by the GitHub Actions CI on the 2.2.0 tag.
- `CodelorColor._get()` now casts `nn.ModuleList` lookups so mypy accepts
  the weight-tying indirection.

## [2.2.0] - 2026-10-21

Architecture weights are unchanged: checkpoints trained for 2.1.0 load as-is.

### Added

- Type hints and docstrings across the whole public API.
- Named constants: `DEFAULT_REF_RGB`, `STEM_W1`, `STEM_W2`, `ROPE_BASE`, `EPS`.
- Structured logging (`logging.getLogger("codelor_color")`).
- `RuntimeWarning` when an image yields more than `max_tokens` spatial tokens
  (previously the positional embeddings were silently skipped).
- Input validation (shapes, dtypes, threshold, batch/color consistency) with
  explicit `ValueError`/`TypeError` messages.
- Batch support in `predict_mask` / `remove_green`: `(B, 3, H, W)` in,
  `(B, H, W)` masks and `(B, 4, H, W)` RGBA out (single-image behaviour unchanged).
- `predict_mask`/`remove_green` now run under `torch.inference_mode()`, force
  eval mode and move the model to the requested device when needed.
- `CodelorColor.config()` now also returns `drop_path_rate`.
- Test suite (`tests/`), GitHub Actions CI, ruff/mypy/pytest configuration.
- Changelog.

### Fixed

- `config()` → `load_from_pretrained()` round-trip (`KeyError` on
  `hyperparameters`); both the flat and the nested format are accepted, and
  unknown keys are ignored with a warning.
- Undeclared runtime dependencies (`numpy`, `safetensors`) missing from
  `pyproject.toml`; `huggingface_hub` is now the optional `hf` extra with a
  clear install hint.
- `requirements.txt` listed unused packages (`pillow`, `matplotlib`, `tqdm`,
  `opencv-python`).
- Unreachable `else` branch in the drop-path schedule.
- Version string duplicated in two files: it now lives only in
  `codelor_color/_version.py` (read by `pyproject.toml`).

## [2.1.0] - 2026-10-20

- Initial public release of the Codelor Color architecture.
