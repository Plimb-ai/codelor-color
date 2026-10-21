# Changelog

All notable changes to this project are documented in this file.

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

### Changed

- Python 3.8/3.9 support dropped: `requires-python` is now `>=3.10`
  (matching current PyTorch). Annotations use modern syntax
  (`tuple[...]`, `X | Y`, `collections.abc.Sequence`).

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
