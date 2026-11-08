#!/data/data/com.termux/files/usr/bin/bash
# Install codelor-color on Termux (Android) for training and inference.
#
# PyTorch has no Android wheels on PyPI, so torch/numpy/pillow come from the
# Termux package repositories (tur-repo provides python-torch).
#
# Usage:  bash scripts/termux_setup.sh            # from PyPI
#         bash scripts/termux_setup.sh --local    # from this checkout (editable)
set -euo pipefail

pkg update -y
pkg install -y tur-repo
pkg install -y python python-pip python-torch python-numpy python-pillow

# safetensors is a Rust extension: try the prebuilt Termux package first,
# then build it with Rust if needed. codelor-color still works without it
# (weights are then saved to model.pt), but Hugging Face checkpoints need it.
if ! python -c "import safetensors" 2>/dev/null; then
    pkg install -y python-safetensors 2>/dev/null \
        || { pkg install -y rust binutils && pip install safetensors; } \
        || echo "WARNING: safetensors unavailable, falling back to model.pt files"
fi

# --no-deps: torch/numpy are already provided by pkg, never let pip rebuild them.
if [ "${1:-}" = "--local" ]; then
    pip install --no-deps -e .
else
    pip install --no-deps --upgrade codelor-color
fi
pip install huggingface_hub || echo "WARNING: huggingface_hub not installed (Hub downloads disabled)"

# Keep the CPU awake while training with the screen off.
command -v termux-wake-lock >/dev/null && termux-wake-lock || true

codelor-color info
