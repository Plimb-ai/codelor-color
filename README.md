# Codelor Color Architecture 🌿🧠

This repository contains the pure PyTorch architecture for Codelor Color, a lightweight Transformer designed for Chroma Key (green screen) segmentation.

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

## 🏗️ Architecture Details

- ConvNeXt-style Stem: Anti-aliased downsampling for crisp edges.
- Transformer Blocks: 8 blocks with 2D RoPE, QK-Norm, and weight sharing (tied_after).
- Color Cross-Attention: Injects the target RGB color at each block.
- Gated Decoder: FPN-like decoder with deep supervision and a Sobel edge head.

Notes:

- The learned positional embedding table holds `max_tokens` tokens (default 1024).
  Larger inputs still run, but raise a `RuntimeWarning` because the positional
  embeddings are skipped beyond that capacity.
- `predict_mask` and `remove_green` switch the model to eval mode and run under
  `torch.inference_mode()`.

## 🧪 Development

    pip install -e ".[dev,hf]"
    pytest                # tests
    ruff check codelor_color tests
    mypy codelor_color

CI runs the same checks on every push and pull request.

## 📜 License

MIT License.
