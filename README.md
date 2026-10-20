# Codelor Color Architecture 🌿🧠

This repository contains the pure PyTorch architecture for Codelor Color, a lightweight Transformer designed for Chroma Key (green screen) segmentation.

Important note: The model weights (.safetensors), training scripts, and inference tools are available on the Hugging Face repository: plimb/codelor-color-1.2

## 📦 Installation

    pip install codelor-color

## 🚀 Usage

You can instantiate the architecture directly and load your own weights.

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

## 🏗️ Architecture Details

- ConvNeXt-style Stem: Anti-aliased downsampling for crisp edges.
- Transformer Blocks: 8 blocks with 2D RoPE, QK-Norm, and weight sharing (tied_after).
- Color Cross-Attention: Injects the target RGB color at each block.
- Gated Decoder: FPN-like decoder with deep supervision and a Sobel edge head.

## 📜 License

MIT License.
