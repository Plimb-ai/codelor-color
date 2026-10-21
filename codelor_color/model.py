"""Codelor Color: lightweight Transformer architecture for chroma-key segmentation.

The model takes an image ``img`` of shape ``(B, 3, H, W)`` and a target color
``ref_rgb`` of shape ``(B, 3)`` and returns a per-pixel logit map ``(B, 1, H, W)``.
Apply ``torch.sigmoid`` to obtain foreground probabilities.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

#: Default reference color used when none is supplied (RGB in [0, 1]).
DEFAULT_REF_RGB: tuple[float, float, float] = (0.20, 0.85, 0.30)

#: Channel width of the stem feature maps at 1/4 (``f4``) resolution.
STEM_W1: int = 32
#: Channel width of the stem feature map at 1/8 (``f8``) resolution.
STEM_W2: int = 64

#: Rotary embedding frequency base.
ROPE_BASE: float = 100.0
#: LayerScale initialization value.
LAYER_SCALE_INIT: float = 1e-6
#: Standard deviation of the learned positional embedding initialization.
POS_EMB_STD: float = 0.02
#: Numerical epsilon for cosine similarity and the Sobel magnitude.
EPS: float = 1e-6

#: Hyper-parameters accepted by :meth:`CodelorColor.config` / ``config.json``.
HYPERPARAM_KEYS: tuple[str, ...] = (
    "dim",
    "depth",
    "heads",
    "tied_after",
    "mlp_ratio",
    "max_tokens",
    "drop_path_rate",
)


# ============================================================
# ARCHITECTURE CODELOR COLOR
# ============================================================

class RMSNorm(nn.Module):
    """Root-mean-square normalization (no mean subtraction)."""

    def __init__(self, dim: int, eps: float = EPS) -> None:
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class SwiGLU(nn.Module):
    """Gated feed-forward module: ``w3(silu(a) * b)`` with a fused ``w12`` projection."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(dim, hidden * 2, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class LayerScale(nn.Module):
    """Per-channel residual scaling (CaiT-style)."""

    def __init__(self, dim: int, init: float = LAYER_SCALE_INIT) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class DropPath(nn.Module):
    """Stochastic depth: drops whole samples with probability ``p`` during training."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep
        return x * mask / keep


def rope_2d(q: Tensor, k: Tensor, h: int, w: int, base: float = ROPE_BASE) -> tuple[Tensor, Tensor]:
    """Apply 2D rotary position embeddings, splitting the head dim in half for x and y."""
    B, Hh, N, D = q.shape
    d4 = D // 4
    freqs = 1.0 / (base ** (torch.arange(0, d4, device=q.device).float() / d4))
    xs = torch.arange(w, device=q.device).float()
    ys = torch.arange(h, device=q.device).float()
    ax = torch.outer(xs, freqs)[None, :, :].expand(h, w, d4).reshape(h * w, d4)
    ay = torch.outer(ys, freqs)[:, None, :].expand(h, w, d4).reshape(h * w, d4)
    ang = torch.cat([ax, ay], dim=-1)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]

    def rot(x: Tensor) -> Tensor:
        x1, x2 = x[..., : D // 2], x[..., D // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return rot(q), rot(k)


class Attention(nn.Module):
    """Multi-head self-attention with QK-norm and 2D RoPE."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.heads = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = RMSNorm(self.dh)
        self.k_norm = RMSNorm(self.dh)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope_2d(q, k, h, w)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class ColorCrossAttention(nn.Module):
    """Cross-attention where spatial tokens (queries) attend to a color token (key/value).

    This is the conditioning path: the target RGB is injected into every block.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.heads = heads
        self.dh = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.q_norm = RMSNorm(self.dh)
        self.k_norm = RMSNorm(self.dh)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, spatial: Tensor, cond: Tensor) -> Tensor:
        B, N, C = spatial.shape
        M = cond.shape[1]
        q = self.q(spatial).reshape(B, N, self.heads, self.dh).transpose(1, 2)
        kv = self.kv(cond).reshape(B, M, 2, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    """Pre-norm Transformer block: self-attention + SwiGLU FFN, both with LayerScale and DropPath."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 3.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.ls1 = LayerScale(dim)
        self.dp1 = DropPath(drop_path)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)
        self.dp2 = DropPath(drop_path)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        x = x + self.dp1(self.ls1(self.attn(self.norm1(x), h, w)))
        return x + self.dp2(self.ls2(self.ffn(self.norm2(x))))


class AntiAliasDownsample(nn.Module):
    """BlurPool-style downsampling: frozen binomial blur, then stride-2 conv + BN + SiLU."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.blur = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False)
        kernel = torch.tensor([1.0, 2.0, 1.0])
        kernel = kernel / kernel.sum()
        bk = kernel[:, None] * kernel[None, :]
        with torch.no_grad():
            self.blur.weight.copy_(bk.expand(c_in, 1, 3, 3))
        self.blur.weight.requires_grad_(False)
        self.conv = nn.Conv2d(c_in, c_out, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(self.blur(x))))


class ConvNeXtStem(nn.Module):
    """Patch-embed stem producing the feature pyramid ``f4``, ``f8``, ``f16``.

    Channel widths come from :data:`STEM_W1` / :data:`STEM_W2`, which are shared
    with :class:`GatedDecoder`; change them together or the decoder will not load.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(3, STEM_W1, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(STEM_W1),
            nn.SiLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(STEM_W1, STEM_W1, 7, padding=3, groups=STEM_W1, bias=False),
            nn.BatchNorm2d(STEM_W1),
            nn.SiLU(inplace=True),
            nn.Conv2d(STEM_W1, STEM_W1, 1, bias=False),
            nn.BatchNorm2d(STEM_W1),
        )
        self.down8 = AntiAliasDownsample(STEM_W1, STEM_W2)
        self.down16 = AntiAliasDownsample(STEM_W2, dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.patch(x)
        x = x + self.block1(x)
        f4 = x
        f8 = self.down8(f4)
        f16 = self.down16(f8)
        return f4, f8, f16


class SobelEdge(nn.Module):
    """Fixed Sobel X/Y filters returning the gradient magnitude of a gray image."""

    sx: Tensor
    sy: Tensor

    def __init__(self) -> None:
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3)
        self.register_buffer("sx", sx)
        self.register_buffer("sy", sy)

    def forward(self, gray: Tensor) -> Tensor:
        ex = F.conv2d(gray, self.sx, padding=1)
        ey = F.conv2d(gray, self.sy, padding=1)
        return (ex**2 + ey**2 + EPS).sqrt()


class GatedDecoder(nn.Module):
    """FPN-like decoder with learnable sigmoid gates, deep supervision and an edge head.

    At full resolution it concatenates ``[decoded, raw image, cosine-similarity to the
    target color, Sobel edge map]`` before the prediction head.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj16 = nn.Conv2d(dim, dim, 1, bias=False)
        self.aux_head = nn.Conv2d(dim, 1, 1)
        self.up8 = nn.Sequential(
            nn.Conv2d(dim + STEM_W2, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(dim + STEM_W1, STEM_W2, 3, padding=1, bias=False),
            nn.BatchNorm2d(STEM_W2),
            nn.SiLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(STEM_W2 + 3 + 1 + 1, STEM_W1, 3, padding=1, bias=False),
            nn.BatchNorm2d(STEM_W1),
            nn.SiLU(inplace=True),
            nn.Conv2d(STEM_W1, STEM_W1, 3, padding=1, bias=False),
            nn.BatchNorm2d(STEM_W1),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Conv2d(STEM_W1, 1, 1)
        self.gate8 = nn.Parameter(torch.zeros(1))
        self.gate4 = nn.Parameter(torch.zeros(1))
        self.edge = SobelEdge()

    def forward(
        self, latent: Tensor, f4: Tensor, f8: Tensor, img: Tensor, ref_rgb: Tensor
    ) -> tuple[Tensor, Tensor]:
        B, _, H, W = img.shape
        x = self.proj16(latent)
        aux = self.aux_head(x)
        aux = F.interpolate(aux, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=f8.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up8(torch.cat([x, torch.sigmoid(self.gate8) * f8], 1))
        x = F.interpolate(x, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up4(torch.cat([x, torch.sigmoid(self.gate4) * f4], 1))
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        ref_map = ref_rgb.view(B, 3, 1, 1).expand(B, 3, H, W)
        cos = F.cosine_similarity(img, ref_map, dim=1, eps=EPS).unsqueeze(1)
        gray = img.mean(1, keepdim=True)
        edge = self.edge(gray)
        x = self.up1(torch.cat([x, img, cos, edge], 1))
        return self.head(x), aux


class CodelorColor(nn.Module):
    """Chroma-key segmentation model: ConvNeXt stem + color-conditioned Transformer + gated decoder.

    Args:
        dim: Transformer width. Must be divisible by ``heads``.
        depth: Number of Transformer blocks executed (weights are shared after ``tied_after``).
        heads: Number of attention heads.
        tied_after: Blocks with index >= this value reuse earlier weights (0 disables tying).
        mlp_ratio: Hidden-size multiplier of the SwiGLU FFN.
        max_tokens: Capacity of the learned positional embedding table.
        drop_path_rate: Maximum stochastic-depth probability (linearly ramped over depth).

    The model returns ``(B, 1, H, W)`` logits, or ``((B, 1, H, W), (B, 1, h4, w4))``
    when ``return_aux=True``.
    """

    def __init__(
        self,
        dim: int = 128,
        depth: int = 8,
        heads: int = 8,
        tied_after: int = 4,
        mlp_ratio: float = 3.0,
        max_tokens: int = 1024,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.tied_after = tied_after
        self.mlp_ratio = mlp_ratio
        self.max_tokens = max_tokens
        self.drop_path_rate = drop_path_rate
        self._pos_warned = False
        self.stem = ConvNeXtStem(dim)
        self.color_mlp = nn.Sequential(
            nn.Linear(3, dim), nn.SiLU(inplace=True), nn.Linear(dim, dim)
        )
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=POS_EMB_STD)
        n_unique = depth if tied_after <= 0 else min(depth, tied_after)
        dp = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            [Block(dim, heads, mlp_ratio, drop_path=dp[i]) for i in range(n_unique)]
        )
        self.color_xattn = nn.ModuleList(
            [ColorCrossAttention(dim, heads) for _ in range(n_unique)]
        )
        self.norm_out = RMSNorm(dim)
        self.decoder = GatedDecoder(dim)

    def _get(self, i: int) -> tuple[Block, ColorCrossAttention]:
        idx = i - self.tied_after if self.tied_after > 0 and i >= self.tied_after else i
        return cast(Block, self.blocks[idx]), cast(ColorCrossAttention, self.color_xattn[idx])

    def forward(
        self, img: Tensor, ref_rgb: Tensor, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Predict foreground logits.

        Args:
            img: ``(B, 3, H, W)`` float tensor, values in ``[0, 1]``.
            ref_rgb: ``(B, 3)`` target color in RGB ``[0, 1]``.
            return_aux: also return the 1/4-resolution deep-supervision logit.

        Returns:
            ``(B, 1, H, W)`` logits, or ``(main, aux)`` if ``return_aux``.
        """
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(f"img must have shape (B, 3, H, W), got {tuple(img.shape)}")
        if ref_rgb.ndim != 2 or ref_rgb.shape[1] != 3:
            raise ValueError(f"ref_rgb must have shape (B, 3), got {tuple(ref_rgb.shape)}")
        if ref_rgb.shape[0] != img.shape[0]:
            raise ValueError(
                f"batch mismatch: img has {img.shape[0]} samples, ref_rgb has {ref_rgb.shape[0]}"
            )

        B = img.shape[0]
        f4, f8, f16 = self.stem(img)
        _, _, h, w = f16.shape
        N = h * w
        tokens = f16.flatten(2).transpose(1, 2)
        if N > self.pos.shape[1]:
            if not self._pos_warned:
                warnings.warn(
                    f"Input yields {N} spatial tokens but max_tokens={self.pos.shape[1]}; "
                    "positional embeddings are skipped for this and subsequent inputs. "
                    "Use a larger max_tokens or a smaller image.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._pos_warned = True
        else:
            tokens = tokens + self.pos[:, :N]
        c_tok = self.color_mlp(ref_rgb).unsqueeze(1)
        x = tokens
        for i in range(self.depth):
            blk, xa = self._get(i)
            x = blk(x, h, w)
            x = x + xa(x, c_tok)
        x = self.norm_out(x)
        latent = x.transpose(1, 2).reshape(B, -1, h, w)
        main, aux = self.decoder(latent, f4, f8, img, ref_rgb)
        return (main, aux) if return_aux else main

    def config(self) -> dict[str, Any]:
        """Return the hyper-parameters needed to rebuild this model.

        The result is accepted both directly and nested under a
        ``"hyperparameters"`` key by :func:`load_from_pretrained`.
        """
        return dict(
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            tied_after=self.tied_after,
            mlp_ratio=self.mlp_ratio,
            max_tokens=self.max_tokens,
            drop_path_rate=self.drop_path_rate,
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, depth={self.depth}, heads={self.heads}, "
            f"tied_after={self.tied_after}, mlp_ratio={self.mlp_ratio}, "
            f"max_tokens={self.max_tokens}"
        )


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def load_from_pretrained(
    path_or_repo: str | Path, device: str | torch.device = "cpu"
) -> CodelorColor:
    """Build a :class:`CodelorColor` and load ``model.safetensors`` from it.

    Args:
        path_or_repo: local directory containing ``config.json`` and
            ``model.safetensors``, or a Hugging Face repository id (requires the
            ``hf`` extra: ``pip install "codelor-color[hf]"``).
        device: device the returned model is moved to.

    Returns:
        The model in eval mode on ``device``.
    """
    p = Path(path_or_repo)
    if not p.exists():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Downloading from the Hugging Face Hub requires the 'hf' extra: "
                'pip install "codelor-color[hf]"'
            ) from exc
        p = Path(snapshot_download(str(path_or_repo)))

    config_path = p / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found in {p}")
    cfg = json.loads(config_path.read_text())

    hyper = cfg.get("hyperparameters", cfg) if isinstance(cfg, dict) else None
    if not isinstance(hyper, dict):
        raise ValueError(f"Unexpected config.json structure in {config_path}")
    unknown = sorted(set(hyper) - set(HYPERPARAM_KEYS))
    if unknown:
        logger.warning("Ignoring unknown hyper-parameters in %s: %s", config_path, unknown)
    params = {k: hyper[k] for k in HYPERPARAM_KEYS if k in hyper}
    logger.info("Building CodelorColor from %s with %s", config_path, params)
    model = CodelorColor(**params)

    weights_path = p / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"model.safetensors not found in {p}")
    from safetensors.torch import load_file

    state = load_file(str(weights_path))
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def _as_batch(image_tensor: Tensor, name: str = "image_tensor") -> tuple[Tensor, bool]:
    """Validate an image tensor and return it with a batch dimension plus a squeeze flag."""
    if not isinstance(image_tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(image_tensor).__name__}")
    if image_tensor.ndim not in (3, 4):
        raise ValueError(
            f"{name} must have shape (3, H, W) or (B, 3, H, W), got {tuple(image_tensor.shape)}"
        )
    if image_tensor.shape[-3] != 3:
        raise ValueError(
            f"{name} must have 3 RGB channels, got shape {tuple(image_tensor.shape)}"
        )
    if not image_tensor.is_floating_point():
        raise TypeError(
            f"{name} must be a float tensor in [0, 1], got dtype {image_tensor.dtype}"
        )
    squeezed = image_tensor.ndim == 3
    return (image_tensor.unsqueeze(0) if squeezed else image_tensor), squeezed


def _sync_model_device(model: nn.Module, device: torch.device) -> None:
    """Move ``model`` to ``device`` if its parameters live elsewhere."""
    try:
        current = next(model.parameters()).device
    except StopIteration:
        return
    if current != device:
        logger.info("Moving model from %s to %s", current, device)
        model.to(device)


def predict_mask(
    model: nn.Module,
    image_tensor: Tensor,
    color_rgb: Sequence[float] | None = None,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
) -> tuple[Any, Any]:
    """Compute a foreground mask for an image.

    Args:
        model: a :class:`CodelorColor` (switched to eval mode).
        image_tensor: ``(3, H, W)`` or ``(B, 3, H, W)`` float tensor in ``[0, 1]``.
        color_rgb: target color as 3 values in ``[0, 1]``; defaults to :data:`DEFAULT_REF_RGB`.
        threshold: probability threshold in ``[0, 1]``.
        device: device used for the forward pass; the model is moved there if needed.

    Returns:
        ``(mask, prob)`` where ``prob`` is a numpy array of shape ``(H, W)`` for a
        single image (or ``(B, H, W)`` for a batch) and ``mask`` is ``prob > threshold``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    img, squeezed = _as_batch(image_tensor)
    if color_rgb is None:
        color_rgb = DEFAULT_REF_RGB
    if len(color_rgb) != 3:
        raise ValueError(f"color_rgb must contain 3 values, got {len(color_rgb)}")

    device = torch.device(device)
    img = img.to(device)
    _sync_model_device(model, device)
    model.eval()

    ref = torch.tensor(list(color_rgb), dtype=torch.float32, device=device).view(1, 3)
    if ref.shape[0] != img.shape[0]:
        ref = ref.expand(img.shape[0], 3)
    with torch.inference_mode():
        logits = model(img, ref)
        prob_t = torch.sigmoid(logits)
    prob = prob_t[:, 0].cpu().numpy()
    if squeezed:
        prob = prob[0]
    return prob > threshold, prob


def remove_green(
    image_tensor: Tensor,
    model: nn.Module,
    color_rgb: Sequence[float] | None = None,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Return an RGBA tensor where the alpha channel is ``1 - P(foreground)``.

    Args:
        image_tensor: ``(3, H, W)`` or ``(B, 3, H, W)`` float tensor in ``[0, 1]``.
        model: a :class:`CodelorColor`.
        color_rgb: target color, defaults to :data:`DEFAULT_REF_RGB`.
        threshold: kept for API compatibility; the alpha channel uses the soft probability.
        device: device used for the forward pass.

    Returns:
        ``(4, H, W)`` or ``(B, 4, H, W)`` float tensor on CPU, with the same dtype
        as the input image.
    """
    _, prob = predict_mask(model, image_tensor, color_rgb, threshold, device)
    alpha = torch.from_numpy((1.0 - prob.astype("float32")).clip(0, 1))

    img = image_tensor.detach()
    img_cpu = img.cpu()
    if img_cpu.ndim == 3:
        alpha = alpha.unsqueeze(0)
    else:
        alpha = alpha.unsqueeze(1)
    return torch.cat([img_cpu, alpha.to(img_cpu.dtype)], dim=0 if img_cpu.ndim == 3 else 1)
