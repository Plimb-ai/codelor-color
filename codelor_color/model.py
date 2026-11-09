"""Codelor Color: lightweight Transformer architecture for chroma-key segmentation.

The model takes an image ``img`` of shape ``(B, 3, H, W)`` and a target color
``ref_rgb`` of shape ``(B, 3)`` and returns a per-pixel logit map ``(B, 1, H, W)``.
Apply ``torch.sigmoid`` to obtain foreground probabilities.

Two tasks share the same backbone (see :data:`TASKS`):

* ``"color"`` -- :class:`CodelorColor`, removes a *target color* (chroma key).
* ``"background"`` -- :class:`CodelorBackground`, removes the *background*
  whatever its color (no reference color needed).

Both predict the probability of each pixel being *removed*, so the same
inference (:func:`predict_mask`) and training code applies to both.

Architecture presets (see :data:`PRESETS`):

* ``"v2"`` (default) -- the original architecture, bit-compatible with every
  published checkpoint (``plimb/codelor-color-1.2``...).
* ``"v3"`` -- FiLM color conditioning, RoPE-only positions (any resolution),
  GroupNorm (stable with the tiny batches used on phones) and an extra
  chroma-distance guide channel. Needs new weights (train or fine-tune).
* ``"v4"`` -- ``"v3"`` with BatchNorm instead of GroupNorm (more accurate in
  the ablation; recommended unless training with very small batches).
"""

from __future__ import annotations

import copy
import functools
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
from torch.utils.checkpoint import checkpoint

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

#: Architecture options added in 2.3 (all default to the original "v2" behaviour).
ARCH_KEYS: tuple[str, ...] = (
    "cond_mode", "pos_mode", "pos_grid", "norm", "chroma_guide", "head"
)

#: Default value of every :data:`ARCH_KEYS` option (= original architecture).
ARCH_DEFAULTS: dict[str, Any] = {
    "cond_mode": "xattn",
    "pos_mode": "learned",
    "pos_grid": None,
    "norm": "bn",
    "chroma_guide": False,
    "head": "conv",
}

#: Named architecture presets accepted by :func:`build_model`.
PRESETS: dict[str, dict[str, Any]] = {
    "v2": dict(ARCH_DEFAULTS),
    "v3": {
        "cond_mode": "film",
        "pos_mode": "none",
        "pos_grid": None,
        "norm": "gn",
        "chroma_guide": True,
        "head": "light",
    },
    # v3 without GroupNorm: the ablation (see README) showed GroupNorm was the
    # component costing accuracy; FiLM + RoPE-only + light head are kept.
    "v4": {
        "cond_mode": "film",
        "pos_mode": "none",
        "pos_grid": None,
        "norm": "bn",
        "chroma_guide": True,
        "head": "light",
    },
}

#: Model families: ``"color"`` removes a target color, ``"background"`` removes the background.
TASKS: tuple[str, ...] = ("color", "background")
#: Task assumed when a ``config.json`` does not specify one (all pre-2.3 checkpoints).
DEFAULT_TASK: str = "color"
#: Number of learned conditioning tokens used by :class:`CodelorBackground`.
BG_COND_TOKENS: int = 4

_COND_MODES = ("xattn", "film")
_POS_MODES = ("learned", "interp", "none")
_NORMS = ("bn", "gn")
_HEADS = ("conv", "light", "quarter")


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


def make_norm2d(channels: int, kind: str = "bn") -> nn.Module:
    """``BatchNorm2d`` (``"bn"``, original) or ``GroupNorm`` (``"gn"``, batch-size independent)."""
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "gn":
        groups = 8 if channels % 8 == 0 else 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"norm must be one of {_NORMS}, got {kind!r}")


@functools.lru_cache(maxsize=64)
def _rope_tables(
    h: int, w: int, d4: int, base: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Cached ``cos``/``sin`` tables of :func:`rope_2d` (same math, computed once per grid)."""
    with torch.inference_mode(False), torch.no_grad():
        freqs = 1.0 / (base ** (torch.arange(0, d4, device=device).float() / d4))
        xs = torch.arange(w, device=device).float()
        ys = torch.arange(h, device=device).float()
        ax = torch.outer(xs, freqs)[None, :, :].expand(h, w, d4).reshape(h * w, d4)
        ay = torch.outer(ys, freqs)[:, None, :].expand(h, w, d4).reshape(h * w, d4)
        ang = torch.cat([ax, ay], dim=-1)
        return ang.cos()[None, None], ang.sin()[None, None]


def rope_2d(q: Tensor, k: Tensor, h: int, w: int, base: float = ROPE_BASE) -> tuple[Tensor, Tensor]:
    """Apply 2D rotary position embeddings, splitting the head dim in half for x and y."""
    B, Hh, N, D = q.shape
    cos, sin = _rope_tables(h, w, D // 4, float(base), q.device)

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

    With a single conditioning token the softmax is identically 1, so the output
    is ``proj(v)`` for every query: that case skips the query path entirely
    (same result, much cheaper).
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
        if M == 1:
            v = self.kv(cond)[..., C:]  # (B, 1, C): softmax over one key == 1
            return cast(Tensor, self.proj(v)).expand(B, N, C)
        q = self.q(spatial).reshape(B, N, self.heads, self.dh).transpose(1, 2)
        kv = self.kv(cond).reshape(B, M, 2, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class FiLM(nn.Module):
    """Feature-wise modulation ``x * (1 + scale) + shift`` from a conditioning vector.

    Zero-initialized, so a freshly built block starts as the identity. Used by
    the ``"v3"`` preset instead of :class:`ColorCrossAttention`.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.to_ss = nn.Linear(dim, dim * 2)
        nn.init.zeros_(self.to_ss.weight)
        nn.init.zeros_(self.to_ss.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.to_ss(cond).unsqueeze(1).chunk(2, dim=-1)
        return x + self.norm(x) * scale + shift


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
    """BlurPool-style downsampling: frozen binomial blur, then stride-2 conv + norm + SiLU."""

    def __init__(self, c_in: int, c_out: int, norm: str = "bn") -> None:
        super().__init__()
        self.blur = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False)
        kernel = torch.tensor([1.0, 2.0, 1.0])
        kernel = kernel / kernel.sum()
        bk = kernel[:, None] * kernel[None, :]
        with torch.no_grad():
            self.blur.weight.copy_(bk.expand(c_in, 1, 3, 3))
        self.blur.weight.requires_grad_(False)
        self.conv = nn.Conv2d(c_in, c_out, 1, stride=2, bias=False)
        self.bn = make_norm2d(c_out, norm)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(self.blur(x))))


class ConvNeXtStem(nn.Module):
    """Patch-embed stem producing the feature pyramid ``f4``, ``f8``, ``f16``.

    Channel widths come from :data:`STEM_W1` / :data:`STEM_W2`, which are shared
    with :class:`GatedDecoder`; change them together or the decoder will not load.
    """

    def __init__(self, dim: int, norm: str = "bn") -> None:
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(3, STEM_W1, kernel_size=4, stride=4, bias=False),
            make_norm2d(STEM_W1, norm),
            nn.SiLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(STEM_W1, STEM_W1, 7, padding=3, groups=STEM_W1, bias=False),
            make_norm2d(STEM_W1, norm),
            nn.SiLU(inplace=True),
            nn.Conv2d(STEM_W1, STEM_W1, 1, bias=False),
            make_norm2d(STEM_W1, norm),
        )
        self.down8 = AntiAliasDownsample(STEM_W1, STEM_W2, norm)
        self.down16 = AntiAliasDownsample(STEM_W2, dim, norm)

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


def color_cosine(img: Tensor, ref_rgb: Tensor) -> Tensor:
    """Per-pixel cosine similarity to the key color, ``(B, 1, H, W)``.

    Same result as ``F.cosine_similarity(img, ref_map, dim=1, eps=EPS)`` on an
    expanded ``ref_map`` (each norm clamped to ``EPS``), but computed with
    plain reductions: about 10x cheaper on CPU at full resolution.
    """
    ref = ref_rgb.view(-1, 3, 1, 1)
    dot = (img * ref).sum(1, keepdim=True)
    img_n = img.pow(2).sum(1, keepdim=True).sqrt().clamp_min(EPS)
    ref_n = ref.pow(2).sum(1, keepdim=True).sqrt().clamp_min(EPS)
    return dot / (img_n * ref_n)


def chroma_distance(img: Tensor, ref_rgb: Tensor) -> Tensor:
    """Distance in the CbCr plane (BT.601) between each pixel and the key color.

    Unlike the RGB cosine similarity it ignores brightness, so shadows and
    highlights on a green screen stay close to the key color. Returns
    ``(B, 1, H, W)``, roughly in ``[0, 1]``.
    """
    def cbcr(x: Tensor) -> tuple[Tensor, Tensor]:
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        return (-0.168736 * r - 0.331264 * g + 0.5 * b), (0.5 * r - 0.418688 * g - 0.081312 * b)

    cb, cr = cbcr(img)
    rb, rr = cbcr(ref_rgb[:, :, None, None])
    return ((cb - rb) ** 2 + (cr - rr) ** 2 + EPS).sqrt().unsqueeze(1)


class GatedDecoder(nn.Module):
    """FPN-like decoder with learnable sigmoid gates, deep supervision and an edge head.

    At full resolution it concatenates ``[decoded, raw image, cosine-similarity to the
    target color, Sobel edge map]`` before the prediction head.

    With ``color_guided=False`` (background removal) the cosine-similarity channel
    is dropped since there is no target color. ``chroma_guide=True`` (preset
    ``"v3"``) adds a :func:`chroma_distance` channel.

    ``head="light"`` (preset ``"v3"``) runs that last stage at half resolution and
    adds a tiny full-resolution refinement (8 channels) driven by the same
    guides. The full-resolution stage is memory-bound and is the most expensive
    part of the original model on CPU; this removes most of its cost while
    keeping pixel-accurate edges from the guides.
    """

    def __init__(
        self, dim: int, color_guided: bool = True, norm: str = "bn",
        chroma_guide: bool = False, head: str = "conv",
    ) -> None:
        super().__init__()
        if head not in _HEADS:
            raise ValueError(f"head must be one of {_HEADS}, got {head!r}")
        self.color_guided = color_guided
        self.chroma_guide = chroma_guide and color_guided
        self.light = head != "conv"
        #: Resolution divisor of the last decoder stage (1 = full resolution).
        self.head_stride = {"conv": 1, "light": 2, "quarter": 4}[head]
        # image + edge (+ cosine) (+ chroma distance)
        n_guides = 3 + 1 + (1 if color_guided else 0) + (1 if self.chroma_guide else 0)
        self.proj16 = nn.Conv2d(dim, dim, 1, bias=False)
        self.aux_head = nn.Conv2d(dim, 1, 1)
        self.up8 = nn.Sequential(
            nn.Conv2d(dim + STEM_W2, dim, 3, padding=1, bias=False),
            make_norm2d(dim, norm),
            nn.SiLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(dim + STEM_W1, STEM_W2, 3, padding=1, bias=False),
            make_norm2d(STEM_W2, norm),
            nn.SiLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(STEM_W2 + n_guides, STEM_W1, 3, padding=1, bias=False),
            make_norm2d(STEM_W1, norm),
            nn.SiLU(inplace=True),
            nn.Conv2d(STEM_W1, STEM_W1, 3, padding=1, bias=False),
            make_norm2d(STEM_W1, norm),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Conv2d(STEM_W1, 1, 1)
        self.refine: nn.Sequential | None = None
        if self.light:
            last = nn.Conv2d(8, 1, 1)
            nn.init.zeros_(last.weight)  # starts as identity
            nn.init.zeros_(cast(Tensor, last.bias))
            self.refine = nn.Sequential(
                nn.Conv2d(1 + n_guides, 8, 3, padding=1), nn.SiLU(inplace=True), last
            )
        self.gate8 = nn.Parameter(torch.zeros(1))
        self.gate4 = nn.Parameter(torch.zeros(1))
        self.edge = SobelEdge()

    def _guides(self, img: Tensor, ref_rgb: Tensor | None) -> list[Tensor]:
        edge = self.edge(img.mean(1, keepdim=True))
        if not self.color_guided:
            return [img, edge]
        if ref_rgb is None:
            raise ValueError("a color-guided decoder needs ref_rgb")
        guides = [img, color_cosine(img, ref_rgb), edge]
        if self.chroma_guide:
            guides.append(chroma_distance(img, ref_rgb))
        return guides

    def forward(
        self, latent: Tensor, f4: Tensor, f8: Tensor, img: Tensor, ref_rgb: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        H, W = img.shape[-2:]
        x = self.proj16(latent)
        aux = self.aux_head(x)
        aux = F.interpolate(aux, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=f8.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up8(torch.cat([x, torch.sigmoid(self.gate8) * f8], 1))
        x = F.interpolate(x, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up4(torch.cat([x, torch.sigmoid(self.gate4) * f4], 1))
        if self.refine is None:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
            x = self.up1(torch.cat([x, *self._guides(img, ref_rgb)], 1))
            return self.head(x), aux
        low = F.avg_pool2d(img, self.head_stride, ceil_mode=True)
        if x.shape[-2:] != low.shape[-2:]:
            x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, *self._guides(low, ref_rgb)], 1))
        logit = F.interpolate(self.head(x), size=(H, W), mode="bilinear", align_corners=False)
        logit = logit + self.refine(torch.cat([logit, *self._guides(img, ref_rgb)], 1))
        return logit, aux


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
        cond_mode: ``"xattn"`` (original color cross-attention) or ``"film"``.
        pos_mode: ``"learned"`` (original: flat learned table), ``"interp"`` (the
            same table read as a ``pos_grid`` 2D grid and resized to the input
            grid, so other resolutions/aspect ratios get correct positions) or
            ``"none"`` (RoPE only, resolution independent).
        pos_grid: ``(rows, cols)`` token grid the learned table was trained on,
            used by ``pos_mode="interp"`` (e.g. ``(10, 10)`` for 160x160 training).
        norm: ``"bn"`` (BatchNorm, original) or ``"gn"`` (GroupNorm).
        chroma_guide: add a CbCr chroma-distance channel to the decoder.
        head: ``"conv"`` (original full-resolution head) or ``"light"`` (head at
            half resolution + small full-resolution refinement, much cheaper).

    The defaults reproduce the original architecture exactly (preset ``"v2"``).
    The model returns ``(B, 1, H, W)`` logits, or ``((B, 1, H, W), (B, 1, h4, w4))``
    when ``return_aux=True``.
    """

    #: Task identifier written to ``config.json`` (see :data:`TASKS`).
    task: str = "color"
    #: Whether :meth:`forward` requires a ``ref_rgb`` target color.
    needs_color: bool = True

    def __init__(
        self,
        dim: int = 128,
        depth: int = 8,
        heads: int = 8,
        tied_after: int = 4,
        mlp_ratio: float = 3.0,
        max_tokens: int = 1024,
        drop_path_rate: float = 0.1,
        cond_mode: str = "xattn",
        pos_mode: str = "learned",
        pos_grid: Sequence[int] | None = None,
        norm: str = "bn",
        chroma_guide: bool = False,
        head: str = "conv",
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if cond_mode not in _COND_MODES:
            raise ValueError(f"cond_mode must be one of {_COND_MODES}, got {cond_mode!r}")
        if pos_mode not in _POS_MODES:
            raise ValueError(f"pos_mode must be one of {_POS_MODES}, got {pos_mode!r}")
        if norm not in _NORMS:
            raise ValueError(f"norm must be one of {_NORMS}, got {norm!r}")
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.tied_after = tied_after
        self.mlp_ratio = mlp_ratio
        self.max_tokens = max_tokens
        self.drop_path_rate = drop_path_rate
        self.cond_mode = cond_mode
        self.pos_mode = pos_mode
        self.pos_grid: tuple[int, int] | None = None
        self.norm = norm
        self.chroma_guide = bool(chroma_guide)
        self.head = head
        #: Recompute Transformer activations in backward to save RAM (training only).
        #: Not saved in ``config()``; enable it with :meth:`set_grad_checkpointing`.
        self.grad_checkpointing = False
        self._pos_warned = False
        self.stem = ConvNeXtStem(dim, norm)
        self._build_conditioning(dim)
        self.pos: nn.Parameter | None
        if pos_mode == "none":
            self.pos = None
        else:
            self.pos = nn.Parameter(torch.zeros(1, max_tokens, dim))
            nn.init.trunc_normal_(self.pos, std=POS_EMB_STD)
        if pos_grid is not None:
            self.set_pos_grid(pos_grid)
        n_unique = depth if tied_after <= 0 else min(depth, tied_after)
        dp = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            [Block(dim, heads, mlp_ratio, drop_path=dp[i]) for i in range(n_unique)]
        )
        self.color_xattn: nn.ModuleList | None
        self.color_film: nn.ModuleList | None
        if cond_mode == "xattn":
            self.color_xattn = nn.ModuleList(
                [ColorCrossAttention(dim, heads) for _ in range(n_unique)]
            )
            self.color_film = None
        else:
            self.color_xattn = None
            self.color_film = nn.ModuleList([FiLM(dim) for _ in range(n_unique)])
        self.norm_out = RMSNorm(dim)
        self.decoder = GatedDecoder(
            dim, color_guided=self.needs_color, norm=norm, chroma_guide=self.chroma_guide,
            head=head,
        )

    # -- conditioning hooks (overridden by CodelorBackground) -----------------

    def _build_conditioning(self, dim: int) -> None:
        """Create the modules producing the conditioning (tokens or FiLM vector)."""
        self.color_mlp = nn.Sequential(
            nn.Linear(3, dim), nn.SiLU(inplace=True), nn.Linear(dim, dim)
        )

    def _condition(self, ref_rgb: Tensor | None, batch: int, tokens: Tensor | None = None) -> Tensor:
        """Return ``(B, M, dim)`` cross-attention tokens (``"xattn"``) or a ``(B, dim)`` vector."""
        if ref_rgb is None:
            raise ValueError("CodelorColor needs ref_rgb of shape (B, 3)")
        c = cast(Tensor, self.color_mlp(ref_rgb))
        return c if self.cond_mode == "film" else c.unsqueeze(1)

    def _get(self, i: int) -> tuple[Block, nn.Module]:
        # cycle through the shared blocks; same as ``i - tied_after`` when depth <= 2 * tied_after
        idx = i % self.tied_after if self.tied_after > 0 and i >= self.tied_after else i
        cond = self.color_xattn if self.color_xattn is not None else self.color_film
        assert cond is not None
        return cast(Block, self.blocks[idx]), cast(nn.Module, cond[idx])

    # -- positions -------------------------------------------------------------

    def set_pos_grid(self, grid: Sequence[int] | None) -> CodelorColor:
        """Declare the ``(rows, cols)`` token grid the learned positions were trained on.

        With a grid set, ``pos_mode="learned"`` becomes ``"interp"``: the table is
        reshaped to that grid and resized to the input grid. At the training
        resolution the result is identical to the original behaviour; at other
        resolutions or aspect ratios each token gets the embedding of its real
        2D position instead of a wrong flat index (and no ``max_tokens`` limit).
        Example for a model trained at 160x160: ``model.set_pos_grid((10, 10))``.
        """
        if grid is None:
            self.pos_grid = None
            if self.pos_mode == "interp":
                self.pos_mode = "learned"
            return self
        if self.pos is None:
            raise ValueError("pos_mode='none' has no learned positions to interpolate")
        gh, gw = (int(g) for g in grid)
        if gh < 1 or gw < 1 or gh * gw > self.pos.shape[1]:
            raise ValueError(f"pos_grid {grid} does not fit max_tokens={self.pos.shape[1]}")
        self.pos_grid = (gh, gw)
        self.pos_mode = "interp"
        return self

    def _positions(self, h: int, w: int) -> Tensor | None:
        """Positional embeddings ``(1, h*w, dim)`` for an ``h x w`` grid, or ``None``."""
        if self.pos is None:
            return None
        N = h * w
        if self.pos_mode == "interp" and self.pos_grid is not None:
            gh, gw = self.pos_grid
            table = self.pos[:, : gh * gw]
            if (gh, gw) == (h, w):
                return table
            grid = table.reshape(1, gh, gw, -1).permute(0, 3, 1, 2)
            grid = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)
            return grid.flatten(2).transpose(1, 2)
        if N > self.pos.shape[1]:
            if not self._pos_warned:
                warnings.warn(
                    f"Input yields {N} spatial tokens but max_tokens={self.pos.shape[1]}; "
                    "positional embeddings are skipped for this and subsequent inputs. "
                    "Use a larger max_tokens, a smaller image, or model.set_pos_grid(...).",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._pos_warned = True
            return None
        return self.pos[:, :N]

    # -- forward ---------------------------------------------------------------

    def forward(
        self, img: Tensor, ref_rgb: Tensor | None = None, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Predict foreground logits.

        Args:
            img: ``(B, 3, H, W)`` float tensor, values in ``[0, 1]``.
            ref_rgb: ``(B, 3)`` target color in RGB ``[0, 1]`` (required for the
                ``"color"`` task, ignored by :class:`CodelorBackground`).
            return_aux: also return the 1/4-resolution deep-supervision logit.

        Returns:
            ``(B, 1, H, W)`` logits, or ``(main, aux)`` if ``return_aux``.
        """
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(f"img must have shape (B, 3, H, W), got {tuple(img.shape)}")
        if self.needs_color:
            if ref_rgb is None:
                raise ValueError("ref_rgb is required for the 'color' task, shape (B, 3)")
            if ref_rgb.ndim != 2 or ref_rgb.shape[1] != 3:
                raise ValueError(f"ref_rgb must have shape (B, 3), got {tuple(ref_rgb.shape)}")
            if ref_rgb.shape[0] != img.shape[0]:
                raise ValueError(
                    f"batch mismatch: img has {img.shape[0]} samples, "
                    f"ref_rgb has {ref_rgb.shape[0]}"
                )
        else:
            ref_rgb = None

        B = img.shape[0]
        f4, f8, f16 = self.stem(img)
        _, _, h, w = f16.shape
        tokens = f16.flatten(2).transpose(1, 2)
        pos = self._positions(h, w)
        if pos is not None:
            tokens = tokens + pos
        c_tok = self._condition(ref_rgb, B, tokens)
        x = tokens
        use_ckpt = self.grad_checkpointing and self.training and torch.is_grad_enabled()
        for i in range(self.depth):
            blk, cond = self._get(i)
            if use_ckpt:
                x = cast(Tensor, checkpoint(blk, x, h, w, use_reentrant=False))
            else:
                x = blk(x, h, w)
            if self.cond_mode == "film":
                x = cast(Tensor, checkpoint(cond, x, c_tok, use_reentrant=False)
                         if use_ckpt else cond(x, c_tok))
            else:
                x = x + cast(Tensor, checkpoint(cond, x, c_tok, use_reentrant=False)
                             if use_ckpt else cond(x, c_tok))
        x = self.norm_out(x)
        latent = x.transpose(1, 2).reshape(B, -1, h, w)
        main, aux = self.decoder(latent, f4, f8, img, ref_rgb)
        return (main, aux) if return_aux else main

    def set_grad_checkpointing(self, enabled: bool = True) -> CodelorColor:
        """Trade compute for memory during training (useful on phones / small GPUs)."""
        self.grad_checkpointing = enabled
        return self

    def config(self) -> dict[str, Any]:
        """Return the hyper-parameters needed to rebuild this model.

        The result is accepted both directly and nested under a
        ``"hyperparameters"`` key by :func:`load_from_pretrained`. Architecture
        options (:data:`ARCH_KEYS`) are only included when they differ from the
        original architecture, so ``v2`` configs stay readable by older code.
        """
        cfg: dict[str, Any] = dict(
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            tied_after=self.tied_after,
            mlp_ratio=self.mlp_ratio,
            max_tokens=self.max_tokens,
            drop_path_rate=self.drop_path_rate,
        )
        arch = {
            "cond_mode": self.cond_mode,
            "pos_mode": self.pos_mode,
            "pos_grid": list(self.pos_grid) if self.pos_grid else None,
            "norm": self.norm,
            "chroma_guide": self.chroma_guide,
            "head": self.head,
        }
        if arch["pos_mode"] == "interp":
            arch["pos_mode"] = "learned"  # rebuilt from pos_grid
        cfg.update({k: v for k, v in arch.items() if v != ARCH_DEFAULTS[k]})
        return cfg

    def extra_repr(self) -> str:
        return (
            f"task={self.task}, dim={self.dim}, depth={self.depth}, heads={self.heads}, "
            f"tied_after={self.tied_after}, mlp_ratio={self.mlp_ratio}, "
            f"max_tokens={self.max_tokens}, cond={self.cond_mode}, pos={self.pos_mode}, "
            f"norm={self.norm}, head={self.head}"
        )


class CodelorBackground(CodelorColor):
    """Background-removal model: same backbone as :class:`CodelorColor`, no target color.

    Instead of a color token, the cross-attention attends to :data:`BG_COND_TOKENS`
    learned "background prototype" tokens, and the decoder drops the
    color-similarity channel. With ``cond_mode="film"`` (preset ``"v3"``) the
    modulation comes from a global context vector pooled from the image itself.
    The output is the logit of each pixel being *background* (i.e. to be
    removed), exactly like the color model predicts the pixels of the key color,
    so :func:`predict_mask` / :func:`remove_background` and the training loop
    work unchanged.

    ``ref_rgb`` is accepted by :meth:`forward` for API compatibility but ignored.
    The constructor takes the same hyper-parameters as :class:`CodelorColor`.
    """

    task = "background"
    needs_color = False

    def _build_conditioning(self, dim: int) -> None:
        if self.cond_mode == "film":
            self.bg_context = nn.Sequential(
                RMSNorm(dim), nn.Linear(dim, dim), nn.SiLU(inplace=True), nn.Linear(dim, dim)
            )
        else:
            self.bg_tokens = nn.Parameter(torch.zeros(1, BG_COND_TOKENS, dim))
            nn.init.trunc_normal_(self.bg_tokens, std=POS_EMB_STD)

    def _condition(self, ref_rgb: Tensor | None, batch: int, tokens: Tensor | None = None) -> Tensor:
        if self.cond_mode == "film":
            assert tokens is not None
            return cast(Tensor, self.bg_context(tokens.mean(1)))
        return self.bg_tokens.expand(batch, -1, -1)

    @classmethod
    def from_color_model(cls, color_model: CodelorColor) -> CodelorBackground:
        """Build a background model initialized from a trained color model.

        The stem, Transformer blocks and most of the decoder are shared, so
        fine-tuning from a chroma-key checkpoint converges much faster than
        training from scratch (handy on low-power devices such as Termux).
        """
        model = cls(**color_model.config())
        transfer_weights(color_model, model)
        return model


#: Mapping ``task name -> model class``.
MODEL_CLASSES: dict[str, type[CodelorColor]] = {
    "color": CodelorColor,
    "background": CodelorBackground,
}


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def build_model(task: str = DEFAULT_TASK, preset: str | None = None, **hyperparams: Any) -> CodelorColor:
    """Instantiate the model for ``task`` (``"color"`` or ``"background"``).

    ``preset`` (``"v2"``, ``"v3"`` or ``"v4"``, see :data:`PRESETS`) fills the
    architecture options; explicit keyword arguments override it.

    Example::

        model = build_model("background", preset="v3", dim=128, depth=8)
    """
    if task not in MODEL_CLASSES:
        raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")
    params: dict[str, Any] = {}
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}, expected one of {tuple(PRESETS)}")
        params.update(PRESETS[preset])
    params.update(hyperparams)
    return MODEL_CLASSES[task](**params)


def transfer_weights(src: nn.Module, dst: nn.Module) -> list[str]:
    """Copy every parameter/buffer of ``src`` whose name and shape match into ``dst``.

    Returns the list of copied keys. Useful to initialize a
    :class:`CodelorBackground` from a :class:`CodelorColor` checkpoint, or a
    ``"v3"`` model from a ``"v2"`` one (Transformer blocks, convolutions...).
    """
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    copied = [
        k for k, v in src_state.items() if k in dst_state and dst_state[k].shape == v.shape
    ]
    dst.load_state_dict({**dst_state, **{k: src_state[k] for k in copied}})
    logger.info("Transferred %d/%d tensors", len(copied), len(dst_state))
    return copied


def save_pretrained(model: CodelorColor, directory: str | Path, **extra: Any) -> Path:
    """Write ``config.json`` + ``model.safetensors`` so :func:`load_from_pretrained` can reload.

    Extra keyword arguments are stored in ``config.json`` next to the
    hyper-parameters (e.g. training metadata). If ``safetensors`` is not
    installed (some Termux setups), the weights are written to ``model.pt``
    with :func:`torch.save` instead.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "architectures": [type(model).__name__],
        "model_type": "codelor-color",
        "codelor_task": model.task,
        "hyperparameters": model.config(),
        **extra,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    try:
        from safetensors.torch import save_file
    except ImportError:
        logger.warning("safetensors not installed: saving weights to model.pt instead")
        torch.save(state, out / "model.pt")
    else:
        save_file(state, str(out / "model.safetensors"))
    return out


def _detect_task(cfg: dict[str, Any], hyper: dict[str, Any]) -> str:
    """Find the Codelor task in a ``config.json``.

    Order: ``codelor_task`` -> ``architectures`` -> ``task`` (only if it is a
    Codelor task: Hub configs use ``"task": "image-segmentation"``) -> ``"color"``.
    """
    explicit = cfg.get("codelor_task", hyper.get("codelor_task"))
    if explicit is not None:
        if explicit not in MODEL_CLASSES:
            raise ValueError(f"unknown codelor_task {explicit!r}, expected one of {TASKS}")
        return str(explicit)
    by_class = {cls.__name__: task for task, cls in MODEL_CLASSES.items()}
    for arch in cfg.get("architectures") or []:
        if arch in by_class:
            return by_class[arch]
    for candidate in (cfg.get("task"), hyper.get("task")):
        if candidate in MODEL_CLASSES:
            return str(candidate)
    return DEFAULT_TASK


def load_from_pretrained(
    path_or_repo: str | Path, device: str | torch.device = "cpu"
) -> CodelorColor:
    """Build a :class:`CodelorColor` (or :class:`CodelorBackground`) and load its weights.

    The model class is picked from ``config.json`` (``codelor_task``,
    ``architectures``, then ``task``; ``"color"`` when absent, so every
    existing checkpoint such as ``plimb/codelor-color-1.2`` keeps loading).
    ``model.safetensors`` is preferred; ``model.pt`` (written by
    :func:`save_pretrained` without safetensors) is used as a fallback.

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
    accepted = set(HYPERPARAM_KEYS) | set(ARCH_KEYS) | {"task", "codelor_task"}
    if hyper is cfg:  # flat config: top-level metadata is not a hyper-parameter
        accepted |= {"architectures", "model_type", "format", "version", "labels",
                     "green_ref", "training"}
    unknown = sorted(set(hyper) - accepted)
    if unknown:
        logger.warning("Ignoring unknown hyper-parameters in %s: %s", config_path, unknown)
    params = {k: hyper[k] for k in (*HYPERPARAM_KEYS, *ARCH_KEYS) if k in hyper}
    task = _detect_task(cfg, hyper)
    logger.info("Building %s from %s with %s", MODEL_CLASSES[task].__name__, config_path, params)
    model = build_model(task, **params)

    weights_path = p / "model.safetensors"
    fallback_path = p / "model.pt"
    if weights_path.is_file():
        from safetensors.torch import load_file

        state = load_file(str(weights_path))
    elif fallback_path.is_file():
        state = torch.load(fallback_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"model.safetensors not found in {p}")
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


# ============================================================
# INFERENCE OPTIMIZATION
# ============================================================

def _fuse_conv_norm(conv: nn.Module, norm: nn.Module) -> nn.Module | None:
    if isinstance(conv, nn.Conv2d) and isinstance(norm, nn.BatchNorm2d):
        from torch.nn.utils.fusion import fuse_conv_bn_eval

        return fuse_conv_bn_eval(conv, norm)
    return None


def optimize_for_inference(
    model: CodelorColor, fuse_bn: bool = True, quantize: bool = False,
    channels_last: bool = False,
) -> CodelorColor:
    """Return an inference-only copy of ``model`` that runs faster on CPU.

    * ``fuse_bn``: folds every BatchNorm into the preceding convolution
      (mathematically identical, removes one pass over each feature map).
    * ``channels_last``: NHWC weights, often faster convolutions with oneDNN
      (x86); benchmark it on your device.
    * ``quantize``: int8 dynamic quantization of the ``nn.Linear`` layers
      (Transformer). Faster and smaller on CPU / ARM (Termux), with a small
      accuracy cost: check the result on your images.

    The original model is left untouched. The copy cannot be trained or saved
    with :func:`save_pretrained` (its ``state_dict`` layout differs).
    """
    opt = copy.deepcopy(model).eval()
    if fuse_bn:
        for module in list(opt.modules()):
            if isinstance(module, nn.Sequential):
                children = list(module.children())
                for i in range(len(children) - 1):
                    fused = _fuse_conv_norm(children[i], children[i + 1])
                    if fused is not None:
                        module[i] = fused
                        module[i + 1] = nn.Identity()
            elif isinstance(module, AntiAliasDownsample):
                fused = _fuse_conv_norm(module.conv, module.bn)
                if fused is not None:
                    module.conv = cast(nn.Conv2d, fused)
                    module.bn = nn.Identity()
    if quantize:
        try:
            engines = torch.backends.quantized.supported_engines
            if "qnnpack" in engines and "fbgemm" not in engines:
                torch.backends.quantized.engine = "qnnpack"  # ARM / Android
            quant = getattr(torch, "ao", torch).quantization
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # torch.ao.quantization deprecation notices
                opt = cast(CodelorColor,
                           quant.quantize_dynamic(opt, {nn.Linear}, dtype=torch.qint8))
        except (AttributeError, RuntimeError, ImportError) as exc:
            warnings.warn(f"int8 quantization unavailable in this torch build ({exc}); "
                          "returning the float model", RuntimeWarning, stacklevel=2)
    if channels_last:
        opt = opt.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    for p in opt.parameters():
        p.requires_grad_(False)
    return opt


# ============================================================
# INFERENCE HELPERS
# ============================================================

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


def _box(x: Tensor, r: int) -> Tensor:
    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


def guided_filter(guide: Tensor, src: Tensor, radius: int = 2, eps: float = 1e-3) -> Tensor:
    """Edge-preserving guided filter (He et al.): smooths ``src`` along ``guide``'s edges.

    ``guide`` and ``src`` are ``(B, 1, H, W)``. Applied to an alpha map with the
    image luminance as guide, it snaps soft or blocky mask borders to the real
    image edges. No parameters, works with any model (including ``codelor-color-1.2``).
    """
    mean_i, mean_p = _box(guide, radius), _box(src, radius)
    cov = _box(guide * src, radius) - mean_i * mean_p
    var = _box(guide * guide, radius) - mean_i * mean_i
    a = cov / (var + eps)
    b = mean_p - a * mean_i
    return _box(a, radius) * guide + _box(b, radius)


def guided_upsample(
    guide_hr: Tensor, src_lr: Tensor, radius: int = 2, eps: float = 1e-3
) -> Tensor:
    """Fast guided filter upsampling: low-res alpha -> full-res alpha that follows image edges.

    The linear coefficients are fitted at low resolution and applied to the
    full-resolution ``guide_hr`` (``(B, 1, H, W)``), so running the model at a
    small ``max_side`` still gives sharp full-resolution borders.
    """
    guide_lr = F.interpolate(guide_hr, size=src_lr.shape[-2:], mode="area")
    mean_i, mean_p = _box(guide_lr, radius), _box(src_lr, radius)
    cov = _box(guide_lr * src_lr, radius) - mean_i * mean_p
    var = _box(guide_lr * guide_lr, radius) - mean_i * mean_i
    a = cov / (var + eps)
    b = mean_p - a * mean_i
    size = guide_hr.shape[-2:]
    a = F.interpolate(_box(a, radius), size=size, mode="bilinear", align_corners=False)
    b = F.interpolate(_box(b, radius), size=size, mode="bilinear", align_corners=False)
    return a * guide_hr + b


#: Accepted values of the ``refine`` argument of :func:`predict_mask`.
REFINE_MODES: tuple[str, ...] = ("none", "guided")


def _fit_size(h: int, w: int, max_side: int) -> tuple[int, int]:
    """Scale ``(h, w)`` so the longest side is ``max_side``, rounded to multiples of 16."""
    scale = max_side / max(h, w)
    return (max(16, round(h * scale / 16) * 16), max(16, round(w * scale / 16) * 16))


def predict_mask(
    model: nn.Module,
    image_tensor: Tensor,
    color_rgb: Sequence[float] | None = None,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
    max_side: int | None = None,
    refine: str | None = None,
) -> tuple[Any, Any]:
    """Compute a foreground mask for an image.

    Args:
        model: a :class:`CodelorColor` (switched to eval mode).
        image_tensor: ``(3, H, W)`` or ``(B, 3, H, W)`` float tensor in ``[0, 1]``.
        color_rgb: target color as 3 values in ``[0, 1]``; defaults to :data:`DEFAULT_REF_RGB`.
        threshold: probability threshold in ``[0, 1]``.
        device: device used for the forward pass; the model is moved there if needed.
        max_side: if the image is larger, run the model on a downscaled copy
            (longest side = ``max_side``) and upsample the probability back.
            Much faster on big images and closer to the training resolution;
            ``None`` (default) runs at full resolution as before.
        refine: ``"guided"`` snaps the probability to the image edges with a
            guided filter (and uses guided upsampling with ``max_side``), guided
            by the chroma distance to the key (color models) or the luminance
            (background models). Small gain (measured +0.2 IoU point when
            upsampling 3x); ``None`` / ``"none"`` keeps the raw output (default).

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
    if max_side is not None and max_side < 16:
        raise ValueError(f"max_side must be >= 16, got {max_side}")
    refine = refine or "none"
    if refine not in REFINE_MODES:
        raise ValueError(f"refine must be one of {REFINE_MODES}, got {refine!r}")

    device = torch.device(device)
    img = img.to(device)
    _sync_model_device(model, device)
    model.eval()

    ref = torch.tensor(list(color_rgb), dtype=torch.float32, device=device).view(1, 3)
    if ref.shape[0] != img.shape[0]:
        ref = ref.expand(img.shape[0], 3)
    H, W = img.shape[-2:]
    with torch.inference_mode():
        x = img
        if max_side is not None and max(H, W) > max_side:
            x = F.interpolate(img, size=_fit_size(H, W, max_side), mode="bilinear",
                              align_corners=False, antialias=True)
        logits = model(x, ref)
        if refine == "guided":
            # color models: the chroma distance to the key has the sharpest edge
            # exactly where the key color ends; background models: luminance.
            if getattr(model, "needs_color", True):
                guide = (chroma_distance(img.float(), ref) * 3).clamp(0, 1)
            else:
                guide = img.float().mean(1, keepdim=True)
            prob_lr = torch.sigmoid(logits.float())
            if prob_lr.shape[-2:] != (H, W):
                prob_t = guided_upsample(guide, prob_lr)
            else:
                prob_t = guided_filter(guide, prob_lr)
            prob_t = prob_t.clamp(0, 1)
        else:
            if logits.shape[-2:] != (H, W):
                logits = F.interpolate(logits, size=(H, W), mode="bilinear",
                                       align_corners=False)
            prob_t = torch.sigmoid(logits)
    prob = prob_t[:, 0].float().cpu().numpy()
    if squeezed:
        prob = prob[0]
    return prob > threshold, prob


def despill(
    rgb: Tensor, alpha: Tensor, color_rgb: Sequence[float] = DEFAULT_REF_RGB,
    strength: float = 1.0,
) -> Tensor:
    """Remove the key-color cast (e.g. green fringes) left on the kept pixels.

    The key's dominant channel is limited to the mean of the two others, the
    classic chroma-key "spill suppression", weighted by ``strength`` and applied
    most on semi-transparent edges (``alpha`` close to 0.5) and less inside.

    Args:
        rgb: ``(3, H, W)`` or ``(B, 3, H, W)`` image.
        alpha: matching ``(H, W)`` / ``(B, H, W)`` or with a channel dim.
    """
    if rgb.shape[-3] != 3:
        raise ValueError("despill expects an RGB image")
    key = torch.tensor(list(color_rgb), dtype=rgb.dtype)
    c = int(key.argmax())
    others = torch.tensor([i for i in range(3) if i != c])
    a = alpha if alpha.ndim == rgb.ndim else alpha.unsqueeze(-3)
    limit = rgb.index_select(-3, others).mean(-3, keepdim=True)
    main = rgb.narrow(-3, c, 1)
    spill = (main - limit).clamp(min=0)
    weight = strength * (1.0 - (2 * a - 1).abs() * 0.5)
    out = rgb.clone()
    out.narrow(-3, c, 1).copy_(main - spill * weight)
    return out


def remove_green(
    image_tensor: Tensor,
    model: nn.Module,
    color_rgb: Sequence[float] | None = None,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
    max_side: int | None = None,
    despill_strength: float = 0.0,
    refine: str | None = None,
) -> Tensor:
    """Return an RGBA tensor where the alpha channel is ``1 - P(foreground)``.

    Args:
        image_tensor: ``(3, H, W)`` or ``(B, 3, H, W)`` float tensor in ``[0, 1]``.
        model: a :class:`CodelorColor`.
        color_rgb: target color, defaults to :data:`DEFAULT_REF_RGB`.
        threshold: kept for API compatibility; the alpha channel uses the soft probability.
        device: device used for the forward pass.
        max_side: see :func:`predict_mask`.
        despill_strength: ``> 0`` applies :func:`despill` to the RGB channels
            (only meaningful for color models; ``0`` = unchanged, as before).
        refine: see :func:`predict_mask` (``"guided"`` for edge-aligned alpha).

    Returns:
        ``(4, H, W)`` or ``(B, 4, H, W)`` float tensor on CPU, with the same dtype
        as the input image.
    """
    _, prob = predict_mask(model, image_tensor, color_rgb, threshold, device, max_side, refine)
    alpha = torch.from_numpy((1.0 - prob.astype("float32")).clip(0, 1))

    img = image_tensor.detach()
    img_cpu = img.cpu()
    if despill_strength > 0:
        img_cpu = despill(img_cpu, alpha.to(img_cpu.dtype),
                          color_rgb or DEFAULT_REF_RGB, despill_strength)
    if img_cpu.ndim == 3:
        alpha = alpha.unsqueeze(0)
    else:
        alpha = alpha.unsqueeze(1)
    return torch.cat([img_cpu, alpha.to(img_cpu.dtype)], dim=0 if img_cpu.ndim == 3 else 1)


def remove_background(
    image_tensor: Tensor,
    model: nn.Module,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
    color_rgb: Sequence[float] | None = None,
    max_side: int | None = None,
    refine: str | None = None,
) -> Tensor:
    """Return an RGBA tensor whose alpha is ``1 - P(removed)``.

    Works with both model families: with a :class:`CodelorBackground` the
    background is removed (``color_rgb`` is ignored), with a
    :class:`CodelorColor` the target color is removed (same as
    :func:`remove_green`).

    Returns:
        ``(4, H, W)`` or ``(B, 4, H, W)`` float tensor on CPU.
    """
    return remove_green(image_tensor, model, color_rgb, threshold, device, max_side,
                        refine=refine)
