import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json

# ============================================================
# ARCHITECTURE CODELOR COLOR
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden * 2, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)

class LayerScale(nn.Module):
    def __init__(self, dim, init=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init)
    def forward(self, x):
        return x * self.gamma

class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep
        return x * mask / keep

def rope_2d(q, k, h, w, base=100.0):
    B, Hh, N, D = q.shape
    d4 = D // 4
    freqs = 1.0 / (base ** (torch.arange(0, d4, device=q.device).float() / d4))
    xs = torch.arange(w, device=q.device).float()
    ys = torch.arange(h, device=q.device).float()
    ax = torch.outer(xs, freqs)[None, :, :].expand(h, w, d4).reshape(h * w, d4)
    ay = torch.outer(ys, freqs)[:, None, :].expand(h, w, d4).reshape(h * w, d4)
    ang = torch.cat([ax, ay], dim=-1)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    def rot(x):
        x1, x2 = x[..., :D//2], x[..., D//2:]
        return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
    return rot(q), rot(k)

class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = RMSNorm(self.dh)
        self.k_norm = RMSNorm(self.dh)
        self.proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x, h, w):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope_2d(q, k, h, w)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))

class ColorCrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dh = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.q_norm = RMSNorm(self.dh)
        self.k_norm = RMSNorm(self.dh)
        self.proj = nn.Linear(dim, dim, bias=False)
    def forward(self, spatial, cond):
        B, N, C = spatial.shape
        M = cond.shape[1]
        q = self.q(spatial).reshape(B, N, self.heads, self.dh).transpose(1, 2)
        kv = self.kv(cond).reshape(B, M, 2, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=3.0, drop_path=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.ls1 = LayerScale(dim)
        self.dp1 = DropPath(drop_path)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)
        self.dp2 = DropPath(drop_path)
    def forward(self, x, h, w):
        x = x + self.dp1(self.ls1(self.attn(self.norm1(x), h, w)))
        return x + self.dp2(self.ls2(self.ffn(self.norm2(x))))

class AntiAliasDownsample(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.blur = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False)
        kernel = torch.tensor([1., 2., 1.])
        kernel = kernel / kernel.sum()
        bk = kernel[:, None] * kernel[None, :]
        with torch.no_grad():
            self.blur.weight.copy_(bk.expand(c_in, 1, 3, 3))
        self.blur.weight.requires_grad_(False)
        self.conv = nn.Conv2d(c_in, c_out, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(self.blur(x))))

class ConvNeXtStem(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True))
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 32, 7, padding=3, groups=32, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32))
        self.down8 = AntiAliasDownsample(32, 64)
        self.down16 = AntiAliasDownsample(64, dim)
    def forward(self, x):
        x = self.patch(x)
        x = x + self.block1(x)
        f4 = x
        f8 = self.down8(f4)
        f16 = self.down16(f8)
        return f4, f8, f16

class SobelEdge(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).float().view(1,1,3,3)
        sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).float().view(1,1,3,3)
        self.register_buffer('sx', sx)
        self.register_buffer('sy', sy)
    def forward(self, gray):
        ex = F.conv2d(gray, self.sx, padding=1)
        ey = F.conv2d(gray, self.sy, padding=1)
        return (ex**2 + ey**2 + 1e-6).sqrt()

class GatedDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj16 = nn.Conv2d(dim, dim, 1, bias=False)
        self.aux_head = nn.Conv2d(dim, 1, 1)
        self.up8 = nn.Sequential(
            nn.Conv2d(dim+64, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.SiLU(inplace=True))
        self.up4 = nn.Sequential(
            nn.Conv2d(dim+32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True))
        self.up1 = nn.Sequential(
            nn.Conv2d(64+3+1+1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True))
        self.head = nn.Conv2d(32, 1, 1)
        self.gate8 = nn.Parameter(torch.zeros(1))
        self.gate4 = nn.Parameter(torch.zeros(1))
        self.edge = SobelEdge()
    def forward(self, latent, f4, f8, img, ref_rgb):
        B, _, H, W = img.shape
        x = self.proj16(latent)
        aux = self.aux_head(x)
        aux = F.interpolate(aux, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=f8.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up8(torch.cat([x, torch.sigmoid(self.gate8) * f8], 1))
        x = F.interpolate(x, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up4(torch.cat([x, torch.sigmoid(self.gate4) * f4], 1))
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        ref_map = ref_rgb.view(B, 3, 1, 1).expand(B, 3, H, W)
        cos = F.cosine_similarity(img, ref_map, dim=1, eps=1e-6).unsqueeze(1)
        gray = img.mean(1, keepdim=True)
        edge = self.edge(gray)
        x = self.up1(torch.cat([x, img, cos, edge], 1))
        return self.head(x), aux

class CodelorColor(nn.Module):
    def __init__(self, dim=128, depth=8, heads=8, tied_after=4, mlp_ratio=3.0, max_tokens=1024, drop_path_rate=0.1):
        super().__init__()
        self.dim, self.depth, self.heads = dim, depth, heads
        self.tied_after, self.mlp_ratio = tied_after, mlp_ratio
        self.max_tokens = max_tokens
        self.stem = ConvNeXtStem(dim)
        self.color_mlp = nn.Sequential(
            nn.Linear(3, dim), nn.SiLU(inplace=True), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        n_unique = depth if tied_after <= 0 else min(depth, tied_after)
        dp = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            Block(dim, heads, mlp_ratio, drop_path=dp[i] if i < len(dp) else 0.0) 
            for i in range(n_unique)])
        self.color_xattn = nn.ModuleList([
            ColorCrossAttention(dim, heads) for _ in range(n_unique)])
        self.norm_out = RMSNorm(dim)
        self.decoder = GatedDecoder(dim)

    def _get(self, i):
        if self.tied_after > 0 and i >= self.tied_after:
            return self.blocks[i - self.tied_after], self.color_xattn[i - self.tied_after]
        return self.blocks[i], self.color_xattn[i]

    def forward(self, img, ref_rgb, return_aux=False):
        B = img.shape[0]
        f4, f8, f16 = self.stem(img)
        _, _, h, w = f16.shape
        N = h * w
        tokens = f16.flatten(2).transpose(1, 2)
        if N <= self.pos.shape[1]:
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

    def config(self):
        return dict(dim=self.dim, depth=self.depth, heads=self.heads, 
                    tied_after=self.tied_after, mlp_ratio=self.mlp_ratio, 
                    max_tokens=self.max_tokens)

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def load_from_pretrained(path_or_repo, device="cpu"):
    p = Path(path_or_repo)
    if not p.exists():
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(path_or_repo))
    cfg = json.loads((p / "config.json").read_text())
    model = CodelorColor(**cfg["hyperparameters"])
    from safetensors.torch import load_file
    model.load_state_dict(load_file(p / "model.safetensors"))
    return model.eval().to(device)

def predict_mask(model, image_tensor, color_rgb=None, threshold=0.5, device="cpu"):
    if color_rgb is None:
        color_rgb = (0.20, 0.85, 0.30)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    ref = torch.tensor(color_rgb, dtype=torch.float32, device=device).view(1, 3)
    with torch.no_grad():
        prob = torch.sigmoid(model(image_tensor, ref))[0, 0].cpu().numpy()
    return prob > threshold, prob

def remove_green(image_tensor, model, color_rgb=None, threshold=0.5, device="cpu"):
    import numpy as np
    _, prob = predict_mask(model, image_tensor, color_rgb, threshold, device)
    alpha = torch.from_numpy((1.0 - prob.astype("float32")).clip(0, 1)).unsqueeze(0)
    return torch.cat([image_tensor.cpu(), alpha], dim=0)
