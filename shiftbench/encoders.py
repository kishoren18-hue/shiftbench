"""Frozen image encoders: camera frames in, one latent vector per frame out.

VJEPA2Encoder is the real one. It loads Meta's open V-JEPA 2 from the Hugging
Face Hub (default facebook/vjepa2-vitl-fpc64-256, ViT-L, 300M parameters, MIT
license) and never trains it. Each frame is encoded on its own (transformers
duplicates a single frame to fill V-JEPA 2's 2-frame tubelet, as V-JEPA 2-AC
does). The 16x16 grid of patch tokens is pooled to

    pool="mean"   one 1024-d vector (cheap; loses where things are)
    pool="grid2"  a 2x2 grid of means, 4096-d (keeps coarse position)

Run it on a GPU (Colab T4 is fine). On a laptop CPU expect roughly a frame or
two per second, which is why the live watchdog runs the detector at a few Hz.

TinyEncoder is a small fixed random conv net for tests and for checking the
pipeline end to end without downloading anything. Do not report results with it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_float_chw(frames) -> torch.Tensor:
    """Accept (T,H,W,3) uint8 / (T,3,H,W) float in [0,1] / a single frame.
    Returns (T,3,H,W) float32 in [0,1]."""
    x = torch.as_tensor(np.asarray(frames) if not torch.is_tensor(frames) else frames)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[-1] == 3 and x.shape[1] != 3:
        x = x.permute(0, 3, 1, 2)
    x = x.float()
    if x.max() > 1.5:
        x = x / 255.0
    return x.contiguous()


def resize_center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    """Resize the shorter side to `size`, then center-crop to size x size."""
    _, _, h, w = x.shape
    s = size / min(h, w)
    nh, nw = max(size, round(h * s)), max(size, round(w * s))
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    top, left = (nh - size) // 2, (nw - size) // 2
    return x[:, :, top:top + size, left:left + size]


class VJEPA2Encoder:
    def __init__(self, model_id: str = "facebook/vjepa2-vitl-fpc64-256", pool: str = "mean",
                 device: str | None = None, dtype: torch.dtype | None = None, batch_size: int = 32):
        from transformers import AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device).eval()
        cfg = self.model.config
        self.size = int(getattr(cfg, "crop_size", 256))
        self.patch = int(cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0])
        self.hidden = int(cfg.hidden_size)
        self.pool = pool
        self.batch_size = batch_size
        self.model_id = model_id

    @property
    def dim(self) -> int:
        return self.hidden * (4 if self.pool == "grid2" else 1)

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            return tokens.mean(1)
        g = self.size // self.patch
        b, n, d = tokens.shape
        t = tokens.reshape(b, n // (g * g), g, g, d).mean(1)          # average over time slots
        t = t.reshape(b, 2, g // 2, 2, g // 2, d).mean((2, 4))        # 2x2 grid
        return t.reshape(b, 4 * d)

    @torch.no_grad()
    def encode(self, frames) -> np.ndarray:
        x = to_float_chw(frames)
        out = []
        for i in range(0, len(x), self.batch_size):
            b = resize_center_crop(x[i:i + self.batch_size], self.size)
            b = ((b - IMAGENET_MEAN) / IMAGENET_STD).to(self.device, self.dtype)
            tokens = self.model.get_vision_features(pixel_values_videos=b.unsqueeze(1))  # (B,1,C,H,W)
            out.append(self._pool(tokens.float()).cpu())
        return torch.cat(out).numpy().astype(np.float32)


class TinyEncoder(nn.Module):
    """Fixed random features. Deterministic given `seed`. For tests only."""

    def __init__(self, dim: int = 64, size: int = 64, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.size, self._dim = size, dim
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.proj = nn.Linear(32 * 4 * 4, dim)
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * (1.0 / np.sqrt(max(1, p[0].numel()))))
        self.eval()

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode(self, frames) -> np.ndarray:
        x = resize_center_crop(to_float_chw(frames), self.size)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        h = F.adaptive_avg_pool2d(torch.tanh(self.conv2(torch.tanh(self.conv1(x)))), 4)
        return self.proj(h.flatten(1)).numpy().astype(np.float32)


def make_encoder(name: str = "vjepa2", **kw):
    if name == "vjepa2":
        return VJEPA2Encoder(**kw)
    if name == "tiny":
        return TinyEncoder(**kw)
    raise ValueError(f"unknown encoder {name!r}")
