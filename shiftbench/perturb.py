"""Image-level stand-ins for the physical shifts, applied to real robot video.

These let the H1/H2 test run on real footage before the arm arrives. They are
*stand-ins*: a brightness change in pixels is not the same as moving a lamp
(no new shadows, no specular highlights), and an affine warp is not the same
as moving a camera (no parallax). The physical ShiftBench data is still the
real test. Every shift is deterministic, so results are reproducible.

`onset_frac` makes a shift start partway through an episode (a mid-run
camera bump), which is what the replay demo uses.
"""
from __future__ import annotations

import numpy as np

SHIFTS = {
    "nominal": {},
    "light_dim_1": {"gain": 0.65},
    "light_dim_2": {"gain": 0.40},
    "light_warm": {"rgb": (1.12, 1.0, 0.78)},
    "cam_bump_1": {"dx": 0.04, "dy": 0.02, "rot": 2.0, "zoom": 1.03},
    "cam_bump_2": {"dx": 0.10, "dy": 0.05, "rot": 6.0, "zoom": 1.08},
    "occlusion": {"box": (0.70, 0.05, 0.22, 0.30)},          # x, y, w, h as fractions: a "distractor" in a corner
    "blur": {"sigma": 2.5},
}


def apply(frames: np.ndarray, name: str, onset_frac: float | None = None) -> np.ndarray:
    """frames: (T,H,W,3) uint8 RGB. Returns a new array."""
    import cv2

    p = SHIFTS[name]
    out = frames.copy()
    T, H, W, _ = frames.shape
    t0 = 0 if onset_frac is None else int(round(onset_frac * T))
    sl = slice(t0, T)
    x = out[sl].astype(np.float32)
    if "gain" in p:
        x = 255.0 * (x / 255.0) ** (1.0 / p["gain"] ** 0.5) * p["gain"] ** 0.5     # darker, gamma-ish
    if "rgb" in p:
        x = x * np.array(p["rgb"], np.float32)
    if "dx" in p:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), p["rot"], p["zoom"])
        M[0, 2] += p["dx"] * W
        M[1, 2] += p["dy"] * H
        x = np.stack([cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT) for f in x])
    if "box" in p:
        bx, by, bw, bh = p["box"]
        x0, y0, x1, y1 = int(bx * W), int(by * H), int((bx + bw) * W), int((by + bh) * H)
        rng = np.random.default_rng(0)
        tex = rng.normal(90, 25, (y1 - y0, x1 - x0, 3)).astype(np.float32)        # same "object" every frame
        tex[..., 2] += 40
        x[:, y0:y1, x0:x1] = tex
    if "sigma" in p:
        x = np.stack([cv2.GaussianBlur(f, (0, 0), p["sigma"]) for f in x])
    out[sl] = np.clip(x, 0, 255).astype(np.uint8)
    return out
