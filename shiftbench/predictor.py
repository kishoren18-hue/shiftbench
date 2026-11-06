"""The small trained part: an action-conditioned latent predictor.

Given the last `context` latent frames, the arm's joint state and the action
the policy just commanded, predict the latent `horizon` steps ahead:

    z_hat_{t+h} = z_t + f([z_{t-c+1..t}, state_t, action_t])

The residual form means "nothing changes" is the default guess, so the
network spends its capacity on how actions move the scene. It is trained on
*successful* episodes only (L1 loss, as in V-JEPA 2-AC), so failures and
unfamiliar scenes show up as large prediction errors.

V-JEPA 2-AC itself is a 300M-parameter block-causal transformer. This is a
few-million-parameter MLP over pooled latents: small enough to train on
Colab in minutes from ~50 demonstrations. If it turns out too weak, the
architecture is the first thing to upgrade; the rest of the pipeline does
not change.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

from .data import Episode


@dataclass
class PredictorConfig:
    latent_dim: int
    state_dim: int
    action_dim: int
    context: int = 2
    horizon: int = 1
    hidden: int = 512
    layers: int = 3
    dropout: float = 0.1


class LatentPredictor(nn.Module):
    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.latent_dim * cfg.context + cfg.state_dim + cfg.action_dim
        mods, d = [], d_in
        for _ in range(cfg.layers):
            mods += [nn.Linear(d, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout)]
            d = cfg.hidden
        mods.append(nn.Linear(d, cfg.latent_dim))
        self.net = nn.Sequential(*mods)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, zc, state, action):
        """zc: (B, context, D) oldest first; state: (B, S); action: (B, A)."""
        x = torch.cat([zc.flatten(1), state, action], dim=1)
        return zc[:, -1] + self.net(x)


@dataclass
class Normalizer:
    """Per-dimension standardization fitted on training episodes."""
    z_mu: np.ndarray
    z_sd: np.ndarray
    s_mu: np.ndarray
    s_sd: np.ndarray
    a_mu: np.ndarray
    a_sd: np.ndarray

    @classmethod
    def fit(cls, episodes: list[Episode]):
        Z = np.concatenate([e.z for e in episodes])
        S = np.concatenate([e.state for e in episodes])
        A = np.concatenate([e.action for e in episodes])
        f = lambda x: (x.mean(0), x.std(0) + 1e-6)
        return cls(*f(Z), *f(S), *f(A))

    def apply(self, e: Episode):
        return ((e.z - self.z_mu) / self.z_sd, (e.state - self.s_mu) / self.s_sd,
                (e.action - self.a_mu) / self.a_sd)


def make_windows(z, s, a, context: int, horizon: int):
    """All (context window, state, action, target) tuples in one episode."""
    T = len(z)
    idx = np.arange(context - 1, T - horizon)
    if len(idx) == 0:
        return None
    zc = np.stack([z[i - context + 1:i + 1] for i in idx])
    return zc, s[idx], a[idx], z[idx + horizon], idx


class WorldModel:
    """Predictor + normalizer, with train / score / save / load."""

    def __init__(self, cfg: PredictorConfig, device: str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = LatentPredictor(cfg).to(self.device)
        self.norm: Normalizer | None = None

    @classmethod
    def for_episodes(cls, episodes: list[Episode], **kw):
        e = episodes[0]
        return cls(PredictorConfig(e.z.shape[1], e.state.shape[1], e.action.shape[1], **kw))

    def _batch(self, episodes):
        parts = [make_windows(*self.norm.apply(e), self.cfg.context, self.cfg.horizon) for e in episodes]
        parts = [p for p in parts if p is not None]
        return [torch.as_tensor(np.concatenate([p[i] for p in parts]), dtype=torch.float32) for i in range(4)]

    def fit(self, train: list[Episode], val: list[Episode] | None = None, epochs: int = 60,
            lr: float = 3e-4, batch_size: int = 256, weight_decay: float = 1e-4, seed: int = 0,
            verbose: bool = True) -> dict:
        train = [e for e in train if e.success]
        if not train:
            raise ValueError("need at least one successful training episode")
        torch.manual_seed(seed)
        self.norm = Normalizer.fit(train)
        zc, s, a, y = self._batch(train)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        hist = {"train": [], "val": []}
        n = len(y)
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, batch_size):
                b = perm[i:i + batch_size]
                pred = self.net(zc[b].to(self.device), s[b].to(self.device), a[b].to(self.device))
                loss = (pred - y[b].to(self.device)).abs().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            sched.step()
            hist["train"].append(tot / n)
            if val:
                hist["val"].append(float(np.mean(np.concatenate([self.errors(e) for e in val if e.success] or [np.zeros(1)]))))
            if verbose and (ep % max(1, epochs // 6) == 0 or ep == epochs - 1):
                v = f"  val {hist['val'][-1]:.4f}" if val else ""
                print(f"epoch {ep:3d}  train L1 {hist['train'][-1]:.4f}{v}")
        return hist

    @torch.no_grad()
    def errors(self, e: Episode) -> np.ndarray:
        """Per-step surprise for one episode, aligned to the predicted frame.
        Length T - context - horizon + 1 (empty if the episode is too short)."""
        self.net.eval()
        w = make_windows(*self.norm.apply(e), self.cfg.context, self.cfg.horizon)
        if w is None:
            return np.zeros(0)
        zc, s, a, y, _ = (torch.as_tensor(x, dtype=torch.float32) for x in w)
        pred = self.net(zc.to(self.device), s.to(self.device), a.to(self.device)).cpu()
        return (pred - y).abs().mean(1).numpy()

    @torch.no_grad()
    def step_error(self, z_hist: np.ndarray, state: np.ndarray, action: np.ndarray, z_next: np.ndarray) -> float:
        """Online version: one prediction error from raw (unnormalized) inputs."""
        n = self.norm
        zc = torch.as_tensor(((z_hist - n.z_mu) / n.z_sd)[None], dtype=torch.float32, device=self.device)
        s = torch.as_tensor(((state - n.s_mu) / n.s_sd)[None], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(((action - n.a_mu) / n.a_sd)[None], dtype=torch.float32, device=self.device)
        y = (z_next - n.z_mu) / n.z_sd
        self.net.eval()
        return float(np.abs(self.net(zc, s, a).cpu().numpy()[0] - y).mean())

    def save(self, path: str):
        torch.save({"cfg": asdict(self.cfg), "state_dict": self.net.state_dict(),
                    "norm": {k: v for k, v in vars(self.norm).items()}}, path)

    @classmethod
    def load(cls, path: str, device: str | None = None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        wm = cls(PredictorConfig(**ck["cfg"]), device=device)
        wm.net.load_state_dict(ck["state_dict"])
        wm.norm = Normalizer(**ck["norm"])
        return wm
