"""A toy pick-and-place world in latent space, for testing the pipeline.

THIS IS NOT DATA. It exists so every script can be run and checked before the
arm arrives, and so the benchmark has known answers to be tested against. The
numbers it produces must never be reported as results.

Hidden scene: 5 joint angles q, object position o (2-d), grasp flag g.
The scripted policy reaches, grasps, carries and releases. In a failure the
grasp slips at a random moment and the object stops following the gripper.

The "camera" renders the scene into a D-dim latent:

    z = tanh(W @ [q, o, g] + bg) + noise

Shifts change the rendering, not the task:
    camera:    W -> W + delta * W2           (view changes how the scene maps to pixels)
    lighting:  gain on W, offset, more noise
    table:     background vector bg changes
    objects:   object columns of W change, and grasps fail more often
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Episode

SHIFTS = ("nominal", "camera", "lighting", "table", "objects")


@dataclass
class World:
    D: int = 48
    T: int = 40
    p_fail: float = 0.2
    noise: float = 0.05
    seed: int = 0

    def __post_init__(self):
        r = np.random.default_rng(self.seed)
        self.W = r.normal(0, 0.6, (self.D, 8))
        self.W[:, 5:7] *= 2.5                          # the object is the salient thing in view
        self.W2 = r.normal(0, 0.6, (self.D, 8))
        self.bg = r.normal(0, 0.3, self.D)
        self.bg2 = r.normal(0, 0.3, self.D)
        self.light_dir = r.normal(0, 1, self.D) / np.sqrt(self.D)

    def render_params(self, shift: str, level: int):
        W, bg, gain, off, noise, p_fail = self.W.copy(), self.bg.copy(), 1.0, 0.0, self.noise, self.p_fail
        if shift == "camera":
            W = W + 0.05 * level * self.W2
        elif shift == "lighting":
            gain, off, noise = 1 - 0.05 * level, 0.25 * level, self.noise * (1 + 0.25 * level)
        elif shift == "table":
            bg = bg + 0.35 * level * self.bg2
        elif shift == "objects":
            W[:, 5:7] = W[:, 5:7] + 0.12 * level * self.W2[:, 5:7]
            p_fail = min(0.6, self.p_fail + 0.1 * level)
        elif shift != "nominal":
            raise ValueError(shift)
        return W, bg, gain, off, noise, p_fail

    def episode(self, shift="nominal", level=0, rng=None, index=-1) -> Episode:
        r = rng or np.random.default_rng()
        W, bg, gain, off, noise, p_fail = self.render_params(shift, level)
        fail = r.random() < p_fail
        onset = int(r.integers(16, 39)) if fail else None    # late slips leave little time
        slip = r.uniform(0.005, 0.03)                   # drift speed after the slip
        lag = r.uniform(0.0, 0.6)                       # partial grip: object still half-follows
        q = np.zeros(5)
        o = np.array([0.6, -0.4]) + r.normal(0, 0.05, 2)
        goal = np.array([-0.5, 0.5]) + r.normal(0, 0.05, 2)
        g = 0.0
        Z, S, A = [], [], []
        for t in range(self.T):
            # scripted policy: joints 0-1 steer the gripper in the plane, 4 = gripper
            ee = q[:2]
            target = o if t < 12 else goal
            a = q.copy()
            a[:2] = ee + 0.35 * (target - ee) + r.normal(0, 0.01, 2)
            a[2:4] = 0.3 * np.sin(t / 6 + np.arange(2))
            a[4] = 1.0 if 10 <= t < 32 else 0.0
            h = np.r_[q, o, 0.0]                         # grasp itself is not visible, only where the object is
            z = np.tanh(gain * (W @ h) + bg + off * self.light_dir) + r.normal(0, noise, self.D)
            Z.append(z), S.append(q + r.normal(0, 0.005, 5)), A.append(a)
            # dynamics
            q = q + 0.6 * (a - q)
            if t >= 10 and np.linalg.norm(q[:2] - o) < 0.25 and a[4] > 0.5:
                g = 1.0
            if fail and t >= onset:
                g = 0.0
            if g > 0.5 and a[4] > 0.5:
                o = q[:2].copy()
            elif fail and t >= onset:
                o = o + lag * (q[:2] - o) + np.array([0.0, -slip])   # slipping: lags and drifts
        success = not fail
        return Episode(z=np.array(Z, np.float32), state=np.array(S, np.float32), action=np.array(A, np.float32),
                       success=bool(success), condition=shift, level=level, episode_index=index,
                       fail_step=onset, fps=4.0)

    def dataset(self, n_per: dict, seed=1) -> list[Episode]:
        """n_per: {(shift, level): n_episodes}"""
        r = np.random.default_rng(seed)
        out, k = [], 0
        for (shift, level), n in n_per.items():
            for _ in range(n):
                out.append(self.episode(shift, level, r, k))
                k += 1
        return out
