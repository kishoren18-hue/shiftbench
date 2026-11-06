"""World model + alarm statistic + cutoff, packaged for offline and live use."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .adaptive import AdaptiveAlarm, adaptive_from_calibration
from .calibrate import conformal_cutoff
from .data import Episode
from .predictor import WorldModel
from .scores import score_trace, episode_score


@dataclass
class DetectorConfig:
    window: int = 4                 # smoothing window, in detector steps
    ref_steps: int | None = None    # self-referenced score if set
    alpha: float = 0.05             # target false-alarm rate per successful episode
    context: int = 2
    horizon: int = 1
    hidden: int = 512
    layers: int = 3
    epochs: int = 60


class Detector:
    def __init__(self, cfg: DetectorConfig, wm: WorldModel | None = None):
        self.cfg = cfg
        self.wm = wm
        self.tau = float("inf")
        self.cal_scores: np.ndarray | None = None

    # ---------------------------------------------------------- offline
    def fit(self, train: list[Episode], cal: list[Episode], verbose=True, seed=0):
        c = self.cfg
        self.wm = WorldModel.for_episodes(train, context=c.context, horizon=c.horizon,
                                          hidden=c.hidden, layers=c.layers)
        self.wm.fit(train, epochs=c.epochs, verbose=verbose, seed=seed)
        self.calibrate(cal)
        return self

    def calibrate(self, cal: list[Episode]):
        s = np.array([self.episode_score(e) for e in cal if e.success])
        self.cal_scores = s
        self.tau = conformal_cutoff(s, self.cfg.alpha)
        return self.tau

    def trace(self, e: Episode, ref_steps: int | None | str = "cfg") -> np.ndarray:
        r = self.cfg.ref_steps if ref_steps == "cfg" else ref_steps
        return score_trace(self.wm.errors(e), self.cfg.window, r)

    def episode_score(self, e: Episode, ref_steps="cfg") -> float:
        return episode_score(self.trace(e, ref_steps))

    def default_adaptive(self, eta_mult: float = 0.5, log: bool = False, alpha: float | None = None) -> AdaptiveAlarm:
        """Adaptive cutoff started at the conformal cutoff, with step size and
        cap set from the spread of nominal calibration scores. With log=True,
        feed it adaptive.to_log(score)."""
        return adaptive_from_calibration(self.cal_scores, self.tau, alpha or self.cfg.alpha,
                                         eta_mult=eta_mult, log=log)

    # ---------------------------------------------------------- persistence
    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.wm.save(out / "world_model.pt")
        (out / "detector.json").write_text(json.dumps({
            "cfg": asdict(self.cfg), "tau": self.tau,
            "cal_scores": None if self.cal_scores is None else self.cal_scores.tolist()}, indent=2))

    @classmethod
    def load(cls, out_dir, device=None):
        out = Path(out_dir)
        meta = json.loads((out / "detector.json").read_text())
        d = cls(DetectorConfig(**meta["cfg"]), WorldModel.load(out / "world_model.pt", device=device))
        d.tau = float(meta["tau"])
        d.cal_scores = None if meta["cal_scores"] is None else np.array(meta["cal_scores"])
        return d


class OnlineScorer:
    """Feeds one detector tick at a time; used by the live watchdog.

    Offline, step t pairs the frame and joint state at t with the action the
    policy commanded during [t, t+1) (see data.encode_lerobot). Live, that
    action is only known at the next tick. So at tick t call

        step(z_t, state_t, action_prev)

    where action_prev is the latest command sent since tick t-1. It is
    filed under step t-1, and the error for target frame t uses exactly the
    same inputs as offline. Returns the alarm statistic, or None until there
    is enough history.
    """

    def __init__(self, det: Detector):
        self.det = det
        self.reset()

    def reset(self):
        self.z: list = []
        self.s: list = []
        self.a: list = []
        self.errs: list[float] = []
        self.trace: list[float] = []

    def step(self, z, state, action_prev=None) -> float | None:
        c = self.det.cfg
        if self.z and action_prev is not None:
            self.a.append(np.asarray(action_prev, np.float32))     # belongs to step t-1
        elif self.z:
            self.a.append(self.a[-1] if self.a else np.zeros(self.det.wm.cfg.action_dim, np.float32))
        self.z.append(np.asarray(z, np.float32))
        self.s.append(np.asarray(state, np.float32))
        t, h = len(self.z) - 1, c.horizon
        i = t - h                                   # last context index
        if i - c.context + 1 < 0:
            return None
        e = self.det.wm.step_error(np.stack(self.z[i - c.context + 1:i + 1]), self.s[i], self.a[i], self.z[t])
        self.errs.append(e)
        errs = np.asarray(self.errs)
        if c.ref_steps:
            if len(errs) < c.ref_steps:
                self.trace.append(float("-inf"))
                return None
            errs = errs - np.median(errs[: c.ref_steps])
        w = errs[-c.window:]
        self.trace.append(float(w.mean()))
        return self.trace[-1]
