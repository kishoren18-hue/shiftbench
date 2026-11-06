"""An alarm cutoff that keeps its false-alarm promise after conditions change.

Same rule as SteadyHand's AdaptiveCutoff (online quantile tracking, as in
adaptive conformal inference, Gibbs & Candes 2021, and conformal PID control,
Angelopoulos, Candes & Tibshirani 2023), applied to episode alarms:

    episode score  s = min(max_t s_t, s_max)
    alarm          if s > tau
    after a *successful* episode:
        tau <- max(tau + eta * (m - alpha), floor),   m = 1 if it alarmed

Failed episodes do not move tau: the promise is about successful episodes.

Guarantee. If tau starts at tau_1 <= s_max and floor <= tau_1, then after T
successful episodes, on *any* sequence of episodes (lighting change, camera
bump, slow drift, adversarial order),

    (#false alarms) / T  <=  alpha + (s_max + eta - tau_1) / (eta * T).

Proof as in SteadyHand: tau only rises after a false alarm, which needs
s > tau, so tau < s_max before the rise and tau < s_max + eta after it, and
tau never exceeds s_max + eta. Summing the update gives
eta * sum (m - alpha) <= tau_{T+1} - tau_1 <= s_max + eta - tau_1.

Labels. The update needs to know whether an alarmed episode would have
succeeded. If the watchdog *stops* the arm, nobody knows. Three options:

  "shadow"        record with the watchdog logging only, never stopping;
                  every episode gets a true label. Use this to collect data.
  "human"         an operator marks each stop as needed or not needed.
  "conservative"  count every alarm on a stopped episode as a false alarm.
                  The bound then covers all alarms, which is stricter: tau
                  rises faster and the detector catches fewer failures.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AdaptiveAlarm:
    alpha: float = 0.05
    start: float = 1.0          # initial cutoff, e.g. the conformal cutoff from nominal data
    eta: float = 0.1            # step size in score units; ~ the spread of nominal scores
    s_max: float = 10.0         # scores are capped here; set well above nominal scores
    floor: float = -np.inf     # lowest tau allowed; must be <= start
    tau: float = field(default=None, init=False)
    tau_start: float = field(default=None, init=False)
    n_success: int = field(default=0, init=False)
    n_false_alarms: int = field(default=0, init=False)

    def __post_init__(self):
        if not np.isfinite(self.start):
            self.start = self.s_max
        self.tau = self.tau_start = float(min(self.start, self.s_max))
        self.floor = min(self.floor, self.tau_start)

    def clip(self, s: float) -> float:
        return float(min(s, self.s_max))

    def alarms(self, score: float) -> bool:
        return self.clip(score) > self.tau

    def update(self, alarmed: bool, success: bool | None) -> float:
        """Feed back one finished episode. success=None means unknown (skipped)."""
        if success is None or not success:
            return self.tau
        m = float(alarmed)
        self.tau = max(self.tau + self.eta * (m - self.alpha), self.floor)
        self.n_success += 1
        self.n_false_alarms += int(m)
        return self.tau

    def bound(self, T: int | None = None) -> float:
        """Guaranteed upper bound on the false-alarm rate after T successful episodes."""
        T = self.n_success if T is None else T
        return self.alpha + (self.s_max + self.eta - self.tau_start) / (self.eta * max(T, 1))


def to_log(s):
    """Log scale for scores: a shift that multiplies errors by c moves every
    score by the same log(c), so one step size fits all shift sizes better."""
    return np.log(np.maximum(np.asarray(s, float), 1e-6))


def adaptive_from_calibration(cal_scores, start, alpha, eta_mult=0.5, cap_mult=20.0, log=False):
    """AdaptiveAlarm with step size eta = eta_mult * IQR of nominal successful
    calibration scores and cap s_max = max + cap_mult * IQR. With log=True the
    same is done on log scores (feed it to_log(score) later)."""
    s = to_log(cal_scores) if log else np.asarray(cal_scores, float)
    st = float(to_log(start)) if (log and np.isfinite(start)) else start
    iqr = float(np.subtract(*np.percentile(s, [75, 25]))) or float(np.std(s)) or 0.1
    st = st if np.isfinite(st) else float(np.max(s))
    return AdaptiveAlarm(alpha=alpha, start=st, eta=eta_mult * iqr,
                         s_max=float(np.max(s) + cap_mult * iqr), floor=float(np.min(s)))


def run_adaptive(scores, success, alpha, start, eta, s_max, label_mode="shadow", stop_on_alarm=True,
                 floor=-np.inf):
    """Replay a stream of episodes through the adaptive cutoff.

    scores:  (n,) episode scores in time order
    success: (n,) bool, the true outcome had the arm not been stopped
    label_mode: see module docstring
    Returns (alarmed (n,), tau_trace (n,)) where tau_trace[k] is the cutoff
    used for episode k.
    """
    ad = AdaptiveAlarm(alpha=alpha, start=start, eta=eta, s_max=s_max, floor=floor)
    scores = np.asarray(scores, float)
    success = np.asarray(success, bool)
    alarmed = np.zeros(len(scores), bool)
    trace = np.empty(len(scores))
    for k, (s, y) in enumerate(zip(scores, success)):
        trace[k] = ad.tau
        a = ad.alarms(s)
        alarmed[k] = a
        if label_mode == "shadow" or not (a and stop_on_alarm):
            ad.update(a, bool(y))
        elif label_mode == "human":
            ad.update(a, bool(y))       # the operator judged correctly
        elif label_mode == "conservative":
            ad.update(True, True)       # every stop counted as a false alarm
        else:
            raise ValueError(label_mode)
    return alarmed, trace
