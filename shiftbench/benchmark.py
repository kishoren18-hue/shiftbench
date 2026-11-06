"""The ShiftBench experiment: does the alarm keep its promise after a shift?

Inputs: a trained Detector calibrated on nominal episodes, and episodes
recorded under each condition, in the order they were recorded.

Methods, all at the same target false-alarm rate alpha:

  fixed        conformal cutoff from nominal calibration, never changed
               (what published detectors do)
  self_ref     same, but with the self-referenced score (each episode's
               error relative to its own first steps); its own nominal cutoff
  recal_k      rerun the arm k times in the new conditions, recalibrate on the
               successful ones; never alarms if fewer than 1/alpha - 1 of the
               k succeeded
  adaptive_*   starts at the fixed cutoff and updates after every successful
               episode (shadow-mode labels), with a stated finite-sample bound;
               one row per step-size / scale variant

Every method is scored on the same evaluation episodes, the ones after the
first k, so recal_k's calibration episodes are never used to test it.
Adaptive methods also learn from the first k, and `false_alarm_full_stream`
charges them for their mistakes there.

Order matters for recal_k and adaptive. `false_alarm` uses the recorded order;
`*_mean_orders` / `*_p90_orders` average over `n_orders` random reorderings of
each condition, which assumes episodes within a condition are exchangeable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .adaptive import run_adaptive, to_log
from .calibrate import conformal_cutoff
from .metrics import auroc, wilson

DEFAULT_ADAPTIVE = (("adaptive", 0.5, False), ("adaptive_log", 1.0, True), ("adaptive_log_fast", 3.0, True))


def _rates(scores, success, tau):
    a = scores > tau
    ns, nf = int(success.sum()), int((~success).sum())
    fa, d = int((a & success).sum()), int((a & ~success).sum())
    return (fa / ns if ns else np.nan), (d / nf if nf else np.nan), fa, ns


def _first_recovery(alarmed, success, alpha, win=10):
    """Index (in successful episodes) after which the rolling false-alarm rate
    over `win` successes first drops to <= 2*alpha. NaN if never."""
    fa = alarmed[success].astype(float)
    if len(fa) < win:
        return np.nan
    roll = np.convolve(fa, np.ones(win) / win, mode="valid")
    ok = np.flatnonzero(roll <= 2 * alpha)
    return float(ok[0] + win) if len(ok) else np.nan


def evaluate(det, episodes_by_condition: dict, nominal_cal, k: int = 20, n_orders: int = 200,
             seed: int = 0, adaptive_variants=DEFAULT_ADAPTIVE) -> pd.DataFrame:
    """Score episodes with a Detector, then run `evaluate_scores`."""
    ref = det.cfg.ref_steps or 4
    cal_raw = np.array([det.episode_score(e, ref_steps=None) for e in nominal_cal if e.success])
    cal_ref = np.array([det.episode_score(e, ref_steps=ref) for e in nominal_cal if e.success])
    conds = {}
    for cond, eps in episodes_by_condition.items():
        conds[cond] = (np.array([det.episode_score(e, ref_steps=None) for e in eps]),
                       np.array([det.episode_score(e, ref_steps=ref) for e in eps]),
                       np.array([e.success for e in eps]))
    return evaluate_scores(conds, cal_raw, cal_ref, det.cfg.alpha, k=k, n_orders=n_orders, seed=seed,
                           adaptive_variants=adaptive_variants)


def evaluate_scores(conditions: dict, cal_raw, cal_ref, alpha: float, k: int = 20, n_orders: int = 200,
                    seed: int = 0, adaptive_variants=DEFAULT_ADAPTIVE) -> pd.DataFrame:
    """The benchmark on precomputed episode scores.

    conditions: {name: (raw_scores, self_ref_scores, success)} in recording order
    cal_raw / cal_ref: episode scores of the nominal successful calibration episodes
    """
    from .adaptive import adaptive_from_calibration

    rng = np.random.default_rng(seed)
    cal_raw = np.asarray(cal_raw, float)
    tau_fixed = conformal_cutoff(cal_raw, alpha)
    tau_ref = conformal_cutoff(np.asarray(cal_ref, float), alpha)
    ads = {name: (adaptive_from_calibration(cal_raw, tau_fixed, alpha, eta_mult=m, log=lg), lg)
           for name, m, lg in adaptive_variants}

    rows = []
    for cond, (s_raw, s_ref, y) in conditions.items():
        s_raw, s_ref, y = np.asarray(s_raw, float), np.asarray(s_ref, float), np.asarray(y, bool)
        n = len(y)
        if n <= k:
            raise ValueError(f"{cond}: need more than k={k} episodes, have {n}")
        base = dict(condition=cond, n_eval=n - k, n_eval_success=int(y[k:].sum()),
                    n_eval_fail=int((~y[k:]).sum()),
                    auroc_raw=auroc(s_raw[k:], ~y[k:]), auroc_ref=auroc(s_ref[k:], ~y[k:]),
                    median_score_success=float(np.median(s_raw[y])) if y.any() else np.nan,
                    tau_fixed=tau_fixed)

        def add(method, far, detr, fa=None, ns=None, **extra):
            lo, hi = wilson(fa, ns) if ns else (np.nan, np.nan)
            rows.append({**base, "method": method, "false_alarm": far, "fa_lo": lo, "fa_hi": hi,
                         "detection": detr, **extra})

        far, detr, fa, ns = _rates(s_raw[k:], y[k:], tau_fixed)
        add("fixed", far, detr, fa, ns)
        far, detr, fa, ns = _rates(s_ref[k:], y[k:], tau_ref)
        add("self_ref", far, detr, fa, ns)

        def recal(order):
            sr, yr = s_raw[order], y[order]
            t = conformal_cutoff(sr[:k][yr[:k]], alpha)
            f, d, *_ = _rates(sr[k:], yr[k:], t)
            return f, d, float(np.isinf(t))

        def adapt(order, ad, lg):
            sr, yr = s_raw[order], y[order]
            sc = to_log(sr) if lg else sr
            alarmed, _ = run_adaptive(sc, yr, ad.alpha, ad.tau_start, ad.eta, ad.s_max, floor=ad.floor)
            a_e, y_e = alarmed[k:], yr[k:]
            f = (a_e & y_e).sum() / max(y_e.sum(), 1)
            d = (a_e & ~y_e).sum() / (~y_e).sum() if (~y_e).any() else np.nan
            full = (alarmed & yr).sum() / max(yr.sum(), 1)
            return f, d, full, _first_recovery(alarmed, yr, alpha)

        orders = [np.arange(n)] + [rng.permutation(n) for _ in range(n_orders)]
        R = np.array([recal(o) for o in orders])
        add("recal_k", R[0, 0], R[0, 1], false_alarm_mean_orders=np.nanmean(R[1:, 0]),
            false_alarm_p90_orders=np.nanpercentile(R[1:, 0], 90), detection_mean_orders=np.nanmean(R[1:, 1]),
            never_alarms_share=np.mean(R[1:, 2]))
        for name, (ad, lg) in ads.items():
            A = np.array([adapt(o, ad, lg) for o in orders], float)
            T = max(int(y.sum()), 1)
            add(name, A[0, 0], A[0, 1], false_alarm_mean_orders=np.nanmean(A[1:, 0]),
                false_alarm_p90_orders=np.nanpercentile(A[1:, 0], 90), detection_mean_orders=np.nanmean(A[1:, 1]),
                false_alarm_full_stream=A[0, 2],
                bound_full_stream=ad.alpha + (ad.s_max + ad.eta - ad.tau_start) / (ad.eta * T),
                successes_to_recover=A[0, 3])
    return pd.DataFrame(rows)


def drift_stream(det, nominal, shifted, eta_mult=1.0, log=True):
    """Nominal episodes followed by shifted ones, for the recovery plot."""
    eps = list(nominal) + list(shifted)
    s = np.array([det.episode_score(e, ref_steps=None) for e in eps])
    y = np.array([e.success for e in eps])
    ad = det.default_adaptive(eta_mult=eta_mult, log=log)
    sc = to_log(s) if log else s
    alarmed, trace = run_adaptive(sc, y, ad.alpha, ad.tau_start, ad.eta, ad.s_max, floor=ad.floor)
    tau_adapt = np.exp(trace) if log else trace
    return pd.DataFrame({"k": np.arange(len(eps)), "phase": ["nominal"] * len(nominal) + ["shifted"] * len(shifted),
                         "score": s, "success": y, "tau_fixed": det.tau, "tau_adaptive": tau_adapt,
                         "alarm_fixed": s > det.tau, "alarm_adaptive": alarmed})
