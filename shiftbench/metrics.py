"""What a detector gets right and wrong, with honest error bars."""
from __future__ import annotations

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def auroc(scores, failed) -> float:
    """Probability that a random failed episode scores above a random success
    (ties count half). NaN if either class is missing."""
    s = np.asarray(scores, float)
    f = np.asarray(failed, bool)
    pos, neg = s[f], s[~f]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.r_[pos, neg]
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):                      # average ranks for ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarize(alarmed, success, alarm_step=None, fail_step=None, n_steps=None) -> dict:
    """Episode-level summary.

    false_alarm_rate: alarms among successful episodes (the promised quantity)
    detection_rate:   alarms among failed episodes
    lead_steps:       for detected failures, steps between alarm and the
                      labeled failure moment (or episode end if not labeled);
                      positive means the alarm came first
    """
    alarmed = np.asarray(alarmed, bool)
    success = np.asarray(success, bool)
    ns, nf = int(success.sum()), int((~success).sum())
    fa, det = int((alarmed & success).sum()), int((alarmed & ~success).sum())
    out = {
        "n_success": ns, "n_fail": nf,
        "false_alarm_rate": fa / ns if ns else float("nan"),
        "false_alarm_ci": wilson(fa, ns),
        "detection_rate": det / nf if nf else float("nan"),
        "detection_ci": wilson(det, nf),
    }
    if alarm_step is not None and n_steps is not None:
        leads = []
        fs = fail_step if fail_step is not None else [None] * len(alarmed)
        for a, y, t_a, t_f, T in zip(alarmed, success, alarm_step, fs, n_steps):
            if a and not y and t_a is not None:
                ref = t_f if t_f is not None else T
                leads.append(ref - t_a)
        out["median_lead_steps"] = float(np.median(leads)) if leads else float("nan")
    return out
