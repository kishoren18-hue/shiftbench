import numpy as np
import pytest

from shiftbench.adaptive import AdaptiveAlarm, run_adaptive, to_log
from shiftbench.calibrate import conformal_cutoff, min_calibration_size
from shiftbench.metrics import auroc, wilson
from shiftbench.scores import smooth, reference, first_alarm, episode_score


# ---------------------------------------------------------------- conformal
def test_conformal_false_alarm_rate_holds_on_exchangeable_data():
    rng = np.random.default_rng(0)
    alpha, n, reps = 0.1, 30, 4000
    fa = 0
    for _ in range(reps):
        s = rng.gamma(2.0, size=n + 1)
        fa += s[-1] > conformal_cutoff(s[:n], alpha)
    assert fa / reps <= alpha + 0.015          # exact bound is alpha; MC slack


def test_conformal_never_alarms_when_too_few_successes():
    assert min_calibration_size(0.05) == 19
    assert conformal_cutoff(np.arange(18.0), 0.05) == np.inf
    assert np.isfinite(conformal_cutoff(np.arange(19.0), 0.05))


def test_conformal_ties_do_not_alarm():
    s = np.ones(50)
    assert not (1.0 > conformal_cutoff(s, 0.1))


# ---------------------------------------------------------------- adaptive
@pytest.mark.parametrize("seq", ["adversarial", "random", "drift", "step"])
def test_adaptive_bound_holds_on_any_sequence(seq):
    rng = np.random.default_rng(1)
    T = 400
    if seq == "adversarial":
        s = np.full(T, 100.0)                  # everything looks terrible, all succeed
    elif seq == "random":
        s = rng.normal(0, 3, T)
    elif seq == "drift":
        s = np.linspace(-2, 20, T) + rng.normal(0, 0.5, T)
    else:
        s = np.r_[rng.normal(0, 1, T // 2), rng.normal(8, 1, T // 2)]
    y = rng.random(T) < 0.8
    for eta in (0.05, 0.3, 2.0):
        ad = AdaptiveAlarm(alpha=0.05, start=1.0, eta=eta, s_max=10.0)
        alarmed, _ = run_adaptive(s, y, ad.alpha, ad.tau_start, ad.eta, ad.s_max)
        Ts = int(y.sum())
        assert (alarmed & y).sum() / Ts <= ad.bound(Ts) + 1e-12


def test_adaptive_ignores_failed_episodes():
    ad = AdaptiveAlarm(alpha=0.05, start=1.0, eta=0.5, s_max=5)
    t0 = ad.tau
    ad.update(alarmed=True, success=False)
    assert ad.tau == t0 and ad.n_success == 0


def test_adaptive_conservative_labels_bound_all_alarms():
    rng = np.random.default_rng(2)
    s = rng.normal(2, 1, 300)
    y = rng.random(300) < 0.7
    ad = AdaptiveAlarm(alpha=0.1, start=0.0, eta=0.3, s_max=6)
    alarmed, _ = run_adaptive(s, y, 0.1, 0.0, 0.3, 6, label_mode="conservative")
    # every episode updates; alarms counted as false -> bound on overall alarm rate
    n_upd = (alarmed | y).sum()
    assert alarmed.sum() / n_upd <= ad.bound(int(n_upd)) + 1e-12


def test_to_log_handles_zero():
    assert np.isfinite(to_log(0.0))


# ---------------------------------------------------------------- scores / metrics
def test_smooth_and_reference():
    e = np.array([1.0, 3.0, 5.0, 7.0])
    np.testing.assert_allclose(smooth(e, 2), [1, 2, 4, 6])
    np.testing.assert_allclose(reference(e, 2), e - 2.0)
    assert first_alarm(smooth(e, 2), 3.5) == 2
    assert first_alarm(e, 100) is None
    assert episode_score(np.zeros(0)) == -np.inf


def test_auroc_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    s = np.round(rng.normal(size=200), 1)       # ties on purpose
    f = rng.random(200) < 0.3
    assert abs(auroc(s, f) - sk.roc_auc_score(f, s)) < 1e-9


def test_wilson_contains_point():
    lo, hi = wilson(3, 60)
    assert lo < 0.05 < hi
