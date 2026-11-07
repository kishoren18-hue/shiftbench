import numpy as np

from shiftbench.data import Episode
from shiftbench.refscores import ReferenceBank


def _ep(z, fps=2.0):
    return Episode(z=z.astype(np.float32), state=np.zeros((len(z), 6), np.float32),
                   action=np.zeros((len(z), 6), np.float32), success=True, fps=fps)


def test_deviating_run_scores_higher():
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(0, 1, (30, 16)), 0)
    bank = ReferenceBank([_ep(base + rng.normal(0, 0.3, base.shape)) for _ in range(10)], n_comp=8)
    good = _ep(base + rng.normal(0, 0.3, base.shape))
    bad = base + rng.normal(0, 0.3, base.shape)
    bad[15:] += 6.0                                   # goes wrong halfway
    bad = _ep(bad)
    dg, db = bank.trajdev(good), bank.trajdev(bad)
    assert db[20:].mean() > 2 * dg[20:].mean()
    assert abs(db[:10].mean() - dg[:10].mean()) < 0.5 * dg[:10].mean() + 0.5   # same before the failure
    assert bank.endstate(bad) > bank.endstate(good)


def test_trajdev_is_causal():
    rng = np.random.default_rng(1)
    eps = [_ep(rng.normal(size=(20, 8))) for _ in range(5)]
    bank = ReferenceBank(eps, n_comp=4)
    x = rng.normal(size=(20, 8))
    full = bank.trajdev(_ep(x))
    head = bank.trajdev(_ep(x[:12]))
    np.testing.assert_allclose(full[:12], head)       # step t never looks at later frames
