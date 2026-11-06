import numpy as np
import pytest
import torch

from shiftbench.data import Episode, load_cache, read_labels, write_label
from shiftbench.detector import Detector, DetectorConfig, OnlineScorer
from shiftbench.encoders import TinyEncoder, resize_center_crop, to_float_chw
from shiftbench.synthetic import World


@pytest.fixture(scope="module")
def trained():
    w = World(seed=0, D=24, T=24)
    train = w.dataset({("nominal", 0): 25}, seed=1)
    cal = w.dataset({("nominal", 0): 40}, seed=2)
    det = Detector(DetectorConfig(alpha=0.1, epochs=8, hidden=64, layers=2)).fit(train, cal, verbose=False)
    return w, det, cal


def test_training_reduces_error(trained):
    w, det, cal = trained
    e = cal[0]
    zero = np.abs((e.z[1:] - e.z[:-1]) / det.wm.norm.z_sd).mean()   # "nothing changes" guess
    assert det.wm.errors(e).mean() < zero


def test_save_load_roundtrip(trained, tmp_path):
    w, det, cal = trained
    det.save(tmp_path / "d")
    d2 = Detector.load(tmp_path / "d")
    np.testing.assert_allclose(det.trace(cal[1]), d2.trace(cal[1]), rtol=1e-5)
    assert d2.tau == det.tau


@pytest.mark.parametrize("ref", [None, 3])
def test_online_scorer_matches_offline(trained, ref):
    w, det, cal = trained
    det.cfg.ref_steps = ref
    e = cal[2]
    off = det.trace(e)
    sc = OnlineScorer(det)
    on = []
    for t in range(len(e)):
        r = sc.step(e.z[t], e.state[t], e.action[t - 1] if t > 0 else None)
        if r is not None:
            on.append(r)
    det.cfg.ref_steps = None
    off = off[np.isfinite(off)]
    np.testing.assert_allclose(on, off, rtol=1e-4, atol=1e-6)


def test_episode_cache_roundtrip(tmp_path):
    e = Episode(z=np.random.rand(5, 8).astype(np.float32), state=np.zeros((5, 6), np.float32),
                action=np.ones((5, 6), np.float32), success=False, condition="light_dim", level=2,
                episode_index=7, fail_step=3)
    e.save(tmp_path / "ep_000007.npz")
    (b,) = load_cache(tmp_path)
    assert b.condition == "light_dim" and b.fail_step == 3 and not b.success
    np.testing.assert_allclose(b.z, e.z, atol=1e-3)


def test_labels_last_write_wins(tmp_path):
    write_label(tmp_path, 3, True, "nominal")
    write_label(tmp_path, 3, False, "nominal", notes="fixed typo")
    assert read_labels(tmp_path)[3]["success"] is False


def test_tiny_encoder_shapes():
    enc = TinyEncoder(dim=16)
    frames = (np.random.rand(3, 120, 160, 3) * 255).astype(np.uint8)
    z = enc.encode(frames)
    assert z.shape == (3, 16)
    np.testing.assert_allclose(enc.encode(frames), z)          # deterministic
    x = resize_center_crop(to_float_chw(frames), 64)
    assert x.shape == (3, 3, 64, 64)


def test_vjepa2_encoder_code_path(monkeypatch):
    """Runs VJEPA2Encoder on a tiny randomly initialised V-JEPA 2 (no download)."""
    tr = pytest.importorskip("transformers")
    from shiftbench import encoders

    cfg = tr.VJEPA2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, crop_size=32,
                          patch_size=8, pred_hidden_size=16, pred_num_hidden_layers=1,
                          pred_num_attention_heads=2, frames_per_clip=2)
    model = tr.VJEPA2Model(cfg)
    monkeypatch.setattr(tr.AutoModel, "from_pretrained", lambda *a, **k: model)
    for pool, dim in (("mean", 32), ("grid2", 128)):
        enc = encoders.VJEPA2Encoder("fake", pool=pool, device="cpu")
        z = enc.encode((np.random.rand(3, 48, 64, 3) * 255).astype(np.uint8))
        assert z.shape == (3, dim) and np.isfinite(z).all()
