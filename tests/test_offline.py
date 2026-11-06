import numpy as np
import pytest

from shiftbench.perturb import SHIFTS, apply
from shiftbench.encoders import ENCODERS, make_encoder


def test_every_shift_changes_frames_and_keeps_shape():
    rng = np.random.default_rng(0)
    fr = (rng.random((6, 48, 64, 3)) * 255).astype(np.uint8)
    for s in SHIFTS:
        o = apply(fr, s)
        assert o.shape == fr.shape and o.dtype == np.uint8
        assert (o == fr).all() if s == "nominal" else (o != fr).any()


def test_onset_leaves_early_frames_untouched():
    fr = np.full((10, 32, 32, 3), 120, np.uint8)
    o = apply(fr, "light_dim_2", onset_frac=0.5)
    assert (o[:5] == fr[:5]).all() and (o[5:] != fr[5:]).all()


def test_shifts_are_deterministic():
    fr = (np.random.default_rng(1).random((3, 40, 40, 3)) * 255).astype(np.uint8)
    for s in SHIFTS:
        np.testing.assert_array_equal(apply(fr, s), apply(fr, s))


def test_encoder_registry():
    assert {"vjepa2", "dinov2_b", "siglip2_b", "clip_b32", "resnet18"} <= set(ENCODERS)
    assert make_encoder("tiny").encode(np.zeros((2, 32, 32, 3), np.uint8)).shape[0] == 2
    with pytest.raises(ValueError):
        make_encoder("nope")


def test_v3_reader_on_local_dataset(tmp_path):
    """Builds a tiny LeRobot v3 dataset (needs lerobot) and checks the direct reader
    returns the exact frames: a dark square moves one pixel per frame."""
    pytest.importorskip("lerobot")
    pytest.importorskip("av")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from shiftbench.lerobot_v3 import V3Dataset

    names = [f"j{i}.pos" for i in range(6)]
    feats = {"observation.state": {"dtype": "float32", "shape": (6,), "names": names},
             "action": {"dtype": "float32", "shape": (6,), "names": names},
             "observation.images.front": {"dtype": "video", "shape": (48, 64, 3), "names": ["height", "width", "channels"]}}
    ds = LeRobotDataset.create("local/t", fps=30, features=feats, root=tmp_path / "ds", use_videos=True)
    for ep in range(2):
        for t in range(40):
            img = np.full((48, 64, 3), 160, np.uint8)
            img[20:28, t + 5:t + 11] = 20
            st = np.full(6, t, np.float32)
            ds.add_frame({"observation.state": st, "action": st + 1, "observation.images.front": img, "task": "x"})
        ds.save_episode()
    ds.finalize()
    v = V3Dataset(tmp_path / "ds")
    ep = v.episode(1, "observation.images.front", target_fps=6)          # stride 5
    assert ep["frames"].shape[0] == 8 and ep["fps"] == 6
    for f, s in zip(ep["frames"], ep["state"]):
        t = int(s[0])
        assert np.where(f[24, :, 0] < 80)[0].min() == t + 5
    np.testing.assert_array_equal(ep["action"][:, 0], np.minimum(np.arange(0, 40, 5) + 4, 39) + 1)
