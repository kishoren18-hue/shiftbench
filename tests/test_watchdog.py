"""The watchdog robot class, driven without hardware.

The motor bus and camera are faked by patching SOFollower's I/O methods, so
this checks the real SO101Watchdog wiring: config registration, observing
frames, holding position after an alarm, shadow mode, and episode resets.
"""
import numpy as np
import pytest

pytest.importorskip("lerobot")

from lerobot.robots.so_follower.so_follower import SOFollower  # noqa: E402
from lerobot.robots.utils import make_robot_from_config  # noqa: E402
from lerobot_robot_shiftbench import Monitor, SO101Watchdog, SO101WatchdogConfig  # noqa: E402

from shiftbench.detector import Detector, DetectorConfig  # noqa: E402
from shiftbench.encoders import TinyEncoder  # noqa: E402
from shiftbench.synthetic import World  # noqa: E402

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@pytest.fixture(scope="module")
def det_dir(tmp_path_factory):
    """A detector trained on TinyEncoder latents of random frames (6-d state/action)."""
    enc = TinyEncoder(dim=16)
    rng = np.random.default_rng(0)
    from shiftbench.data import Episode
    eps = []
    for k in range(12):
        frames = (rng.random((12, 32, 32, 3)) * 40 + 100).astype(np.uint8)   # dull grey scenes
        eps.append(Episode(z=enc.encode(frames), state=rng.normal(size=(12, 6)).astype(np.float32),
                           action=rng.normal(size=(12, 6)).astype(np.float32), success=True))
    det = Detector(DetectorConfig(alpha=0.1, epochs=3, hidden=32, layers=1, window=2)).fit(eps[:6], eps[6:], verbose=False)
    d = tmp_path_factory.mktemp("det")
    det.save(d)
    return str(d)


class FakeIO:
    def __init__(self):
        self.sent = []
        self.frame = np.full((32, 32, 3), 120, np.uint8)
        self.pos = {f"{m}.pos": 0.0 for m in MOTORS}

    def get_observation(self, robot):
        return {**self.pos, "front": self.frame}

    def send_action(self, robot, action):
        self.sent.append(dict(action))
        return dict(action)


@pytest.fixture
def robot(det_dir, monkeypatch, tmp_path):
    io = FakeIO()
    monkeypatch.setattr(SOFollower, "get_observation", lambda self: io.get_observation(self))
    monkeypatch.setattr(SOFollower, "send_action", lambda self, a: io.send_action(self, a))
    cfg = SO101WatchdogConfig(port="/dev/null", id="test", detector_dir=det_dir, encoder="tiny",
                              log_path=str(tmp_path / "log.jsonl"), mode="stop")
    r = make_robot_from_config(cfg)
    assert isinstance(r, SO101Watchdog)
    r.monitor = r._build_monitor()                 # what connect() does, minus the serial port
    r.monitor.enc = TinyEncoder(dim=16)
    return r, io


def test_registered_as_robot_type():
    from lerobot.robots.config import RobotConfig
    assert "so101_watchdog" in RobotConfig.get_known_choices()


def test_holds_position_after_alarm(robot):
    r, io = robot
    r.monitor.cutoff = -np.inf                     # anything alarms
    act = {f"{m}.pos": 10.0 for m in MOTORS}
    for _ in range(5):
        r.get_observation()
        r.send_action(act)
        r.monitor.tick()
    assert r.monitor.tripped.is_set()
    io.pos = {f"{m}.pos": 3.0 for m in MOTORS}
    r.get_observation()
    r.send_action(act)
    assert io.sent[-1] == {f"{m}.pos": 3.0 for m in MOTORS}   # held, not 10.0


def test_shadow_mode_never_overrides(robot):
    r, io = robot
    r.config.mode = r.monitor.mode = "shadow"
    r.monitor.cutoff = -np.inf
    act = {f"{m}.pos": 10.0 for m in MOTORS}
    for _ in range(5):
        r.get_observation()
        r.send_action(act)
        r.monitor.tick()
    assert r.monitor.tripped.is_set()
    assert io.sent[-1] == act


def test_no_alarm_with_infinite_cutoff_and_reset(robot):
    r, io = robot
    r.monitor.cutoff = np.inf
    act = {f"{m}.pos": 1.0 for m in MOTORS}
    for _ in range(6):
        r.get_observation()
        r.send_action(act)
        r.monitor.tick()
    assert not r.monitor.tripped.is_set()
    assert io.sent[-1] == act
    r.monitor.tripped.set()
    r.monitor.new_episode()
    assert not r.monitor.tripped.is_set() and r.monitor.scorer.z == []


def test_log_written(robot):
    import json
    r, io = robot
    for _ in range(4):
        r.get_observation()
        r.send_action({f"{m}.pos": 0.0 for m in MOTORS})
        r.monitor.tick()
    lines = [json.loads(x) for x in open(r.config.log_path)]
    assert len(lines) == 4 and {"score", "cutoff", "alarm"} <= set(lines[0])
