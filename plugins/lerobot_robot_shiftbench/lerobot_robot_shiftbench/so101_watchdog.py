"""SO-101 follower with a ShiftBench watchdog that can stop the arm.

Install this package (pip install -e plugins/lerobot_robot_shiftbench) and
LeRobot picks it up automatically: any LeRobot command that takes
--robot.type accepts `so101_watchdog`, e.g.

    lerobot-rollout --robot.type=so101_watchdog --robot.port=/dev/ttyACM0 \
        --robot.id=my_arm --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
        --robot.detector_dir=runs/detector --robot.watch_camera=front --robot.mode=stop \
        --policy.path=$HF_USER/act_so101_pick --task="put the cube in the bowl" --duration=30

It behaves exactly like so101_follower (same calibration files, same
features) plus:

  * a background thread grabs the newest frame from `watch_camera` at
    `detector_hz`, encodes it with the frozen encoder, and scores it with the
    trained world model (shiftbench.OnlineScorer);
  * when the score crosses the cutoff, mode="stop" latches an alarm and every
    later send_action is replaced by "hold the current joint positions"
    until the episode resets; mode="shadow" only logs (use this when
    collecting the dataset, so every episode gets a true success label);
  * every tick is written to `log_path` as one JSON line (time, score,
    cutoff, alarm) for plots and the demo video overlay.

The detector runs off the control thread, so a slow encoder lowers the
detector rate, not the control rate. On a laptop CPU V-JEPA 2 ViT-L manages
roughly 1-2 frames per second; use a GPU, or set detector_hz to what the
machine sustains (the world model must be trained at the same rate).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from lerobot.robots.config import RobotConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

log = logging.getLogger(__name__)


@RobotConfig.register_subclass("so101_watchdog")
@dataclass
class SO101WatchdogConfig(SOFollowerRobotConfig):
    detector_dir: str = ""                 # output of scripts/train_detector.py
    watch_camera: str = "front"            # key in --robot.cameras
    mode: str = "stop"                     # "stop" or "shadow"
    detector_hz: float = 4.0               # must match the rate the world model was trained at
    encoder: str = "vjepa2"                # "vjepa2" or "tiny" (tests only)
    encoder_model: str = "facebook/vjepa2-vitl-fpc64-256"
    encoder_pool: str = "mean"
    device: str | None = None
    log_path: str = "watchdog_log.jsonl"
    reset_gap_s: float = 1.5               # no actions for this long = new episode
    cutoff: float | None = None            # override the calibrated cutoff
    adaptive: bool = False                 # use the adaptive cutoff (needs labels via label_last())


class Monitor:
    """Detector loop shared by the robot class and the tests."""

    def __init__(self, detector, encoder, hz=4.0, mode="stop", log_path=None, cutoff=None, adaptive=None):
        from shiftbench.detector import OnlineScorer

        self.det = detector
        self.enc = encoder
        self.period = 1.0 / hz
        self.mode = mode
        self.cutoff = detector.tau if cutoff is None else cutoff
        self.adaptive = adaptive
        self.scorer = OnlineScorer(detector)
        self.log_path = log_path
        self._lock = threading.Lock()
        self._frame = None
        self._state = None
        self._action = None
        self._last_action_t = None
        self.tripped = threading.Event()
        self.episode = 0
        self.max_score = -np.inf
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------ inputs from the control loop
    def observe(self, frame, state):
        with self._lock:
            self._frame, self._state = frame, state

    def record_action(self, action_vec, now=None):
        with self._lock:
            self._action = action_vec
            self._last_action_t = now if now is not None else time.monotonic()

    def current_cutoff(self):
        if self.adaptive is not None:          # adaptive cutoffs live on the log scale
            return float(np.exp(self.adaptive.tau))
        return self.cutoff

    # ------------------------------------------------ one detector tick
    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            frame, state, action, last_t = self._frame, self._state, self._action, self._last_action_t
        if frame is None or state is None:
            return None
        z = self.enc.encode(np.asarray(frame)[None])[0]
        score = self.scorer.step(z, state, action)
        cut = self.current_cutoff()
        alarm = score is not None and score > cut
        if score is not None:
            self.max_score = max(self.max_score, score)
        if alarm and not self.tripped.is_set():
            self.tripped.set()
            msg = f"[shiftbench] ALARM episode {self.episode}: score {score:.4f} > cutoff {cut:.4f}"
            msg += " -> holding position" if self.mode == "stop" else " (shadow mode, not stopping)"
            log.warning(msg)
            print(msg, flush=True)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"t": now, "episode": self.episode, "score": score, "cutoff": cut,
                                    "alarm": bool(alarm), "tripped": self.tripped.is_set(),
                                    "mode": self.mode}) + "\n")
        return score

    def new_episode(self):
        self.scorer.reset()
        self.tripped.clear()
        self.max_score = -np.inf
        self.episode += 1

    def label_last(self, success: bool):
        """Feed back the outcome of the episode that just ended (adaptive mode)."""
        if self.adaptive is not None:
            from shiftbench.adaptive import to_log
            self.adaptive.update(self.adaptive.alarms(float(to_log(self.max_score))), success)

    # ------------------------------------------------ thread
    def _loop(self, reset_gap_s):
        nxt = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                last = self._last_action_t
            if last is not None and now - last > reset_gap_s and self.scorer.z:
                self.new_episode()
            try:
                self.tick(now)
            except Exception:                         # never take the control loop down
                log.exception("shiftbench watchdog tick failed")
            nxt += self.period
            time.sleep(max(0.0, nxt - time.monotonic()))

    def start(self, reset_gap_s=1.5):
        self._thread = threading.Thread(target=self._loop, args=(reset_gap_s,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


class SO101Watchdog(SOFollower):
    config_class = SO101WatchdogConfig
    # keep name = "so_follower" so the arm's existing calibration file is reused

    def __init__(self, config: SO101WatchdogConfig):
        super().__init__(config)
        self.config = config
        self.monitor: Monitor | None = None
        self._last_pos: dict[str, float] = {}

    def _build_monitor(self):
        from shiftbench.detector import Detector
        from shiftbench.encoders import make_encoder

        c = self.config
        if not c.detector_dir:
            raise ValueError("--robot.detector_dir is required for so101_watchdog")
        det = Detector.load(c.detector_dir, device=c.device)
        if c.encoder == "vjepa2":
            enc = make_encoder("vjepa2", model_id=c.encoder_model, pool=c.encoder_pool, device=c.device)
        else:
            enc = make_encoder(c.encoder)
        ad = None
        if c.adaptive:
            ad = det.default_adaptive(eta_mult=1.0, log=True)
        return Monitor(det, enc, hz=c.detector_hz, mode=c.mode, log_path=c.log_path, cutoff=c.cutoff, adaptive=ad)

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate)
        self.monitor = self._build_monitor()
        self.monitor.start(self.config.reset_gap_s)
        print(f"[shiftbench] watchdog on '{self.config.watch_camera}' at {self.config.detector_hz} Hz, "
              f"mode={self.config.mode}, cutoff={self.monitor.current_cutoff():.4f}", flush=True)

    def _state_vec(self, obs):
        return np.array([obs[f"{m}.pos"] for m in self.bus.motors], np.float32)

    def get_observation(self):
        obs = super().get_observation()
        self._last_pos = {f"{m}.pos": float(obs[f"{m}.pos"]) for m in self.bus.motors}
        if self.monitor is not None and self.config.watch_camera in obs:
            self.monitor.observe(obs[self.config.watch_camera], self._state_vec(obs))
        return obs

    def send_action(self, action):
        if self.monitor is not None and self.monitor.tripped.is_set() and self.config.mode == "stop" and self._last_pos:
            action = dict(self._last_pos)                     # hold where we are
        sent = super().send_action(action)
        if self.monitor is not None:
            self.monitor.record_action(np.array([sent[f"{m}.pos"] for m in self.bus.motors], np.float32))
        return sent

    def disconnect(self):
        if self.monitor is not None:
            self.monitor.stop()
        super().disconnect()
