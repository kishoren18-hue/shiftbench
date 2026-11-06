"""Replay demo: real SO-101 footage, a camera bump halfway through, and the
watchdog's live score next to it.

For each chosen episode it renders two MP4s (H.264):
  <ep>_bump.mp4     the camera is "bumped" (affine shift, perturb.cam_bump_2) at
                    50% of the episode, from that frame on
  <ep>_control.mp4  the same footage, untouched

The detector runs exactly as live (OnlineScorer, one tick per detector step,
frozen encoder). When the score crosses the cutoff the frame turns grey with
"WATCHDOG WOULD STOP THE ARM HERE". This is recorded footage, so the robot in
the video doesn't actually stop. The caption says so; keep it that way.

  python scripts/replay_demo.py --detector OUT/detector_vjepa2 --encoder vjepa2 \
      --episodes_csv OUT/episodes.csv --traces OUT/traces.parquet --out OUT/demo --n 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from shiftbench.detector import Detector, OnlineScorer
from shiftbench.encoders import make_encoder
from shiftbench.lerobot_v3 import V3Dataset
from shiftbench.perturb import apply

p = argparse.ArgumentParser()
p.add_argument("--repo_id", default="armnet/armnetbench_v01_lerobot_so101")
p.add_argument("--data_dir", default="/content/armnet")
p.add_argument("--camera", default="observation.images.front")
p.add_argument("--detector", required=True)
p.add_argument("--encoder", default="vjepa2")
p.add_argument("--episodes", default=None, help="comma-separated; default: picked from traces")
p.add_argument("--episodes_csv", default=None)
p.add_argument("--traces", default=None)
p.add_argument("--shift", default="cam_bump_2")
p.add_argument("--onset", type=float, default=0.5)
p.add_argument("--n", type=int, default=3)
p.add_argument("--out", required=True)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

LOCAL = a.repo_id.startswith("local:")
ds = V3Dataset(a.repo_id.split(":", 1)[1]) if LOCAL else V3Dataset(a.data_dir)
det = Detector.load(a.detector)
enc = make_encoder(a.encoder)

if a.episodes:
    eps = [int(x) for x in a.episodes.split(",")]
else:   # calmest successful test rollouts under the nominal image: the fairest "it was going fine" clips
    tr = pd.read_parquet(a.traces)
    tr = tr[(tr.encoder == a.encoder) & (tr.variant == "full") & (tr.experiment == "B") & (tr["shift"] == "nominal") & tr.success]
    eps = tr.sort_values("score").episode_index.head(a.n).astype(int).tolist()

W_IMG, W_PLOT, H = 640, 420, 360


def render(frames, times, scores_t, scores, tau, onset_t, bumped, path, title):
    """frames at full rate; scores at detector ticks (times scores_t)."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_IMG + W_PLOT}x{H}",
           "-r", "20", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast", str(path)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    finite = [s for s in scores if s is not None and np.isfinite(s)]
    smax = max(max(finite, default=tau), tau) * 1.25
    T = times[-1] if len(times) else 1
    alarm_t = next((t for t, s in zip(scores_t, scores) if s is not None and s > tau), None)
    for fr, t in zip(frames, times):
        img = cv2.resize(fr, (W_IMG, H))
        stopped = alarm_t is not None and t >= alarm_t
        if stopped:
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = (0.35 * img + 0.65 * np.stack([g] * 3, -1)).astype(np.uint8)
            cv2.rectangle(img, (0, H - 46), (W_IMG, H), (200, 30, 30), -1)
            cv2.putText(img, "WATCHDOG WOULD STOP THE ARM HERE", (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if bumped and t >= onset_t:
            cv2.putText(img, "camera bumped", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)
        cv2.putText(img, f"{t:5.1f}s", (W_IMG - 90, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        # plot panel
        pl = np.full((H, W_PLOT, 3), 250, np.uint8)
        x0, x1, y0, y1 = 50, W_PLOT - 15, 40, H - 40
        X = lambda tt: int(x0 + (x1 - x0) * tt / T)
        Y = lambda s: int(y1 - (y1 - y0) * min(s, smax) / smax)
        cv2.line(pl, (x0, y1), (x1, y1), (120, 120, 120), 1)
        cv2.line(pl, (x0, y0), (x0, y1), (120, 120, 120), 1)
        for xx in range(x0, x1, 12):
            cv2.line(pl, (xx, Y(tau)), (xx + 6, Y(tau)), (0, 0, 0), 1)
        cv2.putText(pl, "alarm cutoff", (x1 - 110, Y(tau) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        if bumped:
            cv2.line(pl, (X(onset_t), y0), (X(onset_t), y1), (0, 170, 230), 1)
        pts = [(X(tt), Y(s)) for tt, s in zip(scores_t, scores) if s is not None and np.isfinite(s) and tt <= t]
        for (p1, p2) in zip(pts, pts[1:]):
            cv2.line(pl, p1, p2, (200, 30, 30) if p2[1] < Y(tau) else (40, 60, 160), 2)
        cv2.putText(pl, "world-model surprise", (x0, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
        cv2.putText(pl, "time", (x1 - 40, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
        cv2.putText(pl, title, (x0, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
        pr.stdin.write(np.concatenate([img, pl], 1).tobytes())
    pr.stdin.close()
    pr.wait()
    return alarm_t


meta_out = []
for ei in eps:
    frames, ts = ds.episode_all_frames(ei, a.camera, max_side=640)
    ts = ts - ts[0]
    fps = 1 / np.median(np.diff(ts))
    stride = max(1, round(fps / 4.0))
    d = ds.episode(ei, a.camera, target_fps=fps / stride)          # state/action on the same ticks
    tick = np.arange(0, len(frames), stride)[: len(d["state"])]
    for bumped in (True, False):
        fr = apply(frames, a.shift, onset_frac=a.onset) if bumped else frames
        z = enc.encode(fr[tick])
        sc = OnlineScorer(det)
        scores = [sc.step(z[i], d["state"][i], d["action"][i - 1] if i else None) for i in range(len(tick))]
        name = f"ep{ei}_{'bump' if bumped else 'control'}.mp4"
        alarm_t = render(fr, ts, ts[tick], scores, det.tau, ts[int(a.onset * len(ts))], bumped, out / name,
                         "replay of recorded ArmnetBench footage")
        meta_out.append({"episode": ei, "clip": name, "bumped": bumped, "onset_s": float(ts[int(a.onset * len(ts))]),
                         "alarm_s": None if alarm_t is None else float(alarm_t), "cutoff": det.tau,
                         "scores_t": ts[tick].tolist(), "scores": [None if s is None or not np.isfinite(s) else float(s) for s in scores]})
        print(name, "alarm at", alarm_t)
json.dump(meta_out, open(out / "demo_meta.json", "w"), indent=1)
