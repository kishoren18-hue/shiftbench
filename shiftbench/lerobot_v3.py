"""Read LeRobot v3 datasets directly: metadata, joint data, and one camera
at a chosen frame rate, without installing LeRobot.

LeRobot v3 packs many episodes into each parquet / mp4 file. The per-episode
rows of meta/episodes/*.parquet say which files hold an episode and where:

    data/chunk_index, data/file_index           -> data/chunk-XXX/file-YYY.parquet
    videos/<cam>/chunk_index, .../file_index    -> videos/<cam>/chunk-XXX/file-YYY.mp4
    videos/<cam>/from_timestamp                 -> where the episode starts in that mp4

Only the files the chosen episodes need are downloaded. Frames are decoded in
one forward pass per episode (seek once, keep the frames nearest the wanted
timestamps), which is far faster than random access per frame.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(template: str, **kw) -> str:
    return template.format(**kw)


class V3Dataset:
    def __init__(self, root):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        files = sorted(glob.glob(str(self.root / "meta/episodes/*/*.parquet")))
        keep = lambda c: not c.startswith("stats/")
        self.episodes = pd.concat([pd.read_parquet(f, columns=None) for f in files], ignore_index=True)
        self.episodes = self.episodes[[c for c in self.episodes.columns if keep(c)]].set_index("episode_index", drop=False)
        self.fps = float(self.info["fps"])
        self._data_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------ download
    @staticmethod
    def download(repo_id: str, local_dir, episodes=None, cameras=("observation.images.front",), revision=None):
        """Fetch meta/, then only the data and video files the episodes need."""
        from huggingface_hub import snapshot_download

        local_dir = Path(local_dir)
        snapshot_download(repo_id, repo_type="dataset", local_dir=local_dir, allow_patterns=["meta/*", "meta/**"],
                          revision=revision)
        if episodes is None:
            return V3Dataset(local_dir)
        ds = V3Dataset(local_dir)
        pats = set()
        for ei in episodes:
            r = ds.episodes.loc[ei]
            pats.add(_fmt(ds.info["data_path"], chunk_index=int(r["data/chunk_index"]), file_index=int(r["data/file_index"])))
            for cam in cameras:
                pats.add(_fmt(ds.info["video_path"], video_key=cam, chunk_index=int(r[f"videos/{cam}/chunk_index"]),
                              file_index=int(r[f"videos/{cam}/file_index"])))
        snapshot_download(repo_id, repo_type="dataset", local_dir=local_dir, allow_patterns=sorted(pats),
                          revision=revision)
        return V3Dataset(local_dir)

    # ------------------------------------------------------------ labels
    def labels(self) -> pd.DataFrame:
        """episode_index, task, success (bool or NaN), success_class, policy_type, length."""
        e = self.episodes
        out = pd.DataFrame({"episode_index": e["episode_index"].astype(int), "length": e["length"].astype(int)})
        out["task"] = e["tasks"].map(lambda t: t[0] if len(t) else "")
        if "success_class" in e:
            out["success_class"] = e["success_class"]
            out["success"] = e["success_class"].map({"successful": True, "failure": False})
        elif "success" in e:
            out["success"] = e["success"].astype(bool)
        for c in ("policy_type", "policy_repo_id"):
            if c in e:
                out[c] = e[c].fillna("").replace("", "teleoperated") if c == "policy_type" else e[c]
        return out.reset_index(drop=True)

    # ------------------------------------------------------------ one episode
    def _data_file(self, r) -> pd.DataFrame:
        p = _fmt(self.info["data_path"], chunk_index=int(r["data/chunk_index"]), file_index=int(r["data/file_index"]))
        if p not in self._data_cache:
            self._data_cache = {p: pd.read_parquet(self.root / p)}          # keep one file in memory
        return self._data_cache[p]

    def episode(self, ei: int, camera: str, target_fps: float = 4.0, max_side: int | None = 480):
        """Returns dict: frames (T,H,W,3) uint8 RGB, state (T,S), action (T,A), t (T,) seconds,
        fps (detector rate). Action at step t is the command at the end of [t, t+stride)."""
        r = self.episodes.loc[ei]
        d = self._data_file(r)
        d = d[d["episode_index"] == ei].sort_values("frame_index")
        stride = max(1, round(self.fps / target_fps))
        n = len(d)
        pick = np.arange(0, n, stride)
        act_idx = np.minimum(pick + stride - 1, n - 1)
        ts = d["timestamp"].to_numpy()[pick].astype(float)
        state = np.stack(d["observation.state"].to_numpy()[pick]).astype(np.float32)
        action = np.stack(d["action"].to_numpy()[act_idx]).astype(np.float32)
        vp = self.root / _fmt(self.info["video_path"], video_key=camera, chunk_index=int(r[f"videos/{camera}/chunk_index"]),
                              file_index=int(r[f"videos/{camera}/file_index"]))
        frames = decode_at(vp, float(r[f"videos/{camera}/from_timestamp"]) + ts, max_side=max_side,
                           tol=0.5 / self.fps)
        return {"frames": frames, "state": state, "action": action, "t": ts, "fps": self.fps / stride}

    def episode_all_frames(self, ei: int, camera: str, max_side: int | None = 480):
        """Every frame (for demo videos) plus timestamps."""
        r = self.episodes.loc[ei]
        d = self._data_file(r)
        ts = np.sort(d[d["episode_index"] == ei]["timestamp"].to_numpy().astype(float))
        vp = self.root / _fmt(self.info["video_path"], video_key=camera, chunk_index=int(r[f"videos/{camera}/chunk_index"]),
                              file_index=int(r[f"videos/{camera}/file_index"]))
        return decode_at(vp, float(r[f"videos/{camera}/from_timestamp"]) + ts, max_side=max_side, tol=0.5 / self.fps), ts


def decode_at(path, times, max_side=None, tol=0.05) -> np.ndarray:
    """Decode the frames nearest to `times` (seconds, increasing) in one pass."""
    import av

    times = np.asarray(times, float)
    out = [None] * len(times)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        tb = float(s.time_base)
        c.seek(max(0, int((times[0] - 1.0) / tb)), stream=s, backward=True, any_frame=False)
        j = 0
        prev = None
        for fr in c.decode(s):
            ft = fr.pts * tb
            while j < len(times) and ft >= times[j] - tol:
                cand = fr if (prev is None or abs(ft - times[j]) <= abs(prev[0] - times[j])) else prev[1]
                if abs((ft if cand is fr else prev[0]) - times[j]) <= tol * 2:
                    img = cand.to_ndarray(format="rgb24")
                    if max_side and max(img.shape[:2]) > max_side:
                        import cv2
                        sc = max_side / max(img.shape[:2])
                        img = cv2.resize(img, (round(img.shape[1] * sc), round(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
                    out[j] = img
                j += 1
            prev = (ft, fr)
            if j >= len(times):
                break
    missing = [i for i, o in enumerate(out) if o is None]
    if missing:
        good = [i for i, o in enumerate(out) if o is not None]
        if not good:
            raise RuntimeError(f"no frames decoded from {path} near t={times[0]:.2f}")
        for i in missing:                                   # fill from nearest decoded neighbour
            out[i] = out[min(good, key=lambda g: abs(g - i))]
    return np.stack(out)
