"""Episodes, labels and feature caches.

An `Episode` holds what the detector needs from one rollout, already encoded:
latents z (T, D) at the detector rate, joint state (T, S), commanded action
(T, A), and the labels: success, shift condition, and optionally the step
where the failure became visible.

Encoding video is the slow part, so `encode_lerobot` writes one .npz per
episode into a cache directory and everything downstream reads the cache.

Labels live in a sidecar file next to the LeRobot dataset,
`meta/shiftbench_episodes.jsonl`, one JSON object per episode:

    {"episode_index": 12, "success": true, "condition": "light_dim",
     "level": 1, "fail_step": null, "notes": ""}

That keeps the stock LeRobot files untouched, so the dataset still loads in
lerobot-dataset-viz and in other people's code. If the sidecar is missing,
labels are read from `success` / `success_class` columns in the LeRobot
episode metadata when they exist (ArmnetBench uses those).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

LABELS_FILE = "meta/shiftbench_episodes.jsonl"


@dataclass
class Episode:
    z: np.ndarray
    state: np.ndarray
    action: np.ndarray
    success: bool
    condition: str = "nominal"
    level: int = 0
    episode_index: int = -1
    fail_step: int | None = None
    fps: float = 4.0
    meta: dict = field(default_factory=dict)
    source: str = ""            # dataset name; with episode_index gives a unique id

    def __len__(self):
        return len(self.z)

    @property
    def uid(self) -> str:
        return f"{self.source}:{self.episode_index}"

    def save(self, path):
        np.savez_compressed(path, z=self.z.astype(np.float16), state=self.state, action=self.action,
                            info=json.dumps({k: v for k, v in asdict(self).items()
                                             if k not in ("z", "state", "action")}))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        info = json.loads(str(d["info"]))
        return cls(z=d["z"].astype(np.float32), state=d["state"], action=d["action"], **info)


def load_cache(cache_dir) -> list[Episode]:
    """All cached episodes, sorted by (source, episode_index). Name sessions
    with the date first (rollout_shiftbench_YYYYMMDD_HHMM_...) and this is
    recording order."""
    eps = [Episode.load(f) for f in Path(cache_dir).glob("ep_*.npz")]
    return sorted(eps, key=lambda e: (e.source, e.episode_index))


# ---------------------------------------------------------------- labels

def read_labels(root) -> dict[int, dict]:
    p = Path(root) / LABELS_FILE
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[int(r["episode_index"])] = r          # last write wins
    return out


def write_label(root, episode_index: int, success: bool, condition: str, level: int = 0,
                fail_step: int | None = None, notes: str = "", **extra):
    p = Path(root) / LABELS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"episode_index": int(episode_index), "success": bool(success), "condition": condition,
           "level": int(level), "fail_step": fail_step, "notes": notes, **extra}
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def labels_from_lerobot_meta(ds) -> dict[int, dict]:
    """Fallback: success columns in LeRobot's own episode metadata."""
    eps = ds.meta.episodes
    if eps is None:
        return {}
    cols = set(eps.column_names)
    out = {}
    for row in eps:
        ei = int(row["episode_index"])
        if "success" in cols:
            out[ei] = {"success": bool(row["success"])}
        elif "success_class" in cols:
            out[ei] = {"success": row["success_class"] == "successful"}
        if ei in out:
            for k in ("policy_type", "policy_repo_id", "task_index"):
                if k in cols:
                    out[ei][k] = row[k]
    return out


# ---------------------------------------------------------------- LeRobot -> cache

def encode_lerobot(repo_id: str, encoder, cache_dir, camera_key: str, root: str | None = None,
                   episodes: list[int] | None = None, target_fps: float = 4.0, labels: dict | None = None,
                   condition_default: str = "nominal", skip_unlabeled: bool = True, verbose: bool = True):
    """Encode episodes of a LeRobot dataset into the cache.

    camera_key: e.g. "observation.images.front"
    target_fps: detector rate; frames are subsampled from the dataset fps.
    The action stored for step t is the policy's command at the end of the
    interval [t, t+stride), i.e. where it was told to go next.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root, episodes=episodes)
    labels = labels if labels is not None else (read_labels(ds.root) or labels_from_lerobot_meta(ds))
    stride = max(1, round(ds.fps / target_fps))
    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    os.makedirs(cache_dir, exist_ok=True)
    source = repo_id.split("/")[-1]
    done = []
    for ei in np.unique(ep_col):
        ei = int(ei)
        out = Path(cache_dir) / f"ep_{source}_{ei:06d}.npz"
        lab = labels.get(ei)
        if lab is None and skip_unlabeled:
            continue
        if out.exists():
            done.append(ei)
            continue
        rel = np.flatnonzero(ep_col == ei)
        pick = rel[::stride]
        frames, states, actions = [], [], []
        for j, r in enumerate(pick):
            item = ds[int(r)]
            frames.append(item[camera_key].numpy())                       # (3,H,W) float [0,1]
            states.append(item["observation.state"].numpy())
            end = rel[min(np.searchsorted(rel, r) + stride - 1, len(rel) - 1)]
            actions.append(ds[int(end)]["action"].numpy() if end != r else item["action"].numpy())
        z = encoder.encode(np.stack(frames))
        lab = lab or {"success": True}
        fail_step = lab.get("fail_step")
        ep = Episode(z=z, state=np.stack(states).astype(np.float32), action=np.stack(actions).astype(np.float32),
                     success=bool(lab["success"]), condition=lab.get("condition", condition_default),
                     level=int(lab.get("level", 0)), episode_index=ei,
                     fail_step=None if fail_step is None else int(fail_step) // stride,
                     fps=ds.fps / stride,
                     meta={k: v for k, v in lab.items() if k in ("policy_type", "policy_repo_id", "task_index", "notes")},
                     source=source)
        ep.save(out)
        done.append(ei)
        if verbose:
            print(f"episode {ei}: {len(z)} steps, success={ep.success}, condition={ep.condition}")
    return done
