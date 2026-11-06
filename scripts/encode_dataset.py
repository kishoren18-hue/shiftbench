"""Encode a LeRobot dataset's episodes into the ShiftBench feature cache.

Run on a GPU (Colab). Examples:

  # your own recordings (labels from meta/shiftbench_episodes.jsonl)
  python scripts/encode_dataset.py --repo_id $HF_USER/shiftbench_so101 --camera observation.images.front \
      --cache cache/front

  # ArmnetBench pilot: one task, 40 episodes per policy, condition = policy_type
  python scripts/encode_dataset.py --repo_id armnet/armnetbench_v01_lerobot_so101 \
      --camera observation.images.front --cache cache/armnet_block_stack \
      --task "Stack the colouful blocks" --condition_column policy_type --max_per_condition 40

Only the selected episodes' files are downloaded.
"""
import argparse
import json

import numpy as np

from shiftbench.data import encode_lerobot, read_labels
from shiftbench.encoders import make_encoder

p = argparse.ArgumentParser()
p.add_argument("--repo_id", required=True)
p.add_argument("--root", default=None, help="local dataset folder (skip download)")
p.add_argument("--camera", default="observation.images.front")
p.add_argument("--cache", required=True)
p.add_argument("--fps", type=float, default=4.0, help="detector rate; V-JEPA 2-AC used 4 fps")
p.add_argument("--encoder", default="vjepa2", choices=["vjepa2", "tiny"])
p.add_argument("--model_id", default="facebook/vjepa2-vitl-fpc64-256")
p.add_argument("--pool", default="mean", choices=["mean", "grid2"])
p.add_argument("--episodes", default=None, help="comma-separated episode indices")
p.add_argument("--task", default=None, help="keep episodes whose task text contains this")
p.add_argument("--condition_column", default=None,
               help="episode-metadata column to use as the condition label (e.g. policy_type)")
p.add_argument("--max_per_condition", type=int, default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--all_success", default=None, metavar="CONDITION",
               help="treat every episode as a success with this condition (for teleop demo datasets)")
a = p.parse_args()

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

meta = LeRobotDatasetMetadata(a.repo_id, root=a.root)
eps = meta.episodes
cols = set(eps.column_names)
print("episode metadata columns:", sorted(c for c in cols if not c.startswith(("stats/", "data/", "videos/", "meta/"))))

side = read_labels(meta.root)
labels = {}
for row in eps:
    ei = int(row["episode_index"])
    if a.task and not any(a.task.lower() in t.lower() for t in (row.get("tasks") or [])):
        continue
    if a.all_success:
        lab = {"success": True, "condition": a.all_success}
    elif ei in side:
        if side[ei].get("skip"):
            continue
        lab = dict(side[ei])
    elif "success" in cols:
        lab = {"success": bool(row["success"])}
    elif "success_class" in cols:
        if row["success_class"] == "suboptimal":
            continue                                   # ambiguous; leave out
        lab = {"success": row["success_class"] == "successful"}
    else:
        continue
    if a.condition_column and not a.all_success:
        lab["condition"] = str(row[a.condition_column] or "unknown")
    for k in ("policy_type", "policy_repo_id"):
        if k in cols:
            lab[k] = row[k]
    labels[ei] = lab

if a.episodes:
    keep = {int(x) for x in a.episodes.split(",")}
    labels = {k: v for k, v in labels.items() if k in keep}
if a.max_per_condition:
    rng = np.random.default_rng(a.seed)
    by = {}
    for ei, lab in labels.items():
        by.setdefault(lab.get("condition", "nominal"), []).append(ei)
    keep = set()
    for c, lst in by.items():
        keep |= set(rng.permutation(lst)[: a.max_per_condition].tolist())
    labels = {k: v for k, v in labels.items() if k in keep}

summary = {}
for lab in labels.values():
    c = lab.get("condition", "nominal")
    s = summary.setdefault(c, [0, 0])
    s[0 if lab["success"] else 1] += 1
print("selected (success, fail) per condition:", json.dumps(summary))
if not labels:
    raise SystemExit("no labeled episodes selected")

kw = dict(model_id=a.model_id, pool=a.pool) if a.encoder == "vjepa2" else {}
enc = make_encoder(a.encoder, **kw)
encode_lerobot(a.repo_id, enc, a.cache, a.camera, root=a.root, episodes=sorted(labels), target_fps=a.fps,
               labels=labels)
print("done ->", a.cache)
