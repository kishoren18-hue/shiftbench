"""Label rollouts right after recording them.

  shiftbench-label --root ~/.cache/huggingface/lerobot/$HF_USER/shiftbench_so101 \
      --condition light_dim --level 1

Asks about every episode that has no label yet: s = success, f = fail,
x = skip (bad recording), q = quit. For failures it asks for the second at
which the failure became visible (optional; press Enter to skip).

Write the success rule down BEFORE recording and apply it the same way every
time (see docs/PROTOCOL.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import read_labels, write_label


def episode_indices(root) -> list[int]:
    try:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        meta = LeRobotDatasetMetadata(repo_id=Path(root).name, root=root)
        return [int(r["episode_index"]) for r in meta.episodes]
    except Exception:
        info = json.loads((Path(root) / "meta/info.json").read_text())
        return list(range(int(info["total_episodes"])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="local LeRobot dataset folder")
    p.add_argument("--condition", required=True, help="e.g. nominal, light_dim, cam_left_10, table_wood, obj_mug")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0, help="dataset fps, to convert seconds to frames")
    p.add_argument("--from_episode", type=int, default=0)
    p.add_argument("--relabel", action="store_true")
    a = p.parse_args(argv)

    done = read_labels(a.root)
    todo = [i for i in episode_indices(a.root) if i >= a.from_episode and (a.relabel or i not in done)]
    print(f"{len(todo)} episodes to label as condition={a.condition} level={a.level}")
    for ei in todo:
        while True:
            r = input(f"episode {ei}: [s]uccess / [f]ail / [x] skip / [q]uit > ").strip().lower()
            if r in ("s", "f", "x", "q"):
                break
        if r == "q":
            break
        if r == "x":
            write_label(a.root, ei, False, a.condition, a.level, notes="skip", skip=True)
            continue
        fail_step = None
        if r == "f":
            t = input("  second when the failure became visible (Enter to skip) > ").strip()
            if t:
                fail_step = int(round(float(t) * a.fps))
        notes = input("  notes (Enter to skip) > ").strip()
        write_label(a.root, ei, r == "s", a.condition, a.level, fail_step=fail_step, notes=notes)
    print("labels ->", Path(a.root) / "meta/shiftbench_episodes.jsonl")


if __name__ == "__main__":
    main()
