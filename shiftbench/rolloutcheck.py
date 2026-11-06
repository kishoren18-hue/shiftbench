"""RolloutCheck (early core): flag suspicious episodes in any LeRobot dataset.

    rolloutcheck --repo_id someone/their_so101_dataset --detector runs/detector \
        --camera observation.images.front --out reports/their_dataset

Encodes every episode (labels optional), scores it with a trained ShiftBench
detector, and writes report.html + scores.csv: which episodes crossed the
cutoff, when, and how far.

Read the flags as "the world model was surprised", not "the robot failed".
A detector trained on your cell will be surprised by someone else's camera
and lighting; that is the very shift ShiftBench measures. The web app with
uploads comes later, once the detector works on your own data.
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .data import encode_lerobot, load_cache
from .detector import Detector
from .encoders import make_encoder
from .scores import first_alarm


def scan(cache_dir, det: Detector) -> pd.DataFrame:
    rows, traces = [], {}
    for e in load_cache(cache_dir):
        tr = det.trace(e)
        fa = first_alarm(tr, det.tau)
        rows.append({"episode": e.episode_index, "steps": len(e), "max_score": float(np.max(tr)) if len(tr) else np.nan,
                     "cutoff": det.tau, "flagged": fa is not None,
                     "first_flag_s": None if fa is None else round((fa + det.cfg.context + det.cfg.horizon - 1) / e.fps, 2),
                     "label": None})
        traces[e.episode_index] = (tr, e.fps)
    return pd.DataFrame(rows).sort_values("max_score", ascending=False), traces


def html_report(df, traces, det, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def fig_b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    imgs = []
    for _, r in df.head(12).iterrows():
        tr, fps = traces[r.episode]
        fig, ax = plt.subplots(figsize=(4, 1.8))
        ax.plot(np.arange(len(tr)) / fps, tr, color="#c0392b" if r.flagged else "#2c3e50")
        ax.axhline(det.tau, ls="--", color="gray", lw=0.8)
        ax.set_title(f"episode {r.episode}", fontsize=8)
        ax.set_xlabel("seconds", fontsize=7)
        ax.tick_params(labelsize=6)
        imgs.append(f'<img src="data:image/png;base64,{fig_b64(fig)}">')
    n, k = len(df), int(df.flagged.sum())
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:2em auto;padding:0 1em;color:#222}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border-bottom:1px solid #ddd;padding:3px 8px;text-align:right}}
img{{margin:4px}}</style></head><body>
<h1>RolloutCheck: {title}</h1>
<p><b>{k}</b> of {n} episodes crossed the cutoff ({det.tau:.4f}, calibrated for a {det.cfg.alpha:.0%} false-alarm rate on
the detector's own setup). A flag means the world model was surprised, not that the episode failed.</p>
<h2>Most surprising episodes</h2>{''.join(imgs)}
<h2>All episodes</h2>{df.to_html(index=False, float_format=lambda x: f"{x:.4f}")}
</body></html>"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--detector", required=True)
    p.add_argument("--camera", default="observation.images.front")
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", default="vjepa2", choices=["vjepa2", "tiny"])
    p.add_argument("--max_episodes", type=int, default=None)
    a = p.parse_args(argv)

    det = Detector.load(a.detector)
    out = Path(a.out)
    cache = out / "cache"
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(a.repo_id, root=a.root)
    eps = [int(r["episode_index"]) for r in meta.episodes][: a.max_episodes]
    labels = {ei: {"success": True, "condition": "unknown", "notes": "unlabeled"} for ei in eps}
    encode_lerobot(a.repo_id, make_encoder(a.encoder), cache, a.camera, root=a.root, episodes=eps,
                   labels=labels, verbose=False)
    df, traces = scan(cache, det)
    df["label"] = None
    df.to_csv(out / "scores.csv", index=False)
    (out / "report.html").write_text(html_report(df, traces, det, a.repo_id))
    print(f"{int(df.flagged.sum())}/{len(df)} flagged -> {out / 'report.html'}")


if __name__ == "__main__":
    main()
