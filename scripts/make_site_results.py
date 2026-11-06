"""Turn an offline-suite output folder into site/results.json + figures.

    python scripts/make_site_results.py --out_dir results/armnet_eyedrops --site site \
        --video_base https://<store>.public.blob.vercel-storage.com/eyedrops_v1/out
"""
import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--out_dir", required=True)
p.add_argument("--site", default="site")
p.add_argument("--video_base", default=None)
p.add_argument("--alpha", type=float, default=0.1)
a = p.parse_args()
od, site = Path(a.out_dir), Path(a.site)
NAMES = {"vjepa2": "V-JEPA 2 (video)", "dinov2_s": "DINOv2-S", "siglip2_b": "SigLIP 2-B", "clip_b32": "CLIP B/32", "resnet18": "ResNet-18"}
SHIFT = {"nominal": "no shift", "light_dim_2": "lamp dimmed", "cam_bump_1": "camera bump (small)",
         "cam_bump_2": "camera bump (large)", "occlusion": "object in view"}


def load(exp):
    fs = sorted(glob.glob(str(od / f"table_{exp}_*_*.csv")))
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else pd.DataFrame()


B, A = load("B"), load("A")
encs = [e for e in NAMES if e in set(B.encoder)] if len(B) else []
tables, headline = [], []

if len(B):
    full = B[B.variant == "full"]
    # main table: fixed vs adaptive false-alarm rate, per encoder x shift
    for meth, title in (("fixed", "Fixed cutoff (published-style): false-alarm rate on successful runs"),
                        ("adaptive_log", "Adaptive cutoff: false-alarm rate on successful runs")):
        rows = []
        for e in encs:
            d = full[(full.encoder == e) & (full.method == meth)].set_index("condition")
            rows.append([NAMES[e]] + [float(d.false_alarm.get(s, np.nan)) for s in SHIFT])
        tables.append({"title": title, "columns": ["vision model"] + list(SHIFT.values()), "rows": rows,
                       "pct": True, "flag": 2 * a.alpha,
                       "note": f"Target α = {a.alpha:.0%}. Red = more than twice the target. Same {int(full.n_eval_success.iloc[0])} successful test runs in every column (image shifts applied to identical episodes)."})
    rows = []
    for e in encs:
        d = full[(full.encoder == e) & (full.method == "fixed")].set_index("condition")
        rows.append([NAMES[e]] + [float(d.auroc_raw.get(s, np.nan)) for s in SHIFT])
    tables.append({"title": "Can it tell failures from successes? (AUROC, 0.5 = chance)", "columns": ["vision model"] + list(SHIFT.values()),
                   "rows": rows, "note": "Cutoff-free ranking quality of the episode score."})
    rows = []
    for e in encs:
        r = [NAMES[e]]
        for v in ("full", "noaction", "nochange"):
            d = B[(B.encoder == e) & (B.variant == v) & (B.method == "fixed") & (B.condition == "nominal")]
            r.append(float(d.auroc_raw.iloc[0]) if len(d) else np.nan)
        rows.append(r)
    tables.append({"title": "Does the action-conditioned world model matter? (AUROC, no shift)",
                   "columns": ["vision model", "world model (action-conditioned)", "same, actions zeroed", "no learning: frame-to-frame change"], "rows": rows,
                   "note": "If the last column is as good, the 'world model' isn't adding anything on this data. Reported either way."})
    # headline: worst fixed false alarm vs adaptive, on the flagship encoder (or the first)
    hero = "vjepa2" if "vjepa2" in encs else encs[0]
    fx = full[(full.encoder == hero) & (full.method == "fixed") & (full.condition != "nominal")]
    ad = full[(full.encoder == hero) & (full.method == "adaptive_log") & (full.condition != "nominal")]
    if len(fx):
        w = fx.loc[fx.false_alarm.idxmax()]
        headline = [
            {"value": f"{w.false_alarm:.0%}", "label": f"false alarms, fixed cutoff, {SHIFT.get(w.condition, w.condition)} ({NAMES[hero]})"},
            {"value": f"{ad.set_index('condition').false_alarm.get(w.condition, np.nan):.0%}", "label": "same shift, adaptive cutoff"},
            {"value": f"{a.alpha:.0%}", "label": "promised rate"},
        ]
    # figure
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor="#161b22")
    ax.set_facecolor("#161b22")
    sh = [s for s in SHIFT if s != "nominal"]
    x = np.arange(len(sh))
    for i, e in enumerate(encs):
        d = full[(full.encoder == e)]
        f = d[d.method == "fixed"].set_index("condition").false_alarm.reindex(sh).values
        g = d[d.method == "adaptive_log"].set_index("condition").false_alarm.reindex(sh).values
        off = (i - (len(encs) - 1) / 2) * 0.16
        ax.scatter(x + off, f, marker="o", s=36, color=plt.cm.tab10(i), label=NAMES[e])
        ax.scatter(x + off, g, marker="^", s=36, color=plt.cm.tab10(i), alpha=0.6)
        ax.vlines(x + off, g, f, color=plt.cm.tab10(i), alpha=0.35)
    ax.axhline(a.alpha, color="#8b949e", ls=":", lw=1)
    ax.set_xticks(x, [SHIFT[s] for s in sh], color="#e6edf3")
    ax.tick_params(colors="#8b949e")
    ax.set_ylabel("false-alarm rate", color="#e6edf3")
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.legend(fontsize=7, ncol=3, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")
    ax.set_title("● fixed cutoff   ▲ adaptive cutoff   (dotted: promised rate)", color="#e6edf3", fontsize=10)
    fig.tight_layout()
    fig.savefig(od / "fig_false_alarm.png", dpi=150, facecolor=fig.get_facecolor())

if len(A):
    fa = A[(A.variant == "full") & (A.method.isin(["fixed", "adaptive_log"]))]
    conds = sorted(set(fa.condition))
    rows = []
    for e in encs:
        for m in ("fixed", "adaptive_log"):
            d = fa[(fa.encoder == e) & (fa.method == m)].set_index("condition")
            rows.append([f"{NAMES[e]} · {'fixed' if m == 'fixed' else 'adaptive'}"] + [float(d.false_alarm.get(c, np.nan)) for c in conds])
    tables.append({"title": "Policy shift: calibrated on human teleop, tested on each robot policy",
                   "columns": ["model · cutoff"] + conds, "rows": rows, "pct": True, "flag": 2 * a.alpha,
                   "note": "A different 'driver' is a shift too. Small samples per policy: read these as rough."})

vids = []
if a.video_base and (od / "demo").exists():
    for f in sorted(os.listdir(od / "demo")):
        if f.endswith(".mp4"):
            vids.append({"src": f"{a.video_base}/{f}", "caption": ("Camera bumped at 50% · " if "bump" in f else "Same episode, no bump (control) · ") + f.replace(".mp4", "")})
res = {"headline": headline, "tables": tables, "videos": vids,
       "figures": [],
       "footer": "Full tables, per-episode scores and logs: <a href='https://github.com/kishoren18-hue/shiftbench/tree/main/results/armnet_eyedrops'>results/armnet_eyedrops</a>. Run on CPU in Vercel sandboxes, Nov 2026."}
(site / "results.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(headline, indent=1))
