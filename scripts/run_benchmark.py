"""Run ShiftBench on encoded episodes and write tables + figures.

  python scripts/run_benchmark.py --cache cache/front --detector runs/detector --out results/real --k 20

Conditions are taken from each episode's `condition` (and `level`, if >0,
as condition_level). Nominal episodes used for training or calibration are
excluded from the nominal test set automatically.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shiftbench.benchmark import evaluate, drift_stream
from shiftbench.data import load_cache
from shiftbench.detector import Detector

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--detector", required=True)
p.add_argument("--out", required=True)
p.add_argument("--k", type=int, default=20, help="reruns used by recal_k; scoring starts after them")
p.add_argument("--n_orders", type=int, default=200)
p.add_argument("--drift_condition", default=None, help="condition for the recovery plot")
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

det = Detector.load(a.detector)
split = json.load(open(Path(a.detector) / "split.json"))
used = set(split["train_episodes"]) | set(split["cal_episodes"])
eps = load_cache(a.cache)                     # recording order
cal = [e for e in eps if e.uid in set(split["cal_episodes"])]

groups = {}
for e in eps:
    if e.uid in used:
        continue
    key = e.condition if not e.level else f"{e.condition}_{e.level}"
    groups.setdefault(key, []).append(e)
groups = {k: v for k, v in groups.items() if len(v) > a.k}
print({k: (sum(e.success for e in v), sum(not e.success for e in v)) for k, v in groups.items()},
      "(success, fail) per condition")

df = evaluate(det, groups, nominal_cal=cal, k=a.k, n_orders=a.n_orders)
df.to_csv(out / "benchmark.csv", index=False)
cols = [m for m in ["fixed", "self_ref", "recal_k", "adaptive", "adaptive_log", "adaptive_log_fast"] if m in set(df.method)]
for val, title in (("false_alarm", "FALSE-ALARM RATE on successful episodes (target %.2f)" % det.cfg.alpha),
                   ("detection", "DETECTION RATE on failed episodes")):
    t = df.pivot_table(index="condition", columns="method", values=val, dropna=False).reindex(columns=cols)
    print("\n" + title)
    print(t.round(3).to_string())
    t.to_csv(out / f"{val}.csv")
print("\nAUROC")
print(df.groupby("condition")[["auroc_raw", "auroc_ref"]].first().round(3).to_string())

# figure: false-alarm rate by condition and method, with Wilson CIs where available
t = df.pivot_table(index="condition", columns="method", values="false_alarm", dropna=False).reindex(columns=cols)
fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(t), 3.6))
w = 0.8 / len(cols)
x = np.arange(len(t))
for i, m in enumerate(cols):
    ax.bar(x + i * w - 0.4 + w / 2, t[m].values, w, label=m)
ax.axhline(det.cfg.alpha, color="k", ls=":", lw=1, label="target")
ax.set_xticks(x, t.index, rotation=30, ha="right")
ax.set_ylabel("false-alarm rate")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(out / "false_alarm_by_condition.png", dpi=150)

if a.drift_condition and a.drift_condition in groups:
    nominal_test = groups.get("nominal", [])
    ds = drift_stream(det, nominal_test, groups[a.drift_condition])
    ds.to_csv(out / f"drift_{a.drift_condition}.csv", index=False)
print("\nwrote", out)
