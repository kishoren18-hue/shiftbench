"""Train the world model on nominal successes and calibrate the alarm.

  python scripts/train_detector.py --cache cache/front --out runs/detector \
      --train_condition nominal --alpha 0.05

Nominal successful episodes are split in recording order: the first
`--train_frac` train the predictor, the rest calibrate the cutoff. Keeping
them disjoint is what makes the conformal guarantee valid. You need at least
1/alpha - 1 calibration successes (19 at 5%, 9 at 10%).
"""
import argparse
import json

import numpy as np

from shiftbench.calibrate import min_calibration_size
from shiftbench.data import load_cache
from shiftbench.detector import Detector, DetectorConfig

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--out", required=True)
p.add_argument("--train_condition", default="nominal")
p.add_argument("--train_frac", type=float, default=0.5)
p.add_argument("--alpha", type=float, default=0.05)
p.add_argument("--window", type=int, default=4)
p.add_argument("--context", type=int, default=2)
p.add_argument("--horizon", type=int, default=1)
p.add_argument("--hidden", type=int, default=512)
p.add_argument("--layers", type=int, default=3)
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

eps = load_cache(a.cache)                     # recording order
nom = [e for e in eps if e.condition == a.train_condition and e.success]
if not nom:
    raise SystemExit(f"no successful episodes with condition {a.train_condition!r}; "
                     f"conditions present: {sorted({e.condition for e in eps})}")
n_tr = int(len(nom) * a.train_frac)
train, cal = nom[:n_tr], nom[n_tr:]
need = min_calibration_size(a.alpha)
print(f"{len(nom)} nominal successes -> {len(train)} train, {len(cal)} calibration (need >= {need})")
if len(cal) < need:
    print(f"WARNING: too few calibration successes for alpha={a.alpha}; the cutoff will be +inf (never alarm)")

cfg = DetectorConfig(alpha=a.alpha, window=a.window, context=a.context, horizon=a.horizon,
                     hidden=a.hidden, layers=a.layers, epochs=a.epochs)
det = Detector(cfg).fit(train, cal, seed=a.seed)
det.save(a.out)
json.dump({"train_episodes": [e.uid for e in train], "cal_episodes": [e.uid for e in cal],
           "tau": det.tau}, open(f"{a.out}/split.json", "w"), indent=2)
print(f"cutoff tau = {det.tau:.4f}   (median calibration score {np.median(det.cal_scores):.4f})")
print("saved ->", a.out)
