"""Run the whole ShiftBench pipeline on the toy latent world.

Checks that the code runs end to end and that each method behaves as the
theory says. These are NOT results about robots. Do not report them.

    python scripts/synthetic_experiment.py --out results/synthetic
"""
import argparse
import json
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shiftbench.benchmark import evaluate, drift_stream
from shiftbench.detector import Detector, DetectorConfig
from shiftbench.synthetic import World

p = argparse.ArgumentParser()
p.add_argument("--out", default="results/synthetic")
p.add_argument("--alpha", type=float, default=0.05)
p.add_argument("--k", type=int, default=20)
p.add_argument("--n_shift", type=int, default=100)
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

w = World(seed=a.seed)
train = w.dataset({("nominal", 0): 60}, seed=a.seed + 1)          # ~50 demos' worth of successes
cal = w.dataset({("nominal", 0): 100}, seed=a.seed + 2)
test = {"nominal": w.dataset({("nominal", 0): a.n_shift}, seed=a.seed + 3)}
for shift in ("camera", "lighting", "table", "objects"):
    for level in (1, 2):
        test[f"{shift}_{level}"] = w.dataset({(shift, level): a.n_shift}, seed=a.seed + 10 + zlib.crc32(f"{shift}{level}".encode()) % 1000)

det = Detector(DetectorConfig(alpha=a.alpha, epochs=a.epochs, hidden=256, layers=2)).fit(train, cal, verbose=True)
print(f"conformal cutoff from {int(sum(e.success for e in cal))} nominal successes: {det.tau:.4f}")
det.save(out / "detector")

df = evaluate(det, test, k=a.k, n_orders=100, seed=a.seed, nominal_cal=cal)
df.to_csv(out / "benchmark.csv", index=False)
piv = df.pivot_table(index="condition", columns="method", values="false_alarm").reindex(list(test))
pivd = df.pivot_table(index="condition", columns="method", values="detection").reindex(list(test))
cols = ["fixed", "self_ref", "recal_k", "adaptive", "adaptive_log", "adaptive_log_fast"]
piv, pivd = piv[cols], pivd[cols]
print("\nFALSE-ALARM RATE on successful episodes (target %.2f)" % a.alpha)
print(piv.round(3).to_string())
print("\nDETECTION RATE on failed episodes")
print(pivd.round(3).to_string())
ex = df[df.method.str.startswith("adaptive") | (df.method == "recal_k")].pivot_table(
    index="condition", columns="method", values=["false_alarm_p90_orders"]).reindex(list(test))
print("\n90th-percentile false-alarm rate over random episode orders")
print(ex.round(3).to_string())
rec = df[df.method.str.startswith("adaptive")].pivot_table(index="condition", columns="method",
      values="successes_to_recover").reindex(list(test))
print("\nSuccessful episodes until rolling false-alarm rate <= 2*alpha (recorded order)")
print(rec.to_string())
print("\nrecal_k: share of orders where too few of the k reruns succeeded, so it never alarms")
print(df[df.method == "recal_k"].set_index("condition")["never_alarms_share"].reindex(list(test)).round(2).to_string())
au = df.groupby("condition")[["auroc_raw", "auroc_ref"]].first().reindex(list(test))
print("\nAUROC (ranking quality, cutoff-free)")
print(au.round(3).to_string())

# recovery plot: 100 nominal then 150 camera_2
ds = drift_stream(det, test["nominal"], w.dataset({("lighting", 1): 150}, seed=99))
ds.to_csv(out / "drift_stream.csv", index=False)
fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
c = np.where(ds.success, "tab:blue", "tab:red")
ax[0].scatter(ds.k, ds.score, c=c, s=8)
ax[0].plot(ds.k, ds.tau_fixed, "k--", label="fixed cutoff")
ax[0].plot(ds.k, ds.tau_adaptive, "tab:green", label="adaptive cutoff")
ax[0].axvline(len(test["nominal"]) - 0.5, color="gray", lw=0.8)
ax[0].set_ylabel("episode score")
ax[0].legend(loc="upper left", fontsize=8)
ax[0].set_title("SYNTHETIC: lamp changed at the gray line (blue = success, red = failure)", fontsize=9)
succ = ds[ds.success]
for col, lab, kw in (("alarm_fixed", "fixed", dict(color="k", ls="--")),
                     ("alarm_adaptive", "adaptive (log scale)", dict(color="tab:green"))):
    run = succ[col].astype(float).rolling(25, min_periods=5).mean()
    ax[1].plot(succ.k, run, label=lab, **kw)
ax[1].axhline(a.alpha, color="gray", ls=":", label=f"target {a.alpha:.0%}")
ax[1].set_ylabel("false-alarm rate\n(last 25 successes)")
ax[1].set_xlabel("episode")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "synthetic_drift.png", dpi=150)
json.dump({"tau": det.tau, "alpha": a.alpha, "note": "synthetic toy world; not robot results"},
          open(out / "summary.json", "w"), indent=2)
print(f"\nwrote {out}/benchmark.csv, drift_stream.csv, synthetic_drift.png")
