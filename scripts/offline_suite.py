"""ShiftBench offline suite on real SO-101 video (ArmnetBench), before any hardware.

Two experiments, five vision models, three world-model variants:

  A  POLICY SHIFT. World model trained and calibrated on human teleop demos;
     tested on rollouts of each of ArmnetBench's policies (ACT, Diffusion,
     SmolVLA, pi0, pi0.5, GR00T, MolmoAct 2). Does the promise survive a
     different driver?
  B  IMAGE SHIFT (paired). Calibrated on half of the policy rollouts' successes;
     the other half is tested unchanged and under each image-level shift
     (dim / warm light, camera bump, occlusion, blur; shiftbench/perturb.py).
     The same episodes appear in every condition, so differences are the shift.

World-model variants per encoder:
  full      action-conditioned predictor (the method)
  noaction  same network with state and action zeroed: do actions matter?
  nochange  no learning: surprise = size of the latent change from the last frame

Output (to --out, e.g. a Google Drive folder):
  traces.parquet       per (encoder, variant, shift, episode) error trace + labels
  episodes.csv         which episodes were used where
  table_A_*.csv / table_B_*.csv, summary.json, run_log.txt

Run on a GPU (Colab T4: roughly 1-2 h for one task with all five encoders).
Re-running skips episodes already encoded (cache in --cache).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from shiftbench.benchmark import evaluate_scores
from shiftbench.data import Episode
from shiftbench.detector import Detector, DetectorConfig
from shiftbench.encoders import make_encoder
from shiftbench.lerobot_v3 import V3Dataset
from shiftbench.perturb import SHIFTS, apply
from shiftbench.scores import score_trace, episode_score
from shiftbench.refscores import ReferenceBank

p = argparse.ArgumentParser()
p.add_argument("--repo_id", default="armnet/armnetbench_v01_lerobot_so101")
p.add_argument("--data_dir", default="/content/armnet")
p.add_argument("--task", default="Stack the colouful blocks")
p.add_argument("--camera", default="observation.images.front")
p.add_argument("--encoders", default="vjepa2,dinov2_b,siglip2_b,clip_b32,resnet18")
p.add_argument("--shifts", default=",".join(SHIFTS))
p.add_argument("--max_per_policy", type=int, default=40)
p.add_argument("--max_teleop", type=int, default=60)
p.add_argument("--fps", type=float, default=4.0)
p.add_argument("--alpha", type=float, default=0.1)
p.add_argument("--k", type=int, default=20)
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--cache", default="/content/sb_cache")
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--dry_run", action="store_true", help="select episodes and print counts only")
p.add_argument("--max_test_B", type=int, default=None, help="cap on experiment-B test episodes (random, seeded)")
p.add_argument("--shard", default=None, help="i/n: encode only every n-th selected episode starting at i, then stop")
p.add_argument("--encode_only", action="store_true", help="stop after encoding (use with --shard)")
p.add_argument("--variants", default="full,noaction,nochange,trajdev,endstate",
               help="score variants: full/noaction/nochange (world model), trajdev/endstate (reference bank)")
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
cache = Path(a.cache)
cache.mkdir(parents=True, exist_ok=True)
log_f = open(out / "run_log.txt", "a")


def log(*x):
    msg = " ".join(str(v) for v in x)
    print(msg, flush=True)
    log_f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    log_f.flush()


rng = np.random.default_rng(a.seed)
encs = a.encoders.split(",")
shifts = a.shifts.split(",")
assert shifts[0] == "nominal"

# ------------------------------------------------------------------ select episodes
LOCAL = a.repo_id.startswith("local:")             # local:<path> skips downloads (for testing)
if LOCAL:
    a.data_dir = a.repo_id.split(":", 1)[1]
meta = V3Dataset(a.data_dir) if LOCAL else V3Dataset.download(a.repo_id, a.data_dir)
lab = meta.labels()
lab = lab[lab.task.str.lower().str.contains(a.task.lower())]
lab = lab[lab.success.notna()]
log("task:", a.task, "| episodes with success/failure labels:", len(lab))
log(lab.groupby("policy_type").success.agg(["count", "sum"]).rename(columns={"sum": "successes"}).to_string())

tele = lab[(lab.policy_type == "teleoperated") & lab.success]
tele = tele.sample(min(len(tele), a.max_teleop), random_state=a.seed).sort_values("episode_index")
pol = []
for pt, g in lab[lab.policy_type != "teleoperated"].groupby("policy_type"):
    pol.append(g.sample(min(len(g), a.max_per_policy), random_state=a.seed))
pol = pd.concat(pol).sort_values("episode_index") if pol else lab.iloc[:0]
if len(tele) < 30:
    raise SystemExit(f"only {len(tele)} successful teleop episodes for this task; need >= 30 (train + calibrate)")

n_tr = len(tele) // 2
tele_train, tele_cal = tele.iloc[:n_tr], tele.iloc[n_tr:]
pol_succ = pol[pol.success]
calB = pol_succ.sample(frac=0.5, random_state=a.seed)
testB = pol.drop(calB.index).sort_values("episode_index")
if a.max_test_B and len(testB) > a.max_test_B:
    testB = testB.sample(a.max_test_B, random_state=a.seed).sort_values("episode_index")
    pol = pd.concat([calB, testB]).sort_values("episode_index")     # unused rollouts are dropped everywhere
use = pd.concat([tele_train.assign(role="teleop_train"), tele_cal.assign(role="teleop_cal"),
                 pol.assign(role=np.where(pol.index.isin(calB.index), "policy_calB", "policy_testB"))])
use.to_csv(out / "episodes.csv", index=False)
log(f"teleop: {len(tele_train)} train / {len(tele_cal)} cal | policy rollouts: {len(pol)} "
    f"({int(pol.success.sum())} successes) | B: {len(calB)} cal successes, {len(testB)} test")
if a.dry_run:
    raise SystemExit(0)

# ------------------------------------------------------------------ download + encode
shifted_eps = set(testB.episode_index)            # image shifts only where they are tested
if a.shard:
    i_sh, n_sh = (int(v) for v in a.shard.split("/"))
    use_enc = use.iloc[i_sh::n_sh]
    log(f"shard {i_sh}/{n_sh}: {len(use_enc)} episodes")
else:
    use_enc = use


def cpath(enc, shift, ei):
    return cache / enc / shift / f"ep_{ei:06d}.npz"


need = [int(r.episode_index) for r in use_enc.itertuples()
        if any(not cpath(e, s, int(r.episode_index)).exists() for e in encs
               for s in (shifts if int(r.episode_index) in shifted_eps else ["nominal"]))]
log(f"{len(need)} episodes still to encode")
ds = meta if (LOCAL or not need) else V3Dataset.download(a.repo_id, a.data_dir, episodes=need, cameras=(a.camera,))
encoders = {}
for e in (encs if need else []):
    t0 = time.time()
    encoders[e] = make_encoder(e)
    log(f"loaded {e} in {time.time() - t0:.0f}s")

t_start = time.time()
for n, row in enumerate(use_enc.itertuples()):
    ei = int(row.episode_index)
    todo = [(e, s) for e in encs for s in (shifts if ei in shifted_eps else ["nominal"])
            if not cpath(e, s, ei).exists()]
    if not todo:
        continue
    ep = ds.episode(ei, a.camera, target_fps=a.fps)
    for s in sorted({s for _, s in todo}):
        fr = ep["frames"] if s == "nominal" else apply(ep["frames"], s)
        for e in [e for e, s2 in todo if s2 == s]:
            z = encoders[e].encode(fr)
            cpath(e, s, ei).parent.mkdir(parents=True, exist_ok=True)
            Episode(z=z, state=ep["state"], action=ep["action"], success=bool(row.success), condition=s,
                    episode_index=ei, fps=ep["fps"], meta={"policy_type": row.policy_type, "role": row.role},
                    source="armnet").save(cpath(e, s, ei))
    if n % 10 == 0:
        el = time.time() - t_start
        log(f"encoded {n + 1}/{len(use_enc)} episodes ({el / 60:.1f} min)")
del encoders
log(f"encoding finished in {(time.time() - t_start) / 60:.1f} min")
if a.encode_only:
    raise SystemExit(0)

# ------------------------------------------------------------------ train, score, evaluate
REF = 4
rows, summary = [], {}
for e in encs:
    def load(ei, s="nominal"):
        return Episode.load(cpath(e, s, ei))

    tr = [load(i) for i in tele_train.episode_index]
    cal = [load(i) for i in tele_cal.episode_index]
    for variant in a.variants.split(","):
        calB_v = calB                                     # calibration set for experiment B
        def prep(x):
            if variant == "noaction":
                return Episode(z=x.z, state=np.zeros_like(x.state), action=np.zeros_like(x.action),
                               success=x.success, condition=x.condition, episode_index=x.episode_index, fps=x.fps, meta=x.meta)
            return x

        errsB = None
        if variant == "nochange":
            mu_sd = np.concatenate([x.z for x in tr]).std(0) + 1e-6
            errs = lambda x: np.abs(np.diff(x.z, axis=0) / mu_sd).mean(1)
        elif variant in ("trajdev", "endstate"):
            # A: bank = teleop train demos. B: bank also gets half the successful policy runs,
            # the other half calibrates (bank and calibration never overlap).
            bankA = ReferenceBank(tr)
            calB_v = calB.iloc[1::2]
            bankB = ReferenceBank(tr + [load(i) for i in calB.iloc[0::2].episode_index])
            if variant == "trajdev":
                errs = lambda x, b=bankA: b.trajdev(x)
                errsB = lambda x, b=bankB: b.trajdev(x)
            else:
                errs = lambda x, b=bankA: np.full(max(len(x.z), 1), b.endstate(x))
                errsB = lambda x, b=bankB: np.full(max(len(x.z), 1), b.endstate(x))
        else:
            det = Detector(DetectorConfig(alpha=a.alpha, epochs=a.epochs)).fit([prep(x) for x in tr], [prep(x) for x in cal],
                                                                              verbose=False, seed=a.seed)
            errs = lambda x, det=det: det.wm.errors(prep(x))
            if variant == "full":
                det.save(out / f"detector_{e}")               # teleop-calibrated, used by replay_demo.py

        def scores(x, f=None):
            er = (f or errs)(x)
            return episode_score(score_trace(er, 4, None)), episode_score(score_trace(er, 4, REF)), er
        errsB = errsB or errs

        cal_s = [scores(x) for x in cal]
        # ---- experiment A: policy shift (image nominal)
        condA = {}
        for pt, g in pol.groupby("policy_type"):
            S = [scores(load(i)) for i in g.sort_values("episode_index").episode_index]
            condA[pt] = (np.array([s[0] for s in S]), np.array([s[1] for s in S]), g.sort_values("episode_index").success.values)
            for (r0, r1, er), (_, rw) in zip(S, g.sort_values("episode_index").iterrows()):
                rows.append({"encoder": e, "variant": variant, "experiment": "A", "shift": "nominal",
                             "condition": pt, "episode_index": int(rw.episode_index), "success": bool(rw.success),
                             "score": r0, "score_ref": r1, "errors": er.astype(np.float32).tolist()})
        dfA = evaluate_scores({k_: v for k_, v in condA.items() if len(v[2]) > a.k},
                              [s[0] for s in cal_s], [s[1] for s in cal_s], a.alpha, k=a.k, n_orders=100, seed=a.seed)
        dfA.insert(0, "variant", variant)
        dfA.insert(0, "encoder", e)
        dfA.to_csv(out / f"table_A_{e}_{variant}.csv", index=False)
        # ---- experiment B: image shifts, paired episodes, calibrated on policy successes
        calB_s = [scores(load(i), errsB) for i in calB_v.episode_index]
        condB = {}
        for s in shifts:
            S = [scores(load(i, s), errsB) for i in testB.episode_index]
            condB[s] = (np.array([x[0] for x in S]), np.array([x[1] for x in S]), testB.success.values)
            for (r0, r1, er), (_, rw) in zip(S, testB.iterrows()):
                rows.append({"encoder": e, "variant": variant, "experiment": "B", "shift": s,
                             "condition": rw.policy_type, "episode_index": int(rw.episode_index), "success": bool(rw.success),
                             "score": r0, "score_ref": r1, "errors": er.astype(np.float32).tolist()})
        for x, (_, rw) in zip(calB_s, calB_v.iterrows()):
            rows.append({"encoder": e, "variant": variant, "experiment": "B_cal", "shift": "nominal",
                         "condition": rw.policy_type, "episode_index": int(rw.episode_index), "success": True,
                         "score": x[0], "score_ref": x[1], "errors": x[2].astype(np.float32).tolist()})
        for x, (_, rw) in zip(cal_s, tele_cal.iterrows()):
            rows.append({"encoder": e, "variant": variant, "experiment": "A_cal", "shift": "nominal",
                         "condition": "teleoperated", "episode_index": int(rw.episode_index), "success": True,
                         "score": x[0], "score_ref": x[1], "errors": x[2].astype(np.float32).tolist()})
        dfB = evaluate_scores(condB, [x[0] for x in calB_s], [x[1] for x in calB_s], a.alpha, k=a.k, n_orders=100, seed=a.seed)
        dfB.insert(0, "variant", variant)
        dfB.insert(0, "encoder", e)
        dfB.to_csv(out / f"table_B_{e}_{variant}.csv", index=False)
        fb = dfB[dfB.method.isin(["fixed", "adaptive_log"])].pivot_table(index="condition", columns="method", values="false_alarm")
        summary[f"{e}/{variant}"] = {"B_false_alarm": fb.round(3).to_dict(), "B_auroc": dfB.groupby("condition").auroc_raw.first().round(3).to_dict(),
                                     "A_auroc": dfA.groupby("condition").auroc_raw.first().round(3).to_dict()}
        log(f"[{e}/{variant}] B false alarms (fixed vs adaptive_log):\n{fb.round(3).to_string()}")
    pd.DataFrame(rows).to_parquet(out / "traces.parquet")           # save as we go
json.dump({"args": vars(a), "summary": summary}, open(out / "summary.json", "w"), indent=1, default=str)
log("DONE ->", out)
