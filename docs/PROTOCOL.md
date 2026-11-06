# ShiftBench protocol

## The question, stated so it can fail

A world-model alarm calibrated on successful rollouts promises: *a run that
would have succeeded raises an alarm at most α of the time*. Published
detectors (FAIL-Detect, Foresight, FoMo-FD) calibrate and test in the same
setup. ShiftBench asks:

* **H1** After a controlled shift (lighting, camera pose, table, distractors,
  new object), the fixed conformal cutoff's false-alarm rate on successful
  runs exceeds α. *Fails if* it stays ≤ α (Wilson upper bound) for every shift.
* **H2** The adaptive cutoff keeps the long-run false-alarm rate within its
  stated bound under every shift, and its rate on the evaluation episodes is
  closer to α than the fixed cutoff's. *Fails if* it doesn't.
* **H3** (exploratory) The self-referenced score cancels some of the
  camera/lighting shift without losing detection (AUROC).

Write these into the README and commit them **before** recording shifted
data, with the date. That's your pre-registration.

## Setup rules

1. One cell, one task: "put the red cube in the bowl" (configs/shifts.yaml).
2. The arm is clamped. The front camera's tripod position, height and angle are taped and photographed.
3. Blinds closed, room lights off. The lamp is the only light source you control, so daylight can't leak in as a shift.
4. Camera exposure and white balance are locked. Note the values.
5. Before every session: a photo of the setup, lux at the cube, the time.
6. Success rule written down before the first rollout (shifts.yaml) and never changed. Ambiguous runs get `x` (skip) with a note. Never guess.

## Data plan

| Block | Rollouts | Purpose |
|---|---|---|
| teleop demos | 50 | train the policy (ACT via LeRobot) |
| nominal A | 100 | 50 train the world model, 50 calibrate the cutoff |
| nominal B, C, D | 20 each, at start, middle and end of the shift sessions | the no-shift control; measures drift from servo heat, wear, etc. |
| each shift × level | 40 | test |

All policy rollouts are recorded in **shadow mode** (`DETECTOR` set,
`--robot.mode=shadow`, or no detector at all), so the arm is never stopped and
every episode gets a true success/fail label. Stopping the arm is only for
the demo video.

Shuffle the order of shift blocks across days and put nominal blocks between
them, so "shift" isn't confused with "later in the day".

## Two-week plan

| Day | Do | Done when |
|---|---|---|
| 1 | Unbox, print parts if needed, assemble both arms (LeRobot SO-101 guide) | both arms assembled |
| 2 | `lerobot-find-port`, `lerobot-setup-motors`, `lerobot-calibrate` for leader and follower, `lerobot-teleoperate` | the follower mirrors the leader smoothly |
| 3 | Build the cell: clamps, tape marks, lamp, tripod; lock camera exposure; photos | shifts.yaml nominal matches reality |
| 3 | **Meanwhile on Colab**: notebook sections 1–3 (sanity check, V-JEPA speed, ArmnetBench pilot) | pilot table printed |
| 4–5 | Record 50 teleop demos with `lerobot-record` (teleop) | 50 clean episodes |
| 6 | Train ACT on Colab (`lerobot-train --policy.type=act`) | checkpoint on the Hub |
| 7 | `lerobot-rollout --strategy.type=base` a few times, check the policy works at all (aim ≥ 60% success; retrain or add demos if not) | policy usable |
| 8–9 | `record_session.sh nominal 0 50` twice = 100 nominal rollouts, labeled | 100 labeled |
| 10 | Encode + `train_detector.py` on Colab | detector saved, cutoff printed |
| 10 | Shadow-mode watchdog run: check the live scores look like the offline ones | log looks sane |
| 11–14 | Shift sessions (start with light_dim 1 and cam_pan 1), nominal blocks between | first shifted results table |

Then: the remaining shifts, `run_benchmark.py`, the stop-on-bump video
(below), push the dataset to the Hub with the shift labels.

## The demo video

`--robot.mode=stop`, one `lerobot-rollout --strategy.type=base --duration=20`
per take, so each take is one episode for the watchdog. Film the arm running
normally, then bump the camera mid-run, and the arm freezes. Overlay the
score and cutoff from `logs/*.jsonl`. Also film one take where nothing is
bumped and the arm finishes, so viewers see it doesn't freeze all the time.
Be honest in the caption: the alarm fires because the scene stopped matching
the world model's prediction, which is exactly what a camera bump does.

## Publishing the dataset

Each session is a LeRobot v3 dataset with `meta/shiftbench_episodes.jsonl`
(success, condition, level, fail_step, notes). Push with
`huggingface-cli upload` or LeRobot's push, include configs/shifts.yaml and
the setup photos, and write a dataset card listing every condition and
count. License: CC-BY-4.0 or Apache-2.0 (ArmnetBench uses Apache-2.0).
