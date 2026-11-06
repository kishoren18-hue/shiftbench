# Related work (checked Nov 6, 2026)

What exists, so the paper claims only what's new.

## World-model / conformal failure detectors (all calibrate in the test setup)

* **FAIL-Detect**, Xu et al., RSS 2025, arXiv 2503.08558. Learns scalar failure scores from successful rollouts only; conformal time-varying thresholds. The baseline idea ShiftBench stress-tests.
* **Foresight**, Zhang et al., 2026, arXiv 2606.23085. Uses **V-JEPA 2-AC** world-model latents for failure detection with functional conformal prediction, across VLA policies, sim plus ReactorX-200 and Franka arms. Closest prior work. A summary of it notes the paper doesn't address how much data or compute recalibration needs when deployment conditions shift. That is ShiftBench's question.
* **FoMo-FD**, 2026, arXiv 2607.27511. Flow-matching world model for surgical-robot failure detection; episode-level conformal threshold on successful executions.
* **Gauge** (world-model latents + conformal decision functions, success / known failure / OOD), and Ward et al. 2026, "Foundational world models accurately detect bimanual manipulator failures".
* **FIPER**: failure prediction for generative policies, calibrated with conformal prediction on successful rollouts.

## Adaptive conformal

* Gibbs & Candès 2021 (adaptive conformal inference); Angelopoulos, Candès & Tibshirani 2023 (conformal PID control). The update rule in `shiftbench/adaptive.py`, same as SteadyHand's.
* Conformalized teleoperation (RSS 2024) uses adaptive conformal quantile regression in robotics, but for action uncertainty, not failure alarms.

## Data

* **ArmnetBench v0.1** (`armnet/armnetbench_v01_lerobot_so101`): 2,499 SO-101 episodes, 8 tasks, 7 policies + teleop, success/failure/suboptimal labels, LeRobot v3, Apache-2.0. It has **no shift-condition labels**: no lighting, camera-pose or table fields. Use it for the policy-shift pilot and cite it. Your dataset's difference is *controlled, labeled* environment shifts.
* **DROID** (V-JEPA 2-AC's post-training data) and the LeRobot Hub SO-100/101 datasets.

## Model

* **V-JEPA 2**, Assran et al., 2025, arXiv 2506.09985. V-JEPA 2-AC post-trains an action-conditioned predictor on a frozen encoder with under 62 hours of unlabeled DROID video. Encoder used here: `facebook/vjepa2-vitl-fpc64-256` (MIT).

## The honest novelty statement

Not "world models detect robot failures" (done) and not "conformal
thresholds for failure detection" (done). It's this: **a controlled,
labeled measurement of whether those thresholds keep their stated
false-alarm rate under everyday shifts on a cheap reproducible arm, plus an
adaptive cutoff with a finite-sample guarantee that holds under any shift,
and a public dataset to check it.**
