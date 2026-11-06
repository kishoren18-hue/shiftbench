# Offline results on real SO-101 video (ArmnetBench), Nov 6 2026

**Setup.** ArmnetBench v0.1, task "Put the eye drops into the basket", front camera, detector at 2 fps.
World model trained on 25 human teleop demos. Experiment B is calibrated on 55 successful policy runs and tested
on 70 others (the first k = 15 are used only by recalibration; 55 scored: 24 successes, 31 failures), each test episode
replayed unchanged and under four image shifts. Five frozen encoders: V-JEPA 2 ViT-L, DINOv2-S, SigLIP 2-B, CLIP B/32, ResNet-18.
Target false-alarm rate α = 10%. Ran on CPU in Vercel Sandbox: 10 parallel encoding workers plus one final job.
Full outputs (per-episode scores, every table, logs, demo clips): https://ulpc6o5irf2ykqv7.public.blob.vercel-storage.com/eyedrops_v1/results.tgz

## 1. The promise can break, and the adaptive cutoff holds it

False-alarm rate on successful runs (24 runs, so one false alarm = 4.2 points):

| model | no shift | lamp dimmed | small bump | large bump | object in view |
|---|---|---|---|---|---|
| V-JEPA 2, fixed | 4% | 4% | 4% | **38%** | 0% |
| V-JEPA 2, adaptive | 4% | 8% | 4% | 12.5% | 8% |
| SigLIP 2-B, fixed | 0% | 12.5% | **54%** | **71%** | 0% |
| SigLIP 2-B, adaptive | 4% | 8% | 12.5% | 8% | 4% |
| DINOv2-S, fixed | 17% | 8% | 4% | 0% | 17% |
| CLIP B/32, fixed | 0% | 4% | 0% | 12.5% | 0% |
| ResNet-18, fixed | 8% | 0% | 8% | 12.5% | 8% |

Adaptive: 4–12.5% for every model and shift. Small shifts mostly did not break the fixed cutoff; large camera bumps did.

## 2. The real weakness: failure detection is near chance here

AUROC (0.5 = coin flip): V-JEPA 2 0.41–0.61, every other model 0.40–0.62. Zeroing the actions or replacing the
learned predictor with "how much did the frame change" gives the same AUROC (V-JEPA 2: 0.611 / 0.614 / 0.577).
On this task, at 2 fps with 25 training demos, the world model is not seeing failures.

## 3. Demo

Large bump at about 4 s, mid-episode: the alarm fires within 0.4 s in both clips, and never fires in the untouched controls.
The score falls back within ~2 s as the predictor adapts to the new view, so a bump that happens before the run starts
would be much harder to catch.

## 4. Policy shift

Only 1–6 successful runs per policy after the first k. Not evidence either way.

## What to change before the physical experiment

1. Detection first: 4 fps or more, 50+ training demos, the `grid2` (spatial) pooling, and a task whose failures are visible
   (dropped object, missed grasp). Get AUROC well above 0.7 on nominal data before measuring shifts.
2. More successful test runs per condition (≥ 60) so false-alarm rates aren't 4-point steps.
3. Keep the image-shift suite as a cheap pre-check; the physical shifts are the real test.
