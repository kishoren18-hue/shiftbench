#!/usr/bin/env bash
# Record one block of policy rollouts in SHADOW mode (watchdog logs, never stops),
# then label them. One block = one condition at one level = one LeRobot dataset.
#
#   bash scripts/record_session.sh nominal 0 20
#   bash scripts/record_session.sh light_dim 1 40
#
# Set once (lerobot-find-port and lerobot-find-cameras give the values):
#   export FOLLOWER_PORT=/dev/ttyACM0 ARM_ID=kishore_so101 HF_USER=<your HF username>
#   export POLICY=$HF_USER/act_so101_cube_bowl
#   export DETECTOR=runs/detector        # optional; leave unset until you have trained one
#
# Uses `lerobot-rollout --strategy.type=episodic` (LeRobot 0.7: lerobot-record is teleop-only).
# Dataset names start with the date so sorting by name = recording order.
set -euo pipefail
COND=${1:?condition}; LEVEL=${2:-0}; N=${3:-20}
: "${FOLLOWER_PORT:?}" "${ARM_ID:?}" "${HF_USER:?}" "${POLICY:?}"
STAMP=$(date +%Y%m%d_%H%M)
NAME="rollout_shiftbench_${STAMP}_${COND}_l${LEVEL}"
ROOT="data/${NAME}"
TASK=$(python -c "import yaml;print(yaml.safe_load(open('configs/shifts.yaml'))['task'])")
CAMS=${CAMS:-'{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}'}

ROBOT_ARGS=(--robot.type=so101_follower)
if [[ -n "${DETECTOR:-}" ]]; then
  mkdir -p logs
  ROBOT_ARGS=(--robot.type=so101_watchdog --robot.detector_dir="$DETECTOR" --robot.watch_camera=front
              --robot.mode=shadow --robot.log_path="logs/${NAME}.jsonl")
fi

echo ">>> Set up '$COND' level $LEVEL exactly as configs/shifts.yaml says."
echo ">>> Take a photo of the setup into docs/setup_photos/${NAME}.jpg, note the lux at the cube, then press Enter."
read -r

lerobot-rollout --strategy.type=episodic \
  "${ROBOT_ARGS[@]}" --robot.port="$FOLLOWER_PORT" --robot.id="$ARM_ID" --robot.cameras="$CAMS" \
  --policy.path="$POLICY" \
  --dataset.repo_id="$HF_USER/$NAME" --dataset.root="$ROOT" --dataset.no_stamp=true \
  --dataset.single_task="$TASK" --dataset.num_episodes="$N" \
  --dataset.episode_time_s=20 --dataset.reset_time_s=15 --dataset.push_to_hub=false

echo ">>> Label the episodes (s/f) with the success rule in configs/shifts.yaml"
shiftbench-label --root "$ROOT" --condition "$COND" --level "$LEVEL"
echo "$ROOT" >> data/sessions.txt
