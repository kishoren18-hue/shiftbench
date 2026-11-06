# Shared setup for ShiftBench cloud workers (Vercel Sandbox, CPU). Sourced, not run.
set -euo pipefail
cd /vercel
if [ ! -d shiftbench ]; then git clone -q --depth 1 https://github.com/kishoren18-hue/shiftbench.git; fi
cd shiftbench && git pull -q || true
if [ ! -d /vercel/venv ]; then
  uv venv -q -p 3.12 /vercel/venv
  . /vercel/venv/bin/activate
  uv pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
  uv pip install -q transformers huggingface_hub av opencv-python-headless pandas pyarrow matplotlib pyyaml scipy imageio-ffmpeg
  uv pip install -q --no-deps -e .
fi
. /vercel/venv/bin/activate
if [ ! -d /vercel/blobcli/node_modules ]; then
  mkdir -p /vercel/blobcli && cp scripts/cloud/blob.mjs /vercel/blobcli/ && (cd /vercel/blobcli && npm init -y >/dev/null && npm i -s @vercel/blob >/dev/null)
fi
BLOB="node /vercel/blobcli/blob.mjs"
RUN="${RUN:-eyedrops_v1}"
# One configuration for every worker and the final job, so shards line up.
SUITE_ARGS=(--task "Put the eye drops into the basket" --fps 2 --max_per_policy 30 --max_teleop 50
            --max_test_B 70 --shifts nominal,light_dim_2,cam_bump_1,cam_bump_2,occlusion
            --alpha 0.1 --k 15 --data_dir /vercel/armnet --cache /vercel/cache --seed 0)
