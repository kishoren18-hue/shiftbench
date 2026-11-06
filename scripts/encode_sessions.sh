#!/usr/bin/env bash
# Encode every recorded session listed in data/sessions.txt into one cache (run on a GPU).
#   bash scripts/encode_sessions.sh cache/front observation.images.front
set -euo pipefail
CACHE=${1:-cache/front}; CAM=${2:-observation.images.front}
while read -r ROOT; do
  [[ -z "$ROOT" ]] && continue
  python scripts/encode_dataset.py --repo_id "local/$(basename "$ROOT")" --root "$ROOT" --camera "$CAM" --cache "$CACHE"
done < data/sessions.txt
