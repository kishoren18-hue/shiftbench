#!/usr/bin/env bash
# Encode one shard for some encoders, then upload the cache to Blob.
#   ENCODERS=vjepa2 SHARD=0/6 bash scripts/cloud/worker.sh
# Stops encoding at LIMIT seconds (default 40 min, under Hobby's 45-min sandbox cap)
# and uploads whatever finished; the final job encodes anything missing.
source "$(dirname "$0")/common.sh"
LIMIT=${LIMIT:-2400}
TAG="${ENCODERS//,/+}_${SHARD//\//of}"
timeout "$LIMIT" python scripts/offline_suite.py "${SUITE_ARGS[@]}" --encoders "$ENCODERS" --shard "$SHARD" \
  --encode_only --out /vercel/out_$TAG > /vercel/worker.log 2>&1 || echo "encoder exited $? (timeout = 124)" >> /vercel/worker.log
tar czf /vercel/cache_$TAG.tgz -C /vercel cache
$BLOB put /vercel/cache_$TAG.tgz "$RUN/cache/cache_$TAG.tgz" >> /vercel/worker.log
$BLOB put /vercel/worker.log "$RUN/logs/worker_$TAG.log" > /dev/null
echo WORKER_DONE >> /vercel/worker.log
