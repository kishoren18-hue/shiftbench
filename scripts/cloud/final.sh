#!/usr/bin/env bash
# Gather every shard's cache from Blob, encode anything missing, train + evaluate all
# encoders, render the replay demos, and upload results to Blob. Each stage is
# time-boxed and saved before the next, so the 45-min Hobby cap can't lose the run.
source "$(dirname "$0")/common.sh"
set +e
ENCODERS=${ENCODERS:-vjepa2,dinov2_s,siglip2_b,clip_b32,resnet18}
OUT=/vercel/out_final
LOG=/vercel/final.log
mkdir -p /vercel/dl "$OUT"
up() { [ -f "$1" ] && $BLOB put "$1" "$RUN/out/$(basename "$1")" >> $LOG 2>&1; }
$BLOB list "$RUN/cache/" | while read -r url path; do
  curl -sfL "$url" -o "/vercel/dl/$(basename "$path")" && tar xzf "/vercel/dl/$(basename "$path")" -C /vercel
done
echo "cache files: $(find /vercel/cache -name '*.npz' | wc -l)" >> $LOG
# 1) encode leftovers, save them
timeout "${ENC_LIMIT:-1200}" python scripts/offline_suite.py "${SUITE_ARGS[@]}" --encoders "$ENCODERS" \
  --encode_only --out "$OUT" >> $LOG 2>&1
tar czf /vercel/cache_final.tgz -C /vercel cache && $BLOB put /vercel/cache_final.tgz "$RUN/cache/cache_zz_final.tgz" >> $LOG 2>&1
# 2) train + evaluate
timeout "${EVAL_LIMIT:-900}" python scripts/offline_suite.py "${SUITE_ARGS[@]}" --encoders "$ENCODERS" --out "$OUT" >> $LOG 2>&1
for f in "$OUT"/summary.json "$OUT"/table_*.csv "$OUT"/episodes.csv "$OUT"/run_log.txt; do up "$f"; done
python - >> $LOG 2>&1 <<'PY'
import pandas as pd
t = pd.read_parquet("/vercel/out_final/traces.parquet").drop(columns=["errors"])
t.to_csv("/vercel/out_final/episode_scores.csv", index=False)
PY
up "$OUT/episode_scores.csv"
# 3) replay demo with V-JEPA 2
timeout "${DEMO_LIMIT:-420}" python scripts/replay_demo.py --repo_id armnet/armnetbench_v01_lerobot_so101 --data_dir /vercel/armnet \
  --detector "$OUT/detector_vjepa2" --encoder vjepa2 --fps 2 --episodes_csv "$OUT/episodes.csv" \
  --traces "$OUT/traces.parquet" --out "$OUT/demo" --n 3 >> $LOG 2>&1 || echo "replay demo failed/timed out" >> $LOG
for f in "$OUT"/demo/*; do up "$f"; done
tar czf /vercel/results.tgz -C /vercel out_final && $BLOB put /vercel/results.tgz "$RUN/results.tgz" >> $LOG 2>&1
echo FINAL_DONE >> $LOG
up $LOG
