#!/usr/bin/env bash
# Gather every shard's cache from Blob, encode anything missing, train + evaluate all
# encoders, render the replay demos, and upload results to Blob.
source "$(dirname "$0")/common.sh"
ENCODERS=${ENCODERS:-vjepa2,dinov2_s,siglip2_b,clip_b32,resnet18}
OUT=/vercel/out_final
mkdir -p /vercel/dl "$OUT"
$BLOB list "$RUN/cache/" | while read -r url path; do
  curl -sfL "$url" -o "/vercel/dl/$(basename "$path")" && tar xzf "/vercel/dl/$(basename "$path")" -C /vercel
done
python scripts/offline_suite.py "${SUITE_ARGS[@]}" --encoders "$ENCODERS" --out "$OUT" > /vercel/final.log 2>&1
python scripts/replay_demo.py --repo_id armnet/armnetbench_v01_lerobot_so101 --data_dir /vercel/armnet \
  --detector "$OUT/detector_vjepa2" --encoder vjepa2 --fps 2 --episodes_csv "$OUT/episodes.csv" \
  --traces "$OUT/traces.parquet" --out "$OUT/demo" --n 3 >> /vercel/final.log 2>&1 || echo "replay demo failed" >> /vercel/final.log
cp /vercel/final.log "$OUT/"
tar czf /vercel/results_$RUN.tgz -C /vercel out_final
$BLOB put /vercel/results_$RUN.tgz "$RUN/results.tgz" >> /vercel/final.log
for f in "$OUT"/summary.json "$OUT"/table_*.csv "$OUT"/episodes.csv "$OUT"/final.log "$OUT"/demo/*; do
  [ -f "$f" ] && $BLOB put "$f" "$RUN/out/$(basename "$f")" >> /vercel/final.log
done
echo FINAL_DONE >> /vercel/final.log
