#!/usr/bin/env bash
# After final.sh: replay demo (downloads its episodes), site results, compact summary; upload.
source "$(dirname "$0")/common.sh"
set +e
OUT=/vercel/out_final; LOG=/vercel/post.log
pip install -q tabulate >> $LOG 2>&1 || uv pip install -q tabulate >> $LOG 2>&1
up() { [ -f "$1" ] && $BLOB put "$1" "$RUN/out/$(basename "$1")" >> $LOG 2>&1; }
timeout "${DEMO_LIMIT:-600}" python scripts/replay_demo.py --repo_id armnet/armnetbench_v01_lerobot_so101 --data_dir /vercel/armnet \
  --detector "$OUT/detector_vjepa2" --encoder vjepa2 --fps 2 --episodes_csv "$OUT/episodes.csv" \
  --traces "$OUT/traces.parquet" --out "$OUT/demo" --n 2 >> $LOG 2>&1 || echo "replay demo failed/timed out" >> $LOG
python scripts/summarize_suite.py "$OUT" >> $LOG 2>&1
mkdir -p /vercel/site_out
python scripts/make_site_results.py --out_dir "$OUT" --site /vercel/site_out --alpha 0.1 \
  --video_base "https://ulpc6o5irf2ykqv7.public.blob.vercel-storage.com/$RUN/out" >> $LOG 2>&1
for f in "$OUT"/demo/* "$OUT"/SUMMARY.md "$OUT"/fig_false_alarm.png /vercel/site_out/results.json; do up "$f"; done
tar czf /vercel/results.tgz -C /vercel out_final && $BLOB put /vercel/results.tgz "$RUN/results.tgz" >> $LOG 2>&1
echo POST_DONE >> $LOG; up $LOG
