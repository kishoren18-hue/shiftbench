#!/usr/bin/env bash
# Re-score cached latents with new variants (no video, no encoding). Uploads tables to Blob.
#   VARIANTS=trajdev,endstate OUTNAME=rescore_v1 bash scripts/cloud/rescore.sh
source "$(dirname "$0")/common.sh"
set +e
OUT=/vercel/out_${OUTNAME:-rescore}; LOG=/vercel/rescore.log; mkdir -p "$OUT"
if [ ! -d /vercel/cache/vjepa2 ]; then
  url=$($BLOB list "$RUN/cache/cache_zz_final" | awk '{print $1}' | head -1)
  curl -sfL "$url" -o /vercel/cache_all.tgz && tar xzf /vercel/cache_all.tgz -C /vercel
fi
echo "cache files: $(find /vercel/cache -name '*.npz' | wc -l)" >> $LOG
python scripts/offline_suite.py "${SUITE_ARGS[@]}" --encoders "${ENCODERS:-vjepa2,dinov2_s,siglip2_b,clip_b32,resnet18}" \
  --variants "${VARIANTS:-trajdev,endstate}" --out "$OUT" >> $LOG 2>&1
tar czf /vercel/$(basename $OUT).tgz -C /vercel $(basename $OUT) && $BLOB put /vercel/$(basename $OUT).tgz "$RUN/$(basename $OUT).tgz" >> $LOG 2>&1
echo RESCORE_DONE >> $LOG
