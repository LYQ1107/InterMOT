#!/usr/bin/env bash
set -u
cd .
PY=envs/sam3_intermot/bin/python
LOG=outputs/n18/logs
mkdir -p "$LOG"

for th in 0.4 0.5 0.6; do
  for s in 0 1 2 3; do
    gpu=$((s + 4))
    PYTHONPATH=. "$PY" scripts/audit_n18_custom_anchor.py \
      --anchors "outputs/n18/tables/verified_anchors_${th}.jsonl" \
      --out-name "verified_anchor_topk_${th}" \
      --gpu "$gpu" --shard "$s" --nshards 4 \
      > "$LOG/audit_verified_anchor_${th}_s${s}.log" 2>&1 &
  done
  wait
done
echo ALL_VERIFIED_ANCHOR_AUDITS_DONE
