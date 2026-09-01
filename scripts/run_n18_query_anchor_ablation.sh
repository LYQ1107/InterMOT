#!/usr/bin/env bash
# N18 query-anchor diagnosis: first-appearance H_i vs fresher last-seen
# (offline upper-bound) anchor for GFN ranking on recorded loop attempts.
set -u
cd .
PY=envs/sam3_intermot/bin/python
LOG=outputs/n18/logs
mkdir -p "$LOG"

for s in 0 1 2 3; do
  gpu=$((s + 4))
  PYTHONPATH=. "$PY" scripts/audit_n18_query_anchor.py --gpu "$gpu" \
    --shard "$s" --nshards 4 > "$LOG/audit_query_anchor_s${s}.log" 2>&1 &
done
wait
echo ALL_QUERY_ANCHOR_DONE
