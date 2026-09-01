#!/usr/bin/env bash
set -u
cd .
PY=envs/sam3_intermot/bin/python
LOG=outputs/n18/logs
mkdir -p "$LOG"

for s in 0 1 2 3; do
  gpu=$((s + 4))
  PYTHONPATH=. "$PY" scripts/audit_n18_health_features.py --gpu "$gpu" \
    --shard "$s" --nshards 4 > "$LOG/audit_health_s${s}.log" 2>&1 &
done
wait
echo ALL_HEALTH_AUDIT_DONE
