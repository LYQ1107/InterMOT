#!/usr/bin/env bash
# N18 F3/F4 split: replay the recorded FULL_LOOP_V0 GFN attempts and audit
# top-1/3/5/10 plus best-detection recall. One blocking command, GPUs 4-7.
set -u
cd .
PY=envs/sam3_intermot/bin/python
LOG=outputs/n18/logs
mkdir -p "$LOG"

for s in 0 1 2 3; do
  gpu=$((s + 4))
  PYTHONPATH=. "$PY" scripts/audit_n18_gfn_topk.py --gpu "$gpu" \
    --shard "$s" --nshards 4 > "$LOG/audit_gfn_topk_s${s}.log" 2>&1 &
done
wait
echo ALL_AUDIT_STAGES_DONE
