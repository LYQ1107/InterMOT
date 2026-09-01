#!/usr/bin/env bash
# N21 Phase-II: true live train30 on-policy FULL_LOOP rollout.
# 4 idle GPUs (5/6/7/8), one sequence per GPU at a time, round-robin shards.
# Per-sequence .done markers enable resume; failed shards are retried at most
# 3 times (resume via markers).
set -u
ROOT=.
PY=$ROOT/envs/sam3_intermot/bin/python
OUT=$ROOT/outputs/n21/train30_true_onpolicy
LOG=$ROOT/outputs/n21/logs
mkdir -p "$OUT" "$LOG"

SEQS=($($PY - <<'EOF'
import json
f=json.load(open('./outputs/n15/n15_frozen.json'))
print(' '.join(sorted(f['split']['train30'])))
EOF
))
echo "TOTAL_SEQS=${#SEQS[@]}" | tee "$LOG/train30_stage.out"

run_shard() {
  local gpu=$1
  local idx=$2
  local shard=()
  local i=0
  for s in "${SEQS[@]}"; do
    if (( i % 4 == idx )); then shard+=("$s"); fi
    i=$((i+1))
  done
  local seqs_csv=$(IFS=,; echo "${shard[*]}")
  echo "GPU$gpu SEQS=$seqs_csv" >> "$LOG/train30_stage.out"
  for attempt in 1 2 3; do
    echo "GPU$gpu ATTEMPT=$attempt START $(date +%F-%T)" >> "$LOG/train30_stage.out"
    "$PY" "$ROOT/scripts/run_n21_train30_onpolicy.py" \
      --gpu "$gpu" --seqs "$seqs_csv" --split train30 \
      --kplus1-model "$ROOT/outputs/n20/models/kplus1_gru.pt" \
      >> "$LOG/train30_onpolicy_gpu${gpu}.log" 2>&1
    rc=$?
    echo "GPU$gpu ATTEMPT=$attempt RC=$rc END $(date +%F-%T)" >> "$LOG/train30_stage.out"
    if (( rc == 0 )); then return 0; fi
  done
  echo "GPU$gpu FAILED_3_ATTEMPTS" >> "$LOG/train30_stage.out"
  return 1
}

pids=()
for idx in 0 1 2 3; do
  run_shard $((5+idx)) $idx >> "$LOG/train30_stage.out" 2>&1 &
  pids+=($!)
done

rc_all=0
for p in "${pids[@]}"; do
  wait "$p" || rc_all=1
done

if (( rc_all == 0 )); then
  echo "N21_TRAIN30_STAGE_DONE" >> "$LOG/train30_stage.out"
  echo "N21_TRAIN30_STAGE_DONE" > "$OUT/STAGE.done"
else
  echo "N21_TRAIN30_STAGE_PARTIAL" >> "$LOG/train30_stage.out"
fi
exit $rc_all
