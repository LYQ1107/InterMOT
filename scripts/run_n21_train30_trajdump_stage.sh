#!/usr/bin/env bash
# N21 Phase-II: second true-live pass over train30 that PERSISTS the
# per-attempt SAM3 shadow trajectories (--dump-trajectories) so the CATIL
# retrain can use the true on-policy distribution. 4 idle GPUs (5/6/7/8),
# round-robin shards, per-seq .done resume, <=3 retries per shard.
set -u
ROOT=.
PY=$ROOT/envs/sam3_intermot/bin/python
OUT=$ROOT/outputs/n21/train30_true_onpolicy_trajdump
LOG=$ROOT/outputs/n21/logs
mkdir -p "$OUT" "$LOG"

SEQS=($($PY - <<'EOF'
import json
f=json.load(open('./outputs/n15/n15_frozen.json'))
print(' '.join(sorted(f['split']['train30'])))
EOF
))
echo "TRAJDUMP_TOTAL_SEQS=${#SEQS[@]}" | tee "$LOG/trajdump_stage.out"

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
  echo "GPU$gpu SEQS=$seqs_csv" >> "$LOG/trajdump_stage.out"
  for attempt in 1 2 3; do
    echo "GPU$gpu ATTEMPT=$attempt START $(date +%F-%T)" >> "$LOG/trajdump_stage.out"
    "$PY" "$ROOT/scripts/run_n21_train30_onpolicy.py" \
      --gpu "$gpu" --seqs "$seqs_csv" --split train30 \
      --kplus1-model "$ROOT/outputs/n20/models/kplus1_gru.pt" \
      --dump-trajectories \
      --out-dir "outputs/n21/train30_true_onpolicy_trajdump" \
      >> "$LOG/trajdump_gpu${gpu}.log" 2>&1
    rc=$?
    echo "GPU$gpu ATTEMPT=$attempt RC=$rc END $(date +%F-%T)" >> "$LOG/trajdump_stage.out"
    if (( rc == 0 )); then return 0; fi
  done
  echo "GPU$gpu FAILED_3_ATTEMPTS" >> "$LOG/trajdump_stage.out"
  return 1
}

pids=()
for idx in 0 1 2 3; do
  run_shard $((5+idx)) $idx >> "$LOG/trajdump_stage.out" 2>&1 &
  pids+=($!)
done

rc_all=0
for p in "${pids[@]}"; do
  wait "$p" || rc_all=1
done

if (( rc_all == 0 )); then
  echo "TRAJDUMP_STAGE_DONE" >> "$LOG/trajdump_stage.out"
  echo "done" > "$OUT/STAGE.done"
else
  echo "TRAJDUMP_STAGE_PARTIAL" >> "$LOG/trajdump_stage.out"
fi
exit $rc_all
