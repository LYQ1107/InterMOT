#!/usr/bin/env bash
# N19.5 data generation: FULL_LOOP_V0 (deployed verifier + real SAM3
# reactivation) on train30 to produce the real causal write-candidate
# distribution. One blocking command; 4 shards on idle GPUs.
set -u

ROOT=.
cd "$ROOT"
PY="$ROOT/envs/sam3_intermot/bin/python"
mkdir -p "$ROOT/outputs/n19/logs"

GPUS=${GPUS:-0,1,2,3}
IFS=, read -ra GPULIST <<< "$GPUS"
NGPU=${#GPULIST[@]}
PIDS=()

for i in "${!GPULIST[@]}"; do
  gpu=${GPULIST[$i]}
  "$PY" scripts/run_n18_full_loop_v0.py \
    --gpu "$gpu" --shard "$i" --nshards "$NGPU" \
    --split train30 \
    --out-tag train30_n19 \
    > "$ROOT/outputs/n19/logs/train30_full_s${i}.log" 2>&1 &
  PIDS+=($!)
  echo "launched gpu=$gpu pid=$!"
done

rc=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || rc=1
done
if [ $rc -ne 0 ]; then
  echo "TRAIN30_FULL_LOOP_FAIL"
  exit 1
fi
echo "TRAIN30_FULL_LOOP_DONE"
