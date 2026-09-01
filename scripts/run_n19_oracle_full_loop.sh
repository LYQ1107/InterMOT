#!/usr/bin/env bash
# N19.3: Oracle causal-refresh FULL_LOOP on calibration10.
# Offline upper-bound diagnostic: oracle anchor (write memory when the
# current delivery is GT-correct) + oracle verifier (accept only GT-correct
# recovery candidates) + real SAM3 reactivation. One blocking command.
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
    --oracle-anchor --oracle-verifier \
    --out-tag oracle_n19 \
    > "$ROOT/outputs/n19/logs/oracle_full_s${i}.log" 2>&1 &
  PIDS+=($!)
  echo "launched gpu=$gpu pid=$!"
done

rc=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || rc=1
done
if [ $rc -ne 0 ]; then
  echo "ORACLE_FULL_LOOP_FAIL"
  exit 1
fi
echo "ORACLE_FULL_LOOP_DONE"
