#!/usr/bin/env bash
# N19.12: FULL_LOOP_N19 (learned writer + deployed verifier + real SAM3
# reactivation) on cal10, 4 shards on idle GPUs. One blocking command.
set -u

ROOT=.
cd "$ROOT"
PY="$ROOT/envs/sam3_intermot/bin/python"
mkdir -p "$ROOT/outputs/n19/logs"

GPUS=${GPUS:-3,5,8,9}
WRITER=${WRITER:-"$ROOT/outputs/n19/models/writer_v0/writer_v0.pt"}
WRCFG=${WRCFG:-"$ROOT/outputs/n19/models/writer_v0/writer_config.json"}
THRESH=${THRESH:-0.95}
T=$THRESH

IFS=, read -ra GPULIST <<< "$GPUS"
NGPU=${#GPULIST[@]}
PIDS=()
for i in "${!GPULIST[@]}"; do
  gpu=${GPULIST[$i]}
  "$PY" scripts/run_n19_full_loop_learned.py \
    --gpu "$gpu" --shard "$i" --nshards "$NGPU" \
    --split cal10 --writer "$WRITER" --writer-config "$WRCFG" \
    --writer-threshold "$T" --memory-k 2 --out-tag learned_n19 \
    > "$ROOT/outputs/n19/logs/learned_full_s${i}.log" 2>&1 &
  PIDS+=($!)
  echo "launched gpu=$gpu pid=$! threshold=$T"
done

rc=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || rc=1
done
if [ $rc -ne 0 ]; then
  echo "LEARNED_FULL_LOOP_FAIL"
  exit 1
fi
echo "LEARNED_FULL_LOOP_DONE"
