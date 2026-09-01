#!/usr/bin/env bash
# N20.2: Oracle Shadow Propagation on 4 idle GPUs (one blocking command).
set -u

ROOT=.
PY="$ROOT/envs/sam3_intermot/bin/python"
cd "$ROOT"
mkdir -p "$ROOT/outputs/n20/logs" "$ROOT/outputs/n20/shadow_cache"

GPUS=${GPUS:-3,5,8,9}
K=${K:-5}
HORIZON=${HORIZON:-120}
ATTEMPT_CSV=${ATTEMPT_CSV:-"$ROOT/outputs/n20/topk_no_commit.csv"}
TAG=${TAG:-oracle_shadow}
LIMIT=${LIMIT:-0}

IFS=, read -ra GPULIST <<< "$GPUS"
NGPU=${#GPULIST[@]}
PIDS=()
for i in "${!GPULIST[@]}"; do
  gpu=${GPULIST[$i]}
  extra=""
  if [ "$LIMIT" -gt 0 ]; then extra="--limit-attempts $LIMIT"; fi
  "$PY" scripts/run_n20_oracle_shadow.py \
    --gpu "$gpu" --shard "$i" --nshards "$NGPU" \
    --attempts-csv "$ATTEMPT_CSV" --k "$K" --horizon "$HORIZON" \
    --out-tag "$TAG" --skip-existing --reset-session $extra \
    > "$ROOT/outputs/n20/logs/shadow_${TAG}_k${K}_s${i}.log" 2>&1 &
  PIDS+=($!)
  echo "shadow gpu=$gpu shard=$i pid=$!"
done

rc=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || rc=1
done
if [ $rc -ne 0 ]; then
  echo "SHADOW_GEN_FAIL"
  exit 1
fi
echo "SHADOW_GEN_DONE k=$K horizon=$HORIZON"
