#!/usr/bin/env bash
# N18 RouteC.3/C.4: one blocking command for R0 overfit sanity or formal
# 4-GPU DDP training. No agent-level polling.
set -u

ROOT=.
cd "$ROOT"

MODE=${1:-train}
TAG=${2:-r0}
GPUS=${GPUS:-4,8,9}
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
LOG="$ROOT/outputs/n18/route_c/logs/${TAG}_${MODE}.log"
mkdir -p "$ROOT/outputs/n18/route_c/logs" \
         "$ROOT/outputs/n18/route_c/models"

EXTRA=()
if [ "$MODE" = "overfit" ]; then
  EXTRA=(--overfit --tag "${TAG}_overfit")
elif [ "$MODE" = "train" ]; then
  EXTRA=(--tag "$TAG")
else
  echo "usage: $0 [overfit|train] [tag]"
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=4
envs/sam3_intermot/bin/torchrun --nproc_per_node=$NGPU \
  scripts/train_route_c_r0.py "${EXTRA[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  echo "R0_RUNNER_FAIL mode=$MODE tag=$TAG"
  exit 1
fi
echo "R0_RUNNER_DONE mode=$MODE tag=$TAG"
