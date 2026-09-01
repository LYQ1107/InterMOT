#!/usr/bin/env bash
# N18 RouteC.8: one blocking command for R1 partial-backbone training.
set -u

ROOT=.
cd "$ROOT"

GPUS=${GPUS:-4,8,9}
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
LOCAL=$(seq -s, 0 $((NGPU - 1)))
TAG=${1:-r1}

export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=4

envs/sam3_intermot/bin/python scripts/train_route_c_r1.py \
  --gpus "$LOCAL" --tag "$TAG" 2>&1 | tee "outputs/n18/route_c/logs/${TAG}_train.log"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  echo "R1_RUNNER_FAIL"
  exit 1
fi
echo "R1_RUNNER_DONE"
