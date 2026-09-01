#!/usr/bin/env bash
# N18 FULL_LOOP_V1: causal trusted-memory query anchor.
# Same three modes as V0, tagged full_v1 / human_v1 / gfn_v1, GPUs 4-7.
set -u
cd .
PY=envs/sam3_intermot/bin/python
SEQS=dancetrack0074,dancetrack0075,dancetrack0080,dancetrack0082,dancetrack0083,dancetrack0086,dancetrack0087,dancetrack0096,dancetrack0098,dancetrack0099
LOG=outputs/n18/logs
mkdir -p "$LOG"

run_stage() {
  local tag=$1
  shift
  for s in 0 1 2 3; do
    gpu=$((s + 4))
    "$PY" scripts/run_n18_full_loop_v0.py --gpu "$gpu" --seqs "$SEQS" \
      --shard "$s" --nshards 4 --fresh-anchor --out-tag "$tag" \
      "$@" > "$LOG/loop_${tag}_s${s}.log" 2>&1 &
  done
  wait
  echo "STAGE_${tag}_DONE"
}

run_stage full_v1
run_stage human_v1 --no-recovery
run_stage gfn_v1 --no-reactivation
echo ALL_LOOP_V1_STAGES_DONE
