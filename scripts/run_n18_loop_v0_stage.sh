#!/usr/bin/env bash
# N18 FULL_LOOP_V0 calibration-10 stage: full / human-one-shot / gfn-rank-only.
# One blocking command, max 4 GPUs (physical 4-7: idle at launch time).
set -u
cd .
PY=envs/sam3_intermot/bin/python
SEQS=dancetrack0074,dancetrack0075,dancetrack0080,dancetrack0082,dancetrack0083,dancetrack0086,dancetrack0087,dancetrack0096,dancetrack0098,dancetrack0099
LOG=outputs/n18/logs
mkdir -p "$LOG"

run_stage() {
  local mode=$1
  shift
  for s in 0 1 2 3; do
    gpu=$((s + 4))
    "$PY" scripts/run_n18_full_loop_v0.py --gpu "$gpu" --seqs "$SEQS" \
      --shard "$s" --nshards 4 "$@" > "$LOG/loop_${mode}_s${s}.log" 2>&1 &
  done
  wait
  echo "STAGE_${mode}_DONE"
}

run_stage full
run_stage human --no-recovery
run_stage gfn --no-reactivation
echo ALL_LOOP_STAGES_DONE
