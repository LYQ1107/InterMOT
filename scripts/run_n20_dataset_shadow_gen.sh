#!/usr/bin/env bash
# N20.5B/6: all top-K candidate shadows for cal10 + train30 dataset attempts.
set -u

ROOT=.
PY="$ROOT/envs/sam3_intermot/bin/python"
cd "$ROOT"
mkdir -p "$ROOT/outputs/n20/logs"

GPUS=${GPUS:-3,5,8,9}
K=${K:-5}
HORIZON=${HORIZON:-8}

run_split () {
  split=$1
  csv=$2
  outdir=$3
  events=$4
  tag=$5
  IFS=, read -ra GPULIST <<< "$GPUS"
  NGPU=${#GPULIST[@]}
  PIDS=()
  for i in "${!GPULIST[@]}"; do
    gpu=${GPULIST[$i]}
    "$PY" scripts/run_n20_all_candidate_shadow.py \
      --gpu "$gpu" --shard "$i" --nshards "$NGPU" \
      --attempts-csv "$csv" --k "$K" --horizon "$HORIZON" \
      --out-tag "$tag" --out-dir "$outdir" --events-jsonl "$events" \
      > "$ROOT/outputs/n20/logs/allcand_${tag}_k${K}_s${i}.log" 2>&1 &
    PIDS+=($!)
  done
  rc=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || rc=1
  done
  if [ $rc -ne 0 ]; then
    echo "ALL_CANDIDATE_FAIL $tag"
    exit 1
  fi
  echo "ALL_CANDIDATE_DONE $tag"
}

run_split cal10 \
  "$ROOT/outputs/n20/dataset_attempts_cal10.csv" \
  full_shadow_cache_cal10 \
  "$ROOT/outputs/n20/full_loop_oracle_shadow/events_dump_only.jsonl" \
  cal10_dataset

run_split train30 \
  "$ROOT/outputs/n20/dataset_attempts_train30.csv" \
  full_shadow_cache_train30 \
  "$ROOT/outputs/n20/full_loop_oracle_shadow/events_dump_train30.jsonl" \
  train30_dataset

echo "DATASET_SHADOW_GEN_DONE"
