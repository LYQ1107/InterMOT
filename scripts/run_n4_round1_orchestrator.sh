#!/usr/bin/env bash
# Round 1 module isolation, parallelized on idle GPUs 1..9.
set -euo pipefail

PROJECT=.
PY="$PROJECT/envs/sam3_intermot/bin/python"
ROOT_OUT="$PROJECT/outputs/n4/round1"
LOG_ROOT="$ROOT_OUT/logs"
mkdir -p "$LOG_ROOT"

declare -a JOBS
GPU_LIST=(1 2 3)
for cfg in R1_C R1_CD R1_CA R1_CR R1_FULL_NOGUARD; do
  for b in 1 2 5; do
    for s in dancetrack0004 dancetrack0005 dancetrack0007; do
      JOBS+=("$cfg $b $s")
    done
  done
done

PIDS=()
IDX=0
for job in "${JOBS[@]}"; do
  read -r cfg b s <<< "$job"
  gpu="${GPU_LIST[$((IDX % 3))]}"
  cfg_out="$ROOT_OUT/$cfg"
  mkdir -p "$cfg_out"
  N4_CONFIG="$cfg" N3_BUDGETS="$b" N3_SEQS="$s" N3_SKIP_TRACKEVAL=1 \
    N4_OUT_DIR="$cfg_out" CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PROJECT" \
    "$PY" "$PROJECT/scripts/run_n3_budget_smoke.py" \
    > "$LOG_ROOT/${cfg}_b${b}_${s}_gpu${gpu}.log" 2>&1 &
  PIDS+=("$!")
  IDX=$((IDX+1))
  if (( IDX % 3 == 0 )); then
    for p in "${PIDS[@]}"; do wait "$p" || exit 1; done
    PIDS=()
  fi
done
for p in "${PIDS[@]}"; do wait "$p" || exit 1; done

echo "ROUND1_RUNS_DONE"

# Official TrackEval per config (b1/b2/b5).
for cfg in R1_C R1_CD R1_CA R1_CR R1_FULL_NOGUARD; do
  cfg_out="$ROOT_OUT/$cfg"
  printf 'name\ndancetrack0004\ndancetrack0005\ndancetrack0007\n' > "$cfg_out/seqmap.txt"
  python \
    ./third_party/MOTIP/TrackEval/scripts/run_mot_challenge.py \
    --GT_FOLDER /path/to/dancetrack/val \
    --TRACKERS_FOLDER "$cfg_out/mot_results" \
    --TRACKERS_TO_EVAL b1 b2 b5 \
    --TRACKER_SUB_FOLDER '' --OUTPUT_SUB_FOLDER '' \
    --SEQMAP_FILE "$cfg_out/seqmap.txt" \
    --BENCHMARK DanceTrack --SPLIT_TO_EVAL val --SKIP_SPLIT_FOL True \
    --DO_PREPROC False --CLASSES_TO_EVAL pedestrian \
    --METRICS HOTA CLEAR Identity \
    --USE_PARALLEL False --PLOT_CURVES False \
    --PRINT_RESULTS True --PRINT_ONLY_COMBINED False \
    --OUTPUT_SUMMARY True --OUTPUT_DETAILED True \
    > "$cfg_out/trackeval.log" 2>&1
done

echo "ROUND1_TRACKEVAL_DONE"
