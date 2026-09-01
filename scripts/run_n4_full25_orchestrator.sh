#!/usr/bin/env bash
# Full 25-sequence evaluation with winner config R2_G2.
set -euo pipefail

PROJECT=.
PY="$PROJECT/envs/sam3_intermot/bin/python"
OUT="$PROJECT/outputs/n4/full25/winner"
LOG_ROOT="$OUT/logs"
mkdir -p "$LOG_ROOT"

SEQS=$(awk 'NR>1 && $1!="" && $1!="name" {print $1}' \
  /path/to/dancetrack/val/val_seqmap.txt \
  | grep -v '^val_seqmap.txt$')

declare -a JOBS
GPU_LIST=(1 2 3 5 6 7)
for b in 0 1 2 5; do
  for s in $SEQS; do
    JOBS+=("$b $s")
  done
done

PIDS=()
IDX=0
for job in "${JOBS[@]}"; do
  read -r b s <<< "$job"
  gpu="${GPU_LIST[$((IDX % 6))]}"
  N4_CONFIG=R2_G2 N3_BUDGETS="$b" N3_SEQS="$s" N3_SKIP_TRACKEVAL=1 \
    N4_OUT_DIR="$OUT" CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PROJECT" \
    "$PY" "$PROJECT/scripts/run_n3_budget_smoke.py" \
    > "$LOG_ROOT/b${b}_${s}_gpu${gpu}.log" 2>&1 &
  PIDS+=("$!")
  IDX=$((IDX+1))
  if (( IDX % 6 == 0 )); then
    for p in "${PIDS[@]}"; do wait "$p" || exit 1; done
    PIDS=()
  fi
done
for p in "${PIDS[@]}"; do wait "$p" || exit 1; done

echo "FULL25_RUNS_DONE"

printf 'name\n' > "$OUT/seqmap.txt"
for s in $SEQS; do echo "$s" >> "$OUT/seqmap.txt"; done
python \
  ./third_party/MOTIP/TrackEval/scripts/run_mot_challenge.py \
  --GT_FOLDER /path/to/dancetrack/val \
  --TRACKERS_FOLDER "$OUT/mot_results" \
  --TRACKERS_TO_EVAL b0 b1 b2 b5 \
  --TRACKER_SUB_FOLDER '' --OUTPUT_SUB_FOLDER '' \
  --SEQMAP_FILE "$OUT/seqmap.txt" \
  --BENCHMARK DanceTrack --SPLIT_TO_EVAL val --SKIP_SPLIT_FOL True \
  --DO_PREPROC False --CLASSES_TO_EVAL pedestrian \
  --METRICS HOTA CLEAR Identity \
  --USE_PARALLEL False --PLOT_CURVES False \
  --PRINT_RESULTS True --PRINT_ONLY_COMBINED False \
  --OUTPUT_SUMMARY True --OUTPUT_DETAILED True \
  > "$OUT/trackeval_full25.log" 2>&1

echo "FULL25_TRACKEVAL_DONE"
