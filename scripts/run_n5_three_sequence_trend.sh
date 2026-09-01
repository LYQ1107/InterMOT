#!/usr/bin/env bash
# N5-3 three-sequence trend validation (P0/P1/P2/P3/P4-B1/B2/B4/B8).
set -euo pipefail

PROJECT=.
PY="$PROJECT/envs/sam3_intermot/bin/python"
CPU_PY=python
SEQS="dancetrack0004 dancetrack0005 dancetrack0007"
OUT="$PROJECT/outputs/n5"
GPU_LIST=(1 2 3 5 6 7)

# --- P1 offline (CPU) -----------------------------------------------------
for seq in $SEQS; do
  N5_PROTOCOL=p1 N5_SEQ="$seq" N5_OUT_DIR="$OUT/p1_oracle_frame_all/$seq" \
    N5_SKIP_TRACKEVAL=1 PYTHONPATH="$PROJECT" "$CPU_PY" \
    "$PROJECT/scripts/run_n5_continuous_observer.py"
done

# --- Stateful jobs (persistent 6-GPU worker pool) -------------------------
N5_SEQS="$SEQS" N5_GPUS="${N5_GPUS:-4 5 6 7 8 9}" N5_OUT_ROOT="$OUT" N5_TAG=trend \
  "$CPU_PY" "$PROJECT/scripts/run_n5_parallel_orchestrator.py"

echo "N5_3_TREND_RUNS_DONE"

# --- Official TrackEval on all protocol streams --------------------------
{ printf 'name\n'; for s in $SEQS; do echo "$s"; done; } > "$OUT/tmp_3seq_seqmap.txt"
eval_stream() {
  local label=$1 dir=$2
  mkdir -p "$OUT/tmp_trackeval/$label/mot_results/pre_mot" \
           "$OUT/tmp_trackeval/$label/mot_results/post_mot"
  for seq in $SEQS; do
    if [ -f "$dir/$seq.txt" ]; then
      cp "$dir/$seq.txt" "$OUT/tmp_trackeval/$label/mot_results/pre_mot/$seq.txt"
      cp "$dir/$seq.txt" "$OUT/tmp_trackeval/$label/mot_results/post_mot/$seq.txt"
    else
      cp "$dir/$seq/pre_mot/$seq.txt" "$OUT/tmp_trackeval/$label/mot_results/pre_mot/" 2>/dev/null || true
      cp "$dir/$seq/post_mot/$seq.txt" "$OUT/tmp_trackeval/$label/mot_results/post_mot/" 2>/dev/null || true
    fi
  done
  "$CPU_PY" \
    ./third_party/MOTIP/TrackEval/scripts/run_mot_challenge.py \
    --GT_FOLDER /path/to/dancetrack/val \
    --TRACKERS_FOLDER "$OUT/tmp_trackeval/$label/mot_results" \
    --TRACKERS_TO_EVAL pre_mot post_mot \
    --TRACKER_SUB_FOLDER '' --OUTPUT_SUB_FOLDER '' \
    --SEQMAP_FILE "$OUT/tmp_3seq_seqmap.txt" \
    --BENCHMARK DanceTrack --SPLIT_TO_EVAL val --SKIP_SPLIT_FOL True \
    --DO_PREPROC False --CLASSES_TO_EVAL pedestrian \
    --METRICS HOTA CLEAR Identity \
    --USE_PARALLEL False --PLOT_CURVES False \
    --PRINT_RESULTS True --PRINT_ONLY_COMBINED False \
    --OUTPUT_SUMMARY True --OUTPUT_DETAILED True \
    > "$OUT/tmp_trackeval/$label.log" 2>&1
}

mkdir -p "$OUT/tmp_trackeval"
eval_stream p0 "$OUT/integrity/canonical_mot_results/b0"
eval_stream p1 "$OUT/p1_oracle_frame_all"
eval_stream p2 "$OUT/p2_oracle_state_all"
eval_stream p3 "$OUT/p3_continuous_id_miss"
eval_stream p4_b1 "$OUT/p4_budget_b1"
eval_stream p4_b2 "$OUT/p4_budget_b2"
eval_stream p4_b4 "$OUT/p4_budget_b4"
eval_stream p4_b8 "$OUT/p4_budget_b8"

echo "N5_3_TRACKEVAL_DONE"
