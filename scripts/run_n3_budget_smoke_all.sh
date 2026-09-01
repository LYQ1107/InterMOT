#!/usr/bin/env bash
# N3 three-sequence budget smoke: one isolated subprocess per (budget, sequence).
set -euo pipefail

PROJECT=.
PY="$PROJECT/envs/sam3_intermot/bin/python"
OUT="$PROJECT/outputs/n3_smoke"
mkdir -p "$OUT"
if [ -d "$OUT/mot_results" ]; then
  mkdir -p "$OUT/old_attempts"
  mv "$OUT/mot_results" "$OUT/old_attempts/mot_results.$(date +%s)"
fi

for b in 0 1 2 5; do
  for s in dancetrack0004 dancetrack0005 dancetrack0007; do
    echo "=== budget=$b sequence=$s start $(date +%H:%M:%S) ==="
    N3_BUDGETS="$b" N3_SEQS="$s" N3_SKIP_TRACKEVAL=1 \
      CUDA_VISIBLE_DEVICES=8 PYTHONPATH="$PROJECT" \
      "$PY" "$PROJECT/scripts/run_n3_budget_smoke.py" \
      >> "$OUT/run_all.log" 2>&1
    echo "=== budget=$b sequence=$s done $(date +%H:%M:%S) ==="
  done
done

# Official TrackEval over all completed budget groups.
printf 'name\ndancetrack0004\ndancetrack0005\ndancetrack0007\n' > "$OUT/seqmap.txt"
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
  > "$OUT/trackeval_final.log" 2>&1

echo "N3_ALL_DONE rc=$?"
