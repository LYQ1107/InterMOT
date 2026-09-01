#!/usr/bin/env bash
# N5-2 two-sequence real interaction gate (P3 stateful, short segments).
set -euo pipefail

PROJECT=.
PY="$PROJECT/envs/sam3_intermot/bin/python"
GATE="$PROJECT/outputs/n5/gate"
FRAMES="${N5_GATE_FRAMES:-100}"

mkdir -p "$GATE" "$GATE/runs"

run_one() {
  local seq=$1 gpu=$2
  N5_PROTOCOL=p3 N5_BUDGET=0 N5_SEQ="$seq" \
    N5_FRAMES="$FRAMES" N5_SKIP_TRACKEVAL=1 \
    N5_OUT_DIR="$GATE/runs/$seq" CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$PROJECT" "$PY" "$PROJECT/scripts/run_n5_continuous_observer.py" \
    > "$GATE/runs/$seq.log" 2>&1
}

run_one dancetrack0004 1 &
PID1=$!
run_one dancetrack0007 2 &
PID2=$!

wait "$PID1" || { echo "GATE_0004_FAIL"; exit 1; }
wait "$PID2" || { echo "GATE_0007_FAIL"; exit 1; }

echo "N5_2_GATE_RUNS_DONE"
