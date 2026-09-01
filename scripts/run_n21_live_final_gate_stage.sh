#!/usr/bin/env bash
# N21 Phase-III true-live cal10 FINAL GATE (L0/L1/L2/L3).
# Global GPU limit: max 4. This stage uses at most 3 GPUs (5,6,8).
# Variants run in parallel (L0/L1/L2), then L3 replaces the first finisher.
# Per-sequence .done resume; each shard retried up to 3 times; an exit code
# is tolerated if all sequences of the shard already have .done markers.
set -u
ROOT=.
PY=$ROOT/envs/sam3_intermot/bin/python
RUN=$ROOT/scripts/run_n21_live_final_gate.py
OUTROOT=$ROOT/outputs/n21/live_final_gate
LOG=$ROOT/outputs/n21/logs
KPLUS=$ROOT/outputs/n20/models/kplus1_gru.pt
CATIL=$ROOT/outputs/n21/models/tracklet_identity_live_base.pt
SEQS="dancetrack0074,dancetrack0075,dancetrack0080,dancetrack0082,dancetrack0083,dancetrack0086,dancetrack0087,dancetrack0096,dancetrack0098,dancetrack0099"
mkdir -p "$OUTROOT" "$LOG"

all_done() {
  local dir=$1
  local n=0
  for s in ${SEQS//,/ }; do
    [ -f "$dir/$s.done" ] && n=$((n+1))
  done
  [ "$n" -eq 10 ]
}

run_variant() {
  local gpu=$1
  local variant=$2
  local extra=""
  local lr=1e-4
  local odir="$OUTROOT/$variant"
  mkdir -p "$odir"
  if [ "$variant" = "L0" ]; then
    extra=""
  else
    extra="--catil-model $CATIL"
  fi
  if [ "$variant" = "L2" ]; then lr=1e-4; fi
  if [ "$variant" = "L3" ]; then lr=3e-5; fi
  for attempt in 1 2 3; do
    echo "$(date +%F-%T) $variant GPU$gpu ATTEMPT=$attempt START" >> "$LOG/live_gate_stage.out"
    "$PY" "$RUN" --gpu "$gpu" --seqs "$SEQS" --split cal10 \
      --variant "$variant" $extra \
      --kplus1-model "$KPLUS" \
      --online-lr "$lr" --online-epochs 10 --online-replay 32 --kl-lambda 2.0 \
      --out-dir "outputs/n21/live_final_gate/$variant" \
      >> "$LOG/live_gate_${variant}_gpu${gpu}.log" 2>&1
    rc=$?
    echo "$(date +%F-%T) $variant GPU$gpu ATTEMPT=$attempt RC=$rc" >> "$LOG/live_gate_stage.out"
    if [ "$rc" -eq 0 ] || all_done "$odir"; then
      return 0
    fi
  done
  echo "$(date +%F-%T) $variant GPU$gpu FAILED_3_ATTEMPTS" >> "$LOG/live_gate_stage.out"
  return 1
}

run_variant 5 L0 >> "$LOG/live_gate_stage.out" 2>&1 & p0=$!
run_variant 6 L1 >> "$LOG/live_gate_stage.out" 2>&1 & p1=$!
run_variant 8 L2 >> "$LOG/live_gate_stage.out" 2>&1 & p2=$!

wait "$p0"; r0=$?
run_variant 5 L3 >> "$LOG/live_gate_stage.out" 2>&1 & p3=$!

wait "$p1"; r1=$?
wait "$p2"; r2=$?
wait "$p3"; r3=$?

echo "GATE_RC L0=$r0 L1=$r1 L2=$r2 L3=$r3 $(date +%F-%T)" \
  | tee -a "$LOG/live_gate_stage.out"
if [ "$r0" -eq 0 ] && [ "$r1" -eq 0 ] && [ "$r2" -eq 0 ] && [ "$r3" -eq 0 ]; then
  "$PY" "$ROOT/scripts/analyze_n21_live_final_gate.py" \
    >> "$LOG/live_gate_stage.out" 2>&1
  echo "done" > "$OUTROOT/STAGE.done"
  echo "LIVE_GATE_STAGE_DONE" | tee -a "$LOG/live_gate_stage.out"
  exit 0
else
  echo "LIVE_GATE_STAGE_PARTIAL" | tee -a "$LOG/live_gate_stage.out"
  exit 1
fi
