#!/usr/bin/env bash
# N21 Phase-III: run the missing L3 (CATIL C2 partial-FT online) variant on
# cal10 with resume-safe per-sequence .done and up to 3 retries. Uses a
# single idle GPU (5). After completion, aggregates the full gate.
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
  local n=0
  for s in ${SEQS//,/ }; do
    [ -f "$OUTROOT/L3/$s.done" ] && n=$((n+1))
  done
  [ "$n" -eq 10 ]
}

for attempt in 1 2 3; do
  echo "$(date +%F-%T) L3 GPU5 ATTEMPT=$attempt START" >> "$LOG/live_gate_stage.out"
  "$PY" "$RUN" --gpu 5 --seqs "$SEQS" --split cal10 --variant L3 \
    --catil-model "$CATIL" --kplus1-model "$KPLUS" \
    --online-lr 3e-5 --online-epochs 10 --online-replay 32 --kl-lambda 2.0 \
    --out-dir "outputs/n21/live_final_gate/L3" \
    >> "$LOG/live_gate_L3_gpu5.log" 2>&1
  rc=$?
  echo "$(date +%F-%T) L3 GPU5 ATTEMPT=$attempt RC=$rc" >> "$LOG/live_gate_stage.out"
  if [ "$rc" -eq 0 ] || all_done; then break; fi
done

if all_done; then
  "$PY" "$ROOT/scripts/analyze_n21_live_final_gate.py" \
    >> "$LOG/live_gate_stage.out" 2>&1
  echo "done" > "$OUTROOT/STAGE.done"
  echo "LIVE_GATE_L3_DONE $(date +%F-%T)" | tee -a "$LOG/live_gate_stage.out"
  exit 0
else
  echo "LIVE_GATE_L3_PARTIAL" | tee -a "$LOG/live_gate_stage.out"
  exit 1
fi
