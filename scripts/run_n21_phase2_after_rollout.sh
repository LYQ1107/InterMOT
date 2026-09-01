#!/usr/bin/env bash
# N21 Phase-II post-rollout CPU pipeline.
# Waits (blocking) until 30/30 train30 .done files exist, then:
#   1. aggregates true-live train30 outputs
#   2. compares true-live vs offline-proxy attempt distributions
# GPU training steps are intentionally NOT started by this script; the
# printed next commands are gated until the user-approved pipeline launch.
set -u
ROOT=.
PY=$ROOT/envs/sam3_intermot/bin/python
OUT=$ROOT/outputs/n21/train30_true_onpolicy
LOG=$ROOT/outputs/n21/logs
mkdir -p "$LOG"

while [ "$(ls "$OUT"/*.done 2>/dev/null | grep -v STAGE.done | wc -l)" -lt 30 ]; do
  echo "$(date +%F-%T) WAITING done=$(ls "$OUT"/*.done 2>/dev/null | grep -v STAGE.done | wc -l)/30" \
    >> "$LOG/phase2_after_rollout.log"
  sleep 600
done

echo "ROLLOUT_30_30_ACHIEVED $(date +%F-%T)" | tee -a "$LOG/phase2_after_rollout.log"
"$PY" "$ROOT/scripts/aggregate_n21_train30_onpolicy.py" \
  >> "$LOG/phase2_after_rollout.log" 2>&1
"$PY" "$ROOT/scripts/build_n21_true_vs_offline_distribution.py" \
  >> "$LOG/phase2_after_rollout.log" 2>&1

cat >> "$LOG/phase2_after_rollout.log" <<'EOF'
NEXT_GATED_GPU_STEPS (do not start while any rollout shard is alive):
  1. rerun live pass with --dump-trajectories (train30, 4 GPUs) to persist
     shadow tracklets for the true on-policy retrain dataset
  2. rebuild tracklet identity dataset + K+1 features from true-live data
  3. retrain CATIL base (C0) on true-live distribution
  4. C1 LoRA / C2 partial-FT offline training + cal10 calibration
  5. FULL_LOOP_N21 (C0/C1/C2/memory-only/offline-retrain) on cal10
EOF
echo "PHASE2_POST_ROLLOUT_DONE" | tee -a "$LOG/phase2_after_rollout.log"
