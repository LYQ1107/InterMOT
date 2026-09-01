#!/usr/bin/env bash
# N21 Phase-II autonomous post-trajdump pipeline.
# Waits for the trajectory-dump pass (30/30), then:
#   1. converts live trajdump -> N20 cache format
#   2. concatenates live events (for memory reconstruction)
#   3. builds the true-live K+1 feature CSV
#   4. builds the true-live visual tracklet identity dataset (npz)
#   5. retrains CATIL (base + C1/C2 + cal10 offline evaluation) on the
#      true-live distribution
# GPU use: after trajdump finishes, a single idle GPU (9) is used for the
# retrain/offline experiment (global <=4 GPU constraint respected because
# the trajdump shards have exited).
set -u
ROOT=.
PY=$ROOT/envs/sam3_intermot/bin/python
TRJ=$ROOT/outputs/n21/train30_true_onpolicy_trajdump
LOG=$ROOT/outputs/n21/logs
mkdir -p "$LOG"

while :; do
  n=$(ls "$TRJ"/*.done 2>/dev/null | grep -v STAGE.done | wc -l)
  shards=$(pgrep -f 'run_n21_train30_onpolicy.py.*dump-trajectories' | wc -l)
  if [ "$n" -ge 30 ] && { [ -f "$TRJ/STAGE.done" ] || [ "$shards" -eq 0 ]; }; then
    break
  fi
  echo "$(date +%F-%T) WAIT_TRAJ=$n/30 shards=$shards" \
    >> "$LOG/true_live_retrain_pipeline.log"
  sleep 600
done
echo "TRAJDUMP_30_30 $(date +%F-%T)" >> "$LOG/true_live_retrain_pipeline.log"

# 1. convert
"$PY" "$ROOT/scripts/convert_live_trajdump_to_cache.py" \
  >> "$LOG/true_live_retrain_pipeline.log" 2>&1 || exit 1

# 2. concat events
cat "$TRJ"/../train30_true_onpolicy/events_*.jsonl > \
  "$ROOT/outputs/n21/events_train30_live.jsonl" 2>/dev/null || \
  cat "$ROOT/outputs/n21/train30_true_onpolicy"/events_*.jsonl > \
  "$ROOT/outputs/n21/events_train30_live.jsonl"
echo "EVENTS_CONCAT rows=$(wc -l < "$ROOT/outputs/n21/events_train30_live.jsonl")" \
  >> "$LOG/true_live_retrain_pipeline.log"

# 3. K+1 features
"$PY" "$ROOT/scripts/build_n20_kplus1_dataset.py" \
  --cache-dir live_traj_cache \
  --events-jsonl "$ROOT/outputs/n21/events_train30_live.jsonl" \
  --evidence-steps 5 --k 5 \
  --out features/shadow_kplus1_live_train30.csv \
  >> "$LOG/true_live_retrain_pipeline.log" 2>&1 || exit 1

# 4. visual tracklet npz
"$PY" "$ROOT/scripts/build_n21_tracklet_identity_dataset.py" \
  --h 8 --cache-dir live_traj_cache --out-name live_train30 \
  >> "$LOG/true_live_retrain_pipeline.log" 2>&1 || exit 1

# backup offline-proxy result tables before overwriting
for f in capacity_ladder.csv online_epochs_ablation.csv \
         offline_tracklet_training.csv lora_param_count.csv \
         representation_shift.csv; do
  if [ -f "$ROOT/outputs/n21/$f" ]; then
    cp "$ROOT/outputs/n21/$f" "$ROOT/outputs/n21/offlineproxy_$f"
  fi
done

# 5. retrain + cal10 offline evaluation (single idle GPU after trajdump)
"$PY" "$ROOT/scripts/n21_tracklet_identity_experiment.py" \
  --gpu 9 \
  --train-csv shadow_kplus1_live_train30.csv \
  --cal-csv shadow_kplus1_cal10.csv \
  --train-npz live_train30.npz \
  --cal-npz cal10.npz \
  --model-prefix tracklet_identity_live \
  --offline-epochs 40 --offline-lr 1e-3 \
  --online-epochs 10 --epochs-ablation 5,20 \
  >> "$LOG/true_live_retrain_pipeline.log" 2>&1 || exit 1

echo "TRUE_LIVE_RETRAIN_DONE $(date +%F-%T)" \
  | tee -a "$LOG/true_live_retrain_pipeline.log"
echo "done" > "$ROOT/outputs/n21/TRUE_LIVE_RETRAIN_DONE"
