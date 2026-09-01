#!/usr/bin/env bash
# Single blocking monitor for the N21 Phase-II long pipeline.
# Phase A: wait for trajdump 30/30 (stall/interruption detection).
# Phase B: wait for the true-live retrain pipeline (TRUE_LIVE_RETRAIN_DONE).
# Internal sleep/process checks only; no agent-level polling needed.
set -u
ROOT=.
TRJ=$ROOT/outputs/n21/train30_true_onpolicy_trajdump
LOG=$ROOT/outputs/n21/logs
MARKER=$ROOT/outputs/n21/TRUE_LIVE_RETRAIN_DONE

echo "$(date +%F-%T) MONITOR_START"

# ---------- Phase A: trajdump ----------
stall=0
prev=-1
while :; do
  if [ -f "$TRJ/STAGE.done" ]; then
    echo "$(date +%F-%T) PHASE_A_OK done=$(ls "$TRJ"/*.done 2>/dev/null | grep -v STAGE.done | wc -l)/30"
    break
  fi
  n=$(ls "$TRJ"/*.done 2>/dev/null | grep -v STAGE.done | wc -l)
  alive=$(pgrep -f 'run_n21_train30_onpolicy.py.*dump-trajectories' | wc -l)
  if [ "$n" -eq 30 ]; then
    # all sequences done but STAGE.done not yet written; give it a minute
    sleep 60
    continue
  fi
  if [ "$alive" -eq 0 ]; then
    stall=$((stall+1))
    echo "$(date +%F-%T) NO_SHARD_ALIVE n=$n stall=$stall"
    if [ "$stall" -ge 2 ]; then
      echo "$(date +%F-%T) PHASE_A_FAIL n=$n/30 no shard alive"
      exit 2
    fi
  elif [ "$n" -eq "$prev" ]; then
    # check log freshness of all four shards
    old=0
    for g in 5 6 7 8; do
      f=$LOG/trajdump_gpu${g}.log
      if [ -f "$f" ] && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 1800 ]; then
        old=$((old+1))
      fi
    done
    if [ "$old" -ge 4 ]; then
      stall=$((stall+1))
      echo "$(date +%F-%T) ALL_LOGS_STALE n=$n stall=$stall"
      if [ "$stall" -ge 3 ]; then
        echo "$(date +%F-%T) PHASE_A_STALL_FAIL n=$n/30 logs stale >30min"
        exit 2
      fi
    fi
  else
    stall=0
  fi
  prev=$n
  echo "$(date +%F-%T) PHASE_A_WAIT n=$n/30 alive=$alive"
  sleep 1800
done

# ---------- Phase B: retrain pipeline ----------
stall=0
prev_log=""
while :; do
  if [ -f "$MARKER" ]; then
    echo "$(date +%F-%T) PHASE_B_OK retrain done"
    exit 0
  fi
  p=$(pgrep -f 'run_n21_true_live_retrain_pipeline' | wc -l)
  logf=$LOG/true_live_retrain_pipeline_resume2.log
  if [ ! -f "$logf" ]; then logf=$LOG/true_live_retrain_pipeline.log; fi
  if [ "$p" -eq 0 ]; then
    stall=$((stall+1))
    echo "$(date +%F-%T) PIPE_NOT_ALIVE stall=$stall log=$logf"
    if [ "$stall" -ge 2 ]; then
      echo "$(date +%F-%T) PHASE_B_FAIL pipeline process missing"
      exit 3
    fi
  else
    stall=0
  fi
  sleep 1800
done
