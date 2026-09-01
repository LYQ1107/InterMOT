#!/usr/bin/env bash
# N20.3: Oracle Delayed-Commit FULL_LOOP variants (CPU, one blocking command).
set -u

ROOT=.
PY="$ROOT/envs/sam3_intermot/bin/python"
cd "$ROOT"
mkdir -p "$ROOT/outputs/n20/logs"

KS=${KS:-"3 5"}
HORIZONS=${HORIZONS:-"0 1 3 5 8"}
JOBS=${JOBS:-4}

PIDS=()
for k in $KS; do
  for h in $HORIZONS; do
    echo "=== delayed loop k=$k h=$h ==="
    "$PY" scripts/run_n20_oracle_delayed_full_loop.py \
      --split cal10 --k "$k" --horizon "$h" \
      --out-tag "k${k}_h${h}" \
      > "$ROOT/outputs/n20/logs/delayed_k${k}_h${h}.log" 2>&1 &
    PIDS+=("$!")
    while [ "${#PIDS[@]}" -ge "$JOBS" ]; do
      for pid in "${PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
          PIDS=($(printf '%s\n' "${PIDS[@]}" | grep -v "^$pid$"))
        fi
      done
      sleep 10
    done
  done
done

rc=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || rc=1
done
if [ $rc -ne 0 ]; then
  echo "DELAYED_LOOP_FAIL"
  exit 1
fi

"$PY" scripts/analyze_n20_oracle_gate.py \
  > "$ROOT/outputs/n20/logs/oracle_gate_agg.log" 2>&1 || {
    echo "GATE_AGG_FAIL"
    exit 1
  }
echo "ORACLE_DELAYED_STAGE_DONE"
