#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="."
PYTHON_BIN="python"
cd "$ROOT"
mkdir -p outputs/n32/logs outputs/n32/policy_rollouts outputs/n32/checkpoints
exec > >(tee -a outputs/n32/logs/long_task.log) 2>&1

START_EPOCH="$(date +%s.%N)"
echo "N32_LONG_TASK_START $(date -Is)"

PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_recompute_policy_oracle.py
PYTHONPATH="third_party/sam3:." CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/n32_policy_regression.py

worker_pids=()
worker_gpus=(0 1 2 3)
for worker_index in 0 1 2 3; do
  gpu="${worker_gpus[$worker_index]}"
  log="outputs/n32/logs/policy_worker_$(printf '%02d' "$worker_index").log"
  echo "N32_WORKER_START index=$worker_index gpu=$gpu log=$log"
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_build_policy_rollouts.py --worker-index "$worker_index" --worker-count 4 > "$log" 2>&1 &
  worker_pids+=("$!")
done

worker_failure=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    worker_failure=1
  fi
done
if [[ "$worker_failure" -ne 0 ]]; then
  echo "N32_WORKER_FAILURE"
  "$PYTHON_BIN" scripts/n32_write_run_summary.py --output outputs/n32/run_summary.json --start-epoch "$START_EPOCH" --exit-code 1
  exit 1
fi

PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_merge_policy_rollouts.py
PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_selector_feature_audit.py
PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_policy_oracle_689.py

ORACLE_STATUS="$($PYTHON_BIN -c 'import json; print(json.load(open("outputs/n32/policy_oracle_689.json"))["status"])')"
if [[ "$ORACLE_STATUS" == "PASS" ]]; then
  PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_train_selector.py
  PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_evaluate_selector.py
  LEARN_STATUS="$($PYTHON_BIN -c 'import json; print(json.load(open("outputs/n32/learn_gate.json"))["status"])')"
  OVERFIT_STATUS="$($PYTHON_BIN -c 'import json; print(json.load(open("outputs/n32/overfit_gate.json"))["status"])')"
  if [[ "$LEARN_STATUS" == "PASS" ]]; then
    PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_selector_full_loop.py
  elif [[ "$OVERFIT_STATUS" == "PASS" ]]; then
    PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_temporal_selector.py
    TEMPORAL_STATUS="$($PYTHON_BIN -c 'import json; print(json.load(open("outputs/n32/temporal_learn_gate.json"))["status"])')"
    if [[ "$TEMPORAL_STATUS" == "PASS" ]]; then
      PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_selector_full_loop.py
    else
      PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_selector_full_loop.py
    fi
  else
    PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_selector_full_loop.py
  fi
else
  "$PYTHON_BIN" scripts/n32_mark_not_run.py --output-dir outputs/n32 --reason "N32-C 689-episode policy Oracle failed; selector training is not authorized" --source-status "$ORACLE_STATUS"
  PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_selector_full_loop.py
fi

if [[ ! -f outputs/n32/temporal_learn_gate.json ]]; then
  "$PYTHON_BIN" scripts/n32_mark_temporal_not_run.py --output outputs/n32/temporal_learn_gate.json --reason "temporal fallback was not authorized by the upstream branch"
fi

PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_association_fallback.py
PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_validate_artifacts.py
VALIDATION_STATUS="$($PYTHON_BIN -c 'import json; print(json.load(open("outputs/n32/artifact_validation.json"))["status"])')"
if [[ "$VALIDATION_STATUS" == "PASS" ]]; then
  VALIDATION_EXIT=0
else
  VALIDATION_EXIT=1
fi
"$PYTHON_BIN" scripts/n32_write_run_summary.py --output outputs/n32/run_summary.json --start-epoch "$START_EPOCH" --exit-code "$VALIDATION_EXIT"
PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_write_report.py
PYTHONPATH="third_party/sam3:." "$PYTHON_BIN" scripts/n32_validate_artifacts.py
FINAL_VALIDATION_STATUS="$($PYTHON_BIN -c 'import json; print(json.load(open("outputs/n32/artifact_validation.json"))["status"])')"
if [[ "$FINAL_VALIDATION_STATUS" != "PASS" ]]; then
  "$PYTHON_BIN" scripts/n32_write_run_summary.py --output outputs/n32/run_summary.json --start-epoch "$START_EPOCH" --exit-code 1
  exit 2
fi
"$PYTHON_BIN" scripts/n32_write_run_summary.py --output outputs/n32/run_summary.json --start-epoch "$START_EPOCH" --exit-code 0
echo "N32_LONG_TASK_END $(date -Is)"
