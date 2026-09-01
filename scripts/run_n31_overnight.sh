#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${N31_PYTHON_BIN:-python}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-9}"
OUT_DIR="${ROOT_DIR}/outputs/n31"
mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

echo '{"phase":"N31","status":"STARTED"}' > "${OUT_DIR}/overnight_status.json"

if [[ "$("${PYTHON_BIN}" - <<'PY'
import json
print(json.load(open("outputs/n31/resume_equivalence_gate.json"))["status"])
PY
)" != "PASS" ]]; then
  echo "N31-B resume gate is not PASS" >&2
  exit 2
fi

"${PYTHON_BIN}" scripts/n31_id_mapping_regression.py
"${PYTHON_BIN}" scripts/n31_protected_identity_scope.py
"${PYTHON_BIN}" scripts/n31_build_episode_manifest.py

# One command owns the long official experiment.  The ablation script writes
# a partial artifact after each episode and can be safely resumed.
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/n31_correction_state_ablation.py --resume

if [[ "$("${PYTHON_BIN}" - <<'PY'
import json
print(json.load(open("outputs/n31/correction_state_gate.json"))["status"])
PY
)" == "PASS" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/n31_build_candidate_rollouts.py --resume
else
  echo "N31-C state gate failed; candidate oracle phase is not authorized" >&2
fi

"${PYTHON_BIN}" scripts/n31_future_gradient_smoke.py

if [[ -f "${OUT_DIR}/candidate_oracle_gate.json" && "$("${PYTHON_BIN}" - <<'PY'
import json
print(json.load(open("outputs/n31/candidate_oracle_gate.json"))["status"])
PY
)" == "PASS" ]]; then
  "${PYTHON_BIN}" scripts/n31_train_state_selector.py
else
  # Preserve a real Oracle FAIL; the selector runner writes an explicit
  # NOT_RUN_ORACLE_FAIL learn artifact without replacing the scientific gate.
  "${PYTHON_BIN}" scripts/n31_train_state_selector.py
fi

"${PYTHON_BIN}" scripts/n31_fallback.py
"${PYTHON_BIN}" scripts/n31_full_loop_gate.py
echo '{"phase":"N31","status":"COMPLETE"}' > "${OUT_DIR}/overnight_status.json"
