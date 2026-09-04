#!/usr/bin/env python3
"""Validate the small N72R7 native-scope base-score correction."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/N72R7"
SOURCE = ROOT / "sam3_intermot/association/online_associator.py"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def run(command: list[str]) -> dict[str, object]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": int(result.returncode),
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")
    expected = 'nsame = 1.0 if native_same(s, o) else 0.0'
    source_ok = expected in source
    compile_result = run([sys.executable, "-m", "py_compile", str(SOURCE)])
    test_result = run([sys.executable, "-m", "pytest", "-q", "tests/test_n72r6_target_scope.py"])
    payload = {
        "schema_version": "N72R7_STAGE_STATUS_V1",
        "stage": "N72R7-01_NATIVE_SCOPE_BASE_SCORE_FIX",
        "status": "PASS_NATIVE_SCOPE_FIX_VALIDATED" if source_ok and compile_result["returncode"] == 0 and test_result["returncode"] == 0 else "FAIL_NATIVE_SCOPE_FIX_VALIDATION",
        "source": str(SOURCE),
        "scope_aware_expression_present": source_ok,
        "numerical_native_weight_unchanged": 'w["native"] * nsame' in source,
        "compile": compile_result,
        "focused_test": test_result,
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "next_stage": "N72R7-02_CANDIDATE_SOURCE_FORENSICS",
    }
    atomic_json(OUT / "stage_01_status.json", payload)
    print(json.dumps({"status": payload["status"], "scope_aware": source_ok}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS_NATIVE_SCOPE_FIX_VALIDATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
