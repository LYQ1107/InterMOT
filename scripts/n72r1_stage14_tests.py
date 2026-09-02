#!/usr/bin/env python3
"""Run N72R1 focused and full CPU regression in the pinned environment."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N72R1_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R1")
PYTHON = Path("/home/lwr/anaconda3/envs/intermot/bin/python")
FOCUSED = [
    "tests/test_n72r1_provenance.py",
    "tests/test_n72r1_assignment.py",
    "tests/test_n72r1_actions_runtime.py",
    "tests/test_n72r1_recorder.py",
    "tests/test_n72r1_ui.py",
]


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(label: str, args: list[str]) -> dict[str, object]:
    command = [str(PYTHON), "-m", "pytest", "-q", *args]
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    output = result.stdout + ("\n" + result.stderr if result.stderr else "")
    log = N72R1_ROOT / "tests" / f"{label}.log"
    atomic_text(log, output)
    return {
        "label": label,
        "command": " ".join(command),
        "exit_code": int(result.returncode),
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "log": str(log),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "environment": str(PYTHON),
    }


def main() -> None:
    focused = run("focused_pytest_attempt1", FOCUSED)
    full = run("full_pytest_attempt1", [])
    source_hashes = {path: sha256(ROOT / path) for path in FOCUSED}
    payload = {
        "schema_version": "N72R1_STAGE_STATUS_V1",
        "stage": "N72R1-14",
        "status": "PASS_FOCUSED_AND_FULL_CPU_REGRESSION" if focused["status"] == "PASS" and full["status"] == "PASS" else "FAIL_TEST_REGRESSION",
        "focused": focused,
        "full": full,
        "test_fixture_scope": "toy/non-scientific for new N72R1 tests",
        "historical_wrong_interpreter_failure": str(N72R1_ROOT / "attempts" / "focused_pytest_wrong_interpreter_attempt1.json"),
        "input_hashes": source_hashes,
        "runtime_future_gt_used": False,
        "production_formula_modified": False,
        "third_party_sam3_modified": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    stage_path = N72R1_ROOT / "status" / "stage_14_status.json"
    atomic_json(stage_path, payload)
    # Preserve every attempt instead of overwriting a prior failed status.
    attempts = sorted((N72R1_ROOT / "attempts").glob("stage_14_attempt*.json"))
    attempt_path = N72R1_ROOT / "attempts" / f"stage_14_attempt{len(attempts) + 1}.json"
    atomic_json(attempt_path, payload)
    print(json.dumps({"status": payload["status"], "focused": focused["status"], "full": full["status"], "path": str(stage_path)}, sort_keys=True))
    raise SystemExit(0 if payload["status"].startswith("PASS") else 1)


if __name__ == "__main__":
    main()
