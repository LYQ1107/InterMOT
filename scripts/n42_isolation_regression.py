#!/usr/bin/env python3
"""Verify N42 isolation from frozen evidence and the MOT/OVMOT boundary."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "scripts/n42_snapshot.py"
BASELINE = ROOT / "outputs/n42/snapshot/baseline_before_n42.json"
FINAL = ROOT / "outputs/n42/snapshot/final_after_n42.json"
OUT = ROOT / "outputs/n42/isolation_regression.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    from scripts.n36_real_eval_common import atomic_json as write_json

    write_json(path, payload)


def run_command(command: list[str], timeout: int = 300) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": int(result.returncode),
            "timeout": False,
            "stdout": result.stdout[-12000:],
            "stderr": result.stderr[-12000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "timeout": True,
            "stdout": str(exc.stdout)[-12000:] if exc.stdout else "",
            "stderr": str(exc.stderr)[-12000:] if exc.stderr else "",
        }


def by_path(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["path"]): record for record in records}


def compare_records(
    baseline: list[dict[str, Any]],
    current: list[dict[str, Any]],
    *,
    allow_new_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    left, right = by_path(baseline), by_path(current)
    missing = sorted(set(left) - set(right))
    changed = sorted(
        path
        for path in set(left) & set(right)
        if left[path].get("sha256") != right[path].get("sha256")
    )
    unexpected_new = sorted(
        path
        for path in set(right) - set(left)
        if not any(path.startswith(prefix) for prefix in allow_new_prefixes)
    )
    return {
        "status": "PASS" if not missing and not changed and not unexpected_new else "FAIL",
        "baseline_count": len(left),
        "current_count": len(right),
        "missing": missing,
        "changed": changed,
        "unexpected_new": unexpected_new,
        "allowed_new": sorted(
            path for path in set(right) - set(left)
            if any(path.startswith(prefix) for prefix in allow_new_prefixes)
        ),
    }


def compare_tree_inventory(baseline: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, Any]:
    left = {str(row["path"]): row for row in baseline}
    right = {str(row["path"]): row for row in current}
    details = {}
    for path in sorted(set(left) | set(right)):
        a, b = left.get(path), right.get(path)
        details[path] = {
            "same": a == b,
            "baseline": a,
            "current": b,
        }
    return {
        "status": "PASS" if all(row["same"] for row in details.values()) else "FAIL",
        "details": details,
    }


def main() -> None:
    started = now()
    result: dict[str, Any] = {
        "protocol": "N42_MOT_OVMOT_ISOLATION_REGRESSION_V1",
        "status": "FAIL",
        "started_at": started,
        "project_root": str(ROOT),
        "production_files_modified_by_n42": False,
        "n42_write_root": str(ROOT / "outputs/n42"),
    }
    try:
        if not BASELINE.is_file():
            raise FileNotFoundError(BASELINE)
        snapshot_command = [sys.executable, str(SNAPSHOT), "--label", "final_after_n42"]
        snapshot_run = run_command(snapshot_command)
        result["final_snapshot_command"] = snapshot_run
        if snapshot_run["returncode"] != 0 or not FINAL.is_file():
            raise RuntimeError(f"final snapshot failed: {snapshot_run}")
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        final = json.loads(FINAL.read_text(encoding="utf-8"))

        result["project_code"] = compare_records(
            baseline.get("project_code_hashes", []),
            final.get("project_code_hashes", []),
            allow_new_prefixes=("scripts/n42_",),
        )
        result["project_config"] = compare_records(
            baseline.get("project_config_hashes", []), final.get("project_config_hashes", [])
        )
        result["protected_text"] = compare_records(
            baseline.get("protected_text_hashes", []), final.get("protected_text_hashes", [])
        )
        result["shared_checkpoints"] = compare_records(
            baseline.get("shared_checkpoint_hashes", []), final.get("shared_checkpoint_hashes", [])
        )
        result["protected_output_trees"] = compare_tree_inventory(
            baseline.get("protected_output_tree_inventories", []),
            final.get("protected_output_tree_inventories", []),
        )
        baseline_boundary = baseline.get("mot_ovmot_boundary", {})
        final_boundary = final.get("mot_ovmot_boundary", {})
        result["mot_ovmot_boundary"] = {
            "status": "PASS" if (
                baseline_boundary.get("sibling_metadata_inventory") == final_boundary.get("sibling_metadata_inventory")
                and baseline_boundary.get("ovmot_directories_under_interactive") == final_boundary.get("ovmot_directories_under_interactive")
                and final_boundary.get("production_files_modified_by_snapshot") is False
            ) else "FAIL",
            "sibling_root": final_boundary.get("sibling_intermot_root"),
            "sibling_exists": final_boundary.get("sibling_exists"),
            "ovmot_directories_under_interactive": final_boundary.get("ovmot_directories_under_interactive", []),
            "sibling_metadata_unchanged": baseline_boundary.get("sibling_metadata_inventory") == final_boundary.get("sibling_metadata_inventory"),
        }

        import_check = run_command(
            [
                sys.executable,
                "-c",
                "import sam3_intermot.association.appearance_memory, "
                "sam3_intermot.association.online_associator, "
                "sam3_intermot.association.ccam_replay; print('INTERMOT_IMPORT_PASS')",
            ],
            timeout=120,
        )
        result["intermot_import_regression"] = {
            **import_check,
            "status": "PASS" if import_check["returncode"] == 0 else "FAIL",
        }
        compile_check = run_command(
            [
                sys.executable,
                "-m",
                "py_compile",
                *[str(path) for path in sorted((ROOT / "scripts").glob("n42_*.py"))],
            ],
            timeout=120,
        )
        result["n42_script_compile"] = {
            **compile_check,
            "status": "PASS" if compile_check["returncode"] == 0 else "FAIL",
        }
        pytest_check = run_command([sys.executable, "-m", "pytest", "-q", "tests"], timeout=600)
        result["intermot_tests"] = {
            **pytest_check,
            "status": "PASS" if pytest_check["returncode"] == 0 else "FAIL",
            "scope": "existing SAM3_InterMOT tests; no MOT/OVMOT writes",
        }
        sibling = ROOT.parent / "InterMOT"
        if sibling.is_dir() and (sibling / ".git").exists():
            sibling_status = run_command(["git", "-C", str(sibling), "status", "--porcelain"], timeout=60)
            result["sibling_intermot_git_status"] = {
                **sibling_status,
                "status": "PASS" if sibling_status["returncode"] == 0 else "NOT_AVAILABLE",
                "read_only": True,
            }
        else:
            result["sibling_intermot_git_status"] = {
                "status": "NOT_AVAILABLE",
                "reason": "sibling is not a directly inspectable git worktree",
                "read_only": True,
            }
        protected_checks = (
            result["project_code"],
            result["project_config"],
            result["protected_text"],
            result["shared_checkpoints"],
            result["protected_output_trees"],
            result["mot_ovmot_boundary"],
            result["intermot_import_regression"],
            result["n42_script_compile"],
            result["intermot_tests"],
        )
        result["status"] = "PASS" if all(row.get("status") == "PASS" for row in protected_checks) else "FAIL"
        result["finished_at"] = now()
        atomic_json(OUT, result)
        print(json.dumps({"status": result["status"], "output": str(OUT)}, sort_keys=True), flush=True)
    except Exception as exc:
        result["status"] = "FAIL"
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["finished_at"] = now()
        atomic_json(OUT, result)
        raise


if __name__ == "__main__":
    main()
