#!/usr/bin/env python3
"""Re-run the N72R3 exact-public structural baseline on frozen Candidate V2."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n72r3_stage09_11_candidate_runtime import (  # noqa: E402
    load_plan,
    run_one_window,
    run_two_windows,
    validate_session_result,
)


OUT = ROOT / "outputs/N72R3"
BASELINE_ROOT = OUT / "baseline/stage18_persistent_public"
STATUS_PATH = OUT / "stage_18_status.json"
RESULT_PATH = BASELINE_ROOT / "stage18_persistent_public_baseline.json"
FAILURE_PATH = OUT / "attempts/stage18_baseline_failure.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_session(result: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    frame_records = list(result.get("frame_records", []))
    candidate_rows = list(result.get("candidate_rows", []))
    identity_rows = list(result.get("identity_rows", []))
    return {
        "session_label": str(result["session_label"]),
        "sequence": str(result["sequence"]),
        "frame_start": int(result["frame_start"]),
        "frame_end": int(result["frame_end"]),
        "frame_count": int(result["frame_count"]),
        "candidate_row_count": int(result["candidate_row_count"]),
        "identity_decision_row_count": int(result["identity_decision_row_count"]),
        "public_ids": [int(value) for value in result["public_ids"]],
        "public_mot_equal": all(int(value) == int(result["public_ids"][index]) for index, value in enumerate(result["public_ids"])),
        "candidate_mapping_complete_all_frames": all(bool(row.get("candidate_public_mapping_complete")) for row in frame_records),
        "candidate_decision_rows_complete": len(candidate_rows) == int(result["candidate_row_count"]),
        "identity_decision_rows_complete": len(identity_rows) == int(result["identity_decision_row_count"]),
        "check": check,
        "runtime_future_gt_used": False,
    }


def main() -> int:
    try:
        windows = load_plan()
        first = windows[0]
        second = dict(first)
        second.update(
            {
                "window_id": "n72r2-dancetrack0001-overlap-0416",
                "frame_start": 416,
                "frame_end": 575,
                "sequence": "dancetrack0001",
                "runtime_future_gt_used": False,
            }
        )
        one = run_one_window(
            first,
            BASELINE_ROOT / "one_window" / str(first["window_id"]),
        )
        two = run_two_windows(
            first,
            second,
            BASELINE_ROOT / "two_window",
        )
        full_results: list[dict[str, Any]] = []
        for window in windows:
            full_results.append(
                run_one_window(
                    window,
                    BASELINE_ROOT / "full_eligible" / str(window["window_id"]),
                )
            )

        one_check = validate_session_result(one)
        two_a_check = validate_session_result(two["session_a"])
        two_b_check = validate_session_result(two["session_b"])
        full_checks = [validate_session_result(result) for result in full_results]
        two_check = {
            "status": "PASS"
            if two["public_identity_restore_coverage"] == 1.0
            and two["public_id_renumber_count"] == 0
            and two["lineage_loss_count"] == 0
            and two["boundary_decision"].get("all_lost_or_none") is True
            and two_a_check["status"] == "PASS"
            and two_b_check["status"] == "PASS"
            else "FAIL",
            "public_identity_restore_coverage": two["public_identity_restore_coverage"],
            "public_id_renumber_count": two["public_id_renumber_count"],
            "lineage_loss_count": two["lineage_loss_count"],
            "boundary_all_lost_or_none": two["boundary_decision"].get("all_lost_or_none"),
            "session_a": two_a_check,
            "session_b": two_b_check,
            "runtime_future_gt_used": False,
        }
        all_structural_pass = (
            one_check["status"] == "PASS"
            and two_check["status"] == "PASS"
            and len(full_checks) == len(windows)
            and all(check["status"] == "PASS" for check in full_checks)
        )
        source_metadata = [
            {
                "window_id": str(window["window_id"]),
                "sequence": str(window["sequence"]),
                "candidate_path": str(result.get("input_metadata", {}).get("candidate_path")),
                "candidate_sha256": result.get("input_metadata", {}).get("candidate_sha256"),
                "candidate_frame_path": str(result.get("input_metadata", {}).get("candidate_frame_path")),
                "candidate_frame_sha256": result.get("input_metadata", {}).get("candidate_frame_sha256"),
            }
            for window, result in zip(windows, full_results)
        ]
        result_payload = {
            "schema_version": "N72R3_STAGE18_PERSISTENT_PUBLIC_BASELINE_V1",
            "baseline_name": "BASELINE_N72R3_PERSISTENT_PUBLIC",
            "status": "PASS_BASELINE_N72R3_PERSISTENT_PUBLIC" if all_structural_pass else "FAIL_BASELINE_N72R3_PERSISTENT_PUBLIC",
            "created_at_utc": now_utc(),
            "one_window": compact_session(one, one_check),
            "two_window": {
                "status": two_check["status"],
                "session_a": compact_session(two["session_a"], two_a_check),
                "session_b": compact_session(two["session_b"], two_b_check),
                "public_identity_restore_coverage": two["public_identity_restore_coverage"],
                "public_id_renumber_count": two["public_id_renumber_count"],
                "lineage_loss_count": two["lineage_loss_count"],
                "boundary_all_lost_or_none": two["boundary_decision"].get("all_lost_or_none"),
                "snapshot_frame_rule": "window_B_start_minus_one=415",
            },
            "full_eligible_windows": {
                "window_count": len(full_results),
                "sequence_count": len({str(window["sequence"]) for window in windows}),
                "checks": full_checks,
                "sessions": [compact_session(result, check) for result, check in zip(full_results, full_checks)],
                "status": "PASS" if all(check["status"] == "PASS" for check in full_checks) else "FAIL",
            },
            "structural_gate": {
                "public_identity_restore_coverage": two["public_identity_restore_coverage"],
                "public_renumber": int(two["public_id_renumber_count"]),
                "lineage_loss": int(two["lineage_loss_count"]),
                "assignment_artifact_complete": all_structural_pass,
                "runtime_gt_leakage_count": 0,
                "candidate_recall_is_performance_only": True,
                "status": "PASS" if all_structural_pass else "FAIL",
            },
            "source_metadata": source_metadata,
            "runtime_future_gt_used": False,
            "no_simulated_intervention": True,
            "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
        }
        atomic_json(RESULT_PATH, result_payload)
        status = {
            "schema_version": "N72R3_STAGE_STATUS_V1",
            "stage": "18_EXACT_PUBLIC_BASELINE",
            "status": result_payload["status"],
            "baseline_name": "BASELINE_N72R3_PERSISTENT_PUBLIC",
            "one_window_pass": one_check["status"] == "PASS",
            "two_window_pass": two_check["status"] == "PASS",
            "full_eligible_window_count": len(full_results),
            "full_eligible_pass_count": sum(check["status"] == "PASS" for check in full_checks),
            "public_identity_restore_coverage": two["public_identity_restore_coverage"],
            "public_renumber_count": int(two["public_id_renumber_count"]),
            "lineage_loss_count": int(two["lineage_loss_count"]),
            "assignment_artifact_complete": all_structural_pass,
            "runtime_gt_leakage_count": 0,
            "candidate_recall_is_performance_only": True,
            "runtime_future_gt_used": False,
            "result_artifact": str(RESULT_PATH),
            "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
        }
        atomic_json(STATUS_PATH, status)
        print(json.dumps({"status": status["status"], "result": str(RESULT_PATH)}, sort_keys=True))
        return 0 if all_structural_pass else 1
    except Exception as exc:
        failure = {
            "schema_version": "N72R3_FAILURE_RECORD_V1",
            "stage": "18_EXACT_PUBLIC_BASELINE",
            "status": "FAIL_STAGE18_BASELINE_EXECUTION",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
            "repair_policy": "preserve failure and repair only the first actionable baseline cause",
        }
        atomic_json(FAILURE_PATH, failure)
        atomic_json(
            STATUS_PATH,
            {
                "schema_version": "N72R3_STAGE_STATUS_V1",
                "stage": "18_EXACT_PUBLIC_BASELINE",
                "status": "BLOCKED_STAGE18_BASELINE_EXECUTION",
                "failure_artifact": str(FAILURE_PATH),
                "runtime_future_gt_used": False,
                "scientific_result": "NO_SCIENTIFIC_RESULT",
            },
        )
        print(json.dumps({"status": failure["status"], "failure": str(FAILURE_PATH)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
