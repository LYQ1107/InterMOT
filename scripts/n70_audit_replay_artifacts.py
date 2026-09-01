"""Lossless, read-only integrity audit for the N70 replay artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/N70"
ARTIFACT_DIR = OUT / "replay/event_artifacts"
BOUNDARY = OUT / "replay/assignment_boundary.jsonl"
RESULTS = OUT / "replay/paired_replay_results.json"
AUDIT = OUT / "replay/replay_integrity_audit.json"
METHODS = {
    "CURRENT_CCAM_BASELINE", "M0", "M1", "M2", "M3", "M4", "BRANCH_A", "BRANCH_B"
}
VARIANTS = {"M0", "M1", "M2", "M3", "M4"}
# One CURRENT row, five frozen M0--M4 rows, and one A/B row for each of the
# five upstream variants: 1 + 5 + 5 + 5 = 16 rows per frame.
ROWS_PER_VARIANT_FRAME = 1 + len(VARIANTS) + 2 * len(VARIANTS)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def add_error(errors: list[dict[str, Any]], kind: str, detail: str, *, event_id: str | None = None, frame: int | None = None) -> None:
    if len(errors) < 100:
        row: dict[str, Any] = {"kind": kind, "detail": detail}
        if event_id is not None:
            row["event_id"] = event_id
        if frame is not None:
            row["frame"] = frame
        errors.append(row)


def check_matrix(value: Any, rows: int, columns: int, errors: list[dict[str, Any]], label: str, event_id: str, frame: int) -> None:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except Exception as exc:  # pragma: no cover - defensive audit path
        add_error(errors, "matrix_parse", f"{label}: {type(exc).__name__}: {exc}", event_id=event_id, frame=frame)
        return
    if matrix.shape != (rows, columns):
        add_error(errors, "matrix_shape", f"{label}: expected {(rows, columns)}, found {matrix.shape}", event_id=event_id, frame=frame)
    elif not np.all(np.isfinite(matrix)):
        add_error(errors, "matrix_nonfinite", label, event_id=event_id, frame=frame)


def audit_frame(
    row: dict[str, Any],
    event_id: str,
    errors: list[dict[str, Any]],
    frame_counter: Counter[str],
    public_widths: Counter[int],
    variant_axis_mismatch: list[dict[str, Any]],
) -> None:
    frame = int(row.get("frame", -1))
    frame_counter[event_id] += 1
    if row.get("schema") != "N70_PAIRED_REPLAY_FRAME_V1" or row.get("status") != "PASS_RUNTIME_FRAME":
        add_error(errors, "frame_status", f"schema/status invalid: {row.get('schema')}/{row.get('status')}", event_id=event_id, frame=frame)
    for flag in ("runtime_future_gt_used", "target_native_id_sent_to_runtime"):
        if row.get(flag) is not False:
            add_error(errors, "runtime_boundary", f"{flag} is not false", event_id=event_id, frame=frame)
    if row.get("interaction_source") != "simulated_from_gt" or row.get("real_human_tape") is not False or row.get("production_authorized") is not False:
        add_error(errors, "provenance", "synthetic/production flags invalid", event_id=event_id, frame=frame)
    candidates = row.get("candidate_rows_mapping")
    pids = row.get("public_id_order")
    if not isinstance(candidates, list) or not isinstance(pids, list):
        add_error(errors, "axis_missing", "candidate_rows_mapping/public_id_order is not a list", event_id=event_id, frame=frame)
        return
    native_ids = [item.get("native_tid") for item in candidates if isinstance(item, dict)]
    candidate_indexes = [item.get("candidate_index") for item in candidates if isinstance(item, dict)]
    if len(native_ids) != len(set(native_ids)):
        add_error(errors, "duplicate_candidate", "native_tid duplicate", event_id=event_id, frame=frame)
    if len(candidate_indexes) != len(set(candidate_indexes)):
        add_error(errors, "duplicate_candidate", "candidate_index duplicate", event_id=event_id, frame=frame)
    if len(pids) != len(set(pids)):
        add_error(errors, "duplicate_public_id", "public_id_order duplicate", event_id=event_id, frame=frame)
    public_widths[len(pids)] += 1
    mapping = row.get("mapping_audit", {})
    if not mapping.get("candidate_chain_complete", False):
        add_error(errors, "mapping_incomplete", "frame mapping_audit is incomplete", event_id=event_id, frame=frame)
    variants = row.get("variants")
    if not isinstance(variants, dict) or set(variants) != VARIANTS:
        add_error(errors, "variant_missing", f"expected {sorted(VARIANTS)}, found {sorted(variants) if isinstance(variants, dict) else type(variants).__name__}", event_id=event_id, frame=frame)
        return
    m0_native = [item.get("native_tid") for item in candidates if isinstance(item, dict)]
    m0_public = list(pids)
    for variant in sorted(VARIANTS):
        item = variants[variant]
        if not isinstance(item, dict):
            add_error(errors, "variant_invalid", f"{variant} is not an object", event_id=event_id, frame=frame)
            continue
        variant_candidates = item.get("candidate_rows_mapping")
        variant_pids = item.get("public_id_order")
        if not isinstance(variant_candidates, list) or not isinstance(variant_pids, list):
            add_error(errors, "variant_axis_missing", f"{variant} candidate_rows_mapping/public_id_order is not a list", event_id=event_id, frame=frame)
            continue
        variant_native = [entry.get("native_tid") for entry in variant_candidates if isinstance(entry, dict)]
        if variant_native != m0_native or list(variant_pids) != m0_public:
            variant_axis_mismatch.append({"event_id": event_id, "frame": frame, "variant": variant})
        if len(variant_pids) != len(set(variant_pids)):
            add_error(errors, "duplicate_public_id", f"{variant} public_id_order duplicate", event_id=event_id, frame=frame)
        assignment = item.get("assignment_columns")
        if not isinstance(assignment, list) or len(assignment) != len(variant_candidates):
            add_error(errors, "assignment_axis", f"{variant} assignment length mismatch", event_id=event_id, frame=frame)
        else:
            if any(int(column) < -1 or int(column) >= len(variant_pids) for column in assignment):
                add_error(errors, "assignment_column", f"{variant} assignment column out of range", event_id=event_id, frame=frame)
        check_matrix(item.get("score_matrix"), len(variant_candidates), len(variant_pids), errors, f"{variant}.score_matrix", event_id, frame)
        variant_mapping = item.get("mapping_audit", {})
        if not variant_mapping.get("candidate_chain_complete", False) or variant_mapping.get("candidate_count") != len(variant_candidates):
            add_error(errors, "mapping_incomplete", f"{variant} mapping_audit is incomplete", event_id=event_id, frame=frame)
        for branch_key, branch_name in (("branch_a", "A"), ("branch_b", "B")):
            branch = item.get(branch_key)
            if not isinstance(branch, dict):
                add_error(errors, "branch_missing", f"{variant}/{branch_key} missing", event_id=event_id, frame=frame)
                continue
            check_matrix(branch.get("score_matrix"), len(variant_candidates), len(variant_pids), errors, f"{variant}/{branch_name}.score_matrix", event_id, frame)
            sidecar = branch.get("sidecar")
            if not isinstance(sidecar, dict) or sidecar.get("branch") != branch_name:
                add_error(errors, "branch_sidecar", f"{variant}/{branch_key} sidecar invalid", event_id=event_id, frame=frame)
            elif sidecar.get("runtime_future_gt_used") is not False or sidecar.get("target_native_id_sent_to_runtime") is not False or sidecar.get("target_column_only") is not True:
                add_error(errors, "branch_runtime_boundary", f"{variant}/{branch_name} runtime/target scope contract invalid", event_id=event_id, frame=frame)


def main() -> None:
    errors: list[dict[str, Any]] = []
    event_files = sorted(ARTIFACT_DIR.glob("*.jsonl"))
    event_ids: list[str] = []
    frame_counter: Counter[str] = Counter()
    public_widths: Counter[int] = Counter()
    expected_frames: dict[str, set[int]] = {}
    event_sequences: dict[str, str] = {}
    action_counts: Counter[str] = Counter()
    total_candidate_rows = 0
    variant_frame_keys: set[tuple[str, str, int]] = set()
    variant_axis_mismatch: list[dict[str, Any]] = []

    for path in event_files:
        event_id = path.stem
        event_ids.append(event_id)
        seen_frames: set[int] = set()
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except Exception as exc:
                        add_error(errors, "json_parse", f"{type(exc).__name__}: {exc}", event_id=event_id)
                        continue
                    if not isinstance(row, dict):
                        add_error(errors, "row_type", f"line {line_no} is not an object", event_id=event_id)
                        continue
                    frame = int(row.get("frame", -1))
                    if frame in seen_frames:
                        add_error(errors, "duplicate_frame", "duplicate event/frame artifact row", event_id=event_id, frame=frame)
                    seen_frames.add(frame)
                    if row.get("event_id") != event_id:
                        add_error(errors, "event_key", f"embedded event_id={row.get('event_id')}", event_id=event_id, frame=frame)
                    if row.get("frame_horizon") != frame - int(row.get("event_frame", frame)):
                        add_error(errors, "horizon", "frame_horizon mismatch", event_id=event_id, frame=frame)
                    event_sequences[event_id] = str(row.get("sequence"))
                    action_counts[str(row.get("action_type"))] += 1
                    candidates = row.get("candidate_rows_mapping", [])
                    total_candidate_rows += len(candidates) if isinstance(candidates, list) else 0
                    for variant in VARIANTS:
                        variant_frame_keys.add((event_id, variant, frame))
                    audit_frame(row, event_id, errors, frame_counter, public_widths, variant_axis_mismatch)
        except Exception as exc:
            add_error(errors, "artifact_read", f"{type(exc).__name__}: {exc}", event_id=event_id)
        if seen_frames:
            min_frame, max_frame = min(seen_frames), max(seen_frames)
            expected_frames[event_id] = set(range(min_frame, max_frame + 1))
            if seen_frames != expected_frames[event_id]:
                add_error(errors, "frame_gap", f"non-contiguous frames {min_frame}..{max_frame}", event_id=event_id)

    boundary_rows = 0
    boundary_keys: set[tuple[str, int, str, str]] = set()
    boundary_method_counts: Counter[str] = Counter()
    boundary_variant_counts: Counter[str] = Counter()
    boundary_runtime_true = 0
    boundary_nonfinite = 0
    if BOUNDARY.is_file():
        with BOUNDARY.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                boundary_rows += 1
                try:
                    row = json.loads(line)
                except Exception as exc:
                    add_error(errors, "boundary_json_parse", f"line {line_no}: {type(exc).__name__}: {exc}")
                    continue
                if not isinstance(row, dict):
                    add_error(errors, "boundary_row_type", f"line {line_no} is not object")
                    continue
                key = (str(row.get("event_id")), int(row.get("frame", -1)), str(row.get("method")), str(row.get("variant")))
                if key in boundary_keys:
                    add_error(errors, "boundary_duplicate", str(key), event_id=key[0], frame=key[1])
                boundary_keys.add(key)
                boundary_method_counts[key[2]] += 1
                boundary_variant_counts[key[3]] += 1
                if row.get("runtime_future_gt_used") is not False:
                    boundary_runtime_true += 1
                for field in ("max_abs_score_delta", "utility_delta"):
                    value = row.get(field)
                    if value is not None and not finite_number(value):
                        boundary_nonfinite += 1
                        add_error(errors, "boundary_nonfinite", field, event_id=key[0], frame=key[1])
                if row.get("method") not in METHODS or row.get("variant") not in VARIANTS:
                    add_error(errors, "boundary_method_variant", str(key), event_id=key[0], frame=key[1])
                if row.get("candidate_integrity") is not True:
                    add_error(errors, "boundary_candidate_integrity", str(key), event_id=key[0], frame=key[1])
    else:
        add_error(errors, "boundary_missing", str(BOUNDARY))

    result = json.loads(RESULTS.read_text()) if RESULTS.is_file() else {}
    expected_boundary_rows = len(event_files) * 100 * ROWS_PER_VARIANT_FRAME
    expected_variant_keys = len(event_files) * 100 * len(VARIANTS)
    unique_sequences = sorted(set(event_sequences.values()))
    summary = {
        "schema": "N70_REPLAY_INTEGRITY_AUDIT_V1",
        "status": "PASS" if not errors and boundary_rows == expected_boundary_rows and len(boundary_keys) == boundary_rows else "FAIL",
        "created_at_utc": now(),
        "artifacts": {
            "directory": str(ARTIFACT_DIR),
            "file_count": len(event_files),
            "event_count": len(set(event_ids)),
            "duplicate_event_files": len(event_ids) - len(set(event_ids)),
            "frame_rows": sum(frame_counter.values()),
            "expected_frame_rows": len(event_files) * 100,
            "unique_event_variant_frame_keys": len(variant_frame_keys),
            "expected_event_variant_frame_keys": expected_variant_keys,
            "variant_axis_mismatch_frame_count": len(variant_axis_mismatch),
            "variant_axis_mismatch_examples": variant_axis_mismatch[:20],
            "candidate_rows_in_frame_artifacts": total_candidate_rows,
            "public_id_width_distribution": {str(k): int(v) for k, v in sorted(public_widths.items())},
            "action_distribution_by_frame": dict(sorted(action_counts.items())),
            "event_sequence_count": len(unique_sequences),
            "sequences": unique_sequences,
        },
        "boundary": {
            "path": str(BOUNDARY),
            "sha256": sha256(BOUNDARY),
            "row_count": boundary_rows,
            "expected_row_count": expected_boundary_rows,
            "unique_key_count": len(boundary_keys),
            "duplicate_key_count": boundary_rows - len(boundary_keys),
            "method_counts": dict(sorted(boundary_method_counts.items())),
            "variant_counts": dict(sorted(boundary_variant_counts.items())),
            "runtime_future_gt_true_count": boundary_runtime_true,
            "nonfinite_count": boundary_nonfinite,
        },
        "results": {
            "path": str(RESULTS),
            "sha256": sha256(RESULTS),
            "reported_event_count": result.get("event_count"),
            "reported_cache_frame_count": result.get("cache_frame_count"),
            "reported_boundary_row_count": result.get("boundary_row_count"),
            "reported_status": result.get("status"),
            "reported_research_gate": result.get("gate", {}).get("research_gate"),
        },
        "runtime_contract": {
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
            "real_sam3_full_loop": False,
            "production_authorized": False,
        },
        "errors": errors,
    }
    atomic_json(AUDIT, summary)
    print(json.dumps({"status": summary["status"], "artifact_files": len(event_files), "frame_rows": summary["artifacts"]["frame_rows"], "boundary_rows": boundary_rows, "errors_recorded": len(errors), "audit": str(AUDIT)}, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
