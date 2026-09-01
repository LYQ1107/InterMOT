"""N68 Stage 01: CPU-only identity-scope audit of frozen N67 runtime.

The audit consumes the already-emitted N67/N54 runtime artifacts and the
offline-simulated N37 event manifest.  It does not rerun SAM3, load raw GT,
or change a production association path.  Its purpose is to distinguish a
missing target candidate from an accepted action that was scoped to the
wrong identity.
"""

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
N37_EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
N67_RUNTIME = ROOT / "outputs/n67/replay/runtime"
N67_STATUS = ROOT / "outputs/n67/replay/runtime_status.json"
N67_INTEGRITY = ROOT / "outputs/n67/replay/stage_06_integrity.json"
N54_RUNTIME = ROOT / "outputs/n54/replay/runtime"
OUT = ROOT / "outputs/n68/diagnosis"
ATTEMPTS = ROOT / "outputs/n68/attempts"
SUMMARY = OUT / "stage_01_identity_scope_summary.json"
ROWS = OUT / "stage_01_identity_scope_audit.jsonl"
STATUS = ROOT / "outputs/n68/stage_01_status.json"

REQUIRED_PROVENANCE = {
    "interaction_source": "simulated_from_gt",
    "not_real_human_evidence": True,
    "runtime_future_gt_used": False,
    "real_human_tape": False,
    "real_sam3_full_loop": False,
    "production_authorized": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def unit(value: Any, dim: int = 512) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != dim or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-6 else None


def vector_meta(value: Any) -> dict[str, Any]:
    arr = unit(value)
    if arr is None:
        return {"finite_512d": False, "norm": None, "sha256": None}
    return {
        "finite_512d": True,
        "norm": float(np.linalg.norm(arr)),
        "sha256": hashlib.sha256(arr.astype(np.float32).tobytes()).hexdigest(),
    }


def matrix_meta(matrix: Any) -> dict[str, Any]:
    try:
        arr = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return {"shape": None, "finite": False, "sha256": None}
    return {
        "shape": list(arr.shape),
        "finite": bool(np.all(np.isfinite(arr))),
        "sha256": hashlib.sha256(arr.astype(np.float32).tobytes()).hexdigest(),
    }


def dot_or_none(left: Any, right: Any) -> float | None:
    a, b = unit(left), unit(right)
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def rank_desc(values: list[float], index: int) -> int | None:
    if index < 0 or index >= len(values):
        return None
    value = finite_float(values[index])
    if value is None:
        return None
    return 1 + sum(1 for item in values if finite_float(item) is not None and float(item) > value)


def get_event_map() -> dict[str, dict[str, Any]]:
    manifest = json.loads(N37_EVENTS.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"PASS", "PARTIAL"}:
        raise RuntimeError(f"unexpected N37 event manifest status: {manifest.get('status')}")
    result: dict[str, dict[str, Any]] = {}
    for item in manifest.get("events", []):
        event = item.get("event", {})
        event_id = str(item.get("protocol_candidate_id") or event.get("event_id"))
        if not event_id or not isinstance(event, dict):
            raise RuntimeError("N37 event manifest has an unaddressable event")
        target = event.get("public_id", event.get("canonical_public_id"))
        if target is None:
            raise RuntimeError(f"event {event_id} has no simulated target public_id")
        result[event_id] = {"manifest_item": item, "event": event, "target_public_id": int(target)}
    if len(result) != 24:
        raise RuntimeError(f"expected 24 frozen N37 events, found {len(result)}")
    return result


def branch_public_rows(frame: dict[str, Any], branch: str) -> list[dict[str, Any]]:
    rows = frame.get(branch, {}).get("rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def target_row_from_source(source_frame: dict[str, Any], target: int) -> int | None:
    rows = branch_public_rows(source_frame, "no_write")
    hits = [idx for idx, row in enumerate(rows) if row.get("public_id") == target]
    if len(hits) > 1:
        raise RuntimeError(f"duplicate target public mapping in source frame: {target}")
    return hits[0] if hits else None


def target_physical_row(candidate_rows: list[dict[str, Any]], target_native_tid: int | None) -> int | None:
    if target_native_tid is None:
        return None
    hits = [idx for idx, row in enumerate(candidate_rows) if row.get("native_tid") == target_native_tid]
    if len(hits) > 1:
        raise RuntimeError(f"duplicate target native candidate row: {target_native_tid}")
    return hits[0] if hits else None


def score_cell(matrix: Any, row: int | None, col: int | None) -> float | None:
    if row is None or col is None:
        return None
    try:
        return finite_float(matrix[row][col])
    except (IndexError, TypeError, KeyError):
        return None


def candidate_summary(source_frame: dict[str, Any], row: int | None) -> dict[str, Any] | None:
    candidates = source_frame.get("candidate_features_512", [])
    rows = source_frame.get("no_write", {}).get("candidate_rows", [])
    if row is None or row < 0 or row >= len(candidates):
        return None
    feature = candidates[row]
    meta: dict[str, Any] = {"row": int(row), "feature": vector_meta(feature)}
    if row < len(rows) and isinstance(rows[row], dict):
        meta.update(
            {
                "native_tid": rows[row].get("native_tid"),
                "native_age": finite_float(rows[row].get("native_age")),
                "confidence": finite_float(rows[row].get("confidence")),
                "feature_available": rows[row].get("feature_available"),
            }
        )
    return meta


def analyze_event(n67_path: Path, event_info: dict[str, Any]) -> list[dict[str, Any]]:
    n67 = json.loads(n67_path.read_text(encoding="utf-8"))
    event_id = str(n67.get("event_id"))
    if event_id not in event_info:
        raise RuntimeError(f"runtime event not in frozen manifest: {event_id}")
    event_record = event_info[event_id]
    target = int(event_record["target_public_id"])
    source_path = N54_RUNTIME / n67_path.name
    if not source_path.is_file():
        raise RuntimeError(f"matching N54 source runtime missing: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("event_id") != event_id:
        raise RuntimeError(f"N54/N67 event mismatch: {event_id} vs {source.get('event_id')}")

    rows: list[dict[str, Any]] = []
    variant = n67.get("variants", {}).get("M2", {})
    for frame_data in variant.get("frames", []):
        probe = frame_data.get("probe", {})
        selected = probe.get("selected_action", {})
        if selected.get("accepted") is not True:
            continue
        frame = int(frame_data["frame"])
        local_m2_frame = int(frame_data.get("local_m2_frame", frame - int(n67.get("event_frame", frame))))
        source_frames = source.get("variants", {}).get("M2", {}).get("frames", [])
        source_frame = next((item for item in source_frames if int(item.get("frame", -1)) == frame), None)
        if source_frame is None:
            raise RuntimeError(f"N54 source frame missing for {event_id}/{frame}")

        public_order = [int(x) for x in frame_data.get("memory_public_id_order", [])]
        target_column = public_order.index(target) if target in public_order else None
        endpoint = selected.get("invalid_public_id")
        competitor = selected.get("valid_public_id")
        endpoint = int(endpoint) if endpoint is not None else None
        competitor = int(competitor) if competitor is not None else None
        candidate_rows = frame_data.get("write_baseline", {}).get("candidate_rows", [])
        target_native_raw = event_record["event"].get("target_native_tid")
        target_native_tid = int(target_native_raw) if target_native_raw is not None else None
        target_physical = target_physical_row(candidate_rows, target_native_tid)
        target_mapped = target_row_from_source(source_frame, target)
        # Scores are observation-row indexed.  Prefer the physical target
        # row for target-side diagnostics; the separate mapped row records
        # whether native->public identity mapping itself drifted.
        target_row = target_physical if target_physical is not None else target_mapped
        if target_row is not None and target_row >= len(candidate_rows):
            raise RuntimeError(f"target row outside N67 candidate rows: {event_id}/{frame}/{target_row}")

        n67_before = frame_data.get("write_baseline", {})
        n67_after = frame_data.get("write_plus_n67r1", {})
        source_no_write = source_frame.get("no_write", {})
        source_write = source_frame.get("write_baseline", {})
        endpoint_col = public_order.index(endpoint) if endpoint in public_order else None
        competitor_col = public_order.index(competitor) if competitor in public_order else None
        endpoint_row = int(selected.get("candidate_row"))
        if endpoint_row < 0 or endpoint_row >= len(candidate_rows):
            raise RuntimeError(f"selected candidate row outside N67 rows: {event_id}/{frame}")

        target_feature = source_frame.get("candidate_features_512", [])[target_row] if target_row is not None else None
        target_memory = None
        if target_column is not None:
            vectors = source_frame.get("memory_vectors_512", [])
            if target_column < len(vectors):
                target_memory = vectors[target_column]
        human_feature = event_record["event"].get("human_embedding")
        updated_prototype = None
        memory_unit = unit(target_memory)
        human_unit = unit(human_feature)
        if memory_unit is not None and human_unit is not None:
            updated_prototype = unit(0.8 * memory_unit + 0.2 * human_unit)

        baseline_matrix = n67_before.get("score_matrix", [])
        after_matrix = n67_after.get("score_matrix", [])
        n67_target_scores_before = (
            list(baseline_matrix[target_row]) if target_row is not None and target_row < len(baseline_matrix) else None
        )
        n67_target_scores_after = (
            list(after_matrix[target_row]) if target_row is not None and target_row < len(after_matrix) else None
        )
        selected_before = score_cell(baseline_matrix, endpoint_row, endpoint_col)
        selected_after = score_cell(after_matrix, endpoint_row, endpoint_col)
        competitor_before = score_cell(baseline_matrix, endpoint_row, competitor_col)
        competitor_after = score_cell(after_matrix, endpoint_row, competitor_col)
        source_no_matrix = source_no_write.get("score_matrix", [])
        source_write_matrix = source_write.get("score_matrix", [])
        source_target_col = public_order.index(target) if target in public_order else None
        source_target_before = score_cell(source_no_matrix, target_row, source_target_col)
        source_target_after = score_cell(source_write_matrix, target_row, source_target_col)
        target_feature_meta = vector_meta(target_feature) if target_feature is not None else {"finite_512d": False, "norm": None, "sha256": None}
        candidate_meta = candidate_summary(source_frame, endpoint_row)
        target_physical_exists = target_physical is not None
        target_public_mapping_exists = target_mapped is not None
        mapping_matches_physical = bool(
            target_physical_exists and target_public_mapping_exists and target_physical == target_mapped
        )
        endpoint_is_target = endpoint == target
        competitor_is_target = competitor == target
        if not target_physical_exists:
            bucket = "A_TARGET_CANDIDATE_RECALL_ABSENT"
        elif not mapping_matches_physical:
            bucket = "B_TARGET_PHYSICAL_PRESENT_PUBLIC_MAPPING_MISMATCH"
        elif not endpoint_is_target and not competitor_is_target:
            bucket = "B_TARGET_CANDIDATE_PRESENT_WRONG_ID_SCOPE"
        else:
            # The selected target pair still needs a direction/scale audit;
            # do not infer C without inspecting its target-side deltas.
            target_delta = None
            if n67_target_scores_before is not None and n67_target_scores_after is not None:
                target_delta = float(np.max(np.abs(np.asarray(n67_target_scores_after) - np.asarray(n67_target_scores_before))))
            bucket = "C_TARGET_SCOPE_REACHED_REQUIRES_DIRECTION_SCALE_AUDIT" if target_delta is not None else "C_TARGET_SCOPE_UNRESOLVED"

        frame_row = {
            "schema": "N68_STAGE_01_IDENTITY_SCOPE_AUDIT_ROW_V1",
            "created_at_utc": utc_now(),
            "event_id": event_id,
            "sequence": n67.get("sequence"),
            "action_type": n67.get("action_type"),
            "event_frame": int(n67.get("event_frame")),
            "frame": frame,
            "local_m2_frame": local_m2_frame,
            "target_public_id_simulated": target,
            "target_native_tid_from_offline_event": target_native_tid,
            "target_source": "N37_offline_simulated_event_manifest",
            "selected_action": {
                "candidate_row": endpoint_row,
                "invalid_endpoint_public_id": endpoint,
                "valid_competitor_public_id": competitor,
                "endpoint_is_target": endpoint_is_target,
                "competitor_is_target": competitor_is_target,
                "either_endpoint_or_competitor_is_target": bool(endpoint_is_target or competitor_is_target),
                "pair_is_unrelated_to_target": bool(not endpoint_is_target and not competitor_is_target),
                "action_probability": finite_float(selected.get("action_probability")),
                "magnitude": finite_float(selected.get("magnitude")),
                "base_gap_invalid_minus_valid": finite_float(selected.get("base_gap_invalid_minus_valid")),
                "adjusted_gap_invalid_minus_valid": finite_float(selected.get("adjusted_gap_invalid_minus_valid")),
                "global_assignment_margin": finite_float(selected.get("global_assignment_margin")),
                "gates": selected.get("gates", {}),
            },
            "classification": bucket,
            "candidate_recall_audit": {
                "target_candidate_row": target_physical,
                "target_candidate_exists_physically_by_native_tid": target_physical_exists,
                "target_public_mapping_row": target_mapped,
                "target_candidate_exists_in_frozen_public_mapping": target_public_mapping_exists,
                "target_candidate_visible_as_mapped_row": target_public_mapping_exists,
                "native_to_public_mapping_matches_target": mapping_matches_physical,
                "target_native_tid": target_native_tid,
                "selected_candidate": candidate_meta,
                "target_candidate": candidate_summary(source_frame, target_physical),
                "candidate_count": len(candidate_rows),
                "candidate_native_ids": [row.get("native_tid") for row in candidate_rows if isinstance(row, dict)],
            },
            "mapping_audit": {
                "public_id_order": public_order,
                "target_column": target_column,
                "endpoint_column": endpoint_col,
                "competitor_column": competitor_col,
                "target_row": target_row,
                "target_physical_row": target_physical,
                "target_public_mapping_row": target_mapped,
                "selected_row": endpoint_row,
                "source_no_write_public_rows": [row.get("public_id") for row in branch_public_rows(source_frame, "no_write")],
                "mapping_consistent_n54_n67_candidate_count": len(source_frame.get("candidate_features_512", [])) == len(candidate_rows),
            },
            "score_audit": {
                "n67_write_baseline_matrix": baseline_matrix,
                "n67_write_plus_n67r1_matrix": after_matrix,
                "n67_matrix_meta_before": matrix_meta(baseline_matrix),
                "n67_matrix_meta_after": matrix_meta(after_matrix),
                "target_scores_before": n67_target_scores_before,
                "target_scores_after": n67_target_scores_after,
                "target_rank_before": rank_desc(n67_target_scores_before, target_column) if n67_target_scores_before is not None and target_column is not None else None,
                "target_rank_after": rank_desc(n67_target_scores_after, target_column) if n67_target_scores_after is not None and target_column is not None else None,
                "selected_cell_before": selected_before,
                "selected_cell_after": selected_after,
                "selected_cell_delta": None if selected_before is None or selected_after is None else selected_after - selected_before,
                "competitor_cell_before": competitor_before,
                "competitor_cell_after": competitor_after,
                "competitor_cell_delta": None if competitor_before is None or competitor_after is None else competitor_after - competitor_before,
                "source_memory_no_write_matrix": source_no_matrix,
                "source_memory_write_baseline_matrix": source_write_matrix,
                "source_target_cell_before": source_target_before,
                "source_target_cell_after": source_target_after,
                "source_target_cell_delta": None if source_target_before is None or source_target_after is None else source_target_after - source_target_before,
                "none_columns": [idx for idx, pid in enumerate(public_order) if pid is None],
            },
            "memory_audit": {
                "memory_write_before_target_vector": vector_meta(target_memory) if target_memory is not None else None,
                "human_anchor_vector": vector_meta(human_feature),
                "updated_prototype_approx_0_8_old_plus_0_2_human": vector_meta(updated_prototype) if updated_prototype is not None else None,
                "candidate_target_vs_human_cosine": dot_or_none(target_feature, human_feature),
                "candidate_selected_vs_human_cosine": dot_or_none(source_frame.get("candidate_features_512", [])[endpoint_row], human_feature) if endpoint_row < len(source_frame.get("candidate_features_512", [])) else None,
                "target_vs_memory_cosine_before": dot_or_none(target_feature, target_memory),
                "target_vs_updated_prototype_cosine": dot_or_none(target_feature, updated_prototype),
                "memory_source": frame_data.get("memory_provenance"),
            },
            "assignments": {
                "n67_write_baseline_public_ids": n67_before.get("assignment_public_ids"),
                "n67_write_plus_n67r1_public_ids": n67_after.get("assignment_public_ids"),
                "n67_probe_assignment_if_applied": selected.get("assignment_public_ids_if_applied"),
                "assignment_changed": bool(probe.get("assignment_changed")),
                "id_set_changed": bool(probe.get("id_set_changes")),
                "other_assignment_changes": bool(probe.get("other_assignment_changes")),
            },
            "provenance": {
                **REQUIRED_PROVENANCE,
                "raw_gt_loaded": False,
                "posthoc_gt_loaded": False,
                "event_frame_memory_read": False,
                "first_memory_visible_frame": int(n67.get("event_frame")) + 1,
                "runtime_source": str(n67_path),
                "source_runtime": str(source_path),
                "runtime_future_gt_used_in_source": frame_data.get("runtime_future_gt_used") is False,
            },
        }
        rows.append(frame_row)
    return rows


def main() -> None:
    start = utc_now()
    event_map = get_event_map()
    if not N67_STATUS.is_file() or not N67_INTEGRITY.is_file():
        raise RuntimeError("N67 runtime status/integrity artifact is missing")
    n67_status = json.loads(N67_STATUS.read_text(encoding="utf-8"))
    integrity = json.loads(N67_INTEGRITY.read_text(encoding="utf-8"))
    if n67_status.get("status") != "PASS" or integrity.get("status") != "PASS":
        raise RuntimeError("N68 Stage 01 requires frozen N67 runtime and independent integrity PASS")
    if n67_status.get("runtime_future_gt_used") is not False or integrity.get("runtime_future_gt_used") is not False:
        raise RuntimeError("N67 runtime provenance is not GT-free")
    paths = sorted(N67_RUNTIME.glob("*.json"))
    if len(paths) != 24:
        raise RuntimeError(f"expected 24 N67 runtime event artifacts, found {len(paths)}")
    all_rows: list[dict[str, Any]] = []
    for path in paths:
        all_rows.extend(analyze_event(path, event_map))

    counts = Counter(row["classification"] for row in all_rows)
    action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    sequence_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in all_rows:
        action_counts[str(row["action_type"])][row["classification"]] += 1
        sequence_counts[str(row["sequence"])][row["classification"]] += 1
    accepted_pairs = [row["selected_action"] for row in all_rows]
    endpoint_target_count = sum(bool(x["endpoint_is_target"]) for x in accepted_pairs)
    competitor_target_count = sum(bool(x["competitor_is_target"]) for x in accepted_pairs)
    physical_target_present_count = sum(
        bool(row["candidate_recall_audit"]["target_candidate_exists_physically_by_native_tid"])
        for row in all_rows
    )
    physical_target_absent_count = len(all_rows) - physical_target_present_count
    target_public_mapping_present_count = sum(
        bool(row["candidate_recall_audit"]["target_candidate_exists_in_frozen_public_mapping"])
        for row in all_rows
    )
    target_public_mapping_mismatch_count = sum(
        row["classification"] == "B_TARGET_PHYSICAL_PRESENT_PUBLIC_MAPPING_MISMATCH"
        for row in all_rows
    )
    target_native_mapping_consistent_count = sum(
        bool(row["candidate_recall_audit"]["native_to_public_mapping_matches_target"])
        for row in all_rows
    )
    wrong_id_scope_count = counts.get("B_TARGET_CANDIDATE_PRESENT_WRONG_ID_SCOPE", 0)
    if physical_target_absent_count:
        root_cause_interpretation = "A_CANDIDATE_RECALL_ABSENCE_PRESENT"
    elif target_public_mapping_mismatch_count and wrong_id_scope_count:
        root_cause_interpretation = "B_MAPPING_MISMATCH_AND_B_ID_SCOPE"
    elif target_public_mapping_mismatch_count:
        root_cause_interpretation = "B_NATIVE_PUBLIC_MAPPING_MISMATCH_DOMINANT"
    elif wrong_id_scope_count:
        root_cause_interpretation = "B_ID_SCOPE_DOMINANT_WITH_PHYSICAL_TARGET_PRESENT"
    else:
        root_cause_interpretation = "C_TARGET_SCOPE_REACHED_REQUIRES_DIRECTION_SCALE_AUDIT"
    summary = {
        "schema": "N68_STAGE_01_IDENTITY_SCOPE_SUMMARY_V1",
        "status": "PASS_READONLY_IDENTITY_SCOPE_AUDIT",
        "created_at_utc": utc_now(),
        "started_at_utc": start,
        "event_count": len(paths),
        "accepted_m2_action_count": len(all_rows),
        "events_with_accepted_m2_action": len({row["event_id"] for row in all_rows}),
        "classification_counts": dict(counts),
        "selected_endpoint_equals_target_count": endpoint_target_count,
        "selected_competitor_equals_target_count": competitor_target_count,
        "selected_pair_either_endpoint_or_competitor_target_count": sum(bool(x["endpoint_is_target"] or x["competitor_is_target"]) for x in accepted_pairs),
        "selected_pair_unrelated_to_target_count": sum(bool(not x["endpoint_is_target"] and not x["competitor_is_target"]) for x in accepted_pairs),
        # Physical recall is determined by the native candidate identity, not
        # by whether the current (possibly drifting) public-ID table contains
        # the target.  Keep the mapping counts separate so a mapping defect is
        # never reported as SAM3 candidate absence.
        "physical_target_candidate_present_count": physical_target_present_count,
        "physical_target_candidate_absent_count": physical_target_absent_count,
        "target_public_mapping_present_count": target_public_mapping_present_count,
        "target_public_mapping_mismatch_count": target_public_mapping_mismatch_count,
        "target_native_mapping_consistent_count": target_native_mapping_consistent_count,
        "target_candidate_present_count": physical_target_present_count,
        "target_candidate_absent_count": physical_target_absent_count,
        "by_action_type": {key: dict(value) for key, value in sorted(action_counts.items())},
        "by_sequence": {key: dict(value) for key, value in sorted(sequence_counts.items())},
        "minimum_case": all_rows[0] if all_rows else None,
        "root_cause_interpretation": root_cause_interpretation,
        "input_hashes": {
            "n37_event_manifest_sha256": digest(N37_EVENTS),
            "n67_runtime_tree_sha256_from_status": n67_status.get("runtime_tree_sha256"),
            "n67_integrity_sha256": digest(N67_INTEGRITY),
            "n67_protocol_sha256": n67_status.get("protocol_sha256"),
        },
        "provenance": {
            **REQUIRED_PROVENANCE,
            "raw_gt_loaded": False,
            "posthoc_gt_loaded": False,
            "candidate_stream_reused": True,
            "n67_evidence_modified": False,
        },
    }
    atomic_jsonl(ROWS, all_rows)
    atomic_json(SUMMARY, summary)
    status = {
        "schema": "N68_STAGE_01_STATUS_V1",
        "status": summary["status"],
        "created_at_utc": utc_now(),
        "inputs": {
            "n37_event_manifest": str(N37_EVENTS),
            "n67_runtime": str(N67_RUNTIME),
            "n54_source_runtime": str(N54_RUNTIME),
            "n67_runtime_status": str(N67_STATUS),
            "n67_integrity": str(N67_INTEGRITY),
        },
        "outputs": {"rows": str(ROWS), "summary": str(SUMMARY), "script": str(Path(__file__))},
        "metrics": {
            "event_count": summary["event_count"],
            "accepted_m2_action_count": summary["accepted_m2_action_count"],
            "classification_counts": summary["classification_counts"],
            "physical_target_candidate_present_count": summary["physical_target_candidate_present_count"],
            "physical_target_candidate_absent_count": summary["physical_target_candidate_absent_count"],
            "target_public_mapping_present_count": summary["target_public_mapping_present_count"],
            "target_public_mapping_mismatch_count": summary["target_public_mapping_mismatch_count"],
            "target_native_mapping_consistent_count": summary["target_native_mapping_consistent_count"],
            "target_candidate_present_count": summary["target_candidate_present_count"],
            "target_candidate_absent_count": summary["target_candidate_absent_count"],
        },
        "gate_checks": {
            "n67_runtime_pass": True,
            "n67_independent_integrity_pass": True,
            "all_24_events_loaded": len(paths) == 24,
            "all_accepted_actions_audited": True,
            "target_public_id_explicit_offline_only": True,
            "candidate_mapping_reconciled": all(bool(row["mapping_audit"]["mapping_consistent_n54_n67_candidate_count"]) for row in all_rows),
            "runtime_future_gt_false": all(bool(row["provenance"]["runtime_future_gt_used_in_source"]) for row in all_rows),
            "raw_gt_loaded": False,
            "posthoc_gt_loaded": False,
            "production_authorized": False,
        },
        "first_actionable_root_cause": summary["root_cause_interpretation"],
        "next_action": "Implement isolated identity-scoped sidecar only after preserving this audit; do not change N67 or production path.",
        "failure_evidence_preserved": True,
        "modified_frozen_outputs": False,
        **REQUIRED_PROVENANCE,
    }
    atomic_json(STATUS, status)
    print(json.dumps({"status": status["status"], "summary": str(SUMMARY), "rows": len(all_rows), "classification": dict(counts)}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        ATTEMPTS.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "N68_STAGE_01_FAILURE_V1",
            "status": "FAIL_PRESERVED",
            "created_at_utc": utc_now(),
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "Repair only the first actionable audit defect, preserve this artifact, and rerun the same frozen input.",
            **REQUIRED_PROVENANCE,
        }
        attempt = len(list(ATTEMPTS.glob("stage_01_failure_attempt*.json"))) + 1
        atomic_json(ATTEMPTS / f"stage_01_failure_attempt{attempt}.json", failure)
        atomic_json(ATTEMPTS / "stage_01_failure.json", failure)
        atomic_json(STATUS, failure)
        raise
