#!/usr/bin/env python3
"""Audit the frozen N72R10 E1 protected regressions without running a model.

The audit is deliberately posthoc: it reads only sealed N72R10 runtime rows,
the already sealed event metrics, and the frozen train annotations.  It never
changes the solver or any historical artifact.  The output is an evidence
table used to decide whether a research-only target-edge bridge is warranted.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

INPUT_ROOT = ROOT / "outputs/N72R10/stage_07_replay_attempt_02"
EVENT_METRICS = INPUT_ROOT / "event_metrics.jsonl"
OUTPUT = ROOT / "outputs/N72R11/protected_regression_audit.json"
STATUS_OUTPUT = ROOT / "outputs/N72R11/stage_01_status.json"
E0 = "BASELINE_B0"
E1 = "TEMPORAL_CURRENT"
HORIZONS = (20, 50, 100)
IOU_THRESHOLD = 0.50
DATA_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def box_iou(left: Any, right: Any) -> float:
    """Posthoc-only IoU implementation; importing a model is unnecessary here."""

    import numpy as np

    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != 4 or b.size != 4:
        return 0.0
    left_edge, top_edge = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    right_edge, bottom_edge = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    intersection = max(0.0, right_edge - left_edge) * max(0.0, bottom_edge - top_edge)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def load_gt(sequence: str) -> dict[int, dict[int, dict[str, Any]]]:
    """Read train GT only for posthoc reconstruction of sealed outcomes."""

    from collections import defaultdict
    import numpy as np

    path = DATA_ROOT / "train" / str(sequence) / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) < 6:
            raise ValueError(f"malformed GT row {path}:{line_number}")
        frame, identity = int(fields[0]) - 1, int(fields[1])
        x, y, width, height = [float(value) for value in fields[2:6]]
        box = [x, y, x + width, y + height]
        if not np.isfinite(np.asarray(box, dtype=np.float64)).all():
            raise ValueError(f"non-finite GT row {path}:{line_number}")
        result[frame][identity] = {"box": box}
    return result


def _solver_public_map(row: Mapping[str, Any]) -> dict[int, str | None]:
    solver = (row.get("assignment") or {}).get("solver") or {}
    result: dict[int, str | None] = {}
    for item in solver.get("public_assignments", []):
        public_id = item.get("public_id")
        if public_id is None:
            continue
        result[int(public_id)] = None if item.get("candidate_uid") is None else str(item["candidate_uid"])
    return result


def _candidate_by_uid(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["candidate_uid"]): dict(item) for item in row.get("candidate_rows", [])}


def _assignment_row_by_uid(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    solver = (row.get("assignment") or {}).get("solver") or {}
    return {str(item["candidate_uid"]): dict(item) for item in solver.get("assignment_rows", [])}


def _model_score_by_uid(row: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in ((row.get("selection_audit") or {}).get("ranked_candidates") or []):
        if item.get("candidate_uid") is not None and item.get("model_score") is not None:
            result[str(item["candidate_uid"])] = float(item["model_score"])
    return result


def _target_edge(row: Mapping[str, Any], uid: str | None) -> float | None:
    if uid is None:
        return None
    candidate = _candidate_by_uid(row).get(str(uid))
    if candidate is None:
        return None
    index = int(candidate.get("candidate_index", -1))
    values = (row.get("score_audit") or {}).get("base_target_scores") or []
    if index < 0 or index >= len(values):
        return None
    value = values[index]
    return None if value is None else float(value)


def _public_iou(row: Mapping[str, Any], public_id: int, gt_box: Any) -> tuple[float, str | None, dict[str, Any] | None]:
    mapping = _solver_public_map(row)
    uid = mapping.get(int(public_id))
    if uid is None:
        return 0.0, None, None
    candidate = _candidate_by_uid(row).get(uid)
    if candidate is None:
        return 0.0, uid, None
    return float(box_iou(candidate.get("box_xyxy"), gt_box)), uid, candidate


def _event_root_from_metrics(item: Mapping[str, Any]) -> Path:
    raw = item.get("runtime_event_sealed")
    if raw is None:
        raise KeyError(f"event metrics row lacks runtime_event_sealed: {item.get('event_id')}")
    path = Path(str(raw))
    if not path.is_absolute():
        path = ROOT / path
    root = path.parent
    if not (root / E0 / "runtime_frames.jsonl").is_file():
        raise FileNotFoundError(f"sealed runtime rows unavailable: {root}")
    return root


def audit() -> dict[str, Any]:
    if not EVENT_METRICS.is_file():
        raise FileNotFoundError(EVENT_METRICS)
    metric_rows = read_jsonl(EVENT_METRICS)
    regressions: list[dict[str, Any]] = []
    evidence_events: list[dict[str, Any]] = []
    input_hashes = {str(EVENT_METRICS): sha256_file(EVENT_METRICS)}
    for metric_item in metric_rows:
        e1_metric = metric_item.get("metrics", {}).get("E1_vs_E0", {})
        expected = max(int(e1_metric.get(str(h), {}).get("protected_regression_count", 0)) for h in HORIZONS)
        if expected <= 0:
            continue
        event_id = str(metric_item["event_id"])
        posthoc_path = Path(str(metric_item["posthoc"]))
        if not posthoc_path.is_absolute():
            posthoc_path = ROOT / posthoc_path
        posthoc = read_json(posthoc_path)
        input_hashes[str(posthoc_path)] = sha256_file(posthoc_path)
        event = posthoc["event"]
        sequence = str(metric_item["sequence"])
        gt = load_gt(sequence)
        event_frame = int(metric_item["event_frame"])
        target_gid = int(metric_item["target_public_id"] if "target_public_id" in metric_item else event["target_dataset_gt_id"])
        target_gid = int(event["target_dataset_gt_id"])
        target_public = int(event["target_public_id"])
        protected_map = {int(gid): int(pid) for gid, pid in event["protected_public_by_gt_posthoc"].items()}
        root = _event_root_from_metrics(metric_item)
        variant_rows: dict[str, dict[int, dict[str, Any]]] = {}
        for variant in (E0, E1):
            path = root / variant / "runtime_frames.jsonl"
            input_hashes[str(path)] = sha256_file(path)
            variant_rows[variant] = {int(row["frame"]): row for row in read_jsonl(path)}
        event_regressions = 0
        for frame in range(event_frame + 1, event_frame + 101):
            e0_row = variant_rows[E0].get(frame)
            e1_row = variant_rows[E1].get(frame)
            if e0_row is None or e1_row is None:
                raise RuntimeError(f"missing sealed frame row: {event_id}:{frame}")
            e0_map, e1_map = _solver_public_map(e0_row), _solver_public_map(e1_row)
            e1_assignment_by_uid = _assignment_row_by_uid(e1_row)
            e1_model = _model_score_by_uid(e1_row)
            e0_target_iou, e0_target_uid, e0_target_candidate = _public_iou(
                e0_row, target_public, gt.get(frame, {}).get(target_gid, {}).get("box", [])
            ) if gt.get(frame, {}).get(target_gid) is not None else (0.0, e0_map.get(target_public), None)
            e1_target_iou, e1_target_uid, e1_target_candidate = _public_iou(
                e1_row, target_public, gt.get(frame, {}).get(target_gid, {}).get("box", [])
            ) if gt.get(frame, {}).get(target_gid) is not None else (0.0, e1_map.get(target_public), None)
            for protected_gid, protected_public in sorted(protected_map.items()):
                gt_item = gt.get(frame, {}).get(protected_gid)
                if gt_item is None:
                    continue
                e0_iou, e0_uid, e0_candidate = _public_iou(e0_row, protected_public, gt_item["box"])
                e1_iou, e1_uid, e1_candidate = _public_iou(e1_row, protected_public, gt_item["box"])
                if e0_iou < IOU_THRESHOLD or e1_iou >= IOU_THRESHOLD:
                    continue
                event_regressions += 1
                target_candidate = e1_target_candidate or (_candidate_by_uid(e1_row).get(e1_target_uid) if e1_target_uid else None)
                target_uid = e1_target_uid
                target_assigned_to_protected_public = e1_map.get(protected_public) == target_uid and target_uid is not None
                target_stole_protected_incumbent = bool(
                    target_assigned_to_protected_public
                    or (target_uid is not None and target_uid == e0_uid)
                )
                protected_assignment = e1_assignment_by_uid.get(e1_uid or "")
                cause = (
                    "TARGET_EDGE_OVER_STEALS_PROTECTED_INCUMBENT"
                    if target_stole_protected_incumbent
                    else "GLOBAL_SOLVER_COMPETITION_OR_OTHER"
                )
                regressions.append(
                    {
                        "event_id": event_id,
                        "sequence": sequence,
                        "event_frame": event_frame,
                        "action_type": str(event["action_type"]),
                        "frame": int(frame),
                        "horizon": int(frame - event_frame),
                        "target_dataset_gt_id": target_gid,
                        "target_public_id": target_public,
                        "protected_dataset_gt_id": int(protected_gid),
                        "protected_public_id": int(protected_public),
                        "e0_protected_candidate_uid": e0_uid,
                        "e0_protected_iou": float(e0_iou),
                        "e1_protected_candidate_uid": e1_uid,
                        "e1_protected_iou": float(e1_iou),
                        "e0_target_candidate_uid": e0_target_uid,
                        "e0_target_iou": float(e0_target_iou),
                        "e1_target_candidate_uid": e1_target_uid,
                        "e1_target_iou": float(e1_target_iou),
                        "target_e1_candidate": {
                            "source": None if target_candidate is None else target_candidate.get("candidate_source"),
                            "source_kind": None if target_candidate is None else target_candidate.get("source_kind"),
                            "incumbent_public_id_if_any": None if target_candidate is None else target_candidate.get("incumbent_public_id_if_any"),
                            "candidate_uid": target_uid,
                        },
                        "protected_e0_candidate": None if e0_candidate is None else {
                            "source": e0_candidate.get("candidate_source"),
                            "source_kind": e0_candidate.get("source_kind"),
                            "incumbent_public_id_if_any": e0_candidate.get("incumbent_public_id_if_any"),
                            "candidate_uid": e0_uid,
                        },
                        "protected_e1_candidate": None if e1_candidate is None else {
                            "source": e1_candidate.get("candidate_source"),
                            "source_kind": e1_candidate.get("source_kind"),
                            "incumbent_public_id_if_any": e1_candidate.get("incumbent_public_id_if_any"),
                            "candidate_uid": e1_uid,
                        },
                        "target_model_score": None if target_uid is None else e1_model.get(target_uid),
                        "legacy_target_edge": _target_edge(e1_row, target_uid),
                        "protected_legacy_edge": _target_edge(e1_row, e1_uid),
                        "e0_global_assignment": {str(k): v for k, v in sorted(e0_map.items())},
                        "e1_global_assignment": {str(k): v for k, v in sorted(e1_map.items())},
                        "protected_public_e1_candidate_uid": e1_map.get(protected_public),
                        "target_public_e1_candidate_uid": e1_map.get(target_public),
                        "target_assigned_to_protected_public": bool(target_assigned_to_protected_public),
                        "target_stole_protected_incumbent": bool(target_stole_protected_incumbent),
                        "protected_e1_solver_assignment": protected_assignment,
                        "classification": cause,
                        "runtime_future_gt_used": False,
                        "posthoc_gt_used": True,
                    }
                )
        evidence_events.append(
            {
                "event_id": event_id,
                "sequence": sequence,
                "action_type": str(event["action_type"]),
                "expected_protected_regressions": expected,
                "audited_protected_regressions": event_regressions,
                "protected_public_by_gt_posthoc": {str(k): v for k, v in sorted(protected_map.items())},
            }
        )
        if event_regressions != expected:
            raise RuntimeError(
                f"protected regression count mismatch for {event_id}: audited={event_regressions} expected={expected}"
            )
    if len(regressions) != sum(item["audited_protected_regressions"] for item in evidence_events):
        raise RuntimeError("internal protected regression count mismatch")
    cause_counts: dict[str, int] = {}
    for row in regressions:
        cause_counts[row["classification"]] = cause_counts.get(row["classification"], 0) + 1
    result = {
        "schema_version": "N72R11_PROTECTED_REGRESSION_AUDIT_V1",
        "status": "PASS_PROTECTED_REGRESSIONS_RECONSTRUCTED",
        "created_at_utc": now_utc(),
        "input": {
            "event_metrics": str(EVENT_METRICS),
            "event_metrics_sha256": sha256_file(EVENT_METRICS),
            "historical_outputs_read_only": True,
            "model_executed": False,
        },
        "expected_from_frozen_metrics": {
            "event_count_with_regression": len(evidence_events),
            "protected_regression_count": len(regressions),
            "by_horizon": {
                str(h): sum(int(item.get("metrics", {}).get("E1_vs_E0", {}).get(str(h), {}).get("protected_regression_count", 0)) for item in metric_rows)
                for h in HORIZONS
            },
        },
        "audited": {
            "event_count_with_regression": len(evidence_events),
            "protected_regression_count": len(regressions),
            "cause_counts": cause_counts,
            "all_runtime_future_gt_used_false": all(not row["runtime_future_gt_used"] for row in regressions),
        },
        "event_evidence": evidence_events,
        "regressions": regressions,
        "root_cause_decision": (
            "TARGET_EDGE_OVER_STEALS_PROTECTED_INCUMBENT"
            if cause_counts.get("TARGET_EDGE_OVER_STEALS_PROTECTED_INCUMBENT", 0) == len(regressions) and regressions
            else "MIXED_OR_NON_TARGET_EDGE_CAUSE"
        ),
        "input_sha256": input_hashes,
    }
    return result


def main() -> int:
    started = now_utc()
    try:
        result = audit()
        atomic_json(OUTPUT, result)
        atomic_json(
            STATUS_OUTPUT,
            {
                "schema_version": "N72R11_STAGE_STATUS_V1",
                "stage": "N72R11-01",
                "status": "PASS_PROTECTED_REGRESSION_AUDIT",
                "started_at_utc": started,
                "finished_at_utc": now_utc(),
                "command": "python scripts/n72r11_audit_current_protected_regressions.py",
                "output": str(OUTPUT),
                "output_sha256": sha256_file(OUTPUT),
                "protected_regression_count": result["audited"]["protected_regression_count"],
                "root_cause_decision": result["root_cause_decision"],
                "model_executed": False,
                "historical_outputs_modified": False,
                "runtime_future_gt_used": False,
            },
        )
        print(json.dumps({"status": "PASS_PROTECTED_REGRESSION_AUDIT", "output": str(OUTPUT), "count": result["audited"]["protected_regression_count"]}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = OUTPUT.parent / "attempts" / "protected_regression_audit.failure.json"
        atomic_json(
            failure,
            {
                "schema_version": "N72R11_STAGE_FAILURE_V1",
                "stage": "N72R11-01",
                "status": "FAIL_PROTECTED_REGRESSION_AUDIT",
                "started_at_utc": started,
                "finished_at_utc": now_utc(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": __import__("traceback").format_exc(),
                "historical_outputs_modified": False,
            },
        )
        print(json.dumps({"status": "FAIL_PROTECTED_REGRESSION_AUDIT", "failure": str(failure), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
