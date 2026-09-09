"""Lossless export and post-hoc bookkeeping for N72R11R5.

This module deliberately does not run SAM3 or alter any runtime artifact.  It
resolves the already sealed N72R11R4 manifest chain, exports intervention-local
MOT windows, and leaves metric computation to the pinned TrackEval checkout.
"""

from __future__ import annotations

import configparser
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HORIZONS = (20, 50, 100)
LOGICAL_VARIANTS = ("E0_BASELINE_B0", "E1A_V3", "E1B_PCTIS")
FROZEN_METRICS = {
    "E1A_V3": "outputs/N72R11R4/formal_e1a_metrics.json",
    "E1B_PCTIS": "outputs/N72R11R4/formal_e1b_metrics_attempt_02.json",
}


class WindowTrackEvalError(RuntimeError):
    """Raised when a sealed input or an exported window is not auditable."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    """Convert finite scalar values to JSON-safe values without hiding errors."""
    if isinstance(value, float) and not math.isfinite(value):
        raise WindowTrackEvalError(f"non-finite value cannot be serialized: {value!r}")
    return value


def write_json_atomic(path: Path | str, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_text_atomic(path: Path | str, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path | str) -> dict[str, Any]:
    location = Path(path)
    if not location.is_file():
        raise WindowTrackEvalError(f"missing JSON artifact: {location}")
    try:
        value = json.loads(location.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WindowTrackEvalError(f"invalid JSON artifact {location}: {exc}") from exc
    if not isinstance(value, dict):
        raise WindowTrackEvalError(f"expected JSON object: {location}")
    return value


def _resolve_recorded_path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise WindowTrackEvalError(f"missing recorded path: {value!r}")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _require_hash(path: Path, expected: Any, label: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise WindowTrackEvalError(f"{label} has no valid recorded SHA-256: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise WindowTrackEvalError(
            f"{label} SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def _assert_false_runtime_flags(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"} and nested:
                raise WindowTrackEvalError(f"runtime GT flag is true at {location}.{key}")
            _assert_false_runtime_flags(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_false_runtime_flags(nested, f"{location}[{index}]")


def _recorded_runtime_entry(
    entry: Mapping[str, Any], *, event_id: str, logical_variant: str, root: Path
) -> dict[str, Any]:
    frames = _resolve_recorded_path(entry.get("frames"), root)
    if not frames.is_file():
        raise WindowTrackEvalError(
            f"{event_id}/{logical_variant}: missing sealed runtime frames {frames}"
        )
    _require_hash(frames, entry.get("frames_sha256"), f"{event_id}/{logical_variant}/frames")
    if entry.get("runtime_future_gt_used") or entry.get("runtime_gt_read"):
        raise WindowTrackEvalError(f"{event_id}/{logical_variant}: runtime GT flag is true")
    if entry.get("posthoc_gt_used"):
        raise WindowTrackEvalError(f"{event_id}/{logical_variant}: posthoc GT flag is true in runtime manifest")
    return dict(entry)


def _load_source_manifest(metrics_path: Path, root: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    metrics = _read_json(metrics_path)
    if metrics.get("complete") is not True:
        raise WindowTrackEvalError(f"sealed metrics are not complete: {metrics_path}")
    if metrics.get("runtime_future_gt_used") is not False:
        raise WindowTrackEvalError(f"metrics runtime GT flag is not false: {metrics_path}")
    manifest_path = _resolve_recorded_path(metrics.get("source_manifest"), root)
    _require_hash(manifest_path, metrics.get("source_manifest_sha256"), "source manifest")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "PASS_ALL_SELECTED":
        raise WindowTrackEvalError(f"source manifest is not PASS_ALL_SELECTED: {manifest_path}")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 32:
        raise WindowTrackEvalError(f"source manifest must contain 32 records: {manifest_path}")
    return metrics, manifest_path, manifest


def _load_protocol(metrics: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], Path]:
    protocol_path = _resolve_recorded_path(metrics.get("protocol"), root)
    _require_hash(protocol_path, metrics.get("protocol_sha256"), "frozen protocol")
    protocol = _read_json(protocol_path)
    events = protocol.get("source_event_selection", {}).get("events")
    if not isinstance(events, list) or len(events) != 32:
        raise WindowTrackEvalError("frozen protocol does not contain 32 source events")
    return protocol, protocol_path


def _load_event_runtime(
    record: Mapping[str, Any], *, root: Path, logical_treatment: str
) -> dict[str, Any]:
    event_id = record.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise WindowTrackEvalError(f"manifest record has invalid event_id: {record!r}")
    done_path = _resolve_recorded_path(record.get("done"), root)
    _require_hash(done_path, record.get("done_sha256"), f"{event_id}/done")
    done = _read_json(done_path)
    if not str(done.get("status", "")).startswith("PASS_"):
        raise WindowTrackEvalError(f"{event_id}: done artifact is not PASS")
    sealed_path = _resolve_recorded_path(done.get("runtime_event_sealed"), root)
    _require_hash(sealed_path, done.get("runtime_event_sealed_sha256"), f"{event_id}/sealed runtime")
    sealed = _read_json(sealed_path)
    if not str(sealed.get("status", "")).startswith("PASS_"):
        raise WindowTrackEvalError(f"{event_id}: sealed runtime artifact is not PASS")
    if sealed.get("runtime_future_gt_used") or sealed.get("runtime_gt_read"):
        raise WindowTrackEvalError(f"{event_id}: sealed runtime has future-GT usage")
    runtime_manifests = sealed.get("runtime_manifests")
    if not isinstance(runtime_manifests, dict):
        raise WindowTrackEvalError(f"{event_id}: sealed runtime has no runtime_manifests")
    actual_names = list(runtime_manifests)
    if len(actual_names) != 2:
        raise WindowTrackEvalError(f"{event_id}: expected E0 plus one treatment, got {actual_names}")

    # The first entry is the actual baseline name recorded by the frozen manifest;
    # no runtime variant name is invented here.
    manifest_variants = record.get("variants")
    baseline_name = None
    if isinstance(manifest_variants, list) and manifest_variants:
        candidate = manifest_variants[0]
        if candidate in runtime_manifests:
            baseline_name = candidate
    if baseline_name is None:
        baseline_name = actual_names[0]
    treatment_names = [name for name in actual_names if name != baseline_name]
    if len(treatment_names) != 1:
        raise WindowTrackEvalError(f"{event_id}: cannot resolve one treatment from {actual_names}")
    treatment_name = treatment_names[0]
    if treatment_name != logical_treatment:
        # The caller passes the actual treatment name found in the paired source.
        raise WindowTrackEvalError(
            f"{event_id}: treatment name mismatch {treatment_name!r} != {logical_treatment!r}"
        )

    entries = {
        baseline_name: _recorded_runtime_entry(
            runtime_manifests[baseline_name], event_id=event_id, logical_variant=baseline_name, root=root
        ),
        treatment_name: _recorded_runtime_entry(
            runtime_manifests[treatment_name], event_id=event_id, logical_variant=treatment_name, root=root
        ),
    }
    event_frame = entries[baseline_name].get("event_frame")
    sequence = entries[baseline_name].get("sequence")
    if not isinstance(event_frame, int) or not isinstance(sequence, str):
        raise WindowTrackEvalError(f"{event_id}: invalid event frame or sequence in runtime manifest")
    for actual_name, entry in entries.items():
        if entry.get("event_id") != event_id or entry.get("event_frame") != event_frame:
            raise WindowTrackEvalError(f"{event_id}: inconsistent event metadata in {actual_name}")
        if entry.get("sequence") != sequence:
            raise WindowTrackEvalError(f"{event_id}: inconsistent sequence in {actual_name}")
        if entry.get("frame_count") != 101:
            raise WindowTrackEvalError(f"{event_id}/{actual_name}: expected 101 sealed frames")
    return {
        "event_id": event_id,
        "sequence": sequence,
        "event_frame": event_frame,
        "action_type": record.get("action_type"),
        "record": dict(record),
        "done_path": str(done_path),
        "sealed_path": str(sealed_path),
        "baseline_name": baseline_name,
        "treatment_name": treatment_name,
        "entries": entries,
    }


def load_frozen_sources(root: Path | str) -> dict[str, Any]:
    """Resolve and validate the exact E1A/E1B -> sealed runtime chain."""
    project_root = Path(root).resolve()
    loaded: dict[str, Any] = {}
    for logical, relative in FROZEN_METRICS.items():
        metrics_path = project_root / relative
        metrics, manifest_path, manifest = _load_source_manifest(metrics_path, project_root)
        protocol, protocol_path = _load_protocol(metrics, project_root)
        loaded[logical] = {
            "metrics_path": metrics_path,
            "metrics": metrics,
            "manifest_path": manifest_path,
            "manifest": manifest,
            "protocol_path": protocol_path,
            "protocol": protocol,
        }

    first_protocol = loaded["E1A_V3"]["protocol"]
    if loaded["E1B_PCTIS"]["metrics"].get("protocol_sha256") != loaded["E1A_V3"]["metrics"].get("protocol_sha256"):
        raise WindowTrackEvalError("E1A and E1B do not reference the same frozen protocol hash")
    protocol_events = {
        event["event_id"]: event
        for event in first_protocol["source_event_selection"]["events"]
        if isinstance(event, dict) and isinstance(event.get("event_id"), str)
    }
    if len(protocol_events) != 32:
        raise WindowTrackEvalError("frozen protocol event IDs are not unique")

    for logical in ("E1A_V3", "E1B_PCTIS"):
        manifest = loaded[logical]["manifest"]
        actual_variants = manifest.get("variants")
        if not isinstance(actual_variants, list) or len(actual_variants) != 2:
            raise WindowTrackEvalError(f"{logical}: source manifest has no single treatment variant")
        actual_treatment = [name for name in actual_variants if name != actual_variants[0]]
        if len(actual_treatment) != 1 or not all(isinstance(name, str) for name in actual_variants):
            raise WindowTrackEvalError(f"{logical}: source manifest has no single treatment variant")
        loaded[logical]["actual_treatment_name"] = actual_treatment[0]

    # Build the paired event table from the actual manifest records and verify
    # E0 bytes/metadata are identical across the two sealed formal replays.
    e1a_records = {r["event_id"]: r for r in loaded["E1A_V3"]["manifest"]["records"]}
    e1b_records = {r["event_id"]: r for r in loaded["E1B_PCTIS"]["manifest"]["records"]}
    if set(e1a_records) != set(e1b_records) or len(e1a_records) != 32:
        raise WindowTrackEvalError("E1A/E1B event sets are not the same 32 unique events")
    events: list[dict[str, Any]] = []
    for event_id in [r["event_id"] for r in loaded["E1A_V3"]["manifest"]["records"]]:
        a = _load_event_runtime(
            e1a_records[event_id], root=project_root,
            logical_treatment=loaded["E1A_V3"]["actual_treatment_name"],
        )
        b = _load_event_runtime(
            e1b_records[event_id], root=project_root,
            logical_treatment=loaded["E1B_PCTIS"]["actual_treatment_name"],
        )
        if a["sequence"] != b["sequence"] or a["event_frame"] != b["event_frame"]:
            raise WindowTrackEvalError(f"{event_id}: E1A/E1B event metadata differ")
        a_e0 = a["entries"][a["baseline_name"]]
        b_e0 = b["entries"][b["baseline_name"]]
        if a_e0.get("frames_sha256") != b_e0.get("frames_sha256"):
            raise WindowTrackEvalError(f"{event_id}: E0 sealed runtime bytes differ across source metrics")
        frozen_event = protocol_events.get(event_id)
        if frozen_event is None:
            raise WindowTrackEvalError(f"{event_id}: missing from frozen source protocol")
        if frozen_event.get("sequence") != a["sequence"] or frozen_event.get("event_frame") != a["event_frame"]:
            raise WindowTrackEvalError(f"{event_id}: runtime metadata differs from frozen event protocol")
        if frozen_event.get("action_type") != a["action_type"]:
            raise WindowTrackEvalError(f"{event_id}: action type differs from frozen event protocol")
        if frozen_event.get("runtime_future_gt_used") is not False:
            raise WindowTrackEvalError(f"{event_id}: frozen protocol runtime GT flag is not false")
        events.append(
            {
                "event_id": event_id,
                "sequence": a["sequence"],
                "event_frame": a["event_frame"],
                "action_type": a["action_type"],
                "split": frozen_event.get("split", "train"),
                "frozen_event": frozen_event,
                "E0_BASELINE_B0": {
                    "sealed_runtime_variant": a["baseline_name"],
                    "entry": a_e0,
                    "frames_sha256": a_e0["frames_sha256"],
                },
                "E1A_V3": {
                    "sealed_runtime_variant": a["treatment_name"],
                    "entry": a["entries"][a["treatment_name"]],
                    "frames_sha256": a["entries"][a["treatment_name"]]["frames_sha256"],
                },
                "E1B_PCTIS": {
                    "sealed_runtime_variant": b["treatment_name"],
                    "entry": b["entries"][b["treatment_name"]],
                    "frames_sha256": b["entries"][b["treatment_name"]]["frames_sha256"],
                },
            }
        )
    return {
        "root": project_root,
        "events": events,
        "sources": loaded,
        "protocol": first_protocol,
        "protocol_path": loaded["E1A_V3"]["protocol_path"],
        "protocol_sha256": loaded["E1A_V3"]["metrics"].get("protocol_sha256"),
    }


def _load_runtime_rows(entry: Mapping[str, Any], event_id: str, variant: str) -> list[dict[str, Any]]:
    path = Path(str(entry["frames"]))
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WindowTrackEvalError(f"{event_id}/{variant}: invalid runtime JSONL line {line_number}") from exc
            if not isinstance(row, dict):
                raise WindowTrackEvalError(f"{event_id}/{variant}: runtime line {line_number} is not an object")
            _assert_false_runtime_flags(row, f"{event_id}/{variant}/line{line_number}")
            rows.append(row)
    event_frame = entry.get("event_frame")
    expected = list(range(int(event_frame), int(event_frame) + 101))
    observed = [row.get("frame") for row in rows]
    if observed != expected:
        raise WindowTrackEvalError(
            f"{event_id}/{variant}: sealed frame axis is not event..event+100: {observed[:2]}..{observed[-2:]}"
        )
    event_rows = [row for row in rows if row.get("frame") == event_frame]
    if len(event_rows) != 1 or event_rows[0].get("record_kind") != "event_frame_correction":
        raise WindowTrackEvalError(f"{event_id}/{variant}: event-frame correction row is missing")
    if event_rows[0].get("event_frame_memory_read") is not False or event_rows[0].get("memory_read") is not False:
        raise WindowTrackEvalError(f"{event_id}/{variant}: event frame reads memory")
    first_future = rows[1]
    if first_future.get("frame") != int(event_frame) + 1:
        raise WindowTrackEvalError(f"{event_id}/{variant}: no event+1 runtime row")
    return rows


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise WindowTrackEvalError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise WindowTrackEvalError(f"{label} is not finite: {result!r}")
    return result


def _prediction_lines(
    rows: Sequence[Mapping[str, Any]], *, event_frame: int, horizon: int, event_id: str, variant: str
) -> tuple[list[str], dict[str, int]]:
    selected = [row for row in rows if event_frame < int(row.get("frame", -1)) <= event_frame + horizon]
    observed = [int(row.get("frame", -1)) for row in selected]
    expected = list(range(event_frame + 1, event_frame + horizon + 1))
    if observed != expected:
        raise WindowTrackEvalError(f"{event_id}/{variant}/H{horizon}: prediction frame axis is incomplete")
    output: list[str] = []
    constant_score_count = 0
    assigned_count = 0
    for runtime_row in selected:
        frame = int(runtime_row["frame"]) - event_frame
        candidates = runtime_row.get("candidate_rows")
        if not isinstance(candidates, list):
            raise WindowTrackEvalError(f"{event_id}/{variant}/frame{frame}: candidate_rows is not a list")
        seen_candidate_uids: set[str] = set()
        seen_public_ids: set[int] = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise WindowTrackEvalError(f"{event_id}/{variant}/frame{frame}: candidate row is not an object")
            uid = candidate.get("candidate_uid")
            if not isinstance(uid, str) or not uid:
                raise WindowTrackEvalError(f"{event_id}/{variant}/frame{frame}: candidate UID missing")
            if uid in seen_candidate_uids:
                raise WindowTrackEvalError(f"{event_id}/{variant}/frame{frame}: duplicate candidate UID {uid}")
            seen_candidate_uids.add(uid)
            if candidate.get("solver_status") != "ASSIGNED_TO_PUBLIC_ID":
                continue
            public_id = candidate.get("solver_public_id")
            if isinstance(public_id, bool) or not isinstance(public_id, int):
                raise WindowTrackEvalError(f"{event_id}/{variant}/frame{frame}: invalid solver public ID")
            if candidate.get("public_id") != public_id or candidate.get("assigned_public_id") != public_id:
                raise WindowTrackEvalError(f"{event_id}/{variant}/frame{frame}: public-ID fields disagree")
            if candidate.get("public_id_authority") != "exact_global_solver_output":
                raise WindowTrackEvalError(f"{event_id}/{variant}/frame{frame}: public ID authority is not exact solver")
            if candidate.get("public_id_inference") is not False:
                raise WindowTrackEvalError(f"{event_id}/{variant}/frame{frame}: inferred public ID is not allowed")
            if public_id in seen_public_ids:
                raise WindowTrackEvalError(
                    f"{event_id}/{variant}/frame{frame}: duplicate (frame, public_id) {public_id}"
                )
            seen_public_ids.add(public_id)
            box = candidate.get("box_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                raise WindowTrackEvalError(f"{event_id}/{variant}/frame{frame}: invalid box")
            x1, y1, x2, y2 = [_finite_float(value, f"{event_id}/{variant}/frame{frame}/box") for value in box]
            if x2 <= x1 or y2 <= y1:
                raise WindowTrackEvalError(f"{event_id}/{variant}/frame{frame}: non-positive box")
            if "confidence" not in candidate:
                score = 1.0
                constant_score_count += 1
            else:
                score = _finite_float(candidate["confidence"], f"{event_id}/{variant}/frame{frame}/confidence")
            output.append(
                f"{frame},{public_id},{x1:.6f},{y1:.6f},{x2 - x1:.6f},{y2 - y1:.6f},{score:.6f},-1,-1,-1"
            )
            assigned_count += 1
    return output, {
        "assigned_row_count": assigned_count,
        "constant_score_row_count": constant_score_count,
    }


def _invalid_assigned_boxes(
    rows: Sequence[Mapping[str, Any]], *, event_frame: int, event_id: str, variant: str
) -> list[dict[str, Any]]:
    """Report invalid exact-solver rows without changing or filtering them."""
    invalid: list[dict[str, Any]] = []
    for runtime_row in rows:
        frame = runtime_row.get("frame")
        if not isinstance(frame, int) or frame <= event_frame:
            continue
        for candidate in runtime_row.get("candidate_rows") or []:
            if not isinstance(candidate, dict) or candidate.get("solver_status") != "ASSIGNED_TO_PUBLIC_ID":
                continue
            box = candidate.get("box_xyxy")
            valid = (
                isinstance(box, list)
                and len(box) == 4
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in box
                )
                and box[2] > box[0]
                and box[3] > box[1]
            )
            if not valid:
                invalid.append(
                    {
                        "event_id": event_id,
                        "sealed_runtime_variant": variant,
                        "original_frame": frame,
                        "relative_future_frame": frame - event_frame,
                        "solver_public_id": candidate.get("solver_public_id"),
                        "candidate_uid": candidate.get("candidate_uid"),
                        "geometry_valid": candidate.get("geometry_valid"),
                        "box_xyxy": box,
                        "solver_status": candidate.get("solver_status"),
                    }
                )
    return invalid


def _gt_window_lines(gt_path: Path, *, event_frame: int, horizon: int, event_id: str) -> list[str]:
    lines: list[str] = []
    frames: set[int] = set()
    if not gt_path.is_file():
        raise WindowTrackEvalError(f"{event_id}/H{horizon}: GT file missing: {gt_path}")
    with gt_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            parts = [part.strip() for part in stripped.split(",")]
            if len(parts) < 8:
                raise WindowTrackEvalError(f"{gt_path}:{line_number}: GT has fewer than 8 columns")
            try:
                original_frame = int(float(parts[0]))
            except ValueError as exc:
                raise WindowTrackEvalError(f"{gt_path}:{line_number}: invalid GT frame") from exc
            if event_frame < original_frame <= event_frame + horizon:
                new_frame = original_frame - event_frame
                parts[0] = str(new_frame)
                lines.append(",".join(parts))
                frames.add(new_frame)
    expected = set(range(1, horizon + 1))
    if frames != expected:
        missing = sorted(expected - frames)
        raise WindowTrackEvalError(f"{event_id}/H{horizon}: GT frame axis incomplete, missing {missing[:8]}")
    return lines


def _pseudo_name(event_id: str, horizon: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", event_id).strip("_")
    return f"iw_h{horizon:03d}_{safe}"


def _seqinfo_text(source_path: Path, pseudo: str, horizon: int) -> str:
    parser = configparser.ConfigParser()
    parser.read(source_path, encoding="utf-8")
    if "Sequence" not in parser:
        raise WindowTrackEvalError(f"missing [Sequence] in {source_path}")
    section = parser["Sequence"]
    if "imWidth" not in section or "imHeight" not in section:
        raise WindowTrackEvalError(f"missing image dimensions in {source_path}")
    # Keep the original metadata values and change only the pseudo-sequence
    # identity and length required by the TrackEval dataset adapter.
    values = {
        "name": pseudo,
        "imDir": section.get("imDir", "img1"),
        "frameRate": section.get("frameRate", "20"),
        "seqLength": str(horizon),
        "imWidth": section["imWidth"],
        "imHeight": section["imHeight"],
        "imExt": section.get("imExt", ".jpg"),
    }
    return "[Sequence]\n" + "".join(f"{key}={value}\n" for key, value in values.items())


def _source_gt_and_seqinfo(data_root: Path, event: Mapping[str, Any]) -> tuple[Path, Path]:
    split = event.get("split")
    sequence = event["sequence"]
    if split not in {"train", "validation", "val", "test"}:
        raise WindowTrackEvalError(f"{event['event_id']}: unsupported frozen split {split!r}")
    split_dir = "val" if split == "validation" else str(split)
    sequence_dir = data_root / split_dir / sequence
    return sequence_dir / "gt" / "gt.txt", sequence_dir / "seqinfo.ini"


def build_protocol(
    sources: Mapping[str, Any], *, output_root: Path, source_branch: str, source_commit: str,
    trackeval_root: Path, data_root: Path, events: Sequence[Mapping[str, Any]], mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": "N72R11R5_WINDOW_TRACKEVAL_V1",
        "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
        "official_dancetrack_benchmark_score": False,
        "mode": mode,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "source_e1a_metrics": str(sources["E1A_V3"]["metrics_path"]),
        "source_e1a_metrics_sha256": sha256_file(sources["E1A_V3"]["metrics_path"]),
        "source_e1b_metrics": str(sources["E1B_PCTIS"]["metrics_path"]),
        "source_e1b_metrics_sha256": sha256_file(sources["E1B_PCTIS"]["metrics_path"]),
        "source_e1a_manifest": str(sources["E1A_V3"]["manifest_path"]),
        "source_e1b_manifest": str(sources["E1B_PCTIS"]["manifest_path"]),
        "frozen_protocol": str(sources["E1A_V3"]["protocol_path"]),
        "frozen_protocol_sha256": sources["E1A_V3"]["metrics"].get("protocol_sha256"),
        "trackeval_root": str(trackeval_root),
        "trackeval_commit": None,
        "data_root": str(data_root),
        "horizons": list(HORIZONS),
        "event_count": len(events),
        "event_ids": [event["event_id"] for event in events],
        "independent_sequence_count": len({event["sequence"] for event in events}),
        "window_start": "event_frame+1",
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "runtime_future_gt_used": False,
        "gt_used_only_for_posthoc_evaluation": True,
        "bootstrap_seed": 7211,
        "bootstrap_repetitions": 2000,
        "prediction_score_policy": "runtime_confidence_else_constant_1",
        "variants": list(LOGICAL_VARIANTS),
        "sealed_runtime_variants": {
            logical: sorted(
                {
                    event[logical]["sealed_runtime_variant"]
                    for event in events
                }
            )
            for logical in LOGICAL_VARIANTS
        },
        "source_event_order_is_frozen": True,
        "e2_evaluated": False,
        "e2_status": "NOT_MEASURED_INCOMPLETE_FORMAL_REPLAY",
    }


def export_windows(
    *, root: Path | str, data_root: Path | str, source_branch: str, source_commit: str,
    trackeval_root: Path | str, event_ids: Iterable[str] | None = None, mode: str = "full",
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    output_root = Path(root).resolve()
    dataset_root = Path(data_root).resolve()
    trackeval_path = Path(trackeval_root).resolve()
    source_root = Path(project_root).resolve() if project_root is not None else output_root.parent.parent
    frozen = load_frozen_sources(source_root)
    wanted = set(event_ids) if event_ids is not None else None
    events = [event for event in frozen["events"] if wanted is None or event["event_id"] in wanted]
    if wanted is not None and len(events) != len(wanted):
        missing = sorted(wanted - {event["event_id"] for event in events})
        raise WindowTrackEvalError(f"requested event IDs are not in sealed manifests: {missing}")
    if not events:
        raise WindowTrackEvalError("no events selected for export")
    events_by_id = {event["event_id"]: event for event in events}
    if list(events_by_id) != [event["event_id"] for event in events]:
        raise WindowTrackEvalError("duplicate event IDs after selection")

    # Preflight every sealed future row before writing a PASS manifest.  An
    # assigned candidate with a non-positive box cannot be repaired in a
    # post-hoc exporter: clipping, dropping, or inventing a one-pixel box would
    # change the sealed solver output and invalidate the diagnostic.
    runtime_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    invalid_boxes: list[dict[str, Any]] = []
    for event in events:
        for logical in LOGICAL_VARIANTS:
            entry = event[logical]["entry"]
            runtime_cache[(event["event_id"], logical)] = _load_runtime_rows(
                entry, event["event_id"], event[logical]["sealed_runtime_variant"]
            )
            invalid_boxes.extend(
                _invalid_assigned_boxes(
                    runtime_cache[(event["event_id"], logical)],
                    event_frame=event["event_frame"],
                    event_id=event["event_id"],
                    variant=event[logical]["sealed_runtime_variant"],
                )
            )
    if invalid_boxes:
        write_json_atomic(
            output_root / "export_failure_invalid_assigned_boxes.json",
            {
                "schema_version": "N72R11R5_EXPORT_FAILURE_V1",
                "status": "BLOCKED_INVALID_SEALED_ASSIGNED_BOXES",
                "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
                "reason": "sealed exact-solver assigned rows violate x2>x1 or y2>y1",
                "event_count": len(events),
                "logical_variant_count": len(LOGICAL_VARIANTS),
                "invalid_assigned_row_count": len(invalid_boxes),
                "unique_event_logical_frame_public_count": len(
                    {
                        (
                            item["event_id"],
                            item["sealed_runtime_variant"],
                            item["original_frame"],
                            item["solver_public_id"],
                        )
                        for item in invalid_boxes
                    }
                ),
                "invalid_rows": invalid_boxes,
                "allowed_repairs": [],
                "historical_runtime_artifacts_modified": False,
            },
        )
        raise WindowTrackEvalError(
            f"{len(invalid_boxes)} sealed exact-solver assigned rows have invalid boxes; "
            "post-hoc export cannot repair or omit them"
        )

    protocol = build_protocol(
        frozen["sources"], output_root=output_root, source_branch=source_branch,
        source_commit=source_commit, trackeval_root=trackeval_path,
        data_root=dataset_root, events=events, mode=mode,
    )
    try:
        protocol["trackeval_commit"] = subprocess.check_output(
            ["git", "-C", str(trackeval_path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        protocol["trackeval_commit"] = None
    write_json_atomic(output_root / "protocol.json", protocol)

    manifest_records: list[dict[str, Any]] = []
    window_records: list[dict[str, Any]] = []
    sequence_maps: dict[int, list[str]] = {horizon: [] for horizon in HORIZONS}
    for event in events:
        gt_source, seqinfo_source = _source_gt_and_seqinfo(dataset_root, event)
        for horizon in HORIZONS:
            pseudo = _pseudo_name(event["event_id"], horizon)
            gt_lines = _gt_window_lines(
                gt_source, event_frame=event["event_frame"], horizon=horizon, event_id=event["event_id"]
            )
            gt_dir = output_root / "pseudo_gt" / pseudo
            write_text_atomic(gt_dir / "gt" / "gt.txt", "\n".join(gt_lines) + "\n")
            write_text_atomic(gt_dir / "seqinfo.ini", _seqinfo_text(seqinfo_source, pseudo, horizon))
            sequence_maps[horizon].append(pseudo)
            window_records.append(
                {
                    "pseudo_sequence": pseudo,
                    "original_sequence": event["sequence"],
                    "event_id": event["event_id"],
                    "action_type": event["action_type"],
                    "event_frame": event["event_frame"],
                    "horizon": horizon,
                    "original_start_frame": event["event_frame"] + 1,
                    "original_end_frame": event["event_frame"] + horizon,
                    "gt_path": str(gt_dir / "gt" / "gt.txt"),
                    "seqinfo_path": str(gt_dir / "seqinfo.ini"),
                    "gt_sha256": sha256_file(gt_dir / "gt" / "gt.txt"),
                }
            )
            for logical in LOGICAL_VARIANTS:
                cache_key = (event["event_id"], logical)
                lines, score_counts = _prediction_lines(
                    runtime_cache[cache_key], event_frame=event["event_frame"], horizon=horizon,
                    event_id=event["event_id"], variant=event[logical]["sealed_runtime_variant"],
                )
                prediction_path = output_root / "trackers" / logical / "data" / f"{pseudo}.txt"
                write_text_atomic(prediction_path, "\n".join(lines) + ("\n" if lines else ""))
                manifest_records.append(
                    {
                        "logical_variant": logical,
                        "sealed_runtime_variant": event[logical]["sealed_runtime_variant"],
                        "runtime_frames_path": str(event[logical]["entry"]["frames"]),
                        "runtime_frames_sha256": event[logical]["frames_sha256"],
                        "pseudo_sequence": pseudo,
                        "original_sequence": event["sequence"],
                        "event_id": event["event_id"],
                        "action_type": event["action_type"],
                        "event_frame": event["event_frame"],
                        "horizon": horizon,
                        "prediction_path": str(prediction_path),
                        "prediction_sha256": sha256_file(prediction_path),
                        "prediction_score_policy": "runtime_confidence_else_constant_1",
                        **score_counts,
                    }
                )
    for horizon, names in sequence_maps.items():
        seqmap = output_root / "seqmaps" / f"seqmap_h{horizon:03d}.txt"
        write_text_atomic(seqmap, "name\n" + "\n".join(names) + "\n")

    expected_records = len(events) * len(HORIZONS) * len(LOGICAL_VARIANTS)
    keys = [
        (record["event_id"], record["logical_variant"], record["horizon"])
        for record in manifest_records
    ]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if len(manifest_records) != expected_records or duplicates:
        raise WindowTrackEvalError(
            f"export record completeness failed: expected {expected_records}, got {len(manifest_records)}, duplicates={duplicates}"
        )
    manifest = {
        "schema_version": "N72R11R5_WINDOW_EXPORT_MANIFEST_V1",
        "status": "PASS_EXPORT",
        "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
        "official_dancetrack_benchmark_score": False,
        "source_protocol_sha256": frozen["protocol_sha256"],
        "source_events": len(events),
        "independent_sequence_count": len({event["sequence"] for event in events}),
        "horizons": list(HORIZONS),
        "logical_variants": list(LOGICAL_VARIANTS),
        "pseudo_sequence_count": len(window_records),
        "record_count": len(manifest_records),
        "runtime_future_gt_used": False,
        "gt_used_only_for_posthoc_evaluation": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "prediction_score_policy": "runtime_confidence_else_constant_1",
        "event_windows": window_records,
        "records": manifest_records,
        "integrity": {
            "expected_records": expected_records,
            "actual_records": len(manifest_records),
            "duplicate_keys": duplicates,
            "missing_keys": [],
            "all_runtime_hashes_verified": True,
            "all_prediction_files_written_atomically": True,
        },
    }
    write_json_atomic(output_root / "export_manifest.json", manifest)
    return manifest
