#!/usr/bin/env python3
"""Run one N72R10 true future-frame re-query event in an isolated process.

The process consumes only a frozen N72R9 runtime row at ``event_frame + 1``
for its causal predicted state.  It then creates a fresh SAM3 session at that
frame, probes the frozen four-query family, chooses by a causal confidence
tie-break, and propagates the selected raw target source through H100.  No GT
or posthoc metric is opened by this worker; scoring is a separate CPU stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.backend.sam3_backend import Sam3Backend  # noqa: E402
from sam3_intermot.reacquisition.future_requery_session import (  # noqa: E402
    FUTURE_FRAME_REQUERY,
    FutureFrameRequerySession,
    QUERY_SPECS,
)
from scripts.run_n35_export_tape import image_files  # noqa: E402
from scripts.n72r5_stage07_official_full_loop import (  # noqa: E402
    CHECKPOINT,
    DATA_ROOT,
    FrozenMachineOSNetN72R5,
    MACHINE_CHECKPOINT,
)


PROTOCOL_PATH = ROOT / "outputs/N72R9/protocol.json"
RUNTIME_ROOT = ROOT / "outputs/N72R9/replay/full"
HORIZON = 100
QUERY_SELECTOR = "CAUSAL_OFFICIAL_CONFIDENCE_THEN_QUERY_ORDER"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
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
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _runtime_flags(value: Mapping[str, Any], label: str) -> None:
    for key in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"):
        if value.get(key) is not False:
            raise RuntimeError(f"{label} has invalid {key}={value.get(key)!r}")


def _target_assignment_uid(row: Mapping[str, Any], target_public_id: int) -> str | None:
    assignment = row.get("assignment")
    if not isinstance(assignment, Mapping):
        return None
    direct = assignment.get("target_assigned_candidate_uid")
    if direct not in (None, ""):
        return str(direct)
    solver = assignment.get("solver")
    if isinstance(solver, Mapping):
        for item in solver.get("public_assignments", []):
            if isinstance(item, Mapping) and int(item.get("public_id", -1)) == int(target_public_id):
                return str(item["candidate_uid"])
    return None


def _finite_box(value: Any, label: str) -> list[float]:
    box = np.asarray(value, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
        raise RuntimeError(f"{label} is not a finite positive XYXY box")
    return [float(item) for item in box]


def _causal_target_from_variant(
    row: Mapping[str, Any], *, target_public_id: int, variant: str
) -> tuple[list[float], int, str, str]:
    pool = row.get("candidate_pool")
    if not isinstance(pool, Mapping):
        raise RuntimeError(f"{variant} causal row has no candidate_pool")
    candidates = pool.get("candidate_rows")
    if not isinstance(candidates, list):
        raise RuntimeError(f"{variant} causal row has no candidate_rows")
    uid = _target_assignment_uid(row, target_public_id)
    if uid is None:
        raise RuntimeError(f"{variant} causal row has no target public assignment")
    matches = [item for item in candidates if isinstance(item, Mapping) and str(item.get("candidate_uid")) == uid]
    if len(matches) != 1:
        raise RuntimeError(f"{variant} causal target UID is not unique: {uid}")
    target = matches[0]
    raw = target.get("official_raw_sam_id", target.get("adapter_external_id"))
    if raw is None:
        raise RuntimeError(f"{variant} causal target has no raw/native ID")
    return _finite_box(target.get("box_xyxy"), f"{variant} causal target"), int(raw), uid, str(target.get("candidate_source"))


def load_causal_state(event_id: str, event_frame: int) -> dict[str, Any]:
    """Read a single frozen runtime row, with E1 fallback for baseline ADD."""

    event_root = RUNTIME_ROOT / event_id
    trigger = int(event_frame) + 1
    variants = ("BASELINE_B0", "TEMPORAL_CURRENT")
    rows_by_variant: dict[str, dict[str, Any]] = {}
    for variant in variants:
        path = event_root / variant / "runtime_frames.jsonl"
        rows = read_jsonl(path)
        expected = list(range(int(event_frame), int(event_frame) + HORIZON + 1))
        if [int(row.get("frame", -1)) for row in rows] != expected:
            raise RuntimeError(f"{event_id}:{variant} frozen frame axis is incomplete")
        for row in rows:
            _runtime_flags(row, f"{event_id}:{variant}:{row.get('frame')}")
        selected = [row for row in rows if int(row["frame"]) == trigger]
        if len(selected) != 1:
            raise RuntimeError(f"{event_id}:{variant}:{trigger} expected one runtime row")
        rows_by_variant[variant] = selected[0]
    b0, current = rows_by_variant["BASELINE_B0"], rows_by_variant["TEMPORAL_CURRENT"]
    _runtime_flags(b0, f"{event_id}:B0:{trigger}")
    _runtime_flags(current, f"{event_id}:CURRENT:{trigger}")
    target_public_id = int(b0.get("target_public_id", -1))
    if target_public_id <= 0 or int(current.get("target_public_id", -1)) != target_public_id:
        raise RuntimeError(f"{event_id}:{trigger} target public authority differs between frozen variants")
    source_variant = "BASELINE_B0"
    try:
        predicted_box, previous_raw, source_uid, source_kind = _causal_target_from_variant(
            b0, target_public_id=target_public_id, variant="BASELINE_B0"
        )
    except RuntimeError as baseline_error:
        predicted_box, previous_raw, source_uid, source_kind = _causal_target_from_variant(
            current, target_public_id=target_public_id, variant="TEMPORAL_CURRENT"
        )
        source_variant = "TEMPORAL_CURRENT"
        baseline_error_text = str(baseline_error)
    else:
        baseline_error_text = None
    causal_state = {
        "previous_raw_sam_id": int(previous_raw),
        "current_raw_sam_id": int(previous_raw),
        "source_runtime_variant": source_variant,
        "source_runtime_record_kind": str(rows_by_variant[source_variant].get("record_kind")),
        "source_candidate_uid": source_uid,
        "source_candidate_kind": source_kind,
        "target_public_id_authority": "frozen_runtime_assignment_input_only",
        "baseline_fallback_reason": baseline_error_text,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }
    return {
        "target_public_id": target_public_id,
        "predicted_box": predicted_box,
        "causal_state": causal_state,
        "trigger_frame": trigger,
        "runtime_paths": {
            variant: str(event_root / variant / "runtime_frames.jsonl") for variant in variants
        },
        "runtime_hashes": {
            variant: sha256_file(event_root / variant / "runtime_frames.jsonl") for variant in variants
        },
    }


def make_backend(device: str) -> Sam3Backend:
    return Sam3Backend(
        checkpoint_path=str(CHECKPOINT),
        max_num_objects=16,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
        session_expiration_sec=1200,
        output_prob_thresh=0.30,
        async_loading_frames=False,
        device=device,
        official_batched_grounding_batch_size=1,
    )


def _select_probe(probe_candidates: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(probe_candidates) != len(QUERY_SPECS):
        raise RuntimeError(f"expected {len(QUERY_SPECS)} successful probe candidates, got {len(probe_candidates)}")
    ranked = sorted(
        (dict(item) for item in probe_candidates),
        key=lambda item: (-float(item.get("confidence", 0.0)), int(item.get("requery_index", 0))),
    )
    if any(item.get("candidate_source") != FUTURE_FRAME_REQUERY for item in ranked):
        raise RuntimeError("probe candidate source is not FUTURE_FRAME_REQUERY")
    if any(item.get("public_id") is not None for item in ranked):
        raise RuntimeError("probe candidate exposed public-ID authority")
    selected = ranked[0]
    return selected, {
        "selector": QUERY_SELECTOR,
        "selection_is_scientific": False,
        "selected_query_name": str(selected["requery_name"]),
        "selected_candidate_uid": str(selected["candidate_uid"]),
        "ranked_candidates": [
            {
                "candidate_uid": str(item["candidate_uid"]),
                "query_name": str(item["requery_name"]),
                "confidence": float(item.get("confidence", 0.0)),
                "query_index": int(item.get("requery_index", 0)),
                "runtime_future_gt_used": False,
            }
            for item in ranked
        ],
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "public_id_inference": False,
    }


def run_event(event_id: str, *, device: str, output_root: Path) -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    event_matches = [
        dict(item)
        for item in protocol.get("source_event_selection", {}).get("events", [])
        if str(item.get("event_id")) == str(event_id)
    ]
    if len(event_matches) != 1:
        raise RuntimeError(f"event is not in the frozen N72R9 protocol: {event_id}")
    event = event_matches[0]
    sequence = str(event["sequence"])
    event_frame = int(event["event_frame"])
    trigger = event_frame + 1
    end_frame = event_frame + HORIZON
    causal = load_causal_state(event_id, event_frame)
    if int(causal["trigger_frame"]) != trigger:
        raise RuntimeError("causal trigger mismatch")
    paths = image_files(DATA_ROOT / "train" / sequence)
    if end_frame >= len(paths):
        raise RuntimeError(f"{sequence} image coverage is incomplete for {event_frame}:{end_frame}")
    if not CHECKPOINT.is_file() or not MACHINE_CHECKPOINT.is_file():
        raise FileNotFoundError("frozen SAM3 or machine feature checkpoint is missing")
    encoder = FrozenMachineOSNetN72R5(device)

    def feature_fn(frame: int, box: Sequence[float]) -> np.ndarray:
        values = encoder.encode(paths[int(frame)], [box])
        return np.asarray(values[0], dtype=np.float32)

    backends: list[Sam3Backend] = []

    def backend_factory() -> Sam3Backend:
        backend = make_backend(device)
        backends.append(backend)
        return backend

    session = FutureFrameRequerySession(
        backend_factory=backend_factory,
        sequence=sequence,
        event_id=event_id,
        event_frame=event_frame,
        target_public_id=int(causal["target_public_id"]),
        frame_paths=paths,
        feature_fn=feature_fn,
    )
    started = time.time()
    try:
        start_info = session.start_from_frame(
            trigger,
            causal["predicted_box"],
            causal["causal_state"],
            end_frame=end_frame,
            main_y_pre_frozen=True,
        )
        probes = session.query_current_frame()
        query_audits = session.audit().get("query_audits", [])
        failures = [item for item in query_audits if str(item.get("status", "")).startswith("FAIL")]
        if failures:
            raise RuntimeError(f"fresh current-frame query failed for {len(failures)} frozen queries")
        selected, selection_audit = _select_probe(probes)
        future = session.propagate_if_selected(
            selected_query_name=str(selected["requery_name"]),
            selected_candidate_uid=str(selected["candidate_uid"]),
            selection_audit=selection_audit,
        )
        audit_before_close = session.audit()
        session.close()
        audit_after_close = session.audit()
        coverage = list(audit_before_close.get("future_frame_coverage", []))
        expected_frames = list(range(trigger, end_frame + 1))
        observed_frames = [int(item.get("global_frame", -1)) for item in coverage]
        if observed_frames != expected_frames:
            raise RuntimeError("future frame coverage is not exactly trigger..event+100")
        if audit_before_close.get("event_frame_memory_read") is not False:
            raise RuntimeError("event-frame memory read boundary failed")
        if int(audit_before_close.get("first_memory_visible_frame", -1)) != trigger + 1:
            raise RuntimeError("future memory first-visible frame is not trigger+1")
        if audit_before_close.get("raw_rebinding", {}).get("public_id_changed") is not False:
            raise RuntimeError("future re-query changed public identity authority")
        if any(row.get("candidate_source") != FUTURE_FRAME_REQUERY for row in future):
            raise RuntimeError("active future stream contains a non-live re-query source")
        if any(row.get("runtime_future_gt_used") is not False for row in future):
            raise RuntimeError("active future stream contains runtime future-GT use")
        trigger_rows = [row for row in future if int(row.get("frame", -1)) == trigger]
        if len(trigger_rows) != 1:
            raise RuntimeError("selected fresh session did not expose exactly one trigger candidate")
        candidate_uids = [str(row["candidate_uid"]) for row in future]
        if len(candidate_uids) != len(set(candidate_uids)):
            raise RuntimeError("active future candidate UID collision")
        if not bool(audit_after_close.get("closed")):
            raise RuntimeError("session did not report closed after explicit close")
        lifecycle = {
            "backend_count": len(backends),
            "probe_backend_count": max(0, len(backends) - 1),
            "active_backend_count": 1,
            "all_backend_sessions_closed": all(getattr(item, "_session_id", None) is None for item in backends),
            "fresh_backend_per_probe_and_active_session": True,
        }
        if not lifecycle["all_backend_sessions_closed"]:
            raise RuntimeError("one or more fresh backend sessions remained open")
        result = {
            "schema_version": "N72R10_TRUE_FUTURE_REQUERY_EVENT_RESULT_V1",
            "status": "PASS_N72R10_TRUE_FUTURE_REQUERY_EVENT",
            "event_id": event_id,
            "sequence": sequence,
            "action_type": str(event.get("action_type", "UNKNOWN")),
            "event_frame": event_frame,
            "trigger_frame": trigger,
            "end_frame": end_frame,
            "target_public_id": int(causal["target_public_id"]),
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "causal_input": {
                "runtime_paths": causal["runtime_paths"],
                "runtime_hashes": causal["runtime_hashes"],
                "source_runtime_variant": causal["causal_state"]["source_runtime_variant"],
                "source_candidate_uid": causal["causal_state"]["source_candidate_uid"],
                "predicted_box_xyxy": causal["predicted_box"],
                "previous_raw_sam_id": causal["causal_state"]["previous_raw_sam_id"],
                "baseline_fallback_reason": causal["causal_state"].get("baseline_fallback_reason"),
                "runtime_future_gt_used": False,
            },
            "frozen_protocol_sha256": sha256_file(PROTOCOL_PATH),
            "sam3_checkpoint": str(CHECKPOINT),
            "sam3_checkpoint_sha256": sha256_file(CHECKPOINT),
            "machine_checkpoint": str(MACHINE_CHECKPOINT),
            "machine_checkpoint_sha256": sha256_file(MACHINE_CHECKPOINT),
            "start": start_info,
            "probe_candidate_count": len(probes),
            "selected_query_name": str(selected["requery_name"]),
            "selected_probe_candidate_uid": str(selected["candidate_uid"]),
            "future_candidate_count": len(future),
            "future_nonempty_frame_count": sum(int(item.get("candidate_count", 0)) > 0 for item in coverage),
            "future_frame_count": len(coverage),
            "future_trigger_candidate_count": len(trigger_rows),
            "probe_candidates_path": "candidates.json#probe_candidates",
            "audit_path": "audit.json",
            "audit_sha256": None,
            "lifecycle": lifecycle,
            "backend_runtime_memory_policies": [backend.runtime_memory_policy() for backend in backends],
            "audit_before_close": audit_before_close,
            "audit_after_close": audit_after_close,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "scientific_effect_result": None,
            "elapsed_sec": time.time() - started,
            "device": device,
        }
        return {
            "result": result,
            "probes": probes,
            "future": future,
            "audit": audit_after_close,
        }
    finally:
        try:
            session.close()
        finally:
            del session
            del encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    event_id = str(args.event_id)
    event_dir = args.output_root / "events" / event_id
    done_path = event_dir / "done.json"
    if done_path.exists() or (event_dir / "result.json").exists():
        raise RuntimeError(f"refusing to overwrite existing event artifact: {event_dir}")
    started = now_utc()
    try:
        payload = run_event(event_id, device=str(args.device), output_root=args.output_root)
        result = dict(payload["result"])
        atomic_json(event_dir / "candidates.json", {
            "schema_version": "N72R10_TRUE_FUTURE_REQUERY_CANDIDATES_V1",
            "event_id": event_id,
            "probe_candidates": payload["probes"],
            "future_candidates": payload["future"],
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
        })
        atomic_json(event_dir / "audit.json", payload["audit"])
        result["audit_sha256"] = sha256_file(event_dir / "audit.json")
        result["candidates_sha256"] = sha256_file(event_dir / "candidates.json")
        result["created_at_utc"] = now_utc()
        atomic_json(event_dir / "result.json", result)
        atomic_json(event_dir / "done.json", {
            "schema_version": "N72R10_TRUE_FUTURE_REQUERY_EVENT_DONE_V1",
            "status": result["status"],
            "event_id": event_id,
            "result": str(event_dir / "result.json"),
            "result_sha256": sha256_file(event_dir / "result.json"),
            "candidates": str(event_dir / "candidates.json"),
            "candidates_sha256": sha256_file(event_dir / "candidates.json"),
            "audit": str(event_dir / "audit.json"),
            "audit_sha256": sha256_file(event_dir / "audit.json"),
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
        })
        print(json.dumps({"status": result["status"], "event_id": event_id, "output": str(event_dir)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "N72R10_TRUE_FUTURE_REQUERY_EVENT_FAILURE_V1",
            "status": "FAIL_N72R10_TRUE_FUTURE_REQUERY_EVENT",
            "event_id": event_id,
            "device": str(args.device),
            "command": [str(item) for item in sys.argv],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
        }
        atomic_json(args.output_root / "attempts" / f"{event_id}.failure.json", failure)
        print(json.dumps({"status": failure["status"], "event_id": event_id, "failure": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
