#!/usr/bin/env python3
"""Run one N72R10 E0/E1/E2 replay event with sealed true future queries.

The audited N72R9 runtime engine is reused as an immutable evaluation
adapter.  Only its source adapter, model checkpoint, and re-query input are
swapped in this isolated child process:

E0 = BASELINE_B0
E1 = TEMPORAL_CURRENT_V2
E2 = TRUE_CLOSED_LOOP_REQUERY_V2

The runtime phase never opens dataset GT.  It seals all three runtime
variants first; the legacy posthoc scorer is called only after that seal.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import scripts.n72r9_temporal_replay as legacy  # noqa: E402
from sam3_intermot.reacquisition.models.n72r10_source_temporal_v2 import (  # noqa: E402
    N72R10SourceAwareTemporalIdentityModel,
)
from sam3_intermot.reacquisition.target_candidate_pool import (  # noqa: E402
    FUTURE_FRAME_REQUERY,
    build_candidate_pool_with_future_requery,
)


PROTOCOL_PATH = ROOT / "outputs/N72R9/protocol.json"
FUTURE_AUDIT_PATH = ROOT / "outputs/N72R10/stage_03_true_future_requery/batch_integrity_audit.json"
CHECKPOINT = ROOT / "outputs/N72R10/training/N72R10SourceAwareTemporalIdentityModel_v2.pt"
HORIZON = 100
OLD_VARIANTS = ("BASELINE_B0", "TEMPORAL_CURRENT", "TEMPORAL_REQUERY")
VARIANT_ALIASES = {
    "BASELINE_B0": "E0_B0",
    "TEMPORAL_CURRENT": "E1_TEMPORAL_CURRENT_V2",
    "TEMPORAL_REQUERY": "E2_TRUE_CLOSED_LOOP_REQUERY_V2",
}
SOURCE_NAMES = (
    "MAIN_B0_CANDIDATE",
    "TARGET_SESSION_CURRENT_RAW",
    "STATIC_EVENT_REQUERY",
    "FUTURE_FRAME_REQUERY",
    "UNKNOWN",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
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


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write(path, json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def resolved_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_future_rows(event_id: str, event_frame: int) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    audit = read_json(FUTURE_AUDIT_PATH)
    if audit.get("status") != "PASS_N72R10_TRUE_FUTURE_REQUERY_BATCH_AUDIT":
        raise RuntimeError("N72R10 true-future batch audit is not PASS")
    matches = [item for item in audit.get("event_rows", []) if str(item.get("event_id")) == str(event_id)]
    if len(matches) != 1:
        raise RuntimeError(f"future audit must contain exactly one event row: {event_id}")
    audit_row = dict(matches[0])
    artifact_dir = resolved_path(str(audit_row["artifact_dir"]))
    candidates_path = artifact_dir / "candidates.json"
    if not candidates_path.is_file() or str(audit_row.get("candidates_sha256")) != sha256_file(candidates_path):
        raise RuntimeError(f"future candidate artifact hash mismatch: {event_id}")
    payload = read_json(candidates_path)
    if payload.get("runtime_future_gt_used") is not False or payload.get("posthoc_gt_used") is not False:
        raise RuntimeError(f"future candidate artifact GT boundary violation: {event_id}")
    by_frame: dict[int, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for candidate in payload.get("future_candidates", []):
        if str(candidate.get("candidate_source")) != FUTURE_FRAME_REQUERY:
            raise RuntimeError(f"future source taxonomy violation: {event_id}")
        if str(candidate.get("candidate_kind")) != "FUTURE_FRAME_REQUERY_CANDIDATE":
            raise RuntimeError(f"future candidate kind violation: {event_id}")
        if candidate.get("public_id") is not None or candidate.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"future candidate authority/GT violation: {event_id}")
        frame = int(candidate.get("frame", -1))
        if frame < event_frame + 1 or frame > event_frame + HORIZON:
            raise RuntimeError(f"future candidate outside sealed window: {event_id}:{frame}")
        uid = str(candidate.get("candidate_uid"))
        if uid in seen:
            raise RuntimeError(f"future candidate UID duplicate: {event_id}:{uid}")
        seen.add(uid)
        by_frame.setdefault(frame, []).append(dict(candidate))
    rows = {
        frame: {
            "frame": frame,
            "candidate_rows": list(by_frame.get(frame, [])),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "public_id_inference": False,
        }
        for frame in range(event_frame + 1, event_frame + HORIZON + 1)
    }
    declared_count = int(audit_row.get("future_candidate_count", len(seen)))
    if declared_count != len(seen):
        raise RuntimeError(f"future candidate count mismatch: {event_id}: {len(seen)} != {declared_count}")
    return rows, {
        "artifact_dir": str(artifact_dir),
        "candidates": str(candidates_path),
        "candidates_sha256": sha256_file(candidates_path),
        "future_candidate_count": len(seen),
        "future_nonempty_frame_count": sum(bool(row["candidate_rows"]) for row in rows.values()),
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
    }


def load_inputs(event: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    # This verifies all frozen N72R9 source hashes and the causal authority
    # axis before the N72R10 future source is attached.
    inputs = legacy._load_rows(event)
    future_rows, future_provenance = load_future_rows(str(event["event_id"]), int(event["event_frame"]))
    inputs["rows"]["requery_source"] = future_rows
    inputs["source_paths"]["requery_source"] = future_provenance["candidates"]
    inputs["source_hashes"]["requery_source"] = future_provenance["candidates_sha256"]
    inputs["future_provenance"] = future_provenance
    return inputs, future_provenance


def load_model(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    config = dict(payload.get("model_config", {}))
    required = {
        "candidate_feature_dim", "source_feature_dim", "temporal_feature_dim", "trusted_slots",
        "distractor_slots", "hidden_dim", "layers", "heads", "dropout",
    }
    if not required.issubset(config) or int(config["source_feature_dim"]) != len(SOURCE_NAMES):
        raise RuntimeError("N72R10 checkpoint configuration is incomplete or has wrong source width")
    model = N72R10SourceAwareTemporalIdentityModel(**{key: config[key] for key in required})
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_schema_version": payload.get("schema_version"),
        "model_config": config,
    }


def run_event(
    event_id: str,
    output_root: Path,
    device: torch.device,
    *,
    started_at_utc: str,
) -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    events = {str(item["event_id"]): item for item in protocol.get("source_event_selection", {}).get("events", [])}
    if event_id not in events:
        raise RuntimeError(f"event is not in frozen N72R9 protocol: {event_id}")
    inputs, future_provenance = load_inputs(events[event_id])
    model, model_provenance = load_model(device)
    runtime_rows: dict[str, list[dict[str, Any]]] = {}
    runtime_manifests: dict[str, dict[str, Any]] = {}
    stats: dict[str, Any] = {}
    event_dir = output_root / event_id
    for variant in OLD_VARIANTS:
        rows, variant_stats = legacy._run_variant(
            inputs, variant, None if variant == "BASELINE_B0" else model, device
        )
        frames_path = event_dir / variant / "runtime_frames.jsonl"
        atomic_jsonl(frames_path, rows)
        manifest = {
            "schema_version": "N72R10_TEMPORAL_RUNTIME_MANIFEST_V1",
            "status": "PASS_N72R10_RUNTIME_ARTIFACT_SEALED",
            "event_id": event_id,
            "sequence": inputs["sequence"],
            "variant": variant,
            "variant_alias": VARIANT_ALIASES[variant],
            "event_frame": inputs["event_frame"],
            "target_public_id": inputs["target_public_id"],
            "frame_count": len(rows),
            "frames": str(frames_path),
            "frames_sha256": sha256_file(frames_path),
            "input_source_hashes": inputs["source_hashes"],
            "future_requery_provenance": future_provenance,
            "model_provenance": model_provenance,
            "stats": variant_stats,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
        }
        atomic_json(event_dir / variant / "runtime_manifest.json", manifest)
        runtime_rows[variant] = rows
        runtime_manifests[variant] = manifest
        stats[variant] = variant_stats
    # This is the causal seal point.  No GT is loaded before this file exists.
    sealed = {
        "schema_version": "N72R10_TEMPORAL_RUNTIME_EVENT_SEALED_V1",
        "status": "PASS_N72R10_ALL_VARIANT_RUNTIME_SEALED",
        "event_id": event_id,
        "variants": list(OLD_VARIANTS),
        "variant_aliases": VARIANT_ALIASES,
        "runtime_manifests": runtime_manifests,
        "future_requery_provenance": future_provenance,
        "model_provenance": model_provenance,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "gt_loaded": False,
        "created_at_utc": now_utc(),
    }
    atomic_json(event_dir / "runtime_event_sealed.json", sealed)
    posthoc = legacy._posthoc_score(inputs, runtime_rows)
    posthoc.update({
        "schema_version": "N72R10_TEMPORAL_POSTHOC_EVENT_V1",
        "variant_aliases": VARIANT_ALIASES,
        "future_requery_provenance": future_provenance,
        "model_provenance": model_provenance,
    })
    atomic_json(event_dir / "posthoc.json", posthoc)
    done = {
        "schema_version": "N72R10_TEMPORAL_EVENT_DONE_V1",
        "status": "PASS_N72R10_RUNTIME_AND_POSTHOC_EVENT",
        "event_id": event_id,
        "runtime_event_sealed": str(event_dir / "runtime_event_sealed.json"),
        "runtime_event_sealed_sha256": sha256_file(event_dir / "runtime_event_sealed.json"),
        "posthoc": str(event_dir / "posthoc.json"),
        "posthoc_sha256": sha256_file(event_dir / "posthoc.json"),
        "stats": stats,
        "future_requery_provenance": future_provenance,
        "model_provenance": model_provenance,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "production_authorized": False,
        "started_at_utc": started_at_utc,
        "finished_at_utc": now_utc(),
    }
    atomic_json(event_dir / "done.json", done)
    return {"status": done["status"], "event_id": event_id, "output": str(event_dir), "stats": stats}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    started = now_utc()
    output_root = resolved_path(args.output_root)
    event_id = str(args.event_id)
    try:
        device = torch.device(str(args.device))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device is unavailable")
        result = run_event(event_id, output_root, device, started_at_utc=started)
        result["started_at_utc"] = started
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        failure = output_root / "attempts" / f"{event_id}.failure.json"
        payload = {
            "schema_version": "N72R10_TEMPORAL_REPLAY_FAILURE_V1",
            "status": "FAIL_N72R10_TEMPORAL_REPLAY_EVENT",
            "event_id": event_id,
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
        }
        atomic_json(failure, payload)
        print(json.dumps({"status": payload["status"], "event_id": event_id, "failure_artifact": str(failure), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    # The legacy engine uses these module globals for source normalization and
    # the re-query builder.  The mutation is process-local and does not touch
    # the frozen N72R9 source or production code.
    legacy.SOURCE_NAMES = SOURCE_NAMES
    legacy.build_candidate_pool_with_requery = build_candidate_pool_with_future_requery
    raise SystemExit(main())
