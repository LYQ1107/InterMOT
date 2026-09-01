#!/usr/bin/env python3
"""Run one isolated N41-02 source/config paired replay worker.

The worker receives one frozen N37 event, one source feature from the N41
source sidecar, and one preregistered weight configuration.  It never imports
DanceTrack GT and never sees future labels.  The output contains the event
frame plus every future frame for all M0--M4 branches, including complete
score matrices, assignments, candidate order and public/native mapping.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.ccam_replay import paired_replay
from sam3_intermot.association.state_manager import StateManager
from scripts.n36_real_eval_common import FEATURE_DIM, atomic_json, variant_config
from scripts.n38r1_sidecar_common import build_event_frame_audit, read_source_rows
from scripts.n39_weight_worker import (
    compact_audit,
    compact_trace,
    memory_summary,
    validate_audit,
)
from scripts.n38r1_sidecar_common import protocol_hash
from scripts.run_n37_replay import build_runtime_tape


N37_MANIFEST = ROOT / "outputs" / "n37" / "real_event_manifest.json"
SOURCE_MANIFEST = ROOT / "outputs" / "n41" / "source_replay" / "source_embedding_manifest.json"
PROTOCOL_PATH = ROOT / "outputs" / "n41" / "source_replay" / "source_protocol.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
PROTOCOL = "N41_GT_CONTROLLED_APPEARANCE_SOURCE_ABLATION_WORKER_V1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_event(event_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_json(N37_MANIFEST)
    if manifest.get("status") != "PASS" or manifest.get("event_count") != 24:
        raise RuntimeError("frozen N37 manifest is not PASS/24")
    for item in manifest.get("events", []):
        if str(item.get("event", {}).get("event_id")) == str(event_id):
            return manifest, item
    raise KeyError(f"N37 event not found: {event_id}")


def load_source(event_id: str, source_id: str) -> dict[str, Any]:
    payload = load_json(SOURCE_MANIFEST)
    if payload.get("status") != "PASS" or payload.get("event_count") != 24:
        raise RuntimeError("N41 source sidecar is not PASS/24")
    for entry in payload.get("events", []):
        if str(entry.get("event_id")) != str(event_id):
            continue
        source = entry.get("sources", {}).get(source_id)
        if not isinstance(source, dict):
            raise KeyError(f"source not found: {event_id}/{source_id}")
        feature = np.asarray(source.get("feature"), dtype=np.float32).reshape(-1)
        if feature.size != FEATURE_DIM or not np.all(np.isfinite(feature)):
            raise ValueError(f"invalid source feature: {event_id}/{source_id}")
        norm = float(np.linalg.norm(feature))
        if norm <= 1e-6:
            raise ValueError(f"zero source feature: {event_id}/{source_id}")
        if source.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"source sidecar runtime future GT flag is not false: {event_id}/{source_id}")
        return source
    raise KeyError(f"event not found in source sidecar: {event_id}")


def install_human_weight(weight: float):
    original_init = StateManager.__init__

    def patched_init(self, config):
        original_init(self, config)
        self.appearance_memory.human_weight = float(weight)

    StateManager.__init__ = patched_init  # type: ignore[method-assign]
    return original_init


def restore_human_weight(original_init: Any) -> None:
    StateManager.__init__ = original_init  # type: ignore[method-assign]


def expected_future(item: dict[str, Any]) -> list[int]:
    event_frame = int(item["event"]["frame"])
    end = min(int(item["sequence_frame_count"]) - 1, event_frame + 100)
    return list(range(event_frame + 1, end + 1))


def replay_signature(replay: dict[str, Any]) -> str:
    """Hash deterministic axes/scores/assignments without serializing state."""
    parts: list[Any] = []
    for branch_name in ("memory_write=False", "memory_write=True"):
        branch = replay["branches"][branch_name]
        for entry in branch.get("future_trace", []):
            audit = entry.get("candidate_audit", {})
            parts.append(
                {
                    "frame": int(entry["frame"]),
                    "candidate_order": audit.get("candidate_order", []),
                    "candidate_native_ids": audit.get("candidate_native_ids", []),
                    "public_id_order": audit.get("public_id_order", []),
                    "assignment": audit.get("assignment_after_scope", audit.get("assignment", [])),
                    "fused_scores": np.asarray(audit.get("fused_scores", []), dtype=np.float64).tolist(),
                }
            )
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_axes_and_boundary(
    artifact: dict[str, Any],
    variant_payload: dict[str, Any],
    event_frame: int,
    future_frames: list[int],
) -> None:
    event_audit = variant_payload["event_frame_audit"]
    event_candidate_audit = event_audit["candidate_audit"]
    validate_audit(event_candidate_audit, event_frame, event_frame)
    if event_audit.get("is_event_frame") is not True or event_audit.get("is_future_frame") is not False:
        raise RuntimeError("event frame flags invalid")
    if event_audit.get("memory_read") is not False or event_audit.get("memory_write") is not False:
        raise RuntimeError("event frame memory was read or written")
    if event_audit.get("current_frame_write_hidden") is not True:
        raise RuntimeError("event frame write visibility flag invalid")
    if event_audit.get("runtime_future_gt_used") is not False:
        raise RuntimeError("event frame runtime future GT flag invalid")
    for branch_name in ("memory_write=False", "memory_write=True"):
        branch = variant_payload["branches"].get(branch_name)
        if not isinstance(branch, dict):
            raise RuntimeError(f"missing branch {branch_name}")
        trace = branch.get("future_trace")
        if not isinstance(trace, list) or [int(row["frame"]) for row in trace] != future_frames:
            raise RuntimeError(f"future frame range invalid for {branch_name}")
        for entry in trace:
            audit = entry.get("candidate_audit", {})
            validate_audit(audit, int(entry["frame"]), event_frame)
            if audit.get("is_event_frame") is not False or audit.get("is_future_frame") is not True:
                raise RuntimeError("future frame flags invalid")
            if audit.get("runtime_future_gt_used") is not False:
                raise RuntimeError("future runtime future GT flag invalid")
    if future_frames:
        first_audit = variant_payload["branches"]["memory_write=True"]["future_trace"][0]["candidate_audit"]
        if int(first_audit.get("frame", -1)) != event_frame + 1:
            raise RuntimeError("first memory-visible frame is not event+1")


def run(event_id: str, source_id: str, config_id: str, output: Path, check_determinism: bool = False) -> dict[str, Any]:
    started = now()
    _n37_manifest, original_item = load_event(event_id)
    source = load_source(event_id, source_id)
    protocol = load_json(PROTOCOL_PATH)
    configs = {str(item["config_id"]): item for item in protocol.get("weight_grid", [])}
    config_entry = configs.get(str(config_id))
    if config_entry is None:
        raise KeyError(f"config not in frozen N41-02 protocol: {config_id}")
    event = original_item["event"]
    if str(event["event_id"]) != str(event_id):
        raise RuntimeError("event ID mismatch")
    feature = np.asarray(source["feature"], dtype=np.float32).reshape(-1)
    item = copy.deepcopy(original_item)
    # Only the event-time appearance evidence is replaced.  The frozen
    # spatial correction, prefix, future tape and candidate stream remain
    # byte-for-byte the N37 input.
    item["event"]["human_embedding"] = feature.tolist()
    item["event"]["human_feature_digest"] = str(source["feature_sha256"])
    event_frame = int(event["frame"])
    future = expected_future(item)
    tape = build_runtime_tape(item, horizon=100)
    source_rows = read_source_rows(item, event_frame, future[-1] if future else event_frame)
    configs_payload: dict[str, Any] = {}
    original_init = install_human_weight(float(config_entry["human_weight"]))
    try:
        for variant in VARIANTS:
            config, description = variant_config(variant)
            config.appearance_score_weight = float(config_entry["lambda_assoc"])
            event_audit = build_event_frame_audit(tape, source_rows[event_frame], item, config)
            validate_audit(event_audit, event_frame, event_frame)
            replay = paired_replay(
                tape,
                config=config,
                feat_dim=FEATURE_DIM,
                write_branch_uses_appearance_memory=(variant != "M0"),
            )
            if replay.get("status") != "PASS":
                raise RuntimeError(f"paired_replay_failed:{variant}:{replay.get('status')}:{replay.get('validation')}")
            first_signature = replay_signature(replay)
            reproducibility = {"checked": False, "status": "NOT_REQUESTED"}
            if check_determinism:
                replay_again = paired_replay(
                    tape,
                    config=config,
                    feat_dim=FEATURE_DIM,
                    write_branch_uses_appearance_memory=(variant != "M0"),
                )
                second_signature = replay_signature(replay_again)
                reproducibility = {
                    "checked": True,
                    "status": "PASS" if first_signature == second_signature else "FAIL",
                    "first_signature": first_signature,
                    "second_signature": second_signature,
                }
                if reproducibility["status"] != "PASS":
                    raise RuntimeError(f"assignment_reproducibility_failed:{variant}")
                del replay_again
            compact_branches: dict[str, Any] = {}
            for branch_name in ("memory_write=False", "memory_write=True"):
                branch = replay["branches"][branch_name]
                actual_write = bool(branch.get("memory_write", False))
                trace = branch.get("future_trace")
                if not isinstance(trace, list) or [int(entry["frame"]) for entry in trace] != future:
                    raise RuntimeError(f"future_trace_incomplete:{variant}:{branch_name}")
                compacted = compact_trace(trace, memory_write=actual_write, memory_read=actual_write)
                for entry in compacted:
                    audit = entry.get("candidate_audit", {})
                    audit["is_event_frame"] = False
                    audit["is_future_frame"] = True
                    audit["runtime_future_gt_used"] = False
                compact_branches[branch_name] = {
                    "memory_write": actual_write,
                    "memory_read": actual_write,
                    "future_trace": compacted,
                    "state_summary": copy.deepcopy(branch.get("state_summary", {})),
                    "appearance_memory": memory_summary(branch.get("appearance_memory", {})),
                }
            configs_payload[variant] = {
                "description": description,
                "status": "PASS",
                "event_frame_audit": {
                    "frame": event_frame,
                    "is_event_frame": True,
                    "is_future_frame": False,
                    "candidate_audit": compact_audit(event_audit),
                    "memory_read": False,
                    "memory_write": False,
                    "current_frame_write_hidden": True,
                    "runtime_future_gt_used": False,
                    "gt_loaded_posthoc": False,
                },
                "branches": compact_branches,
                "reproducibility": reproducibility,
            }
            validate_axes_and_boundary(
                {"event_id": event_id}, configs_payload[variant], event_frame, future
            )
            del replay, event_audit, config
    finally:
        restore_human_weight(original_init)

    payload = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "event_id": str(event_id),
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": event_frame,
        "future_frame_start": event_frame + 1,
        "future_frame_end": future[-1] if future else event_frame,
        "future_frame_count": len(future),
        "source_id": str(source_id),
        "source_role": source.get("role"),
        "source_feature_sha256": str(source["feature_sha256"]),
        "source_feature_origin": source.get("feature_origin"),
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "input": {
            "n37_manifest": str(N37_MANIFEST.relative_to(ROOT)),
            "n37_manifest_sha256": sha256(N37_MANIFEST),
            "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
            "source_manifest_sha256": sha256(SOURCE_MANIFEST),
            "source_protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
            "source_protocol_sha256": sha256(PROTOCOL_PATH),
            "frozen_n38_protocol_hash": protocol_hash(),
            "source_tape": str(original_item["source_tape"]),
            "source_tape_sha256": original_item.get("source_tape_sha256"),
        },
        "weight_configuration": {
            "config_id": str(config_id),
            "lambda_assoc": float(config_entry["lambda_assoc"]),
            "human_weight": float(config_entry["human_weight"]),
            "candidate_definition_unchanged": True,
            "checkpoint_unchanged": True,
            "metric_definition_unchanged": True,
        },
        "runtime_boundary": {
            "runtime_future_gt_used": False,
            "future_gt_fields_sent": [],
            "gt_loaded_in_worker": False,
            "event_frame_memory_read": False,
            "event_frame_memory_write_hidden": True,
            "first_memory_read_frame": event_frame + 1,
        },
        "candidate_stream_contract": {
            "same_frozen_tape_for_all_variants": True,
            "candidate_complete_required": True,
            "candidate_order_preserved": True,
            "public_id_mapping_audited_per_frame": True,
            "no_duplicate_or_missing_future_frames": True,
        },
        "variants": configs_payload,
        "started_at": started,
        "finished_at": now(),
    }
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-determinism", action="store_true")
    args = parser.parse_args()
    try:
        result = run(
            args.event_id,
            args.source_id,
            args.config_id,
            args.output,
            check_determinism=bool(args.check_determinism),
        )
        print(json.dumps({"status": result["status"], "event_id": args.event_id, "source_id": args.source_id, "config_id": args.config_id, "output": str(args.output)}, sort_keys=True), flush=True)
    except Exception as exc:
        failure = {
            "protocol": PROTOCOL,
            "status": "FAIL",
            "event_id": str(args.event_id),
            "source_id": str(args.source_id),
            "config_id": str(args.config_id),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "artifact_is_failure_evidence": True,
            "failed_at": now(),
        }
        atomic_json(args.output, failure)
        print(json.dumps({"status": "FAIL", "event_id": args.event_id, "source_id": args.source_id, "config_id": args.config_id, "error": failure["error"], "output": str(args.output)}, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    main()
