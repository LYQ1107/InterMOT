#!/usr/bin/env python3
"""Freeze the new N72R5 GT-simulated event policy.

This is a CPU-only selection step.  It intentionally does not reuse the old
N72R4 selected event set and never reads any N72R4/N72R5 replay outcome.  The
only GT fields copied into the manifest describe the event-frame interaction
offline; runtime workers must not load them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "N72R5"
ROUND_ROOT = Path(
    os.environ.get(
        "N72R5_STAGE06_ROOT",
        str(OUT / "mechanism_rounds" / "round_06_event_policy"),
    )
)
PROTOCOL_PATH = ROUND_ROOT / "event_policy_protocol.json"
POOL_AUDIT_PATH = ROUND_ROOT / "candidate_pool_audit.json"
MANIFEST_PATH = ROUND_ROOT / "real_event_manifest.json"
STAGE_STATUS = OUT / "stage_status" / "stage_06_status.json"

N37_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/outputs/n37")
N36_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/outputs/n36")
N72R4_PROTOCOL = Path(
    "/data2/usr_for_deadline/SAM3_InterMOT_N72R3R1/worktree/outputs/N72R4/expansion/stage14_event_expansion_protocol.json"
)
KNOWN_FAILURE_PLAN = N37_ROOT / "global_atomic_replacement_plan_attempt2.json"

ACTION_TYPES = (
    "ADD_NEW_IDENTITY",
    "AUTHORITATIVE_REASSIGN",
    "ATOMIC_ID_SWAP",
    "RECOVER_IDENTITY",
)
ACTION_TARGETS = {
    "ADD_NEW_IDENTITY": 5,
    "AUTHORITATIVE_REASSIGN": 14,
    "ATOMIC_ID_SWAP": 8,
    "RECOVER_IDENTITY": 13,
}
TARGET_EVENTS = 40
MIN_SEQUENCE_COUNT = 20
SAME_SEQUENCE_EVENT_GAP = 101
HORIZON = 100
SELECTION_ORDER = (
    "source_sequence_order",
    "frame_ascending",
    "candidate_id_ascending",
)
# Historical dancetrack0015 atomic slots were explicitly held out while the
# native/public mapping semantics were under audit.  They remain excluded
# from any new PASS manifest; no later replay result may override this.
UNRESOLVED_ATOMIC_FRAMES = {
    "dancetrack0015": {772, 773, 774, 796},
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def source_sequences() -> list[str]:
    protocol = read_json(N72R4_PROTOCOL)
    values = [str(item) for item in protocol.get("source_sequences", [])]
    if len(values) != 24 or len(values) != len(set(values)):
        raise RuntimeError(f"frozen train sequence list invalid: {len(values)}")
    return values


def load_tape_index(sequences: list[str]) -> dict[str, dict[str, Any]]:
    manifest_path = N36_ROOT / "real_tape" / "tape_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS" or manifest.get("runtime_future_gt_used") is not False:
        raise RuntimeError("N36 tape manifest is not a valid frozen input")
    index: dict[str, dict[str, Any]] = {}
    for item in manifest.get("completed", []):
        sequence = str(item.get("sequence"))
        if sequence not in sequences:
            continue
        path = N36_ROOT / "real_tape" / "frames" / f"{sequence}.jsonl"
        if item.get("status") != "PASS" or item.get("candidate_complete") is not True or not path.is_file():
            raise RuntimeError(f"N36 tape entry is incomplete: {sequence}")
        index[sequence] = {
            "path": path,
            "sha256": sha256_file(path),
            "frame_count": int(item.get("frame_count", -1)),
            "manifest_record": dict(item),
        }
    if set(index) != set(sequences):
        raise RuntimeError(f"N36 tape coverage mismatch: missing={sorted(set(sequences)-set(index))}")
    return index


def load_failed_ids() -> set[str]:
    if not KNOWN_FAILURE_PLAN.is_file():
        return set()
    payload = read_json(KNOWN_FAILURE_PLAN)
    if payload.get("runtime_future_gt_used") is not False or payload.get("replay_metrics_used") is not False:
        raise RuntimeError("known atomic failure plan has invalid provenance")
    return {str(item) for item in payload.get("failed_original_atomic_slots", [])}


def finite_box(value: Any) -> list[float]:
    box = [float(item) for item in value]
    if len(box) != 4 or not all(math.isfinite(item) for item in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("current event box is not finite positive XYXY")
    return box


def load_pool(sequences: list[str], tape_index: dict[str, dict[str, Any]], failed_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool_root = N37_ROOT / "event_candidates"
    rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    required = {
        "accepted",
        "action_type",
        "candidate_count",
        "candidate_id",
        "dataset_gt_id",
        "event_type",
        "frame",
        "frame_count",
        "gap_length",
        "gt_box",
        "gt_visible_count",
        "h100_available",
        "multi_identity_context",
        "other_dataset_gt_id",
        "pre_box",
        "seen_before",
    }
    for sequence in sequences:
        path = pool_root / f"{sequence}.json"
        document = read_json(path)
        if document.get("status") != "PASS" or document.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"candidate pool is not a valid frozen input: {path}")
        file_info = {
            "sequence": sequence,
            "path": str(path),
            "sha256": sha256_file(path),
            "candidate_count": len(document.get("candidates", [])),
        }
        files.append(file_info)
        for item in document.get("candidates", []):
            candidate_id = str(item.get("candidate_id"))
            reasons: list[str] = []
            missing = sorted(required - set(item))
            if missing:
                reasons.append("missing:" + ",".join(missing))
            action = str(item.get("action_type"))
            if item.get("accepted") is not True:
                reasons.append("not_accepted")
            if action not in ACTION_TYPES:
                reasons.append("unknown_action")
            if candidate_id in failed_ids:
                reasons.append("known_builder_failure_excluded")
            if sequence in UNRESOLVED_ATOMIC_FRAMES and int(item.get("frame", -1)) in UNRESOLVED_ATOMIC_FRAMES[sequence]:
                reasons.append("historical_atomic_mapping_semantics_unresolved")
            if item.get("h100_available") is not True:
                reasons.append("h100_unavailable")
            if item.get("multi_identity_context") is not True:
                reasons.append("not_multi_identity_context")
            try:
                frame = int(item.get("frame"))
                frame_count = int(tape_index[sequence]["frame_count"])
                if frame < 1:
                    reasons.append("no_t_minus_one_prestate")
                if frame + HORIZON >= frame_count:
                    reasons.append("future_window_out_of_tape")
                current_box = finite_box(item.get("gt_box"))
            except (KeyError, TypeError, ValueError):
                reasons.append("invalid_current_frame_box_or_frame")
                frame = -1
                current_box = None
            if reasons:
                excluded.append({"sequence": sequence, "candidate_id": candidate_id, "reasons": reasons})
                continue
            rows.append(
                {
                    "sequence": sequence,
                    "frame": frame,
                    "candidate_id": candidate_id,
                    "action_type": action,
                    "event_type": str(item["event_type"]),
                    "dataset_gt_id": int(item["dataset_gt_id"]),
                    "other_dataset_gt_id": None if item.get("other_dataset_gt_id") is None else int(item["other_dataset_gt_id"]),
                    "gt_box": current_box,
                    "pre_box": None if item.get("pre_box") is None else finite_box(item["pre_box"]),
                    "candidate_count": int(item["candidate_count"]),
                    "frame_count": int(item["frame_count"]),
                    "gap_length": None if item.get("gap_length") is None else int(item["gap_length"]),
                    "gt_visible_count": int(item["gt_visible_count"]),
                    "h100_available": True,
                    "multi_identity_context": True,
                    "seen_before": bool(item["seen_before"]),
                    "source_pool_file": str(path),
                    "source_pool_sha256": file_info["sha256"],
                    "source_candidate_record": {
                        "accepted": True,
                        "action_type": action,
                        "event_type": str(item["event_type"]),
                        "frame": frame,
                        "candidate_count": int(item["candidate_count"]),
                        "multi_identity_context": True,
                        "h100_available": True,
                    },
                }
            )
    return rows, {
        "candidate_pool_root": str(pool_root),
        "files": files,
        "excluded_count": len(excluded),
        "excluded_reason_counts": dict(sorted(Counter(reason for item in excluded for reason in item["reasons"]).items())),
        "excluded_records": excluded,
        "failed_atomic_candidate_ids_excluded": sorted(failed_ids),
        "valid_candidate_count": len(rows),
    }


def compatible(candidate: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
    return all(
        candidate["sequence"] != item["sequence"]
        or abs(int(candidate["frame"]) - int(item["frame"])) >= SAME_SEQUENCE_EVENT_GAP
        for item in selected
    )


def deterministic_select(valid: list[dict[str, Any]], sequences: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    order = {sequence: index for index, sequence in enumerate(sequences)}
    ordered = sorted(valid, key=lambda item: (order[item["sequence"]], int(item["frame"]), str(item["candidate_id"])))
    selected: list[dict[str, Any]] = []

    # First cover twenty independent sequences without consuming more than an
    # action's frozen target.  The action preference is fixed here; it is not
    # inferred from any replay result.  Filling the action targets afterwards
    # gives exactly forty rows because the targets sum to forty.
    coverage_action_order = (
        "ADD_NEW_IDENTITY",
        "ATOMIC_ID_SWAP",
        "AUTHORITATIVE_REASSIGN",
        "RECOVER_IDENTITY",
    )
    selected_actions = Counter()
    for sequence in sequences:
        if len({item["sequence"] for item in selected}) >= MIN_SEQUENCE_COUNT:
            break
        if sequence in {item["sequence"] for item in selected}:
            continue
        chosen = None
        for action in coverage_action_order:
            candidates = [
                item
                for item in ordered
                if item["sequence"] == sequence
                and item["action_type"] == action
                and selected_actions[action] < ACTION_TARGETS[action]
                and compatible(item, selected)
            ]
            if candidates:
                chosen = candidates[0]
                break
        if chosen is not None:
            selected.append(chosen)
            selected_actions[chosen["action_type"]] += 1

    # Fill each action to its preregistered target.  Do not add arbitrary
    # extras: a shortfall must remain a visible quota block.
    for action in ("ADD_NEW_IDENTITY", "ATOMIC_ID_SWAP", "AUTHORITATIVE_REASSIGN", "RECOVER_IDENTITY"):
        for candidate in ordered:
            if selected_actions[action] >= ACTION_TARGETS[action]:
                break
            if candidate in selected or candidate["action_type"] != action or not compatible(candidate, selected):
                continue
            selected.append(candidate)
            selected_actions[action] += 1

    selected.sort(key=lambda item: (order[item["sequence"]], int(item["frame"]), str(item["candidate_id"])))
    counts = Counter(str(item["action_type"]) for item in selected)
    sequences_selected = [sequence for sequence in sequences if sequence in {item["sequence"] for item in selected}]
    return selected, {
        "target_event_count": TARGET_EVENTS,
        "selected_event_count": len(selected),
        "selected_action_counts": dict(sorted(counts.items())),
        "selected_sequence_count": len(sequences_selected),
        "selected_sequences": sequences_selected,
        "minimum_action_quota_each": 4,
        "action_targets": ACTION_TARGETS,
        "minimum_independent_sequences": MIN_SEQUENCE_COUNT,
        "same_sequence_event_gap": SAME_SEQUENCE_EVENT_GAP,
        "selection_order": SELECTION_ORDER,
        "selection_uses_future_metrics": False,
    }


def make_event(row: dict[str, Any], tape: dict[str, Any], ordinal: int) -> dict[str, Any]:
    event_frame = int(row["frame"])
    source_id = str(row["candidate_id"])
    return {
        "schema_version": "N72R5_SIMULATED_EVENT_POLICY_V1",
        "event_id": f"n72r5-pool-{source_id}",
        "sequence": str(row["sequence"]),
        "action_type": str(row["action_type"]),
        "event_frame": event_frame,
        "prefix_range": [0, event_frame - 1],
        "future_window": [event_frame + 1, event_frame + HORIZON],
        "dataset_gt_id": int(row["dataset_gt_id"]),
        "other_dataset_gt_id": row["other_dataset_gt_id"],
        "current_gt_box": list(row["gt_box"]),
        "pre_box": None if row["pre_box"] is None else list(row["pre_box"]),
        "candidate_tape_ref": str(tape["path"]),
        "candidate_tape_sha256": str(tape["sha256"]),
        "candidate_pool_ref": str(row["source_pool_file"]),
        "candidate_pool_sha256": str(row["source_pool_sha256"]),
        "source_candidate_id": source_id,
        "source_candidate_record": row["source_candidate_record"],
        "current_frame_selection_evidence": {
            "candidate_count": int(row["candidate_count"]),
            "event_type": str(row["event_type"]),
            "gap_length": row["gap_length"],
            "gt_visible_count": int(row["gt_visible_count"]),
            "h100_available": True,
            "multi_identity_context": True,
            "pre_box": None if row["pre_box"] is None else list(row["pre_box"]),
            "seen_before": bool(row["seen_before"]),
        },
        "selection_ordinal": int(ordinal),
        "selection_rule": "N72R5 current-frame/prefix-only deterministic policy; no replay outcome fields",
        "selection_post_treatment_fields_used": [],
        "target_public_id": None,
        "public_id_authority": "deferred_to_N72R5_persistent_prestate_and_current_Y_pre_mapping",
        "public_id_from_gt_id": False,
        "public_id_from_candidate_index": False,
        "public_id_from_raw_sam_id": False,
        "gt_used_only_offline_event_generation": True,
        "runtime_gt_read": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "sequence_frame_count": int(tape["frame_count"]),
    }


def main() -> int:
    if ROUND_ROOT.exists() and any(ROUND_ROOT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty Stage06 root: {ROUND_ROOT}")
    ROUND_ROOT.mkdir(parents=True, exist_ok=True)
    sequences = source_sequences()
    tape_index = load_tape_index(sequences)
    failed_ids = load_failed_ids()
    valid, pool_audit = load_pool(sequences, tape_index, failed_ids)
    selected, selection = deterministic_select(valid, sequences)
    protocol = {
        "schema_version": "N72R5_EVENT_POLICY_PROTOCOL_V1",
        "status": "FROZEN_BEFORE_ANY_NEW_FULL_LOOP_OR_REPLAY",
        "stage": "06_NEW_GT_SIMULATED_EVENT_POOL",
        "hypothesis": "TVC_plus_image_grounded_recovery_requires_cross-sequence event coverage",
        "source_split": "train/train_fold",
        "source_sequences": sequences,
        "target_event_count": TARGET_EVENTS,
        "minimum_independent_sequences": MIN_SEQUENCE_COUNT,
        "same_sequence_event_gap": SAME_SEQUENCE_EVENT_GAP,
        "future_horizon": HORIZON,
        "action_types": list(ACTION_TYPES),
        "action_targets": ACTION_TARGETS,
        "selection_order": SELECTION_ORDER,
        "selection_uses": [
            "current-frame candidate-pool fields",
            "current-frame GT box for offline event construction",
            "prefix/event type and past-state descriptors present in candidate pool",
            "N36 tape frame-count completeness",
        ],
        "selection_must_not_use": [
            "N72R4 Stage11/Stage12/Stage13/Stage14 replay outcomes",
            "N72R5 Stage01/Stage02/Stage03/Stage04 outcomes",
            "H20/H50/H100",
            "identity error",
            "IDSW",
            "post-treatment IoU",
            "variant score or assignment",
        ],
        "known_failed_event_slots_excluded": sorted(failed_ids),
        "public_id_authority": "deferred_to_persistent_prestate_and_current_Y_pre; never inferred from GT/candidate index/native ID",
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "created_at_utc": now_utc(),
    }
    atomic_json(PROTOCOL_PATH, protocol)
    atomic_json(POOL_AUDIT_PATH, {
        "schema_version": "N72R5_EVENT_POOL_AUDIT_V1",
        "status": "PASS_POOL_READ_ONLY_AUDIT" if valid else "BLOCKED_EMPTY_POOL",
        "source_sequences": sequences,
        "source_sequence_count": len(sequences),
        "valid_candidate_count": len(valid),
        "valid_action_counts": dict(sorted(Counter(item["action_type"] for item in valid).items())),
        "valid_sequence_count": len({item["sequence"] for item in valid}),
        "pool": pool_audit,
        "runtime_future_gt_used": False,
        "replay_metrics_used": False,
    })
    events = [make_event(row, tape_index[row["sequence"]], ordinal) for ordinal, row in enumerate(selected)]
    manifest_status = "PASS_N72R5_EVENT_POLICY_FROZEN" if (
        len(events) == TARGET_EVENTS
        and len({event["sequence"] for event in events}) >= MIN_SEQUENCE_COUNT
        and all(sum(int(event["action_type"] == action) for event in events) >= 4 for action in ACTION_TYPES)
    ) else "BLOCKED_N72R5_EVENT_POLICY_QUOTA"
    manifest = {
        "schema_version": "N72R5_SIMULATED_EVENT_MANIFEST_V1",
        "status": manifest_status,
        "created_at_utc": now_utc(),
        "events": events,
        "event_count": len(events),
        "independent_sequence_count": len({event["sequence"] for event in events}),
        "action_counts": dict(sorted(Counter(event["action_type"] for event in events).items())),
        "selection": selection,
        "selection_uses_future_metrics": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "pool_audit_sha256": sha256_file(POOL_AUDIT_PATH),
    }
    atomic_json(MANIFEST_PATH, manifest)
    gate = {
        "schema_version": "N72R5_STAGE06_EVENT_POLICY_GATE_V1",
        "status": manifest_status,
        "event_count": len(events),
        "independent_sequence_count": len({event["sequence"] for event in events}),
        "action_counts": manifest["action_counts"],
        "minimum_event_count_met": len(events) >= TARGET_EVENTS,
        "minimum_sequence_count_met": len({event["sequence"] for event in events}) >= MIN_SEQUENCE_COUNT,
        "minimum_action_quota_met": all(manifest["action_counts"].get(action, 0) >= 4 for action in ACTION_TYPES),
        "runtime_future_gt_used": False,
        "full_loop_authorized": manifest_status == "PASS_N72R5_EVENT_POLICY_FROZEN",
        "training_authorized": False,
        "production_authorized": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }
    atomic_json(ROUND_ROOT / "gate.json", gate)
    atomic_json(
        STAGE_STATUS,
        {
            "schema_version": "N72R5_STAGE_STATUS_V1",
            "stage": "06_NEW_GT_SIMULATED_EVENT_POOL",
            "status": manifest_status,
            "protocol": str(PROTOCOL_PATH),
            "pool_audit": str(POOL_AUDIT_PATH),
            "manifest": str(MANIFEST_PATH),
            "gate": str(ROUND_ROOT / "gate.json"),
            "event_count": len(events),
            "independent_sequence_count": len({event["sequence"] for event in events}),
            "action_counts": manifest["action_counts"],
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "full_loop_authorized": gate["full_loop_authorized"],
            "training_authorized": False,
            "production_authorized": False,
        },
    )
    print(json.dumps({"status": manifest_status, "events": len(events), "sequences": len({event['sequence'] for event in events}), "actions": manifest["action_counts"], "manifest": str(MANIFEST_PATH)}, sort_keys=True))
    return 0 if manifest_status == "PASS_N72R5_EVENT_POLICY_FROZEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
