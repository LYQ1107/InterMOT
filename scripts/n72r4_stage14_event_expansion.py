#!/usr/bin/env python3
"""Freeze an independently auditable N72R4 exploratory event expansion.

This stage is deliberately CPU-only.  It reads the N37 *event discovery
pool* and the immutable N36 tape manifest, but it does not read N37 replay
artifacts and it never assigns a public identity.  The output is therefore a
candidate manifest for later N72R4 persistent-prestate/official-SAM3 work,
not a scientific effect result.

The historical N37 pool contains a few smoke/retry files and nested N8
observer records.  The selector uses only a small allow-list of current-frame
and prefix fields.  In particular, old canonical/public IDs, replay metrics,
and nested post-treatment state are intentionally excluded from the new
manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N37_ROOT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/outputs/n37"
)
N36_ROOT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/outputs/n36"
)
OUT = ROOT / "outputs" / "N72R4"
EXPANSION_ROOT = OUT / "expansion"
DEFAULT_PROTOCOL = EXPANSION_ROOT / "stage14_event_expansion_protocol.json"
DEFAULT_AUDIT = EXPANSION_ROOT / "stage14_event_pool_audit.json"
DEFAULT_MANIFEST = EXPANSION_ROOT / "expanded_event_manifest.json"
DEFAULT_STATUS = OUT / "stage_status" / "stage_14_status.json"
DEFAULT_FAILURE_ROOT = OUT / "attempts" / "stage14"
KNOWN_FAILURE_PLAN = N37_ROOT / "global_atomic_replacement_plan_attempt2.json"
PENDING_MAPPING_AUDIT = N37_ROOT / "atomic_id_swap_precondition_consistency_audit_attempt1.json"

ACTION_TYPES = (
    "ADD_NEW_IDENTITY",
    "AUTHORITATIVE_REASSIGN",
    "ATOMIC_ID_SWAP",
    "RECOVER_IDENTITY",
)
MIN_ACTION_QUOTA = 4
MIN_SEQUENCE_QUOTA = 20
TARGET_EVENTS = 40
SAME_SEQUENCE_GAP = 101
HORIZON = 100

# Only fields that describe the candidate at t or the available prefix are
# permitted to influence the frozen expansion.  We do not copy the nested
# ``n8_event`` object, because it contains old observer state and historical
# canonical/public-ID names that are not N72R4 authority.
ALLOWED_POOL_FIELDS = {
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


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def finite_box(value: Any) -> list[float]:
    box = [float(item) for item in value]
    if len(box) != 4 or not all(math.isfinite(item) for item in box):
        raise ValueError("gt_box must contain four finite values")
    if not (box[2] > box[0] and box[3] > box[1]):
        raise ValueError("gt_box has non-positive area")
    return box


def load_source_sequences(protocol: dict[str, Any]) -> list[str]:
    sequences = protocol.get("source_sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("N37 protocol has no frozen source_sequences")
    values = [str(item) for item in sequences]
    if len(values) != len(set(values)):
        raise ValueError("N37 source_sequences contains duplicates")
    return values


def load_tape_index() -> dict[str, dict[str, Any]]:
    manifest_path = N36_ROOT / "real_tape" / "tape_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS":
        raise ValueError("N36 real tape manifest is not PASS")
    if manifest.get("runtime_future_gt_used") is not False:
        raise ValueError("N36 tape manifest permits runtime future GT")
    index: dict[str, dict[str, Any]] = {}
    for record in manifest.get("completed", []):
        sequence = str(record["sequence"])
        path = N36_ROOT / "real_tape" / "frames" / f"{sequence}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"N36 frame tape is missing: {path}")
        if record.get("status") != "PASS" or record.get("candidate_complete") is not True:
            raise ValueError(f"N36 tape record is incomplete: {sequence}")
        index[sequence] = {
            "record": dict(record),
            "path": path,
            "sha256": sha256_file(path),
        }
    if len(index) != int(manifest.get("sequence_count_expected", -1)):
        raise ValueError("N36 tape sequence count does not match its manifest")
    return index


def load_candidate_pool(sequences: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool_root = N37_ROOT / "event_candidates"
    candidates: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    excluded_files: list[str] = []
    for sequence in sequences:
        path = pool_root / f"{sequence}.json"
        if not path.is_file():
            raise FileNotFoundError(f"frozen N37 candidate pool is missing: {path}")
        document = read_json(path)
        if document.get("status") != "PASS":
            raise ValueError(f"N37 candidate pool is not PASS: {path}")
        if document.get("runtime_future_gt_used") is not False:
            raise ValueError(f"N37 candidate pool permits runtime GT: {path}")
        file_record = {
            "sequence": sequence,
            "path": str(path),
            "sha256": sha256_file(path),
            "candidate_count": len(document.get("candidates", [])),
            "eligible_candidate_count": int(document.get("eligible_candidate_count", 0)),
            "candidate_pool_truncated": bool(document.get("candidate_pool_truncated", False)),
        }
        files.append(file_record)
        for item in document.get("candidates", []):
            if not isinstance(item, dict) or item.get("accepted") is not True:
                continue
            # An explicit allow-list is used rather than copying a historical
            # record.  This makes the absence of replay-derived fields
            # machine-auditable and prevents accidental authority leakage.
            missing = sorted(ALLOWED_POOL_FIELDS - set(item))
            if missing:
                raise ValueError(f"candidate {item.get('candidate_id')} missing {missing}")
            row = {key: item.get(key) for key in ALLOWED_POOL_FIELDS}
            row["sequence"] = sequence
            row["source_pool_file"] = str(path)
            row["source_pool_sha256"] = file_record["sha256"]
            candidates.append(row)
    # The source list is authoritative; any similarly named smoke files are
    # intentionally not read and are recorded as excluded by policy.
    for path in sorted(pool_root.glob("*.json")):
        if path.stem not in set(sequences):
            excluded_files.append(str(path))
    return candidates, {
        "files": files,
        "excluded_noncanonical_pool_files": excluded_files,
        "candidate_pool_root": str(pool_root),
    }


def load_known_failed_candidate_ids() -> dict[str, Any]:
    """Load immutable pre-materialization failures as selection exclusions.

    These are not replay outcomes.  Reusing one would repeat a known
    event-builder/precondition failure and would make the new expansion
    manifest claim an event that cannot enter the N72R4 runtime contract.
    """

    document = read_json(KNOWN_FAILURE_PLAN)
    failed = [str(item) for item in document.get("failed_original_atomic_slots", [])]
    if document.get("runtime_future_gt_used") is not False:
        raise ValueError("known atomic-failure plan has an invalid runtime GT flag")
    if document.get("replay_metrics_used") is not False:
        raise ValueError("known atomic-failure plan was selected from replay outcomes")
    mapping_audit = read_json(PENDING_MAPPING_AUDIT)
    if mapping_audit.get("runtime_future_gt_used") is not False or mapping_audit.get("replay_metrics_used") is not False:
        raise ValueError("atomic mapping audit has an invalid future/replay provenance flag")
    if mapping_audit.get("all_candidates_rejected_by_builder_precondition") is not True:
        raise ValueError("atomic mapping audit does not establish a complete rejection set")
    pending = [str(item["candidate_id"]) for item in mapping_audit.get("candidates", [])]
    combined = list(dict.fromkeys(failed + pending))
    return {
        "path": str(KNOWN_FAILURE_PLAN),
        "sha256": sha256_file(KNOWN_FAILURE_PLAN),
        "status": document.get("status"),
        "candidate_ids": combined,
        "prior_materialization_failure_ids": failed,
        "pending_mapping_audit": {
            "path": str(PENDING_MAPPING_AUDIT),
            "sha256": sha256_file(PENDING_MAPPING_AUDIT),
            "candidate_ids": pending,
            "status": mapping_audit.get("status"),
        },
        "reason": "prior N37 materialization/precondition failure or unresolved atomic mapping audit; excluded before N72R4 selection",
    }


def validate_candidate(row: dict[str, Any], tape: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    action = str(row.get("action_type"))
    if action not in ACTION_TYPES:
        reasons.append("unknown_action")
    if not str(row.get("candidate_id", "")):
        reasons.append("missing_candidate_id")
    try:
        frame = int(row["frame"])
        frame_count = int(row["frame_count"])
        if frame < 0 or frame + HORIZON >= frame_count:
            reasons.append("h100_not_inside_sequence")
    except (TypeError, ValueError, KeyError):
        reasons.append("invalid_frame_range")
    if row.get("h100_available") is not True:
        reasons.append("h100_unavailable")
    try:
        if int(row["candidate_count"]) < 2:
            reasons.append("single_candidate_context")
        if int(row["gt_visible_count"]) < 2:
            reasons.append("insufficient_current_visible_context")
    except (TypeError, ValueError, KeyError):
        reasons.append("invalid_context_counts")
    if row.get("multi_identity_context") is not True:
        reasons.append("not_multi_identity_context")
    try:
        finite_box(row["gt_box"])
    except (TypeError, ValueError, KeyError):
        reasons.append("invalid_current_gt_box")
    try:
        target = int(row["dataset_gt_id"])
        if target < 0:
            reasons.append("invalid_target_dataset_gt_id")
    except (TypeError, ValueError, KeyError):
        reasons.append("missing_target_dataset_gt_id")
    if action == "ATOMIC_ID_SWAP":
        try:
            other = int(row["other_dataset_gt_id"])
            if other < 0 or other == int(row["dataset_gt_id"]):
                reasons.append("invalid_atomic_competitor_gt_id")
        except (TypeError, ValueError, KeyError):
            reasons.append("missing_atomic_competitor_gt_id")
    if str(row.get("sequence")) not in tape:
        reasons.append("missing_n36_tape")
    else:
        record = tape[str(row["sequence"])]["record"]
        if int(record["frame_count"]) != int(row["frame_count"]):
            reasons.append("frame_count_disagrees_with_n36_tape")
        source_hash = str(row.get("source_pool_sha256"))
        if not source_hash:
            reasons.append("missing_candidate_pool_hash")
    return not reasons, reasons


def nonoverlap(candidate: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
    sequence = str(candidate["sequence"])
    frame = int(candidate["frame"])
    return all(
        sequence != str(item["sequence"])
        or abs(frame - int(item["frame"])) >= SAME_SEQUENCE_GAP
        for item in selected
    )


def deterministic_select(valid: list[dict[str, Any]], sequence_order: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    order = {sequence: index for index, sequence in enumerate(sequence_order)}
    ordered = sorted(valid, key=lambda item: (order[str(item["sequence"])], int(item["frame"]), str(item["candidate_id"])))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(item: dict[str, Any]) -> bool:
        candidate_id = str(item["candidate_id"])
        if candidate_id in selected_ids or not nonoverlap(item, selected):
            return False
        selected.append(item)
        selected_ids.add(candidate_id)
        return True

    # First guarantee sequence coverage, then action quotas, then fill in the
    # frozen global order.  No score, replay outcome, or future metric enters.
    for sequence in sequence_order:
        for item in ordered:
            if str(item["sequence"]) == sequence and add(item):
                break
    for action in ACTION_TYPES:
        while sum(str(item["action_type"]) == action for item in selected) < MIN_ACTION_QUOTA:
            if not any(str(item["action_type"]) == action and add(item) for item in ordered):
                break
    for item in ordered:
        if len(selected) >= TARGET_EVENTS:
            break
        add(item)

    selected.sort(key=lambda item: (order[str(item["sequence"])], int(item["frame"]), str(item["candidate_id"])))
    counts = Counter(str(item["action_type"]) for item in selected)
    sequences = sorted({str(item["sequence"]) for item in selected}, key=lambda value: order[value])
    return selected, {
        "target_event_count": TARGET_EVENTS,
        "selected_event_count": len(selected),
        "selected_action_counts": dict(sorted(counts.items())),
        "selected_sequences": sequences,
        "selected_sequence_count": len(sequences),
        "minimum_action_quota": MIN_ACTION_QUOTA,
        "minimum_sequence_quota": MIN_SEQUENCE_QUOTA,
        "same_sequence_gap": SAME_SEQUENCE_GAP,
        "selection_sort": "frozen source sequence order, frame ascending, candidate_id ascending",
        "selection_uses_future_metrics": False,
    }


def make_manifest_event(row: dict[str, Any], tape: dict[str, Any], ordinal: int) -> dict[str, Any]:
    sequence = str(row["sequence"])
    frame = int(row["frame"])
    source_id = str(row["candidate_id"])
    event_id = f"n72r4-expansion-{source_id}"
    return {
        "schema_version": "N72R4_EXPANDED_SIMULATED_EVENT_V1",
        "event_id": event_id,
        "source_candidate_id": source_id,
        "source_pool_file": str(row["source_pool_file"]),
        "source_pool_sha256": str(row["source_pool_sha256"]),
        "candidate_tape_ref": str(tape[sequence]["path"]),
        "candidate_tape_sha256": str(tape[sequence]["sha256"]),
        "sequence": sequence,
        "action_type": str(row["action_type"]),
        "event_frame": frame,
        "prefix_range": [0, frame - 1],
        "future_window": [frame + 1, frame + HORIZON],
        "sequence_frame_count": int(row["frame_count"]),
        "dataset_gt_id": int(row["dataset_gt_id"]),
        "other_dataset_gt_id": (
            None if row.get("other_dataset_gt_id") is None else int(row["other_dataset_gt_id"])
        ),
        "current_gt_box": finite_box(row["gt_box"]),
        "current_frame_selection_evidence": {
            "candidate_count": int(row["candidate_count"]),
            "gt_visible_count": int(row["gt_visible_count"]),
            "h100_available": bool(row["h100_available"]),
            "multi_identity_context": bool(row["multi_identity_context"]),
            "event_type": str(row["event_type"]),
            "seen_before": bool(row["seen_before"]),
            "gap_length": None if row.get("gap_length") is None else int(row["gap_length"]),
            "pre_box": row.get("pre_box"),
        },
        "public_id_resolution": "N72R4_event_prestate_and_current_Y_pre_oracle_only",
        "target_public_id": None,
        "other_public_id": None,
        "public_id_from_candidate_index": False,
        "public_id_from_raw_sam_id": False,
        "public_id_from_gt_id": False,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "gt_used_only_offline_event_generation": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "selection_post_treatment_fields_used": [],
        "selection_ordinal": int(ordinal),
    }


def write_failure(path: Path, exc: BaseException, attempt: str) -> None:
    atomic_json(
        path,
        {
            "schema_version": "N72R4_FAILURE_RECORD_V1",
            "stage": "14_EVENT_EXPANSION_POOL_AUDIT",
            "status": "FAIL_PRESERVED",
            "attempt": attempt,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "selection_uses_future_metrics": False,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="attempt1")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    args = parser.parse_args()
    failure_path = DEFAULT_FAILURE_ROOT / f"stage14_event_expansion_{args.attempt}_failure.json"
    try:
        n37_protocol_path = N37_ROOT / "event_protocol_repaired_attempt2.json"
        n37_protocol = read_json(n37_protocol_path)
        sequences = load_source_sequences(n37_protocol)
        tape = load_tape_index()
        missing_sequences = [sequence for sequence in sequences if sequence not in tape]
        if missing_sequences:
            raise RuntimeError(f"N36 tape missing protocol sequences: {missing_sequences}")
        pool, pool_meta = load_candidate_pool(sequences)
        known_failures = load_known_failed_candidate_ids()
        known_failed_ids = set(known_failures["candidate_ids"])
        valid: list[dict[str, Any]] = []
        invalid: list[dict[str, Any]] = []
        known_failure_exclusions: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for row in pool:
            candidate_id = str(row["candidate_id"])
            if candidate_id in seen_ids:
                invalid.append({"candidate_id": candidate_id, "reasons": ["duplicate_candidate_id"]})
                continue
            seen_ids.add(candidate_id)
            ok, reasons = validate_candidate(row, tape)
            if ok:
                if candidate_id in known_failed_ids:
                    exclusion_reason = (
                        "known_prior_materialization_failure"
                        if candidate_id in set(known_failures.get("prior_materialization_failure_ids", []))
                        else "unresolved_atomic_mapping_precondition"
                    )
                    known_failure_exclusions.append(
                        {
                            "candidate_id": candidate_id,
                            "sequence": row.get("sequence"),
                            "action_type": row.get("action_type"),
                            "reason": exclusion_reason,
                        }
                    )
                else:
                    valid.append(row)
            else:
                invalid.append({"candidate_id": candidate_id, "sequence": row.get("sequence"), "reasons": reasons})
        selected, selection = deterministic_select(valid, sequences)
        action_pool_counts = Counter(str(row["action_type"]) for row in valid)
        action_selected_counts = Counter(str(row["action_type"]) for row in selected)
        selected_sequence_counts = Counter(str(row["sequence"]) for row in selected)
        stage14_gate = {
            "selected_count_target_met": len(selected) >= TARGET_EVENTS,
            "minimum_sequence_quota_met": len(selected_sequence_counts) >= MIN_SEQUENCE_QUOTA,
            "minimum_action_quota_met": all(action_selected_counts[action] >= MIN_ACTION_QUOTA for action in ACTION_TYPES),
            "all_h100_complete": all(int(row["frame"]) + HORIZON < int(row["frame_count"]) for row in selected),
            "same_sequence_nonoverlap": all(
                nonoverlap(row, [other for other in selected if other is not row]) for row in selected
            ),
            "candidate_pool_runtime_future_gt_used": False,
            "selection_uses_future_metrics": False,
        }
        # This is a required audit, not a soft warning: do not emit a frozen
        # expansion manifest when the quota protocol is not satisfiable.
        required_gate_values = (
            "selected_count_target_met",
            "minimum_sequence_quota_met",
            "minimum_action_quota_met",
            "all_h100_complete",
            "same_sequence_nonoverlap",
            "candidate_pool_runtime_future_gt_used",
            "selection_uses_future_metrics",
        )
        if not all(bool(stage14_gate[key]) for key in required_gate_values[:5]) or any(
            bool(stage14_gate[key]) for key in required_gate_values[5:]
        ):
            raise RuntimeError(f"Stage14 expansion quotas are not satisfiable: {stage14_gate}")
        protocol = {
            "schema_version": "N72R4_STAGE14_EVENT_EXPANSION_PROTOCOL_V1",
            "status": "PASS_STAGE14_EXPANSION_POLICY_FROZEN",
            "created_at_utc": now_utc(),
            "source_n37_protocol": str(n37_protocol_path),
            "source_n37_protocol_sha256": sha256_file(n37_protocol_path),
            "source_n36_tape_manifest": str(N36_ROOT / "real_tape" / "tape_manifest.json"),
            "source_n36_tape_manifest_sha256": sha256_file(N36_ROOT / "real_tape" / "tape_manifest.json"),
            "known_prior_failure_exclusion_source": known_failures,
            "source_split": "train/train_fold",
            "source_sequences": sequences,
            "event_selection": {
                "target_events": TARGET_EVENTS,
                "minimum_independent_sequences": MIN_SEQUENCE_QUOTA,
                "minimum_action_quota_each": MIN_ACTION_QUOTA,
                "same_sequence_event_gap_at_least": SAME_SEQUENCE_GAP,
                "future_horizon": HORIZON,
                "selection_fields": sorted(ALLOWED_POOL_FIELDS),
                "selection_sort": "frozen source sequence order, frame ascending, candidate_id ascending",
                "selection_uses": ["current frame GT box", "current frame context counts", "current/prefix event type", "N36 frame-count completeness"],
                "selection_must_not_use": ["N37 replay artifacts", "H20/H50/H100 outcomes", "future identity error", "IDSW", "post-treatment IoU", "variant scores", "old canonical/public IDs"],
            },
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "public_id_authority": "deferred_to_N72R4_persistent_event_prestate_and_Y_pre_mapping",
        }
        events = [make_manifest_event(row, tape, index) for index, row in enumerate(selected)]
        manifest = {
            "schema_version": "N72R4_EXPANDED_SIMULATED_EVENT_MANIFEST_V1",
            "status": "PASS_STAGE14_EXPANSION_POLICY_FROZEN",
            "created_at_utc": now_utc(),
            "protocol": str(args.protocol),
            "protocol_sha256": json_hash(protocol),
            "event_count": len(events),
            "independent_sequence_count": len({item["sequence"] for item in events}),
            "action_counts": dict(sorted(Counter(item["action_type"] for item in events).items())),
            "events": events,
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "scientific_result": "EVENT_SELECTION_ONLY_NO_FULL_LOOP_OR_EFFECT_RESULT",
        }
        audit = {
            "schema_version": "N72R4_STAGE14_EVENT_POOL_AUDIT_V1",
            "status": "PASS_STAGE14_EXPANSION_POOL_AUDIT",
            "created_at_utc": now_utc(),
            "source": {
                "n37_root": str(N37_ROOT),
                "n37_protocol": str(n37_protocol_path),
                "n36_tape_root": str(N36_ROOT / "real_tape"),
                "pool": pool_meta,
            },
            "candidate_pool_total": len(pool),
            "candidate_pool_unique": len(seen_ids),
            "candidate_pool_valid": len(valid),
            "candidate_pool_invalid": len(invalid),
            "known_failure_exclusion_count": len(known_failure_exclusions),
            "known_failure_exclusions": known_failure_exclusions,
            "known_failure_source": known_failures,
            "candidate_pool_action_counts": dict(sorted(action_pool_counts.items())),
            "invalid_reason_counts": dict(sorted(Counter(reason for item in invalid for reason in item["reasons"]).items())),
            "invalid_examples": invalid[:100],
            "selection": selection,
            "selected_sequence_counts": dict(selected_sequence_counts),
            "selected_event_ids": [str(item["event_id"]) for item in events],
            "gate": stage14_gate,
            "excluded_public_or_replay_fields": [
                "canonical_public_id",
                "current_public_id",
                "other_canonical_public_id",
                "target_auto_tid",
                "other_auto_tid",
                "n8_event",
                "observer_memory_hash_before/after",
                "system_state_hash_before/after",
                "variant/replay/H20/H50/H100/post-treatment fields",
            ],
            "runtime_future_gt_used": False,
            "selection_uses_future_metrics": False,
            "scientific_result": "EVENT_SELECTION_ONLY_NO_FULL_LOOP_OR_EFFECT_RESULT",
        }
        status = {
            "schema_version": "N72R4_STAGE_STATUS_V1",
            "stage": "14_EVENT_EXPANSION_POLICY",
            "status": "PASS_STAGE14_EXPANSION_POLICY_FROZEN",
            "created_at_utc": now_utc(),
            "event_count": len(events),
            "independent_sequence_count": len({item["sequence"] for item in events}),
            "action_counts": dict(sorted(Counter(item["action_type"] for item in events).items())),
            "candidate_pool_total": len(pool),
            "candidate_pool_valid": len(valid),
            "candidate_pool_invalid": len(invalid),
            "known_failure_exclusion_count": len(known_failure_exclusions),
            "protocol": str(args.protocol),
            "audit": str(args.audit),
            "manifest": str(args.manifest),
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "scientific_result": "EVENT_SELECTION_ONLY_NO_FULL_LOOP_OR_EFFECT_RESULT",
            "next_gate": "N72R4_EVENT_PRESTATE_AND_OFFICIAL_FULL_LOOP_REQUIRED",
        }
        atomic_json(args.protocol, protocol)
        atomic_json(args.manifest, manifest)
        atomic_json(args.audit, audit)
        atomic_json(args.status, status)
        print(json.dumps({"status": status["status"], "events": len(events), "sequences": status["independent_sequence_count"], "actions": status["action_counts"], "manifest": str(args.manifest)}, sort_keys=True))
        return 0
    except Exception as exc:
        write_failure(failure_path, exc, str(args.attempt))
        failure_status = {
            "schema_version": "N72R4_STAGE_STATUS_V1",
            "stage": "14_EVENT_EXPANSION_POLICY",
            "status": "BLOCKED_STAGE14_EXPANSION_AUDIT",
            "created_at_utc": now_utc(),
            "failure_artifact": str(failure_path),
            "runtime_future_gt_used": False,
            "selection_uses_future_metrics": False,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
        }
        atomic_json(args.status, failure_status)
        print(json.dumps({"status": failure_status["status"], "failure": str(failure_path)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
