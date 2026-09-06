#!/usr/bin/env python3
"""Freeze the deterministic N72R11 secondary-interaction schedule.

This stage only audits current-frame availability and writes a preregistered
schedule.  It does not run SAM3, read future GT, select outcomes, or create a
synthetic event artifact.  Actual secondary sessions are generated later by
the isolated runtime worker.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
OUTPUT_ROOT = ROOT / "outputs/N72R11"
EVENT_PROTOCOL = OUTPUT_ROOT / "event_protocol.json"
MANIFEST = OUTPUT_ROOT / "secondary_event_manifest.json"
STATUS = OUTPUT_ROOT / "stage_05_status.json"
DATA_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
OFFSETS = tuple(range(0, 81, 5))
ACTION_TYPES = ("ADD_NEW_IDENTITY", "AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP", "RECOVER_IDENTITY")


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
                raise TypeError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_gt(sequence: str) -> dict[int, dict[int, list[float]]]:
    path = DATA_ROOT / "train" / str(sequence) / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, list[float]]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) < 6:
            raise ValueError(f"malformed GT row {path}:{line_number}")
        frame = int(fields[0]) - 1
        gt_id = int(fields[1])
        x, y, width, height = (float(value) for value in fields[2:6])
        result.setdefault(frame, {})[gt_id] = [x, y, x + width, y + height]
    return result


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def event_target_public(event: Mapping[str, Any]) -> int:
    manifest = read_json(resolve(str(event["source_event_manifest"])))
    value = manifest.get("target_public_id")
    if value is None:
        raise RuntimeError(f"source event has no explicit target public authority: {event['event_id']}")
    return int(value)


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(PROTOCOL)
    source_events = [dict(item) for item in protocol.get("source_event_selection", {}).get("events", [])]
    if len(source_events) != 32 or len({str(item["event_id"]) for item in source_events}) != 32:
        raise RuntimeError(f"expected 32 frozen N72R9 source events, found {len(source_events)}")
    sequences = sorted({str(item["sequence"]) for item in source_events})
    if len(sequences) != 18:
        raise RuntimeError(f"expected 18 development sequences, found {len(sequences)}")
    train_sequences = sequences[:12]
    validation_sequences = sequences[12:]
    events: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    action_counts = {action: 0 for action in ACTION_TYPES}
    skip_counts: dict[str, int] = {}
    for original in sorted(source_events, key=lambda item: str(item["event_id"])):
        sequence = str(original["sequence"])
        original_frame = int(original["event_frame"])
        action = str(original["action_type"])
        target_gid = int(original["dataset_gt_id"])
        target_public = event_target_public(original)
        c0_path = resolve(str(original["c0_source"]))
        if not c0_path.is_file():
            raise FileNotFoundError(c0_path)
        c0_rows = {int(row["frame"]): row for row in read_jsonl(c0_path)}
        gt = load_gt(sequence)
        for offset in OFFSETS:
            secondary_frame = original_frame + int(offset)
            event_id = f"{original['event_id']}:secondary:{offset:03d}"
            row = c0_rows.get(secondary_frame)
            status = "ELIGIBLE"
            reason = None
            target_box = None
            if row is None:
                status, reason = "SKIP_B0_FRAME_UNAVAILABLE", "SKIP_B0_FRAME_UNAVAILABLE"
            elif target_public not in {int(value) for value in row.get("public_id_axis", [])}:
                status, reason = "SKIP_TARGET_PUBLIC_NOT_LIVE", "SKIP_TARGET_PUBLIC_NOT_LIVE"
            elif target_gid not in gt.get(secondary_frame, {}):
                status, reason = "SKIP_CURRENT_TARGET_NOT_VISIBLE", "SKIP_CURRENT_TARGET_NOT_VISIBLE"
            else:
                target_box = list(gt[secondary_frame][target_gid])
            if status != "ELIGIBLE":
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
            else:
                action_counts[action] += 1
            decisions.append(
                {
                    "event_id": event_id,
                    "original_event_id": str(original["event_id"]),
                    "sequence": sequence,
                    "action_type": action,
                    "split": "train" if sequence in train_sequences else "validation",
                    "original_event_frame": original_frame,
                    "secondary_offset": int(offset),
                    "secondary_frame": secondary_frame,
                    "target_dataset_gt_id": target_gid,
                    "target_public_id": target_public,
                    "status": status,
                    "skip_reason": reason,
                    "current_target_box_posthoc_selection_only": target_box,
                    "c0_source": str(c0_path),
                    "c0_source_sha256": sha256_file(c0_path),
                    "runtime_future_gt_used": False,
                    "interaction_source": "simulated_from_gt" if status == "ELIGIBLE" else None,
                    "not_real_human_evidence": True if status == "ELIGIBLE" else None,
                    "post_treatment_fields_used_for_selection": False,
                }
            )
            if status == "ELIGIBLE":
                events.append(decisions[-1])
    protocol_out = {
        "schema_version": "N72R11_SECONDARY_EVENT_PROTOCOL_V1",
        "status": "PASS_SECONDARY_SCHEDULE_FROZEN",
        "created_at_utc": now_utc(),
        "source_protocol": str(PROTOCOL),
        "source_protocol_sha256": sha256_file(PROTOCOL),
        "source_event_count": len(source_events),
        "source_sequence_count": len(sequences),
        "train_sequences": train_sequences,
        "validation_sequences": validation_sequences,
        "offsets": list(OFFSETS),
        "selection_rule": "fixed offsets 0..80 step 5; B0 public axis live; same-frame target GT visible; no future outcome fields",
        "secondary_end_rule": "min(secondary_frame+100, original_event_frame+100)",
        "action_quota_rule": "report actual legal counts; do not relax or outcome-select",
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "runtime_future_gt_used": False,
        "post_treatment_fields_used_for_selection": False,
        "future_metrics_forbidden_during_selection": ["identity_error", "missing", "IoU", "IDSW", "H20", "H50", "H100", "requery_success"],
    }
    manifest = {
        "schema_version": "N72R11_SECONDARY_EVENT_MANIFEST_V1",
        "status": "PASS_SECONDARY_EVENT_SCHEDULE_AUDITED",
        "created_at_utc": now_utc(),
        "protocol": str(EVENT_PROTOCOL),
        "source_event_count": len(source_events),
        "candidate_decision_count": len(decisions),
        "eligible_event_count": len(events),
        "eligible_sequence_count": len({str(item["sequence"]) for item in events}),
        "eligible_by_split": {
            "train": sum(str(item["split"]) == "train" for item in events),
            "validation": sum(str(item["split"]) == "validation" for item in events),
        },
        "eligible_by_action": action_counts,
        "skip_counts": dict(sorted(skip_counts.items())),
        "events": events,
        "all_decisions": decisions,
        "runtime_generated": False,
        "runtime_future_gt_used": False,
        "real_human_evidence": False,
        "not_real_human_evidence": True,
    }
    return protocol_out, manifest


def main() -> int:
    started = now_utc()
    try:
        protocol_out, manifest = build()
        atomic_json(EVENT_PROTOCOL, protocol_out)
        manifest["protocol_sha256"] = sha256_file(EVENT_PROTOCOL)
        atomic_json(MANIFEST, manifest)
        result = {
            "schema_version": "N72R11_STAGE_STATUS_V1",
            "stage": "N72R11-05",
            "status": "PASS_SECONDARY_SCHEDULE_FROZEN",
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
            "command": "python scripts/n72r11_build_secondary_schedule.py",
            "event_protocol": str(EVENT_PROTOCOL),
            "event_protocol_sha256": sha256_file(EVENT_PROTOCOL),
            "manifest": str(MANIFEST),
            "manifest_sha256": sha256_file(MANIFEST),
            "eligible_event_count": manifest["eligible_event_count"],
            "eligible_sequence_count": manifest["eligible_sequence_count"],
            "eligible_by_split": manifest["eligible_by_split"],
            "eligible_by_action": manifest["eligible_by_action"],
            "skip_counts": manifest["skip_counts"],
            "runtime_generated": False,
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
        }
        atomic_json(STATUS, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        failure = OUTPUT_ROOT / "attempts" / f"secondary_schedule_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        payload = {
            "schema_version": "N72R11_STAGE_FAILURE_V1",
            "stage": "N72R11-05",
            "status": "FAIL_SECONDARY_SCHEDULE",
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "historical_outputs_modified": False,
        }
        atomic_json(failure, payload)
        print(json.dumps({"status": payload["status"], "failure_artifact": str(failure), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
