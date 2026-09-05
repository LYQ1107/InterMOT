#!/usr/bin/env python3
"""CPU-only integrity audit for N72R10 true future-frame event artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "outputs/N72R9/protocol.json"


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
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            import os

            os.fsync(handle.fileno())
        import os

        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        import os

        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def frozen_events() -> list[dict[str, Any]]:
    protocol = read_json(PROTOCOL_PATH)
    events = [dict(item) for item in protocol.get("source_event_selection", {}).get("events", [])]
    if len(events) != 32 or len({str(item.get("event_id")) for item in events}) != 32:
        raise RuntimeError(f"frozen protocol does not contain exactly 32 unique events: {len(events)}")
    return events


def assert_false_flags(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in {"runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"} and item is not False:
                raise RuntimeError(f"{path}.{key} is not false: {item!r}")
            assert_false_flags(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_false_flags(item, f"{path}[{index}]")


def finite_vector(value: Any, *, label: str, expected: int | None = None) -> None:
    if not isinstance(value, list) or (expected is not None and len(value) != expected):
        raise RuntimeError(f"{label} has invalid vector shape")
    if not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value):
        raise RuntimeError(f"{label} contains non-finite values")


def audit_event(event: Mapping[str, Any], event_dir: Path) -> dict[str, Any]:
    event_id = str(event["event_id"])
    required = {"done.json", "result.json", "candidates.json", "audit.json"}
    if not event_dir.is_dir() or any(not (event_dir / name).is_file() for name in required):
        raise RuntimeError(f"missing event artifact files: {event_dir}")
    done = read_json(event_dir / "done.json")
    result = read_json(event_dir / "result.json")
    candidates = read_json(event_dir / "candidates.json")
    audit = read_json(event_dir / "audit.json")
    if done.get("status") != "PASS_N72R10_TRUE_FUTURE_REQUERY_EVENT":
        raise RuntimeError(f"{event_id}: done is not PASS")
    if result.get("status") != "PASS_N72R10_TRUE_FUTURE_REQUERY_EVENT":
        raise RuntimeError(f"{event_id}: result is not PASS")
    for name, payload in (("result", result), ("candidates", candidates), ("audit", audit)):
        if done.get(name + "_sha256") != sha256_file(event_dir / (name + ".json")):
            raise RuntimeError(f"{event_id}: {name} hash mismatch")
        assert_false_flags(payload, name)
    if str(result.get("event_id")) != event_id or str(candidates.get("event_id")) != event_id:
        raise RuntimeError(f"{event_id}: event key mismatch")
    trigger = int(event["event_frame"]) + 1
    end = int(event["event_frame"]) + 100
    if int(result.get("trigger_frame", -1)) != trigger or int(result.get("end_frame", -1)) != end:
        raise RuntimeError(f"{event_id}: future window mismatch")
    if result.get("interaction_source") != "simulated_from_gt" or result.get("not_real_human_evidence") is not True:
        raise RuntimeError(f"{event_id}: interaction source taxonomy is invalid")
    if result.get("target_public_id", 0) <= 0:
        raise RuntimeError(f"{event_id}: target public ID is invalid")
    if result.get("causal_input", {}).get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"{event_id}: causal input used future GT")
    if result.get("lifecycle", {}).get("all_backend_sessions_closed") is not True:
        raise RuntimeError(f"{event_id}: backend lifecycle is not closed")
    if result.get("lifecycle", {}).get("fresh_backend_per_probe_and_active_session") is not True:
        raise RuntimeError(f"{event_id}: fresh backend lifecycle flag missing")
    if result.get("audit_before_close", {}).get("raw_rebinding", {}).get("public_id_changed") is not False:
        raise RuntimeError(f"{event_id}: public ID changed")
    if audit.get("status") != "PASS_SELECTED" or audit.get("closed") is not True:
        raise RuntimeError(f"{event_id}: audit status/lifecycle is invalid")
    if audit.get("event_frame_memory_read") is not False or int(audit.get("first_memory_visible_frame", -1)) != trigger + 1:
        raise RuntimeError(f"{event_id}: causal memory boundary is invalid")
    coverage = list(audit.get("future_frame_coverage", []))
    expected_frames = list(range(trigger, end + 1))
    observed_frames = [int(item.get("global_frame", -1)) for item in coverage]
    if observed_frames != expected_frames or len(coverage) != 100:
        raise RuntimeError(f"{event_id}: coverage is not exactly the 100-frame window")
    future = candidates.get("future_candidates")
    probes = candidates.get("probe_candidates")
    if not isinstance(future, list) or not isinstance(probes, list) or len(probes) != 4:
        raise RuntimeError(f"{event_id}: candidate arrays are incomplete")
    if len({str(item.get("candidate_uid")) for item in probes}) != len(probes):
        raise RuntimeError(f"{event_id}: probe UID duplicate")
    if len({str(item.get("candidate_uid")) for item in future}) != len(future):
        raise RuntimeError(f"{event_id}: future UID duplicate")
    by_frame = Counter(int(item.get("frame", -1)) for item in future)
    for row in probes + future:
        if row.get("candidate_source") != "FUTURE_FRAME_REQUERY" or row.get("public_id") is not None:
            raise RuntimeError(f"{event_id}: candidate source/public authority is invalid")
        if row.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"{event_id}: candidate used runtime future GT")
        box = row.get("box_xyxy")
        finite_vector(box, label=f"{event_id}:box", expected=4)
        if float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]):
            raise RuntimeError(f"{event_id}: serialized candidate box is non-positive")
        feature = row.get("feature")
        finite_vector(feature, label=f"{event_id}:feature", expected=512)
        if float(sum(float(item) ** 2 for item in feature)) <= 1.0e-12:
            raise RuntimeError(f"{event_id}: candidate feature is zero norm")
    for item in coverage:
        frame = int(item["global_frame"])
        if int(item.get("candidate_count", -1)) != int(by_frame.get(frame, 0)):
            raise RuntimeError(f"{event_id}: coverage/candidate count mismatch at frame {frame}")
        if item.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"{event_id}: coverage used future GT")
    if int(result.get("future_frame_count", -1)) != 100:
        raise RuntimeError(f"{event_id}: result future_frame_count is not 100")
    invalid = list(audit.get("invalid_observation_audit", []))
    repairs = sum(item.get("status") == "REPAIRED_BOX_FROM_OFFICIAL_NONEMPTY_MASK" for item in invalid)
    absences = sum(item.get("status") == "LEGITIMATELY_ABSENT_OFFICIAL_ZERO_AREA_EMPTY_MASK" for item in invalid)
    if any(item.get("status") not in {
        "REPAIRED_BOX_FROM_OFFICIAL_NONEMPTY_MASK",
        "LEGITIMATELY_ABSENT_OFFICIAL_ZERO_AREA_EMPTY_MASK",
    } for item in invalid):
        raise RuntimeError(f"{event_id}: unknown invalid observation classification")
    return {
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "action_type": str(event.get("action_type", "UNKNOWN")),
        "event_frame": int(event["event_frame"]),
        "trigger_frame": trigger,
        "end_frame": end,
        "target_public_id": int(result["target_public_id"]),
        "future_frame_count": len(coverage),
        "future_candidate_count": len(future),
        "future_nonempty_frame_count": sum(int(item.get("candidate_count", 0)) > 0 for item in coverage),
        "invalid_official_observation_count": len(invalid),
        "deterministic_mask_box_repair_count": int(repairs),
        "legitimate_official_absence_count": int(absences),
        "source": "FUTURE_FRAME_REQUERY",
        "artifact_dir": str(event_dir),
        "result_sha256": sha256_file(event_dir / "result.json"),
        "candidates_sha256": sha256_file(event_dir / "candidates.json"),
        "audit_sha256": sha256_file(event_dir / "audit.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--retry-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    events = frozen_events()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    selected_roots = [args.attempt_root, *args.retry_root]
    for event in events:
        event_id = str(event["event_id"])
        locations = [root / "events" / event_id for root in selected_roots]
        complete = [location for location in locations if (location / "done.json").is_file()]
        if len(complete) != 1:
            failures.append({"event_id": event_id, "reason": "expected exactly one completed artifact location", "locations": [str(item) for item in complete]})
            continue
        try:
            rows.append(audit_event(event, complete[0]))
        except Exception as exc:
            failures.append({"event_id": event_id, "reason": type(exc).__name__, "error": str(exc), "artifact_dir": str(complete[0])})
    by_action = Counter(row["action_type"] for row in rows)
    by_sequence = Counter(row["sequence"] for row in rows)
    summary = {
        "schema_version": "N72R10_TRUE_FUTURE_REQUERY_BATCH_AUDIT_V1",
        "status": "PASS_N72R10_TRUE_FUTURE_REQUERY_BATCH_AUDIT" if len(rows) == 32 and not failures else "FAIL_N72R10_TRUE_FUTURE_REQUERY_BATCH_AUDIT",
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "expected_event_count": 32,
        "audited_event_count": len(rows),
        "failed_event_count": len(failures),
        "unique_event_count": len({row["event_id"] for row in rows}),
        "unique_sequence_count": len(by_sequence),
        "action_counts": dict(sorted(by_action.items())),
        "sequence_counts": dict(sorted(by_sequence.items())),
        "future_frame_count_total": sum(row["future_frame_count"] for row in rows),
        "future_candidate_count_total": sum(row["future_candidate_count"] for row in rows),
        "future_nonempty_frame_count_total": sum(row["future_nonempty_frame_count"] for row in rows),
        "deterministic_mask_box_repair_count_total": sum(row["deterministic_mask_box_repair_count"] for row in rows),
        "legitimate_official_absence_count_total": sum(row["legitimate_official_absence_count"] for row in rows),
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "event_rows": rows,
        "failures": failures,
        "audited_at_utc": now_utc(),
    }
    atomic_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
