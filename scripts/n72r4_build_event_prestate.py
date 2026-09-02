#!/usr/bin/env python3
"""Build auditable t-1 persistent substates for the frozen N72R3 events.

The source is the already passing N72R3 Stage18 persistent candidate-runtime
contract.  This script runs that runtime only through ``event_frame - 1``;
it never opens GT, creates a simulated command, or runs an effect replay.
"""

from __future__ import annotations

import hashlib
import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.identity.persistent_snapshot import PersistentRuntimeSnapshot  # noqa: E402
from sam3_intermot.identity.persistent_runtime import SequencePersistentIdentityRuntime  # noqa: E402
from scripts.n72r3_stage09_11_candidate_runtime import load_plan, load_source_rows, run_session  # noqa: E402


FROZEN_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R3/worktree/outputs/N72R3")
EVENT_MANIFEST = FROZEN_ROOT / "simulation/real_event_manifest.json"
OFFICIAL_ROOT = FROZEN_ROOT / "official_correction/events"
OUT = ROOT / "outputs/N72R4"
PRESTATE_ROOT = OUT / "event_prestate"
PRESTATE_ROOT = Path(os.environ.get("N72R4_PRESTATE_ROOT", str(PRESTATE_ROOT)))
MANIFEST_PATH = Path(os.environ.get("N72R4_PRESTATE_MANIFEST", str(OUT / "event_prestate_manifest.json")))
STAGE6_PATH = Path(os.environ.get("N72R4_STAGE06_PATH", str(OUT / "stage_status/stage_06_status.json")))
STAGE7_PATH = Path(os.environ.get("N72R4_STAGE07_PATH", str(OUT / "stage_status/stage_07_status.json")))
STAGE6_TOP_PATH = Path(os.environ.get("N72R4_STAGE06_TOP_PATH", str(OUT / "stage_06_status.json")))
STAGE7_TOP_PATH = Path(os.environ.get("N72R4_STAGE07_TOP_PATH", str(OUT / "stage_07_status.json")))
FAILURE_ROOT = OUT / "attempts"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Serialize the deep snapshot without changing the in-memory runtime."""

    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Enum):
        return json_safe(value.value)
    if hasattr(value, "tolist") and callable(value.tolist):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {"__type__": f"{value.__class__.__module__}.{value.__class__.__name__}", **json_safe(vars(value))}
    raise TypeError(f"snapshot value is not JSON serializable: {type(value).__name__}")


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_events() -> list[dict[str, Any]]:
    manifest = read_json(EVENT_MANIFEST)
    if manifest.get("status") != "PASS_STAGE14_POLICY_FROZEN":
        raise RuntimeError("event prestate requires frozen N72R3 Stage14 policy")
    events = [dict(item) for item in manifest.get("events", [])]
    if len(events) != 6:
        raise RuntimeError(f"event prestate requires exactly six frozen events, found {len(events)}")
    for event in events:
        if event.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"event manifest permits runtime GT: {event.get('event_id')}")
        if event.get("interaction_source") != "simulated_from_gt":
            raise RuntimeError(f"unexpected interaction source in frozen event: {event.get('event_id')}")
    return sorted(events, key=lambda item: str(item["event_id"]))


def find_window(event: dict[str, Any], windows: list[dict[str, Any]]) -> dict[str, Any]:
    window_id = str(event["current_candidate_v2"]["window_id"])
    matches = [item for item in windows if str(item["window_id"]) == window_id]
    if len(matches) != 1:
        raise RuntimeError(f"candidate window is not uniquely frozen: {event['event_id']}/{window_id}")
    return dict(matches[0])


def axis_from_payload(payload: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    records = list(payload.get("identities", []))
    if not records:
        raise RuntimeError("persistent prestate contains no identity records")
    state_axis = [int(item["association_state_id"]) for item in records]
    public_axis = [int(item["public_id"]) for item in records]
    lineage_axis = [int(item["identity_lineage_id"]) for item in records]
    if len(state_axis) != len(set(state_axis)):
        raise RuntimeError("prestate association_state_id axis is not unique")
    if len(public_axis) != len(set(public_axis)):
        raise RuntimeError("prestate public_id axis is not unique")
    if len(lineage_axis) != len(set(lineage_axis)):
        raise RuntimeError("prestate lineage axis is not unique")
    for item in records:
        if int(item["public_id"]) != int(item["mot_track_id"]):
            raise RuntimeError("prestate violates public_id == mot_track_id")
        if item.get("last_seen_frame") is not None and int(item["last_seen_frame"]) > int(payload["snapshot_frame"]):
            raise RuntimeError("prestate identity points into the future")
        if item.get("current_session_id") is None:
            raise RuntimeError("prestate lost active SAM session binding unexpectedly")
        if item.get("appearance_state", {}).get("last_machine_feature_sha256") is None:
            raise RuntimeError("prestate lacks last machine feature digest")
        if item.get("motion_state_ref") is None:
            raise RuntimeError("prestate lacks motion state reference")
    return state_axis, public_axis, lineage_axis


def build_one(event: dict[str, Any], windows: list[dict[str, Any]]) -> dict[str, Any]:
    event_id = str(event["event_id"])
    sequence = str(event["sequence"])
    event_frame = int(event["event_frame"])
    prestate_frame = event_frame - 1
    if prestate_frame < 0:
        raise RuntimeError(f"event has no valid t-1 prestate: {event_id}")
    window = find_window(event, windows)
    if int(window["frame_start"]) > prestate_frame or int(window["frame_end"]) < event_frame:
        raise RuntimeError(f"frozen candidate window does not bracket t-1/t: {event_id}")
    event_dir = PRESTATE_ROOT / event_id
    if event_dir.exists() and any(event_dir.iterdir()):
        raise RuntimeError(f"prestate event directory is not empty: {event_dir}")
    event_dir.mkdir(parents=True, exist_ok=True)

    # This is the same public persistent runtime implementation used by the
    # passing N72R3 Stage18 baseline.  It consumes candidate rows only through
    # t-1, and run_session has an audited no-future-GT contract.
    rows, source_metadata = load_source_rows(window, end=prestate_frame)
    if not rows:
        raise RuntimeError(f"no candidate rows available before event: {event_id}")
    runtime = SequencePersistentIdentityRuntime(sequence, public_id_start=1000)
    prefix_output = event_dir / "runtime_prefix"
    prefix_result = run_session(
        runtime,
        rows,
        session_label=f"prestate:{event_id}",
        output_dir=prefix_output,
    )
    if int(prefix_result["frame_end"]) != prestate_frame:
        raise RuntimeError(f"runtime prefix ended at the wrong frame: {event_id}/{prefix_result['frame_end']}")
    if prefix_result["runtime_audit"].get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"runtime prefix used future GT: {event_id}")
    snapshot = PersistentRuntimeSnapshot.capture(
        runtime,
        snapshot_frame=prestate_frame,
        next_window_start=event_frame,
    )
    payload = snapshot.payload
    state_axis, public_axis, lineage_axis = axis_from_payload({**payload, "snapshot_frame": prestate_frame})
    official = read_json(OFFICIAL_ROOT / f"{event_id}.json")
    target_public_id = int(official["persistent_identity"]["public_id"])
    if target_public_id not in public_axis:
        raise RuntimeError(f"event target public ID is not present in t-1 persistent prestate: {event_id}/{target_public_id}")
    target_state = int(payload["public_to_state"][str(target_public_id)])
    if target_state not in state_axis:
        raise RuntimeError(f"public-to-state reverse binding failed: {event_id}/{target_public_id}")

    snapshot_path = event_dir / "persistent_runtime_snapshot.json"
    snapshot_document = {
        **snapshot.as_dict(),
        "payload": json_safe(payload),
        "runtime_future_gt_used": False,
        "source_stage18_baseline": str(FROZEN_ROOT / "baseline/stage18_persistent_public"),
        "source_candidate_sha256": source_metadata["candidate_sha256"],
        "source_candidate_frame_sha256": source_metadata["candidate_frame_sha256"],
        "scientific_result": "STRUCTURAL_PRESTATE_ONLY",
    }
    atomic_json(snapshot_path, snapshot_document)
    identity_axis_path = event_dir / "identity_axis.json"
    atomic_json(
        identity_axis_path,
        {
            "schema_version": "N72R4_EVENT_PRESTATE_IDENTITY_AXIS_V1",
            "event_id": event_id,
            "sequence": sequence,
            "prestate_frame": prestate_frame,
            "association_state_axis": state_axis,
            "public_id_axis": public_axis,
            "lineage_axis": lineage_axis,
            "public_id_from_persistent_record": True,
            "candidate_index_to_public_id": False,
            "native_id_to_public_id": False,
            "runtime_future_gt_used": False,
        },
    )
    manager_path = event_dir / "state_manager_snapshot.json"
    atomic_json(
        manager_path,
        {
            "schema_version": "N72R4_EVENT_PRESTATE_STATE_MANAGER_V1",
            "event_id": event_id,
            "sequence": sequence,
            "prestate_frame": prestate_frame,
            "track_manager": json_safe(payload.get("track_manager")),
            "active_session_id": payload.get("active_session_id"),
            "runtime_future_gt_used": False,
        },
    )
    manifest_path = event_dir / "manifest.json"
    event_manifest = {
        "schema_version": "N72R4_EVENT_PRESTATE_MANIFEST_V1",
        "status": "PASS_EVENT_PRESTATE",
        "event_id": event_id,
        "sequence": sequence,
        "action_type": str(event["action_type"]),
        "event_frame": event_frame,
        "prestate_frame": prestate_frame,
        "next_window_start": event_frame,
        "target_public_id": target_public_id,
        "target_association_state_id": target_state,
        "snapshot": str(snapshot_path),
        "snapshot_sha256": sha256(snapshot_path),
        "identity_axis": str(identity_axis_path),
        "identity_axis_sha256": sha256(identity_axis_path),
        "state_manager_snapshot": str(manager_path),
        "state_manager_snapshot_sha256": sha256(manager_path),
        "runtime_prefix": str(prefix_output),
        "runtime_prefix_frame_start": int(prefix_result["frame_start"]),
        "runtime_prefix_frame_end": int(prefix_result["frame_end"]),
        "runtime_prefix_frame_count": int(prefix_result["frame_count"]),
        "runtime_prefix_candidate_row_count": int(prefix_result["candidate_row_count"]),
        "source_candidate_sha256": source_metadata["candidate_sha256"],
        "source_candidate_frame_sha256": source_metadata["candidate_frame_sha256"],
        "public_id_axis": public_axis,
        "association_state_axis": state_axis,
        "lineage_axis": lineage_axis,
        "allocator_state_present": "public_allocator" in payload and "next_state_id" in payload,
        "appearance_memory_state_present": "appearance_memory" in payload,
        "motion_state_present": all(item.get("motion_state_ref") is not None for item in payload.get("identities", [])),
        "active_sam_binding_present": all(item.get("current_session_id") is not None for item in payload.get("identities", [])),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "scientific_result": "STRUCTURAL_PRESTATE_ONLY",
    }
    atomic_json(manifest_path, event_manifest)
    return event_manifest


def write_stage_status(completed: list[dict[str, Any]], stage_path: Path, top_path: Path, stage: str, status: str, **extra: Any) -> None:
    payload = {
        "schema_version": "N72R4_STAGE_STATUS_V1",
        "stage": stage,
        "status": status,
        "created_at_utc": now_utc(),
        "event_count": len(completed),
        "independent_sequence_count": len({item["sequence"] for item in completed}),
        "completed_events": completed,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "scientific_result": "STRUCTURAL_PRESTATE_ONLY",
        **extra,
    }
    atomic_json(stage_path, payload)
    atomic_json(top_path, payload)


def stage7_static_audit() -> dict[str, Any]:
    formal_paths = [
        ROOT / "scripts/n72r4_persistent_effect_replay.py",
    ]
    existing = [path for path in formal_paths if path.is_file()]
    forbidden = ["1000 + index", "1000+index", "1000 + candidate_index", "candidate_index_to_public_id ="]
    hits: list[dict[str, Any]] = []
    for path in existing:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern in text:
                hits.append({"file": str(path), "pattern": pattern})
    audit_status = (
        "PASS_STAGE07_PERSISTENT_AXIS_SOURCE"
        if existing and not hits
        else "PENDING_STAGE07_REPLAY_PATH_NOT_YET_CREATED"
        if not existing
        else "FAIL_STAGE07_CANDIDATE_INDEX_PUBLIC_MAPPING"
    )
    return {
        "schema_version": "N72R4_STAGE07_PERSISTENT_AXIS_AUDIT_V1",
        "status": audit_status,
        "formal_path_files_checked": [str(path) for path in existing],
        "forbidden_candidate_public_patterns": forbidden,
        "hits": hits,
        "public_axis_source": "PersistentIdentityRecord.public_id",
        "association_axis_source": "PersistentIdentityRecord.association_state_id",
        "candidate_index_to_public_id": False,
        "native_id_to_public_id": False,
        "runtime_future_gt_used": False,
        "scientific_result": "STRUCTURAL_AXIS_AUDIT_ONLY",
    }


def write_failure(exc: BaseException) -> Path:
    FAILURE_ROOT.mkdir(parents=True, exist_ok=True)
    existing = sorted(FAILURE_ROOT.glob("stage06_07_prestate_failure_attempt*.json"))
    path = FAILURE_ROOT / f"stage06_07_prestate_failure_attempt{len(existing) + 1}.json"
    atomic_json(
        path,
        {
            "schema_version": "N72R4_FAILURE_RECORD_V1",
            "stage": "06_07_EVENT_PRESTATE_AND_PERSISTENT_AXIS",
            "status": "FAIL_PRESERVED",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
        },
    )
    return path


def main() -> int:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--event-id", help="run one frozen event as a targeted smoke")
        args = parser.parse_args()
        windows = load_plan()
        events = load_events()
        if args.event_id:
            events = [event for event in events if str(event["event_id"]) == str(args.event_id)]
            if not events:
                raise RuntimeError(f"unknown frozen event for targeted smoke: {args.event_id}")
        completed: list[dict[str, Any]] = []
        for event in events:
            completed.append(build_one(event, windows))
            atomic_json(
                MANIFEST_PATH,
                {
                    "schema_version": "N72R4_EVENT_PRESTATE_ROOT_MANIFEST_V1",
                    "status": "IN_PROGRESS" if len(completed) < len(events) else "PASS_EVENT_PRESTATE_SET",
                    "created_at_utc": now_utc(),
                    "expected_event_count": len(events),
                    "completed_event_count": len(completed),
                    "completed": completed,
                    "runtime_future_gt_used": False,
                    "interaction_source": "simulated_from_gt",
                    "real_human_tape": False,
                    "execution_scope": "targeted_smoke" if args.event_id else "full_frozen_event_set",
                },
            )
            print(json.dumps({"events_completed": len(completed), "events_total": len(events)}, sort_keys=True), flush=True)
        stage6_extra = {
            "prestate_manifest": str(MANIFEST_PATH),
            "snapshot_frame_rule": "event_frame_minus_one",
            "all_target_public_ids_in_prestate": all(item["target_public_id"] in item["public_id_axis"] for item in completed),
            "allocator_state_present_all": all(item["allocator_state_present"] for item in completed),
            "appearance_memory_state_present_all": all(item["appearance_memory_state_present"] for item in completed),
            "motion_state_present_all": all(item["motion_state_present"] for item in completed),
            "active_sam_binding_present_all": all(item["active_sam_binding_present"] for item in completed),
            "execution_scope": "targeted_smoke" if args.event_id else "full_frozen_event_set",
        }
        write_stage_status(completed, STAGE6_PATH, STAGE6_TOP_PATH, "06_EVENT_PRESTATE_SNAPSHOT", "PASS_STAGE06_EVENT_PRESTATE", **stage6_extra)
        stage7 = stage7_static_audit()
        write_stage_status(
            completed,
            STAGE7_PATH,
            STAGE7_TOP_PATH,
            "07_REMOVE_EVENT_LOCAL_PUBLIC_AXIS",
            stage7["status"],
            static_audit=stage7,
            prestate_manifest=str(MANIFEST_PATH),
        )
        print(json.dumps({"status": "PASS_STAGE06_07", "stage07": stage7["status"], "prestate_manifest": str(MANIFEST_PATH)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = write_failure(exc)
        print(json.dumps({"status": "FAIL_STAGE06_07", "failure": str(failure)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
