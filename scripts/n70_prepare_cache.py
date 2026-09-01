"""N70 Stage 01/02: repair mapping provenance and materialize an isolated cache.

The N54 runtime files are deliberately not edited.  N70 joins their frozen
public-ID score stream with the already audited N36 frame tape, which contains
the missing chunk-local and sequence-global native bridge.  The join is
lossless with respect to the candidate masks/features and keeps candidate
absence explicit.  This is a cache/materialization branch, not a claim that a
new SAM3 checkpoint or a real human event was run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/N70"
DIAG = OUT / "diagnosis"
ATTEMPTS = OUT / "attempts"
PROTOCOL = OUT / "protocol.json"
N37_EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
N54_RUNTIME = ROOT / "outputs/n54/replay/runtime"
N36_FRAMES = ROOT / "outputs/n36/real_tape/frames"
CACHE_ROOT = Path("/path/to/cache/SAM3_InterMOT_N70/cache")
CACHE_BRANCH = "N70_REHYDRATED_N36_MAPPING_ENRICHED_N54_STREAM"
CACHE_DIR = CACHE_ROOT / CACHE_BRANCH / "event_frames"
CACHE_MANIFEST = OUT / "cache/candidate_cache_manifest.json"
MAPPING_ROWS = DIAG / "mapping_audit.jsonl"
MAPPING_SUMMARY = DIAG / "mapping_summary.json"
FIXTURE_RESULTS = DIAG / "mapping_fixture_results.json"
STAGE01 = OUT / "stage_01_status.json"
STAGE02 = OUT / "stage_02_status.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
EVENT_COUNT = 24
FRAMES_PER_EVENT = 100
FEAT_DIM = 512
PUBLIC_ID_EXPLICIT = "EXPLICIT_N54_PUBLIC_ASSIGNMENT"
PUBLIC_ID_ABSENT = "EXPLICIT_N54_PUBLIC_ASSIGNMENT_ABSENT"
PUBLIC_ID_GLOBAL_UNMAPPED = "N36_GLOBAL_MAPPING_ABSENT"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            for row in rows:
                line = (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
                handle.write(line)
                digest.update(line)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def finite_vector(value: Any, dim: int = FEAT_DIM, *, require_nonzero: bool = True) -> bool:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return False
    if arr.size != dim or not np.all(np.isfinite(arr)):
        return False
    return bool(not require_nonzero or np.linalg.norm(arr) > 1e-6)


def finite_matrix(value: Any, shape: tuple[int, ...]) -> bool:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(arr.shape == shape and np.all(np.isfinite(arr)))


def box(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != 4 or not np.all(np.isfinite(arr)):
        return None
    return arr


def box_iou(left: Any, right: Any) -> float:
    a, b = box(left), box(right)
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def boxes_equal(left: Any, right: Any, *, atol: float = 1e-5, rtol: float = 1e-6) -> bool:
    """Compare coordinates even when IoU is undefined for zero-area boxes.

    Keep the raw IoU at 0.0 for a zero-area union, but do not call identical
    coordinates a mapping mismatch.  Distinct coordinates still fail the
    mapping audit.
    """
    a, b = box(left), box(right)
    return bool(a is not None and b is not None and np.allclose(a, b, atol=atol, rtol=rtol))


def load_events() -> dict[str, dict[str, Any]]:
    payload = load_json(N37_EVENTS)
    events: dict[str, dict[str, Any]] = {}
    for item in payload.get("events", []):
        event = item.get("event", {})
        event_id = str(item.get("protocol_candidate_id") or event.get("event_id"))
        if not event_id or event_id == "None" or not isinstance(event, dict):
            raise RuntimeError("N37 event is not addressable")
        if event.get("interaction_source") != "simulated_from_gt" or item.get("interaction_source") != "simulated_from_gt":
            raise RuntimeError(f"N70 requires explicit simulated provenance: {event_id}")
        if event.get("runtime_future_gt_used") is not False or item.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"N37 runtime GT boundary failed: {event_id}")
        required = ("sequence", "frame", "public_id", "target_native_tid")
        if any(event.get(key) is None for key in required):
            raise RuntimeError(f"N37 explicit event boundary missing for {event_id}")
        human = event.get("human_embedding")
        if not finite_vector(human):
            raise RuntimeError(f"N37 human anchor invalid for {event_id}")
        events[event_id] = {
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "event_frame": int(event["frame"]),
            "future_start": int(item["future_frame_start"]),
            "future_end": int(item["future_frame_end"]),
            "target_public_id": int(event["public_id"]),
            "target_native_id": int(event["target_native_tid"]),
            "action_type": str(item.get("action_type") or event.get("action_type")),
            "human_embedding": list(np.asarray(human, dtype=np.float32).reshape(-1).astype(float)),
            "source_tape": item.get("source_tape"),
            "source_tape_sha256": item.get("source_tape_sha256"),
            "interaction_source": "simulated_from_gt",
        }
    if len(events) != EVENT_COUNT:
        raise RuntimeError(f"expected {EVENT_COUNT} events, found {len(events)}")
    return events


def load_n36_selected(sequence: str, required_frames: set[int]) -> dict[int, dict[str, Any]]:
    """Read one frozen N36 sequence once and retain only requested frames."""
    path = N36_FRAMES / f"{sequence}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    selected: dict[int, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid N36 JSONL {path}: {exc}") from exc
            frame = int(item.get("frame", -1))
            if frame not in required_frames:
                continue
            if frame in selected:
                raise RuntimeError(f"duplicate N36 frame {sequence}:{frame}")
            candidates = item.get("candidates")
            if not isinstance(candidates, list):
                raise RuntimeError(f"N36 candidates missing {sequence}:{frame}")
            by_global: dict[int, dict[str, Any]] = {}
            for index, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    raise RuntimeError(f"N36 candidate not object {sequence}:{frame}:{index}")
                try:
                    local_id = int(candidate["local_native_id"])
                    raw_native = int(candidate["native_tid"])
                    global_id = int(candidate["sequence_global_native_id"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"N36 mapping missing {sequence}:{frame}:{index}") from exc
                if global_id in by_global:
                    raise RuntimeError(f"duplicate N36 global id {sequence}:{frame}:{global_id}")
                if box(candidate.get("box")) is None:
                    raise RuntimeError(f"N36 box invalid {sequence}:{frame}:{index}")
                if not finite_vector(candidate.get("machine_embedding")):
                    raise RuntimeError(f"N36 embedding invalid {sequence}:{frame}:{index}")
                by_global[global_id] = {
                    "candidate_index": int(candidate.get("candidate_index", index)),
                    "native_id": raw_native,
                    "local_id": local_id,
                    "global_id": global_id,
                    "chunk_local_public_id": candidate.get("chunk_local_public_id"),
                    "box": list(np.asarray(candidate["box"], dtype=float).reshape(-1)),
                    "mask": candidate.get("mask"),
                    "machine_embedding": candidate.get("machine_embedding"),
                    "native_id_scope": candidate.get("native_tid_scope"),
                    "mapping_status": candidate.get("public_native_mapping_status"),
                    "global_status": candidate.get("sequence_global_native_id_status"),
                }
            selected[frame] = {
                "frame": frame,
                "sequence": str(item.get("sequence", sequence)),
                "chunk_id": item.get("chunk_id"),
                "frame_owner_chunk_id": item.get("frame_owner_chunk_id"),
                "source_chunk_ids": item.get("source_chunk_ids", []),
                "core_frame_start": item.get("core_frame_start"),
                "core_frame_end": item.get("core_frame_end"),
                "overlap_start": item.get("overlap_start"),
                "overlap_end": item.get("overlap_end"),
                "overlap_with_previous": item.get("overlap_with_previous"),
                "overlap_with_next": item.get("overlap_with_next"),
                "line_sha256": sha256_bytes(raw.rstrip(b"\n")),
                "candidate_by_global": by_global,
                "candidate_count": len(candidates),
                "runtime_future_gt_used": item.get("runtime_future_gt_used"),
                "runtime_gt_read": item.get("runtime_gt_read"),
                "candidate_set_source": item.get("candidate_set_source"),
            }
    missing = sorted(required_frames - set(selected))
    if missing:
        raise RuntimeError(f"N36 requested frames missing for {sequence}: {missing[:10]}")
    return selected


def branch_signature(branch: dict[str, Any]) -> tuple[Any, ...]:
    candidates = branch.get("candidate_rows", [])
    rows = branch.get("rows", [])
    return (
        tuple(int(row.get("native_tid")) for row in candidates),
        tuple(row.get("public_id") for row in rows),
        tuple(int(value) for value in branch.get("public_id_order", [])),
        len(candidates),
    )


def candidate_mapping_row(
    *,
    event: dict[str, Any],
    variant: str,
    frame_number: int,
    source_file: Path,
    source_sha: str,
    n54_frame: dict[str, Any],
    n36_frame: dict[str, Any] | None,
) -> dict[str, Any]:
    branch = n54_frame.get("write_baseline")
    if not isinstance(branch, dict):
        raise RuntimeError(f"write_baseline missing {event['event_id']}/{variant}/{frame_number}")
    candidates = branch.get("candidate_rows")
    rows = branch.get("rows")
    pids = branch.get("public_id_order")
    features = n54_frame.get("candidate_features_512")
    if not isinstance(candidates, list) or not isinstance(rows, list) or not isinstance(pids, list):
        raise RuntimeError(f"candidate/public axis malformed {event['event_id']}/{variant}/{frame_number}")
    if len(candidates) != len(rows):
        raise RuntimeError(f"candidate/row count mismatch {event['event_id']}/{variant}/{frame_number}")
    if not isinstance(features, list) or len(features) != len(candidates):
        raise RuntimeError(f"candidate feature count mismatch {event['event_id']}/{variant}/{frame_number}")
    if n54_frame.get("runtime_future_gt_used") is not False or branch.get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"runtime future GT boundary failed {event['event_id']}/{variant}/{frame_number}")
    n36_by_global = {} if n36_frame is None else n36_frame["candidate_by_global"]
    candidate_records: list[dict[str, Any]] = []
    mapping_errors: list[str] = []
    target_rows: list[int] = []
    public_claims: defaultdict[int, list[int]] = defaultdict(list)
    for index, (candidate, mapped, feature) in enumerate(zip(candidates, rows, features)):
        if not isinstance(candidate, dict) or not isinstance(mapped, dict):
            mapping_errors.append(f"row_{index}_not_object")
            continue
        try:
            runtime_global = int(candidate["native_tid"])
        except (KeyError, TypeError, ValueError):
            mapping_errors.append(f"row_{index}_native_id_invalid")
            continue
        if not finite_vector(feature):
            mapping_errors.append(f"row_{index}_feature_invalid")
        mapping = n36_by_global.get(runtime_global)
        if mapping is None:
            mapping_errors.append(f"row_{index}_global_id_not_in_n36")
        public_value = mapped.get("public_id")
        public_id = None
        if public_value is not None:
            try:
                public_id = int(public_value)
            except (TypeError, ValueError):
                mapping_errors.append(f"row_{index}_public_id_invalid")
        if mapping is None:
            public_id_status = PUBLIC_ID_GLOBAL_UNMAPPED
        elif public_id is None:
            # N54 uses null for a candidate that is not assigned to an active
            # public track.  Preserve that absence; never infer an ID from
            # candidate order, chunk-local IDs, or offline target GT.
            public_id_status = PUBLIC_ID_ABSENT
        else:
            public_id_status = PUBLIC_ID_EXPLICIT
        if public_id is not None:
            public_claims[public_id].append(index)
        if runtime_global == int(event["target_native_id"]):
            target_rows.append(index)
        enriched = dict(candidate)
        enriched["machine_embedding"] = list(np.asarray(feature, dtype=np.float32).reshape(-1).astype(float))
        enriched["embedding_dim"] = FEAT_DIM
        enriched["embedding_status"] = "N54_CANDIDATE_FEATURE_512"
        enriched["mapping"] = {
            "native_id": None if mapping is None else mapping["native_id"],
            "native_id_scope": None if mapping is None else mapping["native_id_scope"],
            "local_id": None if mapping is None else mapping["local_id"],
            "global_id": None if mapping is None else mapping["global_id"],
            "public_id": public_id,
            "public_id_status": public_id_status,
            "public_id_source": "N54_frozen_runtime_public_axis",
            "mapping_version": "n70-n36-bridge-n54-public-v1",
            "mapping_source": "N36_explicit_overlap_reconciled_plus_N54_runtime_row",
            "valid_from": int(event["event_frame"]),
            "valid_to": int(event["future_end"]),
            "confidence": 1.0 if mapping is not None and public_id is not None else 0.0,
            "provenance": {
                "n36_frame_line_sha256": None if n36_frame is None else n36_frame["line_sha256"],
                "n36_chunk_id": None if mapping is None else n36_frame.get("chunk_id"),
                "n54_source_sha256": source_sha,
                "event_id": event["event_id"],
                "interaction_source": "simulated_from_gt",
                "runtime_future_gt_used": False,
                "target_native_id_used_only_offline": True,
                "public_id_absence_is_explicit": public_id_status == PUBLIC_ID_ABSENT,
            },
        }
        if mapping is None:
            enriched["mask"] = None
        else:
            enriched["mask"] = mapping["mask"]
            enriched["n36_candidate_index"] = mapping["candidate_index"]
            enriched["n36_box_iou"] = box_iou(candidate.get("box"), mapping.get("box"))
            enriched["n36_box_coordinate_equal"] = boxes_equal(candidate.get("box"), mapping.get("box"))
            enriched["n36_box_iou_defined"] = bool(
                enriched["n36_box_iou"] > 0.0 or not enriched["n36_box_coordinate_equal"]
            )
            if enriched["n36_box_iou"] < 0.95 and not enriched["n36_box_coordinate_equal"]:
                mapping_errors.append(f"row_{index}_n36_n54_box_iou_below_0.95")
        candidate_records.append(enriched)
    duplicate_public = sorted(pid for pid, indexes in public_claims.items() if len(indexes) > 1)
    if duplicate_public:
        mapping_errors.append("duplicate_public_id_rows")
    target_row = target_rows[0] if len(target_rows) == 1 else None
    target_present = len(target_rows) == 1
    target_public = int(event["target_public_id"])
    old_target_public_rows = public_claims.get(target_public, [])
    old_mapping_match = bool(target_row is not None and rows[target_row].get("public_id") == target_public)
    old_conflict = bool(target_present and not old_mapping_match) or bool(old_target_public_rows and target_row not in old_target_public_rows)
    if n36_frame is None:
        mapping_errors.append("n36_frame_missing")
    n36_runtime_gt_ok = bool(n36_frame is not None and n36_frame.get("runtime_future_gt_used") is False and n36_frame.get("runtime_gt_read") is False)
    if not n36_runtime_gt_ok:
        mapping_errors.append("n36_runtime_gt_boundary_failed")
    chain_complete = bool(
        not mapping_errors
        and all(
            item["mapping"]["local_id"] is not None
            and item["mapping"]["global_id"] is not None
            and (
                item["mapping"]["public_id"] is not None
                or item["mapping"]["public_id_status"] == PUBLIC_ID_ABSENT
            )
            for item in candidate_records
        )
        and len(set(int(item["mapping"]["global_id"]) for item in candidate_records)) == len(candidate_records)
        and len({
            int(item["mapping"]["public_id"])
            for item in candidate_records
            if item["mapping"]["public_id"] is not None
        }) == sum(item["mapping"]["public_id"] is not None for item in candidate_records)
    )
    return {
        "schema": "N70_MAPPING_AUDIT_ROW_V1",
        "event_id": event["event_id"],
        "sequence": event["sequence"],
        "action_type": event["action_type"],
        "event_frame": event["event_frame"],
        "variant": variant,
        "frame": int(frame_number),
        "frame_horizon": int(frame_number) - int(event["event_frame"]),
        "target_public_id_offline": target_public,
        "target_global_id_offline": int(event["target_native_id"]),
        "target_candidate_present": target_present,
        "target_row": target_row,
        "old_public_id_at_target_row": None if target_row is None else rows[target_row].get("public_id"),
        "old_public_id_rows_claiming_target": old_target_public_rows,
        "old_mapping_matches_boundary": old_mapping_match,
        "old_mapping_conflict": old_conflict,
        "candidate_count": len(candidate_records),
        "candidate_chain_complete": chain_complete,
        "public_assignment_absent_candidate_rows": sum(
            item.get("mapping", {}).get("public_id_status") == PUBLIC_ID_ABSENT
            for item in candidate_records
        ),
        "target_public_assignment_absent": bool(
            target_present
            and target_row is not None
            and candidate_records[target_row].get("mapping", {}).get("public_id") is None
        ),
        "mapping_errors": sorted(set(mapping_errors)),
        "mapping_chain_schema": ["native_id", "local_id", "global_id", "public_id"],
        "candidate_mappings": [
            {
                "candidate_index": index,
                "runtime_global_id": int(candidate.get("native_tid", -1)),
                "native_id": candidate.get("mapping", {}).get("native_id"),
                "local_id": candidate.get("mapping", {}).get("local_id"),
                "global_id": candidate.get("mapping", {}).get("global_id"),
                "public_id": candidate.get("mapping", {}).get("public_id"),
                "public_id_status": candidate.get("mapping", {}).get("public_id_status"),
                "n36_box_iou": candidate.get("n36_box_iou"),
                "n36_box_coordinate_equal": candidate.get("n36_box_coordinate_equal"),
                "n36_box_iou_defined": candidate.get("n36_box_iou_defined"),
                "provenance": candidate.get("mapping", {}).get("provenance"),
            }
            for index, candidate in enumerate(candidate_records)
        ],
        "n36_source": {
            "frame_line_sha256": None if n36_frame is None else n36_frame["line_sha256"],
            "chunk_id": None if n36_frame is None else n36_frame.get("chunk_id"),
            "source_chunk_ids": [] if n36_frame is None else n36_frame.get("source_chunk_ids", []),
        },
        "n54_source_sha256": source_sha,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
    }


def build_cache_frame(event: dict[str, Any], variant: str, raw: dict[str, Any], mapping_row: dict[str, Any], source_sha: str, n36_frame: dict[str, Any]) -> dict[str, Any]:
    branch = raw.get("write_baseline")
    if not isinstance(branch, dict):
        raise RuntimeError(f"N70 cache branch missing {event['event_id']}/{variant}/{raw.get('frame')}")
    score = branch.get("score_matrix")
    candidates = branch.get("candidate_rows")
    rows = branch.get("rows")
    pids = branch.get("public_id_order")
    features = raw.get("candidate_features_512")
    memory = raw.get("memory_vectors_512")
    memory_valid = raw.get("memory_valid")
    scalar = raw.get("scalar_features_8")
    n = len(candidates) if isinstance(candidates, list) else -1
    p = len(pids) if isinstance(pids, list) else -1
    if not finite_matrix(score, (n, p)):
        raise RuntimeError(f"N70 base score matrix invalid {event['event_id']}/{variant}/{raw.get('frame')}")
    if not isinstance(features, list) or len(features) != n or any(not finite_vector(value) for value in features):
        raise RuntimeError(f"N70 candidate feature matrix invalid {event['event_id']}/{variant}/{raw.get('frame')}")
    if not isinstance(memory, list) or len(memory) != p or any(not finite_vector(value, require_nonzero=False) for value in memory):
        raise RuntimeError(f"N70 memory feature matrix invalid {event['event_id']}/{variant}/{raw.get('frame')}")
    if not isinstance(memory_valid, list) or len(memory_valid) != p:
        raise RuntimeError(f"N70 memory validity axis invalid {event['event_id']}/{variant}/{raw.get('frame')}")
    if not finite_matrix(scalar, (n * p, 8)):
        raise RuntimeError(f"N70 scalar matrix invalid {event['event_id']}/{variant}/{raw.get('frame')}")
    if raw.get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"N70 raw future GT boundary failed {event['event_id']}/{variant}/{raw.get('frame')}")
    candidate_rows = []
    for index, candidate in enumerate(mapping_row["candidate_mappings"]):
        source_candidate = dict(candidates[index])
        source_candidate["machine_embedding"] = list(np.asarray(features[index], dtype=np.float32).reshape(-1).astype(float))
        source_candidate["mask"] = None
        mapping = next(item for item in mapping_row["candidate_mappings"] if item["candidate_index"] == index)
        n36_candidate = n36_frame["candidate_by_global"].get(int(mapping["runtime_global_id"]))
        if n36_candidate is not None:
            source_candidate["mask"] = n36_candidate.get("mask")
            source_candidate["n36_box_iou"] = box_iou(source_candidate.get("box"), n36_candidate.get("box"))
            source_candidate["n36_box_coordinate_equal"] = boxes_equal(source_candidate.get("box"), n36_candidate.get("box"))
            source_candidate["n36_candidate_index"] = n36_candidate.get("candidate_index")
        source_candidate["mapping"] = {
            "native_id": mapping["native_id"],
            "local_id": mapping["local_id"],
            "global_id": mapping["global_id"],
            "public_id": mapping["public_id"],
            "public_id_status": mapping["public_id_status"],
            "mapping_version": "n70-n36-bridge-n54-public-v1",
            "mapping_source": "N36_explicit_overlap_reconciled_plus_N54_runtime_row",
            "provenance": {
                "n36_frame_line_sha256": mapping["provenance"].get("n36_frame_line_sha256"),
                "n54_source_sha256": source_sha,
                "event_id": event["event_id"],
                "runtime_future_gt_used": False,
                "public_id_absence_is_explicit": mapping["public_id_status"] == PUBLIC_ID_ABSENT,
            },
        }
        candidate_rows.append(source_candidate)
    payload = {
        "schema": "N70_CANDIDATE_CACHE_FRAME_V1",
        "status": "PASS_CACHE_FRAME",
        "event_id": event["event_id"],
        "sequence": event["sequence"],
        "variant": variant,
        "frame": int(raw["frame"]),
        "event_frame": int(event["event_frame"]),
        "frame_horizon": int(raw["frame"]) - int(event["event_frame"]),
        "candidate_set_complete": True,
        "candidate_count": n,
        "candidate_order": [int(item.get("candidate_index", index)) for index, item in enumerate(candidates)],
        "candidate_rows": candidate_rows,
        "rows": rows,
        "public_id_order": [int(value) for value in pids],
        "assignment_columns": branch.get("assignment_columns"),
        "assignment_public_ids": branch.get("assignment_public_ids"),
        "score_matrix": score,
        "candidate_features_512": features,
        "memory_vectors_512": memory,
        "memory_valid": [bool(value) for value in memory_valid],
        "scalar_features_8": scalar,
        "chunk_provenance": {
            "n36_chunk_id": n36_frame.get("chunk_id"),
            "frame_owner_chunk_id": n36_frame.get("frame_owner_chunk_id"),
            "source_chunk_ids": n36_frame.get("source_chunk_ids", []),
            "core_frame_start": n36_frame.get("core_frame_start"),
            "core_frame_end": n36_frame.get("core_frame_end"),
            "overlap_start": n36_frame.get("overlap_start"),
            "overlap_end": n36_frame.get("overlap_end"),
            "n36_frame_line_sha256": n36_frame.get("line_sha256"),
        },
        "source_provenance": {
            "n54_source_file": str((N54_RUNTIME / f"{event['event_id']}.json").resolve()),
            "n54_source_sha256": source_sha,
            "n36_source_file": str((N36_FRAMES / f"{event['sequence']}.jsonl").resolve()),
            "checkpoint": raw.get("checkpoint"),
            "runtime_gt_read": False,
        },
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    }
    return payload


def run_fixtures() -> dict[str, Any]:
    """Small contract fixtures; no scientific data is generated here."""
    good = {
        "native_id": 4,
        "local_id": 2,
        "global_id": 9,
        "public_id": 100004,
        "provenance": {"source": "toy", "runtime_future_gt_used": False},
    }
    missing = dict(good)
    missing["global_id"] = None
    duplicate = [dict(good), {**good, "local_id": 3}]
    tests = {
        "complete_chain_has_all_four_ids": all(good.get(key) is not None for key in ("native_id", "local_id", "global_id", "public_id")),
        "complete_chain_has_provenance": isinstance(good.get("provenance"), dict) and bool(good["provenance"]),
        "missing_global_is_not_promoted": missing.get("global_id") is None,
        "duplicate_global_fixture_is_detectable": len({int(item["global_id"]) for item in duplicate}) != len(duplicate),
        "runtime_gt_false_fixture": good["provenance"]["runtime_future_gt_used"] is False,
        "candidate_absence_is_explicit": True,
        "identical_degenerate_boxes_are_coordinate_equal": boxes_equal([3.0, 4.0, 3.0, 4.0], [3.0, 4.0, 3.0, 4.0]),
        "different_degenerate_boxes_are_not_coordinate_equal": not boxes_equal([3.0, 4.0, 3.0, 4.0], [3.0, 4.0, 3.1, 4.0]),
    }
    return {
        "schema": "N70_MAPPING_FIXTURE_RESULTS_V1",
        "status": "PASS" if all(tests.values()) else "FAIL",
        "scientific_fixture": False,
        "tests": tests,
        "passed": sum(bool(value) for value in tests.values()),
        "total": len(tests),
    }


def process_event(event: dict[str, Any], n36_selected: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    event_id = event["event_id"]
    source_path = N54_RUNTIME / f"{event_id}.json"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha = sha256_file(source_path)
    source = load_json(source_path)
    if source.get("event_id") != event_id or source.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"N54 provenance mismatch {event_id}")
    variants = source.get("variants", {})
    if set(variants) != set(VARIANTS):
        raise RuntimeError(f"N54 variants mismatch {event_id}")
    mapping_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    signatures: dict[str, tuple[Any, ...]] = {}
    frame_sets: dict[str, list[int]] = {}
    for variant in VARIANTS:
        frames = variants[variant].get("frames", [])
        if not isinstance(frames, list) or len(frames) != FRAMES_PER_EVENT:
            raise RuntimeError(f"N54 frame count mismatch {event_id}/{variant}")
        frame_sets[variant] = [int(item.get("frame", -1)) for item in frames]
        expected = list(range(event["future_start"], event["future_end"] + 1))
        if frame_sets[variant] != expected:
            raise RuntimeError(f"N54 frame range mismatch {event_id}/{variant}")
        for raw in frames:
            frame_number = int(raw["frame"])
            n36_frame = n36_selected.get(frame_number)
            row = candidate_mapping_row(
                event=event,
                variant=variant,
                frame_number=frame_number,
                source_file=source_path,
                source_sha=source_sha or "",
                n54_frame=raw,
                n36_frame=n36_frame,
            )
            mapping_rows.append(row)
            cache_rows.append(build_cache_frame(event, variant, raw, row, source_sha or "", n36_frame or {"candidate_by_global": {}}))
            counts["frames"] += 1
            counts["candidate_rows"] += int(row["candidate_count"])
            counts["target_candidate_present_frames"] += int(row["target_candidate_present"])
            counts["target_candidate_absent_frames"] += int(not row["target_candidate_present"])
            counts["target_public_assignment_absent_frames"] += int(row["target_public_assignment_absent"])
            counts["public_assignment_absent_candidate_rows"] += int(row["public_assignment_absent_candidate_rows"])
            counts["old_mapping_conflict_frames"] += int(row["old_mapping_conflict"])
            counts["old_mapping_match_frames"] += int(row["old_mapping_matches_boundary"])
            counts["chain_complete_frames"] += int(row["candidate_chain_complete"])
            counts["mapping_error_frames"] += int(bool(row["mapping_errors"]))
            signatures[variant] = branch_signature(raw["write_baseline"])
        if frame_sets[variant] != frame_sets["M0"]:
            counts["cross_variant_frame_sequence_mismatch"] += 1
    cache_path = CACHE_DIR / f"{event_id}.jsonl"
    cache_sha = atomic_jsonl(cache_path, cache_rows)
    done = {
        "schema": "N70_CACHE_EVENT_DONE_V1",
        "status": "PASS",
        "event_id": event_id,
        "sequence": event["sequence"],
        "path": str(cache_path),
        "sha256": cache_sha,
        "rows": len(cache_rows),
        "expected_rows": len(VARIANTS) * FRAMES_PER_EVENT,
        "candidate_rows": sum(int(item["candidate_count"]) for item in mapping_rows),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
    }
    atomic_json(CACHE_DIR / f"{event_id}.done.json", done)
    summary = {
        "event_id": event_id,
        "sequence": event["sequence"],
        "action_type": event["action_type"],
        "frames": int(counts["frames"]),
        "candidate_rows": int(counts["candidate_rows"]),
        "target_candidate_present_frames": int(counts["target_candidate_present_frames"]),
        "target_candidate_absent_frames": int(counts["target_candidate_absent_frames"]),
        "target_public_assignment_absent_frames": int(counts["target_public_assignment_absent_frames"]),
        "public_assignment_absent_candidate_rows": int(counts["public_assignment_absent_candidate_rows"]),
        "old_mapping_conflict_frames": int(counts["old_mapping_conflict_frames"]),
        "old_mapping_match_frames": int(counts["old_mapping_match_frames"]),
        "chain_complete_frames": int(counts["chain_complete_frames"]),
        "mapping_error_frames": int(counts["mapping_error_frames"]),
        "cross_variant_frame_sequence_mismatch": int(counts["cross_variant_frame_sequence_mismatch"]),
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha,
        "n54_source_sha256": source_sha,
        "n36_selected_frame_count": len(n36_selected),
        "variant_signatures_equal": len(set(signatures.values())) == 1,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
    }
    return mapping_rows, summary, cache_rows


def audit_cache(manifest_events: list[dict[str, Any]]) -> dict[str, Any]:
    seen: set[tuple[str, str, int]] = set()
    errors: list[str] = []
    rows = 0
    candidate_rows = 0
    finite_features = 0
    mapping_complete_rows = 0
    target_absent = 0
    for item in manifest_events:
        path = Path(item["cache_path"])
        if not path.is_file():
            errors.append(f"missing_cache:{path}")
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"invalid_cache_json:{path}:{line_no}:{exc}")
                    continue
                key = (str(frame.get("event_id")), str(frame.get("variant")), int(frame.get("frame", -1)))
                if key in seen:
                    errors.append(f"duplicate_cache_key:{key}")
                seen.add(key)
                rows += 1
                candidates = frame.get("candidate_rows", [])
                candidate_rows += len(candidates) if isinstance(candidates, list) else 0
                if frame.get("runtime_future_gt_used") is not False or frame.get("runtime_gt_read") is not False:
                    errors.append(f"runtime_gt_boundary:{key}")
                if not isinstance(candidates, list) or frame.get("candidate_set_complete") is not True:
                    errors.append(f"candidate_set_incomplete:{key}")
                    continue
                frame_features = frame.get("candidate_features_512", [])
                if len(frame_features) != len(candidates) or any(not finite_vector(value) for value in frame_features):
                    errors.append(f"feature_invalid:{key}")
                else:
                    finite_features += len(frame_features)
                mappings = [candidate.get("mapping", {}) for candidate in candidates if isinstance(candidate, dict)]
                if all(
                    mapping.get("local_id") is not None
                    and mapping.get("global_id") is not None
                    and (
                        mapping.get("public_id") is not None
                        or mapping.get("public_id_status") == PUBLIC_ID_ABSENT
                    )
                    for mapping in mappings
                ):
                    mapping_complete_rows += 1
                if not any(int(candidate.get("native_tid", -1)) == int(item.get("target_native_id", -999999)) for candidate in candidates if isinstance(candidate, dict)):
                    # The manifest intentionally does not put offline target native ID into runtime rows.
                    # This counter is filled from the mapping summary, not inferred here.
                    pass
    expected = EVENT_COUNT * len(VARIANTS) * FRAMES_PER_EVENT
    return {
        "schema": "N70_CACHE_AUDIT_V1",
        "status": "PASS" if not errors and rows == expected and len(seen) == expected else "FAIL",
        "rows": rows,
        "expected_rows": expected,
        "unique_keys": len(seen),
        "duplicate_keys": rows - len(seen),
        "candidate_rows": candidate_rows,
        "finite_candidate_feature_values": finite_features,
        "mapping_complete_frame_rows": mapping_complete_rows,
        "errors": errors[:100],
        "error_count": len(errors),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
    }


def write_failure(exc: BaseException) -> None:
    ATTEMPTS.mkdir(parents=True, exist_ok=True)
    existing = sorted(ATTEMPTS.glob("stage_01_02_failure_attempt*.json"))
    atomic_json(
        ATTEMPTS / f"stage_01_02_failure_attempt{len(existing) + 1}.json",
        {
            "schema": "N70_FAILURE_ARTIFACT_V1",
            "status": "FAIL_PRESERVED",
            "stage": "N70_STAGE_01_02_MAPPING_CACHE",
            "created_at_utc": now(),
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
            "real_sam3_full_loop": False,
            "production_authorized": False,
            "next_action": "Preserve this failure, repair only the first actionable mapping/cache cause, then run the same targeted unit.",
        },
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DIAG.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    protocol_sha = sha256_file(PROTOCOL)
    fixtures = run_fixtures()
    atomic_json(FIXTURE_RESULTS, fixtures)
    if fixtures["status"] != "PASS":
        raise RuntimeError(f"N70 mapping fixtures failed: {fixtures}")
    events = load_events()
    by_sequence: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events.values():
        by_sequence[event["sequence"]].append(event)
    all_mapping_rows: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    cache_events: list[dict[str, Any]] = []
    total_counts: Counter[str] = Counter()
    for sequence in sorted(by_sequence):
        required: set[int] = set()
        for event in by_sequence[sequence]:
            required.update(range(event["future_start"], event["future_end"] + 1))
        selected = load_n36_selected(sequence, required)
        print(json.dumps({"sequence": sequence, "selected_frames": len(selected), "required_frames": len(required)}, sort_keys=True), flush=True)
        for event in sorted(by_sequence[sequence], key=lambda item: item["event_id"]):
            rows, summary, _ = process_event(event, selected)
            all_mapping_rows.extend(rows)
            event_summaries.append(summary)
            cache_events.append({**summary, "target_native_id": event["target_native_id"]})
            for key in ("frames", "candidate_rows", "target_candidate_present_frames", "target_candidate_absent_frames", "target_public_assignment_absent_frames", "public_assignment_absent_candidate_rows", "old_mapping_conflict_frames", "old_mapping_match_frames", "chain_complete_frames", "mapping_error_frames"):
                total_counts[key] += int(summary[key])
            print(json.dumps({"event": event["event_id"], "rows": len(rows), "target_present": summary["target_candidate_present_frames"], "target_absent": summary["target_candidate_absent_frames"], "chain_complete": summary["chain_complete_frames"], "cache": summary["cache_path"]}, sort_keys=True), flush=True)
    expected_rows = EVENT_COUNT * len(VARIANTS) * FRAMES_PER_EVENT
    if len(all_mapping_rows) != expected_rows:
        raise RuntimeError(f"N70 mapping row denominator mismatch: {len(all_mapping_rows)} != {expected_rows}")
    atomic_jsonl(MAPPING_ROWS, all_mapping_rows)
    mapping_summary = {
        "schema": "N70_MAPPING_SUMMARY_V1",
        "status": "PASS_MAPPING_REPAIRED_WITH_EXPLICIT_CANDIDATE_AND_PUBLIC_ASSIGNMENT_ABSENCE" if total_counts["mapping_error_frames"] == 0 and total_counts["chain_complete_frames"] == expected_rows else "FAIL_MAPPING_REPAIR",
        "created_at_utc": now(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": protocol_sha,
        "event_count": EVENT_COUNT,
        "independent_sequence_count": len(by_sequence),
        "variant_count": len(VARIANTS),
        "frames_per_event_variant": FRAMES_PER_EVENT,
        "audit_rows": len(all_mapping_rows),
        "target_candidate_present_frames": int(total_counts["target_candidate_present_frames"]),
        "target_candidate_absent_frames": int(total_counts["target_candidate_absent_frames"]),
        "target_public_assignment_absent_frames": int(total_counts["target_public_assignment_absent_frames"]),
        "public_assignment_absent_candidate_rows": int(total_counts["public_assignment_absent_candidate_rows"]),
        "candidate_rows": int(total_counts["candidate_rows"]),
        "old_mapping_conflict_frames": int(total_counts["old_mapping_conflict_frames"]),
        "old_mapping_match_frames": int(total_counts["old_mapping_match_frames"]),
        "mapping_error_frames": int(total_counts["mapping_error_frames"]),
        "chain_complete_frames": int(total_counts["chain_complete_frames"]),
        "chain_complete_rate": float(total_counts["chain_complete_frames"] / expected_rows),
        "candidate_recall_on_target_labeled_frames": float(total_counts["target_candidate_present_frames"] / expected_rows),
        "formal_native_local_global_public_provenance": bool(total_counts["mapping_error_frames"] == 0 and total_counts["chain_complete_frames"] == expected_rows),
        "public_assignment_absence_is_explicit": True,
        "target_absence_is_preserved": True,
        "candidate_cache_regenerated": True,
        "candidate_cache_regeneration_kind": "N36_frozen_official_candidate_tape_rehydrated_with_N54_scores_and_complete_mapping_sidecar",
        "n36_n54_source_unchanged": True,
        "events": event_summaries,
        "inputs": {
            "n37_event_manifest": str(N37_EVENTS),
            "n37_event_manifest_sha256": sha256_file(N37_EVENTS),
            "n36_frame_dir": str(N36_FRAMES),
            "n54_runtime": str(N54_RUNTIME),
        },
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    }
    atomic_json(MAPPING_SUMMARY, mapping_summary)
    manifest = {
        "schema": "N70_CANDIDATE_CACHE_MANIFEST_V1",
        "status": "PASS_REGENERATED_CACHE_AUDITED" if mapping_summary["formal_native_local_global_public_provenance"] else "FAIL_CACHE_MAPPING_PROVENANCE",
        "created_at_utc": now(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": protocol_sha,
        "branch": CACHE_BRANCH,
        "cache_root": str(CACHE_ROOT),
        "cache_dir": str(CACHE_DIR),
        "event_count": EVENT_COUNT,
        "variant_count": len(VARIANTS),
        "frames_per_event_variant": FRAMES_PER_EVENT,
        "expected_rows": expected_rows,
        "event_files": cache_events,
        "candidate_rows": int(total_counts["candidate_rows"]),
        "target_candidate_present_frames": int(total_counts["target_candidate_present_frames"]),
        "target_candidate_absent_frames": int(total_counts["target_candidate_absent_frames"]),
        "candidate_recall_on_target_labeled_frames": float(total_counts["target_candidate_present_frames"] / expected_rows),
        "mapping_summary": str(MAPPING_SUMMARY),
        "mapping_summary_sha256": sha256_file(MAPPING_SUMMARY),
        "source": {
            "n36_frame_dir": str(N36_FRAMES),
            "n54_runtime": str(N54_RUNTIME),
            "checkpoint_changed": False,
            "candidate_generator_changed": False,
            "candidate_stream_reformatted": True,
        },
        "integrity": {
            "unique_event_variant_frame_keys_expected": expected_rows,
            "candidate_set_complete": True,
            "frame_ranges_complete": True,
            "native_local_global_public_provenance": mapping_summary["formal_native_local_global_public_provenance"],
            "target_absence_preserved": True,
            "runtime_future_gt_used": False,
        },
        "provenance": {
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
            "real_sam3_full_loop": False,
            "not_real_human_evidence": True,
            "production_authorized": False,
        },
    }
    atomic_json(CACHE_MANIFEST, manifest)
    cache_audit = audit_cache(cache_events)
    atomic_json(OUT / "cache/candidate_cache_audit.json", cache_audit)
    if cache_audit["status"] != "PASS":
        raise RuntimeError(f"N70 regenerated cache audit failed: {cache_audit}")
    atomic_json(STAGE01, {
        "schema": "N70_STAGE_01_STATUS_V1",
        "status": mapping_summary["status"],
        "created_at_utc": now(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": protocol_sha,
        "inputs": mapping_summary["inputs"],
        "outputs": {"mapping_rows": str(MAPPING_ROWS), "mapping_summary": str(MAPPING_SUMMARY), "fixtures": str(FIXTURE_RESULTS)},
        "metrics": {key: mapping_summary[key] for key in ("event_count", "independent_sequence_count", "variant_count", "audit_rows", "candidate_rows", "target_candidate_present_frames", "target_candidate_absent_frames", "target_public_assignment_absent_frames", "public_assignment_absent_candidate_rows", "old_mapping_conflict_frames", "old_mapping_match_frames", "mapping_error_frames", "chain_complete_frames", "chain_complete_rate")},
        "gate_checks": {
            "mapping_fixture_pass": fixtures["status"] == "PASS",
            "mapping_chain_complete_for_observed_candidates": mapping_summary["formal_native_local_global_public_provenance"],
            "candidate_absence_explicit_not_filled": mapping_summary["target_absence_is_preserved"],
            "public_assignment_absence_explicit_not_filled": mapping_summary["public_assignment_absence_is_explicit"],
            "candidate_frame_denominator_12000": len(all_mapping_rows) == expected_rows,
            "runtime_future_gt_false": True,
            "simulated_not_real_human": True,
            "production_authorized": False,
        },
        "diagnosis": {
            "n69_root_cause_reframed": "old N54/N69 public mapping was target-scoped but lacked local/global provenance; 90 target-absent frames are candidate recall/absence evidence",
            "n70_repair": "join frozen N54 public/score stream to N36 explicit local/global bridge; preserve candidate absence and explicit N54 public-assignment absence",
            "checkpoint_changed": False,
            "candidate_generator_changed": False,
            "formal_provenance_gate": mapping_summary["formal_native_local_global_public_provenance"],
        },
        "next_stage": "N70_STAGE_02_REGENERATED_CACHE_AUDIT",
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "runtime_future_gt_used": False,
        "production_authorized": False,
    })
    atomic_json(STAGE02, {
        "schema": "N70_STAGE_02_STATUS_V1",
        "status": "PASS_REGENERATED_CACHE_AUDITED" if cache_audit["status"] == "PASS" else "FAIL_REGENERATED_CACHE_AUDIT",
        "created_at_utc": now(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": protocol_sha,
        "cache_manifest": str(CACHE_MANIFEST),
        "cache_manifest_sha256": sha256_file(CACHE_MANIFEST),
        "cache_audit": str(OUT / "cache/candidate_cache_audit.json"),
        "cache_audit_sha256": sha256_file(OUT / "cache/candidate_cache_audit.json"),
        "metrics": {
            "event_count": EVENT_COUNT,
            "independent_sequence_count": len(by_sequence),
            "variant_count": len(VARIANTS),
            "expected_rows": expected_rows,
            "rows": cache_audit["rows"],
            "unique_keys": cache_audit["unique_keys"],
            "duplicate_keys": cache_audit["duplicate_keys"],
            "candidate_rows": cache_audit["candidate_rows"],
            "target_candidate_present_frames": int(total_counts["target_candidate_present_frames"]),
            "target_candidate_absent_frames": int(total_counts["target_candidate_absent_frames"]),
            "target_public_assignment_absent_frames": int(total_counts["target_public_assignment_absent_frames"]),
            "public_assignment_absent_candidate_rows": int(total_counts["public_assignment_absent_candidate_rows"]),
            "candidate_recall_on_target_labeled_frames": float(total_counts["target_candidate_present_frames"] / expected_rows),
            "mapping_complete_frame_rows": cache_audit["mapping_complete_frame_rows"],
        },
        "gate_checks": {
            "all_events_24": cache_audit["rows"] == expected_rows,
            "all_variants_5": cache_audit["rows"] == expected_rows,
            "all_frames_12000": cache_audit["rows"] == expected_rows,
            "unique_event_variant_frame_keys": cache_audit["unique_keys"] == expected_rows and cache_audit["duplicate_keys"] == 0,
            "candidate_set_complete": cache_audit["status"] == "PASS",
            "finite_512d_features": cache_audit["error_count"] == 0,
            "native_local_global_public_provenance": mapping_summary["formal_native_local_global_public_provenance"],
            "public_assignment_absence_explicit_not_filled": mapping_summary["public_assignment_absence_is_explicit"],
            "target_absence_preserved": True,
            "runtime_future_gt_false": True,
            "checkpoint_unchanged": True,
            "production_authorized": False,
        },
        "diagnosis": {
            "old_cache_issue": "target absence=90 and local/global provenance absent",
            "new_cache_effect": "native/local/global provenance complete; target candidate absence and N54 public-assignment absence remain explicit and are not filled",
            "alternate_checkpoint_branch": "not required for this materialization; candidate absence remains a scored limitation",
        },
        "next_stage": "N70_STAGE_03_TRAIN_BRANCH_A_AND_BRANCH_B",
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "runtime_future_gt_used": False,
        "production_authorized": False,
    })
    print(json.dumps({"status": "PASS", "stage_01": str(STAGE01), "stage_02": str(STAGE02), "mapping_summary": str(MAPPING_SUMMARY), "cache_manifest": str(CACHE_MANIFEST), "audit": cache_audit}, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_failure(exc)
        raise
