"""CPU-side public-association replay for the N72R5R1 continuation.

The N72R5 official workers intentionally exported candidate streams without a
public-ID axis.  This module supplies that missing outer layer without
touching the SAM3 workers or their JSONL files.  It keeps three axes explicit:

* session-local/native candidate identifiers;
* persistent association-state identifiers;
* public MOT identifiers owned by :class:`SequencePersistentIdentityRuntime`.

The only global assignment call is ``solve_effect_assignment``.  Unmatched
candidate rows are first recorded as the solver's explicit ``NONE`` decision;
the frozen outer birth policy may then allocate a new persistent public
identity.  This distinction is retained in the sidecar.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sam3_intermot.association.effect_assignment import solve_effect_assignment
from sam3_intermot.association.online_associator import score_matrix_pairwise
from sam3_intermot.association.state_manager import StateManager, StateManagerConfig
from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.identity.persistent_runtime import (
    PersistentIdentityRecord,
    SequencePersistentIdentityRuntime,
)
from sam3_intermot.identity.public_authority import PublicAuthorityBridge


BRANCHES = (
    "B0_NO_INTERVENTION",
    "B1_SPATIAL_CORRECTION_ONLY",
    "B2_SPATIAL_CORRECTION_PLUS_IMAGE_RECOVERY",
    "B3_SPATIAL_CORRECTION_PLUS_TVC",
    "B4_SPATIAL_CORRECTION_PLUS_RECOVERY_PLUS_TVC",
)
TVC_BRANCHES = {
    "B3_SPATIAL_CORRECTION_PLUS_TVC",
    "B4_SPATIAL_CORRECTION_PLUS_RECOVERY_PLUS_TVC",
}
HORIZON = 100
FEATURE_DIM = 512
IOU_THRESHOLD = 0.5
TVC_TRUST_RADIUS = 1.0
TVC_COMPETITOR_TOP_K = 3
TVC_MAD_SCALE_FACTOR = 1.4826
TVC_SCALE_EPS = 1.0e-6
DATA_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")


def now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    """Convert runtime objects into an audit-only JSON representation."""

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite value cannot be written to JSON")
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(child) for child in value]
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return jsonable(value.as_dict())
    if hasattr(value, "__dict__"):
        return {
            "__class__": f"{type(value).__module__}.{type(value).__name__}",
            "attributes": jsonable(vars(value)),
        }
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def atomic_json(path: Path, payload: Any) -> None:
    atomic_write(
        path,
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(
            json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
    )


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
                raise TypeError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def feature_hash(feature: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(feature, dtype="<f4").tobytes()).hexdigest()


def finite_feature(value: Any, *, label: str) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float32).reshape(-1)
    if feature.size != FEATURE_DIM or not np.all(np.isfinite(feature)):
        raise ValueError(f"{label}: expected finite {FEATURE_DIM}-D feature, got {feature.shape}")
    norm = float(np.linalg.norm(feature))
    if norm <= 1.0e-6:
        raise ValueError(f"{label}: zero-norm feature")
    return feature / norm


def finite_box(value: Any, *, label: str, allow_empty: bool = False) -> np.ndarray:
    box = np.asarray(value, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)):
        raise ValueError(f"{label}: invalid box")
    if allow_empty:
        valid = box[2] >= box[0] and box[3] >= box[1]
    else:
        valid = box[2] > box[0] and box[3] > box[1]
    if not valid:
        raise ValueError(f"{label}: non-positive box")
    return box


def box_iou(left: Any, right: Any) -> float:
    # Official Stage07 preserves finite zero-area disappearance observations.
    # They remain valid candidate rows but contribute zero geometric overlap.
    a = finite_box(left, label="left", allow_empty=True)
    b = finite_box(right, label="right", allow_empty=True)
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def normalize_candidate(
    candidate: Mapping[str, Any],
    *,
    sequence: str,
    frame: int,
    source_kind: str,
) -> dict[str, Any]:
    """Normalize N36 tape or Stage07 official rows without using public IDs."""

    raw_feature = candidate.get("feature", candidate.get("machine_embedding"))
    feature = finite_feature(raw_feature, label=f"{source_kind}:{sequence}:{frame}")
    raw = candidate.get("official_raw_sam_id", candidate.get("raw_native_id"))
    if raw is None:
        raw = candidate.get("native_tid", candidate.get("local_native_id"))
    adapter = candidate.get("adapter_external_id", candidate.get("local_native_id"))
    if adapter is None:
        adapter = candidate.get("native_tid", raw)
    if raw is None or adapter is None:
        raise ValueError(f"{source_kind}:{sequence}:{frame}: native mapping is incomplete")
    raw = int(raw)
    adapter = int(adapter)
    index = int(candidate.get("candidate_index", 0))
    digest = str(candidate.get("feature_sha256") or feature_hash(feature))
    uid = candidate.get("candidate_uid")
    if uid in (None, "", "None"):
        uid = f"{source_kind}:{sequence}:{frame}:{index}:{raw}:{adapter}:{digest[:20]}"
    box_key = "box_xyxy" if candidate.get("box_xyxy") is not None else "box"
    box = finite_box(
        candidate.get(box_key),
        label=f"{source_kind}:{sequence}:{frame}:{index}",
        allow_empty=True,
    )
    global_native = candidate.get("sequence_global_native_id")
    normalized = {
        "candidate_uid": str(uid),
        "candidate_index": index,
        "sequence": str(sequence),
        "frame": int(frame),
        "source_kind": str(source_kind),
        "official_raw_sam_id": raw,
        "raw_native_id": raw,
        "adapter_external_id": adapter,
        "adapter_visible_id": candidate.get("adapter_visible_id"),
        "local_native_id": candidate.get("local_native_id"),
        "sequence_global_native_id": None if global_native is None else int(global_native),
        "native_id_source": candidate.get("native_id_source", candidate.get("raw_native_id_source")),
        "native_tid_scope": candidate.get("native_tid_scope", "session_local_adapter_external_id"),
        "native_scope": candidate.get(
            "native_scope",
            candidate.get("native_tid_scope", "session_local_adapter_external_id"),
        ),
        "native_tid": adapter,
        "binding_sam_id": adapter,
        "adapter_mapping_collision": False,
        "association_native_id_source": "adapter_external_id",
        "native_age": float(candidate.get("native_age", 0.0)),
        "confidence": float(candidate.get("confidence", candidate.get("presence_score", 1.0))),
        "presence_score": float(candidate.get("presence_score", candidate.get("confidence", 1.0))),
        "box_xyxy": [float(value) for value in box],
        "feature": feature.astype(float).tolist(),
        "feature_sha256": digest,
        "feature_dim": FEATURE_DIM,
        "feature_source": candidate.get("feature_source", candidate.get("embedding_status", "frozen_candidate")),
        "mask_sha256": (candidate.get("mask") or {}).get("sha256") if isinstance(candidate.get("mask"), dict) else None,
        "public_id_source_field_ignored": candidate.get("public_id") is not None,
    }
    if not math.isfinite(normalized["confidence"]) or not math.isfinite(normalized["presence_score"]):
        raise ValueError(f"{source_kind}:{sequence}:{frame}:{index}: non-finite confidence")
    return normalized


def finalize_native_axis(candidates: Sequence[dict[str, Any]], *, label: str) -> list[dict[str, Any]]:
    """Preserve rows when an official adapter slot is duplicated in a frame."""

    rows = [dict(item) for item in candidates]
    raw_values = [int(item["official_raw_sam_id"]) for item in rows]
    if len(raw_values) != len(set(raw_values)):
        raise ValueError(f"{label}: duplicate official raw SAM IDs")
    counts: dict[int, int] = defaultdict(int)
    for item in rows:
        counts[int(item["adapter_external_id"])] += 1
    for item in rows:
        adapter = int(item["adapter_external_id"])
        if counts[adapter] == 1:
            item["native_tid"] = adapter
            item["binding_sam_id"] = adapter
            item["adapter_mapping_collision"] = False
            item["association_native_id_source"] = "adapter_external_id"
        else:
            # The official adapter value stays untouched.  This negative key
            # is only a collision-free association/TrackManager handle based
            # on the separately preserved raw candidate axis.
            raw = int(item["official_raw_sam_id"])
            item["native_tid"] = -(1_000_000 + raw)
            item["binding_sam_id"] = -(1_000_000 + raw)
            item["adapter_mapping_collision"] = True
            item["association_native_id_source"] = "official_raw_sam_id_disambiguation_after_adapter_collision"
    native_values = [int(item["native_tid"]) for item in rows]
    if len(native_values) != len(set(native_values)):
        raise ValueError(f"{label}: disambiguated association native IDs collide")
    return rows


def candidate_obs(candidate: Mapping[str, Any], frame: int, *, source: str) -> PromptObjectObservation:
    return PromptObjectObservation(
        frame_idx=int(frame),
        sam_object_id=int(candidate.get("binding_sam_id", candidate["native_tid"])),
        raw_sam_object_id=int(candidate["official_raw_sam_id"]),
        mask=np.ones((1, 1), dtype=bool),
        box_xyxy=np.asarray(candidate["box_xyxy"], dtype=float),
        confidence=float(candidate["confidence"]),
        presence_score=float(candidate["presence_score"]),
        source=str(source),
        is_human_verified=source == "simulated_human_correction",
        source_run_id=str(candidate.get("source_run_id", "n72r5r1")),
        session_id=str(candidate.get("session_id", "n72r5r1")),
        candidate_index=int(candidate["candidate_index"]),
    )


def candidate_for_score(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "obs_id": int(candidate["candidate_index"]),
        "candidate_uid": str(candidate["candidate_uid"]),
        "box": np.asarray(candidate["box_xyxy"], dtype=float),
        "feat": np.asarray(candidate["feature"], dtype=np.float32),
        "has_feat": 1.0,
        "native_tid": int(candidate["native_tid"]),
        "native_scope": candidate.get("native_scope", candidate.get("native_tid_scope")),
        "native_age": float(candidate.get("native_age", 0.0)),
        "conf": float(candidate.get("confidence", 1.0)),
    }


def load_prefix_rows(tape_path: Path, event_frame: int, sequence: str) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    with tape_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{tape_path}:{line_number}: expected object")
            frame = int(row.get("frame", -1))
            if frame >= int(event_frame):
                break
            if frame < 0:
                raise ValueError(f"{tape_path}:{line_number}: negative frame")
            if frame in by_frame:
                raise ValueError(f"duplicate prefix frame: {sequence}:{frame}")
            if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False:
                raise ValueError(f"prefix runtime GT flag is not false: {sequence}:{frame}")
            candidates = row.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError(f"prefix candidates missing: {sequence}:{frame}")
            normalized = [
                normalize_candidate(
                    candidate,
                    sequence=sequence,
                    frame=frame,
                    source_kind="PREFIX_N36_TAPE",
                )
                for candidate in candidates
            ]
            if len({str(item["candidate_uid"]) for item in normalized}) != len(normalized):
                raise ValueError(f"prefix candidate UID collision: {sequence}:{frame}")
            normalized = finalize_native_axis(normalized, label=f"prefix:{sequence}:{frame}")
            by_frame[frame] = normalized
    expected = set(range(int(event_frame)))
    if set(by_frame) != expected:
        raise ValueError(
            f"prefix frame coverage mismatch for {sequence}: missing={sorted(expected-set(by_frame))[:8]} "
            f"extra={sorted(set(by_frame)-expected)[:8]}"
        )
    return by_frame


def load_stage07_event_rows(
    stage07_root: Path,
    event_id: str,
    expected_event_frame: int,
    sequence: str,
) -> tuple[dict[str, dict[int, list[dict[str, Any]]]], dict[str, dict[int, dict[str, Any]]], dict[str, str]]:
    manifest_path = stage07_root / "official_full_loop_manifest.json"
    manifest = read_json(manifest_path)
    workers = [item for item in manifest.get("worker_records", []) if str(item.get("event_id")) == str(event_id)]
    if len(workers) != len(BRANCHES):
        raise ValueError(f"Stage07 worker coverage mismatch: {event_id}: {len(workers)}")
    by_branch: dict[str, dict[int, list[dict[str, Any]]]] = {}
    raw_by_branch: dict[str, dict[int, dict[str, Any]]] = {}
    paths: dict[str, str] = {}
    for worker in workers:
        branch = str(worker.get("branch"))
        if branch not in BRANCHES or branch in by_branch:
            raise ValueError(f"invalid/duplicate Stage07 branch: {event_id}/{branch}")
        output = Path(str(worker.get("output")))
        if not output.is_file():
            raise FileNotFoundError(output)
        paths[branch] = str(output)
        rows = read_jsonl(output)
        if len(rows) != HORIZON + 1:
            raise ValueError(f"Stage07 frame count mismatch: {event_id}/{branch}/{len(rows)}")
        normalized_by_frame: dict[int, list[dict[str, Any]]] = {}
        raw_by_frame: dict[int, dict[str, Any]] = {}
        for row in rows:
            frame = int(row.get("frame", -1))
            if frame in normalized_by_frame:
                raise ValueError(f"duplicate Stage07 frame: {event_id}/{branch}/{frame}")
            if row.get("event_id") != event_id or row.get("sequence") != sequence:
                raise ValueError(f"Stage07 event/sequence mismatch: {event_id}/{branch}/{frame}")
            for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
                if row.get(flag) is not False:
                    raise ValueError(f"Stage07 {flag} is not false: {event_id}/{branch}/{frame}")
            candidates = row.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError(f"Stage07 candidates missing: {event_id}/{branch}/{frame}")
            normalized = [
                normalize_candidate(
                    candidate,
                    sequence=sequence,
                    frame=frame,
                    source_kind=f"STAGE07_{branch}",
                )
                for candidate in candidates
            ]
            if len({str(item["candidate_uid"]) for item in normalized}) != len(normalized):
                raise ValueError(f"Stage07 candidate UID collision: {event_id}/{branch}/{frame}")
            normalized = finalize_native_axis(normalized, label=f"stage07:{event_id}/{branch}/{frame}")
            normalized_by_frame[frame] = normalized
            raw_by_frame[frame] = row
        expected = set(range(int(expected_event_frame), int(expected_event_frame) + HORIZON + 1))
        if set(normalized_by_frame) != expected:
            raise ValueError(f"Stage07 frame range mismatch: {event_id}/{branch}")
        by_branch[branch] = normalized_by_frame
        raw_by_branch[branch] = raw_by_frame
    return by_branch, raw_by_branch, paths


def load_gt(sequence: str, *, data_root: Path = DATA_ROOT) -> dict[int, dict[int, dict[str, Any]]]:
    path = data_root / "train" / sequence / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 6:
                raise ValueError(f"malformed GT row {path}:{line_number}")
            frame = int(parts[0]) - 1
            gt_id = int(parts[1])
            x, y, width, height = [float(item) for item in parts[2:6]]
            box = [x, y, x + width, y + height]
            finite_box(box, label=f"GT:{sequence}:{frame}:{gt_id}")
            result[frame][gt_id] = {"box": box, "visibility": None if len(parts) <= 8 else float(parts[8])}
    return result


def current_gt_input(gt_frames: Mapping[int, Mapping[int, Mapping[str, Any]]], frame: int) -> dict[str, Any]:
    items = gt_frames.get(int(frame), {})
    return {
        "boxes": [list(item["box"]) for _, item in sorted(items.items())],
        "gt_ids": [int(gt_id) for gt_id in sorted(items)],
    }


def new_runtime(sequence: str, event_id: str) -> SequencePersistentIdentityRuntime:
    bridge = PublicAuthorityBridge(f"n72r5r1:{event_id}", sequence)
    return SequencePersistentIdentityRuntime(
        sequence,
        authority_bridge=bridge,
        public_id_start=1000,
        state_id_start=1,
    )


def new_state_manager(
    runtime: SequencePersistentIdentityRuntime,
    *,
    max_lost_gap: int = 90,
) -> StateManager:
    manager = StateManager(
        StateManagerConfig(
            score_threshold=0.0,
            variant="reid",
            max_lost_gap=int(max_lost_gap),
            external_identity_authority=True,
        ),
        public_authority_resolver=runtime.authority,
    )
    for record in sorted(runtime.identities.values(), key=lambda item: item.association_state_id):
        feature = np.asarray(record.appearance_state.get("last_machine_feature", []), dtype=np.float32)
        if feature.size != FEATURE_DIM or not np.all(np.isfinite(feature)) or float(np.linalg.norm(feature)) <= 1.0e-6:
            feature = np.zeros(FEATURE_DIM, dtype=np.float32)
            feature[0] = 1.0
        box = np.asarray(record.last_box or [0.0, 0.0, 1.0, 1.0], dtype=float)
        native = int(record.current_adapter_external_id if record.current_adapter_external_id is not None else -1)
        state = manager.register_from_persistent_identity(
            record,
            {
                "feat": feature,
                "box": box,
                "native_tid": native,
                "native_scope": getattr(record, "last_native_scope", None),
            },
            int(record.last_seen_frame if record.last_seen_frame is not None else record.created_frame),
        )
        if record.status == "LOST":
            state.mark_lost(int(record.last_seen_frame if record.last_seen_frame is not None else record.created_frame))
        elif record.status == "TERMINATED":
            state.terminate()
    return manager


def persist_machine_feature(record: PersistentIdentityRecord, candidate: Mapping[str, Any], frame: int) -> None:
    feature = finite_feature(candidate["feature"], label=f"candidate:{candidate['candidate_uid']}")
    record.appearance_state["last_machine_feature"] = feature.astype(float).tolist()
    record.appearance_state["last_machine_feature_sha256"] = str(candidate["feature_sha256"])
    record.appearance_state["last_machine_feature_frame"] = int(frame)
    record.motion_state_ref = {
        "last_box": [float(value) for value in candidate["box_xyxy"]],
        "last_frame": int(frame),
    }


def state_records(
    runtime: SequencePersistentIdentityRuntime,
    states: Sequence[Any],
) -> list[PersistentIdentityRecord]:
    records: list[PersistentIdentityRecord] = []
    for state in states:
        record = runtime.get_identity_by_state_id(int(state.pid))
        if record is None:
            raise RuntimeError(f"state has no persistent owner: {state.pid}")
        records.append(record)
    return records


def score_existing(
    manager: StateManager,
    candidates: Sequence[Mapping[str, Any]],
    frame: int,
    states: Sequence[Any] | None = None,
) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
    selected = list(manager.candidates(int(frame)) if states is None else states)
    observations = [candidate_for_score(candidate) for candidate in candidates]
    audit: dict[str, Any] = {}
    base = score_matrix_pairwise(
        selected,
        observations,
        int(frame),
        None,
        reid_weights={"sim": 1.5, "iou": 1.0, "native": 0.5, "gap": 0.1},
        positive_bonus=5.0,
        native_bonus=3.0,
        authority_mode="permanent",
        hard_frames=1,
        decay_frames=8,
        refresh_threshold=0.5,
        appearance_memory=None,
        score_audit=audit,
    )
    return selected, np.asarray(base, dtype=np.float64), audit


def exact_solve(
    runtime: SequencePersistentIdentityRuntime,
    states: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
    fused_candidate_state: np.ndarray,
    *,
    event_id: str,
    branch: str,
    frame: int,
    session_id: str,
) -> dict[str, Any]:
    records = state_records(runtime, states)
    matrix = np.asarray(fused_candidate_state, dtype=np.float64)
    if matrix.shape != (len(candidates), len(states)):
        raise RuntimeError(
            f"candidate×state matrix shape mismatch: {event_id}/{branch}/{frame}: {matrix.shape} "
            f"!= {(len(candidates), len(states))}"
        )
    return solve_effect_assignment(
        candidate_rows=[
            {"candidate_uid": str(candidate["candidate_uid"]), "candidate_index": int(candidate["candidate_index"])}
            for candidate in candidates
        ],
        persistent_states=records,
        fused_state_candidate_scores=matrix.T,
        source_run_id=f"n72r5r1:{event_id}:{branch}:{frame}",
        session_id=session_id,
        none_score=0.0,
    )


def assignment_by_uid(artifact: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = artifact.get("assignment_rows")
    if not isinstance(rows, list):
        raise ValueError("exact solver artifact has no assignment_rows")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = str(row.get("candidate_uid"))
        if uid in result:
            raise ValueError(f"duplicate solver assignment UID: {uid}")
        result[uid] = dict(row)
    return result


def apply_exact_frame(
    runtime: SequencePersistentIdentityRuntime,
    manager: StateManager,
    states: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
    base_candidate_state: np.ndarray,
    fused_candidate_state: np.ndarray,
    solver: Mapping[str, Any],
    *,
    frame: int,
    event_id: str,
    branch: str,
    session_id: str,
    event_frame: int,
    source_path: str | None = None,
    candidate_role: str | None = None,
    tvc: Mapping[str, Any] | None = None,
    birth_none: bool = True,
    memory_read: bool = False,
    memory_read_source: str | None = None,
    freeze_public_ids: Iterable[int] | None = None,
    persistence_mode: str | None = None,
    birth_none_excluded_uids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Apply an exact solver result and retain solver-NONE vs birth semantics."""

    decisions = assignment_by_uid(solver)
    expected = {str(candidate["candidate_uid"]) for candidate in candidates}
    if set(decisions) != expected:
        raise RuntimeError(f"solver candidate coverage mismatch: {event_id}/{branch}/{frame}")
    state_by_id = {int(state.pid): state for state in states}
    frozen_public_ids = {int(value) for value in (freeze_public_ids or ())}
    no_birth_uids = {str(value) for value in (birth_none_excluded_uids or ())}
    matched_state_ids: set[int] = set()
    candidate_decisions: list[dict[str, Any]] = []
    public_to_uid: dict[int, str] = {}
    for candidate in candidates:
        uid = str(candidate["candidate_uid"])
        decision = decisions[uid]
        solver_public = decision.get("public_id")
        final_public: int | None
        assignment_status: str
        birth_reason: str | None = None
        if solver_public is not None:
            final_public = int(solver_public)
            record = runtime.get_identity_by_public_id(final_public)
            if record is None:
                raise RuntimeError(f"solver returned unknown public ID: {event_id}/{branch}/{frame}/{final_public}")
            state = state_by_id.get(int(record.association_state_id))
            if state is None:
                raise RuntimeError(f"solver public has no current state axis: {event_id}/{branch}/{frame}/{final_public}")
            matched_state_ids.add(int(state.pid))
            runtime.bind_candidate(
                record,
                uid,
                candidate_obs(candidate, frame, source="frozen_candidate"),
                int(frame),
                session_id=session_id,
                adapter_external_id=int(candidate["adapter_external_id"]),
                raw_sam_id=int(candidate["official_raw_sam_id"]),
                native_scope=candidate.get("native_scope", candidate.get("native_tid_scope")),
            )
            freeze_appearance = int(record.public_id) in frozen_public_ids
            state.update_machine(
                np.asarray(candidate["feature"], dtype=np.float32),
                np.asarray(candidate["box_xyxy"], dtype=float),
                int(frame),
                int(candidate["native_tid"]),
                0.9,
                update_prototype=not freeze_appearance,
                native_scope=candidate.get("native_scope", candidate.get("native_tid_scope")),
            )
            if freeze_appearance:
                # Motion/native continuity still follows the assigned
                # candidate, but the identity-scoped appearance state remains
                # the event-corrected evidence.  This opt-in probe prevents a
                # wrong future assignment from overwriting the protected
                # appearance prototype without changing the solver itself.
                record.motion_state_ref = {
                    "last_box": [float(value) for value in candidate["box_xyxy"]],
                    "last_frame": int(frame),
                }
            else:
                persist_machine_feature(record, candidate, frame)
            assignment_status = "ASSIGNED_TO_PUBLIC_ID"
        elif birth_none and uid not in no_birth_uids:
            record = runtime.create_identity(
                int(frame),
                candidate_obs(candidate, frame, source="frozen_candidate"),
                session_id=session_id,
                adapter_external_id=int(candidate["adapter_external_id"]),
                raw_sam_id=int(candidate["official_raw_sam_id"]),
                native_scope=candidate.get("native_scope", candidate.get("native_tid_scope")),
                candidate_uid=uid,
                appearance_state={
                    "last_machine_feature": list(candidate["feature"]),
                    "last_machine_feature_sha256": str(candidate["feature_sha256"]),
                    "last_machine_feature_frame": int(frame),
                },
                motion_state_ref={"last_box": list(candidate["box_xyxy"]), "last_frame": int(frame)},
            )
            manager.register_from_persistent_identity(
                record,
                {
                    "feat": np.asarray(candidate["feature"], dtype=np.float32),
                    "box": np.asarray(candidate["box_xyxy"], dtype=float),
                    "native_tid": int(candidate["native_tid"]),
                    "native_scope": candidate.get("native_scope", candidate.get("native_tid_scope")),
                },
                int(frame),
            )
            final_public = int(record.public_id)
            matched_state_ids.add(int(record.association_state_id))
            assignment_status = "ASSIGNED_TO_PUBLIC_ID"
            birth_reason = "frozen_outer_birth_policy_after_explicit_none"
        else:
            final_public = None
            assignment_status = "EXPLICIT_NONE"
        if final_public is not None:
            if final_public in public_to_uid:
                raise RuntimeError(f"duplicate final public assignment: {event_id}/{branch}/{frame}/{final_public}")
            public_to_uid[final_public] = uid
        candidate_decisions.append(
            {
                **{key: deepcopy(value) for key, value in candidate.items() if key != "feature"},
                "feature_dim": FEATURE_DIM,
                "solver_status": str(decision.get("status")),
                "solver_public_id": None if solver_public is None else int(solver_public),
                "solver_association_state_id": decision.get("association_state_id"),
                "solver_score": float(decision.get("score", 0.0)),
                "public_id": final_public,
                "assignment_status": assignment_status,
                "outer_birth_assigned": bool(birth_reason is not None),
                "birth_reason": birth_reason,
            }
        )
    for state in states:
        if int(state.pid) in matched_state_ids:
            continue
        record = runtime.get_identity_by_state_id(int(state.pid))
        if getattr(state, "state", None) == "ACTIVE":
            state.mark_lost(int(frame))
        else:
            state.advance_lost()
        if record is not None and record.status != "TERMINATED":
            record.status = "LOST"
            record.current_session_id = None
            record.current_adapter_external_id = None
            record.current_raw_sam_id = None
    identity_rows = runtime.record_frame_decisions(int(frame), public_to_uid)
    base = np.asarray(base_candidate_state, dtype=np.float64)
    fused = np.asarray(fused_candidate_state, dtype=np.float64)
    state_axis = [int(state.pid) for state in states]
    # Outer births append a new persistent state after the solver's input
    # axis was captured.  The frame artifact must expose the complete public
    # axis, otherwise a valid birth looks like a mapping hole to Stage09.
    records = sorted(runtime.identities.values(), key=lambda item: item.public_id)
    frame_record = {
        "schema_version": "N72R5R1_PUBLIC_ASSOCIATION_FRAME_V1",
        "record_kind": "public_assignment_frame",
        "event_id": str(event_id),
        "sequence": runtime.sequence,
        "branch": str(branch),
        "session_id": str(session_id),
        "event_frame": int(event_frame),
        "frame": int(frame),
        "frame_horizon": int(frame - event_frame),
        "phase": "EVENT_Y_PRE" if frame == event_frame and candidate_role == "PRE_INTERVENTION_Y_PRE" else (
            "EVENT_Y_POST" if frame == event_frame else "FUTURE_ASSOCIATION"
        ),
        "candidate_role": candidate_role,
        "candidate_stream_source": source_path,
        "candidate_rows": candidate_decisions,
        "candidate_count": len(candidate_decisions),
        "assignment_decision_coverage": 1.0,
        "assignment_statuses": sorted({str(row["assignment_status"]) for row in candidate_decisions}),
        "identity_rows": identity_rows,
        "identity_count": len(identity_rows),
        "public_id_axis": [int(record.public_id) for record in records],
        "association_state_axis": state_axis,
        "base_score_matrix": base.tolist(),
        "fused_score_matrix": fused.tolist(),
        "score_matrix_orientation": "candidate_x_association_state",
        "solver": deepcopy(dict(solver)),
        "tvc": None if tvc is None else deepcopy(dict(tvc)),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "public_id_inference": False,
        "memory_read": bool(memory_read),
        "memory_read_source": memory_read_source,
        "event_frame_memory_read": False,
        "first_memory_visible_frame": int(event_frame + 1),
        "public_id_immutability": True,
        "candidate_index_to_public_id": False,
        "raw_sam_id_to_public_id": False,
        "persistence_mode": persistence_mode,
        "appearance_prototype_frozen_public_ids": sorted(frozen_public_ids),
    }
    return {
        "frame_record": frame_record,
        "candidate_decisions": candidate_decisions,
        "identity_rows": identity_rows,
        "public_to_uid": public_to_uid,
        "solver": deepcopy(dict(solver)),
        "states": list(states),
        "base_candidate_state": base,
        "fused_candidate_state": fused,
    }


def robust_tvc_scale(base_state_candidate: np.ndarray) -> dict[str, Any]:
    values = np.asarray(base_state_candidate, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("TVC event score distribution is empty")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "median": median,
        "mad": mad,
        "mad_scale_factor": TVC_MAD_SCALE_FACTOR,
        "scale": max(TVC_SCALE_EPS, TVC_MAD_SCALE_FACTOR * mad),
        "epsilon": TVC_SCALE_EPS,
        "source": "event_frame_current_public_state_x_candidate_score_matrix",
        "future_metrics_excluded": True,
    }


def choose_tvc_competitors(
    states: Sequence[Any],
    runtime: SequencePersistentIdentityRuntime,
    base_state_candidate: np.ndarray,
    target_public: int,
) -> dict[str, Any]:
    ranked: list[tuple[float, int]] = []
    for index, state in enumerate(states):
        record = runtime.get_identity_by_state_id(int(state.pid))
        if record is None or int(record.public_id) == int(target_public):
            continue
        row = np.asarray(base_state_candidate[index], dtype=float)
        value = float(np.max(row)) if row.size else -math.inf
        ranked.append((value, int(record.public_id)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = [public for _, public in ranked[:TVC_COMPETITOR_TOP_K]]
    return {
        "target_public_id": int(target_public),
        "top_k": TVC_COMPETITOR_TOP_K,
        "ranked_public_ids": [public for _, public in ranked],
        "selected_public_ids": selected,
        "selection_source": "event_frame_base_score_max_over_candidates",
        "future_metrics_excluded": True,
        "runtime_future_gt_used": False,
    }


def tvc_residual(
    states: Sequence[Any],
    runtime: SequencePersistentIdentityRuntime,
    candidates: Sequence[Mapping[str, Any]],
    *,
    target_public: int,
    competitor_publics: Sequence[int],
    human_anchor: np.ndarray,
    persistent_target: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    target_index = None
    competitor_features: list[tuple[int, np.ndarray]] = []
    for index, state in enumerate(states):
        record = runtime.get_identity_by_state_id(int(state.pid))
        if record is None:
            continue
        if int(record.public_id) == int(target_public):
            target_index = index
        if int(record.public_id) in {int(value) for value in competitor_publics}:
            competitor_features.append((int(record.public_id), finite_feature(record.appearance_state.get("last_machine_feature", state.prototype), label="tvc competitor")))
    residual = np.zeros((len(states), len(candidates)), dtype=np.float64)
    details: list[dict[str, Any]] = []
    if target_index is None:
        raise RuntimeError(f"TVC target public is not on current state axis: {target_public}")
    human = finite_feature(human_anchor, label="tvc human anchor")
    persistent = finite_feature(persistent_target, label="tvc persistent target")
    competitors = {public: feature for public, feature in competitor_features}
    for candidate_index, candidate in enumerate(candidates):
        feature = finite_feature(candidate["feature"], label=f"tvc candidate:{candidate['candidate_uid']}")
        human_similarity = float(np.dot(feature, human))
        persistent_similarity = float(np.dot(feature, persistent))
        competitor_values = [
            (float(np.dot(feature, competitor_feature)), public)
            for public, competitor_feature in competitors.items()
        ]
        if competitor_values:
            best_competitor_similarity, best_competitor_public = max(
                competitor_values, key=lambda item: (item[0], -item[1])
            )
        else:
            best_competitor_similarity, best_competitor_public = 0.0, None
        relative = human_similarity + persistent_similarity - best_competitor_similarity
        normalized = relative / max(float(scale), TVC_SCALE_EPS)
        bounded = float(np.clip(normalized, -TVC_TRUST_RADIUS, TVC_TRUST_RADIUS))
        residual[target_index, candidate_index] = bounded
        details.append(
            {
                "candidate_uid": str(candidate["candidate_uid"]),
                "candidate_index": int(candidate["candidate_index"]),
                "target_public_id": int(target_public),
                "human_target_similarity": human_similarity,
                "persistent_target_similarity": persistent_similarity,
                "best_competitor_similarity": best_competitor_similarity,
                "best_competitor_public_id": best_competitor_public,
                "relative_margin": relative,
                "normalized_margin": normalized,
                "bounded_target_row_residual": bounded,
                "scale": float(scale),
                "trust_radius": TVC_TRUST_RADIUS,
            }
        )
    return residual, details


def learned_tvc_residual(
    states: Sequence[Any],
    runtime: SequencePersistentIdentityRuntime,
    candidates: Sequence[Mapping[str, Any]],
    *,
    target_public: int,
    human_anchor: np.ndarray,
    persistent_target: np.ndarray | None,
    model: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply one frozen, candidate-wise TVC verifier to the target row.

    The verifier is deliberately an interface probe: it uses the current
    candidate feature and the current-frame event anchor, plus the prefix
    prototype when one exists.  It never uses GT or a future observation and
    leaves every non-target state row exactly zero.
    """

    target_index = None
    for index, state in enumerate(states):
        record = runtime.get_identity_by_state_id(int(state.pid))
        if record is not None and int(record.public_id) == int(target_public):
            target_index = index
            break
    if target_index is None:
        raise RuntimeError(f"learned TVC target public is not on current state axis: {target_public}")
    weights = np.asarray(model.get("weights", []), dtype=np.float64).reshape(-1)
    mean = np.asarray(model.get("standardization_mean", []), dtype=np.float64).reshape(-1)
    scale = np.asarray(model.get("standardization_scale", []), dtype=np.float64).reshape(-1)
    if weights.size != 3 or mean.size != 3 or scale.size != 3 or np.any(scale <= 0.0):
        raise ValueError("TVC_V1 model has invalid three-feature parameters")
    bias = float(model.get("bias", 0.0))
    max_abs = float(model.get("max_abs_residual", 8.0))
    if not math.isfinite(bias) or not math.isfinite(max_abs) or max_abs <= 0.0:
        raise ValueError("TVC_V1 model has invalid bias/residual bound")
    human = finite_feature(human_anchor, label="learned tvc human anchor")
    prototype = None if persistent_target is None else finite_feature(persistent_target, label="learned tvc prefix prototype")
    residual = np.zeros((len(states), len(candidates)), dtype=np.float64)
    details: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        feature = finite_feature(candidate["feature"], label=f"learned tvc candidate:{candidate['candidate_uid']}")
        raw_features = np.asarray(
            [
                float(np.dot(feature, human)),
                0.0 if prototype is None else float(np.dot(feature, prototype)),
                0.0 if prototype is None else 1.0,
            ],
            dtype=np.float64,
        )
        standardized = (raw_features - mean) / scale
        logit = float(np.dot(standardized, weights) + bias)
        bounded = float(max_abs * np.tanh(logit))
        residual[target_index, candidate_index] = bounded
        details.append(
            {
                "candidate_uid": str(candidate["candidate_uid"]),
                "candidate_index": int(candidate["candidate_index"]),
                "target_public_id": int(target_public),
                "raw_features": raw_features.tolist(),
                "standardized_features": standardized.tolist(),
                "verifier_logit": logit,
                "bounded_target_row_residual": bounded,
                "max_abs_residual": max_abs,
                "prefix_prototype_available": prototype is not None,
            }
        )
    return residual, details


def candidate_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = str(row["candidate_uid"])
        if uid in result:
            raise ValueError(f"duplicate candidate UID in output: {uid}")
        result[uid] = dict(row)
    return result


def best_candidate_for_box(
    candidates: Sequence[Mapping[str, Any]],
    box: Sequence[float],
    *,
    excluded_uids: Iterable[str] = (),
) -> tuple[float, dict[str, Any] | None]:
    excluded = {str(value) for value in excluded_uids}
    ranked = [
        (box_iou(candidate["box_xyxy"], box), str(candidate["candidate_uid"]), candidate)
        for candidate in candidates
        if str(candidate["candidate_uid"]) not in excluded
    ]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return (float(ranked[0][0]), ranked[0][2]) if ranked else (0.0, None)


def public_axis(runtime: SequencePersistentIdentityRuntime) -> list[dict[str, Any]]:
    return [
        {
            "association_state_id": int(record.association_state_id),
            "public_id": int(record.public_id),
            "mot_track_id": int(record.mot_track_id),
            "identity_lineage_id": int(record.identity_lineage_id),
            "status": str(record.status),
            "created_frame": int(record.created_frame),
            "last_seen_frame": record.last_seen_frame,
        }
        for record in sorted(runtime.identities.values(), key=lambda item: item.public_id)
    ]


def runtime_invariants(runtime: SequencePersistentIdentityRuntime) -> dict[str, Any]:
    audit = runtime.audit()
    return {
        "identity_count": int(audit["identity_count"]),
        "public_ids": list(audit["public_ids"]),
        "public_id_immutable": bool(audit["public_id_immutable"]),
        "mot_track_id_equals_public_id": bool(audit["mot_track_id_equals_public_id"]),
        "candidate_is_not_identity_owner": bool(audit["candidate_is_not_identity_owner"]),
        "candidate_bindings_are_session_local": bool(audit["candidate_bindings_are_session_local"]),
        "track_manager_instance_count": int(audit["track_manager_instance_count"]),
        "auxiliary_track_manager_count": int(audit["auxiliary_track_manager_count"]),
        "invariant_violations": list(audit["invariant_violations"]),
        "runtime_future_gt_used": False,
    }


__all__ = [
    "BRANCHES",
    "DATA_ROOT",
    "HORIZON",
    "IOU_THRESHOLD",
    "TVC_BRANCHES",
    "TVC_COMPETITOR_TOP_K",
    "TVC_MAD_SCALE_FACTOR",
    "TVC_TRUST_RADIUS",
    "atomic_json",
    "atomic_jsonl",
    "atomic_write",
    "assignment_by_uid",
    "apply_exact_frame",
    "best_candidate_for_box",
    "box_iou",
    "candidate_map",
    "candidate_obs",
    "choose_tvc_competitors",
    "current_gt_input",
    "exact_solve",
    "finite_feature",
    "learned_tvc_residual",
    "json_hash",
    "jsonable",
    "load_gt",
    "load_prefix_rows",
    "load_stage07_event_rows",
    "new_runtime",
    "new_state_manager",
    "now_utc",
    "persist_machine_feature",
    "public_axis",
    "read_json",
    "read_jsonl",
    "robust_tvc_scale",
    "runtime_invariants",
    "score_existing",
    "sha256_file",
    "state_records",
    "tvc_residual",
]
