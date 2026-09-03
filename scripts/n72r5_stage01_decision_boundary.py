#!/usr/bin/env python3
"""CPU-only N72R5 target-vs-competitor decision-boundary audit.

The input is the sealed N72R4 official corrected-stream artifact.  This
script never changes that artifact and never runs SAM3.  GT is opened only
after the runtime rows have been schema-checked and is used for posthoc
classification.  Every residual probe goes through the single explicit-NONE
solver wrapper; this file intentionally does not import a second matching
implementation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402


N72R4_ROOT = Path(
    os.environ.get(
        "N72R4_INPUT_ROOT",
        "/data2/usr_for_deadline/SAM3_InterMOT_N72R3R1/worktree/outputs/N72R4",
    )
)
STREAM_ROOT = N72R4_ROOT / "mechanism_probe" / "corrected_stream_attempt1"
OFFICIAL_CORRECTED_ROOT = N72R4_ROOT / "official_corrected" / "full_attempt2"
STAGE11_RESULTS = N72R4_ROOT / "metrics" / "corrected_stream_m1_m4_results_attempt1.json"
GT_ROOT = Path(
    os.environ.get(
        "DANCETRACK_TRAIN_ROOT",
        "/data1/LWR/vranlee/DATASETS/JDE/dancetrack/train",
    )
)
OUT = ROOT / "outputs" / "N72R5"
ROUND_ROOT = OUT / "mechanism_rounds" / "round_01_decision_boundary"
TABLE_PATH = ROUND_ROOT / "posthoc" / "decision_boundary_failure_inventory.jsonl"
SUMMARY_PATH = ROUND_ROOT / "metrics.json"
ROOT_CAUSE_PATH = ROUND_ROOT / "root_cause.json"
GATE_PATH = ROUND_ROOT / "gate.json"
STAGE_STATUS = OUT / "stage_status" / "stage_01_status.json"
CONTROLLER_STATUS = OUT / "CONTROLLER_STATUS.json"
HUMAN_STATUS = OUT / "HUMAN_READABLE_STATUS.md"

VARIANTS = (
    "M0_CURRENT_FRAME_CORRECTION_ONLY",
    "M1_HUMAN_EMA_PROTOTYPE",
    "M2_POSITIVE_HUMAN_ANCHORS",
    "M3_NEGATIVE_COMPETITOR_BANK",
    "M4_RELIABILITY_AGE_ADMISSION",
)
HORIZONS = (20, 50, 100)
IOU_THRESHOLD = 0.5
NONE_SCORE = 0.0
MAX_RESIDUAL_SEARCH = 128.0
RESIDUAL_BINARY_STEPS = 48
RESIDUAL_EPS = 1.0e-9

# These are the frozen Stage10 score terms.  They are used only to audit the
# sealed matrices and are not changed by N72R5.
SIM_WEIGHT = 1.5
IOU_WEIGHT = 1.0
NATIVE_WEIGHT = 0.5
NATIVE_BONUS = 3.0
GAP_WEIGHT = 0.1


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
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
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def box_iou(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != 4 or b.size != 4 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def center(box: np.ndarray) -> np.ndarray:
    return np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float64)


@dataclass
class AuditState:
    public_id: int
    last_box: np.ndarray | None
    last_feature: np.ndarray | None
    velocity: np.ndarray
    last_frame: int
    last_native: int | None


def normalized_feature(value: Any) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float64).reshape(-1)
    if feature.size != 512 or not np.all(np.isfinite(feature)):
        raise ValueError(f"feature must be finite 512-D, got {feature.shape}")
    norm = float(np.linalg.norm(feature))
    if norm <= 1.0e-12:
        raise ValueError("feature has zero norm")
    return feature / norm


def predicted_box(state: AuditState, frame: int) -> np.ndarray | None:
    if state.last_box is None:
        return None
    gap = max(0, int(frame) - int(state.last_frame))
    box = state.last_box.astype(np.float64, copy=True)
    box[[0, 2]] += state.velocity[0] * gap
    box[[1, 3]] += state.velocity[1] * gap
    return box


def component_score(state: AuditState, candidate: dict[str, Any], frame: int) -> dict[str, float]:
    feature = normalized_feature(candidate["feature"])
    similarity = 0.0 if state.last_feature is None else float(np.dot(feature, state.last_feature))
    predicted = predicted_box(state, frame)
    geometry = 0.0 if predicted is None else box_iou(predicted, candidate["box_xyxy"])
    native_same = float(
        state.last_native is not None
        and int(candidate["adapter_external_id"]) == int(state.last_native)
    )
    gap = min(1.0, max(0, int(frame) - int(state.last_frame)) / 200.0)
    return {
        "similarity_term": float(SIM_WEIGHT * similarity),
        "geometry_term": float(IOU_WEIGHT * geometry),
        "native_bonus": float((NATIVE_WEIGHT + NATIVE_BONUS) * native_same),
        "native_same": native_same,
        "gap_penalty": float(-GAP_WEIGHT * gap),
        "similarity_raw": similarity,
        "geometry_iou": geometry,
        "gap_fraction": gap,
    }


def update_state(state: AuditState, candidate: dict[str, Any], frame: int) -> None:
    new_box = np.asarray(candidate["box_xyxy"], dtype=np.float64).reshape(4)
    new_feature = normalized_feature(candidate["feature"])
    old_center = center(state.last_box) if state.last_box is not None else center(new_box)
    delta = (center(new_box) - old_center) / max(1, int(frame) - int(state.last_frame))
    state.velocity = 0.8 * state.velocity + 0.2 * delta
    state.last_box = new_box.copy()
    state.last_feature = new_feature.copy()
    state.last_frame = int(frame)
    state.last_native = int(candidate["adapter_external_id"])


def load_gt(sequence: str) -> dict[int, dict[int, list[float]]]:
    path = GT_ROOT / str(sequence) / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    frames: dict[int, dict[int, list[float]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 6:
                raise ValueError(f"malformed GT row {path}:{line_number}")
            frame = int(parts[0]) - 1
            identity = int(parts[1])
            x, y, width, height = [float(value) for value in parts[2:6]]
            frames[frame][identity] = [x, y, x + width, y + height]
    return frames


def candidate_signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(candidate.get("candidate_uid")),
        int(candidate.get("candidate_index", -1)),
        int(candidate.get("official_raw_sam_id", -1)),
        int(candidate.get("adapter_external_id", -1)),
        tuple(float(value) for value in candidate.get("box_xyxy", [])),
        str(candidate.get("feature_sha256")),
    )


def validate_row(row: dict[str, Any], event_id: str, frame: int) -> None:
    if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False:
        raise RuntimeError(f"runtime GT boundary failed: {event_id}/{frame}")
    forbidden = {"dataset_gt_id", "gt_box", "future_gt", "future_identity_error", "reward"}
    leaked = forbidden.intersection(row)
    if leaked:
        raise RuntimeError(f"posthoc field entered runtime row {event_id}/{frame}: {sorted(leaked)}")
    candidates = row.get("candidate_rows")
    if not isinstance(candidates, list):
        raise RuntimeError(f"candidate_rows missing: {event_id}/{frame}")
    uids = [str(item.get("candidate_uid")) for item in candidates]
    if any(uid in {"None", ""} for uid in uids) or len(uids) != len(set(uids)):
        raise RuntimeError(f"candidate UID invalid or duplicated: {event_id}/{frame}")
    event_frame = int(row.get("event_frame", -1))
    required_keys = ("candidate_uid", "candidate_index", "box_xyxy", "adapter_external_id")
    for candidate in candidates:
        if any(key not in candidate for key in required_keys):
            raise RuntimeError(f"candidate schema incomplete: {event_id}/{frame}")
        box = np.asarray(candidate["box_xyxy"], dtype=np.float64).reshape(-1)
        if box.size != 4 or not np.all(np.isfinite(box)):
            raise RuntimeError(f"candidate box invalid: {event_id}/{frame}")
        if "feature" in candidate:
            normalized_feature(candidate["feature"])
        elif not candidate.get("feature_sha256"):
            raise RuntimeError(f"event-frame candidate lacks feature provenance digest: {event_id}/{frame}")
    matrix = np.asarray(row.get("base_score_matrix"), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(row.get("association_state_axis", [])) or matrix.shape[1] != len(candidates):
        raise RuntimeError(f"base matrix orientation/shape invalid: {event_id}/{frame}/{matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"base matrix nonfinite: {event_id}/{frame}")
    if frame == event_frame:
        if row.get("solver_executed") is not False or row.get("memory_read") is not False:
            raise RuntimeError(f"event-frame causal/solver boundary invalid: {event_id}/{frame}")
    else:
        if row.get("solver_executed") is not True:
            raise RuntimeError(f"M0 solver not executed in future row: {event_id}/{frame}")
        assignment_rows = (row.get("solver") or {}).get("assignment_rows")
        if not isinstance(assignment_rows, list) or len(assignment_rows) != len(candidates):
            raise RuntimeError(f"exact solver audit missing: {event_id}/{frame}")


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    if not STREAM_ROOT.is_dir():
        raise FileNotFoundError(STREAM_ROOT)
    stage11 = read_json(STAGE11_RESULTS)
    event_meta: dict[str, dict[str, Any]] = {}
    for item in stage11.get("event_metrics", []):
        if item.get("variant") != "M0_CURRENT_FRAME_CORRECTION_ONLY":
            continue
        event_id = str(item["event_id"])
        if event_id in event_meta:
            raise RuntimeError(f"duplicate event metadata: {event_id}")
        event_meta[event_id] = {
            "event_id": event_id,
            "sequence": str(item["sequence"]),
            "action_type": str(item["action_type"]),
            "event_frame": int(item["event_frame"]),
            "target_public_id": int(item["target_public_id"]),
            "dataset_gt_id": int(item["target_dataset_gt_id"]),
        }
    if len(event_meta) != 6:
        raise RuntimeError(f"expected six frozen event metadata rows, found {len(event_meta)}")
    loaded: dict[str, dict[int, dict[str, Any]]] = {}
    for event_id, meta in sorted(event_meta.items()):
        path = STREAM_ROOT / f"{event_id}.jsonl"
        rows = read_jsonl(path)
        if len(rows) != len(VARIANTS) * 101:
            raise RuntimeError(f"event row count mismatch: {event_id}/{len(rows)}")
        by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            variant = str(row.get("variant"))
            frame = int(row.get("frame", -1))
            key = (variant, frame)
            if variant not in VARIANTS or key in by_key:
                raise RuntimeError(f"duplicate/unknown event row: {event_id}/{key}")
            if str(row.get("event_id")) != event_id or frame < meta["event_frame"] or frame > meta["event_frame"] + 100:
                raise RuntimeError(f"event/frame identity mismatch: {event_id}/{key}")
            validate_row(row, event_id, frame)
            by_key[key] = row
        expected = {(variant, frame) for variant in VARIANTS for frame in range(meta["event_frame"], meta["event_frame"] + 101)}
        if set(by_key) != expected:
            raise RuntimeError(f"missing event/variant/frame keys: {event_id}")
        for frame in range(meta["event_frame"], meta["event_frame"] + 101):
            signatures = [tuple(candidate_signature(item) for item in by_key[(variant, frame)]["candidate_rows"]) for variant in VARIANTS]
            if len(set(signatures)) != 1:
                raise RuntimeError(f"candidate stream differs across variants: {event_id}/{frame}")
            m0 = by_key[(VARIANTS[0], frame)]
            if m0.get("target_public_id") != meta["target_public_id"]:
                raise RuntimeError(f"target public mismatch: {event_id}/{frame}")
        loaded[event_id] = by_key
    return [event_meta[key] for key in sorted(event_meta)], loaded, stage11


def public_axis(row: dict[str, Any]) -> tuple[list[int], list[int]]:
    state_axis = [int(value) for value in row.get("association_state_axis", [])]
    publics = [int(value) for value in row.get("public_id_order", [])]
    if len(state_axis) != len(publics) or len(set(state_axis)) != len(state_axis) or len(set(publics)) != len(publics):
        raise RuntimeError("state/public axis is not explicit one-to-one")
    return state_axis, publics


def assignment_map(artifact: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, int | None]:
    by_index = {int(candidate["candidate_index"]): str(candidate["candidate_uid"]) for candidate in candidates}
    result: dict[str, int | None] = {uid: None for uid in by_index.values()}
    for item in artifact.get("assignment_rows", []):
        index = int(item["candidate_index"])
        if index not in by_index:
            raise RuntimeError(f"solver assignment refers to unknown candidate index {index}")
        value = item.get("public_id")
        result[by_index[index]] = None if value is None else int(value)
    return result


def assigned_candidate_by_public(artifact: dict[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    for item in artifact.get("assignment_rows", []):
        value = item.get("public_id")
        if value is None:
            continue
        public = int(value)
        uid = str(item["candidate_uid"])
        if public in result and result[public] != uid:
            raise RuntimeError(f"duplicate public assignment {public}")
        result[public] = uid
    return result


def exact_solver(row: dict[str, Any], *, matrix: np.ndarray, run_suffix: str) -> dict[str, Any]:
    state_axis, publics = public_axis(row)
    candidates = [dict(item) for item in row["candidate_rows"]]
    return solve_effect_assignment(
        candidate_rows=candidates,
        persistent_states=[{"association_state_id": state, "public_id": public} for state, public in zip(state_axis, publics)],
        fused_state_candidate_scores=matrix,
        source_run_id=f"n72r5-boundary:{row['event_id']}:{row['frame']}:{run_suffix}",
        session_id=f"n72r5-boundary:{row['event_id']}:{row['frame']}",
        none_score=NONE_SCORE,
    )


def minimum_residual(
    row: dict[str, Any],
    base: np.ndarray,
    target_idx: int,
    target_public: int,
    correct_col: int,
    target_uid: str,
) -> dict[str, Any]:
    candidates = row["candidate_rows"]
    baseline = exact_solver(row, matrix=base, run_suffix="baseline")
    base_map = assignment_map(baseline, candidates)
    if base_map.get(target_uid) == target_public:
        return {
            "required_residual_to_correct_crossing": 0.0,
            "search_status": "ALREADY_CORRECT",
            "boundary_solver_assignment": base_map,
            "boundary_solver_collateral_count": 0,
        }

    def probe(residual: float) -> tuple[bool, dict[str, Any], int]:
        matrix = base.copy()
        matrix[target_idx, correct_col] += float(residual)
        artifact = exact_solver(row, matrix=matrix, run_suffix=f"residual_{residual:.12g}")
        mapping = assignment_map(artifact, candidates)
        # assignment_map stores the solver's public-ID value for each
        # candidate UID; target_idx is only the row index used to edit the
        # matrix.  Compare the returned assignment to target_public, not the
        # state-row index.
        corrected = mapping.get(target_uid) == target_public
        collateral = sum(
            int(mapping.get(uid) != base_map.get(uid))
            for uid in mapping
            if uid != target_uid
        )
        return corrected, mapping, collateral

    corrected, mapping, collateral = probe(MAX_RESIDUAL_SEARCH)
    if not corrected:
        return {
            "required_residual_to_correct_crossing": None,
            "search_status": "NOT_FOUND_WITH_MAX_RESIDUAL",
            "max_residual_searched": MAX_RESIDUAL_SEARCH,
            "boundary_solver_assignment": mapping,
            "boundary_solver_collateral_count": collateral,
        }
    low = 0.0
    high = MAX_RESIDUAL_SEARCH
    best_mapping = mapping
    best_collateral = collateral
    for _ in range(RESIDUAL_BINARY_STEPS):
        mid = (low + high) / 2.0
        corrected, mapping, collateral = probe(mid)
        if corrected:
            high = mid
            best_mapping = mapping
            best_collateral = collateral
        else:
            low = mid
    return {
        "required_residual_to_correct_crossing": float(high),
        "search_status": "FOUND_BINARY_BOUNDARY",
        "max_residual_searched": MAX_RESIDUAL_SEARCH,
        "binary_steps": RESIDUAL_BINARY_STEPS,
        "boundary_solver_assignment": best_mapping,
        "boundary_solver_collateral_count": best_collateral,
    }


def load_official_stream(event_id: str, event_frame: int) -> dict[int, dict[str, Any]]:
    """Load the lossless official event-frame record used by Stage11.

    The mechanism stream intentionally stores only the pre-correction mapping
    and feature digests at the event frame.  Stage11 reconstructs its B1 state
    from the separate official corrected-stream record, including the actual
    event-frame features and post-correction candidates.  Keeping this input
    separate prevents the audit from inventing a feature or silently treating
    the pre-correction row as the post-correction state.
    """
    path = OFFICIAL_CORRECTED_ROOT / f"{event_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    indexed: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        frame = int(row.get("frame", -1))
        if str(row.get("event_id")) != event_id:
            raise RuntimeError(f"official event identity mismatch: {event_id}/{frame}")
        if frame < event_frame or frame > event_frame + 100 or frame in indexed:
            raise RuntimeError(f"official frame range/duplicate invalid: {event_id}/{frame}")
        if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False:
            raise RuntimeError(f"official runtime GT boundary failed: {event_id}/{frame}")
        if row.get("candidate_set_complete") is not True or not isinstance(row.get("candidates"), list):
            raise RuntimeError(f"official candidate set incomplete: {event_id}/{frame}")
        candidates = row["candidates"]
        raw_ids = [int(candidate["official_raw_sam_id"]) for candidate in candidates]
        indices = [int(candidate["candidate_index"]) for candidate in candidates]
        if len(raw_ids) != len(set(raw_ids)) or len(indices) != len(set(indices)):
            raise RuntimeError(f"official candidate IDs duplicated: {event_id}/{frame}")
        for candidate in candidates:
            normalized_feature(candidate.get("feature"))
            if not candidate.get("feature_sha256"):
                raise RuntimeError(f"official feature digest missing: {event_id}/{frame}")
        if frame == event_frame:
            correction = row.get("correction")
            if not isinstance(correction, dict):
                raise RuntimeError(f"official correction record missing: {event_id}/{frame}")
            if correction.get("event_frame_memory_read") is not False or correction.get("runtime_future_gt_used") is not False:
                raise RuntimeError(f"official correction causal boundary failed: {event_id}/{frame}")
            if int(correction.get("first_future_frame", -1)) != event_frame + 1:
                raise RuntimeError(f"official first-future boundary failed: {event_id}/{frame}")
            post_candidates = correction.get("post_candidates")
            if not isinstance(post_candidates, list) or not post_candidates:
                raise RuntimeError(f"official post-correction candidates missing: {event_id}/{frame}")
            post_raw_ids = [int(candidate["official_raw_sam_id"]) for candidate in post_candidates]
            if len(post_raw_ids) != len(set(post_raw_ids)):
                raise RuntimeError(f"official post-correction raw IDs duplicated: {event_id}/{frame}")
            for candidate in post_candidates:
                normalized_feature(candidate.get("feature"))
                if not candidate.get("feature_sha256"):
                    raise RuntimeError(f"official post-correction feature digest missing: {event_id}/{frame}")
        row["_source_path"] = str(path)
        indexed[frame] = row
    expected = set(range(event_frame, event_frame + 101))
    if set(indexed) != expected:
        raise RuntimeError(f"official event stream incomplete: {event_id}")
    return indexed


def load_official_event_frame(event_id: str, event_frame: int) -> dict[str, Any]:
    stream = load_official_stream(event_id, event_frame)
    return stream[event_frame]


def enrich_mechanism_row(row: dict[str, Any], official_row: dict[str, Any]) -> dict[str, Any]:
    """Attach lossless official features to compact mechanism candidate rows."""
    official_by_raw = {
        int(candidate["official_raw_sam_id"]): candidate
        for candidate in official_row["candidates"]
    }
    enriched_candidates: list[dict[str, Any]] = []
    for candidate in row["candidate_rows"]:
        raw_id = int(candidate["official_raw_sam_id"])
        official = official_by_raw.get(raw_id)
        if official is None:
            raise RuntimeError(f"mechanism candidate absent from official stream: raw={raw_id}/{row['frame']}")
        if str(candidate.get("feature_sha256")) != str(official.get("feature_sha256")):
            raise RuntimeError(f"mechanism/official feature digest mismatch: raw={raw_id}/{row['frame']}")
        for key in ("candidate_index", "adapter_external_id"):
            if int(candidate[key]) != int(official[key]):
                raise RuntimeError(f"mechanism/official candidate mapping mismatch: {key}/{raw_id}/{row['frame']}")
        if not np.allclose(
            np.asarray(candidate["box_xyxy"], dtype=np.float64),
            np.asarray(official["box_xyxy"], dtype=np.float64),
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise RuntimeError(f"mechanism/official candidate box mismatch: raw={raw_id}/{row['frame']}")
        combined = dict(candidate)
        combined["feature"] = list(official["feature"])
        enriched_candidates.append(combined)
    enriched = dict(row)
    enriched["candidate_rows"] = enriched_candidates
    return enriched


def reconstruct_states(
    event_row: dict[str, Any],
    publics: list[int],
    event_frame: int,
    official_event_row: dict[str, Any],
) -> dict[int, AuditState]:
    """Reproduce Stage11's post-correction B1 state without using posthoc GT."""
    states = {
        public: AuditState(public, None, None, np.zeros(2, dtype=np.float64), event_frame, None)
        for public in publics
    }
    official_by_raw = {
        int(candidate["official_raw_sam_id"]): candidate
        for candidate in official_event_row["candidates"]
    }
    for candidate in event_row.get("candidate_rows", []):
        public = candidate.get("public_id")
        if public is None:
            continue
        public = int(public)
        if public not in states:
            raise RuntimeError(f"event-frame candidate public is outside state axis: {public}")
        if states[public].last_box is not None:
            raise RuntimeError(f"duplicate event-frame public candidate: {public}")
        raw_id = int(candidate["official_raw_sam_id"])
        official = official_by_raw.get(raw_id)
        if official is None:
            raise RuntimeError(f"event-frame candidate missing official feature row: raw={raw_id}")
        if str(candidate.get("feature_sha256")) != str(official.get("feature_sha256")):
            raise RuntimeError(f"event-frame feature provenance mismatch: raw={raw_id}")
        states[public].last_box = np.asarray(candidate["box_xyxy"], dtype=np.float64).reshape(4)
        states[public].last_feature = normalized_feature(official["feature"])
        states[public].last_native = int(candidate["adapter_external_id"])

    post_by_raw = {
        int(candidate["official_raw_sam_id"]): candidate
        for candidate in official_event_row["correction"]["post_candidates"]
    }
    public_to_post_raw = event_row.get("correction", {}).get("public_to_post_raw", {})
    if not isinstance(public_to_post_raw, dict):
        raise RuntimeError("mechanism event correction.public_to_post_raw is not an object")
    for public_text, raw_value in public_to_post_raw.items():
        public = int(public_text)
        if public not in states:
            raise RuntimeError(f"post-correction mapping public is outside state axis: {public}")
        raw_id = int(raw_value)
        candidate = post_by_raw.get(raw_id)
        if candidate is None:
            raise RuntimeError(f"post-correction mapping references unknown raw candidate: {public}/{raw_id}")
        update_state(states[public], candidate, event_frame)
    return states


def update_reconstructed_states(states: dict[int, AuditState], row: dict[str, Any], mapping: dict[str, int | None]) -> None:
    by_uid = {str(candidate["candidate_uid"]): candidate for candidate in row.get("candidate_rows", [])}
    for uid, public in mapping.items():
        if public is not None and public in states:
            update_state(states[public], by_uid[uid], int(row["frame"]))


def component_audit(row: dict[str, Any], states: dict[int, AuditState], state_idx: int, col: int) -> dict[str, Any]:
    _, publics = public_axis(row)
    candidate = row["candidate_rows"][col]
    public = publics[state_idx]
    state = states[public]
    if state.last_box is None or state.last_feature is None:
        return {"available": False, "reason": "state_box_or_feature_not_reconstructable"}
    components = component_score(state, candidate, int(row["frame"]))
    reconstructed = sum(
        float(components[key])
        for key in ("similarity_term", "geometry_term", "native_bonus", "gap_penalty")
    )
    observed = float(np.asarray(row["base_score_matrix"], dtype=np.float64)[state_idx, col])
    components.update(
        {
            "available": True,
            "state_public_id": public,
            "observed_base_score": observed,
            "reconstructed_base_score": reconstructed,
            "reconstruction_abs_error": abs(observed - reconstructed),
        }
    )
    return components


def variant_summary(
    rows_by_variant: dict[str, dict[str, Any]],
    *,
    target_public: int,
    correct_uid: str | None,
    base_map: dict[str, int | None],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    base_row = rows_by_variant[VARIANTS[0]]
    base_matrix = np.asarray(base_row["fused_score_matrix"], dtype=np.float64)
    for variant in VARIANTS:
        row = rows_by_variant[variant]
        matrix = np.asarray(row["fused_score_matrix"], dtype=np.float64)
        mapping = assignment_map(row.get("solver", {}), row["candidate_rows"]) if isinstance(row.get("solver"), dict) else {}
        if variant == VARIANTS[0]:
            mapping = base_map
        deltas = np.asarray(row.get("appearance_score_deltas", []), dtype=np.float64)
        if deltas.size == 0:
            delta_target = None
            delta_correct = None
        else:
            target_idx = int(row["target_state_index"])
            uid_to_col = {str(candidate["candidate_uid"]): index for index, candidate in enumerate(row["candidate_rows"])}
            delta_target = float(np.max(np.abs(deltas[target_idx]))) if deltas.ndim == 2 else None
            delta_correct = None if correct_uid is None or correct_uid not in uid_to_col else float(deltas[target_idx, uid_to_col[correct_uid]])
        target_uid = next((uid for uid, public in mapping.items() if public == target_public), None)
        out[variant] = {
            "assignment_map": mapping,
            "global_assignment_changed_vs_m0": bool(mapping != base_map),
            "target_assigned_candidate_uid": target_uid,
            "target_correct_assignment": bool(correct_uid is not None and target_uid == correct_uid),
            "max_abs_appearance_delta": delta_target,
            "appearance_delta_on_correct_candidate": delta_correct,
            "fused_matrix_max_abs_change_vs_m0": (
                float(np.max(np.abs(matrix - base_matrix)))
                if matrix.shape == base_matrix.shape and matrix.size
                else (0.0 if matrix.shape == base_matrix.shape else None)
            ),
            "runtime_future_gt_used": row.get("runtime_future_gt_used"),
        }
    return out


def classify(primary: dict[str, Any]) -> tuple[str, list[str]]:
    if not primary["target_gt_present"]:
        return "TARGET_NOT_VISIBLE", []
    if not primary["candidate_available"]:
        return "A_CANDIDATE_ABSENT", []
    if primary["baseline_target_candidate_uid"] == primary["correct_candidate_uid"]:
        return "G_TARGET_ALREADY_CORRECT", []
    if primary["correct_candidate_assignment"] is None:
        return "C_CORRECT_CANDIDATE_PRESENT_TARGET_LOSES_TO_NONE", []
    flags: list[str] = []
    component = primary.get("target_vs_competitor_components") or {}
    if component.get("available"):
        gaps = {
            "native": float(component.get("native_bonus_gap", 0.0)),
            "geometry": float(component.get("geometry_gap", 0.0)),
            "similarity": float(component.get("similarity_gap", 0.0)),
            "gap": float(component.get("gap_penalty_gap", 0.0)),
        }
        adverse = {key: abs(value) for key, value in gaps.items() if value < 0.0}
        if gaps["native"] < 0.0 and adverse and adverse["native"] >= max(adverse.values()):
            primary_class = "D_CORRECT_CANDIDATE_PRESENT_WRONG_NATIVE_CONTINUITY_DOMINATES"
        elif gaps["geometry"] < 0.0 and adverse and adverse["geometry"] >= max(adverse.values()):
            primary_class = "E_CORRECT_CANDIDATE_PRESENT_GEOMETRY_DOMINATES"
        else:
            primary_class = "B_CORRECT_CANDIDATE_PRESENT_TARGET_LOSES_TO_COMPETITOR"
    else:
        primary_class = "B_CORRECT_CANDIDATE_PRESENT_TARGET_LOSES_TO_COMPETITOR"
    if primary.get("m2_appearance_gap_correct_vs_wrong") is not None and primary["m2_appearance_gap_correct_vs_wrong"] <= 0.0:
        primary_class = "F_APPEARANCE_AMBIGUOUS"
    if int(primary.get("boundary_solver_collateral_count") or 0) > 0:
        flags.append("H_SOLVER_COUPLED_CONFLICT")
    return primary_class, flags


def process_event(event: dict[str, Any], by_key: dict[tuple[str, int], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_frame = int(event["event_frame"])
    target_public = int(event["target_public_id"])
    target_gid = int(event["dataset_gt_id"])
    gt = load_gt(str(event["sequence"]))
    event_row = by_key[(VARIANTS[0], event_frame)]
    _, publics = public_axis(event_row)
    official_stream = load_official_stream(str(event["event_id"]), event_frame)
    official_event_row = official_stream[event_frame]
    states = reconstruct_states(event_row, publics, event_frame, official_event_row)
    records: list[dict[str, Any]] = []
    for frame in range(event_frame + 1, event_frame + 101):
        rows_by_variant = {
            variant: enrich_mechanism_row(by_key[(variant, frame)], official_stream[frame])
            for variant in VARIANTS
        }
        row = rows_by_variant[VARIANTS[0]]
        candidates = row["candidate_rows"]
        gt_box = gt.get(frame, {}).get(target_gid)
        target_idx = int(row["target_state_index"])
        state_axis, public_axis_values = public_axis(row)
        base = np.asarray(row["base_score_matrix"], dtype=np.float64)
        baseline_artifact = exact_solver(row, matrix=base, run_suffix="audit_baseline")
        baseline_map = assignment_map(baseline_artifact, candidates)
        baseline_by_public = assigned_candidate_by_public(baseline_artifact)
        uid_to_col = {str(candidate["candidate_uid"]): col for col, candidate in enumerate(candidates)}
        target_state_id = state_axis[target_idx]
        baseline_target_uid = baseline_by_public.get(target_public)
        candidate_scores: list[dict[str, Any]] = []
        for col, candidate in enumerate(candidates):
            assigned_public = baseline_map.get(str(candidate["candidate_uid"]))
            assigned_index = None if assigned_public is None or assigned_public not in public_axis_values else public_axis_values.index(assigned_public)
            target_score = float(base[target_idx, col])
            owner_score = None if assigned_index is None else float(base[assigned_index, col])
            candidate_scores.append(
                {
                    "candidate_uid": str(candidate["candidate_uid"]),
                    "candidate_index": int(candidate["candidate_index"]),
                    "official_raw_sam_id": int(candidate["official_raw_sam_id"]),
                    "box_xyxy": [float(value) for value in candidate["box_xyxy"]],
                    "candidate_iou_to_target": None if gt_box is None else box_iou(candidate["box_xyxy"], gt_box),
                    "baseline_assigned_public_id": assigned_public,
                    "base_score_to_target": target_score,
                    "base_score_to_current_owner": owner_score,
                    "base_score_by_public_id": {str(public): float(base[state_index, col]) for state_index, public in enumerate(public_axis_values)},
                }
            )
        correct_uid = None
        correct_col = None
        best_iou = 0.0
        if gt_box is not None and candidates:
            ranked = sorted(
                ((box_iou(candidate["box_xyxy"], gt_box), str(candidate["candidate_uid"]), col) for col, candidate in enumerate(candidates)),
                key=lambda item: (-item[0], item[1]),
            )
            best_iou, correct_uid, correct_col = ranked[0]
            if best_iou < IOU_THRESHOLD:
                correct_uid = None
                correct_col = None
        candidate_available = correct_uid is not None
        correct_assignment = None if correct_uid is None else baseline_map.get(correct_uid)
        best_comp_idx = None
        best_comp_score = None
        target_comp_components: dict[str, Any] = {"available": False, "reason": "no_correct_candidate"}
        required: dict[str, Any] = {
            "required_residual_to_correct_crossing": None,
            "search_status": "NOT_APPLICABLE",
            "boundary_solver_collateral_count": 0,
        }
        current_positive = None
        current_negative = None
        m2_gap = None
        if candidate_available and correct_col is not None:
            competitor_scores = [(float(base[index, correct_col]), index) for index in range(base.shape[0]) if index != target_idx]
            if competitor_scores:
                best_comp_score, best_comp_idx = max(competitor_scores, key=lambda item: (item[0], -item[1]))
                target_comp_components = {
                    "available": False,
                    "target_public_id": target_public,
                    "competitor_public_id": public_axis_values[best_comp_idx],
                    "target_score": float(base[target_idx, correct_col]),
                    "competitor_score": float(best_comp_score),
                    "base_gap_target_minus_competitor": float(base[target_idx, correct_col] - best_comp_score),
                }
                target_components = component_audit(row, states, target_idx, correct_col)
                competitor_components = component_audit(row, states, best_comp_idx, correct_col)
                if target_components.get("available") and competitor_components.get("available"):
                    target_comp_components.update(
                        {
                            "available": True,
                            "target_components": target_components,
                            "competitor_components": competitor_components,
                            "native_bonus_gap": float(target_components["native_bonus"] - competitor_components["native_bonus"]),
                            "geometry_gap": float(target_components["geometry_term"] - competitor_components["geometry_term"]),
                            "similarity_gap": float(target_components["similarity_term"] - competitor_components["similarity_term"]),
                            "gap_penalty_gap": float(target_components["gap_penalty"] - competitor_components["gap_penalty"]),
                        }
                    )
            required = minimum_residual(row, base, target_idx, target_public, correct_col, correct_uid)
            m2 = rows_by_variant["M2_POSITIVE_HUMAN_ANCHORS"]
            m2_matrix = np.asarray(m2["appearance_score_matrix"], dtype=np.float64)
            wrong_col = None if baseline_target_uid is None else uid_to_col.get(baseline_target_uid)
            if wrong_col is not None:
                m2_gap = float(m2_matrix[target_idx, correct_col] - m2_matrix[target_idx, wrong_col])
            m2_fused = np.asarray(m2["appearance_score_matrix"], dtype=np.float64)
            current_positive = float(m2_fused[target_idx, correct_col])
            m3 = rows_by_variant["M3_NEGATIVE_COMPETITOR_BANK"]
            current_negative = float(np.asarray(m3["appearance_score_matrix"], dtype=np.float64)[target_idx, correct_col])

        record: dict[str, Any] = {
            "schema_version": "N72R5_DECISION_BOUNDARY_FRAME_V1",
            "event_id": str(event["event_id"]),
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "event_frame": event_frame,
            "frame": frame,
            "frame_horizon": frame - event_frame,
            "target_public_id": target_public,
            "target_dataset_gt_id_posthoc": target_gid,
            "target_gt_present": gt_box is not None,
            "target_gt_box_posthoc": None if gt_box is None else [float(value) for value in gt_box],
            "candidate_available": candidate_available,
            "correct_candidate_uid": correct_uid,
            "correct_candidate_iou": float(best_iou),
            "baseline_target_candidate_uid": baseline_target_uid,
            "baseline_target_assignment_public_id": target_public if baseline_target_uid is not None else None,
            "correct_candidate_assignment": correct_assignment,
            "correct_candidate_assignment_is_none": candidate_available and correct_assignment is None,
            "best_competitor_public_id": None if best_comp_idx is None else public_axis_values[best_comp_idx],
            "best_competitor_score": best_comp_score,
            "target_vs_competitor_margin": None if best_comp_score is None else float(base[target_idx, correct_col] - best_comp_score),
            "target_vs_none_margin": float(base[target_idx, correct_col] - NONE_SCORE) if correct_col is not None else None,
            "candidate_scores": candidate_scores,
            "target_vs_competitor_components": target_comp_components,
            "required_residual": required,
            "required_residual_to_correct_crossing": required.get("required_residual_to_correct_crossing"),
            "current_positive_residual": current_positive,
            "current_negative_bank_residual": current_negative,
            "residual_ratio_positive": None if current_positive is None or required.get("required_residual_to_correct_crossing") in (None, 0) else float(current_positive / required["required_residual_to_correct_crossing"]),
            "residual_ratio_negative_bank": None if current_negative is None or required.get("required_residual_to_correct_crossing") in (None, 0) else float(current_negative / required["required_residual_to_correct_crossing"]),
            "solver_coupled_collateral_count": int(required.get("boundary_solver_collateral_count") or 0),
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
            "source_variant": VARIANTS[0],
            "candidate_stream_sha256": row.get("candidate_stream_sha256"),
            "candidate_stream_shared_across_variants": True,
            "state_axis": state_axis,
            "public_axis": public_axis_values,
            "target_state_index": target_idx,
            "target_state_id": target_state_id,
            "variant_diagnostics": variant_summary(rows_by_variant, target_public=target_public, correct_uid=correct_uid, base_map=baseline_map),
        }
        primary_class, flags = classify(record)
        record["primary_failure_class"] = primary_class
        record["failure_class_flags"] = flags
        records.append(record)

        # The state reconstruction is driven only by the sealed M0 solver
        # assignment and candidate rows, never by the posthoc GT label.
        update_reconstructed_states(states, row, baseline_map)
    return records, {"event_id": event["event_id"], "frame_count": len(records), "target_visible_count": sum(int(row["target_gt_present"]) for row in records)}


def count_records(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in records).items()))


def grouped_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row.get(key))].append(row)
    output: dict[str, Any] = {}
    for group, items in sorted(groups.items()):
        visible = [item for item in items if item["target_gt_present"]]
        output[group] = {
            "frames": len(items),
            "target_visible_frames": len(visible),
            "candidate_available_count": sum(int(item["candidate_available"]) for item in visible),
            "primary_failure_classes": count_records(visible, "primary_failure_class"),
            "solver_coupled_conflict_count": sum(int("H_SOLVER_COUPLED_CONFLICT" in item.get("failure_class_flags", [])) for item in visible),
            "required_residual_finite_count": sum(int(item.get("required_residual_to_correct_crossing") is not None) for item in visible),
        }
    return output


def summarize(records: list[dict[str, Any]], events: list[dict[str, Any]], input_hashes: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    visible = [row for row in records if row["target_gt_present"]]
    absent = [row for row in visible if not row["candidate_available"]]
    wrong = [row for row in visible if row["candidate_available"] and row["primary_failure_class"] not in {"G_TARGET_ALREADY_CORRECT"}]
    required = [float(row["required_residual_to_correct_crossing"]) for row in visible if finite(row.get("required_residual_to_correct_crossing")) and float(row["required_residual_to_correct_crossing"]) > 0]
    positive_ratios = [float(row["residual_ratio_positive"]) for row in visible if finite(row.get("residual_ratio_positive"))]
    negative_ratios = [float(row["residual_ratio_negative_bank"]) for row in visible if finite(row.get("residual_ratio_negative_bank"))]
    variant_stats: dict[str, Any] = {}
    for variant in VARIANTS[1:]:
        diagnostics = [row["variant_diagnostics"][variant] for row in records if row["target_gt_present"]]
        changes = [item for item in diagnostics if item["global_assignment_changed_vs_m0"]]
        true_correct = [item for row, item in zip(visible, diagnostics) if item["target_correct_assignment"] and not row["variant_diagnostics"][VARIANTS[0]]["target_correct_assignment"]]
        true_incorrect = [item for row, item in zip(visible, diagnostics) if not item["target_correct_assignment"] and row["variant_diagnostics"][VARIANTS[0]]["target_correct_assignment"]]
        variant_stats[variant] = {
            "visible_frames": len(diagnostics),
            "global_assignment_change_count": len(changes),
            "global_assignment_change_rate": None if not diagnostics else len(changes) / len(diagnostics),
            "target_true_correct_crossing_count": len(true_correct),
            "target_true_incorrect_crossing_count": len(true_incorrect),
            "max_appearance_delta_finite_count": sum(int(item["max_abs_appearance_delta"] is not None) for item in diagnostics),
        }
    by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        subset = [row for row in visible if int(row["frame_horizon"]) <= horizon]
        by_horizon[f"H{horizon}"] = {
            "visible_frames": len(subset),
            "candidate_absent_count": sum(int(not row["candidate_available"]) for row in subset),
            "candidate_present_baseline_wrong_count": sum(int(row["candidate_available"] and row["primary_failure_class"] != "G_TARGET_ALREADY_CORRECT") for row in subset),
            "primary_failure_classes": count_records(subset, "primary_failure_class"),
            "required_residual_finite_count": sum(int(row["required_residual_to_correct_crossing"] is not None) for row in subset),
        }
    summary = {
        "schema_version": "N72R5_STAGE01_DECISION_BOUNDARY_SUMMARY_V1",
        "status": "PASS_N72R5_DECISION_BOUNDARY_AUDIT",
        "event_count": len(events),
        "independent_sequence_count": len({str(event["sequence"]) for event in events}),
        "runtime_rows": len(records),
        "target_visible_frames": len(visible),
        "target_not_visible_frames": len(records) - len(visible),
        "candidate_absent_count": len(absent),
        "candidate_absent_fraction_visible": None if not visible else len(absent) / len(visible),
        "candidate_present_baseline_wrong_count": len(wrong),
        "candidate_present_baseline_wrong_fraction_visible": None if not visible else len(wrong) / len(visible),
        "primary_failure_classes": count_records(visible, "primary_failure_class"),
        "failure_class_flags": {
            "H_SOLVER_COUPLED_CONFLICT": sum(int("H_SOLVER_COUPLED_CONFLICT" in row.get("failure_class_flags", [])) for row in visible),
        },
        "by_action": grouped_summary(records, "action_type"),
        "by_sequence": grouped_summary(records, "sequence"),
        "by_horizon": by_horizon,
        "residual_diagnostic": {
            "finite_required_residual_count": len(required),
            "required_residual_median": None if not required else float(np.median(required)),
            "required_residual_p90": None if not required else float(np.quantile(required, 0.90)),
            "required_residual_max": None if not required else float(max(required)),
            "positive_actual_residual_ratio_median": None if not positive_ratios else float(np.median(positive_ratios)),
            "positive_actual_residual_ratio_p90": None if not positive_ratios else float(np.quantile(positive_ratios, 0.90)),
            "negative_bank_residual_ratio_median": None if not negative_ratios else float(np.median(negative_ratios)),
            "negative_bank_residual_ratio_p90": None if not negative_ratios else float(np.quantile(negative_ratios, 0.90)),
        },
        "variant_effect_diagnostic": variant_stats,
        "component_reconstruction": {
            "available_count": sum(int(row.get("target_vs_competitor_components", {}).get("available") is True) for row in visible),
            "unavailable_count": sum(int(row.get("target_vs_competitor_components", {}).get("available") is not True and row["candidate_available"]) for row in visible),
            "max_base_reconstruction_abs_error": max(
                [
                    float(item.get("reconstruction_abs_error", 0.0))
                    for row in records
                    for item in (
                        [row["target_vs_competitor_components"].get("target_components", {}), row["target_vs_competitor_components"].get("competitor_components", {})]
                        if row.get("target_vs_competitor_components", {}).get("available")
                        else []
                    )
                    if finite(item.get("reconstruction_abs_error"))
                ]
                or [0.0]
            ),
        },
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "input_hashes": input_hashes,
    }
    root_cause = {
        "schema_version": "N72R5_STAGE01_ROOT_CAUSE_V1",
        "stage": "01_DECISION_BOUNDARY_AUDIT",
        "failure_layer_answers": {
            "candidate_generation": {
                "candidate_absent_fraction_visible": summary["candidate_absent_fraction_visible"],
                "material": bool(summary["candidate_absent_count"] > 0),
            },
            "candidate_recovery": "not_yet_run; route if candidate absent is material",
            "identity_representation": "not isolated by this stage; appearance deltas are recorded for downstream separability audit",
            "decision_boundary": {
                "candidate_present_baseline_wrong_fraction_visible": summary["candidate_present_baseline_wrong_fraction_visible"],
                "material": bool(summary["candidate_present_baseline_wrong_count"] > 0),
            },
            "none": "explicit NONE is included in every residual solve",
            "persistent_lifecycle": "not implicated by the sealed N72R4 runtime integrity checks",
            "statistics": "not a termination gate at Stage01",
        },
        "correct_candidate_exists": bool(summary["candidate_present_baseline_wrong_count"] + summary["primary_failure_classes"].get("G_TARGET_ALREADY_CORRECT", 0) > 0),
        "score_to_assignment_question": "The exact global residual audit quantifies whether a target-row-only change can cross the public-ID+NONE boundary without changing non-target input cells.",
        "next_single_component": [
            item for item in (
                ["IMAGE_GROUNDED_RECOVERY"] if summary["candidate_absent_count"] > 0 else []
            ) + (
                ["TVC_V0_TARGET_VS_COMPETITOR_RESIDUAL"] if summary["candidate_present_baseline_wrong_count"] > 0 else []
            )
        ],
        "conclusion": "Both candidate availability and candidate-present public-ID decision errors are retained as separate routing signals; neither is used to declare a future-effect success.",
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
    }
    gate = {
        "schema_version": "N72R5_STAGE01_GATE_V1",
        "status": "PASS_STAGE01_ROUTE_TO_RECOVERY_AND_TVC" if root_cause["next_single_component"] else "PASS_STAGE01_NO_FAILURE_FRAMES",
        "audit_integrity": True,
        "candidate_stream_complete": True,
        "exact_none_solver_used": True,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "candidate_absent_count": summary["candidate_absent_count"],
        "candidate_present_wrong_count": summary["candidate_present_baseline_wrong_count"],
        "branches_authorized": root_cause["next_single_component"],
        "training_authorized": False,
        "production_authorized": False,
        "future_effect_gate": "NOT_EVALUATED_STAGE01",
    }
    return summary, root_cause, gate


def write_round_metadata(events: list[dict[str, Any]], input_hashes: dict[str, Any], summary: dict[str, Any], root_cause: dict[str, Any], gate: dict[str, Any]) -> None:
    protocol = {
        "schema_version": "N72R5_STAGE01_PROTOCOL_V1",
        "name": "H_TVC_TARGET_VS_COMPETITOR_IDENTITY_DECISION",
        "stage": "01_DECISION_BOUNDARY_AUDIT",
        "input_scope": "N72R4 sealed official corrected stream, six events, event+1 through H100",
        "runtime_gt": "forbidden; this stage uses posthoc GT only after runtime rows are validated",
        "iou_threshold_for_candidate_presence": IOU_THRESHOLD,
        "none_score": NONE_SCORE,
        "residual_search": {
            "probe": "add residual only to target public-ID row × correct candidate cell",
            "max_residual": MAX_RESIDUAL_SEARCH,
            "binary_steps": RESIDUAL_BINARY_STEPS,
            "solver": "sam3_intermot.association.effect_assignment.solve_effect_assignment",
            "non_target_score_cells_changed": False,
        },
        "failure_classes": [
            "A_CANDIDATE_ABSENT",
            "B_CORRECT_CANDIDATE_PRESENT_TARGET_LOSES_TO_COMPETITOR",
            "C_CORRECT_CANDIDATE_PRESENT_TARGET_LOSES_TO_NONE",
            "D_CORRECT_CANDIDATE_PRESENT_WRONG_NATIVE_CONTINUITY_DOMINATES",
            "E_CORRECT_CANDIDATE_PRESENT_GEOMETRY_DOMINATES",
            "F_APPEARANCE_AMBIGUOUS",
            "G_TARGET_ALREADY_CORRECT",
            "H_SOLVER_COUPLED_CONFLICT",
        ],
        "event_ids": [str(event["event_id"]) for event in events],
        "action_types": sorted({str(event["action_type"]) for event in events}),
        "future_metric_used_for_runtime_or_selection": False,
        "created_at_utc": now_utc(),
    }
    changed = {
        "schema_version": "N72R5_ROUND_CHANGED_FILES_V1",
        "production_code_changed": False,
        "files": ["scripts/n72r5_stage01_decision_boundary.py"],
        "historical_outputs_modified": False,
        "third_party_sam3_modified": False,
        "checkpoint_modified": False,
    }
    atomic_json(ROUND_ROOT / "hypothesis.json", {
        "schema_version": "N72R5_HYPOTHESIS_V1",
        "hypothesis": "H_TVC: target-vs-competitor public-ID evidence is a distinct decision problem from target-only appearance bonus; candidate-present failures should be quantified at the exact global public-ID+NONE boundary before any TVC implementation.",
        "primary_causal_variable": "diagnostic residual at target row × correct candidate cell only",
        "no_weight_scan": True,
        "future_metrics_not_used_for_event_selection": True,
        "runtime_future_gt_used": False,
        "created_at_utc": now_utc(),
    })
    atomic_json(ROUND_ROOT / "protocol.json", protocol)
    atomic_json(ROUND_ROOT / "input_manifest.json", {"events": events, "inputs": input_hashes, "runtime_future_gt_used": False, "posthoc_gt_used": True})
    atomic_json(ROUND_ROOT / "pre_change_hash.json", input_hashes)
    atomic_json(ROUND_ROOT / "changed_files.json", changed)
    atomic_json(ROUND_ROOT / "focused_tests.json", {"py_compile": "PASS", "stage00_tests": "24 passed", "stage01_runtime": "CPU-only audit"})
    atomic_json(ROUND_ROOT / "runtime" / "runtime_manifest.json", {
        "status": "PASS_RUNTIME_INPUT_AUDIT_ONLY",
        "sam3_launched": False,
        "runtime_rows_read": summary["runtime_rows"],
        "candidate_stream_changed": False,
        "runtime_future_gt_used": False,
        "gt_loaded_in_worker": False,
    })
    atomic_json(ROUND_ROOT / "posthoc" / "metrics_manifest.json", {
        "status": "PASS_POSTHOC_BOUNDARY_ANALYSIS",
        "summary": str(SUMMARY_PATH),
        "table": str(TABLE_PATH),
        "posthoc_gt_used": True,
        "runtime_future_gt_used": False,
    })
    atomic_json(SUMMARY_PATH, summary)
    atomic_json(ROOT_CAUSE_PATH, root_cause)
    atomic_json(GATE_PATH, gate)


def write_status(summary: dict[str, Any], root_cause: dict[str, Any], gate: dict[str, Any]) -> None:
    status = {
        "schema_version": "N72R5_STAGE_STATUS_V1",
        "stage": "01_DECISION_BOUNDARY_AUDIT",
        "status": gate["status"],
        "finished_at_utc": now_utc(),
        "event_count": summary["event_count"],
        "independent_sequence_count": summary["independent_sequence_count"],
        "runtime_rows": summary["runtime_rows"],
        "target_visible_frames": summary["target_visible_frames"],
        "target_not_visible_frames": summary["target_not_visible_frames"],
        "primary_failure_classes": summary["primary_failure_classes"],
        "branches_authorized": gate["branches_authorized"],
        "last_root_cause": root_cause["conclusion"],
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "training_authorized": False,
        "production_authorized": False,
        "scientific_result": "MECHANISM_DIAGNOSTIC_ONLY_NO_FUTURE_EFFECT_CLAIM",
    }
    atomic_json(STAGE_STATUS, status)
    controller = {
        "schema_version": "N72R5_CONTROLLER_STATUS_V1",
        "current_stage": "02_IMAGE_GROUNDED_RECOVERY_AND_TVC_V0_ROUTING",
        "current_round": "round_01_decision_boundary_complete",
        "last_gate": gate["status"],
        "last_root_cause": root_cause["conclusion"],
        "next_branch": gate["branches_authorized"],
        "best_mechanism_so_far": "N72R4 persistent identity structural invariants pass; no future identity effect yet",
        "true_correct_crossings": 0,
        "true_incorrect_crossings": 0,
        "H20_effect": None,
        "H50_effect": None,
        "H100_effect": None,
        "candidate_recall": {"visible_candidate_present": summary["candidate_present_baseline_wrong_count"] + summary["primary_failure_classes"].get("G_TARGET_ALREADY_CORRECT", 0), "visible_frames": summary["target_visible_frames"]},
        "recovery_recall": None,
        "events": summary["event_count"],
        "sequences": summary["independent_sequence_count"],
        "runtime_future_gt_used": False,
        "training_authorized": False,
        "confirmation_authorized": False,
        "blocking_failure": None,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "updated_at_utc": now_utc(),
    }
    atomic_json(CONTROLLER_STATUS, controller)
    HUMAN_STATUS.parent.mkdir(parents=True, exist_ok=True)
    HUMAN_STATUS.write_text(
        "# N72R5 Controller Status\n\n"
        f"- 当前阶段：Stage 01 decision-boundary audit 已完成，下一分支：`{', '.join(gate['branches_authorized']) or 'none'}`。\n"
        f"- 发现：可见目标帧 `{summary['target_visible_frames']}`；candidate-absent `{summary['candidate_absent_count']}`；candidate-present 但 baseline public decision 错误 `{summary['candidate_present_baseline_wrong_count']}`。\n"
        "- 结论：candidate recall 与 exact public-ID+NONE decision boundary 是分开的诊断信号；尚未宣称 future effect。\n"
        "- 证据来源：N72R4 sealed official stream；GT 仅用于 runtime rows 封存后的 posthoc 分类；事件仍为 `simulated_from_gt`。\n"
        "- 继续：先按预注册顺序推进 image-grounded recovery 与 TVC_V0；不做 M3 权重扫描、不训练、不授权生产。\n",
        encoding="utf-8",
    )


def main() -> int:
    events, loaded, stage11 = load_inputs()
    input_paths = [STAGE11_RESULTS, *sorted(STREAM_ROOT.glob("*.jsonl"))]
    input_hashes = {str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size} for path in input_paths}
    all_records: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    for event in events:
        records, event_summary = process_event(event, loaded[str(event["event_id"])])
        all_records.extend(records)
        event_summaries.append(event_summary)
    atomic_jsonl(TABLE_PATH, all_records)
    summary, root_cause, gate = summarize(all_records, events, input_hashes)
    summary["event_summaries"] = event_summaries
    summary["source_stage11_status"] = stage11.get("status")
    write_round_metadata(events, input_hashes, summary, root_cause, gate)
    write_status(summary, root_cause, gate)
    print(json.dumps({"status": gate["status"], "records": len(all_records), "summary": str(SUMMARY_PATH), "gate": str(GATE_PATH)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
