#!/usr/bin/env python3
"""Build the N72R11R4 exact-solver-on-policy V3 corpus.

The historical N72R11R3 corpus is used only for the frozen sequence split and
offline labels.  Every runtime candidate pool in this builder is reconstructed
from the sealed C0/secondary artifacts, then follows the same causal order as
formal runtime:

    raw pool -> exact BASE solve -> state-dependent features -> V3 selection
    -> legacy injection -> exact public solve -> shared temporal-state update

No dataset GT is opened until all runtime tensors for an event have been
constructed.  The upstream interaction source is explicitly
``simulated_from_gt`` and is not real-human evidence.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.association.secondary_public_score import (  # noqa: E402
    build_secondary_public_score_frame,
)
from sam3_intermot.association.target_edge_interface import (  # noqa: E402
    apply_legacy_injection,
    select_candidate_from_logits,
)
from sam3_intermot.reacquisition.public_competition_features import (  # noqa: E402
    PUBLIC_COMPETITION_FEATURE_DIM,
    PUBLIC_COMPETITION_FEATURE_SCHEMA,
    build_public_competition_feature,
)
from sam3_intermot.reacquisition.temporal_scorer_adapter import (  # noqa: E402
    PublicCompetitionScorerAdapter,
    V3TemporalScorerAdapter,
)
from sam3_intermot.reacquisition.target_candidate_pool import (  # noqa: E402
    build_candidate_pool_with_future_requery,
    serializable_candidate,
)
from sam3_intermot.reacquisition.target_id_features import (  # noqa: E402
    CANDIDATE_FEATURE_DIM,
    candidate_feature_vector,
)
from sam3_intermot.reacquisition.temporal_state_policy import (  # noqa: E402
    TEMPORAL_FEATURE_SCHEMA,
    build_temporal_features,
    initialize_temporal_state,
    state_audit,
    state_memory_arrays,
    update_temporal_state,
)
from scripts import n72r11_build_causal_corpus as causal  # noqa: E402
from scripts import n72r11_train_v3 as train_v3  # noqa: E402


DEFAULT_SOURCE_ROOT = ROOT / "outputs/N72R11R3/bootstrap_corpus"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/N72R11R4/exact_onpolicy_v3"
DEFAULT_STAGE_PATH = ROOT / "outputs/N72R11R4/stage_02_exact_onpolicy_corpus.json"
HORIZON = 100
SOURCE_FEATURE_DIM = causal.SOURCE_FEATURE_DIM
MEMORY_DIM = 512
RECENT_SLOTS = 4
LONG_TERM_SLOTS = 4
DISTRACTOR_SLOTS = 8


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, content: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        if isinstance(content, bytes):
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
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


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(temporary, **dict(arrays))
        with open(temporary, "rb") as handle:
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
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def device_from(value: str) -> torch.device:
    device = torch.device(str(value))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    if device.type == "cuda" and device.index is not None and int(device.index) >= torch.cuda.device_count():
        raise RuntimeError(f"requested {device}, only {torch.cuda.device_count()} CUDA devices are visible")
    return device


def _resolve(path_value: Any) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else ROOT / path


def _find_target_uid(solver: Mapping[str, Any], target_public_id: int) -> str | None:
    matches = [
        row
        for row in solver.get("assignment_rows", [])
        if row.get("public_id") is not None and int(row["public_id"]) == int(target_public_id)
    ]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate target public assignment: {target_public_id}")
    return None if not matches else str(matches[0]["candidate_uid"])


def _state_objects(state_axis: Sequence[int], public_axis: Sequence[int]) -> list[Any]:
    if len(state_axis) != len(public_axis):
        raise ValueError("state/public axes are not aligned")
    return [
        type(
            "N72R11R4OnPolicyState",
            (),
            {"association_state_id": int(state_id), "public_id": int(public_id)},
        )()
        for state_id, public_id in zip(state_axis, public_axis)
    ]


def _source_catalog(source_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = read_json(source_root / "corpus_manifest.json")
    if not str(manifest.get("status", "")).startswith("PASS_"):
        raise RuntimeError(f"source corpus is not sealed PASS: {manifest.get('status')}")
    catalog: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        metadata_path = source_root / f"{split}_metadata.jsonl"
        for row in read_jsonl(metadata_path):
            event_id = str(row["event_id"])
            item = catalog.get(event_id)
            identity = {
                "event_id": event_id,
                "sequence": str(row["sequence"]),
                "split": str(split),
                "action_type": str(row["action_type"]),
                "event_frame": int(row["event_frame"]),
                "target_public_id": int(row["target_public_id_for_offline_audit"]),
                "target_dataset_gt_id": int(row["target_dataset_gt_id_for_offline_label"]),
            }
            if item is None:
                catalog[event_id] = identity
            elif item != identity:
                raise RuntimeError(f"source metadata disagrees within event: {event_id}")
    if len(catalog) != int(manifest.get("event_count", -1)):
        raise RuntimeError(f"source event catalog count mismatch: {len(catalog)}")
    return catalog, manifest


def _schedule_and_batch(catalog: Mapping[str, Mapping[str, Any]], source_manifest: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    schedule_path = _resolve(source_manifest.get("source_schedule"))
    schedule = read_json(schedule_path)
    schedule_items = {
        str(item["event_id"]): dict(item)
        for item in schedule.get("events", [])
        if item.get("event_id") is not None
    }
    if any(event_id not in schedule_items for event_id in catalog):
        missing = sorted(event_id for event_id in catalog if event_id not in schedule_items)
        raise RuntimeError(f"source schedule lacks catalog events: {missing[:3]}")
    batch_path = _resolve(source_manifest.get("source_batch_manifest"))
    batch = read_json(batch_path)
    records: dict[str, dict[str, Any]] = {}
    for record in batch.get("records", []):
        event_id = str(record.get("event_id"))
        if event_id in records:
            raise RuntimeError(f"duplicate source batch record: {event_id}")
        records[event_id] = dict(record)
    if any(event_id not in records for event_id in catalog):
        missing = sorted(event_id for event_id in catalog if event_id not in records)
        raise RuntimeError(f"source batch lacks catalog events: {missing[:3]}")
    for event_id in catalog:
        if records[event_id].get("status") != "PASS":
            raise RuntimeError(f"source batch record is not PASS: {event_id}")
        if schedule_items[event_id].get("status") != "ELIGIBLE":
            raise RuntimeError(f"source schedule event is not eligible: {event_id}")
    return schedule_items, records


def _load_c0_window(item: Mapping[str, Any], start: int, end: int) -> dict[int, dict[str, Any]]:
    path = _resolve(item["c0_source"])
    if not path.is_file() or sha256_file(path) != str(item["c0_source_sha256"]):
        raise RuntimeError(f"frozen C0 hash mismatch: {item['event_id']}")
    rows = read_jsonl(path)
    by_frame: dict[int, dict[str, Any]] = {}
    for row in rows:
        frame = int(row.get("frame", -1))
        if frame in by_frame:
            raise RuntimeError(f"duplicate C0 frame: {item['event_id']}:{frame}")
        for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"):
            if row.get(flag) is not False:
                raise RuntimeError(f"C0 violates {flag}: {item['event_id']}:{frame}")
        if not isinstance(row.get("candidate_rows"), list):
            raise RuntimeError(f"C0 candidate rows missing: {item['event_id']}:{frame}")
        by_frame[frame] = dict(row)
    expected = set(range(int(start), int(end) + 1))
    if not expected.issubset(by_frame):
        raise RuntimeError(f"C0 window incomplete: {item['event_id']}")
    return by_frame


def _runtime_values(
    *,
    event: Mapping[str, Any],
    frame: int,
    pool: Sequence[Mapping[str, Any]],
    score_frame: Any,
    state: Any,
) -> dict[str, Any]:
    """Recompute every state-dependent training input from this frame's pool."""

    width, height = causal.image_dimensions(str(event["sequence"]), int(frame))
    target_col = int(score_frame.target_column)
    matrix = np.asarray(score_frame.matrix, dtype=np.float64)
    target_scores = matrix[:, target_col].astype(np.float64)
    other_scores = np.asarray(
        [
            max(
                (float(matrix[index, column]) for column in range(matrix.shape[1]) if column != target_col),
                default=0.0,
            )
            for index in range(len(pool))
        ],
        dtype=np.float64,
    )
    candidate_values = np.stack(
        [
            candidate_feature_vector(
                candidate,
                anchor_feature=event["anchor"],
                anchor_box=event["anchor_box"],
                predicted_box=state.predicted_box,
                previous_raw_sam_id=state.previous_raw_sam_id,
                previous_native_scope=state.previous_native_scope,
                image_width=width,
                image_height=height,
                candidate_count=len(pool),
                base_target_score=float(target_scores[index]),
            )
            for index, candidate in enumerate(pool)
        ],
        axis=0,
    ).astype(np.float32)
    if candidate_values.shape != (len(pool), CANDIDATE_FEATURE_DIM) or not np.all(np.isfinite(candidate_values)):
        raise RuntimeError(f"candidate feature recomputation failed: {event['event_id']}:{frame}")

    trusted = state.recent_trusted or state.long_term_trusted
    causal_scores = np.asarray(
        [
            causal.causal_score(
                candidate,
                float(target_scores[index]),
                trusted,
                state.predicted_box,
            )
            for index, candidate in enumerate(pool)
        ],
        dtype=np.float64,
    )
    order = sorted(range(len(pool)), key=lambda index: (-float(causal_scores[index]), str(pool[index]["candidate_uid"])))
    top = float(causal_scores[order[0]]) if order else 0.0
    second = float(causal_scores[order[1]]) if len(order) > 1 else 0.0
    margin = float(top - second)
    motion_iou = np.asarray(
        [causal.box_iou(candidate["box_xyxy"], state.predicted_box) for candidate in pool],
        dtype=np.float32,
    )
    neighbor = causal.normalized_mean(
        [candidate_values[index, :MEMORY_DIM] for index in order[1:] if np.linalg.norm(candidate_values[index, :MEMORY_DIM]) > 1.0e-6],
        np.asarray(event["anchor"], dtype=np.float32),
    )
    recent, recent_mask, long_term, long_mask, distractors, distractor_mask = state_memory_arrays(state)
    temporal = build_temporal_features(
        state,
        frame_horizon=int(frame) - int(event["event_frame"]),
        causal_top_score=top,
        causal_second_score=second,
        causal_margin=margin,
        has_future_requery=any(str(candidate["candidate_source"]) == "FUTURE_FRAME_REQUERY" for candidate in pool),
    )
    source_values = np.stack(
        [causal.source_vector(str(candidate["candidate_source"])) for candidate in pool],
        axis=0,
    ).astype(np.float32)
    base_solver = solve_effect_assignment(
        candidate_rows=pool,
        persistent_states=_state_objects(score_frame.state_axis, score_frame.public_axis),
        fused_state_candidate_scores=matrix.T,
        source_run_id=f"n72r11r4:base:{event['event_id']}:{frame}",
        session_id=f"n72r11r4:{event['event_id']}",
        none_score=0.0,
    )
    base_target_uid = _find_target_uid(base_solver, int(event["target_public_id"]))
    competition_values = np.stack(
        [
            build_public_competition_feature(
                candidate,
                candidate_uid=str(candidate["candidate_uid"]),
                target_public_id=int(event["target_public_id"]),
                legacy_target_score=float(target_scores[index]),
                legacy_best_other_score=float(other_scores[index]),
                base_target_uid=base_target_uid,
            )
            for index, candidate in enumerate(pool)
        ],
        axis=0,
    ).astype(np.float32)
    if competition_values.shape != (len(pool), PUBLIC_COMPETITION_FEATURE_DIM) or not np.all(np.isfinite(competition_values)):
        raise RuntimeError(f"public competition feature construction failed: {event['event_id']}:{frame}")
    base_order = sorted(range(len(pool)), key=lambda index: (-float(target_scores[index]), str(pool[index]["candidate_uid"])))
    base_top = float(target_scores[base_order[0]]) if base_order else 0.0
    base_second = float(target_scores[base_order[1]]) if len(base_order) > 1 else 0.0
    base_margin = float(base_top - base_second)
    return {
        "candidate_values": candidate_values,
        "source_values": source_values,
        "competition_values": competition_values,
        "recent_array": recent,
        "recent_mask": recent_mask,
        "long_array": long_term,
        "long_mask": long_mask,
        "distractor_array": distractors,
        "distractor_mask": distractor_mask,
        "neighbor": neighbor,
        "temporal": temporal,
        "motion_iou": motion_iou,
        "legacy_target_scores": target_scores,
        "legacy_best_other_scores": other_scores,
        "causal_scores": causal_scores,
        "causal_order": order,
        "causal_top": top,
        "causal_second": second,
        "causal_margin": margin,
        "base_top": base_top,
        "base_second": base_second,
        "base_margin": base_margin,
        "base_solver_target_uid": base_target_uid,
        "base_solver": base_solver,
        "target_column": target_col,
    }


def _model_logits(model: Any, values: Mapping[str, Any], pool: Sequence[Mapping[str, Any]], device: torch.device) -> tuple[np.ndarray, float]:
    count = len(pool)
    if not hasattr(model, "score"):
        raise TypeError("N72R11R4 on-policy corpus requires a TemporalScorerAdapter")
    output = np.asarray(
        model.score(
            candidate_features=values["candidate_values"][None],
            candidate_mask=np.ones((1, count), dtype=np.bool_),
            source_features=values["source_values"][None],
            public_competition_features=values["competition_values"][None],
            human_anchor=np.asarray(values["human_anchor"], dtype=np.float32)[None],
            recent_trusted_memory=values["recent_array"][None],
            recent_trusted_mask=values["recent_mask"][None],
            long_term_trusted_memory=values["long_array"][None],
            long_term_trusted_mask=values["long_mask"][None],
            distractor_memory=values["distractor_array"][None],
            distractor_mask=values["distractor_mask"][None],
            neighbor_feature=values["neighbor"][None],
            temporal_features=values["temporal"][None],
        ),
        dtype=np.float64,
    ).reshape(-1)
    if output.shape != (count + 1,) or not np.all(np.isfinite(output)):
        raise RuntimeError(f"temporal scorer returned invalid logits: {output.shape}")
    return output[:count].astype(np.float64), float(output[count])


def _solver_rows(pool: Sequence[Mapping[str, Any]], solver: Mapping[str, Any]) -> list[dict[str, Any]]:
    decisions = {str(row["candidate_uid"]): row for row in solver.get("assignment_rows", [])}
    expected = {str(candidate["candidate_uid"]) for candidate in pool}
    if set(decisions) != expected:
        raise RuntimeError("exact solver output does not cover the complete pool")
    output: list[dict[str, Any]] = []
    for candidate in pool:
        decision = decisions[str(candidate["candidate_uid"])]
        item = serializable_candidate(candidate, include_feature=False)
        feature = candidate.get("feature")
        norm = None if feature is None else float(np.linalg.norm(np.asarray(feature, dtype=np.float32).reshape(-1)))
        public_id = decision.get("public_id")
        item.update(
            {
                "feature_finite": bool(feature is not None and norm is not None and np.isfinite(norm) and norm > 1.0e-6),
                "feature_norm": norm,
                "solver_public_id": None if public_id is None else int(public_id),
                "solver_association_state_id": decision.get("association_state_id"),
                "solver_status": str(decision["status"]),
                "solver_score": float(decision["score"]),
                "public_id": None if public_id is None else int(public_id),
                "public_id_authority": "exact_global_solver_output" if public_id is not None else None,
                "assigned_public_id": None if public_id is None else int(public_id),
                "assignment_status": "EXPLICIT_NONE" if public_id is None else "ASSIGNED_TO_PUBLIC_ID",
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "posthoc_gt_used": False,
            }
        )
        output.append(item)
    return output


def _build_event_runtime(item: Mapping[str, Any], record: Mapping[str, Any], model: torch.nn.Module, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray]], dict[str, Any]]:
    event_id = str(item["event_id"])
    target_by_frame, active_by_frame, details = causal.load_secondary_artifact(item, record)
    start = int(item["secondary_frame"])
    end = int(details["end_frame"])
    c0_by_frame = _load_c0_window(item, start, end)
    anchor = causal.unit(details["anchor"]["feature"], f"{event_id} human anchor")
    anchor_box = causal.box_xyxy(item["current_target_box_posthoc_selection_only"], f"{event_id} anchor box")
    target_event_rows = list(target_by_frame[start].get("candidate_rows", []))
    if not target_event_rows:
        raise RuntimeError(f"event frame lacks target-session candidate: {event_id}")
    initial_raw = target_event_rows[0].get("official_raw_sam_id")
    initial_scope = target_event_rows[0].get("native_scope", target_event_rows[0].get("native_tid_scope"))
    event = {
        "event_id": event_id,
        "sequence": str(item["sequence"]),
        "split": str(item["split"]),
        "action_type": str(item["action_type"]),
        "event_frame": start,
        "target_public_id": int(item["target_public_id"]),
        "target_dataset_gt_id": int(item["target_dataset_gt_id"]),
        "anchor": anchor,
        "anchor_box": [float(value) for value in anchor_box],
    }
    state = initialize_temporal_state(
        anchor_feature=anchor,
        anchor_box=anchor_box,
        previous_raw_sam_id=None if initial_raw is None else int(initial_raw),
        previous_native_scope=None if initial_scope is None else str(initial_scope),
    )
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {
        "candidate_features": [],
        "candidate_mask": [],
        "source_features": [],
        "public_competition_features": [],
        "human_anchor": [],
        "recent_trusted_memory": [],
        "recent_trusted_mask": [],
        "long_term_trusted_memory": [],
        "long_term_trusted_mask": [],
        "distractor_memory": [],
        "distractor_mask": [],
        "neighbor_feature": [],
        "temporal_features": [],
        "labels_placeholder": [],
        "legacy_target_scores": [],
        "legacy_best_other_scores": [],
        "incumbent_target": [],
        "incumbent_other": [],
        "confidence": [],
        "presence": [],
        "motion_iou": [],
        "protected_candidate_mask": [],
    }
    for frame in range(start + 1, end + 1):
        c0_row = c0_by_frame[frame]
        main_rows = [dict(value) for value in c0_row["candidate_rows"]]
        current_rows = [dict(value) for value in target_by_frame[frame].get("candidate_rows", [])]
        active_future_rows = [dict(value) for value in active_by_frame.get(frame, [])]
        pool, pool_audit = build_candidate_pool_with_future_requery(
            main_rows,
            current_rows,
            active_future_rows,
            sequence=str(item["sequence"]),
            frame=frame,
        )
        if not pool:
            raise RuntimeError(f"empty exact on-policy pool: {event_id}:{frame}")
        score_frame = build_secondary_public_score_frame(
            c0_row=c0_row,
            main_candidates=main_rows,
            pool=pool,
            target_public_id=int(item["target_public_id"]),
            anchor_feature=anchor,
            predicted_box=state.predicted_box,
            event_id=event_id,
            frame=frame,
        )
        values = _runtime_values(event=event, frame=frame, pool=pool, score_frame=score_frame, state=state)
        before = state_audit(state)
        arrays["candidate_features"].append(values["candidate_values"])
        arrays["candidate_mask"].append(np.ones(len(pool), dtype=np.bool_))
        arrays["source_features"].append(values["source_values"])
        arrays["public_competition_features"].append(values["competition_values"])
        arrays["human_anchor"].append(anchor.copy())
        arrays["recent_trusted_memory"].append(values["recent_array"])
        arrays["recent_trusted_mask"].append(values["recent_mask"])
        arrays["long_term_trusted_memory"].append(values["long_array"])
        arrays["long_term_trusted_mask"].append(values["long_mask"])
        arrays["distractor_memory"].append(values["distractor_array"])
        arrays["distractor_mask"].append(values["distractor_mask"])
        arrays["neighbor_feature"].append(values["neighbor"])
        arrays["temporal_features"].append(values["temporal"])
        arrays["legacy_target_scores"].append(values["legacy_target_scores"].astype(np.float32))
        arrays["legacy_best_other_scores"].append(values["legacy_best_other_scores"].astype(np.float32))
        arrays["incumbent_target"].append(
            np.asarray(
                [
                    float(candidate.get("incumbent_public_id_if_any") == int(item["target_public_id"]))
                    for candidate in pool
                ],
                dtype=np.float32,
            )
        )
        arrays["incumbent_other"].append(
            np.asarray(
                [
                    float(
                        candidate.get("incumbent_public_id_if_any") is not None
                        and candidate.get("incumbent_public_id_if_any") != int(item["target_public_id"])
                    )
                    for candidate in pool
                ],
                dtype=np.float32,
            )
        )
        arrays["confidence"].append(np.asarray([float(candidate["confidence"]) for candidate in pool], dtype=np.float32))
        arrays["presence"].append(np.asarray([float(candidate["presence_score"]) for candidate in pool], dtype=np.float32))
        arrays["motion_iou"].append(values["motion_iou"])
        arrays["protected_candidate_mask"].append(
            np.asarray(
                [
                    bool(
                        candidate.get("incumbent_public_id_if_any") is not None
                        and candidate.get("incumbent_public_id_if_any") != int(item["target_public_id"])
                    )
                    for candidate in pool
                ],
                dtype=np.bool_,
            )
        )

        logits, none_logit = _model_logits(
            model,
            {**values, "human_anchor": anchor},
            pool,
            device,
        )
        candidate_uids = [str(candidate["candidate_uid"]) for candidate in pool]
        selection = select_candidate_from_logits(logits, none_logit, candidate_uids)
        fused, injected_delta = apply_legacy_injection(
            score_frame.matrix,
            target_column=int(score_frame.target_column),
            candidate_uids=candidate_uids,
            selection=selection,
        )
        solver = solve_effect_assignment(
            candidate_rows=pool,
            persistent_states=_state_objects(score_frame.state_axis, score_frame.public_axis),
            fused_state_candidate_scores=fused.T,
            source_run_id=f"n72r11r4:exact:{event_id}:{frame}",
            session_id=f"n72r11r4:{event_id}",
            none_score=0.0,
        )
        target_uid = _find_target_uid(solver, int(item["target_public_id"]))
        assigned = next((candidate for candidate in pool if str(candidate["candidate_uid"]) == str(target_uid)), None)
        selected_uid = selection.get("selected_candidate_uid")
        state_update = update_temporal_state(
            state,
            candidates=pool,
            target_uid=target_uid,
            selected_uid=None if selected_uid is None else str(selected_uid),
            selected_score=selection.get("selected_score"),
            selected_margin=selection.get("best_minus_second_margin"),
            fused_target_scores=fused[:, int(score_frame.target_column)],
            frame_horizon=frame - start,
            assigned_candidate=assigned,
            base_top_score=values["base_top"],
            base_second_score=values["base_second"],
        )
        source_rows = [serializable_candidate(candidate, include_feature=False) for candidate in pool]
        for source_row in source_rows:
            source_row["public_id"] = None
            source_row["public_id_authority"] = None
        rows.append(
            {
                "schema_version": "N72R11R4_EXACT_ONPOLICY_RUNTIME_ROW_V1",
                "record_kind": "exact_solver_onpolicy_future_frame",
                "event_id": event_id,
                "sequence": str(item["sequence"]),
                "split": str(item["split"]),
                "action_type": str(item["action_type"]),
                "event_frame": start,
                "frame": frame,
                "frame_horizon": frame - start,
                "target_public_id_for_offline_audit": int(item["target_public_id"]),
                "target_dataset_gt_id_for_offline_label": int(item["target_dataset_gt_id"]),
                "anchor_box": event["anchor_box"],
                "candidate_uids": candidate_uids,
                "candidate_boxes": [list(map(float, candidate["box_xyxy"])) for candidate in pool],
                "candidate_sources": [str(candidate["candidate_source"]) for candidate in pool],
                "candidate_feature_sha256": [candidate.get("feature_sha256") for candidate in pool],
                "candidate_incumbent_public_ids": [candidate.get("incumbent_public_id_if_any") for candidate in pool],
                "candidate_confidence": [float(candidate["confidence"]) for candidate in pool],
                "candidate_presence": [float(candidate["presence_score"]) for candidate in pool],
                "candidate_official_raw_sam_ids": [candidate.get("official_raw_sam_id") for candidate in pool],
                "candidate_native_scopes": [candidate.get("native_scope") for candidate in pool],
                "candidate_adapter_external_ids": [candidate.get("adapter_external_id") for candidate in pool],
                "base_target_scores": values["legacy_target_scores"].astype(float).tolist(),
                "base_best_other_scores": values["legacy_best_other_scores"].astype(float).tolist(),
                "base_solver_target_uid": values["base_solver_target_uid"],
                "base_solver": values["base_solver"],
                "base_solver_public_axis": [int(value) for value in score_frame.public_axis],
                "base_solver_state_axis": [int(value) for value in score_frame.state_axis],
                "causal_top_score": float(values["causal_top"]),
                "causal_second_score": float(values["causal_second"]),
                "causal_margin": float(values["causal_margin"]),
                "causal_order": [int(value) for value in values["causal_order"]],
                "causal_scores": values["causal_scores"].astype(float).tolist(),
                "temporal_feature_schema": list(TEMPORAL_FEATURE_SCHEMA),
                "public_competition_feature_schema": list(PUBLIC_COMPETITION_FEATURE_SCHEMA),
                "public_competition_features": values["competition_values"].astype(float).tolist(),
                "score_audit": {
                    "association_state_axis": [int(value) for value in score_frame.state_axis],
                    "public_id_axis": [int(value) for value in score_frame.public_axis],
                    "target_column": int(score_frame.target_column),
                    "base_score_matrix": score_frame.matrix.astype(float).tolist(),
                    "fused_score_matrix": fused.astype(float).tolist(),
                    "base_target_scores": values["legacy_target_scores"].astype(float).tolist(),
                    "base_best_other_scores": values["legacy_best_other_scores"].astype(float).tolist(),
                    "fused_target_scores": fused[:, int(score_frame.target_column)].astype(float).tolist(),
                    "model_logits": logits.astype(float).tolist(),
                    "none_logit": float(none_logit),
                    "model_scores": selection["candidate_scores"],
                    "legacy_injection_delta": float(injected_delta),
                    "base_top1_score": float(values["base_top"]),
                    "base_top2_score": float(values["base_second"]),
                    "base_assignment_margin": float(values["base_margin"]),
                    "runtime_future_gt_used": False,
                },
                "selection": {
                    **selection,
                    "runtime_future_gt_used": False,
                    "public_id_inference": False,
                },
                "assignment": {
                    "base_solver_target_uid": values["base_solver_target_uid"],
                    "target_assigned_candidate_uid": target_uid,
                    "target_selected_candidate_uid": selected_uid,
                    "target_selector_and_solver_agree": bool(selected_uid is not None and str(selected_uid) == str(target_uid)),
                    "solver": solver,
                    "runtime_future_gt_used": False,
                },
                "candidate_pool": {**pool_audit, "candidate_rows": source_rows},
                "temporal_state_before": before,
                "temporal_state_update": state_update,
                "temporal_state_after": state_audit(state),
                "memory_read": True,
                "memory_write": bool(state_update["trusted_admitted"]),
                "event_frame_memory_read": False,
                "first_memory_visible_frame": start + 1,
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "posthoc_gt_used": False,
                "gt_used_offline": False,
                "public_id_inference": False,
                "public_id_immutable": True,
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
            }
        )
    if not rows:
        raise RuntimeError(f"event has no future runtime rows: {event_id}")
    return rows, arrays, {
        "event_id": event_id,
        "event_frame": start,
        "end_frame": end,
        "example_count": len(rows),
        "future_requery_row_count": int(details["future_requery_row_count"]),
        "selected_trigger_count": int(details["selected_trigger_count"]),
        "runtime_future_gt_used": False,
    }


def _stack_split(
    payloads: Sequence[tuple[list[dict[str, Any]], dict[str, list[np.ndarray]], dict[str, Any]]],
    gt_by_sequence: Mapping[str, Mapping[int, Mapping[int, Sequence[float]]]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    for event_rows, event_arrays, _details in payloads:
        sequence = str(event_rows[0]["sequence"])
        causal.attach_offline_labels(event_rows, gt_by_sequence[sequence])
        for row in event_rows:
            row["gt_used_offline"] = True
        rows.extend(event_rows)
        for key, values in event_arrays.items():
            if key != "labels_placeholder":
                collected[key].extend(values)
    if not rows:
        raise RuntimeError("exact on-policy split is empty")
    max_candidates = max(len(row["candidate_uids"]) for row in rows)
    count = len(rows)
    candidate_features = np.zeros((count, max_candidates, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    candidate_mask = np.zeros((count, max_candidates), dtype=np.bool_)
    source_features = np.zeros((count, max_candidates, SOURCE_FEATURE_DIM), dtype=np.float32)
    competition_features = np.zeros((count, max_candidates, PUBLIC_COMPETITION_FEATURE_DIM), dtype=np.float32)
    padded_float_names = (
        "legacy_target_scores", "legacy_best_other_scores", "incumbent_target", "incumbent_other",
        "confidence", "presence", "motion_iou",
    )
    padded_float = {key: np.zeros((count, max_candidates), dtype=np.float32) for key in padded_float_names}
    protected = np.zeros((count, max_candidates), dtype=np.bool_)
    for index, row in enumerate(rows):
        width = len(row["candidate_uids"])
        candidate_features[index, :width] = collected["candidate_features"][index]
        candidate_mask[index, :width] = True
        source_features[index, :width] = collected["source_features"][index]
        competition_features[index, :width] = collected["public_competition_features"][index]
        for key in padded_float_names:
            padded_float[key][index, :width] = collected[key][index]
        protected[index, :width] = collected["protected_candidate_mask"][index]
    labels = np.asarray(
        [
            int(row["label_index_raw"]) if int(row["label_index_raw"]) < len(row["candidate_uids"]) else max_candidates
            for row in rows
        ],
        dtype=np.int64,
    )
    arrays: dict[str, np.ndarray] = {
        "candidate_features": candidate_features,
        "candidate_mask": candidate_mask,
        "source_features": source_features,
        "public_competition_features": competition_features,
        "human_anchor": np.stack(collected["human_anchor"], axis=0).astype(np.float32),
        "recent_trusted_memory": np.stack(collected["recent_trusted_memory"], axis=0).astype(np.float32),
        "recent_trusted_mask": np.stack(collected["recent_trusted_mask"], axis=0).astype(np.bool_),
        "long_term_trusted_memory": np.stack(collected["long_term_trusted_memory"], axis=0).astype(np.float32),
        "long_term_trusted_mask": np.stack(collected["long_term_trusted_mask"], axis=0).astype(np.bool_),
        "distractor_memory": np.stack(collected["distractor_memory"], axis=0).astype(np.float32),
        "distractor_mask": np.stack(collected["distractor_mask"], axis=0).astype(np.bool_),
        "neighbor_feature": np.stack(collected["neighbor_feature"], axis=0).astype(np.float32),
        "temporal_features": np.stack(collected["temporal_features"], axis=0).astype(np.float32),
        "labels": labels,
        "candidate_counts": np.asarray([len(row["candidate_uids"]) for row in rows], dtype=np.int64),
        **padded_float,
        "protected_candidate_mask": protected,
    }
    for key, value in arrays.items():
        if value.dtype.kind not in "biu" and not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite exact on-policy array: {key}")
    return arrays, rows, {
        "example_count": count,
        "event_count": len({str(row["event_id"]) for row in rows}),
        "sequence_count": len({str(row["sequence"]) for row in rows}),
        "max_candidates": max_candidates,
        "label_counts": dict(sorted(__import__("collections").Counter(str(row["label_reason"]) for row in rows).items())),
        "future_positive_count": sum(row["label_kind"] == "TARGET_CANDIDATE" for row in rows),
    }


def _stage_base(stage: str) -> dict[str, Any]:
    return {
        "schema_version": "N72R11R4_STAGE_STATUS_V1",
        "stage": stage,
        "started_at_utc": now_utc(),
        "runtime_future_gt_used": False,
        "gt_used_for_runtime_decision": False,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    }


def main() -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--model-checkpoint", type=Path, default=ROOT / "outputs/N72R11R3/v3_bootstrap/v3_bootstrap.pt")
    parser.add_argument("--model-kind", choices=("v3", "pctis"), default="v3")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage-path", type=Path, default=DEFAULT_STAGE_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("all", "train", "validation"), default="all")
    parser.add_argument("--resource-censored", action="store_true")
    args = parser.parse_args()
    source_root = _resolve(args.source_root)
    checkpoint = _resolve(args.model_checkpoint)
    output_root = _resolve(args.output_root)
    stage_path = _resolve(args.stage_path)
    status = _stage_base("N72R11R4-02-EXACT-SOLVER-ON-POLICY-CORPUS")
    status.update({
        "source_root": str(source_root),
        "model_checkpoint": str(checkpoint),
        "model_kind": str(args.model_kind),
        "output_root": str(output_root),
        "split_requested": str(args.split),
        "resource_censored_development": bool(args.resource_censored),
    })
    try:
        source_catalog, source_manifest = _source_catalog(source_root)
        if bool(source_manifest.get("resource_censored_development", False)) != bool(args.resource_censored):
            raise RuntimeError("resource-censored source requires matching --resource-censored")
        schedule_items, batch_records = _schedule_and_batch(source_catalog, source_manifest)
        device = device_from(str(args.device))
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if str(args.model_kind) == "v3":
            raw_model, checkpoint_payload = train_v3.load_checkpoint(checkpoint, device)
            model = V3TemporalScorerAdapter(raw_model, device)
            model_schema = checkpoint_payload.get("temporal_feature_schema")
            rollout_model = "N72R11R3_V3_CHECKPOINT"
        else:
            from sam3_intermot.reacquisition.models.n72r11r4_public_competition_temporal import load_pctis_checkpoint

            raw_model = load_pctis_checkpoint(checkpoint, device)
            checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = PublicCompetitionScorerAdapter(raw_model, device)
            model_schema = checkpoint_payload.get("public_competition_feature_schema")
            rollout_model = "N72R11R4_PCTIS_CHECKPOINT"
        groups: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
        for event_id, catalog_item in source_catalog.items():
            item = dict(schedule_items[event_id])
            item.update({
                "split": catalog_item["split"],
                "target_public_id": catalog_item["target_public_id"],
                "target_dataset_gt_id": catalog_item["target_dataset_gt_id"],
                "event_id": event_id,
            })
            groups[str(catalog_item["split"])].append(item)
        splits = ("train", "validation") if args.split == "all" else (str(args.split),)
        payloads_by_split: dict[str, list[tuple[list[dict[str, Any]], dict[str, list[np.ndarray]], dict[str, Any]]]] = {"train": [], "validation": []}
        split_summaries: dict[str, Any] = {}
        for split in splits:
            for item in sorted(groups[split], key=lambda value: str(value["event_id"])):
                payloads_by_split[split].append(_build_event_runtime(item, batch_records[str(item["event_id"])], model, device))
            sequences = {str(item["sequence"]) for item in groups[split]}
            gt_by_sequence = {sequence: causal.load_gt(sequence) for sequence in sorted(sequences)}
            arrays, metadata, summary = _stack_split(payloads_by_split[split], gt_by_sequence)
            npz_path = output_root / f"{split}.npz"
            metadata_path = output_root / f"{split}_metadata.jsonl"
            atomic_npz(npz_path, arrays)
            atomic_jsonl(metadata_path, metadata)
            split_summaries[split] = {
                **summary,
                "npz": str(npz_path),
                "npz_sha256": sha256_file(npz_path),
                "metadata": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
            }
            atomic_json(output_root / f"split_summary_{split}.json", split_summaries[split])
        complete = all(split in split_summaries for split in ("train", "validation"))
        manifest = {
            "schema_version": "N72R11R4_EXACT_ONPOLICY_CORPUS_V2",
            "status": "PASS_N72R11R4_EXACT_SOLVER_ON_POLICY_CORPUS" if complete else "PARTIAL_N72R11R4_EXACT_SOLVER_ON_POLICY_CORPUS_SPLIT",
            "created_at_utc": now_utc(),
            "source_corpus_manifest": str(source_root / "corpus_manifest.json"),
            "source_corpus_manifest_sha256": sha256_file(source_root / "corpus_manifest.json"),
            "source_batch_manifest": str(_resolve(source_manifest["source_batch_manifest"])),
            "source_batch_manifest_sha256": sha256_file(_resolve(source_manifest["source_batch_manifest"])),
            "model_checkpoint": str(checkpoint),
            "model_checkpoint_sha256": sha256_file(checkpoint),
            "model_checkpoint_schema": model_schema,
            "model_kind": str(args.model_kind),
            "event_count": len(source_catalog),
            "sequence_count": len({item["sequence"] for item in source_catalog.values()}),
            "splits": split_summaries,
            "rollout_model": rollout_model,
            "edge_mode": "LEGACY_INJECTION",
            "global_solver": "solve_effect_assignment",
            "candidate_features_recomputed": True,
            "motion_iou_recomputed": True,
            "causal_scores_recomputed": True,
            "raw_scope_preserved": True,
            "public_competition_features_saved": True,
            "public_competition_feature_schema": list(PUBLIC_COMPETITION_FEATURE_SCHEMA),
            "runtime_future_gt_used": False,
            "gt_used_only_offline_label_generation": True,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
            "not_real_human_evidence": True,
            "production_authorized": False,
            "split_requested": str(args.split),
            "resource_censored_development": bool(args.resource_censored),
        }
        manifest_path = output_root / "corpus_manifest.json"
        if complete:
            atomic_json(manifest_path, manifest)
        status.update({
            "status": manifest["status"],
            "device": str(device),
            "corpus_manifest": str(manifest_path) if complete else None,
            "corpus_manifest_sha256": sha256_file(manifest_path) if complete else None,
            "model_checkpoint_sha256": sha256_file(checkpoint),
            "split_summaries": split_summaries,
            "candidate_features_recomputed": True,
            "motion_iou_recomputed": True,
            "causal_scores_recomputed": True,
            "raw_scope_preserved": True,
            "exact_solver_in_rollout": True,
            "public_competition_features_saved": True,
            "finished_at_utc": now_utc(),
            "model_kind": str(args.model_kind),
            "rollout_model": rollout_model,
        })
        atomic_json(stage_path, status)
        print(json.dumps(status, sort_keys=True))
        return 0
    except Exception as exc:
        failure = output_root / "attempts" / f"exact_onpolicy_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        payload = {
            **status,
            "status": "FAIL_N72R11R4_EXACT_SOLVER_ON_POLICY_CORPUS",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failure_artifact": str(failure),
            "finished_at_utc": now_utc(),
        }
        atomic_json(failure, payload)
        atomic_json(stage_path, payload)
        print(json.dumps({"status": payload["status"], "failure_artifact": str(failure), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
