#!/usr/bin/env python3
"""N30-B: real multi-identity decomposition of spatial and B10 writes.

The case manifest is frozen by a separate ``--freeze-only`` invocation before
any M0--M4 branch is run.  Every branch uses the same public identities and
the same original-plus-official-decoder candidate construction, followed by
one global Hungarian assignment with per-row NONE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = ROOT / "third_party" / "sam3"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3_intermot.adaptation.corrected_mask_teacher import (  # noqa: E402
    BOX_DERIVED_PSEUDO_MASK,
)
from sam3_intermot.adaptation.decoder_update_transaction import (  # noqa: E402
    DecoderCorrectionEvent,
    DecoderUpdateConfig,
    DecoderUpdateTransaction,
)
from sam3_intermot.adaptation.sam3_decoder_lit import (  # noqa: E402
    DecoderLITConfig,
    SAM3DecoderLITAdapter,
)
from scripts.n29_lit_online_replay import (  # noqa: E402
    DecoderCapture,
    _clone_tree,
    _get_official_decoder,
    _image_files,
    _make_backend,
    _read_gt,
    _session,
    _slot_tensor,
    _tensor_status_tree,
)
from scripts.n29r_real_association import (  # noqa: E402
    FrozenClipReID,
    assignment_audit,
    bind_official_multi,
    b10_scores,
    box_iou,
    decoder_candidates,
    load_audited_negative_memory,
    original_candidate,
    sha256,
    state_slot_ids,
)
from scripts.n29r_paired_replay import _update_dict  # noqa: E402


TRAIN_ROOT = Path("/path/to/dancetrack/train")
CHECKPOINT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
CLIP_CHECKPOINT = ROOT / "outputs/n15/checkpoints/clip_reid_hf/person_vit_clip_reid.pth"
N26_ARRAYS = ROOT / "outputs/n26/dense_dataset/round0_train30.npz"
N26_PARENTS = ROOT / "outputs/n26/dense_dataset/round0_train30_parents.jsonl"

BRANCHES = (
    "M0_no_correction",
    "M1_official_spatial_write_only",
    "M2_b10_identity_write_only",
    "M3_official_spatial_plus_b10",
    "M4_official_spatial_plus_b10_plus_online_lora",
)

# These are frozen before running any branch.  The first three are exactly
# N29-R's predeclared real association cases; the remaining seven are fixed
# train-fold additions chosen by sequence/frame/identity coordinates, not by
# any N30 branch outcome.
FIXED_CASES = (
    {"sequence": "dancetrack0001", "start_frame": 79, "interaction_frame": 82, "identity_ids": (4, 5), "source": "N29-R-predeclared"},
    {"sequence": "dancetrack0002", "start_frame": 556, "interaction_frame": 559, "identity_ids": (0, 7), "source": "N29-R-predeclared"},
    {"sequence": "dancetrack0006", "start_frame": 1116, "interaction_frame": 1119, "identity_ids": (3, 5), "source": "N29-R-predeclared"},
    {"sequence": "dancetrack0008", "start_frame": 59, "interaction_frame": 62, "identity_ids": (1, 2), "source": "N30-fixed-train-addition"},
    {"sequence": "dancetrack0012", "start_frame": 82, "interaction_frame": 85, "identity_ids": (7, 11), "source": "N30-fixed-train-addition"},
    {"sequence": "dancetrack0015", "start_frame": 375, "interaction_frame": 378, "identity_ids": (0, 7), "source": "N30-fixed-train-addition"},
    {"sequence": "dancetrack0016", "start_frame": 1352, "interaction_frame": 1355, "identity_ids": (1, 5), "source": "N30-fixed-train-addition"},
    {"sequence": "dancetrack0020", "start_frame": 276, "interaction_frame": 279, "identity_ids": (22, 24), "source": "N30-fixed-train-addition"},
    {"sequence": "dancetrack0023", "start_frame": 175, "interaction_frame": 178, "identity_ids": (3, 7), "source": "N30-fixed-train-addition"},
    {"sequence": "dancetrack0024", "start_frame": 554, "interaction_frame": 557, "identity_ids": (0, 1), "source": "N30-fixed-train-addition"},
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _overlap(a: Sequence[float], b: Sequence[float]) -> float:
    return box_iou(a, b)


def _freeze_cases(manifest_path: Path, train_root: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for index, fixed in enumerate(FIXED_CASES):
        sequence = str(fixed["sequence"])
        sequence_dir = train_root / sequence
        gt = _read_gt(sequence_dir)
        images = _image_files(sequence_dir)
        start = int(fixed["start_frame"])
        interaction = int(fixed["interaction_frame"])
        query_end = interaction + 5
        ids = [int(value) for value in fixed["identity_ids"]]
        if query_end >= len(images):
            raise ValueError(f"fixed case extends past train video: {fixed}")
        for frame in (start, interaction, query_end):
            if any(identity not in gt.get(frame, {}) for identity in ids):
                raise ValueError(f"fixed case identity is not visible at frame {frame}: {fixed}")
        interaction_overlap = _overlap(gt[interaction][ids[0]], gt[interaction][ids[1]])
        case = {
            "case_id": f"{sequence}:{start}:{interaction}:{ids[0]}-{ids[1]}",
            "sequence": sequence,
            "sequence_path": str(sequence_dir),
            "split": "train/train_fold",
            "start_frame": start,
            "interaction_frame": interaction,
            "query_end": query_end,
            "identity_ids": ids,
            "public_ids": [1000 + identity for identity in ids],
            "correction_dataset_identity": ids[0],
            "correction_public_id": 1000 + ids[0],
            "event_type": "CORRECT_OBJECT",
            "fixed_source": str(fixed["source"]),
            "interaction_gt_box_iou_between_pair": interaction_overlap,
        }
        cases.append(case)
    payload = {
        "protocol": "N30-B-MULTI-IDENTITY-WRITE-ABLATION",
        "status": "PASS",
        "selection_frozen_before_branches": True,
        "future_gt_used_for_selection": False,
        "val25_read": False,
        "test_labels_used": False,
        "train_root": str(train_root),
        "case_count": len(cases),
        "cases": cases,
        "selection_rule": "ten fixed train-fold sequence/frame/identity coordinates; GT only validates legal prompt/query availability and is post-hoc audit data",
    }
    _write_json(manifest_path, payload)
    payload["manifest_sha256"] = sha256(manifest_path)
    return payload


def _load_cases(manifest_path: Path, train_root: Path, minimum: int = 10) -> tuple[dict[str, Any], str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("selection_frozen_before_branches") is not True:
        raise ValueError("N30-B case manifest is not a frozen PASS artifact")
    if payload.get("val25_read") is not False or payload.get("future_gt_used_for_selection") is not False:
        raise ValueError("N30-B manifest violates train-only causal boundary")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) < minimum:
        raise ValueError(f"N30-B requires at least {minimum} frozen cases")
    for case in cases:
        sequence = str(case.get("sequence", ""))
        split = str(case.get("split", ""))
        path = Path(case.get("sequence_path", ""))
        if "val" in sequence.lower() or "test" in sequence.lower() or "val" in split.lower() or "test" in split.lower():
            raise ValueError(f"N30-B refused non-train case {case}")
        if path != train_root / sequence:
            raise ValueError(f"N30-B case path is outside frozen train root: {case}")
    return payload, sha256(manifest_path)


class B10IdentityMemory:
    """The audited frozen-CLIP positive/explicit-negative B10 memory."""

    def __init__(self, identity_data: list[dict[str, Any]]) -> None:
        self.identity_data = identity_data
        self.write_ledger: list[dict[str, Any]] = []

    def snapshot(self) -> dict[str, Any]:
        return {
            "items": [
                {
                    "positive_features": np.asarray(item["positive_features"], dtype=np.float32).copy(),
                    "negative_features": np.asarray(item["negative_features"], dtype=np.float32).copy(),
                    "root_feature": np.asarray(item["root_feature"], dtype=np.float32).copy(),
                    "latest_anchor_feature": None
                    if item.get("latest_anchor_feature") is None
                    else np.asarray(item["latest_anchor_feature"], dtype=np.float32).copy(),
                }
                for item in self.identity_data
            ],
            "ledger": list(self.write_ledger),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        for item, old in zip(self.identity_data, snapshot["items"]):
            item["positive_features"] = np.asarray(old["positive_features"], dtype=np.float32).copy()
            item["negative_features"] = np.asarray(old["negative_features"], dtype=np.float32).copy()
            item["root_feature"] = np.asarray(old["root_feature"], dtype=np.float32).copy()
            item["latest_anchor_feature"] = None if old["latest_anchor_feature"] is None else np.asarray(old["latest_anchor_feature"], dtype=np.float32).copy()
        self.write_ledger = list(snapshot["ledger"])

    def state_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "public_id": int(item["public_id"]),
                "positive_count": int(len(item["positive_features"])),
                "negative_count": int(len(item["negative_features"])),
                "latest_anchor_present": item.get("latest_anchor_feature") is not None,
            }
            for item in self.identity_data
        ]

    def write_positive(self, row: int, feature: np.ndarray, *, frame: int, public_id: int) -> dict[str, Any]:
        item = self.identity_data[row]
        before = self.state_summary()[row]
        value = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        if value.shape[1] != 1280 or not np.isfinite(value).all():
            raise ValueError("B10 correction feature must be finite 1280-D CLIP-ReID")
        item["positive_features"] = np.concatenate([item["positive_features"], value], axis=0)
        item["latest_anchor_feature"] = value[0].copy()
        self.write_ledger.append({"type": "positive_anchor", "frame": int(frame), "public_id": int(public_id)})
        after = self.state_summary()[row]
        return {"status": "WRITTEN", "before": before, "after": after, "feature_dim": 1280}


def _clone_identity_data(data: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **{key: value for key, value in item.items() if key not in {"root_feature", "positive_features", "negative_features", "latest_anchor_feature", "target_box"}},
            "root_feature": np.asarray(item["root_feature"], dtype=np.float32).copy(),
            "positive_features": np.asarray(item["positive_features"], dtype=np.float32).copy(),
            "negative_features": np.asarray(item["negative_features"], dtype=np.float32).copy(),
            "latest_anchor_feature": None if item.get("latest_anchor_feature") is None else np.asarray(item["latest_anchor_feature"], dtype=np.float32).copy(),
            "target_box": np.asarray(item["target_box"], dtype=float).copy(),
        }
        for item in data
    ]


def _official_action_history(backend: Any, public_ids: Sequence[int], start: int) -> None:
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    state["action_history"] = [{"type": "add", "obj_ids": [int(value) for value in public_ids], "frame_idx": int(start)}]


def _bind_official_multi_clean(
    backend: Any,
    frame_idx: int,
    public_ids: Sequence[int],
    boxes: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Bind public IDs and discard transient semantic-prompt raw-ID caches.

    ``bind_official_multi`` first calls the official semantic box API to
    materialize the image cache.  That API numbers its temporary outputs from
    zero, while the subsequent explicit tracker binding uses the requested
    public IDs.  Keeping both in ``cached_frame_outputs`` makes the official
    merge path ask the public score table for raw ID 0.  The semantic outputs
    are only bootstrap material; the explicit rectangle tracker state is the
    state under audit, so remove the transient cache before propagation.
    """

    binding = bind_official_multi(backend, frame_idx, public_ids, boxes)
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    requested = {int(value) for value in public_ids}
    stale_cache_ids: set[int] = set()
    filtered_cache: dict[int, dict[Any, Any]] = {}
    for cached_frame, frame_cache in state.get("cached_frame_outputs", {}).items():
        kept = {}
        for object_id, mask in frame_cache.items():
            object_id_int = int(object_id)
            if object_id_int in requested:
                kept[object_id] = mask
            else:
                stale_cache_ids.add(object_id_int)
        if kept:
            filtered_cache[int(cached_frame)] = kept
    state["cached_frame_outputs"] = filtered_cache
    binding["transient_semantic_cache_ids_removed"] = sorted(stale_cache_ids)
    return binding


def _restart_anchor(
    backend: Any,
    capture: DecoderCapture,
    sequence_dir: Path,
    case: Mapping[str, Any],
    prompt_boxes: Sequence[np.ndarray],
) -> dict[int, list[Any]]:
    _session(backend, sequence_dir)
    public_ids = [int(value) for value in case["public_ids"]]
    _bind_official_multi_clean(backend, int(case["start_frame"]), public_ids, prompt_boxes)
    _official_action_history(backend, public_ids, int(case["start_frame"]))
    capture.reset()
    return backend.propagate(
        int(case["start_frame"]),
        int(case["query_end"]),
        start_frame_index=int(case["start_frame"]),
    )


def _run_lora_update(
    *,
    backend: Any,
    adapter: SAM3DecoderLITAdapter,
    decoder: torch.nn.Module,
    case: Mapping[str, Any],
    support_kwargs: Optional[Mapping[str, Any]],
    current_output_recorded: bool,
    target_slot: int,
) -> tuple[Any, dict[str, Any]]:
    public_id = int(case["correction_public_id"])
    frame = int(case["interaction_frame"])
    state = adapter.new_state(
        f"{case['sequence']}:{case['case_id']}",
        public_id,
        device=adapter.device,
    )
    if support_kwargs is None:
        return state, {"status": "NOT_RUN", "committed": False, "reason": "support_inputs_not_exposed"}
    if not current_output_recorded:
        return state, {"status": "NOT_RUN", "committed": False, "reason": "current_output_not_recorded"}
    correction_box = np.asarray(case["correction_box"], dtype=float)
    event = DecoderCorrectionEvent(
        video_id=f"{case['sequence']}:{case['case_id']}",
        public_id=public_id,
        frame_idx=frame,
        provenance=BOX_DERIVED_PSEUDO_MASK,
        box_xyxy=correction_box,
        image_size=(int(backend._frame_h), int(backend._frame_w)),
        current_output_recorded=True,
        metadata={"branch": "M4", "target_slot": str(target_slot)},
    )
    config = DecoderUpdateConfig(
        inner_steps=5,
        learning_rate=1.0e-4,
        weight_decay=0.0,
        optimizer_enabled=True,
        require_loss_decrease=False,
        require_observable_update=True,
    )
    support_kwargs = _clone_tree(support_kwargs)

    def forward_fn(_supervision: Any, _step: int) -> torch.Tensor:
        with torch.inference_mode(False), torch.enable_grad():
            raw = decoder(**support_kwargs)
        return _slot_tensor(raw["masks"], slot=target_slot)

    def deterministic_forward(_supervision: Any) -> torch.Tensor:
        was_training = decoder.training
        decoder.eval()
        try:
            with torch.inference_mode(False), torch.no_grad():
                raw = decoder(**support_kwargs)
            return _slot_tensor(raw["masks"], slot=target_slot)
        finally:
            decoder.train(was_training)

    result = DecoderUpdateTransaction(adapter, config).apply(
        event,
        state,
        forward_fn=forward_fn,
        deterministic_forward_fn=deterministic_forward,
    )
    update = _update_dict(result)
    update["multi_identity_slot"] = int(target_slot)
    update["identity_isolation_note"] = "official decoder LoRA state is activated for the decoder call; per-slot isolation beyond target-slot supervision is not separately proven"
    return state, update


def _candidate_assignment(
    *,
    backend: Any,
    capture: DecoderCapture,
    clip: FrozenClipReID,
    gt: Mapping[int, Mapping[int, np.ndarray]],
    images: Sequence[Path],
    case: Mapping[str, Any],
    identity_data: list[dict[str, Any]],
    b10_memory_written: bool,
) -> dict[str, Any]:
    frame = int(case["query_end"])
    public_ids = [int(value) for value in case["public_ids"]]
    observations = []
    for observation in backend._output_cache.get(frame, []):
        if int(observation.sam_object_id) in public_ids:
            observations.append(observation)
    observation_by_public = {int(item.sam_object_id): item for item in observations}
    original_items = [observation_by_public[public_id] for public_id in public_ids if public_id in observation_by_public]
    original_features = (
        clip.encode(images[frame], [item.box_xyxy for item in original_items])
        if original_items
        else np.zeros((0, 1280), dtype=np.float32)
    )
    original = [
        original_candidate(item, frame, feature)
        for item, feature in zip(original_items, original_features)
    ]
    raw = capture.last_raw
    slot_ids = state_slot_ids(backend)
    decoder = decoder_candidates(raw, frame, slot_ids)
    if decoder:
        decoder_features = clip.encode(images[frame], [item.box_xyxy for item in decoder])
        decoder = [replace(item, clip_feature=feature) for item, feature in zip(decoder, decoder_features)]
    candidates = tuple([*original, *decoder])
    if not candidates:
        raise RuntimeError(f"no merged candidates at {case['case_id']}:{frame}")
    scores = b10_scores(candidates, identity_data)
    original_scores = b10_scores(original, identity_data) if original else np.zeros((len(public_ids), 0), dtype=np.float64)
    from sam3_intermot.association.decoder_candidate_bridge import build_decoder_assignment

    bridge = build_decoder_assignment(
        scores,
        candidates,
        public_ids,
        none_scores=np.zeros(len(public_ids), dtype=np.float64),
    )
    original_bridge = build_decoder_assignment(
        original_scores,
        original,
        public_ids,
        none_scores=np.zeros(len(public_ids), dtype=np.float64),
    )
    audit = assignment_audit(bridge, original_bridge, candidates, original, identity_data)
    assignment_rows: list[dict[str, Any]] = []
    correct_count = 0
    wrong_count = 0
    none_count = 0
    none_correct = 0
    for row_index, (identity, public_id) in enumerate(zip(case["identity_ids"], public_ids)):
        column = int(bridge.assignment.assignment[row_index])
        selected = None if column < 0 else candidates[column]
        target_box = gt[frame].get(int(identity))
        selected_iou = None if selected is None or target_box is None else box_iou(selected.box_xyxy, target_box)
        all_ious = [
            None if target_box is None else box_iou(candidate.box_xyxy, target_box)
            for candidate in candidates
        ]
        best_identity = None
        best_identity_iou = None
        if selected is not None:
            identity_ious = {
                int(other): box_iou(selected.box_xyxy, gt[frame][int(other)])
                for other in case["identity_ids"]
                if int(other) in gt.get(frame, {})
            }
            if identity_ious:
                best_identity, best_identity_iou = max(identity_ious.items(), key=lambda item: item[1])
        correct = bool(selected_iou is not None and selected_iou >= 0.5 and best_identity == int(identity))
        wrong = bool(selected is not None and best_identity is not None and best_identity != int(identity) and float(best_identity_iou) >= 0.5)
        if correct:
            correct_count += 1
        if wrong:
            wrong_count += 1
        if selected is None:
            none_count += 1
            if target_box is None or not any(value is not None and value >= 0.5 for value in all_ious):
                none_correct += 1
        original_column = int(original_bridge.assignment.assignment[row_index])
        original_selected = None if original_column < 0 else original[original_column]
        original_iou = None if original_selected is None or target_box is None else box_iou(original_selected.box_xyxy, target_box)
        assignment_rows.append({
            "dataset_identity": int(identity),
            "public_id": int(public_id),
            "selected_candidate": column,
            "selected_source": None if selected is None else selected.source,
            "selected_source_public_id": None if selected is None else selected.source_public_id,
            "selected_box_iou_to_gt": selected_iou,
            "best_identity_for_selected": best_identity,
            "best_identity_iou_for_selected": best_identity_iou,
            "correct_identity_assignment": correct,
            "wrong_public_id_assignment": wrong,
            "original_only_candidate": original_column,
            "original_only_box_iou_to_gt": original_iou,
        })
    target_count = len(public_ids)
    mean_iou = float(np.mean([row["selected_box_iou_to_gt"] for row in assignment_rows if row["selected_box_iou_to_gt"] is not None])) if any(row["selected_box_iou_to_gt"] is not None for row in assignment_rows) else None
    none_denominator = sum(int(identity) not in gt.get(frame, {}) for identity in case["identity_ids"])
    return {
        "frame": frame,
        "candidate_count": len(candidates),
        "original_candidate_count": len(original),
        "decoder_candidate_count": len(decoder),
        "candidate_sources": [candidate.source for candidate in candidates],
        "candidate_source_public_ids": [candidate.source_public_id for candidate in candidates],
        "official_decoder_raw_present": raw is not None,
        "official_slot_public_ids": slot_ids,
        "matrix_shape": list(bridge.matrix.shape),
        "matrix": bridge.matrix.tolist(),
        "assignment": bridge.assignment.assignment.tolist(),
        "assignment_selected_scores": bridge.assignment.selected_scores.tolist(),
        "original_only_assignment": original_bridge.assignment.assignment.tolist(),
        "assignment_rows": assignment_rows,
        "future_delivered_box_iou": mean_iou,
        "id_assignment_accuracy": float(correct_count / target_count) if target_count else None,
        "id_switch_count": wrong_count,
        "wrong_public_id_count": wrong_count,
        "none_selected_count": none_count,
        "none_precision": None if none_count == 0 else float(none_correct / none_count),
        "none_recall": None if none_denominator == 0 else float(none_correct / none_denominator),
        "b10_memory_written": bool(b10_memory_written),
        "unaffected_identity_regression": bool(audit["unaffected_identity_regression"]),
        "bridge_audit": audit,
    }


def _metrics_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "future_delivered_box_iou",
        "id_assignment_accuracy",
        "id_switch_count",
        "wrong_public_id_count",
        "none_precision",
        "none_recall",
    )
    return {
        key: None
        if left.get(key) is None or right.get(key) is None
        else float(left[key]) - float(right[key])
        for key in keys
    }


def _run_case(
    *,
    backend: Any,
    adapter: SAM3DecoderLITAdapter,
    decoder: torch.nn.Module,
    capture: DecoderCapture,
    clip: FrozenClipReID,
    case: Mapping[str, Any],
    train_root: Path,
) -> dict[str, Any]:
    sequence_dir = train_root / str(case["sequence"])
    gt = _read_gt(sequence_dir)
    images = _image_files(sequence_dir)
    start = int(case["start_frame"])
    interaction = int(case["interaction_frame"])
    query_end = int(case["query_end"])
    ids = [int(value) for value in case["identity_ids"]]
    public_ids = [int(value) for value in case["public_ids"]]
    prompt_boxes = [np.asarray(gt[start][identity], dtype=float) for identity in ids]
    correction_box = np.asarray(gt[interaction][int(case["correction_dataset_identity"])], dtype=float)

    identity_data: list[dict[str, Any]] = []
    initial_features = clip.encode(images[start], prompt_boxes)
    for identity, public_id, feature in zip(ids, public_ids, initial_features):
        memory = load_audited_negative_memory(str(case["sequence"]), identity, start)
        identity_data.append({
            "dataset_identity": identity,
            "public_id": public_id,
            "target_box": np.asarray(gt[query_end][identity], dtype=float),
            "root_feature": np.asarray(feature, dtype=np.float32),
            "positive_features": np.asarray([feature], dtype=np.float32),
            "negative_features": np.asarray(memory["negative_features"], dtype=np.float32),
            "latest_anchor_feature": None,
            "memory_audit": {key: value for key, value in memory.items() if key != "negative_features"},
        })

    branches: dict[str, Any] = {}
    for branch_name in BRANCHES:
        started = time.perf_counter()
        _session(backend, sequence_dir)
        binding = _bind_official_multi_clean(backend, start, public_ids, prompt_boxes)
        _official_action_history(backend, public_ids, start)
        capture.reset(target_call=max(0, interaction - start))
        pre_outputs = backend.propagate(start, interaction, start_frame_index=start)
        support_kwargs = None if capture.target_inputs is None else _clone_tree(capture.target_inputs)
        current_recorded = bool(interaction in pre_outputs)
        branch_data = _clone_identity_data(identity_data)
        memory = B10IdentityMemory(branch_data)
        correction_row = ids.index(int(case["correction_dataset_identity"]))
        memory_before = memory.snapshot()
        memory_write: dict[str, Any] = {"status": "NOT_RUN", "reason": "branch_has_no_b10_write"}
        official_write: dict[str, Any] = {"status": "NOT_RUN", "reason": "branch_has_no_official_spatial_write"}
        lora_state = None
        lora_update: dict[str, Any] = {"status": "NOT_RUN", "committed": False, "reason": "branch_has_no_lora_write"}
        if branch_name in ("M1_official_spatial_write_only", "M3_official_spatial_plus_b10", "M4_official_spatial_plus_b10_plus_online_lora"):
            backend.correct_object(
                interaction,
                int(case["correction_public_id"]),
                box_xyxy=correction_box,
            )
            official_write = {
                "status": "WRITTEN",
                "method": "official_backend_correct_object",
                "public_id": int(case["correction_public_id"]),
                "frame": interaction,
            }
        if branch_name in ("M2_b10_identity_write_only", "M3_official_spatial_plus_b10", "M4_official_spatial_plus_b10_plus_online_lora"):
            correction_feature = clip.encode(images[interaction], [correction_box])[0]
            memory_write = memory.write_positive(
                correction_row,
                correction_feature,
                frame=interaction,
                public_id=int(case["correction_public_id"]),
            )
        if branch_name == "M3_official_spatial_plus_b10" and official_write["status"] == "WRITTEN" and memory_write["status"] != "WRITTEN":
            memory.restore(memory_before)
            official_write["rollback_attempted"] = True
            official_write["rollback_status"] = "RESTART_REQUIRED_TO_RESTORE_OFFICIAL_STATE"
            raise RuntimeError("M3 atomic transaction failed during B10 write")
        if branch_name == "M4_official_spatial_plus_b10_plus_online_lora":
            slot_ids_before = state_slot_ids(backend)
            target_public = int(case["correction_public_id"])
            target_slot = slot_ids_before.index(target_public) if target_public in slot_ids_before else 0
            lora_state, lora_update = _run_lora_update(
                backend=backend,
                adapter=adapter,
                decoder=decoder,
                case={**case, "correction_box": correction_box.tolist()},
                support_kwargs=support_kwargs,
                current_output_recorded=current_recorded,
                target_slot=target_slot,
            )

        if branch_name in ("M0_no_correction", "M2_b10_identity_write_only"):
            future_outputs = _restart_anchor(backend, capture, sequence_dir, case, prompt_boxes)
            future_adapter_active = False
        elif branch_name == "M4_official_spatial_plus_b10_plus_online_lora" and bool(lora_update.get("committed", False)):
            with adapter.activate(lora_state):
                future_outputs = backend.propagate(interaction + 1, query_end, start_frame_index=interaction + 1)
            future_adapter_active = True
        else:
            future_outputs = backend.propagate(interaction + 1, query_end, start_frame_index=interaction + 1)
            future_adapter_active = False
        # The official stream writes into the backend cache; make the branch
        # output available to the common candidate builder without serializing
        # masks or tensors.
        for frame, observations in future_outputs.items():
            backend._output_cache[int(frame)] = observations
        assignment = _candidate_assignment(
            backend=backend,
            capture=capture,
            clip=clip,
            gt=gt,
            images=images,
            case=case,
            identity_data=branch_data,
            b10_memory_written=memory_write["status"] == "WRITTEN",
        )
        branches[branch_name] = {
            "status": "PASS" if lora_update.get("status") != "NOT_RUN" or branch_name != "M4_official_spatial_plus_b10_plus_online_lora" else "NOT_RUN",
            "current_output_recorded": current_recorded,
            "official_spatial_write": official_write,
            "b10_identity_write": memory_write,
            "b10_state_after": memory.state_summary(),
            "lora_update": lora_update,
            "future_adapter_active": future_adapter_active,
            "future_frame_count": len(future_outputs),
            "assignment": assignment,
            "timing": {
                "wall_seconds": float(time.perf_counter() - started),
                "peak_gpu_memory_allocated_bytes": None if not torch.cuda.is_available() else int(torch.cuda.max_memory_allocated()),
            },
        }
    paired = {}
    for horizon in ("query",):
        paired[horizon] = {
            "M1_minus_M0": _metrics_difference(branches["M1_official_spatial_write_only"]["assignment"], branches["M0_no_correction"]["assignment"]),
            "M2_minus_M0": _metrics_difference(branches["M2_b10_identity_write_only"]["assignment"], branches["M0_no_correction"]["assignment"]),
            "M3_minus_max_M1_M2": {
                key: None
                if branches["M3_official_spatial_plus_b10"]["assignment"].get(key) is None
                else float(branches["M3_official_spatial_plus_b10"]["assignment"][key])
                - max(
                    float(branches["M1_official_spatial_write_only"]["assignment"].get(key, -math.inf)),
                    float(branches["M2_b10_identity_write_only"]["assignment"].get(key, -math.inf)),
                )
                for key in ("future_delivered_box_iou", "id_assignment_accuracy", "id_switch_count", "wrong_public_id_count")
            },
            "M4_minus_M3": _metrics_difference(branches["M4_official_spatial_plus_b10_plus_online_lora"]["assignment"], branches["M3_official_spatial_plus_b10"]["assignment"]),
        }
    return {
        "status": "PASS" if all(branch["status"] == "PASS" for branch in branches.values()) else "PARTIAL",
        "case_id": str(case["case_id"]),
        "sequence": str(case["sequence"]),
        "split": str(case["split"]),
        "identity_ids": ids,
        "public_ids": public_ids,
        "correction_dataset_identity": int(case["correction_dataset_identity"]),
        "correction_public_id": int(case["correction_public_id"]),
        "start_frame": start,
        "interaction_frame": interaction,
        "query_end": query_end,
        "correction_type": "box",
        "supervision_provenance": BOX_DERIVED_PSEUDO_MASK,
        "official_multi_id_binding": binding,
        "branches": branches,
        "paired_delta": paired,
        "gt_role": "legal current correction box and post-hoc train-fold assignment/IoU audit only",
    }


def _bootstrap(values: Sequence[float], seed: int, draws: int = 2000) -> Optional[list[float]]:
    if not values:
        return None
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    index = rng.integers(0, len(data), size=(draws, len(data)))
    means = data[index].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _cluster_bootstrap(rows: Sequence[Mapping[str, Any]], seed: int, draws: int = 2000) -> Optional[list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get("value") is not None:
            grouped.setdefault(str(row["sequence"]), []).append(float(row["value"]))
    values = [float(np.mean(items)) for items in grouped.values() if items]
    return _bootstrap(values, seed, draws)


def _summary(results: Sequence[Mapping[str, Any]], manifest_path: Path, manifest_sha: str) -> dict[str, Any]:
    metrics = ("future_delivered_box_iou", "id_assignment_accuracy", "id_switch_count", "wrong_public_id_count", "none_precision", "none_recall")
    branch_metrics: dict[str, Any] = {}
    for branch in BRANCHES:
        branch_metrics[branch] = {}
        for metric in metrics:
            values = [
                row["branches"][branch]["assignment"].get(metric)
                for row in results
                if row.get("status") == "PASS" and row["branches"].get(branch, {}).get("assignment", {}).get(metric) is not None
            ]
            branch_metrics[branch][metric] = {"mean": None if not values else float(np.mean(values)), "sample_count": len(values)}
    pairs = {
        "M1_minus_M0": ("M1_official_spatial_write_only", "M0_no_correction"),
        "M2_minus_M0": ("M2_b10_identity_write_only", "M0_no_correction"),
        "M4_minus_M3": ("M4_official_spatial_plus_b10_plus_online_lora", "M3_official_spatial_plus_b10"),
    }
    comparisons: dict[str, Any] = {}
    for name, (left, right) in pairs.items():
        comparisons[name] = {}
        for metric in metrics:
            rows: list[dict[str, Any]] = []
            for result in results:
                if result.get("status") != "PASS":
                    continue
                a = result["branches"].get(left, {}).get("assignment", {}).get(metric)
                b = result["branches"].get(right, {}).get("assignment", {}).get(metric)
                if a is not None and b is not None:
                    rows.append({"sequence": result["sequence"], "value": float(a) - float(b)})
            values = [item["value"] for item in rows]
            comparisons[name][metric] = {
                "mean": None if not values else float(np.mean(values)),
                "sample_count": len(values),
                "negative_rate": None if not values else float(np.mean(np.asarray(values) < 0.0)),
                "episode_bootstrap_ci95": _bootstrap(values, 3001 + len(name) + len(metric)),
                "sequence_cluster_bootstrap_ci95": _cluster_bootstrap(rows, 7001 + len(name) + len(metric)),
            }
    m3_synergy_rows = []
    for result in results:
        if result.get("status") != "PASS":
            continue
        m3 = result["branches"]["M3_official_spatial_plus_b10"]["assignment"]
        m1 = result["branches"]["M1_official_spatial_write_only"]["assignment"]
        m2 = result["branches"]["M2_b10_identity_write_only"]["assignment"]
        for metric in ("future_delivered_box_iou", "id_assignment_accuracy"):
            if m3.get(metric) is not None and m1.get(metric) is not None and m2.get(metric) is not None:
                m3_synergy_rows.append({"sequence": result["sequence"], "metric": metric, "value": float(m3[metric]) - max(float(m1[metric]), float(m2[metric]))})
    comparisons["M3_minus_max_M1_M2"] = {
        metric: {
            "mean": None if not (values := [row["value"] for row in m3_synergy_rows if row["metric"] == metric]) else float(np.mean(values)),
            "sample_count": len([row for row in m3_synergy_rows if row["metric"] == metric]),
            "episode_bootstrap_ci95": _bootstrap([row["value"] for row in m3_synergy_rows if row["metric"] == metric], 5001 + len(metric)),
        }
        for metric in ("future_delivered_box_iou", "id_assignment_accuracy")
    }
    return {
        "protocol": "N30-B-MULTI-IDENTITY-WRITE-ABLATION",
        "status": "PASS" if results and all(result.get("status") == "PASS" for result in results) else "PARTIAL",
        "case_count": len(results),
        "branch_order": list(BRANCHES),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
        "candidate_protocol": "same public IDs x merged original+official decoder candidates plus per-row NONE; one global Hungarian per branch",
        "branch_metrics": branch_metrics,
        "comparisons": comparisons,
        "bootstrap": {"episode_unit_draws": 2000, "sequence_cluster_draws": 2000, "seed_base_episode": 3001, "seed_base_sequence": 7001},
    }


def run(
    *,
    manifest_path: Path,
    checkpoint: Path,
    output: Path,
    summary_output: Path,
    train_root: Path,
    case_limit: Optional[int],
    smoke: bool,
) -> dict[str, Any]:
    manifest, manifest_sha = _load_cases(manifest_path, train_root)
    cases = list(manifest["cases"])
    if case_limit is not None:
        cases = cases[:case_limit]
    if len(cases) < 10 and not smoke:
        raise ValueError("N30-B run requires at least 10 cases after smoke")
    backend = _make_backend(checkpoint)
    decoder = None
    adapter = None
    capture = None
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        # The CLIP model is frozen and is used only for B10 score features.
        clip = FrozenClipReID(torch.device("cuda"))
        for case in cases:
            if decoder is None:
                # Materialize the official model/decoder without opening a
                # throw-away video session.  Closing that bootstrap session
                # leaves stale multiplex tracker metadata in the pinned
                # model; the first real multi-ID propagation can then emit a
                # mask for raw object 0 while its score table is keyed by the
                # requested public IDs.  N29-R initializes the decoder from
                # the model-only path and starts the first real session only
                # inside the case runner, which is the causal initialization
                # we preserve here.
                backend._ensure_model()
                decoder = _get_official_decoder(backend)
                adapter = SAM3DecoderLITAdapter(decoder, DecoderLITConfig(rank=4, alpha=4.0, dropout=0.1))
                capture = DecoderCapture(decoder)
            if decoder is None or adapter is None or capture is None:
                raise RuntimeError("N30-B decoder adapter initialization failed")
            try:
                results.append(_run_case(backend=backend, adapter=adapter, decoder=decoder, capture=capture, clip=clip, case=case, train_root=train_root))
            except Exception as exc:
                results.append({
                    "status": "NOT_RUN",
                    "case_id": str(case["case_id"]),
                    "sequence": str(case["sequence"]),
                    "split": str(case["split"]),
                    "failure": f"{type(exc).__name__}: {exc}",
                    "failure_traceback": traceback.format_exc(limit=16),
                })
    finally:
        if capture is not None:
            capture.close()
        backend.close()
    result = {
        "protocol": "N30-B-MULTI-IDENTITY-WRITE-ABLATION",
        "status": "PASS" if len(results) == len(cases) and all(row.get("status") == "PASS" for row in results) else "PARTIAL" if results else "NOT_RUN",
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
        "selection_frozen_before_branches": True,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "case_count_requested": len(cases),
        "case_count_processed": len(results),
        "case_count_pass": int(sum(row.get("status") == "PASS" for row in results)),
        "case_count_failed": int(sum(row.get("status") != "PASS" for row in results)),
        "branch_order": list(BRANCHES),
        "b10": {"backbone": "frozen CLIP-ReID ViT-B/16 1280-D", "lambda_negative": 0.8, "margin": 0.02, "n26_memory_sha256": sha256(N26_ARRAYS)},
        "candidate_protocol": "merged original+official decoder candidates, source_public_id provenance only, one global Hungarian with NONE",
        "case_results": results,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _write_json(output, result)
    _write_json(summary_output, _summary(results, manifest_path, manifest_sha))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/n30/multi_identity_case_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/n30/multi_identity_write_ablation.json")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "outputs/n30/multi_identity_write_summary.json")
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="allow the explicit 3-case N30-B smoke run")
    args = parser.parse_args()
    if args.freeze_only:
        payload = _freeze_cases(args.manifest, args.train_root)
        print(json.dumps({"status": payload["status"], "case_count": payload["case_count"], "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest)}, indent=2))
        return 0
    result = run(
        manifest_path=args.manifest,
        checkpoint=args.checkpoint,
        output=args.output,
        summary_output=args.summary_output,
        train_root=args.train_root,
        case_limit=args.case_limit,
        smoke=args.smoke,
    )
    print(json.dumps({key: result[key] for key in ("status", "case_count_requested", "case_count_processed", "case_count_pass", "elapsed_seconds", "val25_read")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
