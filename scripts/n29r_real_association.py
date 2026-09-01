#!/usr/bin/env python3
"""N29-R5 real multi-identity decoder-candidate association smoke.

This is intentionally a small, train-fold-only association delivery rather
than a TrackEval claim.  It runs the pinned SAM3.1 propagation decoder on two
real DanceTrack identities, extracts frozen CLIP-ReID features for the
observed boxes, builds B10 scores with an audited pre-existing positive /
explicit-negative memory, merges original and decoder candidates, and solves
one coupled Hungarian matrix with per-identity NONE columns.

The three stress cases are predeclared below.  GT is read only for legal box
prompts and post-hoc auditing; it is never used to select a candidate or to
alter the assignment matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = ROOT / "third_party" / "sam3"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3_intermot.association.decoder_candidate_bridge import (  # noqa: E402
    DecoderCandidate,
    build_decoder_assignment,
    official_output_to_decoder_candidate,
)
from scripts.n29_lit_online_replay import (  # noqa: E402
    DecoderCapture,
    _get_official_decoder,
    _image_files,
    _make_backend,
    _read_gt,
    _session,
    _slot_output,
)


CHECKPOINT = ROOT / "checkpoints" / "sam3.1_mirror" / "sam3.1_multiplex.pt"
CLIP_CHECKPOINT = ROOT / "outputs" / "n15" / "checkpoints" / "clip_reid_hf" / "person_vit_clip_reid.pth"
N26_ARRAYS = ROOT / "outputs" / "n26" / "dense_dataset" / "round0_train30.npz"
N26_PARENTS = ROOT / "outputs" / "n26" / "dense_dataset" / "round0_train30_parents.jsonl"

# Frozen before this R5 runner was executed.  Each pair has a high-overlap
# train-fold frame and is used only to make the multi-ID smoke exercise
# non-trivial.  It is not a validation/test selection and does not affect
# candidate scores.
PREDECLARED_CASES = (
    {"sequence": "dancetrack0001", "start_frame": 79, "interaction_frame": 82, "identity_ids": (4, 5)},
    {"sequence": "dancetrack0002", "start_frame": 556, "interaction_frame": 559, "identity_ids": (0, 7)},
    {"sequence": "dancetrack0006", "start_frame": 1116, "interaction_frame": 1119, "identity_ids": (3, 5)},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(x) for x in a)
    bx1, by1, bx2, by2 = (float(x) for x in b)
    x1, y1, x2, y2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (aa + bb - inter)) if aa + bb - inter > 0 else 0.0


def crop(image: Image.Image, box: Sequence[float]) -> Image.Image:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    return image.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else Image.new("RGB", (8, 8))


class FrozenClipReID:
    """The same frozen 1280-D CLIP-ReID representation used by B10."""

    def __init__(self, device: torch.device) -> None:
        from scripts.run_n15_extract_features import build_clipreid

        self.device = device
        self.model = build_clipreid(str(CLIP_CHECKPOINT), device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.transform = T.Compose(
            [
                T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def encode(self, image_path: Path, boxes: Sequence[Sequence[float]]) -> np.ndarray:
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            tensors = torch.stack([self.transform(crop(image, box)) for box in boxes]).to(self.device)
        with torch.inference_mode():
            _, x12, xproj = self.model(tensors)
            values = F.normalize(torch.cat([x12[:, 0], xproj[:, 0]], dim=1).float(), dim=1)
        result = values.cpu().numpy().astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("frozen CLIP-ReID emitted a non-finite feature")
        return result


def _rectangle_masks(boxes: Sequence[Sequence[float]], height: int, width: int, device: torch.device) -> torch.Tensor:
    masks = torch.zeros((len(boxes), height, width), dtype=torch.float32, device=device)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (float(value) for value in box)
        left = max(0, min(width - 1, int(np.floor(x1))))
        top = max(0, min(height - 1, int(np.floor(y1))))
        right = max(left + 1, min(width, int(np.ceil(x2))))
        bottom = max(top + 1, min(height, int(np.ceil(y2))))
        masks[index, top:bottom, left:right] = 1.0
    return masks


def bind_official_multi(backend: Any, frame_idx: int, public_ids: Sequence[int], boxes: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Install a legal box-rectangle prompt for several public objects."""

    predictor = backend._predictor
    model = predictor.model
    state = predictor._all_inference_states[backend._session_id]["state"]
    ids = [int(value) for value in public_ids]
    if len(ids) != len(boxes) or len(set(ids)) != len(ids):
        raise ValueError("multi-ID binding IDs/boxes are not one-to-one")
    # Let the official semantic prompt path materialize the requested frame's
    # image cache first.  The pinned low-level ``_tracker_add_new_objects``
    # helper used below expects that cache even when the semantic prompt did
    # not produce a tracker masklet for a large DanceTrack box.
    for object_id, box in zip(ids, boxes):
        backend.add_box(frame_idx, object_id, np.asarray(box, dtype=float))
    state = predictor._all_inference_states[backend._session_id]["state"]
    state["sam2_inference_states"] = []
    device = state["device"]
    rectangles = _rectangle_masks(boxes, int(state["orig_height"]), int(state["orig_width"]), device)
    tracker_states = model._tracker_add_new_objects(
        frame_idx=int(frame_idx),
        num_frames=int(state["num_frames"]),
        new_obj_ids=ids,
        new_obj_masks=rectangles,
        tracker_states_local=[],
        orig_vid_height=int(state["orig_height"]),
        orig_vid_width=int(state["orig_width"]),
        feature_cache=state["feature_cache"],
    )
    observed_ids = [int(value) for item in tracker_states for value in np.asarray(item.get("obj_ids", [])).reshape(-1)]
    if set(observed_ids) != set(ids):
        raise RuntimeError(f"official multi-ID binding mismatch: requested={ids}, observed={observed_ids}")
    state["sam2_inference_states"] = tracker_states

    metadata = model._initialize_metadata()
    rank = int(getattr(model, "rank", 0))
    metadata["obj_ids_per_gpu"][rank] = np.asarray(ids, dtype=np.int64)
    metadata["num_obj_per_gpu"][rank] = len(ids)
    metadata["obj_ids_all_gpu"] = np.asarray(ids, dtype=np.int64)
    metadata["max_obj_id"] = max(ids)
    for object_id in ids:
        metadata["obj_id_to_score"][object_id] = 1.0
        metadata["obj_id_to_sam2_score_frame_wise"][int(frame_idx)][object_id] = torch.tensor(
            1.0, dtype=torch.float32, device=device
        )
        metadata["rank0_metadata"]["obj_first_frame_idx"][object_id] = int(frame_idx)
        metadata["rank0_metadata"]["trk_keep_alive"][object_id] = int(model.init_trk_keep_alive)
    metadata["gpu_metadata"] = {
        "N_obj": len(ids),
        "obj_first_frame": torch.full((len(ids),), int(frame_idx), dtype=torch.long, device=device),
        "consecutive_unmatch_count": torch.zeros(len(ids), dtype=torch.long, device=device),
        "trk_keep_alive": torch.full((len(ids),), int(model.init_trk_keep_alive), dtype=torch.long, device=device),
        "removed_mask": torch.zeros(len(ids), dtype=torch.bool, device=device),
        "overlap_pair_counts": torch.zeros((len(ids), len(ids)), dtype=torch.long, device=device),
        "last_occluded_tensor": torch.full((len(ids),), -1, dtype=torch.long, device=device),
    }
    metadata["num_buc_per_gpu"][rank] = model._count_buckets_in_states(tracker_states)
    state["tracker_metadata"] = metadata
    backend._objects = {
        object_id: {
            "box": np.asarray(box, dtype=float).copy(),
            "human_box": np.asarray(box, dtype=float).copy(),
            "frame": int(frame_idx),
            "source": "human_add",
        }
        for object_id, box in zip(ids, boxes)
    }
    backend._ext_to_sam = {object_id: object_id for object_id in ids}
    backend._sam_to_ext = {object_id: object_id for object_id in ids}
    backend._last_prompt_frame = int(frame_idx)
    return {
        "status": "BOUND",
        "frame": int(frame_idx),
        "public_ids": ids,
        "official_tracker_ids": observed_ids,
        "tracker_state_count": len(tracker_states),
        "mask_source": "explicit_box_rectangle",
        "provenance": "BOX_DERIVED_PSEUDO_MASK",
    }


def slot_count(raw: Optional[Mapping[str, Any]]) -> int:
    if not raw or raw.get("masks") is None:
        return 0
    values = np.asarray(raw["masks"])
    if values.ndim >= 5:
        return int(values.shape[1])
    if values.ndim == 4:
        return int(values.shape[1]) if values.shape[0] == 1 else int(values.shape[0])
    return 0


def state_slot_ids(backend: Any) -> list[int]:
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    metadata = state.get("tracker_metadata", {})
    values = metadata.get("obj_ids_all_gpu")
    if values is not None:
        return [int(value) for value in np.asarray(values).reshape(-1)]
    return [int(value) for item in state.get("sam2_inference_states", []) for value in np.asarray(item.get("obj_ids", [])).reshape(-1)]


def original_candidate(observation: Any, frame_idx: int, feature: np.ndarray) -> DecoderCandidate:
    mask = np.asarray(observation.mask, dtype=bool)
    return DecoderCandidate(
        frame_idx=int(frame_idx),
        mask_logits=mask.astype(np.float32),
        mask=mask,
        box_xyxy=tuple(float(value) for value in observation.box_xyxy),
        presence=float(observation.presence_score or observation.confidence),
        iou_pred=float(observation.confidence),
        decoder_token=None,
        clip_feature=feature.copy(),
        source="original_anchor",
        source_public_id=None,
    )


def decoder_candidates(raw: Optional[Mapping[str, Any]], frame_idx: int, public_ids: Sequence[int]) -> list[DecoderCandidate]:
    count = min(slot_count(raw), len(public_ids))
    result: list[DecoderCandidate] = []
    for slot in range(count):
        candidate = official_output_to_decoder_candidate(
            _slot_output(raw, slot=slot),
            frame_idx=frame_idx,
            source_public_id=int(public_ids[slot]),
            source="sam3_decoder_official",
            min_presence=0.0,
            keep_rejected=True,
        )
        if candidate is not None:
            result.append(candidate)
    return result


def load_audited_negative_memory(sequence: str, identity: int, cutoff_frame: int) -> dict[str, Any]:
    """Load only a train-fold explicit-negative prefix from N26's frozen B10 data."""

    with N26_ARRAYS.open("rb"):
        payload = np.load(N26_ARRAYS, allow_pickle=False)
        memory_clip = payload["memory_clip"]
        memory_kind = payload["memory_kind"]
        memory_mask = payload["memory_mask"]
    matches: list[dict[str, Any]] = []
    parent_index = None
    for index, line in enumerate(N26_PARENTS.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("sequence") != sequence or int(row.get("gid", -1)) != int(identity):
            continue
        if int(row.get("frame", 10**9)) >= int(cutoff_frame) or not row.get("explicit_negative_written"):
            continue
        canonical = int(row["parent_event_id"])
        if canonical >= len(memory_clip):
            continue
        negative = memory_clip[canonical][memory_mask[canonical] & (memory_kind[canonical] == 2)].astype(np.float32)
        if len(negative):
            parent_index = canonical
            matches.append({
                "source": "outputs/n26/dense_dataset/round0_train30.npz",
                "parent_event_id": int(row["parent_event_id"]),
                "event_key": row["event_key"],
                "frame": int(row["frame"]),
                "negative_count": int(len(negative)),
                "negative_features": negative,
            })
            break
    if not matches:
        return {
            "positive_count": 0,
            "negative_count": 0,
            "negative_features": np.zeros((0, 1280), dtype=np.float32),
            "source": "NONE_AVAILABLE_BEFORE_R5_ROLLOUT",
        }
    return {
        "positive_count": 0,
        "negative_count": matches[0]["negative_count"],
        "negative_features": matches[0]["negative_features"],
        "source": matches[0]["source"],
        "event_key": matches[0]["event_key"],
        "frame": matches[0]["frame"],
        "parent_event_id": matches[0]["parent_event_id"],
    }


def b10_scores(candidates: Sequence[DecoderCandidate], identity_data: Sequence[dict[str, Any]]) -> np.ndarray:
    matrix = np.zeros((len(identity_data), len(candidates)), dtype=np.float64)
    for row, item in enumerate(identity_data):
        root = item["root_feature"]
        positives = item["positive_features"]
        negatives = item["negative_features"]
        values = np.asarray([candidate.clip_feature for candidate in candidates], dtype=np.float32)
        root_similarity = values @ root
        positive_similarity = np.max(values @ positives.T, axis=1) if len(positives) else np.zeros(len(values), dtype=np.float32)
        negative_similarity = np.max(values @ negatives.T, axis=1) if len(negatives) else np.zeros(len(values), dtype=np.float32)
        positive_base = np.maximum(root_similarity, positive_similarity) if len(positives) else root_similarity
        penalty = np.maximum(0.0, negative_similarity - positive_base + 0.02) if len(negatives) else np.zeros(len(values), dtype=np.float32)
        matrix[row] = positive_base - 0.8 * penalty
    return matrix


def assignment_audit(
    bridge: Any,
    original_bridge: Any,
    candidates: Sequence[DecoderCandidate],
    original: Sequence[DecoderCandidate],
    identity_data: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for index, item in enumerate(identity_data):
        target_box = item["target_box"]
        column = int(bridge.assignment.assignment[index])
        original_column = int(original_bridge.assignment.assignment[index])
        selected = None if column < 0 else candidates[column]
        selected_original = None if original_column < 0 else original[original_column]
        rows.append({
            "public_id": int(item["public_id"]),
            "dataset_identity": int(item["dataset_identity"]),
            "selected_candidate": column,
            "selected_source": None if selected is None else selected.source,
            "selected_source_public_id": None if selected is None else selected.source_public_id,
            "selected_box_iou_to_gt": None if selected is None else box_iou(selected.box_xyxy, target_box),
            "original_only_candidate": original_column,
            "original_only_source": None if selected_original is None else selected_original.source,
            "original_only_box_iou_to_gt": None if selected_original is None else box_iou(selected_original.box_xyxy, target_box),
        })
    return {"rows": rows, "unaffected_identity_regression": any(
        row["selected_box_iou_to_gt"] is not None
        and row["original_only_box_iou_to_gt"] is not None
        and float(row["selected_box_iou_to_gt"]) + 1e-12 < float(row["original_only_box_iou_to_gt"])
        for row in rows
    )}


def run_case(case: Mapping[str, Any], backend: Any, clip: FrozenClipReID, capture: DecoderCapture, train_root: Path) -> dict[str, Any]:
    sequence = str(case["sequence"])
    sequence_dir = train_root / sequence
    gt = _read_gt(sequence_dir)
    images = _image_files(sequence_dir)
    start = int(case["start_frame"])
    interaction = int(case["interaction_frame"])
    end = interaction + 3
    dataset_ids = [int(value) for value in case["identity_ids"]]
    if end >= len(images) or any(identity not in gt.get(start, {}) for identity in dataset_ids):
        raise RuntimeError(f"predeclared case is unavailable: {case}")
    if any(identity not in gt.get(interaction, {}) for identity in dataset_ids):
        raise RuntimeError(f"predeclared interaction frame is unavailable: {case}")
    if any(identity not in gt.get(end, {}) for identity in dataset_ids):
        raise RuntimeError(f"predeclared trace frame is unavailable: {case}")
    target_boxes = [np.asarray(gt[end][identity], dtype=float) for identity in dataset_ids]
    prompt_boxes = [np.asarray(gt[start][identity], dtype=float) for identity in dataset_ids]
    public_ids = [1000 + identity for identity in dataset_ids]

    _session(backend, sequence_dir)
    binding = bind_official_multi(backend, start, public_ids, prompt_boxes)
    trace = []
    identity_data = []
    initial_features = clip.encode(images[start], prompt_boxes)
    for identity, public_id, target_box, root_feature in zip(dataset_ids, public_ids, target_boxes, initial_features):
        memory = load_audited_negative_memory(sequence, identity, start)
        identity_data.append({
            "dataset_identity": identity,
            "public_id": public_id,
            "target_box": target_box,
            "root_feature": root_feature,
            "positive_features": np.asarray([root_feature], dtype=np.float32),
            "negative_features": memory["negative_features"],
            "memory_audit": {key: value for key, value in memory.items() if key != "negative_features"},
        })

    # The official semantic prompt path may precompute VG predictions for the
    # whole video.  Force one genuine SAM2/decoder partial propagation from
    # the multi-object prompt, so the hook observes the decoder rather than a
    # fetch-only cache path.  The final frame is the audited bridge frame.
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    state["action_history"] = [{"type": "add", "obj_ids": list(public_ids), "frame_idx": start}]
    capture.reset()
    outputs = backend.propagate(start, end, start_frame_index=start)
    frame = end
    observations = outputs.get(frame, [])
    observation_by_public = {
        int(observation.sam_object_id): observation
        for observation in observations
        if int(observation.sam_object_id) in public_ids
    }
    original_items = [
        observation_by_public[public_id]
        for public_id in public_ids
        if public_id in observation_by_public
    ]
    original_features = clip.encode(
        images[frame],
        [observation.box_xyxy for observation in original_items] or [prompt_boxes[0]],
    )
    original = [
        original_candidate(observation, frame, feature)
        for observation, feature in zip(original_items, original_features)
    ]
    raw = capture.last_raw
    slot_ids = state_slot_ids(backend)
    decoder = decoder_candidates(raw, frame, slot_ids)
    if decoder:
        decoder_features = clip.encode(images[frame], [candidate.box_xyxy for candidate in decoder])
        decoder = [
            replace(candidate, clip_feature=feature)
            for candidate, feature in zip(decoder, decoder_features)
        ]
    candidates = tuple([*original, *decoder])
    if not candidates:
        raise RuntimeError(f"no valid original/decoder candidates at {sequence}:{frame}")
    # Preserve the actual source order used in the matrix: first original
    # candidates, then decoder candidates.  Missing original observations are
    # represented by NONE rather than guessed boxes.
    scores = b10_scores(candidates, identity_data)
    original_scores = (
        b10_scores(original, identity_data)
        if original
        else np.zeros((len(public_ids), 0), dtype=np.float64)
    )
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
    trace.append({
        "frame": frame,
        "interaction_frame": interaction,
        "official_observation_count": len(observations),
        "decoder_hook_calls": capture.call_count,
        "official_decoder_raw_present": raw is not None,
        "official_slot_public_ids": slot_ids,
        "original_candidate_count": len(original),
        "decoder_candidate_count": len(decoder),
        "candidate_sources": [candidate.source for candidate in candidates],
        "candidate_source_public_ids": [candidate.source_public_id for candidate in candidates],
        "b10_anchor_scores": scores.tolist(),
        "matrix_shape": list(bridge.matrix.shape),
        "matrix": bridge.matrix.tolist(),
        "assignment": bridge.assignment.assignment.tolist(),
        "assignment_selected_scores": bridge.assignment.selected_scores.tolist(),
        "original_only_assignment": original_bridge.assignment.assignment.tolist(),
        **audit,
    })
    max_overlap = max(
        box_iou(gt[interaction][dataset_ids[0]], gt[interaction][dataset_ids[1]]),
        0.0,
    )
    return {
        "status": "PASS",
        "sequence": sequence,
        "split": "train/train_fold",
        "case": dict(case),
        "interaction_box_iou": max_overlap,
        "identity_bindings": [
            {
                "dataset_identity": int(item["dataset_identity"]),
                "public_id": int(item["public_id"]),
                "sam_object_id": int(item["public_id"]),
                "memory": item["memory_audit"],
            }
            for item in identity_data
        ],
        "official_multi_id_binding": binding,
        "trace": trace,
        "unaffected_identity_regression": bool(any(row["unaffected_identity_regression"] for row in trace)),
        "gt_role": "post_hoc_box_iou_audit_and_legal_train_box_prompt_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=9)
    parser.add_argument("--case-limit", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "n29r" / "association_results.json")
    args = parser.parse_args()
    if not CHECKPOINT.is_file() or not CLIP_CHECKPOINT.is_file() or not N26_ARRAYS.is_file() or not N26_PARENTS.is_file():
        raise FileNotFoundError("R5 requires the verified SAM3 checkpoint, frozen CLIP checkpoint, and N26 memory audit")
    train_root = Path("/path/to/dancetrack/train")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    started = time.perf_counter()
    result: dict[str, Any] = {
        "protocol": "N29-R5",
        "status": "NOT_RUN",
        "delivery_status": "REAL_ASSOCIATION",
        "val25_read": False,
        "test_labels_used": False,
        "trackeval_authorized": False,
        "relation_cache_used": False,
        "random_relation_features_used": False,
        "candidate_bridge": "merge original+official decoder candidates then one global Hungarian with NONE",
        "b10": {
            "backbone": "frozen CLIP-ReID ViT-B/16 1280-D",
            "lambda_negative": 0.8,
            "margin": 0.02,
            "memory_source": str(N26_ARRAYS.relative_to(ROOT)),
            "memory_source_sha256": sha256(N26_ARRAYS),
        },
        "predeclared_cases": [dict(case) for case in PREDECLARED_CASES[: args.case_limit]],
    }
    backend = None
    capture = None
    try:
        clip = FrozenClipReID(device)
        backend = _make_backend(CHECKPOINT)
        backend._ensure_model()
        decoder = _get_official_decoder(backend)
        capture = DecoderCapture(decoder)
        cases = []
        for case in PREDECLARED_CASES[: args.case_limit]:
            print(f"R5_CASE_START {case['sequence']} {case['identity_ids']} frame={case['interaction_frame']}", flush=True)
            cases.append(run_case(case, backend, clip, capture, train_root))
            print(f"R5_CASE_DONE {case['sequence']}", flush=True)
        if len(cases) < 3:
            raise RuntimeError("R5 requires three real train-fold multi-ID cases")
        if any(not row["trace"] or not any(item["decoder_candidate_count"] > 0 for item in row["trace"]) for row in cases):
            raise RuntimeError("R5 did not deliver an official decoder candidate in every case")
        result.update({
            "status": "PASS",
            "cases": cases,
            "case_count": len(cases),
            "identity_count": sum(len(row["identity_bindings"]) for row in cases),
            "decoder_candidate_frames": sum(
                sum(item["decoder_candidate_count"] > 0 for item in row["trace"])
                for row in cases
            ),
            "unaffected_identity_regression": any(row["unaffected_identity_regression"] for row in cases),
            "elapsed_seconds": time.perf_counter() - started,
        })
    except Exception as exc:
        result.update({
            "status": "NOT_RUN",
            "reason": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.perf_counter() - started,
        })
        write_json(args.output, result)
        raise
    finally:
        if capture is not None:
            capture.close()
        if backend is not None:
            backend.close()
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
