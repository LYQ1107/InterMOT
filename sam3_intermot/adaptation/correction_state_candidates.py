"""Official, target-scoped correction-state writers used by N31.

The functions here deliberately sit above the pinned SAM3 checkout.  A mask
write is sent to the official ``VideoTrackingMultiplexDemo.add_new_masks``
with ``reconditioning=True`` and exactly one raw object ID; no tracker source
file is patched and no other public identity is guessed.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from sam3_intermot.backend.sam3_state_snapshot import (
    restore_continuation_state,
    snapshot_continuation_state,
)


BOX_RECTANGLE_MASKLET = "BOX_RECTANGLE_MASKLET"
BOX_SANITIZED_RECTANGLE_MASKLET = "BOX_SANITIZED_RECTANGLE_MASKLET"
BOX_PROMPTED_SAM_PSEUDO_MASK = "BOX_PROMPTED_SAM_PSEUDO_MASK"


def _state(backend: Any) -> dict[str, Any]:
    predictor = backend._predictor
    entry = predictor._all_inference_states[backend._session_id]
    return entry["state"]


def tracker_ids(backend: Any) -> list[int]:
    result: list[int] = []
    for tracker_state in _state(backend).get("sam2_inference_states", []):
        result.extend(int(value) for value in np.asarray(tracker_state.get("obj_ids", [])).reshape(-1))
    return result


def find_tracker_state(backend: Any, public_id: int) -> tuple[int, Mapping[str, Any]]:
    """Resolve one public ID to one official tracker state and raw ID."""

    public_id = int(public_id)
    mapped = backend._ext_to_sam.get(public_id)
    matches: list[tuple[int, Mapping[str, Any]]] = []
    for tracker_state in _state(backend).get("sam2_inference_states", []):
        ids = [int(value) for value in np.asarray(tracker_state.get("obj_ids", [])).reshape(-1)]
        for raw_id in ids:
            if (mapped is not None and raw_id == int(mapped)) or (mapped is None and raw_id == public_id):
                matches.append((raw_id, tracker_state))
    if len(matches) != 1:
        raise RuntimeError(
            f"target-scoped writer requires one official state for public_id={public_id}; "
            f"mapping={mapped}, matches={len(matches)}, all_ids={tracker_ids(backend)}"
        )
    return matches[0]


def rectangle_mask(box_xyxy: Sequence[float], height: int, width: int, *, device: Any) -> torch.Tensor:
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    mask = torch.zeros((height, width), dtype=torch.float32, device=device)
    mask[top:bottom, left:right] = 1.0
    return mask


def _state_id_signature(tracker_state: Mapping[str, Any], *, raw_id: int) -> dict[str, Any]:
    """Small per-ID structure signature; tensors are intentionally omitted."""

    signature: dict[str, Any] = {
        "obj_ids": [int(value) for value in np.asarray(tracker_state.get("obj_ids", [])).reshape(-1)],
        "keys": sorted(str(key) for key in tracker_state.keys()),
    }
    for key in ("obj_id_to_idx", "obj_idx_to_id", "first_ann_frame_idx"):
        value = tracker_state.get(key)
        if isinstance(value, Mapping):
            signature[key] = {str(k): int(v) if isinstance(v, (int, np.integer)) else str(v) for k, v in value.items()}
    for key in ("mask_inputs_per_obj", "point_inputs_per_obj", "output_dict_per_obj", "temp_output_dict_per_obj"):
        value = tracker_state.get(key)
        if isinstance(value, Mapping):
            obj_to_idx = tracker_state.get("obj_id_to_idx", {})
            obj_index = obj_to_idx.get(raw_id, obj_to_idx.get(str(raw_id))) if isinstance(obj_to_idx, Mapping) else None
            selected = value.get(raw_id, value.get(str(raw_id)))
            if selected is None and obj_index is not None:
                selected = value.get(obj_index, value.get(str(obj_index)))
            signature[key] = {
                str(raw_id): sorted(str(frame) for frame in selected.keys()) if isinstance(selected, Mapping) else str(type(selected).__name__)
            }
        elif isinstance(value, (list, tuple)):
            obj_to_idx = tracker_state.get("obj_id_to_idx", {})
            obj_index = obj_to_idx.get(raw_id, obj_to_idx.get(str(raw_id))) if isinstance(obj_to_idx, Mapping) else None
            selected = value[int(obj_index)] if obj_index is not None and 0 <= int(obj_index) < len(value) else None
            signature[key] = [
                sorted(str(frame) for frame in selected.keys()) if isinstance(selected, Mapping) else str(type(selected).__name__)
            ]
    return signature


def protected_state_signatures(backend: Any, *, exclude_public_id: int) -> dict[str, Any]:
    """Return per-raw-ID signatures for all non-target tracker objects."""

    target = int(exclude_public_id)
    mapped_target = backend._ext_to_sam.get(target, target)
    result: dict[str, Any] = {}
    for index, tracker_state in enumerate(_state(backend).get("sam2_inference_states", [])):
        ids = [int(value) for value in np.asarray(tracker_state.get("obj_ids", [])).reshape(-1)]
        if mapped_target in ids:
            keep_ids = [value for value in ids if value != mapped_target]
        else:
            keep_ids = ids
        for raw_id in keep_ids:
            result[str(raw_id)] = {
                "state_index": int(index),
                "state": _state_id_signature(tracker_state, raw_id=raw_id),
            }
    return result


def write_target_mask(
    backend: Any,
    *,
    frame_idx: int,
    public_id: int,
    mask: torch.Tensor,
    provenance: str,
) -> dict[str, Any]:
    """Write one masklet through official reconditioning and audit scope."""

    before_ids = tracker_ids(backend)
    protected_before = protected_state_signatures(backend, exclude_public_id=public_id)
    rollback = snapshot_continuation_state(backend)
    raw_id, tracker_state = find_tracker_state(backend, public_id)
    state = _state(backend)
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
        raise ValueError("target-scoped mask writer expects a two-dimensional tensor")
    device = state.get("device", mask.device)
    input_mask = mask.detach().to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    # The outer multiplex predictor stores the already computed frame as
    # ``feature_cache[frame]``.  The official demo tracker state uses the
    # same tuple under its per-state ``cached_features`` name; wiring the
    # existing tuple across avoids a second image read and keeps the write on
    # the official add_new_masks path.  This is an adapter cache bridge, not
    # a model/source modification.
    feature_entry = _outer_feature_cache_entry(backend, int(frame_idx))
    cached_features = tracker_state.setdefault("cached_features", {})
    cached_features[int(frame_idx)] = feature_entry
    try:
        backend._predictor.model.tracker.add_new_masks(
            inference_state=tracker_state,
            frame_idx=int(frame_idx),
            obj_ids=[int(raw_id)],
            masks=input_mask.unsqueeze(0),
            add_mask_to_memory=False,
            reconditioning=True,
        )
    except Exception:
        restore_continuation_state(backend, rollback)
        raise
    after_ids = tracker_ids(backend)
    protected_after = protected_state_signatures(backend, exclude_public_id=public_id)
    protected_ids_unchanged = sorted(before_ids) == sorted(after_ids) and set(protected_before) == set(protected_after)
    return {
        "status": "WRITTEN",
        "frame": int(frame_idx),
        "public_id": int(public_id),
        "raw_id": int(raw_id),
        "provenance": str(provenance),
        "method": "official_tracker.add_new_masks_reconditioning",
        "feature_cache_bridge": "outer_feature_cache_to_inner_cached_features",
        "mask_shape": list(input_mask.shape),
        "mask_area": float((input_mask > 0.5).sum().item()),
        "object_ids_before": sorted(before_ids),
        "object_ids_after": sorted(after_ids),
        "target_state_present": int(raw_id) in after_ids,
        "protected_ids_before": sorted(int(value) for value in protected_before),
        "protected_ids_after": sorted(int(value) for value in protected_after),
        "protected_identity_namespace_unchanged": bool(protected_ids_unchanged),
        "protected_state_before": protected_before,
        "protected_state_after": protected_after,
    }


def _outer_feature_cache_entry(backend: Any, frame_idx: int) -> Any:
    state = _state(backend)
    entry = state.get("feature_cache", {}).get(int(frame_idx))
    if entry is None:
        # The official streaming path keeps only a one-frame outer cache and
        # may evict the last yielded frame while its generator advances one
        # step ahead.  Recompute exactly that already-observed current frame
        # through the official model helper before invoking the official
        # reconditioning API; this never consults a future label/frame.
        prepare = getattr(backend._predictor.model, "_prepare_backbone_feats", None)
        if prepare is not None:
            with torch.inference_mode():
                prepare(state, int(frame_idx), False)
            entry = state.get("feature_cache", {}).get(int(frame_idx))
    if entry is None:
        raise RuntimeError(f"official feature cache is unavailable at frame {frame_idx}")
    return entry


def _feature_cache_entry(backend: Any, frame_idx: int) -> Any:
    entry = _outer_feature_cache_entry(backend, int(frame_idx))
    if isinstance(entry, (tuple, list)) and len(entry) >= 2:
        return entry[1]
    return entry


def interactive_box_candidates(
    backend: Any,
    *,
    frame_idx: int,
    box_xyxy: Sequence[float],
) -> dict[str, Any]:
    """Run the local official interactive SAM decoder for one box.

    The returned masks are detached tensors for immediate writes.  No ground
    truth or future frame is accessed; the only ranking signal is the
    decoder's own predicted IoU head.
    """

    model = backend._predictor.model
    tracker = model.tracker
    prompt_encoder = getattr(tracker, "interactive_sam_prompt_encoder", None)
    mask_decoder = getattr(tracker, "interactive_sam_mask_decoder", None)
    if prompt_encoder is None or mask_decoder is None:
        raise RuntimeError("official interactive_sam prompt encoder/decoder is unavailable")
    state = _state(backend)
    raw_cache = _feature_cache_entry(backend, int(frame_idx))
    prepared = tracker._prepare_backbone_features(raw_cache)
    interactive = prepared.get("interactive") if isinstance(prepared, Mapping) else None
    if interactive is None:
        raise RuntimeError("feature cache has no official interactive neck")
    vision_feats = interactive["vision_feats"]
    feat_sizes = interactive["feat_sizes"]
    pix_feat = tracker._get_interactive_pix_mem(vision_feats, feat_sizes)
    high_res_features = None
    if len(vision_feats) > 1:
        high_res_features = [
            feature.permute(1, 2, 0).view(feature.size(1), feature.size(2), *size)
            for feature, size in zip(vision_feats[:-1], feat_sizes[:-1])
        ]
    input_size = getattr(prompt_encoder, "input_image_size", (1008, 1008))
    input_h, input_w = int(input_size[0]), int(input_size[1])
    video_h, video_w = int(state["orig_height"]), int(state["orig_width"])
    box = torch.as_tensor(box_xyxy, device=pix_feat.device, dtype=torch.float32).reshape(1, 4)
    scale = box.new_tensor([input_w / video_w, input_h / video_h, input_w / video_w, input_h / video_h])
    scaled_box = box * scale
    point_coords = torch.zeros((1, 1, 2), device=pix_feat.device, dtype=torch.float32)
    point_labels = -torch.ones((1, 1), device=pix_feat.device, dtype=torch.int32)
    with torch.inference_mode():
        sparse, dense = prompt_encoder(
            points=(point_coords, point_labels),
            boxes=scaled_box,
            masks=None,
        )
        low_res, ious, _, _ = mask_decoder(
            image_embeddings=pix_feat,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=True,
            high_res_features=high_res_features,
        )
        probabilities = F.interpolate(
            low_res.float(),
            size=(video_h, video_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        predicted_iou = ious.reshape(ious.shape[0], -1)[0].float()
        masks = [probabilities[0, index].detach().clone() for index in range(probabilities.shape[1])]
        iou_values = [float(value.detach().cpu()) for value in predicted_iou[: len(masks)]]
    if not masks:
        raise RuntimeError("official interactive decoder returned no mask token")
    order = sorted(range(len(masks)), key=lambda index: (-iou_values[index], index))
    return {
        "status": "AVAILABLE",
        "frame": int(frame_idx),
        "box_xyxy": [float(value) for value in box_xyxy],
        "token_count": len(masks),
        "predicted_iou": iou_values,
        "rank_order": order,
        "masks": masks,
        "quality_available": True,
        "decoder": "local_official_interactive_sam_mask_decoder",
    }


def candidate_mask_features(candidate: Mapping[str, Any]) -> list[float]:
    """Small causal selector features (no public ID and no future labels)."""

    box = np.asarray(candidate.get("box_xyxy", [0, 0, 1, 1]), dtype=float)
    width = max(1.0, float(box[2] - box[0]))
    height = max(1.0, float(box[3] - box[1]))
    iou = candidate.get("predicted_iou")
    if isinstance(iou, list):
        iou_value = float(iou[int(candidate.get("token_index", 0))]) if iou else 0.0
    else:
        iou_value = float(iou or 0.0)
    area_ratio = float(candidate.get("mask_area_ratio", 0.0))
    return [
        float(box[0]), float(box[1]), float(box[2]), float(box[3]),
        width, height, width / max(1.0, float(candidate.get("video_width", 1))),
        height / max(1.0, float(candidate.get("video_height", 1))),
        area_ratio, iou_value, float(candidate.get("token_index", 0)),
    ]


__all__ = [
    "BOX_PROMPTED_SAM_PSEUDO_MASK",
    "BOX_RECTANGLE_MASKLET",
    "BOX_SANITIZED_RECTANGLE_MASKLET",
    "candidate_mask_features",
    "find_tracker_state",
    "interactive_box_candidates",
    "protected_state_signatures",
    "rectangle_mask",
    "tracker_ids",
    "write_target_mask",
]
